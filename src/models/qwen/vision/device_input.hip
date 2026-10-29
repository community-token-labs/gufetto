#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

#include "src/core/reclaimable_memory.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::models::qwen::vision {
namespace {
void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

// Metadata survives idle eviction so first encodes and re-encodes remain
// distinguishable. Each stream's last read is fenced, including draft readers.
struct PooledEmbeddings final : OutputPool::Storage {
  explicit PooledEmbeddings(std::size_t count)
      : embeddings(count), ever_encoded(count) {}
  ~PooledEmbeddings() override {
    if (uncertain_reader)
      (void)hipDeviceSynchronize();
    for (const auto& [stream, event] : readers) {
      (void)hipEventSynchronize(event);
      (void)hipEventDestroy(event);
    }
  }
  bool TryEvict() override {
    const std::lock_guard lock(mutex);
    if (uncertain_reader)
      throw std::runtime_error("vision reader completion unknown; reload the model");
    for (const auto& [stream, event] : readers) {
      const auto status = hipEventQuery(event);
      if (status == hipErrorNotReady)
        return false;
      Check(status);
    }
    for (auto& embedding : embeddings)
      embedding.reset();
    for (const auto& [stream, event] : readers)
      (void)hipEventDestroy(event);
    readers.clear();
    return true;
  }
  hipEvent_t Reader(hipStream_t stream) {
    for (auto it = readers.begin(); it != readers.end();) {
      if (it->first != stream && hipEventQuery(it->second) == hipSuccess) {
        Check(hipEventDestroy(it->second));
        it = readers.erase(it);
      } else {
        ++it;
      }
    }
    auto [it, inserted] = readers.try_emplace(stream, nullptr);
    if (inserted) {
      const auto status = hipEventCreateWithFlags(&it->second, hipEventDisableTiming);
      if (status != hipSuccess) {
        readers.erase(it);
        Check(status);
      }
    }
    return it->second;
  }
  std::mutex mutex;
  std::vector<std::shared_ptr<const Encoder::Embedding>> embeddings;
  std::vector<bool> ever_encoded;
  std::map<hipStream_t, hipEvent_t> readers;
  bool uncertain_reader{};
};

__global__ void InjectRows(float* hidden, const float* embedding,
                           std::uint32_t rows, std::uint32_t width,
                           std::uint32_t hc) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width * hc)
    return;
  const auto row = i / (width * hc);
  hidden[i] = embedding[row * width + i % width];
}
}  // namespace

DeviceInput::~DeviceInput() {
  if (descriptor_ != nullptr)
    (void)hipFree(descriptor_);
  (void)ReleasePositions();
}

hipError_t DeviceInput::ReleasePositions() noexcept {
  if (positions_ == nullptr)
    return hipSuccess;
  auto status = hipSuccess;
  const auto release = [&] {
    status = hipFree(positions_);
    return status == hipSuccess;
  };
  if (positions_budget_)
    positions_budget_->RetireImage(capacity_ * 3 * sizeof(std::int32_t), release);
  else
    (void)release();
  if (status == hipSuccess) {
    positions_ = nullptr;
    capacity_ = 0;
    positions_budget_.reset();
  }
  return status;
}

void DeviceInput::PrepareRope(hipStream_t stream) {
  if (descriptor_ != nullptr)
    return;
  if (continuation_hooks_)
    continuation_hooks_->BeforeVision(nullptr, &layout_);
  Check(hipMalloc(&descriptor_, sizeof(DeviceRope)));
  const DeviceRope descriptor{};
  Check(hipMemcpyAsync(descriptor_, &descriptor, sizeof(descriptor),
                       hipMemcpyHostToDevice, stream));
  Check(hipStreamSynchronize(stream));
}

void DeviceInput::Configure(std::shared_ptr<const Prompt> prompt,
                            std::shared_ptr<Encoder> encoder,
                            hipStream_t stream) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeVision(prompt.get(), nullptr);
  if (prompt && !prompt->images.empty() && !encoder) {
    throw std::invalid_argument("image input has no vision encoder");
  }
  const bool pooled = encoder && encoder->Outputs();
  const bool reuse = prompt_ && prompt && !prompt->cache_identity.empty() &&
                     encoder_ == encoder &&
                     prompt_->cache_identity == prompt->cache_identity;
  std::shared_ptr<OutputPool::Owner> owner;
  if (pooled && prompt && !prompt->images.empty()) {
    for (const auto& image : prompt->images) {
      if (!image.pixels ||
          std::uint64_t{image.grid.width} * 32 != image.pixels->width ||
          std::uint64_t{image.grid.height} * 32 != image.pixels->height ||
          std::uint64_t{image.grid.width} * image.grid.height > 16384)
        throw std::invalid_argument("image grid exceeds encoder workspace envelope");
    }
    owner = encoder->Outputs()->GetOwner(
        std::string(prompt->cache_identity.begin(), prompt->cache_identity.end()),
        prompt->ImageRows());
  }
  prompt_ = std::move(prompt);
  encoder_ = std::move(encoder);
  if (encoder_)
    memory_budget_ = encoder_->MemoryBudget();
  output_owner_ = std::move(owner);
  if (pooled || !reuse)
    embeddings_.clear();
  if (!pooled && prompt_)
    embeddings_.resize(prompt_->images.size());
  RestoreLayout(prompt_ ? prompt_->rope : RopeLayout{}, stream);
}

void DeviceInput::RestoreLayout(const RopeLayout& layout, hipStream_t stream) {
  if (continuation_hooks_)
    continuation_hooks_->BeforeVision(nullptr, &layout);
  layout.Validate(std::numeric_limits<std::int32_t>::max());
  if (layout == layout_ && (descriptor_ != nullptr || layout.images.empty()))
    return;
  const auto length = layout.PrefixLength();
  std::vector<std::array<std::int32_t, 3>> positions;
  const auto prepare_positions = [&] {
    positions.resize(length);
    for (std::uint32_t token = 0; token < length; ++token)
      positions[token] = layout.Position(token);
  };
  // Host allocation/validation failures must not discard a bound old array.
  if (memory_budget_)
    prepare_positions();
  if (descriptor_ == nullptr)
    Check(hipMalloc(&descriptor_, sizeof(DeviceRope)));
  if (length > capacity_) {
    Check(hipStreamSynchronize(stream));
    // Free before growing so the per-slot high-water claim need not cover
    // two arrays. A failed release retains both the old pointer and charge.
    Check(ReleasePositions());
    const auto bytes = std::size_t{length} * 3 * sizeof(std::int32_t);
    const auto allocate = [&] { Check(hipMalloc(&positions_, bytes)); };
    if (memory_budget_)
      memory_budget_->AllocateImage(bytes, allocate);
    else
      allocate();
    capacity_ = length;
    positions_budget_ = memory_budget_;
  }
  if (!memory_budget_)
    prepare_positions();
  if (length != 0) {
    Check(hipMemcpyAsync(positions_, positions.data(),
                         positions.size() * sizeof(positions[0]),
                         hipMemcpyHostToDevice, stream));
  }
  const DeviceRope descriptor{positions_, length, layout.Delta()};
  Check(hipMemcpyAsync(descriptor_, &descriptor, sizeof(descriptor),
                       hipMemcpyHostToDevice, stream));
  Check(hipStreamSynchronize(stream));
  layout_ = layout;
}

void DeviceInput::Inject(float* hidden, std::uint32_t position,
                         std::uint32_t count, std::uint32_t width,
                         std::uint32_t hc, hipStream_t stream) {
  if (!prompt_)
    return;
  std::shared_ptr<OutputPool::Lease> lease;
  std::shared_ptr<PooledEmbeddings> cache;
  std::unique_lock<std::mutex> lock;
  if (encoder_ && encoder_->Outputs()) {
    // Pooled text-only spans must work even after idle outputs were evicted.
    if (!prompt_->IntersectsImages(position, count))
      return;
    lease = encoder_->Outputs()->TryAcquire(output_owner_);
    if (!lease)
      throw std::runtime_error("vision output budget unavailable; admit the whole prompt before execution");
    cache = std::dynamic_pointer_cast<PooledEmbeddings>(
        lease->GetStorage([&] { return std::make_shared<PooledEmbeddings>(prompt_->images.size()); }));
    if (!cache || cache->embeddings.size() != prompt_->images.size())
      throw std::logic_error("invalid shared vision output storage");
    lock = std::unique_lock(cache->mutex);
  }
  auto& embeddings = cache ? cache->embeddings : embeddings_;
  for (std::size_t index = 0; index < prompt_->images.size(); ++index) {
    const auto& image = prompt_->images[index];
    const auto image_end =
        image.grid.offset + image.grid.height * image.grid.width;
    const auto begin = std::max(position, image.grid.offset);
    const auto end = std::min(position + count, image_end);
    if (begin >= end)
      continue;
    if (!embeddings[index]) {
      if (cache) {
        auto embedding = encoder_->Encode(*image.pixels, {}, is_cancelled_);
        if (embedding->width() != width ||
            embedding->rows() != image.grid.height * image.grid.width)
          throw std::logic_error("vision embedding does not match reserved input");
        lease->RecordEncode(cache->ever_encoded[index], embedding->rows());
        embeddings[index] = std::move(embedding);
        cache->ever_encoded[index] = true;
      } else {
        embeddings[index] = encoder_->Encode(*image.pixels, {}, is_cancelled_);
      }
    }
    const auto& embedding = *embeddings[index];
    if (embedding.width() != width ||
        embedding.rows() != image.grid.height * image.grid.width) {
      throw std::logic_error("vision embedding does not match language input");
    }
    const auto rows = end - begin;
    const auto reader = cache ? cache->Reader(stream) : nullptr;
    try {
      InjectRows<<<(rows * width * hc + 255) / 256, 256, 0, stream>>>(
          hidden + std::size_t{begin - position} * width * hc,
          embedding.data() + std::size_t{begin - image.grid.offset} * width, rows,
          width, hc);
      Check(hipGetLastError());
      if (cache)
        Check(hipEventRecord(reader, stream));
    } catch (...) {
      // A failed event record must never make an in-flight source evictable.
      // Retain its credits if the fallback drain cannot establish completion.
      if (cache && hipStreamSynchronize(stream) != hipSuccess)
        cache->uncertain_reader = true;
      throw;
    }
  }
}
}  // namespace gufo::models::qwen::vision
