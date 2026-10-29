#ifndef GUFO_MODELS_QWEN_VISION_MEMORY_BUDGET_HPP_
#define GUFO_MODELS_QWEN_VISION_MEMORY_BUDGET_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace gufo::models::qwen::vision {

/// Encoder::Encode produces the whole image (at most 65536 ViT patches),
/// independently of the language-model prefill chunk. Output rows are charged
/// separately. A 128-row attention-tile envelope covers the cheaper special
/// 4096-patch tile and its more expensive neighboring shapes.
[[nodiscard]] constexpr std::size_t EncoderWorkspaceBytes(
    std::uint32_t max_image_rows = 16384) {
  const std::uint64_t patches = 4ULL * std::min(max_image_rows, 16384U);
  const auto tile = std::min<std::uint64_t>(patches, 128);
  const auto buffers = 40416 * patches + 96 * tile * patches + 4608 * tile;
  const auto upload = std::min<std::uint64_t>(16ULL << 20, 768 * patches);
  return static_cast<std::size_t>(buffers + upload);
}

/// Full-context input allowance. Image-token policies may separately bound
/// embeddings while retaining context-sized positional storage.
[[nodiscard]] inline std::size_t FullContextInputBytes(std::uint32_t context,
                                                       std::uint32_t hidden) {
  const std::uint64_t stride = std::uint64_t{hidden} * sizeof(float) + 12;
  if (context &&
      stride > (std::numeric_limits<std::size_t>::max() - 64) / context)
    throw std::overflow_error("vision input reservation overflows");
  return static_cast<std::size_t>(stride * context + 64);
}

/// The admission claim covers one workspace per encoder, including callers
/// outside the single serving worker. Hold through all scratch destruction.
class SerializedEncoderWorkspace {
public:
  [[nodiscard]] std::unique_lock<std::mutex> Acquire() {
    return std::unique_lock(mutex_);
  }

private:
  std::mutex mutex_;
};

}  // namespace gufo::models::qwen::vision
#endif
