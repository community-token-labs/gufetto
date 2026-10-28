#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string_view>
#include <utility>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#ifdef GUFO_ENABLE_NPU
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"
#endif

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

#ifdef GUFO_ENABLE_NPU
bool NpuFlag(const char* name, bool fallback = false) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : std::string_view(value) == "1";
}

std::uint32_t NpuCount(const char* name, std::uint32_t fallback,
                       std::uint32_t maximum) {
  const char* value = std::getenv(name);
  if (value == nullptr)
    return fallback;
  char* end = nullptr;
  const long long count = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0')
    return fallback;
  if (count <= 0)
    return 0;
  return static_cast<std::uint32_t>(
      std::min(count, static_cast<long long>(maximum)));
}
#endif

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

/// The trunk and the optional MTP block as one traversal's output.
struct Uploaded {
  DeviceTensor token_embd;
  DeviceTensor output;
  DeviceMixer hc_head;
  std::vector<DeviceLayer> layers;
  DeviceLayer mtp;
};

/// Walks the model's tensors in upload order. With `weights` null the walk
/// only measures: it sums the bytes every allocation would request and does
/// no device work, so the weight placement can be decided from the tensors.
struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator* weights;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  std::uint64_t requested{0};

  [[nodiscard]] bool Measuring() const noexcept { return weights == nullptr; }

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t size) {
    requested += size;
    if (Measuring()) {
      return hipSuccess;
    }
    return weights->Allocate(pointer, size);
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || (!Measuring() && !stager.Finish(error))) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (Allocate(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      if (Measuring()) {
        continue;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    if (Measuring()) {
      return;
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (!Measuring()) {
      allocations.push_back(ptr);
      bytes += size + kTailMargin;
      std::size_t offset = 0;
      for (const TensorRef* t : parts) {
        if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                         static_cast<std::uint8_t*>(ptr) + offset, error)) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += t->SizeBytes();
      }
    }
    if (type == core::GgmlType::kQ8_0) {
      if (Measuring()) {
        return d;
      }
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (Allocate(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  Uploaded Model(const ModelWeights& w, const MtpWeights* mtp,
                 std::uint32_t trunk_shards) {
    Uploaded u;
    u.token_embd = Copy(w.token_embd);
    u.output =
        w.output.data == w.token_embd.data ? u.token_embd : Copy(w.output);
    u.hc_head = Mixer(w.hc_head);
    u.layers.reserve(w.layers.size());
    for (const auto& l : w.layers) {
      u.layers.push_back(Layer(l));
      if (!ok) {
        return u;
      }
    }
    if (mtp != nullptr) {
      shard_base = trunk_shards;
      u.mtp = Layer(mtp->block);
    }
    return u;
  }
};

}  // namespace

std::size_t DeviceModel::resident_bytes() const noexcept {
#if defined(GUFO_ENABLE_NPU) && GUFO_ENABLE_NPU
  if (npu_ != nullptr)
    return bytes_ + npu_->MemoryFootprintBytes().total_backing_bytes;
#endif
  return bytes_;
}

DeviceModel::~DeviceModel() {
#ifdef GUFO_ENABLE_NPU
  // Drain NPU callbacks before releasing GPU weights. The executor must keep
  // registered streams alive until the runtime has released/drained them.
  npu_.reset();
#endif
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

bool DeviceModel::SetDraftVocab(std::uint32_t prefix_rows,
                                std::vector<std::int32_t> specials,
                                std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  if (output_.empty() || output_.rows == 0 || output_bytes_ % output_.rows != 0)
    return fail("draft vocabulary needs a row-addressable output head");
  if (prefix_rows == 0 || prefix_rows >= output_.rows)
    return fail("draft vocabulary prefix must leave rows out");
  for (const auto id : specials) {
    if (id < static_cast<std::int32_t>(prefix_rows) ||
        static_cast<std::uint32_t>(id) >= output_.rows)
      return fail("draft special tokens must lie above the prefix");
  }
  // The prefix is the head's own leading rows: no copy.
  draft_output_ = output_;
  draft_output_.rows = prefix_rows;
  draft_specials_ = std::move(specials);
  return true;
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // Q6_K is supported only for the explicitly bound MTP dense/HC roles.
  // Embeddings, vocabulary heads and routed experts still cannot decode it;
  // reject those before allocating device weights.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
#ifdef GUFO_ENABLE_NPU
  // Wake the NPU before heavy GPU work: resume touches shared SMU/PSP state.
  // A directory alone never opts in. Only GUFO_STRIX_NPU=1 enables offload.
  std::unique_ptr<NpuGemm> early_npu;
  if (NpuFlag("GUFO_STRIX_NPU")) {
    const char* dir = std::getenv("GUFO_NPU_DIR");
    if (dir == nullptr || dir[0] == '\0') {
      if (error_msg != nullptr)
        *error_msg = "GUFO_STRIX_NPU=1 requires GUFO_NPU_DIR";
      return nullptr;
    }
    NpuGemm::Config cfg;
    cfg.xclbin = std::string(dir) + "/final_4096x2560x1024_128x64x128.xclbin";
    cfg.insts = std::string(dir) + "/insts_4096x2560x1024_128x64x128.txt";
    early_npu = NpuGemm::Create(cfg, error_msg);
    if (early_npu == nullptr)
      return nullptr;
    if (!early_npu->Healthy() || early_npu->k() != 2560 ||
        early_npu->n() != 1024 || early_npu->tokens() != 4096) {
      if (error_msg != nullptr)
        *error_msg =
            "NPU runtime unhealthy or fixed projection geometry mismatch";
      return nullptr;
    }
    // Buffer isolation is a runtime policy: leave its default-on private BO
    // mode intact; only an explicit GUFO_NPU_ISOLATE=0 opts into import.
  }
#endif
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  // Measure first: the placement is decided once, from the bytes this walk
  // will request of the device heap.
  std::uint64_t measured = 0;
  {
    std::vector<Conversion> conversions;
    std::vector<void*> allocations;
    std::size_t bytes = 0;
    std::size_t max_half_cols = 1;
    std::size_t max_q8_cols = 32;
    Uploader measure{*stager, nullptr,       conversions, allocations,
                     bytes,   max_half_cols, max_q8_cols, error_msg};
    (void)measure.Model(w, mtp, shard_count);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  std::vector<Conversion> conversions;
  Uploader up{*stager,   &weights,          conversions,     m->allocations_,
              m->bytes_, m->max_half_cols_, m->max_q8_cols_, error_msg};
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  m->output_bytes_ = w.output.SizeBytes();
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
#ifdef GUFO_ENABLE_NPU
  if (early_npu != nullptr) {
    auto npu = std::move(early_npu);
    std::size_t staged = 0;
    std::uint32_t z_layers = 0, gate_layers = 0, shexp_layers = 0;
    std::uint32_t whole_layers = 0;
    const auto z_slices = NpuCount("GUFO_NPU_SLICES", 6, 12);
    const auto gate_slices = NpuCount("GUFO_NPU_GATE_SLICES", 12, 12);
    const auto layer_limit =
        NpuCount("GUFO_NPU_LAYERS", UINT32_MAX, UINT32_MAX);
    const bool whole = NpuFlag("GUFO_NPU_WHOLE");
    const bool shexp = NpuFlag("GUFO_NPU_SHEXP", true);
    const auto eligible = [&](const DeviceTensor& t, std::uint32_t rows) {
      return !t.empty() && t.type == core::GgmlType::kQ8_0 && t.experts == 1 &&
             t.cols == npu->k() && t.rows == rows;
    };
    const auto fail = [&](const std::string& message) {
      if (error_msg != nullptr)
        *error_msg = "NPU weight staging failed: " + message;
      return false;
    };
    // Keep GPU originals immutable. Only the encoded private NPU copies and
    // their ids are additional storage, deliberately excluded from bytes_.
    const auto stage = [&](const DeviceTensor& t, std::uint32_t row0,
                           std::uint32_t slices, std::uint32_t rows_per_slice,
                           std::vector<int>& ids) {
      std::string error;
      hipError_t launch_status = hipSuccess;
      for (std::uint32_t i = 0; i < slices; ++i) {
        const int id = npu->NewBuffer(npu->BBytes(), &error);
        if (id < 0)
          break;
        NpuEncodeWeights(t.data, t.cols, row0 + i * npu->n(), rows_per_slice,
                         npu->k(), 0, npu->k(), npu->Device(id), nullptr);
        ids.push_back(id);
        staged += npu->BBytes();
        launch_status = hipGetLastError();
        if (launch_status != hipSuccess)
          break;
      }
      // Also drain partial uploads on allocation/launch failure before BOs
      // can be destroyed. Never turn a runtime failure into GPU fallback.
      const auto status = hipDeviceSynchronize();
      if (launch_status != hipSuccess)
        return fail(hipGetErrorString(launch_status));
      if (status != hipSuccess)
        return fail(hipGetErrorString(status));
      if (ids.size() != slices)
        return fail(error.empty() ? "buffer allocation failed" : error);
      for (const int id : ids) {
        npu->FinalizeWeights(id);
        if (!npu->Healthy())
          return fail("FinalizeWeights left runtime unhealthy");
      }
      return true;
    };
    for (auto& l : m->layers_) {
      if (l.linear && eligible(l.ssm_in, 16384)) {
        // Stage the whole matrix first, then alias its tail for z. This avoids
        // duplicate private backing when whole-projection execution is on.
        if (whole) {
          if (!stage(l.ssm_in, 0, 16, npu->n(), l.npu_ssm_all))
            return nullptr;
          ++whole_layers;
        }
        if (z_slices != 0 && z_layers < layer_limit) {
          if (!l.npu_ssm_all.empty()) {
            l.npu_z.assign(l.npu_ssm_all.end() - z_slices, l.npu_ssm_all.end());
          } else if (!stage(l.ssm_in, (16 - z_slices) * npu->n(), z_slices,
                            npu->n(), l.npu_z)) {
            return nullptr;
          }
          ++z_layers;
        }
      }
      if (!l.linear && eligible(l.attn_qkv, 13312)) {
        if (whole) {
          if (!stage(l.attn_qkv, 0, 13, npu->n(), l.npu_qkv_all))
            return nullptr;
          ++whole_layers;
        }
        if (gate_slices != 0) {
          if (!l.npu_qkv_all.empty()) {
            l.npu_gate.assign(l.npu_qkv_all.begin(),
                              l.npu_qkv_all.begin() + gate_slices);
          } else if (!stage(l.attn_qkv, 0, gate_slices, npu->n(), l.npu_gate)) {
            return nullptr;
          }
          ++gate_layers;
        }
      }
      // Only the expected 640-row Q8_0 shared expert is supported; Q4 and
      // other nightly formats stay on their original GPU paths unchanged.
      if (shexp && eligible(l.shexp_gate, 640) && eligible(l.shexp_up, 640)) {
        std::vector<int> up_ids;
        if (!stage(l.shexp_gate, 0, 1, 640, l.npu_shexp) ||
            !stage(l.shexp_up, 0, 1, 640, up_ids))
          return nullptr;
        l.npu_shexp.push_back(up_ids.front());
        ++shexp_layers;
      }
    }
    if (!npu->Healthy()) {
      fail("runtime unhealthy after staging");
      return nullptr;
    }
    std::fprintf(stderr,
                 "NPU: z %u, attention query/gate %u, shared expert %u, "
                 "whole %u layers; %.2f GiB logical encoded weights staged "
                 "(not total backing memory)\n",
                 z_layers, gate_layers, shexp_layers, whole_layers,
                 static_cast<double>(staged) / (1024.0 * 1024.0 * 1024.0));
    const auto memory = npu->MemoryFootprintBytes();
    std::fprintf(stderr,
                 "NPU allocation ledger: pinned=%zu private=%zu "
                 "imported_alias=%zu instructions=%zu backing=%zu bytes; "
                 "requested sizes, not measured RSS/driver overhead; "
                 "executor buffers added after staging\n",
                 memory.pinned_host_bytes, memory.private_bo_bytes,
                 memory.imported_alias_bytes, memory.instruction_bo_bytes,
                 memory.total_backing_bytes);
    // Shared-expert-only and attention-only staging must retain the engine.
    if (z_layers != 0 || gate_layers != 0 || shexp_layers != 0 ||
        whole_layers != 0)
      m->npu_ = std::move(npu);
  }
#endif
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
