#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kv_q8_load.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

// One transient view per executor, rebuilt from canonical packed bytes on
// its compute stream. No view survives logically beyond this attention call.
__global__ void DequantizeQ8KvKernel(const void* k_cache, const void* v_cache,
                                     void* k_f16, void* v_f16,
                                     std::size_t groups, bool key_q4,
                                     bool value_q4) {
  const std::size_t group = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (group >= groups)
    return;
  const auto* source = blockIdx.y == 0 ? k_cache : v_cache;
  auto* target = static_cast<uint4*>(blockIdx.y == 0 ? k_f16 : v_f16);
  if (blockIdx.y == 0 ? key_q4 : value_q4) {
    const auto& block = static_cast<const KvBlockQ4_0*>(source)[group / 4];
    target[group] =
        detail::LoadQ4Kv8Packed(block.values, (group % 4) * 8, block.scale);
  } else {
    const auto& block = static_cast<const KvBlockQ8_0*>(source)[group / 4];
    target[group] =
        detail::LoadQ8Kv8Packed(block.values + (group % 4) * 8,
                                __builtin_bit_cast(_Float16, block.scale));
  }
}

}  // namespace

bool DequantizeQ8Kv(const void* k_cache, const void* v_cache, void* k_f16,
                    void* v_f16, std::uint32_t rows, std::uint32_t width,
                    hipStream_t stream, KvCacheType type) {
  if (!IsKnownKvCacheType(type) || !KvIsPacked(type))
    return false;
  if (k_cache == nullptr || v_cache == nullptr || k_f16 == nullptr ||
      v_f16 == nullptr || rows == 0 || width == 0 || width % 32 != 0 ||
      std::size_t{rows} > std::numeric_limits<std::size_t>::max() / width)
    return false;
  const std::size_t groups = std::size_t{rows} * width / 8;
  const std::size_t blocks = (groups - 1) / 256 + 1;
  if (blocks > std::numeric_limits<std::uint32_t>::max())
    return false;
  hipLaunchKernelGGL(DequantizeQ8KvKernel,
                     dim3(static_cast<std::uint32_t>(blocks), 2), dim3(256), 0,
                     stream, k_cache, v_cache, k_f16, v_f16, groups,
                     KvKeyType(type) == KvCacheType::kQ4_0,
                     KvValueType(type) == KvCacheType::kQ4_0);
  return hipGetLastError() == hipSuccess;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
