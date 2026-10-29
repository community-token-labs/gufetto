// Host-only bounds test: actual allocator, layout and lane-store source slices.
// HIP allocation/launch and warp sums are proxies, not GPU numerical coverage.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
namespace core {
enum class GgmlType { kBF16, kF32, kQ8_0 };
}
using hipStream_t = int;
constexpr unsigned char kCanary = 0xA5;
void* allocation = nullptr;
std::size_t allocated_bytes = 0;
std::size_t guard_bytes = 0;
std::size_t allocations = 0;

int hipMalloc(void** ptr, std::size_t bytes) {
  // Extra storage makes the old overflow observable without invoking host UB.
  *ptr = std::malloc(bytes + guard_bytes);
  assert(*ptr != nullptr);
  allocation = *ptr;
  allocated_bytes = bytes;
  ++allocations;
  std::memset(*ptr, kCanary, bytes + guard_bytes);
  return 0;
}
int hipFree(void* ptr) {
  std::free(ptr);
  allocation = nullptr;
  return 0;
}
void LaunchQuantizeActivationQ8_1(const void*, void*, std::size_t, std::size_t,
                                  hipStream_t);
void LaunchBatchedQuantGEMMPreQuantized(core::GgmlType, const void*,
                                        const void*, float*, std::size_t,
                                        std::size_t, std::size_t, hipStream_t) {
}
// All simulated lanes hold q=1, so this is the exact broadcast sum for that
// constant input. Addressing and stores below are the production helper.
int __shfl_xor(int value, int) {
  return value;
}
#define __host__
#define __device__
#define __forceinline__ inline
#define HIP_CHECK(call) assert((call) == 0)
#include "prefill_scratch_body.inc"
#undef HIP_CHECK
#undef __forceinline__
#undef __device__
#undef __host__

void LaunchQuantizeActivationQ8_1(const void*, void* out, std::size_t batch,
                                  std::size_t k, hipStream_t) {
  // The real quantizer writes live token lanes and zeroes padded token lanes.
  // Use the same layout/store helper, including a sidecar store for each
  // lane-0.
  const auto padded = (batch + 15) / 16 * 16;
  for (std::size_t t = 0; t < padded; ++t)
    for (std::size_t block = 0; block < k / 32; ++block)
      for (std::size_t lane = 0; lane < 32; ++lane)
        StoreQ8ActLane<true>(out, batch, k / 32, t, block, lane,
                             t < batch ? 1.0F : 0.0F, t < batch ? 1 : 0);
}

bool Run(std::size_t batch, std::size_t k) {
  const auto padded = (batch + 15) / 16 * 16;
  // Independent specification: 36-byte payload + 4-byte sum per 32 elements.
  const auto expected = padded * (k / 32) * 40;
  guard_bytes = expected + 64;
  LaunchBatchedQuantGEMM(core::GgmlType::kQ8_0, nullptr, nullptr, nullptr,
                         batch, 64, k, 0);
  const auto* guard =
      static_cast<const unsigned char*>(allocation) + allocated_bytes;
  // Only inspect 64 bytes: enough for the first sidecar write, including when
  // a smaller call reuses an earlier (larger) allocation and guard.
  const bool untouched =
      std::all_of(guard, guard + 64, [](auto byte) { return byte == kCanary; });
  const bool fits = allocated_bytes >= expected;
  std::printf("batch=%zu k=%zu allocated=%zu required=%zu guard=%s\n", batch, k,
              allocated_bytes, expected, untouched ? "intact" : "CORRUPTED");
  return untouched && fits;
}
}  // namespace

int main() {
  bool ok = true;
  ok &= Run(100, 160);     // ragged tile
  ok &= Run(1024, 16384);  // 18 MiB payload, 2 MiB sidecar
  const auto count = allocations;
  ok &= Run(1, 32);    // existing allocation reused, no shrink
  ok &= Run(16, 160);  // whole tile reused
  assert(count == allocations);
  hipFree(allocation);
  return ok ? 0 : 1;
}
