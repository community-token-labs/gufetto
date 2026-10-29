#include "src/core/hip/host_allocations.hpp"

#include "src/core/platform/page_pin.hpp"

namespace gufo::hip {

HostAllocations::~HostAllocations() {
  Reset();
}

hipError_t HostAllocations::Allocate(void** pointer, std::size_t bytes,
                                     unsigned flags) {
  if (!pointer)
    return hipErrorInvalidValue;
  *pointer = nullptr;
  // Admit bookkeeping before physical allocation. Otherwise a vector failure
  // needs an unchecked free rollback and can lose a still-resident allocation.
  try {
    allocations_.push_back({nullptr, 0, -1});
  } catch (const std::bad_alloc&) {
    return hipErrorOutOfMemory;
  }
  void* data{};
  const auto status = hipHostMalloc(&data, bytes, flags);
  if (status != hipSuccess || !data) {
    allocations_.pop_back();
    return status;
  }
  allocations_.back() = {data, bytes, platform::PinHostPages(data, bytes)};
  *pointer = data;
  return hipSuccess;
}

void HostAllocations::Reset() noexcept {
  for (const auto& allocation : allocations_) {
    platform::UnpinHostPages(allocation.pin);
    (void)hipHostFree(allocation.data);
  }
  allocations_.clear();
}

bool HostAllocations::TryReset() noexcept {
  std::size_t kept = 0;
  for (auto allocation : allocations_) {
    if (platform::TryUnpinHostPages(allocation.pin)) {
      // If HIP free fails, retry only that remaining stage, never the consumed
      // pin token (whose descriptor may already have been reused).
      allocation.pin = -1;
      if (hipHostFree(allocation.data) == hipSuccess)
        continue;
    }
    allocations_[kept++] = allocation;
  }
  // Compact only failed POD records; shrinking cannot allocate or throw.
  allocations_.resize(kept);
  return allocations_.empty();
}

std::size_t HostAllocations::PinnedBytes() const noexcept {
  std::size_t bytes{};
  for (const auto& allocation : allocations_)
    if (allocation.pin >= 0)
      bytes += allocation.bytes;
  return bytes;
}

}  // namespace gufo::hip
