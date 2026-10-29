#ifndef GUFO_CORE_HIP_HOST_ALLOCATIONS_HPP_
#define GUFO_CORE_HIP_HOST_ALLOCATIONS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>

namespace gufo::hip {

// Owns HIP host buffers and their long-term pins together. Allocations retain
// HIP's original coherence and huge-page policy. Drain readers and GPU work
// before destroying the owner. Page pinning lives in platform/page_pin.hpp.
class HostAllocations {
public:
  HostAllocations() = default;
  ~HostAllocations();
  HostAllocations(const HostAllocations&) = delete;
  HostAllocations& operator=(const HostAllocations&) = delete;

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t bytes,
                                    unsigned flags = hipHostMallocDefault);
  template<typename T>
  [[nodiscard]] hipError_t Allocate(T** pointer, std::size_t bytes,
                                    unsigned flags = hipHostMallocDefault) {
    return Allocate(reinterpret_cast<void**>(pointer), bytes, flags);
  }
  void Reset() noexcept;
  // Checked retirement for capacity accounting. True means every allocation
  // and pin has actually been released. On false, successful releases have
  // been removed and failed records remain retryable; the caller must retain
  // its charge AND this owner until a later successful retry. In particular,
  // PinnedBytes()==0 does not prove the HIP allocations have been freed.
  // Legacy Reset/destruction remain best-effort and discard failure records;
  // do not fall back to those when checked retirement must remain retryable.
  [[nodiscard]] bool TryReset() noexcept;
  [[nodiscard]] std::size_t PinnedBytes() const noexcept;

private:
  struct Allocation {
    void* data;
    std::size_t bytes;
    int pin;
  };
  std::vector<Allocation> allocations_;
};

}  // namespace gufo::hip
#endif
