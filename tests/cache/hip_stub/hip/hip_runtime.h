#ifndef GUFO_CACHE_TEST_HIP_RUNTIME_H_
#define GUFO_CACHE_TEST_HIP_RUNTIME_H_

#include <cstddef>

// Deliberately minimal CPU test boundary for host_allocations.cpp. This include
// directory is private to cache_host_allocations_test, never a production or
// gufo_cache dependency. No HIP runtime or device is loaded by that test.
enum hipError_t {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
};
inline constexpr unsigned hipHostMallocDefault = 0;

extern "C" hipError_t hipHostMalloc(void**, std::size_t, unsigned);
extern "C" hipError_t hipHostFree(void*);

#endif
