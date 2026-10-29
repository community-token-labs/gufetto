#ifndef GUFO_CACHE_TEST_HIP_RUNTIME_H_
#define GUFO_CACHE_TEST_HIP_RUNTIME_H_

#include <cstddef>

// Private CPU boundary for production host-allocation/transfer owners. Never a
// production or gufo_cache include dependency; no HIP runtime/device is loaded.
enum hipError_t {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorNotReady = 600,
  hipErrorUnknown = 999,
};
struct HipTestStream;
struct HipTestEvent;
using hipStream_t = HipTestStream*;
using hipEvent_t = HipTestEvent*;
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost };
inline constexpr unsigned hipStreamNonBlocking = 1, hipEventDisableTiming = 2;
extern "C" {
const char* hipGetErrorString(hipError_t);
hipError_t hipStreamCreateWithFlags(hipStream_t*, unsigned);
hipError_t hipStreamDestroy(hipStream_t);
hipError_t hipStreamSynchronize(hipStream_t);
hipError_t hipEventCreateWithFlags(hipEvent_t*, unsigned);
hipError_t hipEventDestroy(hipEvent_t);
hipError_t hipEventRecord(hipEvent_t, hipStream_t);
hipError_t hipEventQuery(hipEvent_t);
hipError_t hipEventSynchronize(hipEvent_t);
hipError_t hipMemcpyAsync(void*, const void*, std::size_t, hipMemcpyKind,
                          hipStream_t);
}
inline constexpr unsigned hipHostMallocDefault = 0;

extern "C" hipError_t hipHostMalloc(void**, std::size_t, unsigned);
extern "C" hipError_t hipHostFree(void*);

#endif
