#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_PERF_TRACE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_PERF_TRACE_HPP_

#include <hip/hip_runtime.h>

#include "src/core/perf_trace.hpp"

namespace gufo::models::qwen38_flash_next::rocm {

// Reused on the executor's single compute stream, never inside graph capture.
// Stream elapsed time includes host submission bubbles, not just kernel busy.
class PerfEvents {
public:
  void Init() {
    if (!perf::Enabled())
      return;
    if (hipEventCreate(&start_) != hipSuccess ||
        hipEventCreate(&end_) != hipSuccess)
      return;
    ready_ = true;
  }
  ~PerfEvents() {
    if (start_ != nullptr)
      (void)hipEventDestroy(start_);
    if (end_ != nullptr)
      (void)hipEventDestroy(end_);
  }
  void Begin(hipStream_t stream) {
    if (perf::current == nullptr || !ready_)
      return;
    host_start_ = perf::Now();
    active_ = hipEventRecord(start_, stream) == hipSuccess;
    stopped_ = false;
  }
  void Stop(hipStream_t stream) {
    if (active_ && !stopped_) {
      stopped_ = hipEventRecord(end_, stream) == hipSuccess;
    }
  }
  // Called only after the forward's existing stream synchronization.
  void Finish() {
    if (active_ && stopped_) {
      float ms = 0;
      if (hipEventElapsedTime(&ms, start_, end_) == hipSuccess)
        perf::current->gpu.push_back(
            {perf::forward_id, host_start_, perf::Now(),
             static_cast<std::uint64_t>(ms * 1000000.0)});
    }
    Cancel();
  }
  void Cancel() { active_ = false; }

private:
  hipEvent_t start_{nullptr}, end_{nullptr};
  std::uint64_t host_start_{0};
  bool ready_{false}, active_{false}, stopped_{false};
};

class PerfTarget {
public:
  PerfTarget(PerfEvents& events, hipStream_t stream) : events_(events) {
    events_.Begin(stream);
  }
  ~PerfTarget() { events_.Cancel(); }
  void Finish() { events_.Finish(); }

private:
  PerfEvents& events_;
};

}  // namespace gufo::models::qwen38_flash_next::rocm

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_ROCM_PERF_TRACE_HPP_
