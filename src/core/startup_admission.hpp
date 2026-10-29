#ifndef GUFO_CORE_STARTUP_ADMISSION_HPP_
#define GUFO_CORE_STARTUP_ADMISSION_HPP_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>

#include "memory_budget.hpp"

namespace gufo::core {

/// Quiescent execution capacity, with each lane's reserve already removed.
/// automatic is the provider's AUTO cache basis, not another capacity lane.
struct ExecutionCapacityObservation {
  std::size_t host;
  std::size_t device;
  std::size_t automatic;

  [[nodiscard]] constexpr std::size_t Capacity() const {
    return std::min(host, device);
  }
};

/// Observed capacity loss, not logical payload or an execution ledger claim.
struct ExecutionDebit {
  std::size_t host;
  std::size_t device;
};

[[nodiscard]] constexpr ExecutionDebit MeasuredExecutionDebit(
    ExecutionCapacityObservation before, ExecutionCapacityObservation after) {
  return {before.host > after.host ? before.host - after.host : 0,
          before.device > after.device ? before.device - after.device : 0};
}

[[nodiscard]] constexpr ExecutionDebit MaxExecutionDebit(ExecutionDebit a,
                                                         ExecutionDebit b) {
  return {std::max(a.host, b.host), std::max(a.device, b.device)};
}

/// pending_bytes contains still-unallocated obligations exactly once; it must
/// not include resident allocations already reflected in after. The controller
/// owns whether max_debit is applicable to the remaining configurations.
[[nodiscard]] constexpr bool FitsMeasuredExecution(
    ExecutionCapacityObservation after, ExecutionDebit max_debit,
    std::size_t remaining_slots, std::size_t pending_bytes) {
  const auto fits = [=](std::size_t capacity, std::size_t debit) {
    return FitsMemoryBudget(capacity, 0, pending_bytes) &&
           (debit == 0 ||
            remaining_slots <= (capacity - pending_bytes) / debit);
  };
  return fits(after.host, max_debit.host) &&
         fits(after.device, max_debit.device);
}

struct ContextSearchLimits {
  std::size_t granularity = 4096;
  std::size_t max_probes = 8;
  std::chrono::steady_clock::duration max_time = std::chrono::seconds(30);
};

struct ContextSearchResult {
  std::optional<std::size_t> verified_context;
  std::size_t probes = 0;
  bool step_limit_hit = false;
  bool time_limit_hit = false;
  std::chrono::steady_clock::duration elapsed{};
};

/// Invoke only after the caller's requested-context attempt fails. Each probe
/// must fully verify the staged pool and final guard, release all its
/// resources, and cooperatively observe the supplied deadline. Exceptions
/// propagate.
///
/// Binary search assumes smaller contexts tend to fit, but reports only the
/// largest actually successful, on-time probe, never an exact maximum. Capacity
/// changes/nonmonotonic behavior can hide other fits. A synchronous callback
/// cannot be preempted: elapsed can exceed max_time, but results at or beyond
/// the deadline are discarded. now_callback must be monotonic.
/// Zero granularity is invalid; nonpositive time or zero probes permits no
/// work. Limit flags mean the corresponding budget stopped an unresolved
/// search.
template<class FullProbe, class Now = decltype(&std::chrono::steady_clock::now)>
[[nodiscard]] ContextSearchResult SearchVerifiedContext(
    std::size_t requested_context, FullProbe&& full_probe_callback,
    ContextSearchLimits limits = {},
    Now now_callback = &std::chrono::steady_clock::now) {
  using Clock = std::chrono::steady_clock;
  if (limits.granularity == 0)
    throw std::invalid_argument("context search granularity must be positive");

  ContextSearchResult result;
  const auto start = now_callback();
  // Saturate deadline addition, including custom limits near the clock's end.
  const auto budget = std::max(limits.max_time, Clock::duration::zero());
  const auto deadline = start > Clock::time_point::max() - budget
                            ? Clock::time_point::max()
                            : start + budget;
  auto current = start;
  std::size_t low = 1;
  std::size_t high =
      requested_context == 0 ? 0 : (requested_context - 1) / limits.granularity;
  while (low <= high) {
    current = now_callback();
    if (current >= deadline || result.probes >= limits.max_probes)
      break;

    const auto middle = low + (high - low) / 2;
    const auto context = middle * limits.granularity;
    ++result.probes;
    const bool verified = full_probe_callback(context, deadline);
    current = now_callback();
    if (current >= deadline)
      break;

    if (verified) {
      result.verified_context = context;
      // middle <= requested_context - 1, so this cannot overflow.
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  if (low <= high) {
    result.step_limit_hit = result.probes >= limits.max_probes;
    result.time_limit_hit = current >= deadline;
  }
  result.elapsed = current - start;
  return result;
}

}  // namespace gufo::core
#endif
