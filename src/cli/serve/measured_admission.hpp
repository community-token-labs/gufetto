#ifndef GUFO_SERVER_MEASURED_ADMISSION_HPP_
#define GUFO_SERVER_MEASURED_ADMISSION_HPP_

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "src/cli/serve/component_text_cache.hpp"
#include "src/cli/serve/diagnostic_memory.hpp"
#include "src/cli/serve/logging.hpp"

namespace gufo::server::admission {
using Clock = std::chrono::steady_clock;

/// Only this exception is a measured capacity refusal, never a HIP/load error.
class CapacityRefusal : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

struct StagedExecution {
  core::ExecutionCapacityObservation capacity;
  std::size_t pending_execution;
};

// The requested pool and every recommendation use exactly this construction
// and guard. Slot one stays in the real pool on success. Failed trial owners
// destroy slots/idle workers before their context-specific adapter.
template<class Entry, class Create>
std::optional<StagedExecution> StageExecution(
    const ComponentStartupAdmission& startup, std::size_t staging_bytes,
    Create& create, cache::Adapter& adapter, std::size_t context,
    std::size_t count, std::size_t permanent_bytes,
    std::vector<std::unique_ptr<Entry>>& staged,
    Clock::time_point deadline = Clock::time_point::max()) {
  const auto diagnostic_bytes =
      DiagnosticMemoryClaim::Process().Snapshot().bytes;
  if (startup.now() >= deadline)
    return std::nullopt;
  auto pending = startup.pending(adapter, context);
  auto capacity = startup.observe();
  core::ExecutionDebit maximum{};
  std::size_t max_slot_execution{}, max_slot_images{};
  const auto fixed = core::AddMemoryBytes(
      core::AddMemoryBytes(staging_bytes, permanent_bytes), diagnostic_bytes);
  const auto fits = [&](std::size_t remaining,
                        std::size_t first_slot_bytes = 0) {
    RequireDiagnosticMemoryFits(diagnostic_bytes, capacity.Capacity());
    // Pending slot counters are independent of the shared counters: shared
    // materialization may lower the latter during construction.
    std::size_t required = core::AddMemoryBytes(
        core::AddMemoryBytes(pending.shared_execution, pending.slot_execution),
        core::AddMemoryBytes(pending.shared_images, pending.slot_images));
    required = core::AddMemoryBytes(required, fixed);
    const auto per_slot =
        core::AddMemoryBytes(max_slot_execution, max_slot_images);
    if (per_slot &&
        remaining >
            (std::numeric_limits<std::size_t>::max() - required) / per_slot)
      return false;
    required += remaining * per_slot;
    const bool accepted =
        core::FitsMemoryBudget(capacity.host, first_slot_bytes, required) &&
        core::FitsMemoryBudget(capacity.device, first_slot_bytes, required) &&
        core::FitsMeasuredExecution(capacity, maximum, remaining, required);
    Logger::Info(
        "cache",
        "event=measured_startup context=" + std::to_string(context) +
            " slots=" + std::to_string(staged.size()) +
            " remaining_slots=" + std::to_string(remaining) +
            " host_capacity_bytes=" + std::to_string(capacity.host) +
            " device_capacity_bytes=" + std::to_string(capacity.device) +
            " max_host_debit_bytes=" + std::to_string(maximum.host) +
            " max_device_debit_bytes=" + std::to_string(maximum.device) +
            " required_bytes=" + std::to_string(required) +
            " first_slot_logical_bytes=" + std::to_string(first_slot_bytes) +
            " fits=" + (accepted ? "1" : "0"));
    return accepted;
  };
  // Bypass only the old aggregate gate. Even calibration must not knowingly
  // spend either lane's reserve: the complete logical size of slot one plus
  // existing pending/fixed obligations must fit before allocating that slot.
  // After creation its actual debit and pending counters replace this bound.
  if (!fits(0, startup.slot_bytes(context)))
    return std::nullopt;
  for (std::size_t i = 0; i < count; ++i) {
    if (startup.now() >= deadline)
      return std::nullopt;
    staged.push_back(create(adapter, i));
    if (startup.now() >= deadline)
      return std::nullopt;
    const auto next_pending = startup.pending(adapter, context);
    const auto after = startup.observe();
    maximum = core::MaxExecutionDebit(
        maximum, core::MeasuredExecutionDebit(capacity, after));
    max_slot_execution =
        std::max(max_slot_execution,
                 core::PendingMemoryBytes(next_pending.slot_execution,
                                          pending.slot_execution));
    max_slot_images = std::max(
        max_slot_images, core::PendingMemoryBytes(next_pending.slot_images,
                                                  pending.slot_images));
    pending = next_pending;
    capacity = after;
    if (startup.now() >= deadline || !fits(count - i - 1))
      return std::nullopt;
  }
  return StagedExecution{
      capacity,
      core::AddMemoryBytes(core::AddMemoryBytes(pending.shared_execution,
                                                pending.slot_execution),
                           diagnostic_bytes)};
}

template<class Entry, class Create>
StagedExecution AdmitMeasuredExecution(
    ComponentCacheResources& resources, std::size_t requested_context,
    std::size_t count, std::size_t permanent_bytes, std::size_t staging_bytes,
    std::vector<std::unique_ptr<Entry>>& entries, Create create) {
  const auto& startup = *resources.startup_admission;
  if (!count || startup.context != requested_context || !startup.now ||
      !startup.slot_bytes || !startup.observe || !startup.pending ||
      !startup.make_probe || !startup.freeze_capacity ||
      !resources.freeze_execution_liability)
    throw std::logic_error("incomplete measured startup admission provider");
  if (const auto admitted =
          StageExecution(startup, staging_bytes, create, *resources.adapter,
                         startup.context, count, permanent_bytes, entries))
    return *admitted;

  // No production slot survives a refusal. Probes neither touch the original
  // adapter's cumulative counters nor bind another backend to the Model.
  entries.clear();
  const core::ContextSearchLimits limits;
  const auto search = core::SearchVerifiedContext(
      startup.context,
      [&](std::size_t context, Clock::time_point deadline) {
        auto adapter = startup.make_probe(context);
        if (!adapter)
          throw std::logic_error("startup probe adapter is null");
        std::vector<std::unique_ptr<Entry>> trial;
        return StageExecution(startup, staging_bytes, create, *adapter, context,
                              count, permanent_bytes, trial, deadline)
            .has_value();
      },
      limits, startup.now);
  Logger::Info(
      "cache",
      "event=startup_context_search probes=" + std::to_string(search.probes) +
          " probe_limit=" + std::to_string(limits.max_probes) +
          " time_limit_ms=30000 elapsed_ms=" +
          std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                             search.elapsed)
                             .count()) +
          " step_limit_hit=" + std::to_string(search.step_limit_hit) +
          " time_limit_hit=" + std::to_string(search.time_limit_hit));
  std::string message =
      "measured startup execution exceeds capacity for --context " +
      std::to_string(startup.context) + "; ";
  if (search.verified_context)
    message += "verified to fit: --context " +
               std::to_string(*search.verified_context) +
               " (staged execution pool, not a maximum)";
  else
    message += "no fitting context found within the search limits";
  throw CapacityRefusal(message);
}

}  // namespace gufo::server::admission
#endif
