#ifndef GUFO_SERVER_COMPONENT_CACHE_METRICS_HPP_
#define GUFO_SERVER_COMPONENT_CACHE_METRICS_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cache/ledger.hpp"
#include "src/cache/slot.hpp"
#include "src/cache/streaming.hpp"
#include "src/core/json.hpp"

namespace gufo::server {
// Acquisition outcomes describe observed resident state, not an all-history
// reuse oracle. An absent indexed prefix does not prove a first-ever prompt.
enum class CacheReuseReason : std::uint8_t {
  kNotAcquired,
  kLive,
  kRam,
  kDisk,
  kDisabled,
  kUnknownCompatibility,
  kNoIndexedPrefix,
  kNoCompatibleBoundary,
  kInputIdentity,
  kMissingComponent,
  kStablePrefixBoundary,
  kBoundaryGap,
  kRestoreFailed,
  kFrontierNotReady,
  kCancelled,
  kCount,
};
inline constexpr std::array<std::string_view,
                            static_cast<std::size_t>(CacheReuseReason::kCount)>
    kCacheReuseReasonNames{"not_acquired",
                           "live",
                           "ram",
                           "disk",
                           "disabled",
                           "unknown_compatibility",
                           "no_indexed_prefix",
                           "no_compatible_boundary",
                           "input_identity",
                           "missing_component",
                           "stable_prefix_boundary",
                           "boundary_gap",
                           "restore_failed",
                           "frontier_not_ready",
                           "cancelled"};
inline std::string_view CacheReuseReasonName(CacheReuseReason reason) {
  return kCacheReuseReasonNames.at(static_cast<std::size_t>(reason));
}
// Process-lifetime acquisition counts, including acquisitions later cancelled.
// Never increment on metadata-only lookup or repeated metrics serialization.
inline auto& CacheReuseReasonCounts() {
  static std::array<std::atomic<std::uint64_t>, kCacheReuseReasonNames.size()>
      counts{};
  return counts;
}
inline json::Value CacheReuseCountersJson() {
  auto result = json::Value::object();
  for (std::size_t i = 1; i < kCacheReuseReasonNames.size(); ++i)
    result[std::string(kCacheReuseReasonNames[i])] =
        CacheReuseReasonCounts()[i].load(std::memory_order_relaxed);
  return result;
}
struct ComponentCacheMetrics {
  std::string reuse{"none"};
  CacheReuseReason reuse_reason{CacheReuseReason::kNotAcquired};
  // Identity-qualified resident index agreement before retiring the live slot.
  // Not all-history reusable tokens; evicted and unrecorded state is absent.
  std::size_t indexed_common_tokens{}, indexed_gap_tokens{};
  std::size_t unready_frontiers{};
  std::uint64_t used_entry_id{};
  std::uint32_t selection_reason{};
  std::uint64_t acquisition_id{}, admission_index_version{},
      selected_entry_id{};
  std::size_t restore_attempts{}, restore_failures{};
  // Static reason codes only, never exception messages or prompt content.
  std::string restore_failure_reason{"none"};
  std::size_t deepest_live{}, deepest_ram{}, deepest_disk{}, selected{};
  std::size_t restored_bytes{}, prefilled{}, captures{}, captured_bytes{};
  std::size_t borrowed_capacity_bytes{}, copied_payload_bytes{};
  std::vector<std::uint64_t> captured_checkpoint_ids;
  std::size_t disk_queued_bytes{};
  double disk_enqueue_ms{};
  cache::ResourceSnapshot ledger;
  std::optional<cache::PersistenceQueueStats> persistence;
  cache::SpillMetrics spill;
};
inline json::Value ComponentCacheJson(const ComponentCacheMetrics& value) {
  auto result = json::Value::object();
  result["schema"] = "component-cache-v1";
  result["reuse"] = value.reuse;
  result["reuse_reason"] =
      std::string(CacheReuseReasonName(value.reuse_reason));
  result["indexed_common_tokens"] = value.indexed_common_tokens;
  result["indexed_gap_tokens"] = value.indexed_gap_tokens;
  result["unready_frontiers"] = value.unready_frontiers;
  result["used_entry_id"] = value.used_entry_id;
  result["selection_reason"] = static_cast<std::size_t>(value.selection_reason);
  result["deepest_live_tokens"] = value.deepest_live;
  result["deepest_ram_tokens"] = value.deepest_ram;
  result["deepest_disk_tokens"] = value.deepest_disk;
  result["selected_tokens"] = value.selected;
  result["acquisition_id"] = value.acquisition_id;
  result["admission_index_version"] = value.admission_index_version;
  result["selected_entry_id"] = value.selected_entry_id;
  result["restore_attempts"] = value.restore_attempts;
  result["restore_failures"] = value.restore_failures;
  result["restore_failure_reason"] = value.restore_failure_reason;
  result["restored_bytes"] = value.restored_bytes;
  result["prefilled_tokens"] = value.prefilled;
  result["captures"] = value.captures;
  result["captured_capacity_bytes"] = value.captured_bytes;
  result["borrowed_capacity_bytes"] = value.borrowed_capacity_bytes;
  result["copied_payload_bytes"] = value.copied_payload_bytes;
  auto checkpoints = json::Value::array();
  for (const auto id : value.captured_checkpoint_ids)
    checkpoints.push_back(id);
  result["captured_checkpoint_ids"] = std::move(checkpoints);
  auto spill = json::Value::object();
  spill["idle_bytes"] = value.spill.idle_bytes;
  spill["foreground_bytes"] = value.spill.foreground_bytes;
  spill["idle_copy_ms"] = value.spill.idle_copy_ns / 1e6;
  spill["foreground_copy_ms"] = value.spill.foreground_copy_ns / 1e6;
  spill["stream_wait_ms"] = value.spill.stream_wait_ns / 1e6;
  spill["residual_wait_ms"] = value.spill.residual_wait_ns / 1e6;
  spill["reassignments"] = value.spill.reassignments;
  spill["idle_failures"] = value.spill.idle_failures;
  result["global_spill"] = std::move(spill);
  // These are contemporaneous global gauges, not request-local attribution.
  result["global_ledger_bytes"] = value.ledger.total_bytes;
  result["global_ram_bytes"] = value.ledger.ram_bytes;
  result["global_peak_ledger_bytes"] = value.ledger.peak_total_bytes;
  result["global_peak_ram_bytes"] = value.ledger.peak_ram_bytes;
  result["global_persistence_pinned_bytes"] =
      value.ledger.persistence_pinned_bytes;
  result["global_peak_persistence_pinned_bytes"] =
      value.ledger.peak_persistence_pinned_bytes;
  auto categories = json::Value::object();
  constexpr const char* names[]{"backing_free",         "backing_assigned",
                                "backing_materialized", "private_state",
                                "private_tail",         "metadata",
                                "transfer_staging"};
  for (std::size_t i = 0; i < cache::kResourceCategoryCount; ++i) {
    auto gauge = json::Value::object();
    gauge["bytes"] = value.ledger.bytes[i];
    gauge["reserved_bytes"] = value.ledger.reserved_bytes[i];
    gauge["peak_bytes"] = value.ledger.peak_bytes[i];
    categories[names[i]] = std::move(gauge);
  }
  result["global_categories"] = std::move(categories);
  if (value.persistence) {
    const auto& stats = *value.persistence;
    auto queue = json::Value::object();
    queue["accepted_jobs"] = stats.accepted;
    queue["coalesced_jobs"] = stats.coalesced;
    queue["superseded_jobs"] = stats.superseded;
    queue["skipped_jobs"] = stats.skipped;
    queue["published_jobs"] = stats.completed;
    queue["failed_jobs"] = stats.failed;
    queue["pending_jobs"] = stats.pending;
    queue["peak_pending_jobs"] = stats.peak_pending;
    // Queue capacities sum source ownership per job. The ledger above counts
    // shared physical allocations once across all persistence pins.
    queue["pinned_source_capacity_bytes"] = stats.pinned_bytes;
    queue["peak_pinned_source_capacity_bytes"] = stats.peak_pinned_bytes;
    queue["transferred_bytes"] = stats.timing.bytes;
    queue["yield_ms"] = stats.timing.yield_ns / 1e6;
    queue["filesystem_ms"] = stats.timing.filesystem_ns / 1e6;
    queue["fsync_ms"] = stats.timing.fsync_ns / 1e6;
    queue["checksum_ms"] = stats.timing.checksum_ns / 1e6;
    result["global_persistence"] = std::move(queue);
  }
  return result;
}
}  // namespace gufo::server
#endif
