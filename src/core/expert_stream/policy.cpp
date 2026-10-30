#include "src/core/expert_stream/policy.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace gufo::core::expert_stream {
namespace {

// Rows at the end of a prefill batch that are always admitted, and the
// decay horizon in layer prepares (about 40 tokens of a 40-layer model).
constexpr std::uint64_t kRecentRows = 16;
constexpr double kDecayTicks = 1600.0;
// Upper bound on one plan's read tasks, checked before any state changes.
constexpr std::uint64_t kMaxReadTasks = 1ull << 20;

bool AddChecked(std::uint64_t a, std::uint64_t b, std::uint64_t* sum) {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    return false;
  }
  *sum = a + b;
  return true;
}

bool ValidRoute(const ExpertRoute& route) {
  std::uint64_t total = 0;
  for (const SourceRange& range : route.source) {
    std::uint64_t end = 0;
    if (range.bytes == 0 || !AddChecked(range.offset, range.bytes, &end) ||
        !AddChecked(total, range.bytes, &total)) {
      return false;
    }
  }
  return total == route.encoded_bytes;
}

bool SameSource(const ExpertRoute& a, const ExpertRoute& b) {
  for (std::uint32_t p = 0; p < kProjections; ++p) {
    if (a.source[p].shard != b.source[p].shard ||
        a.source[p].offset != b.source[p].offset ||
        a.source[p].bytes != b.source[p].bytes) {
      return false;
    }
  }
  return a.encoded_bytes == b.encoded_bytes;
}

}  // namespace

ExpertPool::ExpertPool(PoolConfig config) : config_(std::move(config)) {
  const auto& classes = config_.class_bytes;
  valid_ = !classes.empty() && classes.front() > 0 && config_.cap_bytes > 0 &&
           config_.max_read_bytes > 0 &&
           std::adjacent_find(classes.begin(), classes.end(),
                              std::greater_equal<>()) == classes.end();
  free_.resize(classes.size());
}

double ExpertPool::Decayed(const Entry& entry) const {
  return entry.score /
         (1.0 + static_cast<double>(tick_ - entry.last) / kDecayTicks);
}

bool ExpertPool::Grow(std::uint32_t size_class) {
  const std::uint64_t slot_bytes = config_.class_bytes[size_class];
  const std::uint64_t room = config_.cap_bytes - stats_.allocated_bytes;
  if (room < slot_bytes) {
    return false;
  }
  const std::uint64_t count = std::max<std::uint64_t>(
      1, std::min(config_.chunk_bytes, room) / slot_bytes);
  for (std::uint64_t i = 0; i < count; ++i) {
    free_[size_class].push_back(static_cast<std::uint32_t>(slots_.size()));
    slots_.push_back({size_class, 0, std::nullopt, 0});
  }
  stats_.allocated_bytes += count * slot_bytes;
  stats_.peak_allocated_bytes =
      std::max(stats_.peak_allocated_bytes, stats_.allocated_bytes);
  return true;
}

std::optional<std::uint32_t> ExpertPool::Acquire(std::uint32_t size_class,
                                                 double newcomer,
                                                 bool allow_evict) {
  if (free_[size_class].empty()) {
    Grow(size_class);
  }
  // Once a class cannot grow, the smallest larger free slot serves.
  for (std::size_t c = size_class; c < free_.size(); ++c) {
    if (!free_[c].empty()) {
      const std::uint32_t slot = free_[c].back();
      free_[c].pop_back();
      return slot;
    }
  }
  if (!allow_evict) {
    return std::nullopt;
  }
  std::optional<std::uint32_t> victim;
  double lowest = std::numeric_limits<double>::infinity();
  for (std::uint32_t i = 0; i < slots_.size(); ++i) {
    const Slot& slot = slots_[i];
    // Pinned or filling slots are never victims; nor is a smaller class.
    if (slot.pins != 0 || !slot.key || slot.size_class < size_class) {
      continue;
    }
    const Entry& entry = entries_.at(*slot.key);
    const double rank = config_.policy == Policy::kLru
                            ? static_cast<double>(entry.last)
                            : Decayed(entry);
    if (rank < lowest) {
      lowest = rank;
      victim = i;
    }
  }
  if (!victim ||
      (config_.policy == Policy::kScanResistant && newcomer < lowest)) {
    return std::nullopt;
  }
  Slot& slot = slots_[*victim];
  entries_.at(*slot.key).slot.reset();
  stats_.resident_encoded_bytes -= slot.encoded_bytes;
  slot.key.reset();
  slot.encoded_bytes = 0;
  ++stats_.evictions;
  return victim;
}

PlanError ExpertPool::Prepare(std::span<const ExpertRoute> routes,
                              bool optional, Plan* plan) {
  if (!valid_) {
    return PlanError::kInvalidPool;
  }
  if (routes.empty()) {
    return PlanError::kEmpty;
  }
  struct Use {
    const ExpertRoute* route = nullptr;
    std::uint64_t count = 0;
    bool recent = false;
  };
  const std::uint32_t layer = routes.front().layer;
  std::uint64_t first = routes.front().step, last = first;
  for (const ExpertRoute& route : routes) {
    if (route.layer != layer) {
      return PlanError::kMixedLayers;
    }
    if (!ValidRoute(route)) {
      return PlanError::kBadRange;
    }
    first = std::min(first, route.step);
    last = std::max(last, route.step);
  }
  const std::uint64_t span = last - first;
  const std::uint64_t tokens =
      span == std::numeric_limits<std::uint64_t>::max() ? span : span + 1;
  std::map<std::uint32_t, Use> uses;  // ordered: deterministic plans
  for (const ExpertRoute& route : routes) {
    Use& use = uses[route.expert];
    if (use.route && !SameSource(*use.route, route)) {
      return PlanError::kBadRange;
    }
    use.route = &route;
    ++use.count;
    use.recent |= tokens > kRecentRows && last - route.step < kRecentRows;
  }
  std::uint64_t task_bound = 0;
  for (const auto& [expert, use] : uses) {
    const auto it = entries_.find(Key(layer, expert));
    if (it != entries_.end() && it->second.source &&
        !SameSource(*it->second.source, *use.route)) {
      return PlanError::kBadRange;
    }
    for (const SourceRange& range : use.route->source) {
      const std::uint64_t max_read = config_.max_read_bytes;
      task_bound += range.bytes / max_read + (range.bytes % max_read ? 1 : 0);
      if (task_bound > kMaxReadTasks) {
        return PlanError::kTooManyReads;
      }
    }
  }
  std::vector<ReadTask> reads;
  reads.reserve(task_bound);  // the only unbounded allocation, before mutation

  ++tick_;
  *plan = Plan{next_plan_++, layer, optional, {}, std::move(reads), {}};
  Open open{{}, layer, 0};
  for (const auto& [expert, use] : uses) {
    const std::uint64_t key = Key(layer, expert);
    const std::uint64_t bytes = use.route->encoded_bytes;
    Entry& entry = entries_[key];
    if (!entry.source) {
      entry.source = *use.route;
    }
    if (!optional) {
      (entry.slot ? stats_.route_hits : stats_.route_misses) += use.count;
      entry.score = Decayed(entry) +
                    static_cast<double>(std::min<std::uint64_t>(use.count, 4)) +
                    (use.recent ? 4.0 : 0.0);
      entry.last = tick_;
    }
    if (entry.slot) {
      if (!optional) {
        ++slots_[*entry.slot].pins;
        plan->leases.push_back({LeaseKind::kHit, expert, *entry.slot, bytes});
      }
      continue;
    }
    if (in_flight_.contains(key)) {
      if (!optional) {
        plan->deferred.push_back({expert, DeferReason::kInFlight});
        ++stats_.deferred;
      }
      continue;
    }
    // A large prefill admits only its final rows and unusually frequent
    // experts, so a scan does not displace the decode working set.
    const bool admit =
        !optional &&
        (config_.policy == Policy::kLru || use.route->phase == Phase::kDecode ||
         tokens <= kRecentRows || use.recent ||
         use.count >= std::max<std::uint64_t>(4, tokens / 32));
    const auto& classes = config_.class_bytes;
    const auto fit = std::lower_bound(classes.begin(), classes.end(), bytes);
    std::optional<std::uint32_t> slot;
    if (fit != classes.end()) {
      slot = Acquire(static_cast<std::uint32_t>(fit - classes.begin()),
                     Decayed(entry), admit);
    }
    if (slot) {
      Slot& s = slots_[*slot];
      s.key = key;
      s.encoded_bytes = bytes;
      s.pins = 1;
      in_flight_.insert(key);
      plan->leases.push_back({LeaseKind::kFill, expert, *slot, bytes});
    } else if (optional) {
      continue;
    } else if (bytes <= config_.stage_bytes - stats_.stage_in_use_bytes) {
      stats_.stage_in_use_bytes += bytes;
      open.stage_bytes += bytes;
      in_flight_.insert(key);
      plan->leases.push_back({LeaseKind::kStage, expert, 0, bytes});
    } else {
      const bool oversized =
          fit == classes.end() && bytes > config_.stage_bytes;
      plan->deferred.push_back(
          {expert, oversized ? DeferReason::kOversized : DeferReason::kNoSlot});
      ++stats_.deferred;
    }
  }

  // Coalesce file-adjacent projections of missing experts, by shard/offset;
  // split any projection larger than max_read_bytes.
  struct Piece {
    std::uint32_t projection;
    SourceRange range;
    std::uint32_t lease;
  };
  std::vector<Piece> pieces;
  for (std::uint32_t i = 0; i < plan->leases.size(); ++i) {
    const Lease& lease = plan->leases[i];
    if (lease.kind == LeaseKind::kHit) {
      continue;
    }
    const ExpertRoute& route = *uses.at(lease.expert).route;
    for (std::uint32_t p = 0; p < kProjections; ++p) {
      pieces.push_back({p, route.source[p], i});
    }
  }
  std::sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) {
    return std::tie(a.projection, a.range.shard, a.range.offset) <
           std::tie(b.projection, b.range.shard, b.range.offset);
  });
  const std::uint64_t max_read = config_.max_read_bytes;
  std::uint64_t read_bytes = 0;
  for (const Piece& piece : pieces) {
    read_bytes += piece.range.bytes;
    if (!plan->reads.empty()) {
      ReadTask& task = plan->reads.back();
      if (task.projection == piece.projection &&
          task.range.shard == piece.range.shard && task.skip == 0 &&
          task.range.offset + task.range.bytes == piece.range.offset &&
          piece.range.bytes <= max_read - task.range.bytes) {
        task.range.bytes += piece.range.bytes;
        task.leases.push_back(piece.lease);
        continue;
      }
    }
    // Advance by the emitted length: skip never passes the projection size.
    for (std::uint64_t skip = 0; skip < piece.range.bytes;) {
      const std::uint64_t bytes = std::min(max_read, piece.range.bytes - skip);
      plan->reads.push_back(
          {piece.projection,
           {piece.range.shard, piece.range.offset + skip, bytes},
           {piece.lease},
           skip});
      skip += bytes;
    }
  }
  (optional ? stats_.optional_read_bytes : stats_.demand_read_bytes) +=
      read_bytes;
  stats_.read_tasks += plan->reads.size();
  open.leases = plan->leases;
  open_.emplace(plan->id, std::move(open));
  return PlanError::kNone;
}

PlanError ExpertPool::Finish(std::uint64_t plan_id, bool publish) {
  const auto it = open_.find(plan_id);
  if (it == open_.end()) {
    return PlanError::kUnknownPlan;
  }
  const Open& open = it->second;
  for (const Lease& lease : open.leases) {
    const std::uint64_t key = Key(open.layer, lease.expert);
    if (lease.kind == LeaseKind::kStage) {
      in_flight_.erase(key);
      continue;
    }
    Slot& slot = slots_[lease.slot];
    --slot.pins;
    if (lease.kind != LeaseKind::kFill) {
      continue;
    }
    in_flight_.erase(key);
    if (publish) {
      entries_[key].slot = lease.slot;
      stats_.resident_encoded_bytes += slot.encoded_bytes;
    } else {
      slot.key.reset();
      slot.encoded_bytes = 0;
      free_[slot.size_class].push_back(lease.slot);
    }
  }
  stats_.stage_in_use_bytes -= open.stage_bytes;
  open_.erase(it);
  return PlanError::kNone;
}

PlanError ExpertPool::Complete(std::uint64_t plan_id) {
  return Finish(plan_id, true);
}

PlanError ExpertPool::Cancel(std::uint64_t plan_id) {
  return Finish(plan_id, false);
}

bool ExpertPool::Resident(std::uint32_t layer, std::uint32_t expert) const {
  const auto it = entries_.find(Key(layer, expert));
  return it != entries_.end() && it->second.slot.has_value();
}

void ExpertPool::ResetCounters() {
  const Stats gauges = stats_;
  stats_ = Stats{};
  stats_.allocated_bytes = gauges.allocated_bytes;
  stats_.peak_allocated_bytes = gauges.allocated_bytes;
  stats_.resident_encoded_bytes = gauges.resident_encoded_bytes;
  stats_.stage_in_use_bytes = gauges.stage_in_use_bytes;
}

std::uint64_t ExpertPool::FragmentationBytes() const {
  std::uint64_t payload = 0;
  for (const Slot& slot : slots_) {
    payload += slot.encoded_bytes;
  }
  return stats_.allocated_bytes - payload;
}

}  // namespace gufo::core::expert_stream
