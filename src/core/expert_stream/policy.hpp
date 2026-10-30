#ifndef GUFO_CORE_EXPERT_STREAM_POLICY_HPP_
#define GUFO_CORE_EXPERT_STREAM_POLICY_HPP_

// CPU policy for streaming MoE experts from storage into a byte-bounded slot
// pool. It plans reads and pins slots; it performs no I/O and owns no memory.
// The caller issues the planned reads and later completes or cancels the plan.
//
// Adapted from franzmoca/gufo feat/ds41 110594c9 runtime/expert_store.cpp
// (MIT, gufo contributors): slots per size class grown in chunks, the
// smallest-larger-free-slot fallback (a larger expert never takes a smaller
// slot), and the scan-resistant admission and decayed-score expressions and
// constants. The plan/lease state machine and range-based read planner are
// new; the donor's HIP and POSIX I/O are not used.
//
// Precondition: an expert's source ranges and size are fixed for the pool's
// lifetime; a Prepare whose metadata conflicts with an earlier one is refused.

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gufo::core::expert_stream {

enum class Phase : std::uint8_t { kDecode, kPrefill };
enum class Policy : std::uint8_t { kLru, kScanResistant };

inline constexpr std::uint32_t kProjections = 3;  // gate, up, down

struct SourceRange {
  std::uint32_t shard = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

/// One routed (token, expert) pair. All routes of one Prepare call share a
/// layer; duplicates of an expert are deduplicated to one read and one slot.
struct ExpertRoute {
  std::uint64_t step = 0;  // token position
  Phase phase = Phase::kDecode;
  std::uint32_t layer = 0;
  std::uint32_t expert = 0;
  std::array<SourceRange, kProjections> source{};
  std::uint64_t encoded_bytes = 0;  // must equal the sum of source bytes
};

struct PoolConfig {
  std::vector<std::uint64_t> class_bytes;  // strictly ascending slot sizes
  std::uint64_t cap_bytes = 0;             // total slot backing
  std::uint64_t chunk_bytes = 0;           // growth granularity per class
  std::uint64_t stage_bytes = 0;           // read-through staging, all plans
  std::uint64_t max_read_bytes = 32ull << 20;  // every ReadTask is <= this
  Policy policy = Policy::kScanResistant;
};

enum class LeaseKind : std::uint8_t {
  kHit,    // resident slot, read in place
  kFill,   // slot being filled by this plan's reads
  kStage,  // read-through: staged for this plan only, not cached
};

struct Lease {
  LeaseKind kind = LeaseKind::kHit;
  std::uint32_t expert = 0;
  std::uint32_t slot = 0;  // pool slot for kHit/kFill
  std::uint64_t bytes = 0;
};

enum class DeferReason : std::uint8_t {
  kNoSlot,     // every eligible slot is pinned and staging is full
  kInFlight,   // another open plan is still filling or staging this expert
  kOversized,  // larger than the largest slot class and the stage
};

struct Deferred {
  std::uint32_t expert = 0;
  DeferReason reason = DeferReason::kNoSlot;
};

/// One read of adjacent expert projections; `leases` index Plan::leases in
/// file order. A projection larger than max_read_bytes is split into several
/// one-lease tasks; `skip` is the task's byte offset within that projection
/// (0 for whole or merged projections).
struct ReadTask {
  std::uint32_t projection = 0;
  SourceRange range;
  std::vector<std::uint32_t> leases;
  std::uint64_t skip = 0;
};

struct Plan {
  std::uint64_t id = 0;
  std::uint32_t layer = 0;
  bool optional = false;
  std::vector<Lease> leases;  // pinned until Complete or Cancel
  std::vector<ReadTask> reads;
  std::vector<Deferred> deferred;  // retry after other plans release
};

enum class PlanError : std::uint8_t {
  kNone,
  kInvalidPool,
  kEmpty,
  kMixedLayers,
  kBadRange,  // overflow, size mismatch, or metadata conflicting with an
              // earlier plan for the same expert
  kUnknownPlan,
  kTooManyReads,  // more than 2^20 read tasks at this max_read_bytes
};

struct Stats {
  std::uint64_t route_hits = 0;
  std::uint64_t route_misses = 0;
  std::uint64_t demand_read_bytes = 0;
  std::uint64_t optional_read_bytes = 0;
  std::uint64_t read_tasks = 0;
  std::uint64_t evictions = 0;
  std::uint64_t deferred = 0;
  std::uint64_t allocated_bytes = 0;  // slot backing, all classes
  std::uint64_t peak_allocated_bytes = 0;
  std::uint64_t resident_encoded_bytes = 0;  // payload in resident slots
  std::uint64_t stage_in_use_bytes = 0;
};

class ExpertPool {
public:
  explicit ExpertPool(PoolConfig config);

  [[nodiscard]] bool Valid() const { return valid_; }

  /// Plans one layer's routes. Optional (prefetch) plans never evict, stage or
  /// count route hits; their reads are reported as optional bytes.
  [[nodiscard]] PlanError Prepare(std::span<const ExpertRoute> routes,
                                  bool optional, Plan* plan);
  /// Publishes the plan's fills and unpins its slots and stage bytes.
  [[nodiscard]] PlanError Complete(std::uint64_t plan_id);
  /// Releases the plan's fills unpublished and unpins everything.
  [[nodiscard]] PlanError Cancel(std::uint64_t plan_id);

  [[nodiscard]] bool Resident(std::uint32_t layer, std::uint32_t expert) const;
  [[nodiscard]] const Stats& stats() const { return stats_; }
  /// Clears counters (for example after a seed) but keeps byte gauges.
  void ResetCounters();
  /// Bytes allocated but not holding payload: free slots plus slack between
  /// an expert and its slot class.
  [[nodiscard]] std::uint64_t FragmentationBytes() const;

private:
  struct Slot {
    std::uint32_t size_class = 0;
    std::uint32_t pins = 0;
    std::optional<std::uint64_t> key;  // resident or filling expert
    std::uint64_t encoded_bytes = 0;
  };
  struct Entry {
    double score = 0;
    std::uint64_t last = 0;
    std::optional<std::uint32_t> slot;  // published only
    std::optional<ExpertRoute> source;  // first metadata seen (step unused)
  };
  struct Open {
    std::vector<Lease> leases;
    std::uint32_t layer = 0;
    std::uint64_t stage_bytes = 0;
  };

  static std::uint64_t Key(std::uint32_t layer, std::uint32_t expert) {
    return (std::uint64_t{layer} << 32) | expert;
  }
  [[nodiscard]] double Decayed(const Entry& entry) const;
  bool Grow(std::uint32_t size_class);
  std::optional<std::uint32_t> Acquire(std::uint32_t size_class,
                                       double newcomer, bool allow_evict);
  PlanError Finish(std::uint64_t plan_id, bool publish);

  PoolConfig config_;
  bool valid_ = false;
  std::vector<Slot> slots_;
  std::vector<std::vector<std::uint32_t>> free_;  // per class
  std::unordered_map<std::uint64_t, Entry> entries_;
  std::unordered_map<std::uint64_t, Open> open_;
  std::unordered_set<std::uint64_t> in_flight_;  // open fills and stages
  std::uint64_t tick_ = 0;
  std::uint64_t next_plan_ = 1;
  Stats stats_;
};

}  // namespace gufo::core::expert_stream

#endif  // GUFO_CORE_EXPERT_STREAM_POLICY_HPP_
