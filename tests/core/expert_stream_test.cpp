#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include "src/core/expert_stream/policy.hpp"

namespace {

using namespace gufo::core::expert_stream;

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

// Projection-major layout: projection p of expert e of size `bytes` starts at
// p * kBlock + e * bytes / 3, so consecutive experts are file-adjacent.
constexpr std::uint64_t kBlock = 1ull << 30;

ExpertRoute Route(std::uint32_t expert, std::uint64_t bytes,
                  std::uint64_t step = 0, Phase phase = Phase::kDecode,
                  std::uint32_t layer = 0) {
  ExpertRoute route;
  route.step = step;
  route.phase = phase;
  route.layer = layer;
  route.expert = expert;
  route.encoded_bytes = bytes;
  for (std::uint32_t p = 0; p < kProjections; ++p) {
    route.source[p] = {0, p * kBlock + expert * (bytes / 3), bytes / 3};
  }
  return route;
}

PoolConfig TwoSlots(Policy policy = Policy::kScanResistant) {
  PoolConfig config;
  config.class_bytes = {150, 300};
  config.cap_bytes = 600;  // two 300-byte slots
  config.chunk_bytes = 300;
  config.policy = policy;
  return config;
}

Plan Prepare(ExpertPool& pool, std::vector<ExpertRoute> routes,
             bool optional = false) {
  Plan plan;
  Expect(pool.Prepare(routes, optional, &plan) == PlanError::kNone,
         "prepare succeeds");
  return plan;
}

void TestDuplicateRoutesDedupe() {
  ExpertPool pool(TwoSlots());
  const Plan plan = Prepare(pool, {Route(5, 300, 0, Phase::kPrefill),
                                   Route(5, 300, 1, Phase::kPrefill),
                                   Route(5, 300, 2, Phase::kPrefill)});
  Expect(plan.leases.size() == 1, "one lease for one expert");
  Expect(plan.leases[0].kind == LeaseKind::kFill, "miss fills a slot");
  Expect(plan.reads.size() == 3, "one read per projection");
  Expect(pool.stats().demand_read_bytes == 300, "expert read once");
  Expect(pool.stats().route_misses == 3, "every route counted");
  Expect(pool.stats().allocated_bytes == 300, "one backing slot");
  Expect(pool.Complete(plan.id) == PlanError::kNone, "complete");
  Expect(pool.Resident(0, 5), "published");
  const Plan hit = Prepare(pool, {Route(5, 300)});
  Expect(hit.leases[0].kind == LeaseKind::kHit && hit.reads.empty(),
         "resident expert is a hit without reads");
  Expect(pool.Complete(hit.id) == PlanError::kNone, "complete hit");
  Expect(pool.stats().route_hits == 1, "hit counted");
}

void TestPinnedSlotsRefuseThenRetry() {
  ExpertPool pool(TwoSlots());
  const Plan a = Prepare(pool, {Route(1, 300)});
  const Plan b = Prepare(pool, {Route(2, 300)});
  Expect(pool.stats().allocated_bytes == 600, "cap reached");

  // Both slots are pinned by held completions: the third expert is deferred.
  const Plan c = Prepare(pool, {Route(3, 300)});
  Expect(c.leases.empty() && c.reads.empty(), "no lease past the cap");
  Expect(c.deferred.size() == 1 && c.deferred[0].reason == DeferReason::kNoSlot,
         "third expert deferred");
  Expect(pool.stats().allocated_bytes == 600 &&
             pool.stats().peak_allocated_bytes == 600,
         "cap never exceeded");
  // A second plan for an expert still being filled waits for that fill.
  const Plan dup = Prepare(pool, {Route(1, 300)});
  Expect(dup.deferred.size() == 1 &&
             dup.deferred[0].reason == DeferReason::kInFlight,
         "in-flight fill is not read twice");
  Expect(pool.stats().deferred == 2, "both deferrals counted");
  Expect(pool.Complete(c.id) == PlanError::kNone, "empty plan completes");
  Expect(pool.Complete(dup.id) == PlanError::kNone, "deferred plan completes");

  // Cancel releases b's slot unpublished; complete publishes a.
  Expect(pool.Cancel(b.id) == PlanError::kNone, "cancel");
  Expect(!pool.Resident(0, 2), "cancelled fill not published");
  Expect(pool.Complete(a.id) == PlanError::kNone, "complete");
  Expect(pool.Resident(0, 1), "completed fill published");
  Expect(pool.Complete(a.id) == PlanError::kUnknownPlan, "complete once");
  Expect(pool.Cancel(b.id) == PlanError::kUnknownPlan, "cancel once");
  Expect(pool.stats().resident_encoded_bytes == 300, "exact resident bytes");
  Expect(pool.FragmentationBytes() == 300, "one free slot");

  // Retry: the free slot serves; then a hotter fourth expert (four routes)
  // evicts an unpinned resident one rather than growing.
  const Plan retry = Prepare(pool, {Route(3, 300)});
  Expect(retry.leases.size() == 1 && retry.leases[0].kind == LeaseKind::kFill,
         "retry fills");
  Expect(pool.Complete(retry.id) == PlanError::kNone, "complete retry");
  const Plan weak = Prepare(pool, {Route(4, 300)});
  Expect(weak.deferred.size() == 1 && pool.stats().evictions == 0,
         "a colder newcomer does not displace the working set");
  Expect(pool.Complete(weak.id) == PlanError::kNone, "complete weak");
  const Plan fourth = Prepare(
      pool,
      {Route(4, 300, 10, Phase::kPrefill), Route(4, 300, 11, Phase::kPrefill),
       Route(4, 300, 12, Phase::kPrefill), Route(4, 300, 13, Phase::kPrefill)});
  Expect(fourth.leases.size() == 1 && pool.stats().evictions == 1,
         "unpinned resident slot evicted");
  Expect(pool.Complete(fourth.id) == PlanError::kNone, "complete fourth");
  Expect(pool.stats().allocated_bytes == 600, "still at cap");
}

void TestStageReadThrough() {
  PoolConfig config = TwoSlots();
  config.stage_bytes = 300;
  ExpertPool pool(config);
  const Plan a = Prepare(pool, {Route(1, 300), Route(2, 300)});
  const Plan b = Prepare(pool, {Route(3, 300), Route(4, 300)});
  Expect(b.leases.size() == 1 && b.leases[0].kind == LeaseKind::kStage,
         "pinned cap stages one expert");
  Expect(b.deferred.size() == 1, "stage full defers the other");
  Expect(pool.stats().stage_in_use_bytes == 300, "stage accounted");
  Expect(pool.Complete(b.id) == PlanError::kNone, "complete");
  Expect(pool.stats().stage_in_use_bytes == 0, "stage released");
  Expect(!pool.Resident(0, 3), "staged expert not cached");
  Expect(pool.Complete(a.id) == PlanError::kNone, "complete");
}

void TestSizeClasses() {
  PoolConfig config = TwoSlots();
  config.chunk_bytes = 150;  // one slot per growth step
  ExpertPool pool(config);
  const Plan small = Prepare(pool, {Route(1, 150)});
  Expect(small.leases[0].kind == LeaseKind::kFill,
         "small expert in its own class");
  const Plan big = Prepare(pool, {Route(2, 300)});
  Expect(pool.stats().allocated_bytes == 450, "two classes allocated");
  Expect(pool.FragmentationBytes() == 0, "no slack");
  // Only 150 bytes of cap remain: a large expert cannot grow or use a smaller
  // slot; a small one can grow its own class.
  const Plan big2 = Prepare(pool, {Route(3, 300)});
  Expect(big2.deferred.size() == 1, "large never takes a small slot");
  const Plan small2 = Prepare(pool, {Route(4, 150)});
  Expect(small2.leases.size() == 1 && pool.stats().allocated_bytes == 600,
         "small class grows to the cap");
  for (const Plan* plan : {&small, &big, &big2, &small2}) {
    Expect(pool.Complete(plan->id) == PlanError::kNone, "complete");
  }
  // Cap exhausted and the small class has no free slot: once the large slot
  // is free, a small expert may take it.
  ExpertPool spill(TwoSlots());
  const Plan l1 = Prepare(spill, {Route(1, 300)});
  const Plan l2 = Prepare(spill, {Route(2, 300)});
  Expect(spill.Cancel(l2.id) == PlanError::kNone, "free one large slot");
  const Plan s = Prepare(spill, {Route(3, 150)});
  Expect(s.leases.size() == 1 && spill.FragmentationBytes() == 150,
         "small expert takes the smallest larger free slot");
  Expect(spill.Complete(l1.id) == PlanError::kNone &&
             spill.Complete(s.id) == PlanError::kNone,
         "complete");
}

void TestCoalescing() {
  PoolConfig config = TwoSlots();
  config.cap_bytes = 3000;
  ExpertPool pool(config);
  const Plan plan =
      Prepare(pool, {Route(4, 300), Route(5, 300), Route(9, 300)});
  Expect(plan.reads.size() == 6, "4+5 coalesce per projection, 9 separate");
  for (const ReadTask& read : plan.reads) {
    Expect(read.range.bytes == (read.leases.size() == 2 ? 200u : 100u),
           "coalesced range covers both experts");
  }
  Expect(pool.stats().demand_read_bytes == 900, "bytes unchanged by merging");
  Expect(pool.Complete(plan.id) == PlanError::kNone, "complete");

  config.max_read_bytes = 150;
  ExpertPool capped(config);
  const Plan split = Prepare(capped, {Route(4, 300), Route(5, 300)});
  Expect(split.reads.size() == 6, "merge cap splits adjacent reads");
  Expect(capped.Complete(split.id) == PlanError::kNone, "complete");
}

void TestRefusals() {
  PoolConfig config = TwoSlots();
  config.stage_bytes = 500;
  ExpertPool pool(config);
  const Plan big = Prepare(pool, {Route(1, 450)});
  Expect(big.leases.size() == 1 && big.leases[0].kind == LeaseKind::kStage,
         "larger than every class stages");
  Expect(pool.Complete(big.id) == PlanError::kNone, "complete");
  const Plan huge = Prepare(pool, {Route(2, 900)});
  Expect(huge.deferred.size() == 1 &&
             huge.deferred[0].reason == DeferReason::kOversized,
         "oversized refused");
  Expect(pool.Complete(huge.id) == PlanError::kNone, "complete");

  Plan plan;
  ExpertRoute overflow = Route(3, 300);
  overflow.source[1].offset = std::numeric_limits<std::uint64_t>::max() - 10;
  Expect(
      pool.Prepare(std::vector{overflow}, false, &plan) == PlanError::kBadRange,
      "offset overflow refused");
  ExpertRoute mismatch = Route(3, 300);
  mismatch.encoded_bytes = 301;
  Expect(
      pool.Prepare(std::vector{mismatch}, false, &plan) == PlanError::kBadRange,
      "encoded bytes must match ranges");
  ExpertRoute alias = Route(3, 300);
  alias.source[0].offset += 3;
  Expect(pool.Prepare(std::vector{Route(3, 300), alias}, false, &plan) ==
             PlanError::kBadRange,
         "one expert, one source");
  Expect(pool.Prepare(
             std::vector{Route(3, 300), Route(4, 300, 0, Phase::kDecode, 1)},
             false, &plan) == PlanError::kMixedLayers,
         "one layer per plan");
  Expect(pool.Prepare({}, false, &plan) == PlanError::kEmpty, "empty");
  Expect(pool.Cancel(12345) == PlanError::kUnknownPlan, "unknown plan");

  PoolConfig bad = TwoSlots();
  bad.class_bytes = {300, 300};
  ExpertPool invalid(bad);
  Expect(!invalid.Valid(), "classes must ascend");
  Expect(invalid.Prepare(std::vector{Route(1, 300)}, false, &plan) ==
             PlanError::kInvalidPool,
         "invalid pool refuses");
  Expect(pool.stats().allocated_bytes == 0, "refusals allocate nothing");
}

// A hot decode working set (eight rounds), then a 64-token prefill scan of new
// experts, then the working set again. The scan's final 16 rows are admitted
// (score 5) but cannot displace the hotter working set.
std::uint64_t WorkingSetHitsAfterScan(Policy policy) {
  PoolConfig config = TwoSlots(policy);
  config.cap_bytes = 1200;  // four slots
  config.stage_bytes = 300 * 64;
  ExpertPool pool(config);
  std::uint64_t step = 0;
  auto decode = [&] {
    for (std::uint32_t e = 0; e < 4; ++e) {
      const Plan plan = Prepare(pool, {Route(e, 300, step++)});
      Expect(pool.Complete(plan.id) == PlanError::kNone, "complete decode");
    }
  };
  for (int round = 0; round < 8; ++round) {
    decode();
  }
  std::vector<ExpertRoute> scan;
  for (std::uint32_t t = 0; t < 64; ++t) {
    scan.push_back(Route(100 + t, 300, step + t, Phase::kPrefill));
  }
  step += 64;
  const Plan plan = Prepare(pool, scan);
  Expect(pool.Complete(plan.id) == PlanError::kNone, "complete scan");
  const std::uint64_t before = pool.stats().route_hits;
  decode();
  return pool.stats().route_hits - before;
}

void TestScanResistance() {
  Expect(WorkingSetHitsAfterScan(Policy::kScanResistant) == 4,
         "scored policy keeps the decode working set");
  Expect(WorkingSetHitsAfterScan(Policy::kLru) == 0,
         "LRU baseline loses it to the scan");
}

void TestOptionalNeverEvicts() {
  ExpertPool pool(TwoSlots());
  const Plan seed = Prepare(pool, {Route(1, 300), Route(2, 300)}, true);
  Expect(seed.leases.size() == 2, "optional fills free capacity");
  Expect(pool.stats().optional_read_bytes == 600 &&
             pool.stats().demand_read_bytes == 0,
         "optional bytes reported separately");
  Expect(pool.stats().route_misses == 0, "optional routes not counted");
  Expect(pool.Complete(seed.id) == PlanError::kNone, "complete seed");
  const Plan extra = Prepare(pool, {Route(3, 300)}, true);
  Expect(extra.leases.empty() && extra.deferred.empty() &&
             pool.stats().evictions == 0,
         "optional plan drops instead of evicting");
  Expect(pool.Complete(extra.id) == PlanError::kNone, "complete");
  pool.ResetCounters();
  Expect(pool.stats().optional_read_bytes == 0 &&
             pool.stats().allocated_bytes == 600 &&
             pool.stats().resident_encoded_bytes == 600,
         "reset keeps gauges");
}

}  // namespace

int main() {
  TestDuplicateRoutesDedupe();
  TestPinnedSlotsRefuseThenRetry();
  TestStageReadThrough();
  TestSizeClasses();
  TestCoalescing();
  TestRefusals();
  TestScanResistance();
  TestOptionalNeverEvicts();
  std::cout << "expert_stream_test: PASS\n";
  return 0;
}
