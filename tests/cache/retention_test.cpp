#include "src/cache/retention.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <future>
#include <map>
#include <set>
#include <stdexcept>

#include "src/cache/backing.hpp"
#include "tests/cache/fake_adapter.hpp"
#include "tests/cache/prefix_index_fixture.hpp"
using namespace gufo::cache;
using namespace gufo::cache::testing;
namespace {
struct Log final : RetentionEventSink {
  std::vector<RetentionEvent> events;
  void Emit(const RetentionEvent& e) noexcept override { events.push_back(e); }
};
struct Fixture {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  Log log;
  RetentionPolicy policy;
  explicit Fixture(std::size_t limit = 128)
      : policy(ledger, index, &log, limit) {
    index.Register({1}, kIndexComponents);
  }
  CheckpointId Add(std::vector<Token> tokens,
                   RetentionPurpose purpose = RetentionPurpose::kContinuation,
                   Rows stable = 0, CheckpointId source = {},
                   bool preserve = true, Identity input = {}) {
    assert(policy.Admit(
        {tokens, {1}, input, purpose, stable, source, preserve, {}},
        [&] { return IndexCheckpoint(ledger, tokens, {1}, input); }));
    return log.events.back().checkpoint;
  }
  Rows Cached(std::vector<Token> tokens, Identity input = {}) {
    return index.CachedPrefixTokens(
        {{1}, tokens, InputIdentity(tokens.size(), input), 0, true});
  }
};
void Planning() {
  FakeAdapter adapter;
  struct Empty final : MutationGuard {
    void BeforeOverwrite(ComponentId, Rows, Rows) override {}
    void BeforeRelease(ComponentId, Rows, Rows) noexcept override {}
  } guard;
  auto slot = adapter.CreateSlot(guard);
  auto plan = PlanCaptures(adapter, *slot, {10000, 0, 1500, 0, true, true, {}});
  std::vector<Rows> expected{1500, 2048, 4096, 6144, 8192, 10000};
  assert(plan.size() == expected.size());
  for (std::size_t i = 0; i < plan.size(); ++i) {
    assert(plan[i].boundary == expected[i]);
    assert(plan[i].splits_pass == (plan[i].boundary == 1500));
  }
  plan = PlanCaptures(adapter, *slot, {2176, 0, 0, 0, true, true, {}});
  assert(plan.size() == 1 && plan[0].boundary == 2176);
  plan = PlanCaptures(adapter, *slot, {2177, 0, 0, 0, true, true, {}});
  assert(plan.size() == 2 && plan[0].boundary == 2048);
  plan = PlanCaptures(adapter, *slot, {5000, 4000, 4500, 4511, true, true, {}});
  assert(plan.size() == 3);  // less than 512 improvement: no learned point/grid
  plan = PlanCaptures(adapter, *slot, {6000, 4000, 4500, 4512, true, true, {}});
  assert(plan.size() == 3 && plan[1].boundary == 4500 && plan[1].required);
  plan = PlanCaptures(adapter, *slot, {6000, 4000, 4500, 4600, true, true, {}});
  assert(plan.size() == 4 && plan[2].purpose == CheckpointPurpose::kLearned);
  adapter.SetPrefillPassRows(2000);
  plan = PlanCaptures(adapter, *slot, {10000, 0, 1501, 0, true, true, {}});
  assert(plan[0].boundary == 1501 && plan[0].required && plan[0].splits_pass);
  assert(plan[1].boundary == 2048 && plan[1].splits_pass);
  assert(plan[2].boundary == 4000 && !plan[2].splits_pass);
  assert(PlanCaptures(adapter, *slot, {}).empty());
  std::array<Rows, 1> retained{2048};
  plan = PlanCaptures(adapter, *slot, {4096, 0, 0, 0, true, false, retained});
  assert(plan.size() == 1 && plan[0].boundary == 4096);
  // A small budget keeps only the shallowest grid; a retained one still counts.
  adapter.SetPrefillPassRows(2048);
  plan = PlanCaptures(adapter, *slot,
                      {10000, 0, 1500, 0, true, false, {}, std::size_t{1}});
  assert(plan.size() == 3 && plan[1].boundary == 2048 &&
         plan[1].purpose == CheckpointPurpose::kGrid);
  plan = PlanCaptures(adapter, *slot,
                      {10000, 0, 1500, 0, true, false, retained, 1});
  assert(plan.size() == 2 && plan[1].boundary == 10000);
  assert(PlanCaptures(adapter, *slot, {10000, 0, 1500, 0, true, false, {}, 0})
             .size() == 2);
  // Flash-Next keeps about 133 MB of private state per checkpoint.
  assert(GridLimit(2ULL << 30, 133'100'000) == 1);
  assert(GridLimit(8ULL << 30, 133'100'000) == 4);
  assert(GridLimit(1ULL << 30, 133'100'000) == 0);
  assert(GridLimit(1ULL << 30, 0) == 4);
}
void Branches() {
  for (bool branched : {false, true}) {
    Fixture f(4);
    f.Add({9, 9, 9});
    const auto root = f.Add({1, 2, 3});
    f.Add({1, 2, 3, 4, 4});
    if (branched)
      f.Add({1, 2, 3, 5, 5});
    else
      f.Add({7, 7});
    assert(f.policy.Rank(root) == (branched ? 3 : 2));
    f.Add({8, 8, 8, 8});
    assert(f.Cached({1, 2, 3, 6}) == (branched ? 3 : 0));
    assert(f.Cached({9, 9, 9, 1}) == (branched ? 0 : 3));
  }
  {
    Fixture f(8);
    f.Add({9, 9, 9});
    f.Add({7, 7, 7});
    const auto learned = f.Add({1, 2, 3}, RetentionPurpose::kBranchPoint);
    f.Add({1, 2, 3, 4, 4});
    std::vector<Token> incoming{1, 2, 3, 5, 5};
    int attempt = 0;
    assert(f.policy.Admit(
        {incoming, {1}, {}, RetentionPurpose::kContinuation, 0, learned, false},
        [&] {
          if (!attempt++)
            throw ResourceExhausted();
          return IndexCheckpoint(f.ledger, incoming);
        }));
    assert(f.Cached({1, 2, 3, 6}) == 3 && f.Cached({9, 9, 9, 1}) == 0);
  }
  {
    Fixture f(4);
    f.Add({9, 9, 9});
    const auto shallow = f.Add({1, 2, 3}, RetentionPurpose::kBranchPoint);
    const auto deep = f.Add({1, 2, 3, 4, 5}, RetentionPurpose::kBranchPoint);
    f.Add({1, 2, 3, 4, 5, 6});
    assert(f.policy.Rank(shallow) == 2 && f.policy.Rank(deep) == 3);
    f.Add({8, 8, 8});
    assert(f.Cached({1, 2, 3, 7}) == 0 && f.Cached({1, 2, 3, 4, 5, 7}) == 5);
  }
  {
    Fixture f(2);
    f.Add({1, 2, 3}, RetentionPurpose::kBranchPoint);
    f.Add({1, 2, 3});  // exact replacement retains learned status
    const auto id = f.log.events.back().checkpoint;
    f.Add({1, 2, 3, 4});
    assert(f.policy.Rank(id) == 3);
  }
}
void RanksAndSources() {
  Fixture f(4);
  auto root = f.Add({1, 2});
  auto retry = f.Add({1, 2, 3}, RetentionPurpose::kRetry, 2);
  auto history = f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory);
  auto tail = f.Add({1, 2, 3, 4, 5});
  assert(f.policy.Rank(retry) == 0 && f.policy.Rank(history) == 1 &&
         f.policy.Rank(root) == 2 && f.policy.Rank(tail) == 3);
  f.Add({7, 7});
  assert(f.log.events[f.log.events.size() - 2].checkpoint == retry);
  f.Add({8, 8});
  assert(f.log.events[f.log.events.size() - 2].checkpoint == history);
  for (const bool preserve : {true, false}) {
    Fixture g(2);
    const auto other = g.Add({9, 9});
    const auto source = g.Add({1, 2});
    g.Add({1, 2, 3}, RetentionPurpose::kContinuation, 0, source, preserve);
    assert(g.Cached({9, 9, 1}) == 2 &&
           g.log.events[g.log.events.size() - 2].checkpoint == source);
    g.policy.Touch(other);
    g.Add({7, 7});
    assert(g.Cached({9, 9, 1}) == 2);  // touch advances LRU
  }
  {
    Fixture g(3);
    const auto root_id = g.Add({1, 2});
    g.Add({9, 9});
    const auto old_tail = g.Add({1, 2, 3, 4});
    g.Add({1, 2, 5}, RetentionPurpose::kContinuation, 0, root_id);
    assert(g.log.events[g.log.events.size() - 2].checkpoint == old_tail);
    assert(g.log.events[g.log.events.size() - 2].reason ==
           RetentionReason::kEditedTail);
  }
  {
    Fixture g(1);
    g.Add({1, 2});
    std::vector<Token> t{9};
    bool called = false;
    assert(!g.policy.Admit({t, {1}, {}, RetentionPurpose::kHistory}, [&] {
      called = true;
      return IndexCheckpoint(g.ledger, t);
    }));
    assert(!called && g.policy.Size() == 1);
  }
  {
    Fixture g(2);
    auto a = g.Add({1, 2}, RetentionPurpose::kContinuation, 0, {}, true, {1});
    g.Add({1, 2, 3}, RetentionPurpose::kContinuation, 0, {}, true, {2});
    assert(g.policy.Rank(a) == 3);  // image identities never cover each other
  }
}
void LongPrefixRelationsAfterReplacement() {
  Fixture f(4);
  const std::vector<Token> prefix(131072, 7);
  auto left = prefix, right = prefix;
  left.push_back(8);
  right.push_back(9);
  const auto root = f.Add(prefix);
  const auto a = f.Add(left);
  const auto b = f.Add(right);
  assert(f.policy.Rank(root) == 3);  // both continuations need their fork
  f.policy.Remove(a);
  assert(f.policy.Rank(root) == 2);
  // Reuse the removed record's slot with a shorter, unrelated conversation.
  // Its tokens cannot be indexed at the old parent's boundary.
  const auto unrelated = f.Add({1, 2});
  assert(f.policy.Rank(root) == 2);
  f.policy.Remove(b);
  assert(f.policy.Rank(root) == 3);
  (void)f.Add(right, RetentionPurpose::kContinuation, 0, {}, true, {42});
  assert(f.policy.Rank(root) == 3);  // another input cannot cover the prefix
  f.policy.Remove(unrelated);
  f.Add(right);
  assert(f.policy.Rank(root) == 2);
  f.Add(right, RetentionPurpose::kBranchPoint);  // replace the same boundary
  assert(f.policy.Rank(root) == 2);
  f.Add({10, 11});
  f.Add({12, 13});
  assert(f.Cached(prefix) == 0);  // covered root still loses at record pressure
  assert(f.Cached(right) == right.size());
  assert(f.Cached(right, {42}) == right.size());
}
void ShortPeersDoNotCoverHistory() {
  for (const auto purpose :
       {RetentionPurpose::kContinuation, RetentionPurpose::kBranchPoint}) {
    Fixture f(3);
    const auto root = f.Add({1, 2}, purpose);
    const auto short_tail = f.Add({1, 2, 7});
    const auto history = f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory);
    // A shared system prefix cannot restore the deeper history's rows.
    assert(f.policy.Rank(history) == 3);
    f.Add({1, 2, 8}, RetentionPurpose::kContinuation, 0, root);
    assert(f.log.events[f.log.events.size() - 2].checkpoint == short_tail);
    assert(f.Cached({1, 2, 3, 4, 5}) == 4);
    assert(f.Cached({1, 2, 8, 9}) == 3);
    assert(f.Cached({1, 2, 7, 9}) == 2);
  }
  {
    Fixture f(3);
    const auto history = f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory);
    f.Add({1, 2, 3, 4, 5});
    assert(f.policy.Rank(history) == 1);  // a deeper peer really covers it
  }
  {
    Fixture f(3);
    const auto history =
        f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory, 0, {}, true, {1});
    f.Add({1, 2, 3, 4, 5}, RetentionPurpose::kContinuation, 0, {}, true, {2});
    assert(f.policy.Rank(history) == 3);  // different inputs cannot cover it
  }
}
void HistoryBytePressureCanReplaceOldFamilies() {
  Fixture f(4);
  const auto old = f.Add({9, 9});
  f.Add({8, 8});
  std::vector<Token> history{1, 2};
  unsigned attempts = 0;
  assert(f.policy.Admit({history, {1}, {}, RetentionPurpose::kHistory}, [&] {
    if (!attempts++)
      throw ResourceExhausted();
    return IndexCheckpoint(f.ledger, history);
  }));
  assert(attempts == 2 && f.Cached({1, 2, 3}) == 2);
  assert(f.Cached({9, 9, 1}) == 0 && f.Cached({8, 8, 1}) == 2);
  assert(f.log.events[f.log.events.size() - 2].checkpoint == old);
  assert(f.log.events[f.log.events.size() - 2].reason ==
         RetentionReason::kByteCapacity);
}
void CurrentTurnKeepsItsEditBoundaries() {
  Fixture f(4);
  const auto old = f.Add({9, 9});
  const auto early = f.Add({1, 2}, RetentionPurpose::kHistory);
  const auto later = f.Add({1, 2, 3, 4}, RetentionPurpose::kHistory);
  const auto source = f.Add({1, 2, 3, 4, 5});
  std::array protected_ids{early, later};
  std::vector<Token> incoming{1, 2, 3, 4, 5, 6};
  RetentionRequest request{
      incoming, {1}, {}, RetentionPurpose::kContinuation, 0, source, true};
  request.protected_checkpoints = protected_ids;
  assert(f.policy.Reclaim(request));
  assert(f.log.events.back().checkpoint == old);
  assert(f.Cached({1, 2, 7}) == 2 && f.Cached({1, 2, 3, 4, 7}) == 4);
  // Protection lasts for the current turn only; a later independent turn may
  // reclaim these covered histories under the same finite capacity. The
  // deeper one goes first: the shallow boundary may serve other prompts.
  request.protected_checkpoints = {};
  assert(f.policy.Reclaim(request));
  assert(f.log.events.back().checkpoint == later);
  assert(f.Cached({1, 2, 7}) == 2 && f.Cached({1, 2, 3, 4, 7}) == 2);
}
// Eight tokens of rows cost as much as one private state: shorter checkpoints
// are cheap to recompute relative to the budget they hold.
constexpr std::array<ComponentDescriptor, 2> kPressureComponents{
    {{{1}, 1, ComponentKind::kAppendRows, 64, 4, 0},
     {{2}, 1, ComponentKind::kPrivateState, 0, 0, 4096}}};
struct PressureFixture {
  ResourceLedger ledger;
  PrefixIndex index{ledger};
  Log log;
  RetentionPolicy policy;
  explicit PressureFixture(std::size_t ram = 1ULL << 30)
      : ledger({ram, ram, 0, ram}), policy(ledger, index, &log) {
    index.Register({1}, kPressureComponents);
  }
  // Each conversation executes separately, so no two share chunks. Admission
  // reclaims through the byte-pressure order when the ledger is full.
  CheckpointId Add(std::vector<Token> tokens,
                   RetentionPurpose purpose = RetentionPurpose::kContinuation,
                   bool reused = false) {
    RetentionRequest request{tokens, {1}, {}, purpose};
    request.reused = reused;
    assert(policy.Admit(request, [&] {
      return IndexCheckpoint(ledger, tokens, {1}, {}, kPressureComponents);
    }));
    return log.events.back().checkpoint;
  }
  CheckpointId Reclaim() {
    std::vector<Token> incoming{99, 99, 99, 99, 99, 99, 99, 99, 99};
    assert(policy.Reclaim({incoming, {1}, {}}));
    return log.events.back().checkpoint;
  }
  Rows Cached(std::vector<Token> tokens) {
    return index.CachedPrefixTokens(
        {{1}, tokens, InputIdentity(tokens.size()), 0, true});
  }
};
std::vector<Token> Conversation(Token first, std::size_t size) {
  std::vector<Token> tokens(size, first);
  for (std::size_t i = 0; i < size; ++i)
    tokens[i] += static_cast<Token>(i * 1000);
  return tokens;
}
void CheapCheckpointsGiveWayFirst() {
  // An idle deep conversation outlives newer short requests (cache-pressure
  // idle resume). Retry copies still go before any cheap checkpoint.
  PressureFixture f;
  const auto idle = f.Add(Conversation(1, 16));
  const auto first = f.Add(Conversation(2, 7));
  const auto second = f.Add(Conversation(3, 7));
  auto retry_tokens = Conversation(3, 7);
  retry_tokens.push_back(5);
  RetentionRequest retry{retry_tokens, {1}, {}, RetentionPurpose::kRetry, 7};
  assert(f.policy.Admit(retry, [&] {
    return IndexCheckpoint(f.ledger, retry_tokens, {1}, {},
                           kPressureComponents);
  }));
  const auto copy = f.log.events.back().checkpoint;
  assert(f.Reclaim() == copy);
  assert(f.Reclaim() == first && f.Reclaim() == second);
  assert(f.Reclaim() == idle);
}
void ReusedCheapCheckpointKeepsRecency() {
  // A short checkpoint restored one request ago is not cheap any more: its
  // next turn would be a full miss (state-edges). It competes by recency.
  PressureFixture f;
  const auto idle = f.Add(Conversation(1, 16));
  const auto restored = f.Add(Conversation(2, 7));
  const auto other = f.Add(Conversation(3, 7));
  f.policy.Touch(restored);
  assert(f.Reclaim() == other);
  assert(f.Reclaim() == idle);
  assert(f.Reclaim() == restored);
}
void PersistingCheckpointIsNotAVictim() {
  // While a disk job writes a checkpoint, removing it frees none of its
  // storage and only drops a restorable boundary from the index.
  PressureFixture f;
  const auto older = f.Add(Conversation(1, 16));
  auto tokens = Conversation(2, 16);
  std::vector<ResourceCharge> charges;
  RetentionRequest request{tokens, {1}, {}, RetentionPurpose::kContinuation};
  assert(f.policy.Admit(request, [&] {
    auto history = ExecutionHistory::Cold(f.ledger, kPressureComponents, {1});
    std::vector<ComponentPosition> positions;
    for (const auto& d : kPressureComponents)
      positions.push_back({d.id, tokens.size()});
    return history.Capture(
        {tokens, InputIdentity(tokens.size()), positions,
         CheckpointPurpose::kPrompt, 1},
        [&](const PayloadRequest& payload) {
          auto pool =
              f.ledger.Reserve(ResourceCategory::kBackingFree, payload.bytes)
                  .Convert();
          auto charge = pool.ReserveBacking(payload.category).Convert();
          charges.push_back(charge);
          return Payload::Committed(std::move(charge),
                                    std::make_shared<int>(0));
        });
  }));
  const auto written = f.log.events.back().checkpoint;
  std::vector<PersistencePin> pins;
  for (const auto& charge : charges)
    pins.push_back(charge.PinPersistence());
  // The written checkpoint is newer, so it would otherwise go second.
  f.policy.Touch(older);
  assert(f.Reclaim() == older);
  assert(!f.policy.Reclaim({std::vector<Token>(9, 99), {1}, {}}));
  pins.clear();
  assert(f.Reclaim() == written);
}
void ExecutionPressureOrder() {
  for (int repeat = 0; repeat < 2; ++repeat) {
    PressureFixture f;
    const auto old = f.Add(Conversation(1, 16));
    const auto reused =
        f.Add(Conversation(2, 16), RetentionPurpose::kContinuation, true);
    const auto cheap = f.Add(Conversation(3, 7));
    const auto source = f.Add(Conversation(4, 7));
    auto retry = Conversation(4, 7);
    retry.push_back(5);
    assert(f.policy.Admit({retry, {1}, {}, RetentionPurpose::kRetry, 7}, [&] {
      return IndexCheckpoint(f.ledger, retry, {1}, {}, kPressureComponents);
    }));
    const auto copy = f.log.events.back().checkpoint;
    assert(f.policy.Rank(copy) == 0);
    for (const auto expected : {copy, cheap, source, old, reused}) {
      const auto size = f.policy.Size();
      assert(f.policy.ReclaimForExecution());
      assert(f.policy.Size() == size - 1);
      assert(f.log.events.back().checkpoint == expected);
      assert(f.log.events.back().reason == RetentionReason::kByteCapacity);
    }
    assert(!f.policy.ReclaimForExecution());
  }
  // Protection is absolute, including a nearby retry source; external
  // pressure has neither an incoming rank ceiling nor a source exception.
  Fixture f;
  const auto source = f.Add({1, 2});
  const auto other = f.Add({9, 9});
  const std::array protected_ids{source};
  assert(f.policy.ReclaimForExecution(protected_ids));
  assert(f.log.events.back().checkpoint == other);
  const auto events = f.log.events.size();
  assert(!f.policy.ReclaimForExecution(protected_ids));
  assert(f.log.events.size() == events && f.Cached({1, 2}) == 2);
  assert(f.policy.ReclaimForExecution());
  assert(f.log.events.back().checkpoint == source);
}
void ExecutionPressureExternalHandles() {
  for (bool drop_index : {false, true}) {
    Fixture f;
    const std::vector<Token> tokens{1, 2, 3, 4};
    f.Add(tokens);
    auto lookup = f.index.Lookup({{1}, tokens, InputIdentity(tokens.size())});
    assert(!f.policy.ReclaimForExecution());
    auto checkpoint = lookup.selected->checkpoint;
    const auto entry = lookup.selected->entry;
    lookup = {};
    assert(checkpoint.use_count() == 3);  // policy, index, external handle
    if (drop_index) {
      f.index.DropResident(entry);
      assert(checkpoint.use_count() == 2);
    }
    // Two owners can mean policy + external, NOT necessarily policy + index.
    assert(!f.policy.ReclaimForExecution());
    checkpoint.reset();
    assert(f.policy.ReclaimForExecution());
    assert(f.policy.Size() == 0);
  }
}
void ExecutionPressurePins() {
  constexpr std::array<ComponentDescriptor, 3> components{
      {{{1}, 1, ComponentKind::kAppendRows, 64, 4, 0},
       {{2}, 1, ComponentKind::kPrivateState, 0, 0, 32},
       {{3}, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
  for (bool private_states : {false, true}) {
    const auto descriptors =
        std::span(components).first(private_states ? 3 : 1);
    for (int kind = 0; kind < (private_states ? 7 : 5); ++kind) {
      ResourceLedger ledger{kIndexLimits};
      PrefixIndex index{ledger};
      index.Register({1}, descriptors);
      Log log;
      RetentionPolicy policy{ledger, index, &log};
      const std::vector<Token> tokens(9, 1);
      assert(policy.Admit({tokens, {1}, {}}, [&] {
        return IndexCheckpoint(ledger, tokens, {1}, {}, descriptors);
      }));
      auto lookup = index.Lookup({{1}, tokens, InputIdentity(tokens.size())});
      const auto& c = lookup.selected->checkpoint->Components()[0];
      std::optional<ChunkReference> reference;
      std::optional<CheckpointChunk> chunk_copy;
      PersistencePin pin;
      if (kind == 0)
        reference = c.chunks[0].PinReader();
      else if (kind == 1)
        reference = c.chunks[0].PinPersistence();
      else if (kind == 2)
        reference.emplace(c.chunks[0]);  // escaped checkpoint-kind reference
      else if (kind == 3)
        chunk_copy = c.chunks[0];
      else if (kind == 4)
        pin = c.tail->CommittedCharge().PinPersistence();
      else
        pin = lookup.selected->checkpoint->Components()[kind - 4]
                  .private_state->CommittedCharge()
                  .PinPersistence();
      lookup = {};
      const auto events = log.events.size();
      assert(!policy.ReclaimForExecution());
      assert(policy.Size() == 1 && log.events.size() == events);
      reference.reset();
      chunk_copy.reset();
      pin = {};
      assert(policy.ReclaimForExecution());
      assert(policy.Size() == 0);
    }
  }
  // Skip a pinned lowest-value record, not the whole scan.
  PressureFixture f;
  auto tokens = Conversation(1, 7);
  const auto pinned = f.Add(tokens);
  const auto eligible = f.Add(Conversation(2, 16));
  auto lookup = f.index.Lookup({{1}, tokens, InputIdentity(tokens.size())});
  auto reader =
      lookup.selected->checkpoint->Components()[0].chunks[0].PinReader();
  lookup = {};
  assert(f.policy.ReclaimForExecution());
  assert(f.log.events.back().checkpoint == eligible);
  assert(f.policy.Size() == 1 && f.policy.Rank(pinned) == 3);
}
void ExecutionPressureSharedAndInFlight() {
  ResourceLedger ledger{kIndexLimits};
  const auto descriptors = std::span(kPressureComponents).first(1);
  PrefixIndex index{ledger};
  index.Register({1}, descriptors);
  RetentionPolicy policy{ledger, index};
  auto history = ExecutionHistory::Cold(ledger, descriptors, {1});
  std::vector<Token> tokens(8, 1);
  const auto save = [&](const PayloadRequest& r) {
    auto pool =
        ledger.Reserve(ResourceCategory::kBackingFree, r.bytes).Convert();
    return Payload::Committed(pool.ReserveBacking(r.category).Convert(),
                              std::make_shared<int>(0));
  };
  const auto capture = [&] {
    return history.Capture({tokens,
                            InputIdentity(tokens.size()),
                            {{{1}, tokens.size()}},
                            CheckpointPurpose::kPrompt,
                            1},
                           save);
  };
  assert(policy.Admit({tokens, {1}, {}}, capture));
  tokens.push_back(2);
  bool prepared = false;
  try {
    (void)history.Capture({tokens,
                           InputIdentity(tokens.size()),
                           {{{1}, tokens.size()}},
                           CheckpointPurpose::kPrompt,
                           1},
                          save, [&](std::span<const PayloadRequest>) {
                            prepared = true;
                            // Unpublished capture already owns the shared
                            // chunks, but no Checkpoint handle has escaped and
                            // no reader-kind pin exists.
                            assert(!policy.ReclaimForExecution());
                            throw std::runtime_error("cancel capture");
                          });
    assert(false);
  } catch (const std::runtime_error&) {
    assert(prepared);
  }
  assert(policy.Admit({tokens, {1}, {}}, capture));
  // Two retained checkpoints share complete chunks, with only weak history
  // bookkeeping outside retention. Normal sharing must not exclude both.
  assert(policy.ReclaimForExecution());
  assert(policy.Size() == 1);
  assert(policy.ReclaimForExecution());
}
void ExecutionPressureDirectRows() {
  ResourceLedger ledger{kIndexLimits};
  const auto descriptors = std::span(kPressureComponents).first(1);
  PrefixIndex index{ledger};
  index.Register({1}, descriptors);
  RetentionPolicy policy{ledger, index};
  std::vector<ResourceCharge> pools;
  const std::vector<Token> tokens{1, 2, 3};
  auto history = ExecutionHistory::Cold(ledger, descriptors, {1});
  assert(policy.Admit({tokens, {1}, {}}, [&] {
    return history.Capture(
        {tokens,
         InputIdentity(tokens.size()),
         {{{1}, tokens.size()}},
         CheckpointPurpose::kPrompt,
         1},
        [&](const PayloadRequest& r) {
          pools.push_back(
              ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                  .Convert());
          return Payload::Committed(
              pools.back().ReserveBacking(r.category).Convert(),
              std::make_shared<std::vector<std::byte>>(r.bytes));
        });
  }));
  auto lookup = index.Lookup({{1}, tokens, InputIdentity(tokens.size())});
  const std::array protected_ids{lookup.selected->checkpoint->Id()};
  auto reader = lookup.selected->checkpoint->Components()[0].tail->PinRows();
  lookup = {};
  // Direct RowPins have no public count: the owner must publish protection.
  assert(!policy.ReclaimForExecution(protected_ids));
  // Demonstrate the documented safety boundary even if a caller omits that
  // protection. Logical removal cannot release the reader's backing token.
  assert(policy.ReclaimForExecution());
  assert(reader.Owner() &&
         pools[0].Info().category == ResourceCategory::kPrivateTail);
  reader = {};
  assert(pools[0].Info().category == ResourceCategory::kBackingFree);
}
void ExecutionPressureBorrowedRows() {
  ResourceLedger ledger{kIndexLimits};
  FakeAdapter adapter;
  FakeStream stream{true};
  LeasedSlot slot{ledger, adapter, stream, {1}};
  auto lease = slot.Acquire();
  const std::vector<Token> tokens{1, 2, 3, 4, 5};
  adapter.Append(lease.Execution(), tokens, tokens);
  const auto hash = adapter.RecurrentHash(lease.Execution());
  const auto invalidations = adapter.Invalidations();
  PrefixIndex index{ledger};
  index.Register(adapter.CompatibilityIdentity(), adapter.Components());
  RetentionPolicy policy{ledger, index};
  auto history = ExecutionHistory::Cold(ledger, adapter.Components(),
                                        adapter.CompatibilityIdentity());
  assert(policy.Admit({tokens, adapter.CompatibilityIdentity(), {}}, [&] {
    return history.Capture(
        {tokens, InputIdentity(tokens.size()),
         adapter.Positions(lease.Execution()), CheckpointPurpose::kPrompt, 1},
        [&](const PayloadRequest& r) {
          auto pool =
              ledger.Reserve(ResourceCategory::kBackingFree, r.bytes).Convert();
          auto reservation = pool.ReserveBacking(r.category);
          auto bytes = std::make_shared<std::vector<std::byte>>(r.bytes);
          if (r.category == ResourceCategory::kPrivateState) {
            auto copy = adapter.CapturePrivate(lease.Execution(), r.component,
                                               *bytes, stream);
            assert(copy.Wait() == TransferResult::kSucceeded);
            return Payload::Committed(reservation.Convert(), bytes);
          }
          return Payload::Borrowed(lease.Borrow(r.component, r.first, r.end,
                                                std::move(reservation), bytes,
                                                *bytes));
        });
  }));
  auto lookup = index.Lookup(
      {adapter.CompatibilityIdentity(), tokens, InputIdentity(tokens.size())});
  auto borrowed = lookup.selected->checkpoint->Components()[0]
                      .chunks[0]
                      .Storage()
                      .BorrowedStorage();
  auto reader = borrowed->Pin();
  lookup = {};
  assert(!policy.ReclaimForExecution());
  reader = {};
  // Materialization alone is not enough: direct backing pins still block.
  adapter.GuardRows(lease.Execution(), kTarget, 0, tokens.size());
  adapter.GuardRows(lease.Execution(), kDraft, 0, tokens.size());
  assert(!borrowed->Location());
  assert(borrowed->RetentionEvictable());
  reader = borrowed->Pin();
  auto persistence = borrowed->PinPersistence();
  borrowed.reset();
  assert(!policy.ReclaimForExecution());
  reader = {};
  assert(!policy.ReclaimForExecution());
  persistence = {};
  assert(policy.ReclaimForExecution());  // no permanent borrowed exclusion
  assert(policy.Size() == 0 && adapter.Invalidations() == invalidations);
  assert(adapter.RecurrentHash(lease.Execution()) == hash);
}
void ExecutionPressureIdleBorrowedLifecycle() {
  ResourceLedger ledger{kIndexLimits};
  FakeAdapter adapter;
  FakeStream stream{true};
  std::promise<void> copying, resume;
  auto started = copying.get_future();
  auto allowed = resume.get_future();
  // One target chunk, one idle copy. Hold stream acquisition after CopyPiece
  // acquires its source/copy pin, without timing-dependent sleeps or GPU work.
  IdleSpillConfig spill{4 * sizeof(Token),
                        [&]() -> std::unique_ptr<Stream> {
                          copying.set_value();
                          allowed.wait();
                          return std::make_unique<FakeStream>(true);
                        },
                        [] { return true; }};
  LeasedSlot slot{ledger, adapter, stream, {1}, spill};
  auto lease = slot.Acquire();
  const std::vector<Token> tokens{1, 2, 3, 4};
  adapter.Append(lease.Execution(), tokens, tokens);
  const auto hash = adapter.RecurrentHash(lease.Execution());
  const auto invalidations = adapter.Invalidations();
  const auto live = lease.Location();
  const auto descriptors = adapter.Components().first(1);
  PrefixIndex index{ledger};
  index.Register(adapter.CompatibilityIdentity(), descriptors);
  RetentionPolicy policy{ledger, index};
  auto history = ExecutionHistory::Cold(ledger, descriptors,
                                        adapter.CompatibilityIdentity());
  assert(policy.Admit({tokens, adapter.CompatibilityIdentity(), {}}, [&] {
    return history.Capture(
        {tokens,
         InputIdentity(tokens.size()),
         {{kTarget, tokens.size()}},
         CheckpointPurpose::kPrompt,
         1},
        [&](const PayloadRequest& r) {
          auto pool =
              ledger.Reserve(ResourceCategory::kBackingFree, r.bytes).Convert();
          auto bytes = std::make_shared<std::vector<std::byte>>(r.bytes);
          return Payload::Borrowed(lease.Borrow(r.component, r.first, r.end,
                                                pool.ReserveBacking(r.category),
                                                bytes, *bytes));
        });
  }));
  auto lookup = index.Lookup(
      {adapter.CompatibilityIdentity(), tokens, InputIdentity(tokens.size())});
  const std::array protected_ids{lookup.selected->checkpoint->Id()};
  auto rows = lookup.selected->checkpoint->Components()[0]
                  .chunks[0]
                  .Storage()
                  .BorrowedStorage();
  const std::weak_ptr<BorrowedRows> observed = rows;
  lookup = {};
  assert(!rows->RetentionEvictable() && !policy.ReclaimForExecution());
  lease.Commit();
  assert(started.wait_for(std::chrono::seconds(3)) ==
         std::future_status::ready);
  assert(!rows->RetentionEvictable() && !policy.ReclaimForExecution());
  auto source_reader = rows->Pin();
  assert(source_reader.Location());
  resume.set_value();
  // Exact-live acquisition drains the current idle copy, not old external
  // source readers. The old source pin must still exclude the now-materialized
  // checkpoint, without misclassifying it as an active backing reader.
  lease = slot.Acquire(live);
  assert(!rows->Location());
  assert(slot.Metrics().idle_bytes == 4 * sizeof(Token));
  assert(!rows->RetentionEvictable() && !policy.ReclaimForExecution());
  auto backing_reader = rows->Pin();
  assert(!backing_reader.Location() && backing_reader.Owner());
  RowPin moved_reader = std::move(backing_reader);
  source_reader = {};
  // Backing readers must not enter the source-overwrite wait counter. This
  // guard must finish while moved_reader still owns materialized backing.
  adapter.GuardRows(lease.Execution(), kTarget, 0, tokens.size());
  assert(!rows->RetentionEvictable() && !policy.ReclaimForExecution());
  moved_reader = {};
  assert(rows->RetentionEvictable());
  // ChunkReference copies clone their RowPins; the final clone releases the
  // backing-reader count, independent of checkpoint/index shared ownership.
  lookup = index.Lookup(
      {adapter.CompatibilityIdentity(), tokens, InputIdentity(tokens.size())});
  std::optional<ChunkReference> first =
      lookup.selected->checkpoint->Components()[0].chunks[0].PinReader();
  std::optional<ChunkReference> clone = *first;
  lookup = {};
  first.reset();
  assert(!rows->RetentionEvictable() && !policy.ReclaimForExecution());
  clone.reset();
  assert(rows->RetentionEvictable());
  auto persistence = rows->PinPersistence();
  assert(!rows->RetentionEvictable() && !policy.ReclaimForExecution());
  persistence = {};
  assert(rows->RetentionEvictable());
  assert(!policy.ReclaimForExecution(protected_ids));
  // Unbacked observations have no managed materialization/pin proof and
  // continue to fail closed; only real managed backing can become eligible.
  auto unbacked = lease.Observe(
      kTarget, 0, tokens.size(),
      adapter.BorrowRowsView(lease.Execution(), kTarget, 0, tokens.size()));
  assert(!unbacked->RetentionEvictable());
  rows.reset();
  assert(policy.ReclaimForExecution());
  assert(policy.Size() == 0 && observed.expired());
  assert(adapter.RecurrentHash(lease.Execution()) == hash);
  assert(lease.Location() == live && adapter.Invalidations() == invalidations);
}
void ExecutionPressureKeepsDurable() {
  struct Directory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("gufo-retention-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    Directory() { assert(std::filesystem::create_directory(path)); }
    ~Directory() { std::filesystem::remove_all(path); }
  } directory;
  ResourceLedger ledger{kIndexLimits};
  DiskStore disk{ledger, directory.path, 1 << 20};
  PrefixIndex index{ledger};
  CompatibilityDigest digest{};
  index.Register({1}, kIndexComponents, digest);
  RetentionPolicy policy{ledger, index};
  const std::vector<Token> tokens{1, 2, 3, 4};
  assert(policy.Admit({tokens, {1}, {}},
                      [&] { return IndexCheckpoint(ledger, tokens); }));
  CheckpointId id;
  IndexEntryId entry;
  {
    auto lookup = index.Lookup({{1}, tokens, InputIdentity(tokens.size())});
    const auto& checkpoint = *lookup.selected->checkpoint;
    id = checkpoint.Id();
    entry = lookup.selected->entry;
    DiskManifest manifest;
    manifest.checkpoint = id;
    manifest.lineage = checkpoint.Lineage();
    manifest.compatibility = digest;
    manifest.tokens = tokens;
    std::vector<std::vector<std::uint8_t>> payloads;
    std::vector<DiskWriteBuffer> buffers;
    const auto add = [&](const Payload& payload, bool private_file) {
      auto& bytes = payloads.emplace_back(payload.Bytes(), 0);
      DiskFileId file{};
      file[0] = static_cast<std::uint8_t>(payloads.size());
      buffers.push_back({private_file, file, bytes});
      return DiskPayload{file, bytes.size(), DiskChecksum(bytes)};
    };
    for (const auto& c : checkpoint.Components()) {
      DiskComponent component{c.descriptor, c.position, {}, {}, {}};
      for (const auto& chunk : c.chunks)
        component.chunks.push_back(add(chunk.Storage(), false));
      if (c.tail)
        component.tail = add(*c.tail, true);
      if (c.private_state)
        component.private_state = add(*c.private_state, true);
      manifest.components.push_back(std::move(component));
    }
    DiskFileId file{};
    file[0] = 100;
    disk.Publish(file, manifest, buffers);
    index.AttachDurable(entry, disk.Describe(id));
  }
  assert(policy.ReclaimForExecution());
  assert(policy.Size() == 0);
  const auto lookup = index.Lookup({{1}, tokens, InputIdentity(tokens.size())});
  assert(lookup.selected && lookup.selected->entry == entry);
  assert(!lookup.selected->checkpoint && lookup.selected->durable);
  assert(lookup.selected->durable->Manifest().checkpoint == id);
  assert(lookup.selected->boundary == tokens.size());
}
void ExecutionPressureDoesNotChangeCaptureReclaim() {
  Fixture f;
  const std::vector<Token> tokens{1, 2, 3, 4};
  const auto source = f.Add(tokens);
  const std::vector<Token> retry{1, 2, 3, 4, 5};
  RetentionRequest request{retry, {1}, {}, RetentionPurpose::kRetry, 4, source};
  auto lookup = f.index.Lookup({{1}, tokens, InputIdentity(tokens.size())});
  assert(!f.policy.ReclaimForExecution());
  // Existing capture semantics intentionally permit advancing a leased,
  // nearby retry source. The new stricter eligibility must not leak into it.
  assert(f.policy.Reclaim(request));
  assert(lookup.selected->checkpoint->IsValid() && f.policy.Size() == 0);
}
void RetryCopyReplacesItsNearbySource() {
  // An unchanged retry restored from its stable boundary may replace that
  // boundary with the full-prompt copy (state-edges), never another
  // conversation's checkpoint, and not when the gap exceeds the framing.
  PressureFixture f;
  const auto other = f.Add(Conversation(1, 16));
  const auto stable = Conversation(2, 7);
  const auto source = f.Add(stable);
  auto retry = stable;
  retry.push_back(5);
  RetentionRequest request{retry, {1}, {}, RetentionPurpose::kRetry, 7};
  request.source = source;
  // Disk-only staging must explicitly protect its source: unlike an in-RAM
  // retry copy, it does not replace that checkpoint. The nearby-source
  // shortcut must honor this protection, too.
  const std::array protected_ids{source};
  request.protected_checkpoints = protected_ids;
  assert(!f.policy.Reclaim(request));
  assert(f.Cached(stable) == stable.size());
  assert(f.Cached(Conversation(1, 16)) == 16);
  request.protected_checkpoints = {};
  assert(f.policy.Reclaim(request));
  assert(f.log.events.back().checkpoint == source);
  assert(!f.policy.Reclaim(request));
  auto far = Conversation(1, 16);
  far.resize(16 + 65, 3);
  RetentionRequest distant{far, {1}, {}, RetentionPurpose::kRetry, 16};
  distant.source = other;
  assert(!f.policy.Reclaim(distant));
}
void ResumedRetryReclaimsCoveredHistory() {
  PressureFixture f;
  const auto old = Conversation(1, 16);
  const auto ancestor = f.Add({old.begin(), old.begin() + 8});
  f.Add(old);
  const auto unrelated = Conversation(2, 16);
  f.Add(unrelated);
  const auto fork = Conversation(3, 8);
  f.Add(fork);
  auto left = fork, right = fork;
  left.push_back(10);
  right.push_back(20);
  f.Add(left);
  f.Add(right);
  const auto current = Conversation(4, 7);
  const auto source = f.Add({current.begin(), current.begin() + 4});
  const auto stable = f.Add(current);
  const std::array protected_ids{source, stable};
  auto prompt = current;
  prompt.push_back(5);
  RetentionRequest request{prompt, {1}, {}, RetentionPurpose::kRetry, 7};
  request.protected_checkpoints = protected_ids;
  // An optional cold retry cannot evict another conversation's history.
  assert(!f.policy.Reclaim(request));
  auto longer = prompt;
  longer.resize(prompt.size() + 65, 77);
  RetentionRequest distant{longer, {1}, {}, RetentionPurpose::kRetry, 7};
  distant.reused = true;
  distant.reused_tokens = 4;
  distant.protected_checkpoints = protected_ids;
  assert(!f.policy.Reclaim(distant) &&
         "large resumed prompt displaced covered history for a retry");
  request.reused = true;
  request.reused_tokens = current.size();
  assert(f.policy.Reclaim(request) &&
         "resumed retry starved behind covered history");
  assert(f.log.events.back().checkpoint == ancestor);
  assert(f.Cached(old) == old.size());
  assert(f.Cached(unrelated) == unrelated.size());
  assert(f.Cached(fork) == fork.size());
  assert(f.Cached({current.begin(), current.begin() + 4}) == 4);
  assert(f.Cached(current) == current.size());
  // Last copies, branching prefixes and the current turn's edit boundaries
  // remain protected even if a full-prompt retry would still need more space.
  assert(!f.policy.Reclaim(request));
}
void LongResumedRetryReclaimsOnlyOwnAncestors() {
  for (const bool named_source : {false, true}) {
    PressureFixture f;
    const auto unrelated = Conversation(1, 16);
    f.Add({unrelated.begin(), unrelated.begin() + 8});
    f.Add(unrelated);
    const auto current = Conversation(2, 16);
    const auto ancestor = f.Add({current.begin(), current.begin() + 8});
    const auto source = f.Add(current);
    auto prompt = current;
    prompt.resize(216, 77);
    const auto stable = f.Add({prompt.begin(), prompt.end() - 5});
    const std::array protected_ids{source, stable};
    RetentionRequest request{prompt, {1}, {}, RetentionPurpose::kRetry, 211};
    request.reused = true;
    request.reused_tokens = current.size();
    request.source = named_source ? source : CheckpointId{};
    request.protected_checkpoints = protected_ids;
    assert(f.policy.Reclaim(request) &&
           "long resumed retry starved behind its own covered ancestor");
    assert(f.log.events.back().checkpoint == ancestor);
    assert(f.Cached(current) == current.size());
    assert(f.Cached(unrelated) == unrelated.size());
    // The older, unrelated ancestor remains ineligible for this larger turn.
    assert(!f.policy.Reclaim(request));
  }
}
void SharedShortPrefixIsNotCheap() {
  // A new conversation sharing a short system/tools prefix must not evict
  // that prefix to admit itself (tool-agent peers).
  PressureFixture f;
  const auto idle = f.Add(Conversation(1, 16));
  const auto prefix = Conversation(2, 7);
  const auto shared = f.Add(prefix);
  auto incoming = prefix;
  incoming.insert(incoming.end(), {4, 4, 4});
  assert(f.policy.Reclaim({incoming, {1}, {}}));
  assert(f.log.events.back().checkpoint == idle);
  assert(f.Cached(incoming) == 7);
  (void)shared;
}
void UnreusedConversationKeepsItsStem() {
  // Compaction keeps a stale conversation's system prompt (cache-pressure
  // compacted): its leaf goes before its shallow history boundary. Once a
  // conversation is reused, its covered records go before any leaf.
  PressureFixture f;
  auto old_conversation = Conversation(1, 16);
  const auto old_stem =
      f.Add({old_conversation.begin(), old_conversation.begin() + 8},
            RetentionPurpose::kHistory);
  const auto old_leaf = f.Add(old_conversation);
  auto live = Conversation(2, 16);
  const auto live_stem =
      f.Add({live.begin(), live.begin() + 8}, RetentionPurpose::kHistory);
  const auto live_old = f.Add({live.begin(), live.begin() + 12});
  const auto live_leaf = f.Add(live, RetentionPurpose::kContinuation, true);
  assert(f.Reclaim() == old_leaf);
  auto compacted = old_conversation;
  compacted.resize(9);
  compacted[8] = 7;
  assert(f.Cached(compacted) == 8);
  assert(f.Reclaim() == old_stem);
  // The reused conversation drops its prior turn, then its stem, deepest
  // covered record first, before its own continuation.
  assert(f.Reclaim() == live_old && f.Reclaim() == live_stem);
  assert(f.Reclaim() == live_leaf);
}
void ForkedParentOutlivesItsChildren() {
  // Concurrent subagents share the parent turn but diverge a few tokens later
  // (cache-pressure fan-out), so the parent is no exact branch point. The
  // parent must still outlive the children for its own next turn.
  PressureFixture f;
  auto parent = Conversation(1, 12);
  const auto prior = f.Add({parent.begin(), parent.begin() + 8},
                           RetentionPurpose::kContinuation, true);
  const auto fork = f.Add(parent, RetentionPurpose::kContinuation, true);
  std::array<CheckpointId, 2> children;
  for (Token child = 0; child < 2; ++child) {
    auto tokens = parent;
    tokens.insert(tokens.end(), {500, 501, Token(600 + child), 502});
    children[child] = f.Add(tokens, RetentionPurpose::kContinuation, true);
  }
  assert(f.Reclaim() == prior);
  assert(f.Reclaim() == children[0] && f.Reclaim() == children[1]);
  parent.push_back(7);
  assert(f.Cached(parent) == 12);
  assert(f.Reclaim() == fork);
}
void ReuseProtectionLastsOneTurnover() {
  // A reused conversation outlives newer one-use requests (cache-pressure UI
  // tasks) for one turnover of the RAM budget, then ages out like any other.
  constexpr std::size_t kRam = 64 * 1024;
  PressureFixture f(kRam);
  auto conversation = Conversation(1, 16);
  f.Add(conversation);
  f.policy.Touch(f.log.events.back().checkpoint);
  Token next = 100;
  std::size_t admitted = 0;
  // Unique admitted bytes per one-use request: private state plus rows.
  constexpr std::size_t kRequest = 4096 + 16 * 64;
  while (admitted + kRequest < kRam) {
    f.Add(Conversation(next++, 16));
    admitted += kRequest;
  }
  // The budget filled; least recently used order alone would have removed
  // the conversation before any of these newer requests.
  assert(std::ranges::any_of(f.log.events, [](const auto& e) {
    return e.action == RetentionAction::kRemoved;
  }));
  assert(f.Cached(conversation) == 16);
  while (admitted < 2 * kRam) {
    f.Add(Conversation(next++, 16));
    admitted += kRequest;
    assert(f.ledger.Snapshot().total_bytes <= kRam);
  }
  assert(f.Cached(conversation) == 0);
  assert(std::ranges::any_of(f.log.events, [](const auto& e) {
    return e.action == RetentionAction::kRemoved &&
           e.reason == RetentionReason::kByteCapacity;
  }));
}
void RecordSourceReplacement() {
  for (const bool preserve : {true, false}) {
    Fixture f(1);
    const auto source = f.Add({1, 2});
    f.Add({1, 2, 3}, RetentionPurpose::kContinuation, 0, source, preserve);
    assert(f.Cached({1, 2, 3, 4}) == 3);
    assert(f.log.events[f.log.events.size() - 2].checkpoint == source);
  }
  // The byte-pressure flag remains effective while capture is incomplete.
  Fixture f(4);
  const auto other = f.Add({9, 9});
  const auto source = f.Add({1, 2});
  std::vector<Token> incoming{1, 2, 3};
  int attempt = 0;
  assert(f.policy.Admit(
      {incoming, {1}, {}, RetentionPurpose::kContinuation, 0, source, true},
      [&] {
        if (!attempt++)
          throw ResourceExhausted();
        return IndexCheckpoint(f.ledger, incoming);
      }));
  assert(f.log.events[f.log.events.size() - 2].checkpoint == other);
  assert(f.policy.Rank(source) == 2);
}
void FullOptionalExactReplacement() {
  for (const auto purpose :
       {RetentionPurpose::kRetry, RetentionPurpose::kHistory}) {
    for (const std::size_t limit : {1U, 2U}) {
      Fixture f(limit);
      const std::vector<Token> root{1, 2};
      const auto id = f.Add(root);
      if (limit == 2)
        f.Add({9, 9});
      bool called = false;
      assert(!f.policy.Admit({root, {1}, {}, purpose, 0, id}, [&] {
        called = true;
        return IndexCheckpoint(f.ledger, root);
      }));
      assert(!called && f.policy.Size() == limit && f.Cached({1, 2, 5}) == 2);
      assert(f.log.events.back().reason == RetentionReason::kRecordCapacity);
    }
  }
  {
    Fixture f(3);
    const std::vector<Token> root{1, 2};
    const auto id = f.Add(root);
    f.Add({1, 2, 3});
    f.Add({1, 2, 4});
    f.policy.Touch(
        id);  // mirrors Acquire's use before its optional reservation
    assert(f.policy.Rank(id) == 3);
    assert(!f.policy.Admit({root, {1}, {}, RetentionPurpose::kHistory, 0, id},
                           [&] { return IndexCheckpoint(f.ledger, root); }));
    assert(f.policy.Rank(id) == 3);
    f.Add({9, 9});
    assert(f.Cached({1, 2, 5}) == 2);
  }
  {
    Fixture f(1);
    const std::vector<Token> root{1, 2};
    const auto id = f.Add(root, RetentionPurpose::kBranchPoint);
    assert(!f.policy.Admit({root, {1}, {}, RetentionPurpose::kRetry, 0, id},
                           [&] { return IndexCheckpoint(f.ledger, root); }));
    assert(f.log.events.back().purpose == RetentionPurpose::kRetry &&
           f.log.events.back().rank == 0);
  }
  {
    // The original guard permits an optional exact replacement if another
    // eligible record exists; an exact learned point must stay learned.
    Fixture f(3);
    const auto id = f.Add({1, 2}, RetentionPurpose::kBranchPoint);
    f.Add({7}, RetentionPurpose::kHistory);
    f.Add({7, 8});
    const std::vector<Token> root{1, 2};
    assert(f.policy.Admit({root, {1}, {}, RetentionPurpose::kHistory, 0, id},
                          [&] { return IndexCheckpoint(f.ledger, root); }));
    assert(f.policy.Rank(f.log.events.back().checkpoint) == 3);
    assert(f.policy.Size() == 3);
  }
}
void FailureAndReplay() {
  ResourceLedger ledger{kIndexLimits};
  {
    PrefixIndex index{ledger};
    index.Register({1}, kIndexComponents);
    Log log;
    {
      RetentionPolicy policy(ledger, index, &log, 2);
      std::vector<Token> t{1, 2, 3};
      std::stop_source stop;
      stop.request_stop();
      bool called = false;
      assert(!policy.Admit({t,
                            {1},
                            {},
                            RetentionPurpose::kContinuation,
                            0,
                            {},
                            true,
                            stop.get_token()},
                           [&] {
                             called = true;
                             return IndexCheckpoint(ledger, t);
                           }));
      assert(!called);
      assert(!policy.Admit({t, {1}, {}},
                           [&]() -> std::shared_ptr<const Checkpoint> {
                             throw std::runtime_error("transfer");
                           }));
      assert(policy.Size() == 0);
      assert(policy.Admit({t, {1}, {}},
                          [&] { return IndexCheckpoint(ledger, t); }));
      const auto id = log.events.back().checkpoint;
      auto pin = index.Lookup({{1}, t, InputIdentity(t.size())});
      const auto before = ledger.Snapshot().total_bytes;
      assert(policy.RetainedPayloadBytes() > 0);
      policy.Remove(id);
      assert(policy.RetainedPayloadBytes() == 0);
      assert(log.events.back().retained_payload_bytes == 0);
      assert(pin.selected && pin.selected->checkpoint->IsValid());
      assert(log.events.back().unique_bytes_freed ==
             before - ledger.Snapshot().total_bytes);
      assert(log.events.back().unique_bytes_freed <
             pin.selected->checkpoint->MetadataBytes());
      pin = {};
      assert(policy.Admit({t, {1}, {}},
                          [&] { return IndexCheckpoint(ledger, t); }));
      policy.Touch(log.events.back().checkpoint);
    }
    std::map<std::uint64_t, std::uint64_t> retained;
    std::uint64_t sequence = 0;
    for (const auto& e : log.events) {
      assert(e.sequence == ++sequence);
      if (e.action == RetentionAction::kAdmitted)
        assert(retained.emplace(e.checkpoint.value, e.last_used).second);
      if (e.action == RetentionAction::kTouched) {
        assert(retained.contains(e.checkpoint.value));
        retained[e.checkpoint.value] = e.last_used;
      }
      if (e.action == RetentionAction::kRemoved)
        assert(retained.erase(e.checkpoint.value) == 1);
      assert(retained.size() == e.retained_records);
    }
    assert(retained.empty());
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void ImpossibleAndAllocationFailure() {
  Fixture f(1);
  const auto id = f.Add({1, 2});
  std::vector<Token> incoming{9};
  RetentionRequest r{incoming, {1}, {}};
  r.new_payload_bytes = kIndexLimits.total_bytes + 1;
  bool called = false;
  assert(!f.policy.Admit(r, [&] {
    called = true;
    return IndexCheckpoint(f.ledger, incoming);
  }));
  assert(!called && f.policy.Size() == 1 && f.Cached({1, 2, 3}) == 2);
  r.new_payload_bytes = 32;
  assert(!f.policy.Admit(r, []() -> std::shared_ptr<const Checkpoint> {
    throw std::bad_alloc();
  }));
  assert(f.policy.Size() == 1 && f.policy.Rank(id) == 3);
  assert(f.log.events.back().reason == RetentionReason::kCaptureFailure);
}
void CapacityAndSharing() {
  // Exact runtime payloads through the fake adapter; shared full chunks remain
  // identical and state/tails are fresh copies. Delayed copies finish before
  // the next Append changes recurrent state.
  ResourceLedger ledger{kIndexLimits};
  {
    FakeAdapter adapter;
    FakeStream stream{true};
    LeasedSlot slot(ledger, adapter, stream, {1});
    auto lease = slot.Acquire();
    PrefixIndex index{ledger};
    index.Register(adapter.CompatibilityIdentity(), adapter.Components());
    Log log;
    RetentionPolicy policy(ledger, index, &log);
    auto history = ExecutionHistory::Cold(ledger, adapter.Components(),
                                          adapter.CompatibilityIdentity());
    std::vector<Token> tokens;
    std::size_t copies = 0;
    auto capture = [&] {
      return history.Capture(
          {tokens, InputIdentity(tokens.size()),
           adapter.Positions(lease.Execution()), CheckpointPurpose::kPrompt, 3},
          [&](const PayloadRequest& r) {
            auto pool = ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                            .Convert();
            auto reservation = pool.ReserveBacking(r.category);
            auto bytes = std::make_shared<std::vector<std::byte>>(r.bytes);
            auto copy =
                r.category == ResourceCategory::kPrivateState
                    ? adapter.CapturePrivate(lease.Execution(), r.component,
                                             *bytes, stream)
                    : adapter.CopyRowsOut(lease.Execution(), r.component,
                                          r.first, r.end, *bytes, stream);
            assert(copy.Wait() == TransferResult::kSucceeded);
            ++copies;
            return Payload::Committed(reservation.Convert(), bytes);
          });
    };
    for (int turn = 0; turn < 2; ++turn) {
      std::vector<Token> more{1, 2, 3, 4};
      adapter.Append(lease.Execution(), more, more);
      tokens.insert(tokens.end(), more.begin(), more.end());
      assert(
          policy.Admit({tokens, adapter.CompatibilityIdentity(), {}}, capture));
    }
    assert(copies == 6);  // target+draft chunks and recurrent state each turn
    auto hits = index.Lookup({adapter.CompatibilityIdentity(), tokens,
                              InputIdentity(tokens.size())});
    assert(hits.candidates.size() == 2);
    assert(hits.candidates[0].checkpoint->Components()[0].chunks[0].Id() ==
           hits.candidates[1].checkpoint->Components()[0].chunks[0].Id());
  }
  assert(ledger.Snapshot().total_bytes == 0);
  // A simulated byte-admission failure chooses the same rank/LRU victim and
  // retries only at the same boundary. Pinned storage frees no payload bytes.
  Fixture f(4);
  const auto first = f.Add({1, 2});
  f.Add({9, 9});
  std::vector<Token> t{7, 7};
  int attempts = 0;
  assert(f.policy.Admit({t, {1}, {}}, [&] {
    if (!attempts++)
      throw ResourceExhausted();
    return IndexCheckpoint(f.ledger, t);
  }));
  assert(attempts == 2 &&
         f.log.events[f.log.events.size() - 2].checkpoint == first);
  assert(f.log.events[f.log.events.size() - 2].reason ==
         RetentionReason::kByteCapacity);
}
void InvalidInventoriesPreserveReplacement() {
  for (int fault = 0; fault < 5; ++fault) {
    Fixture f(1);
    const std::vector<Token> tokens{1, 2, 3, 4};
    const auto original = f.Add(tokens);
    auto components = std::vector<ComponentDescriptor>(kIndexComponents.begin(),
                                                       kIndexComponents.end());
    if (fault == 0)
      components.resize(1);
    if (fault == 1)
      ++components[0].layout_version;
    if (fault == 2)
      ++components[0].row_bytes;
    if (fault == 3)
      ++components[0].rows_per_chunk;
    const auto before = f.ledger.Snapshot().total_bytes;
    assert(!f.policy.Admit({tokens, {1}, {}}, [&] {
      return IndexCheckpoint(
          f.ledger, tokens, {1}, {}, components,
          fault == 4 ? std::optional<Rows>{3} : std::nullopt);
    }));
    assert(f.policy.Size() == 1 && f.Cached(tokens) == tokens.size());
    assert(f.policy.Rank(original) == 3);
    assert(f.ledger.Snapshot().total_bytes == before);
    assert(f.log.events.back().action == RetentionAction::kRefused &&
           f.log.events.back().reason == RetentionReason::kInvalid);
    assert(f.log.events.back().ledger_bytes == before);
  }
}
void SharedChunkAccounting() {
  for (const std::size_t chunks : {0, 1, 7, 64, 2048}) {
    constexpr std::size_t capacity = 64ULL << 20;
    ResourceLedger ledger({capacity, capacity, 0, capacity});
    constexpr std::array<ComponentDescriptor, 3> components{
        {{{1}, 1, ComponentKind::kAppendRows, 8, 4, 0},
         {{2}, 1, ComponentKind::kAppendRows, 16, 8, 0},
         {{3}, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
    PrefixIndex index(ledger);
    index.Register({1}, components);
    Log log;
    RetentionPolicy policy(ledger, index, &log);
    const auto capture = [&](ExecutionHistory& history,
                             const std::vector<Token>& tokens) {
      const auto end = tokens.size();
      return history.Capture(
          {tokens,
           InputIdentity(end),
           {{{1}, end}, {{2}, end}, {{3}, end}},
           CheckpointPurpose::kPrompt,
           3},
          [&](const PayloadRequest& r) {
            auto pool = ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                            .Convert();
            return Payload::Committed(pool.ReserveBacking(r.category).Convert(),
                                      std::make_shared<int>(0));
          });
    };
    std::vector<std::shared_ptr<const Checkpoint>> retained;
    const auto check = [&] {
      // Independent exhaustive oracle: count each actual chunk identity once,
      // without relying on lineage, prefix length or record insertion order.
      std::set<std::uint64_t> seen;
      std::size_t expected = 0;
      for (const auto& checkpoint : retained)
        for (const auto& c : checkpoint->Components()) {
          if (c.private_state)
            expected += c.private_state->Bytes();
          if (c.tail)
            expected += c.tail->Bytes();
          for (const auto& chunk : c.chunks)
            if (seen.insert(chunk.Id().value).second)
              expected += chunk.Storage().Bytes();
        }
      assert(policy.RetainedPayloadBytes() == expected);
      assert(log.events.back().retained_payload_bytes == expected);
    };
    const auto admit = [&](std::shared_ptr<const Checkpoint> checkpoint) {
      assert(policy.Admit({checkpoint->Tokens(), {1}, {}},
                          [&] { return checkpoint; }));
      retained.push_back(std::move(checkpoint));
      check();
    };
    auto history = ExecutionHistory::Cold(ledger, components, {1});
    std::vector<Token> tokens(chunks * 4 + 1, 1);
    const auto root = capture(history, tokens);
    tokens.resize(tokens.size() + chunks * 4 + 3, 2);
    const auto branch = capture(history, tokens);
    admit(branch);  // Deepest first; shorter shared prefixes arrive later.
    admit(root);
    auto sibling = ExecutionHistory::Restored(ledger, *root);
    tokens.assign(root->Tokens().begin(), root->Tokens().end());
    tokens.resize(tokens.size() + chunks * 8 + 5, 3);
    admit(capture(sibling, tokens));
    auto child = ExecutionHistory::Restored(ledger, *branch);
    tokens.assign(branch->Tokens().begin(), branch->Tokens().end());
    tokens.resize(tokens.size() + chunks * 4 + 7, 4);
    admit(capture(child, tokens));
    auto unrelated = ExecutionHistory::Cold(ledger, components, {1});
    tokens.assign(chunks * 4 + 11, 9);
    admit(capture(unrelated, tokens));
    // Removing records changes which owner accounts for shared bytes. External
    // checkpoint handles keep storage alive, but no longer count as retained.
    for (const std::size_t remove : {1, 0, 1, 0, 0}) {
      const auto id = retained[remove]->Id();
      retained.erase(retained.begin() + remove);
      policy.Remove(id);
      check();
      if (!retained.empty()) {
        policy.Touch(retained.back()->Id());
        check();
      }
    }
    assert(policy.Size() == 0);
    if (chunks == 7) {
      // Exercise repeated incremental additions/removals across many shared
      // checkpoints. Keep external owners alive so ledger release cannot
      // stand in for the logical retained-byte total.
      auto growing = ExecutionHistory::Cold(ledger, components, {1});
      std::vector<std::shared_ptr<const Checkpoint>> saved;
      tokens.clear();
      for (std::size_t n = 0; n < 40; ++n) {
        tokens.resize((n + 1) * 4 + 1, 71);
        saved.push_back(capture(growing, tokens));
        admit(saved.back());
      }
      for (std::size_t n = 0; n < saved.size(); ++n) {
        const auto& checkpoint = saved[n * 17 % saved.size()];
        const auto found =
            std::find(retained.begin(), retained.end(), checkpoint);
        assert(found != retained.end());
        retained.erase(found);
        policy.Remove(checkpoint->Id());
        check();
      }
      for (auto it = saved.rbegin(); it != saved.rend(); ++it)
        admit(*it);
      while (!retained.empty()) {
        const auto id = retained.back()->Id();
        retained.pop_back();
        policy.Remove(id);
        check();
      }
      assert(policy.Size() == 0);
    }
    if (chunks == 0) {
      // A reader may retain only a late chunk after the source checkpoint
      // disappears. Two restored histories then recreate earlier chunks
      // independently while still sharing that late chunk.
      auto source = ExecutionHistory::Cold(ledger, components, {1});
      tokens.assign(8, 41);
      auto original = capture(source, tokens);
      auto left = ExecutionHistory::Restored(ledger, *original);
      auto right = ExecutionHistory::Restored(ledger, *original);
      auto late = original->Components()[0].chunks.back().PinReader();
      original.reset();
      tokens.push_back(42);
      admit(capture(left, tokens));
      tokens.back() = 43;
      admit(capture(right, tokens));
      assert(retained[0]->Components()[0].chunks[0].Id() !=
             retained[1]->Components()[0].chunks[0].Id());
      assert(retained[0]->Components()[0].chunks[1].Id() ==
             retained[1]->Components()[0].chunks[1].Id());
    }
  }
}
void RealCapacityAndUniqueAdmission() {
  ResourceLedger ledger({32768, 32768, 0, 0});
  {
    constexpr std::array<ComponentDescriptor, 2> components{
        {{{1}, 1, ComponentKind::kAppendRows, 1024, 4, 0},
         {{2}, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
    PrefixIndex index(ledger);
    index.Register({1}, components);
    Log log;
    RetentionPolicy policy(ledger, index, &log);
    auto history = ExecutionHistory::Cold(ledger, components, {1});
    std::vector<Token> tokens;
    for (std::size_t end = 4; end <= 12; end += 4) {
      tokens.resize(end, 1);
      std::vector<ComponentPosition> positions{{{1}, end}, {{2}, end}};
      auto bytes = history.NewPayloadBytes(positions);
      assert(bytes == 4096 + 32);
      RetentionRequest request{tokens, {1}, {}};
      request.new_payload_bytes = bytes;
      assert(policy.Admit(request, [&] {
        return history.Capture(
            {tokens, InputIdentity(end), positions, CheckpointPurpose::kPrompt,
             3},
            [&](const PayloadRequest& r) {
              auto pool =
                  ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                      .Convert();
              return Payload::Committed(
                  pool.ReserveBacking(r.category).Convert(),
                  std::make_shared<int>(0));
            });
      }));
      assert(policy.RetainedPayloadBytes() == end * 1024 + end / 4 * 32);
      assert(log.events.back().retained_payload_bytes ==
             policy.RetainedPayloadBytes());
    }
    assert(policy.Size() == 3);
    assert(policy.RetainedPayloadBytes() == 12 * 1024 + 3 * 32);
    const auto latest = log.events.back().checkpoint;
    policy.Touch(latest);
    assert(log.events.back().retained_payload_bytes == 12 * 1024 + 3 * 32);
    assert(ledger.Snapshot().total_bytes <= 32768);
    // An independent large branch needs real reclamation, with no simulated
    // pressure exception. Full payload exceeds the old available capacity.
    tokens.assign(16, 9);
    auto cold = ExecutionHistory::Cold(ledger, components, {1});
    std::vector<ComponentPosition> positions{{{1}, 16}, {{2}, 16}};
    RetentionRequest request{tokens, {1}, {}};
    request.new_payload_bytes = cold.NewPayloadBytes(positions);
    assert(policy.Admit(request, [&] {
      return cold.Capture(
          {tokens, InputIdentity(tokens.size()), positions,
           CheckpointPurpose::kPrompt, 3},
          [&](const PayloadRequest& r) {
            auto pool = ledger.Reserve(ResourceCategory::kBackingFree, r.bytes)
                            .Convert();
            return Payload::Committed(pool.ReserveBacking(r.category).Convert(),
                                      std::make_shared<int>(0));
          });
    }));
    assert(std::ranges::any_of(log.events, [](const auto& e) {
      return e.action == RetentionAction::kRemoved &&
             e.reason == RetentionReason::kByteCapacity;
    }));
    assert(ledger.Snapshot().total_bytes <= 32768);
  }
  assert(ledger.Snapshot().total_bytes == 0);
  {
    Fixture f;
    for (Token token = 1; token <= 129; ++token)
      f.Add({token});
    assert(f.policy.Size() == 128 && f.Cached({1}) == 0 &&
           f.Cached({129}) == 1);
  }
}
RetentionRequest Prediction(std::span<const Token> tokens,
                            std::uint64_t owner = 1) {
  RetentionRequest request{tokens, {1}, {}, RetentionPurpose::kHistory};
  request.admission = AdmissionPolicy::kNoEviction;
  request.prediction_owner = owner;
  return request;
}
CheckpointId AddPrediction(Fixture& f, const std::vector<Token>& tokens,
                           std::uint64_t owner = 1) {
  assert(f.policy.Admit(Prediction(tokens, owner),
                        [&] { return IndexCheckpoint(f.ledger, tokens); }));
  return f.log.events.back().checkpoint;
}
void NoEvictionRecordAndDuplicate() {
  Fixture full;
  for (Token token = 1; token <= 128; ++token)
    full.Add({token});
  const std::vector<Token> incoming{129};
  auto request = Prediction(incoming);
  const auto before = full.ledger.Snapshot();
  bool called = false;
  assert(!full.policy.Admit(request, [&] {
    called = true;
    return IndexCheckpoint(full.ledger, incoming);
  }));
  assert(!called && full.policy.Size() == 128);
  assert(full.log.events.back().reason == RetentionReason::kRecordCapacity);
  assert(!full.policy.Reclaim(request));
  assert(full.ledger.Snapshot().total_bytes == before.total_bytes);
  for (Token token = 1; token <= 128; ++token)
    assert(full.Cached({token}) == 1);
  // The default still evicts and admits the 129th ordinary record.
  full.Add(incoming);
  assert(full.Cached({1}) == 0 && full.Cached(incoming) == 1);
  assert(full.policy.PredictionUsage().records == 0);

  Fixture duplicate;
  const std::vector<Token> tokens{1, 2, 3};
  const auto id = duplicate.Add(tokens, RetentionPurpose::kBranchPoint);
  assert(!duplicate.policy.Admit(Prediction(tokens), [&] {
    called = true;
    throw ResourceExhausted();
    return std::shared_ptr<const Checkpoint>{};
  }));
  assert(!called && duplicate.policy.Size() == 1);
  assert(duplicate.policy.Rank(id) == 3 && duplicate.Cached(tokens) == 3);
  assert(duplicate.log.events.back().reason == RetentionReason::kDuplicate);
}
void PredictionCountQuotaAndObservation() {
  Fixture f;
  std::array<CheckpointId, 4> ids;
  for (std::size_t i = 0; i < ids.size(); ++i)
    ids[i] = AddPrediction(f, {static_cast<Token>(i + 1)}, i + 10);
  const auto budget = f.policy.PredictionLimits();
  assert(budget.bytes == kIndexLimits.ram_bytes / 12 && budget.records == 4);
  const auto used = f.policy.PredictionUsage();
  assert(used.records == 4 && used.bytes > 0 && used.bytes <= budget.bytes);
  const auto rank = f.policy.Rank(ids[0]);
  assert(!f.policy.ObservePrediction(ids[0], 0));
  assert(!f.policy.ObservePrediction(ids[0], 10));
  assert(f.policy.ObservePrediction(ids[0], 99));
  assert(f.policy.ObservePrediction(ids[0], 99));
  assert(f.policy.Rank(ids[0]) == rank);
  assert(!f.policy.WithdrawUnprovenPrediction(ids[0], 10));
  assert(f.policy.PredictionUsage().bytes == used.bytes);
  const std::vector<Token> fifth{5};
  bool called = false;
  assert(!f.policy.Admit(Prediction(fifth), [&] {
    called = true;
    return IndexCheckpoint(f.ledger, fifth);
  }));
  assert(!called && f.log.events.back().reason ==
                        RetentionReason::kPredictionRecordCapacity);
  f.policy.Remove(ids[0]);
  assert(f.policy.PredictionUsage().records == 3);
  AddPrediction(f, fifth);
  assert(f.policy.PredictionUsage().records == 4);
  assert(!f.policy.WithdrawUnprovenPrediction(ids[1], 0));
  assert(!f.policy.WithdrawUnprovenPrediction(ids[1], 999));
  assert(f.policy.WithdrawUnprovenPrediction(ids[1], 11));
  assert(f.policy.PredictionUsage().records == 3);
  // Ordinary exact replacement ends prediction provenance, even if observed.
  assert(f.policy.ObservePrediction(ids[2], 99));
  const auto replacement = f.Add({3});
  assert(f.policy.PredictionUsage().records == 2);
  assert(!f.policy.WithdrawUnprovenPrediction(replacement, 12));
  assert(!f.policy.ObservePrediction(replacement, 99));
}
void PredictionByteQuota() {
  constexpr std::array<ComponentDescriptor, 1> components{
      {{{1}, 1, ComponentKind::kPrivateState, 0, 0, 32768}}};
  ResourceLedger ledger{{1 << 20, 1 << 20, 0, 0}};
  PrefixIndex index{ledger};
  index.Register({1}, components);
  Log log;
  RetentionPolicy policy{ledger, index, &log};
  const auto add = [&](Token token, std::size_t estimate = 0) {
    const std::vector<Token> tokens{token};
    auto request = Prediction(tokens);
    request.new_payload_bytes = estimate;
    return policy.Admit(request, [&] {
      return IndexCheckpoint(ledger, tokens, {1}, {}, components);
    });
  };
  assert(add(1));
  const auto first = log.events.back().checkpoint;
  assert(add(2));
  const auto before = policy.PredictionUsage();
  assert(before.records == 2);
  // Even an underestimated preflight cannot bypass the actual payload cap.
  assert(!add(3));
  assert(log.events.back().reason == RetentionReason::kPredictionByteCapacity);
  assert(policy.PredictionUsage().bytes == before.bytes);
  assert(!add(4, policy.PredictionLimits().bytes + 1));
  assert(log.events.back().reason == RetentionReason::kPredictionByteCapacity);
  policy.Remove(first);
  assert(add(3));
  assert(policy.PredictionUsage().records == 2);
}
void PredictionFailureAndCancellation() {
  for (int failure = 0; failure < 4; ++failure) {
    Fixture f;
    const auto anchor = f.Add({1, 2, 3}, RetentionPurpose::kBranchPoint);
    const auto before = f.ledger.Snapshot().total_bytes;
    const std::vector<Token> tokens{9, 9};
    auto request = Prediction(tokens);
    std::stop_source stop;
    request.stop = stop.get_token();
    int calls = 0;
    assert(!f.policy.Admit(request, [&]() -> std::shared_ptr<const Checkpoint> {
      ++calls;
      if (failure == 0)
        throw ResourceExhausted();
      if (failure == 1)
        throw std::bad_alloc();
      if (failure == 2)
        throw std::runtime_error("copy failed");
      auto checkpoint = IndexCheckpoint(f.ledger, tokens);
      stop.request_stop();
      return checkpoint;
    }));
    assert(calls == 1 && f.policy.Size() == 1 && f.policy.Rank(anchor) == 3);
    assert(f.Cached({1, 2, 3}) == 3 && f.Cached(tokens) == 0);
    assert(f.policy.PredictionUsage().records == 0);
    assert(f.ledger.Snapshot().total_bytes == before);
  }
  Fixture f;
  const std::vector<Token> tokens{1};
  bool called = false;
  assert(!f.policy.Admit(Prediction(tokens, 0), [&] {
    called = true;
    return IndexCheckpoint(f.ledger, tokens);
  }));
  assert(!called && f.log.events.back().reason == RetentionReason::kInvalid);
}
void PredictionReaderAndPinProtection() {
  constexpr std::array<ComponentDescriptor, 2> components{
      {{{1}, 1, ComponentKind::kAppendRows, 64, 4, 0},
       {{2}, 1, ComponentKind::kPrivateState, 0, 0, 32}}};
  for (int kind = 0; kind < 9; ++kind) {
    ResourceLedger ledger{kIndexLimits};
    PrefixIndex index{ledger};
    index.Register({1}, components);
    Log log;
    RetentionPolicy policy{ledger, index, &log};
    const std::vector<Token> tokens(9, 1);
    assert(policy.Admit(Prediction(tokens), [&] {
      return IndexCheckpoint(ledger, tokens, {1}, {}, components);
    }));
    const auto id = log.events.back().checkpoint;
    auto lookup = index.Lookup({{1}, tokens, InputIdentity(tokens.size())});
    const auto& c = lookup.selected->checkpoint->Components()[0];
    std::optional<ChunkReference> chunk;
    std::optional<CheckpointChunk> escaped;
    std::shared_ptr<const Checkpoint> checkpoint;
    std::shared_ptr<const void> owner;
    PersistencePin persistence;
    RowPin reader;
    if (kind == 0)
      chunk = c.chunks[0].PinReader();
    else if (kind == 1)
      chunk = c.chunks[0].PinPersistence();
    else if (kind == 2)
      escaped = c.chunks[0];
    else if (kind == 3)
      persistence = c.tail->CommittedCharge().PinPersistence();
    else if (kind == 4)
      persistence = lookup.selected->checkpoint->Components()[1]
                        .private_state->CommittedCharge()
                        .PinPersistence();
    else if (kind == 5)
      reader = c.tail->PinRows();
    else if (kind == 6)
      owner =
          lookup.selected->checkpoint->Components()[1].private_state->Owner();
    else
      checkpoint = lookup.selected->checkpoint;
    if (kind == 8)
      index.DropResident(lookup.selected->entry);
    lookup = {};
    if (kind == 8)
      assert(checkpoint.use_count() == 2);  // policy + external, NOT index
    assert(!policy.WithdrawUnprovenPrediction(id, 1));
    assert(policy.Size() == 1 && policy.PredictionUsage().records == 1);
    chunk.reset();
    escaped.reset();
    checkpoint.reset();
    owner.reset();
    persistence = {};
    reader = {};
    // Lookup/pin acquisition alone did not promote the prediction.
    assert(policy.WithdrawUnprovenPrediction(id, 1));
    assert(policy.Size() == 0 && policy.PredictionUsage().bytes == 0);
  }
  Fixture fork;
  const auto root = AddPrediction(fork, {1, 2});
  fork.Add({1, 2, 3});
  fork.Add({1, 2, 4});
  assert(!fork.policy.WithdrawUnprovenPrediction(root, 1));
  assert(fork.Cached({1, 2, 9}) == 2);
}
void PredictionConcurrentReader() {
  Fixture f;
  const std::vector<Token> tokens{1, 2, 3};
  const auto id = AddPrediction(f, tokens);
  std::mutex publication;
  std::promise<void> pinned, release;
  auto ready = pinned.get_future();
  auto released = release.get_future();
  std::jthread reader([&] {
    std::shared_ptr<const Checkpoint> checkpoint;
    {
      std::lock_guard lock(publication);
      checkpoint = f.index.Lookup({{1}, tokens, InputIdentity(tokens.size())})
                       .selected->checkpoint;
    }
    pinned.set_value();
    released.wait();
  });
  assert(ready.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
  {
    std::lock_guard lock(publication);
    assert(!f.policy.WithdrawUnprovenPrediction(id, 1));
  }
  release.set_value();
  reader.join();
  std::lock_guard lock(publication);
  assert(f.policy.WithdrawUnprovenPrediction(id, 1));
}
void PredictionBorrowedLifecycle() {
  ResourceLedger ledger{kIndexLimits};
  FakeAdapter adapter;
  FakeStream stream{true};
  std::promise<void> copying, resume;
  auto started = copying.get_future();
  auto allowed = resume.get_future();
  IdleSpillConfig spill{4 * sizeof(Token),
                        [&]() -> std::unique_ptr<Stream> {
                          copying.set_value();
                          allowed.wait();
                          return std::make_unique<FakeStream>(true);
                        },
                        [] { return true; }};
  LeasedSlot slot{ledger, adapter, stream, {1}, spill};
  auto lease = slot.Acquire();
  const auto live = lease.Location();
  const std::vector<Token> tokens{1, 2, 3, 4};
  adapter.Append(lease.Execution(), tokens, tokens);
  const auto descriptors = adapter.Components().first(1);
  PrefixIndex index{ledger};
  index.Register(adapter.CompatibilityIdentity(), descriptors);
  RetentionPolicy policy{ledger, index};
  auto history = ExecutionHistory::Cold(ledger, descriptors,
                                        adapter.CompatibilityIdentity());
  auto request = Prediction(tokens);
  request.compatibility = adapter.CompatibilityIdentity();
  assert(policy.Admit(request, [&] {
    return history.Capture(
        {tokens,
         InputIdentity(tokens.size()),
         {{kTarget, tokens.size()}},
         CheckpointPurpose::kPrompt,
         1},
        [&](const PayloadRequest& r) {
          auto pool =
              ledger.Reserve(ResourceCategory::kBackingFree, r.bytes).Convert();
          auto bytes = std::make_shared<std::vector<std::byte>>(r.bytes);
          return Payload::Borrowed(lease.Borrow(r.component, r.first, r.end,
                                                pool.ReserveBacking(r.category),
                                                bytes, *bytes));
        });
  }));
  auto lookup = index.Lookup(
      {request.compatibility, tokens, InputIdentity(tokens.size())});
  const auto id = lookup.selected->checkpoint->Id();
  auto rows = lookup.selected->checkpoint->Components()[0]
                  .chunks[0]
                  .Storage()
                  .BorrowedStorage();
  lookup = {};
  assert(!policy.WithdrawUnprovenPrediction(id, 1));  // still source-backed
  lease.Commit();
  assert(started.wait_for(std::chrono::seconds(3)) ==
         std::future_status::ready);
  assert(
      !policy.WithdrawUnprovenPrediction(id, 1));  // active preservation copy
  auto source_reader = rows->Pin();
  resume.set_value();
  lease = slot.Acquire(live);  // drain the copy, but not the old source reader
  assert(!rows->Location());
  assert(!policy.WithdrawUnprovenPrediction(id, 1));
  auto backing_reader = rows->Pin();
  source_reader = {};
  assert(!policy.WithdrawUnprovenPrediction(id, 1));
  auto persistence = rows->PinPersistence();
  backing_reader = {};
  assert(!policy.WithdrawUnprovenPrediction(id, 1));
  persistence = {};
  rows.reset();
  assert(policy.WithdrawUnprovenPrediction(id, 1));
  assert(policy.PredictionUsage().records == 0);
}
void PredictionWithdrawalBeforeRequiredCapture() {
  constexpr std::array<ComponentDescriptor, 1> components{
      {{{1}, 1, ComponentKind::kPrivateState, 0, 0, 4096}}};
  ResourceLedger ledger{{65536, 65536, 0, 0}};
  PrefixIndex index{ledger};
  index.Register({1}, components);
  Log log;
  RetentionPolicy policy{ledger, index, &log};
  const std::vector<Token> ordinary{1}, predicted{2}, required{3};
  const auto capture = [&](const auto& tokens) {
    return IndexCheckpoint(ledger, tokens, {1}, {}, components);
  };
  assert(policy.Admit({ordinary, {1}, {}, RetentionPurpose::kBranchPoint},
                      [&] { return capture(ordinary); }));
  const auto anchor = log.events.back().checkpoint;
  assert(policy.Admit(Prediction(predicted, 17),
                      [&] { return capture(predicted); }));
  const auto prediction = log.events.back().checkpoint;
  const auto room = 65536 - ledger.Snapshot().total_bytes;
  assert(room > 2048);
  auto pressure =
      ledger.Reserve(ResourceCategory::kBackingFree, room - 2048).Convert();
  bool exhausted = false;
  try {
    (void)capture(required);
  } catch (const ResourceExhausted&) {
    exhausted = true;
  }
  assert(exhausted);
  const auto before = ledger.Snapshot().total_bytes;
  assert(policy.WithdrawUnprovenPrediction(prediction, 17));
  assert(ledger.Snapshot().total_bytes + 4096 <= before);
  assert(policy.Admit({required, {1}, {}}, [&] { return capture(required); }));
  assert(policy.Rank(anchor) == 3 && policy.Size() == 2);
  assert(policy.PredictionUsage().records == 0);
  for (const auto& event : log.events)
    if (event.action == RetentionAction::kRemoved)
      assert(event.checkpoint == prediction);
}
void RequiredAdmissionWithdrawsOnlyUnderPressure() {
  for (const bool full : {false, true}) {
    Fixture f(full ? 2 : 4);
    const auto ordinary = f.Add({1});
    const auto root = AddPrediction(f, {2}, 17);
    const std::array owned{root};
    const std::vector<Token> tokens{2, 3};
    RetentionRequest request{tokens, {1}, {}};
    request.source = root;
    request.prediction_owner = 17;
    request.owned_predictions = owned;
    if (full) {
      auto optional = request;
      optional.admission = AdmissionPolicy::kNoEviction;
      assert(!f.policy.Admit(
          optional, [&] { return IndexCheckpoint(f.ledger, tokens); }));
      assert(f.policy.PredictionUsage().records == 1);
    }
    assert(f.policy.Admit(request,
                          [&] { return IndexCheckpoint(f.ledger, tokens); }));
    assert(f.policy.Rank(ordinary) == 3);
    assert(f.policy.PredictionUsage().records == (full ? 0 : 1));
    for (const auto& event : f.log.events)
      if (event.action == RetentionAction::kRemoved) {
        assert(full && event.checkpoint == root);
        assert(event.reason == RetentionReason::kExplicit);
      }
  }
  // A fitting exact replacement needs no additional slot, even in a full table.
  Fixture f(2);
  f.Add({1});
  const auto root = AddPrediction(f, {2}, 17);
  const std::array owned{root};
  const std::vector<Token> tokens{1};
  RetentionRequest request{tokens, {1}, {}};
  request.prediction_owner = 17;
  request.owned_predictions = owned;
  assert(f.policy.Admit(request,
                        [&] { return IndexCheckpoint(f.ledger, tokens); }));
  assert(f.policy.PredictionUsage().records == 1 && f.Cached({2}) == 1);
}
void RequiredAdmissionKeepsExplicitProtection() {
  Fixture f(2);
  f.Add({1});
  const auto root = AddPrediction(f, {2}, 17);
  const std::array owned{root};
  const std::vector<Token> tokens{3};
  RetentionRequest request{tokens, {1}, {}};
  request.prediction_owner = 17;
  request.owned_predictions = owned;
  request.protected_checkpoints = owned;
  assert(f.policy.Admit(request,
                        [&] { return IndexCheckpoint(f.ledger, tokens); }));
  assert(f.policy.PredictionUsage().records == 1 && f.Cached({2}) == 1);
  assert(f.Cached({1}) == 0);
  for (const auto& event : f.log.events)
    if (event.action == RetentionAction::kRemoved)
      assert(event.checkpoint != root &&
             event.reason != RetentionReason::kExplicit);
}
void RequiredPublicationMetadataPressure() {
  for (int scenario = 0; scenario < 3; ++scenario) {
    const bool withdraw = scenario != 0;
    const bool proof_room = scenario != 2;
    const bool removed_prediction = withdraw;
    Fixture f;
    const auto ordinary = f.Add({1});
    const auto root = AddPrediction(f, {2}, 17);
    const std::array owned{root};
    const std::vector<Token> tokens{2, 3};
    // Capture is complete. Exhaust publication metadata capacity, including
    // a zero-headroom case. The exact owner witness needs no lookup allocation;
    // the owned prediction must give way before the ordinary anchor even then.
    const auto required = IndexCheckpoint(f.ledger, tokens);
    auto pressure = f.ledger
                        .Reserve(ResourceCategory::kBackingFree,
                                 f.ledger.Limits().ram_bytes -
                                     f.ledger.Snapshot().ram_bytes -
                                     (proof_room ? sizeof(PrefixCandidate) : 0))
                        .Convert();
    RetentionRequest request{tokens, {1}, {}};
    request.source = root;  // ordinary ranking cannot sacrifice its source
    request.prediction_owner = 17;
    if (withdraw)
      request.owned_predictions = owned;
    int captures = 0;
    assert(f.policy.Admit(request, [&] {
      ++captures;
      return required;
    }));
    assert(captures ==
           2);  // retry actual admission, never assume reclaimed bytes
    assert(f.policy.PredictionUsage().records == (removed_prediction ? 0 : 1));
    if (removed_prediction)
      assert(f.policy.Rank(ordinary) == 3);
    std::size_t removed = 0;
    for (const auto& event : f.log.events)
      if (event.action == RetentionAction::kRemoved) {
        ++removed;
        assert(event.checkpoint == (removed_prediction ? root : ordinary));
        assert(event.reason == (removed_prediction
                                    ? RetentionReason::kExplicit
                                    : RetentionReason::kByteCapacity));
      }
    assert(removed == 1);
  }
}
void RequiredSharedChunksPreventWithdrawal() {
  ResourceLedger ledger{kIndexLimits};
  const auto descriptors = std::span(kPressureComponents).first(1);
  PrefixIndex index{ledger};
  index.Register({1}, descriptors);
  Log log;
  RetentionPolicy policy{ledger, index, &log};
  const std::vector<Token> ordinary{1};
  assert(policy.Admit({ordinary, {1}, {}}, [&] {
    return IndexCheckpoint(ledger, ordinary, {1}, {}, descriptors);
  }));
  const auto anchor = log.events.back().checkpoint;
  auto history = ExecutionHistory::Cold(ledger, descriptors, {1});
  std::vector<Token> tokens(8, 2);
  const auto capture = [&] {
    return history.Capture(
        {tokens,
         InputIdentity(tokens.size()),
         {{{1}, tokens.size()}},
         CheckpointPurpose::kPrompt,
         1},
        [&](const PayloadRequest& r) {
          auto pool =
              ledger.Reserve(ResourceCategory::kBackingFree, r.bytes).Convert();
          return Payload::Committed(pool.ReserveBacking(r.category).Convert(),
                                    std::make_shared<int>(0));
        });
  };
  assert(policy.Admit(Prediction(tokens, 17), capture));
  const auto root = log.events.back().checkpoint;
  const auto usage = policy.PredictionUsage();
  tokens.push_back(3);
  const auto required = capture();  // genuinely shares the root's full chunks
  assert(!policy.WithdrawUnprovenPrediction(root, 17));
  auto pressure =
      ledger
          .Reserve(ResourceCategory::kBackingFree,
                   ledger.Limits().ram_bytes - ledger.Snapshot().ram_bytes)
          .Convert();
  const std::array owned{root};
  RetentionRequest request{tokens, {1}, {}};
  request.source = root;
  request.prediction_owner = 17;
  request.owned_predictions = owned;
  int captures = 0;
  assert(policy.Admit(request, [&] {
    ++captures;
    return required;
  }));
  assert(captures == 2 && policy.PredictionUsage().bytes == usage.bytes);
  for (const auto& event : log.events)
    if (event.action == RetentionAction::kRemoved) {
      assert(event.checkpoint == anchor);
      assert(event.reason == RetentionReason::kByteCapacity);
    }
}
void NoEvictionBackingFragmentation() {
  constexpr std::array<ComponentDescriptor, 1> components{
      {{{1}, 1, ComponentKind::kPrivateState, 0, 0, 8192}}};
  for (bool fragmented : {false, true}) {
    ResourceLedger ledger{kIndexLimits};
    PrefixIndex index{ledger};
    index.Register({1}, components);
    Log log;
    RetentionPolicy policy{ledger, index, &log};
    CommittedBackingPool pool{
        ledger,
        {4 * 4096 + 131072, 4096, 131072},
        {[](std::size_t bytes) {
           auto memory = std::make_shared<std::vector<std::byte>>(bytes);
           return BackingPages{memory->data(), memory};
         },
         [](BackingPages& pages, std::size_t) {
           pages = {};
           return true;
         }}};
    std::array<std::optional<BackingBlock>, 4> occupied;
    for (auto& block : occupied) {
      block = pool.TryAcquire(ResourceCategory::kPrivateState, 4096);
      assert(block);
    }
    if (fragmented) {
      occupied[0].reset();
      occupied[2].reset();
    }
    const std::vector<Token> ordinary{1}, predicted{2};
    assert(
        policy.Admit({ordinary, {1}, {}, RetentionPurpose::kBranchPoint}, [&] {
          return IndexCheckpoint(ledger, ordinary, {1}, {}, components);
        }));
    const auto anchor = log.events.back().checkpoint;
    const std::array demand{
        PayloadRequest{{1}, ResourceCategory::kPrivateState, 0, 1, 8192}};
    assert(pool.CanFit(demand) == fragmented);
    assert(ledger.Snapshot().ram_bytes + 8192 < ledger.Limits().ram_bytes);
    int attempts = 0;
    assert(!policy.Admit(Prediction(predicted), [&] {
      ++attempts;
      auto history = ExecutionHistory::Cold(ledger, components, {1});
      return history.Capture({predicted,
                              InputIdentity(1),
                              {{{1}, 1}},
                              CheckpointPurpose::kPrompt,
                              1},
                             [&](const PayloadRequest& r) {
                               auto block =
                                   pool.TryAcquire(r.category, r.bytes);
                               if (!block)
                                 throw ResourceExhausted();
                               block->Convert();
                               return std::move(*block).TakePayload();
                             });
    }));
    assert(attempts == 1 && policy.Size() == 1 && policy.Rank(anchor) == 3);
    assert(log.events.back().reason == RetentionReason::kByteCapacity);
    assert(policy.PredictionUsage().records == 0);
  }
}
}  // namespace
int main() {
  NoEvictionRecordAndDuplicate();
  PredictionCountQuotaAndObservation();
  PredictionByteQuota();
  PredictionFailureAndCancellation();
  PredictionReaderAndPinProtection();
  PredictionConcurrentReader();
  PredictionBorrowedLifecycle();
  PredictionWithdrawalBeforeRequiredCapture();
  NoEvictionBackingFragmentation();
  RequiredAdmissionWithdrawsOnlyUnderPressure();
  RequiredAdmissionKeepsExplicitProtection();
  RequiredPublicationMetadataPressure();
  RequiredSharedChunksPreventWithdrawal();
  Planning();
  Branches();
  RanksAndSources();
  LongPrefixRelationsAfterReplacement();
  ShortPeersDoNotCoverHistory();
  HistoryBytePressureCanReplaceOldFamilies();
  CurrentTurnKeepsItsEditBoundaries();
  CheapCheckpointsGiveWayFirst();
  ReusedCheapCheckpointKeepsRecency();
  PersistingCheckpointIsNotAVictim();
  ExecutionPressureOrder();
  ExecutionPressureExternalHandles();
  ExecutionPressurePins();
  ExecutionPressureSharedAndInFlight();
  ExecutionPressureDirectRows();
  ExecutionPressureBorrowedRows();
  ExecutionPressureIdleBorrowedLifecycle();
  ExecutionPressureKeepsDurable();
  ExecutionPressureDoesNotChangeCaptureReclaim();
  RetryCopyReplacesItsNearbySource();
  ResumedRetryReclaimsCoveredHistory();
  LongResumedRetryReclaimsOnlyOwnAncestors();
  SharedShortPrefixIsNotCheap();
  UnreusedConversationKeepsItsStem();
  ForkedParentOutlivesItsChildren();
  ReuseProtectionLastsOneTurnover();
  RecordSourceReplacement();
  FullOptionalExactReplacement();
  FailureAndReplay();
  CapacityAndSharing();
  ImpossibleAndAllocationFailure();
  RealCapacityAndUniqueAdmission();
  SharedChunkAccounting();
  InvalidInventoriesPreserveReplacement();
}
