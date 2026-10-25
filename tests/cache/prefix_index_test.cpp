#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>

#include "tests/cache/prefix_index_fixture.hpp"

namespace {
bool watch_allocations = false;
std::size_t largest_allocation = 0;
}  // namespace
// Observe actual edge allocation order when a ledger reservation is rejected.
void* operator new(std::size_t bytes) {
  if (watch_allocations)
    largest_allocation = std::max(largest_allocation, bytes);
  if (void* pointer = std::malloc(bytes ? bytes : 1))
    return pointer;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

using namespace gufo::cache;
using namespace gufo::cache::testing;
namespace {
using Tokens = std::vector<Token>;
PrefixQuery Query(const Tokens& tokens, Rows stable = 0, Identity input = {},
                  std::vector<InputPrefix> prefixes = {}) {
  return {{1},
          tokens,
          InputIdentity(tokens.size(), std::move(input), std::move(prefixes)),
          stable,
          true};
}
template<class F>
void Invalid(F action) {
  bool threw = false;
  try {
    action();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}
void SnapshotParity(const PrefixIndex& index, const PrefixQuery& query) {
  const auto expected = index.Lookup(query);
  const auto actual = index.Snapshot().Lookup(query);
  assert(actual.size() == expected.candidates.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto& a = actual[i];
    const auto& e = expected.candidates[i];
    assert(a.entry == e.entry && a.boundary == e.boundary);
    assert(a.live == bool(e.live) && a.ram == bool(e.checkpoint) &&
           a.disk == bool(e.durable));
    assert(a.transfer_bytes == e.transfer_bytes && a.full_ready);
  }
}
void InvalidationTelemetryDoesNotOwnOrBlockEntries() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens tokens{1, 2, 3};
  const auto entry =
      index.Insert(IndexCheckpoint(ledger, tokens), ResidentComponents());
  std::size_t calls = 0;
  index.SetInvalidationObserver(
      [&](IndexEntryId id, Rows boundary, std::string_view reason) {
        assert(id == entry && boundary == 3 && reason == "erased");
        ++calls;
        throw std::runtime_error("telemetry is best effort");
      });
  index.Erase(entry);
  index.Erase(entry);  // Absent IDs do not produce fictitious retirements.
  assert(calls == 1 && !index.Lookup(Query(tokens)).selected);
  index.SetInvalidationObserver({});
}
void SelectionTable() {
  struct Case {
    Tokens prompt;
    Rows stable;
    bool shorter, marked, live, available;
    Rows expected;
    SelectionReason reason;
  };
  const std::vector<Case> cases{
      {{1, 2, 3, 4},
       0,
       true,
       false,
       true,
       true,
       3,
       SelectionReason::kExactLiveContinuation},
      {{1, 2, 3, 4},
       0,
       true,
       false,
       false,
       true,
       3,
       SelectionReason::kDeepestCheckpoint},
      {{1, 2, 3, 4},
       2,
       false,
       false,
       false,
       true,
       0,
       SelectionReason::kStablePrefixBoundary},
      {{1, 2, 3, 4},
       2,
       true,
       false,
       false,
       true,
       3,
       SelectionReason::kDeepestCheckpoint},
      {{1, 2, 3, 4},
       2,
       false,
       true,
       false,
       true,
       3,
       SelectionReason::kDeepestCheckpoint},
      {{1, 2, 3, 4},
       1,
       false,
       true,
       false,
       true,
       0,
       SelectionReason::kStablePrefixBoundary},
      {{1, 2, 3, 4},
       2,
       false,
       false,
       true,
       true,
       0,
       SelectionReason::kStablePrefixBoundary},
      {{1, 2, 3, 4},
       0,
       true,
       false,
       true,
       false,
       3,
       SelectionReason::kDeepestCheckpoint},
      {{1, 2},
       0,
       false,
       false,
       false,
       true,
       0,
       SelectionReason::kNoCompatibleBoundary},
      {{1, 9, 3},
       0,
       false,
       false,
       false,
       true,
       0,
       SelectionReason::kNoCompatibleBoundary},
  };
  for (const auto& c : cases) {
    ResourceLedger ledger{kIndexLimits};
    {
      PrefixIndex index{ledger};
      index.Register({1}, kIndexComponents);
      const auto deep = IndexCheckpoint(ledger, Tokens{1, 2, 3});
      (void)index.Insert(deep, ResidentComponents(), c.marked ? 2 : 0);
      if (c.shorter)
        (void)index.Insert(IndexCheckpoint(ledger, Tokens{1, 2}),
                           ResidentComponents());
      if (c.live) {
        auto frontier = IndexLive({1, 2, 3});
        frontier.available = c.available;
        (void)index.Insert(std::move(frontier));
      }
      const auto result = index.Lookup(Query(c.prompt, c.stable));
      assert(result.reason == c.reason);
      SnapshotParity(index, Query(c.prompt, c.stable));
      assert((result.selected ? result.selected->boundary : 0) == c.expected);
      if (result.selected && result.selected->checkpoint)
        assert(result.selected->transfer_bytes == 3 * 4 + 1 * 2 + 32);
      if (c.shorter && c.expected == 3)
        assert(result.candidates.back().boundary == 2);
      assert(index.CachedPrefixTokens(Query(c.prompt, c.stable)) ==
             (c.prompt.size() >= 3 && c.prompt[1] == 2 ? 3 : 0));
      const auto checkpoints = index.Lookup(Query(c.prompt));
      for (Rows minimum = 0; minimum <= c.prompt.size() + 1; ++minimum) {
        const bool expected = std::ranges::any_of(
            checkpoints.candidates, [&](const auto& candidate) {
              return !candidate.live && candidate.boundary >= minimum;
            });
        auto query = Query(c.prompt, c.stable);
        query.reuse = false;
        assert(index.HasCheckpoint(query, minimum) == expected);
      }
    }
    assert(ledger.Snapshot().total_bytes == 0);
  }
}
void CheckpointPresence() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens tokens{1, 2, 3, 4};
  (void)index.Insert(IndexLive(tokens));
  assert(!index.HasCheckpoint(Query(tokens), 0));
  const auto short_id =
      index.Insert(IndexCheckpoint(ledger, Tokens{1, 2}), ResidentComponents());
  const auto long_id = index.Insert(IndexCheckpoint(ledger, tokens, {1}, {9}),
                                    ResidentComponents());
  assert(index.HasCheckpoint(Query(tokens), 2));
  assert(!index.HasCheckpoint(Query(tokens), 3));
  assert(index.HasCheckpoint(Query(tokens, 0, {9}), 4));
  assert(!index.HasCheckpoint(Query(Tokens{1, 2, 9, 4}, 0, {9}), 4));
  index.SetAvailability(
      long_id, {{{1}, true, false}, {{2}, true, false}, {{3}, false, false}});
  assert(!index.HasCheckpoint(Query(tokens, 0, {9}), 4));
  index.Erase(short_id);
  assert(!index.HasCheckpoint(Query(tokens), 2));
  index.SetAvailability(long_id, ResidentComponents());
  auto invalid = Query(tokens);
  invalid.input = InputIdentity(1);
  Invalid([&] { (void)index.HasCheckpoint(invalid, 0); });
  // A resident presence check needs no candidate-list reservation, even when
  // every further metadata admission is refused.
  const auto before = ledger.Snapshot().total_bytes;
  ledger.FailAfter(LedgerStep::kReserve, 0);
  assert(index.HasCheckpoint(Query(tokens, 0, {9}), 4));
  assert(ledger.Snapshot().total_bytes == before);
}
void ImagesAndLearning() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens text{1, 2, 3}, image{1, 2, 3, 248056, 4},
      appended{1, 2, 3, 248056, 4, 248056, 5};
  auto before =
      index.Insert(IndexCheckpoint(ledger, text), ResidentComponents());
  const auto first = index.Insert(IndexCheckpoint(ledger, image, {1}, {10}),
                                  ResidentComponents());
  auto live = index.Insert(IndexLive(image, {7}, {10}));
  auto query = Query(appended, 0, {30}, {{3, {}}, {5, {10}}});
  assert(index.Lookup(query).selected->live->slot == SlotId{7});
  assert(index.CachedPrefixTokens(query) == 5);
  assert(index.CommonPrefixTokens(query) == 5);
  query = Query(image, 0, {20}, {{3, {}}});
  assert(index.Lookup(query).selected->boundary == 3);
  assert(index.CommonPrefixTokens(query) == 3);
  assert(index.Lookup(Query(image)).selected->boundary == 3);
  index.Erase(before);
  assert(index.Lookup(query).reason == SelectionReason::kInputIdentity);
  assert(index.CommonPrefixTokens(query) == 0);
  index.Erase(first);
  index.SetLiveAvailable(live, false);
  assert(index.CachedPrefixTokens(Query(image, 0, {10})) == 0);
  assert(index.CommonPrefixTokens(Query(image, 0, {10})) == 5);
  assert(index.CommonPrefixTokens(Query(Tokens{1, 2}, 0, {10})) == 2);
  assert(index.CommonPrefixTokens(Query(Tokens{1, 2})) == 0);
  index.Erase(live);
  // Port the legacy shared-prefix peeks: branches teach their divergence even
  // when no exact checkpoint exists at that boundary; availability is ignored.
  (void)index.Insert(IndexCheckpoint(ledger, Tokens{1, 2, 3, 4, 5}),
                     ResidentComponents());
  auto unavailable = index.Insert(IndexLive({1, 2, 3, 6, 7}, {8}));
  index.SetLiveAvailable(unavailable, false);
  assert(index.CommonPrefixTokens(Query(Tokens{1, 2, 3, 6, 9})) == 4);
  assert(index.CommonPrefixTokens(Query(Tokens{1, 2})) == 2);
  assert(index.CommonPrefixTokens(Query(Tokens{9, 9})) == 0);
}
void CoherenceAndFallback() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens tokens{1, 2, 3, 4};
  auto full = IndexCheckpoint(ledger, tokens);
  auto missing = ResidentComponents();
  missing.pop_back();
  const auto id = index.Insert(full, missing);
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  index.SetAvailability(
      id, {{{1}, true, true}, {{2}, true, false}, {{3}, false, true}});
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  index.SetAvailability(id, ResidentComponents());
  assert(index.Lookup(Query(tokens)).selected->checkpoint == full);
  // A live frontier cannot stand in for the stable fallback snapshot.
  auto short_live = index.Insert(IndexLive({1, 2}, {4}));
  assert(index.Lookup(Query(tokens, 2)).selected->boundary == 2);
  index.Erase(short_live);
  const auto shorter =
      index.Insert(IndexCheckpoint(ledger, Tokens{1, 2}), missing);
  assert(index.Lookup(Query(tokens, 2)).reason ==
         SelectionReason::kStablePrefixBoundary);
  index.SetAvailability(shorter, ResidentComponents());
  assert(index.Lookup(Query(tokens, 2)).selected->boundary == 4);
  index.Erase(shorter);
  // A mark on another compatible later checkpoint also attests the fallback.
  const auto marked = index.Insert(IndexCheckpoint(ledger, Tokens{1, 2, 3}),
                                   ResidentComponents(), 2);
  assert(index.Lookup(Query(tokens, 2)).selected->boundary == 4);
  index.Erase(marked);
  // Missing draft components and changed layouts must never become hits, even
  // if the producer supplied the same compatibility bytes by mistake.
  auto incomplete = IndexCheckpoint(ledger, tokens, {1}, {},
                                    std::span(kIndexComponents).first(1));
  const auto partial = index.Insert(incomplete, ResidentComponents());
  index.Erase(id);
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  index.Erase(partial);
  auto changed = kIndexComponents;
  changed[0].layout_version++;
  (void)index.Insert(IndexCheckpoint(ledger, tokens, {1}, {}, changed),
                     ResidentComponents());
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  auto frontier = IndexLive(tokens);
  frontier.positions.back().valid_rows--;
  (void)index.Insert(frontier);
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
  frontier.positions.pop_back();
  (void)index.Insert(frontier);
  assert(index.Lookup(Query(tokens)).reason ==
         SelectionReason::kMissingComponent);
}
void CheckpointPrivateBoundary() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens tokens{1, 2, 3, 4};
  for (const Rows position : {3ULL, 5ULL}) {
    auto malformed =
        IndexCheckpoint(ledger, tokens, {1}, {}, kIndexComponents, position);
    const auto entry = index.Insert(malformed, ResidentComponents());
    assert(index.Lookup(Query(tokens)).reason ==
           SelectionReason::kMissingComponent);
    assert(index.CachedPrefixTokens(Query(tokens)) == 0);
    assert(!index.HasCheckpoint(Query(tokens), 0));
    index.Erase(entry);
  }
  const auto deep =
      index.Insert(IndexCheckpoint(ledger, tokens), ResidentComponents());
  // An incoherent shorter checkpoint cannot supply a stable fallback.
  auto malformed =
      IndexCheckpoint(ledger, Tokens{1, 2}, {1}, {}, kIndexComponents, 1);
  auto entry = index.Insert(malformed, ResidentComponents());
  assert(index.Lookup(Query(tokens, 2)).reason ==
         SelectionReason::kStablePrefixBoundary);
  index.Erase(entry);
  // An incoherent later checkpoint cannot attest an established fallback
  // either.
  malformed =
      IndexCheckpoint(ledger, Tokens{1, 2, 3}, {1}, {}, kIndexComponents, 2);
  entry = index.Insert(malformed, ResidentComponents(), 2);
  assert(index.Lookup(Query(tokens, 2)).reason ==
         SelectionReason::kStablePrefixBoundary);
  index.Erase(entry);
  (void)index.Insert(IndexCheckpoint(ledger, Tokens{1, 2}),
                     ResidentComponents());
  assert(index.Lookup(Query(tokens, 2)).selected->entry == deep);
}
void TiesAndIsolation() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  index.Register({2}, kIndexComponents);
  const Tokens tokens{1, 2, 3}, prompt{1, 2, 3, 4};
  auto earlier = IndexCheckpoint(ledger, tokens);
  auto later = IndexCheckpoint(ledger, tokens);
  const auto second = index.Insert(later, ResidentComponents());
  (void)index.Insert(earlier, ResidentComponents());
  assert(index.Lookup(Query(prompt)).selected->checkpoint == earlier);
  auto b = index.Insert(IndexLive(tokens, {2}));
  auto a = index.Insert(IndexLive(tokens, {1}));
  assert(index.Lookup(Query(prompt)).selected->entry == a);
  index.Erase(a);
  assert(index.Lookup(Query(prompt)).selected->entry == b);
  index.Erase(b);
  auto deeper =
      index.Insert(IndexCheckpoint(ledger, prompt), ResidentComponents());
  (void)index.Insert(IndexLive(tokens));
  assert(index.Lookup(Query(prompt)).selected->entry == deeper);
  auto query = Query(prompt);
  query.compatibility = {2};
  assert(!index.Lookup(query).selected);
  query.compatibility = {3};
  assert(index.Lookup(query).reason == SelectionReason::kUnknownCompatibility);
  query = Query(prompt);
  query.reuse = false;
  assert(index.Lookup(query).reason == SelectionReason::kDisabled);
  assert(index.CachedPrefixTokens(query) == 4);
  auto pin = index.Lookup(Query(prompt)).selected->checkpoint;
  index.Erase(deeper);
  assert(pin->Boundary() == 4);
  index.Erase(second);
  index.Erase(second);
}
void LongPromptAndMemory() {
  ResourceLedger ledger{kIndexLimits};
  {
    std::vector<Token> prompt(131072, 7);
    std::vector<std::shared_ptr<const Checkpoint>> checkpoints;
    for (std::size_t count = 1024; count <= prompt.size(); count += 1024)
      checkpoints.push_back(
          IndexCheckpoint(ledger, std::span(prompt).first(count)));
    PrefixIndex forward{ledger}, reverse{ledger};
    forward.Register({1}, kIndexComponents);
    reverse.Register({1}, kIndexComponents);
    const auto empty = forward.MetadataBytes();
    std::vector<IndexEntryId> entries;
    for (const auto& cp : checkpoints)
      (void)forward.Insert(cp, ResidentComponents());
    for (auto i = checkpoints.rbegin(); i != checkpoints.rend(); ++i)
      entries.push_back(reverse.Insert(*i, ResidentComponents()));
    assert(forward.MetadataBytes() == reverse.MetadataBytes());
    assert(forward.MetadataBytes() <
           prompt.size() * sizeof(Token) + 128 * 1024);
    const auto result = reverse.Lookup(Query(prompt));
    assert(result.selected->boundary == 131072);
    assert(result.candidates.size() == 128);
    prompt.back() = 9;
    assert(reverse.CachedPrefixTokens(Query(prompt)) == 130048);
    assert(reverse.CommonPrefixTokens(Query(prompt)) == 131071);
    for (const auto id : entries)
      reverse.Erase(id);
    assert(reverse.MetadataBytes() == empty);
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void PrefixChurn() {
  ResourceLedger ledger{kIndexLimits};
  {
    PrefixIndex index{ledger};
    index.Register({1}, kIndexComponents);
    std::vector<Token> prompt(131072, 7);
    (void)index.Insert(IndexCheckpoint(ledger, prompt), ResidentComponents());
    const auto retained = index.MetadataBytes();
    for (std::size_t count = 128; count < prompt.size(); count += 128) {
      auto entry = index.Insert(IndexLive(std::vector<Token>(count, 7)));
      index.Erase(entry);
      assert(index.MetadataBytes() == retained);
    }
    const auto result = index.Lookup(Query(prompt));
    assert(result.candidates.size() == 1);
    assert(result.selected->boundary == prompt.size());
  }
  assert(ledger.Snapshot().total_bytes == 0);
}
void DeferredCompaction() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens tokens{1, 2, 3, 4, 5};
  (void)index.Insert(IndexCheckpoint(ledger, tokens), ResidentComponents());
  const auto retained = index.MetadataBytes();
  for (const auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    auto short_live = index.Insert(IndexLive({1, 2}));
    ledger.FailAfter(step, 0);
    index.Erase(short_live);
    assert(index.Lookup(Query(tokens)).candidates.size() == 1);
    assert(index.CachedPrefixTokens(Query(Tokens{1, 2})) == 0);
    assert(index.MetadataBytes() > retained);
    auto retry = index.Insert(IndexLive({1, 2, 3}));
    index.Erase(retry);
    assert(index.MetadataBytes() == retained);
  }
}
void AdmissionBeforeAllocation() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  std::vector<Token> prompt(131072, 7);
  auto checkpoint = IndexCheckpoint(ledger, prompt);
  auto availability = ResidentComponents();
  const auto before = ledger.Snapshot().total_bytes;
  ledger.FailAfter(LedgerStep::kReserve, 1);
  largest_allocation = 0;
  watch_allocations = true;
  bool rejected = false;
  try {
    (void)index.Insert(checkpoint, std::move(availability));
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  watch_allocations = false;
  assert(rejected);
  assert(largest_allocation < prompt.size() * sizeof(Token));
  assert(ledger.Snapshot().total_bytes == before);
  (void)index.Insert(checkpoint, ResidentComponents());
  auto prefix = IndexCheckpoint(ledger, std::span(prompt).first(1024));
  availability = ResidentComponents();
  const auto split_before = ledger.Snapshot().total_bytes;
  ledger.FailAfter(LedgerStep::kReserve, 2);
  largest_allocation = 0;
  watch_allocations = true;
  rejected = false;
  try {
    (void)index.Insert(prefix, std::move(availability));
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  watch_allocations = false;
  assert(rejected);
  assert(largest_allocation < (prompt.size() - 1024) * sizeof(Token));
  assert(ledger.Snapshot().total_bytes == split_before);

  std::vector<ComponentDescriptor> inventory(1024);
  for (std::size_t i = 0; i < inventory.size(); ++i)
    inventory[i] = {{static_cast<std::uint32_t>(i)},
                    1,
                    ComponentKind::kPrivateState,
                    0,
                    0,
                    32};
  ledger.FailAfter(LedgerStep::kReserve, 0);
  largest_allocation = 0;
  watch_allocations = true;
  rejected = false;
  try {
    index.Register({2}, inventory);
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  watch_allocations = false;
  assert(rejected);
  assert(largest_allocation < inventory.size() * sizeof(ComponentDescriptor));
  assert(ledger.Snapshot().total_bytes == split_before);
}
void AdmissionAndValidation() {
  ResourceLedger ledger{kIndexLimits};
  const Tokens original{1, 2, 3, 4}, split{1, 2, 9}, prefix{1, 2};
  auto retained = IndexCheckpoint(ledger, original);
  auto branch = IndexCheckpoint(ledger, split);
  auto parent = IndexCheckpoint(ledger, prefix);
  for (const auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    for (std::size_t failure = 0; failure < 5; ++failure) {
      PrefixIndex index{ledger};
      index.Register({1}, kIndexComponents);
      (void)index.Insert(retained, ResidentComponents());
      const auto bytes = ledger.Snapshot().total_bytes;
      ledger.FailAfter(step, failure);
      bool failed = false;
      try {
        (void)index.Insert(branch, ResidentComponents());
      } catch (const std::bad_alloc&) {
        failed = true;
      }
      ledger.ClearFaults();
      assert(index.Lookup(Query(original)).selected->checkpoint == retained);
      if (failed)
        assert(ledger.Snapshot().total_bytes == bytes);
      else
        assert(index.Lookup(Query(split)).selected->checkpoint == branch);
    }
  }
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const auto original_id = index.Insert(retained, ResidentComponents());
  const auto parent_id = index.Insert(parent, ResidentComponents());
  const auto branch_id = index.Insert(branch, ResidentComponents());
  index.Erase(parent_id);
  assert(index.Lookup(Query(original)).selected->entry == original_id);
  assert(index.Lookup(Query(split)).selected->entry == branch_id);
  index.Erase(original_id);
  index.Erase(branch_id);
  assert(index.CachedPrefixTokens(Query(original)) == 0);
  Invalid([&] { index.Register({1}, std::span(kIndexComponents).first(1)); });
  Invalid([&] { (void)index.Insert(retained, {{{9}, true, false}}); });
  Invalid([&] {
    (void)index.Insert(retained, {{{1}, true, false}, {{1}, true, false}});
  });
  Invalid([&] { (void)index.Insert(retained, ResidentComponents(), 5); });
  Invalid([&] { (void)index.Lookup(Query(original, 5)); });
  auto query = Query(original);
  query.input = InputIdentity(1);
  Invalid([&] { (void)index.Lookup(query); });
  const auto id = index.Insert(retained, ResidentComponents());
  const auto old_bytes = ledger.Snapshot().total_bytes;
  ledger.FailAfter(LedgerStep::kReserve, 0);
  bool failed = false;
  try {
    index.SetAvailability(id, {});
  } catch (const std::bad_alloc&) {
    failed = true;
  }
  assert(failed);
  assert(ledger.Snapshot().total_bytes == old_bytes);
  assert(index.Lookup(Query(original)).selected->entry == id);
}
struct SnapshotDirectory {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("gufo-prefix-snapshot-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  SnapshotDirectory() { assert(std::filesystem::create_directory(path)); }
  ~SnapshotDirectory() { std::filesystem::remove_all(path); }
};
DiskFileId SnapshotFile() {
  static std::uint64_t next = 1;
  const auto value = next++;
  DiskFileId file{};
  for (unsigned i = 0; i < 8; ++i)
    file[i] = value >> (i * 8);
  return file;
}
std::shared_ptr<const DiskDescription> SnapshotPublish(
    DiskStore& store, const Checkpoint& checkpoint,
    CompatibilityDigest digest) {
  DiskManifest manifest;
  manifest.checkpoint = checkpoint.Id();
  manifest.lineage = checkpoint.Lineage();
  manifest.compatibility = digest;
  manifest.input = checkpoint.Input();
  manifest.tokens.assign(checkpoint.Tokens().begin(),
                         checkpoint.Tokens().end());
  std::vector<std::vector<std::uint8_t>> payloads;
  std::vector<DiskWriteBuffer> buffers;
  auto add = [&](const Payload& payload, bool private_file) {
    auto& bytes = payloads.emplace_back(payload.Bytes(), 0);
    const auto file = SnapshotFile();
    buffers.push_back({private_file, file, bytes});
    return DiskPayload{file, bytes.size(), DiskChecksum(bytes)};
  };
  for (const auto& component : checkpoint.Components()) {
    DiskComponent copy{component.descriptor, component.position, {}, {}, {}};
    for (const auto& chunk : component.chunks)
      copy.chunks.push_back(add(chunk.Storage(), false));
    if (component.tail)
      copy.tail = add(*component.tail, true);
    if (component.private_state)
      copy.private_state = add(*component.private_state, true);
    manifest.components.push_back(std::move(copy));
  }
  store.Publish(SnapshotFile(), manifest, buffers);
  return store.Describe(checkpoint.Id());
}
void SnapshotDiskCandidates() {
  PrefixIndexSnapshot snapshot;
  const Tokens tokens{1, 2, 3, 4};
  {
    ResourceLedger ledger{kIndexLimits};
    {
      SnapshotDirectory directory;
      DiskStore store{ledger, directory.path, 1 << 20};
      PrefixIndex index{ledger};
      CompatibilityDigest digest{};
      digest[0] = 42;
      index.Register({1}, kIndexComponents, digest);
      auto ram = IndexCheckpoint(ledger, tokens);
      auto mixed = IndexCheckpoint(ledger, tokens);
      auto disk = IndexCheckpoint(ledger, tokens);
      const auto ram_id = index.Insert(ram, ResidentComponents(), 2);
      const auto mixed_id = index.Insert(mixed, ResidentComponents(), 2);
      auto ram_description = SnapshotPublish(store, *ram, digest);
      auto mixed_description = SnapshotPublish(store, *mixed, digest);
      auto disk_description = SnapshotPublish(store, *disk, digest);
      index.AttachDurable(ram_id, ram_description);
      index.AttachDurable(mixed_id, mixed_description);
      index.SetAvailability(
          mixed_id, {{{1}, true, true}, {{2}, false, true}, {{3}, true, true}});
      const auto disk_id = index.Insert({1}, disk_description, 2);
      const auto live_id = index.Insert(IndexLive(tokens));
      SnapshotParity(index, Query(tokens));
      SnapshotParity(index, Query(tokens, 2));
      const auto ram_owners = ram.use_count();
      const auto description_owners = disk_description.use_count();
      snapshot = index.Snapshot();
      assert(ram.use_count() == ram_owners);
      assert(disk_description.use_count() == description_owners);
      const auto result = snapshot.Lookup(Query(tokens, 2));
      assert(result.size() == 4);
      assert(result[0].entry == live_id && result[0].live);
      assert(result[1].entry == ram_id && result[1].ram && !result[1].disk);
      assert(result[2].entry == mixed_id && result[2].ram && result[2].disk);
      assert(result[3].entry == disk_id && !result[3].ram && result[3].disk);
      for (std::size_t i = 1; i < result.size(); ++i)
        assert(result[i].transfer_bytes == 4 * 4 + 2 * 2 + 32);
      // Quarantine is an atomic validity change outside index publication.
      // It also invalidates a wholly RAM candidate bound to that publication.
      store.Invalidate(ram->Id(), ram_description->Epoch());
      store.Invalidate(mixed->Id(), mixed_description->Epoch());
      store.Invalidate(disk->Id(), disk_description->Epoch());
      assert(index.Lookup(Query(tokens)).candidates.size() == 1);
      assert(snapshot.Lookup(Query(tokens)).size() == 1);
      // The only live frontier cannot attest the now-invalid stable fallback.
      assert(snapshot.Lookup(Query(tokens, 2)).empty());
      // Neither construction nor lookup retains disk pins.
      assert(store.Retire(ram->Id()));
      assert(store.Retire(mixed->Id()));
      assert(store.Retire(disk->Id()));
    }
    assert(ledger.Snapshot().total_bytes == 0);
  }
  assert(snapshot.Lookup(Query(tokens)).size() == 1);
}
// A size-only adapter for real managed borrowing. Release deliberately fails
// preservation so validity changes without any PrefixIndex mutation.
struct SnapshotStream final : Stream {
  TransferResult Synchronize() noexcept override {
    return TransferResult::kSucceeded;
  }
};
struct SnapshotAdapter final : Adapter {
  struct TestSlot final : Slot {
    MutationGuard& guard;
    Rows rows{};
    explicit TestSlot(MutationGuard& g) : guard(g) {}
    bool IsValid() const noexcept override { return true; }
  };
  std::size_t component_count{1};
  Capabilities GetCapabilities() const override { return {}; }
  std::span<const ComponentDescriptor> Components() const override {
    return std::span(kIndexComponents).first(component_count);
  }
  Identity CompatibilityIdentity() const override { return {1}; }
  std::unique_ptr<Slot> CreateSlot(MutationGuard& guard) override {
    return std::make_unique<TestSlot>(guard);
  }
  std::vector<ComponentPosition> Positions(const Slot& slot) const override {
    std::vector<ComponentPosition> positions;
    for (const auto& descriptor : Components())
      positions.push_back(
          {descriptor.id,
           static_cast<const TestSlot&>(slot).rows / descriptor.id.value});
    return positions;
  }
  std::vector<Rows> PlanPrefill(const Slot&, Rows, Rows) const override {
    return {};
  }
  void BeginRestore(Slot&, std::span<const ComponentPosition>) override {
    throw std::logic_error("unused restore");
  }
  Completion CapturePrivate(const Slot&, ComponentId, std::span<std::byte>,
                            Stream&) override {
    throw std::logic_error("unused private capture");
  }
  Completion CopyRowsOut(const Slot&, ComponentId, Rows, Rows,
                         std::span<std::byte>, Stream&) override {
    throw std::runtime_error("preservation failure");
  }
  Completion CopyRowsIn(Slot&, ComponentId, Rows, Rows,
                        std::span<const std::byte>, Stream&) override {
    throw std::logic_error("unused row restore");
  }
  Completion LoadPrivate(Slot&, ComponentId, std::span<const std::byte>,
                         Stream&) override {
    throw std::logic_error("unused private restore");
  }
  bool Validate(Slot&, std::span<const ComponentPosition>) override {
    return false;
  }
  bool Invalidate(Slot& slot) noexcept override {
    auto& guard = static_cast<TestSlot&>(slot).guard;
    for (const auto& descriptor : Components())
      guard.BeforeRelease(descriptor.id, 0, std::numeric_limits<Rows>::max());
    static_cast<TestSlot&>(slot).rows = 0;
    guard.AfterReset();
    return true;
  }
};
std::shared_ptr<const Checkpoint> SnapshotBorrow(ResourceLedger& ledger,
                                                 SlotLease& lease,
                                                 const SnapshotAdapter& adapter,
                                                 const Tokens& tokens) {
  auto history = ExecutionHistory::Cold(ledger, adapter.Components(), {1});
  std::vector<ComponentPosition> positions;
  for (const auto& descriptor : adapter.Components())
    positions.push_back({descriptor.id, tokens.size() / descriptor.id.value});
  return history.Capture(
      {tokens, InputIdentity(tokens.size()), positions,
       CheckpointPurpose::kPrompt, 1},
      [&](const PayloadRequest& request) {
        auto pool =
            ledger.Reserve(ResourceCategory::kBackingFree, request.bytes)
                .Convert();
        auto backing = std::make_shared<std::vector<std::byte>>(request.bytes);
        return Payload::Borrowed(lease.Borrow(
            request.component, request.first, request.end,
            pool.ReserveBacking(request.category), backing, *backing));
      });
}
void SnapshotMixedValidity() {
  ResourceLedger ledger{kIndexLimits};
  SnapshotDirectory directory;
  DiskStore store{ledger, directory.path, 1 << 20};
  SnapshotAdapter adapter;
  adapter.component_count = 2;
  SnapshotStream stream;
  LeasedSlot slot{ledger, adapter, stream, {1}};
  auto lease = slot.Acquire();
  auto& execution = static_cast<SnapshotAdapter::TestSlot&>(lease.Execution());
  execution.rows = 4;
  PrefixIndex index{ledger};
  const Tokens tokens{1, 2, 3, 4};
  CompatibilityDigest digest{};
  index.Register({1}, adapter.Components(), digest);
  auto checkpoint = SnapshotBorrow(ledger, lease, adapter, tokens);
  const auto id =
      index.Insert(checkpoint, {{{1}, true, false}, {{2}, true, false}});
  auto description = SnapshotPublish(store, *checkpoint, digest);
  index.AttachDurable(id, description);
  // The second RAM component dies before construction. Candidate resolves
  // just that component to disk; whole-checkpoint IsValid would over-reject.
  execution.guard.BeforeRelease({2}, 0, 2);
  assert(!checkpoint->IsValid());
  SnapshotParity(index, Query(tokens));
  const auto bytes = ledger.Snapshot().total_bytes;
  const auto snapshot = index.Snapshot();
  assert(ledger.Snapshot().total_bytes == bytes);
  const auto mixed = snapshot.Lookup(Query(tokens));
  assert(mixed.size() == 1 && mixed[0].ram && mixed[0].disk);
  execution.guard.BeforeRelease({1}, 0, 4);
  // A selected RAM source dying after publication fails closed. A newly
  // constructed snapshot can resolve the entirely durable replacement plan.
  assert(snapshot.Lookup(Query(tokens)).empty());
  const auto refreshed = index.Snapshot().Lookup(Query(tokens));
  assert(refreshed.size() == 1 && !refreshed[0].ram && refreshed[0].disk);
}
void SnapshotBorrowedInvalidation() {
  ResourceLedger ledger{kIndexLimits};
  SnapshotAdapter adapter;
  SnapshotStream stream;
  LeasedSlot slot{ledger, adapter, stream, {1}};
  auto lease = slot.Acquire();
  static_cast<SnapshotAdapter::TestSlot&>(lease.Execution()).rows = 4;
  PrefixIndex index{ledger};
  index.Register({1}, adapter.Components());
  const Tokens shallow{1, 2}, deep{1, 2, 3, 4};
  auto borrowed = SnapshotBorrow(ledger, lease, adapter, shallow);
  (void)index.Insert(borrowed, {{{1}, true, false}});
  const auto deep_id =
      index.Insert(IndexCheckpoint(ledger, deep, {1}, {}, adapter.Components()),
                   {{{1}, true, false}});
  const auto snapshot = index.Snapshot();
  assert(borrowed->IsValid());
  assert(snapshot.Lookup(Query(deep, 2)).front().entry == deep_id);
  lease.Reset();
  assert(!borrowed->IsValid());
  assert(index.Lookup(Query(deep, 2)).candidates.empty());
  assert(snapshot.Lookup(Query(deep, 2)).empty());
  assert(snapshot.Lookup(Query(deep)).size() == 1);
  assert(snapshot.Lookup(Query(deep)).front().entry == deep_id);
  SnapshotParity(index, Query(deep));
  // Expiration is also checked before deciding whether a stable fallback
  // exists; a deep checkpoint cannot keep using a dead shorter attestation.
  const auto fallback = index.Insert(
      IndexCheckpoint(ledger, shallow, {1}, {}, adapter.Components()),
      {{{1}, true, false}});
  const auto expiring = index.Snapshot();
  assert(expiring.Lookup(Query(deep, 2)).front().entry == deep_id);
  index.Erase(fallback);
  assert(index.Lookup(Query(deep, 2)).candidates.empty());
  assert(expiring.Lookup(Query(deep, 2)).empty());
}
void SnapshotIdentityAndReadiness() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  index.Register({2}, kIndexComponents);
  const Tokens tokens{1, 2, 3, 4, 5};
  const auto short_id =
      index.Insert(IndexCheckpoint(ledger, Tokens{1, 2}), ResidentComponents());
  const auto image_id =
      index.Insert(IndexCheckpoint(ledger, Tokens{1, 2, 3, 4}, {1}, {10}),
                   ResidentComponents(), 2);
  const auto live_id = index.Insert(IndexLive({1, 2, 3, 4}, {7}, {10}));
  const auto bad_id =
      index.Insert(IndexCheckpoint(ledger, tokens), {{{1}, true, false}});
  (void)bad_id;
  for (Rows stable = 0; stable <= tokens.size(); ++stable) {
    SnapshotParity(index, Query(tokens, stable));
    SnapshotParity(index, Query(tokens, stable, {20}, {{2, {}}, {4, {10}}}));
    SnapshotParity(index, Query(tokens, stable, {20}, {{2, {}}}));
  }
  auto query = Query(tokens, 2, {20}, {{2, {}}, {4, {10}}});
  const auto metadata = index.MetadataBytes();
  const auto bytes = ledger.Snapshot().total_bytes;
  std::size_t calls{};
  const auto snapshot = index.Snapshot([&](const PrefixCandidate& candidate) {
    ++calls;
    assert(candidate.entry != bad_id);
    return candidate.entry == image_id;
  });
  assert(calls == 3);
  assert(index.MetadataBytes() == metadata);
  assert(ledger.Snapshot().total_bytes == bytes);
  const auto result = snapshot.Lookup(query);
  assert(result.size() == 3);
  assert(result[0].entry == live_id && result[0].live && !result[0].full_ready);
  assert(result[1].entry == image_id && result[1].ram && result[1].full_ready);
  assert(result[2].entry == short_id && !result[2].full_ready);
  // Readiness annotates observations; it does not change ordering/fallback.
  query.reuse = false;
  SnapshotParity(index, query);
  assert(snapshot.Lookup(query).empty());
  query.reuse = true;
  for (const auto& identity : {Identity{2}, Identity{3}}) {
    query.compatibility = identity;
    SnapshotParity(index, query);
    assert(snapshot.Lookup(query).empty());
  }
  Invalid([&] { (void)snapshot.Lookup(Query(tokens, tokens.size() + 1)); });
  query = Query(tokens);
  query.input = InputIdentity(1);
  Invalid([&] { (void)snapshot.Lookup(query); });
  Invalid([&] { (void)PrefixIndexSnapshot{}.Lookup(query); });
  // Snapshot creation and HTTP lookup need no ledger admission for coherent
  // RAM/live metadata, even when the ledger refuses every new reservation.
  ledger.FailAfter(LedgerStep::kReserve, 0);
  const auto independent = index.Snapshot();
  assert(independent.Lookup(Query(tokens)).size() == 1);
  assert(snapshot.Lookup(Query(tokens)).size() == 1);
  assert(calls == 3);
  assert(ledger.Snapshot().total_bytes == bytes);
  ledger.ClearFaults();
}
void SnapshotOrderingAndLifetime() {
  PrefixIndexSnapshot snapshot;
  const Tokens tokens{1, 2, 3};
  assert(snapshot.Lookup(Query(tokens)).empty());
  {
    ResourceLedger ledger{kIndexLimits};
    std::shared_ptr<const Checkpoint> retained;
    {
      PrefixIndex index{ledger};
      index.Register({1}, kIndexComponents);
      auto earlier = IndexCheckpoint(ledger, tokens);
      auto later = IndexCheckpoint(ledger, tokens);
      const auto late = index.Insert(later, ResidentComponents());
      const auto early = index.Insert(earlier, ResidentComponents());
      const auto duplicate = index.Insert(earlier, ResidentComponents());
      auto frontier = IndexLive(tokens, {2});
      frontier.location.generation = 2;
      const auto generation2 = index.Insert(frontier);
      frontier.location.generation = 1;
      const auto generation1 = index.Insert(frontier);
      const auto first_slot = index.Insert(IndexLive(tokens, {1}));
      SnapshotParity(index, Query(tokens));
      assert(earlier.use_count() == 3 && later.use_count() == 2);
      snapshot = index.Snapshot();
      assert(earlier.use_count() == 3 && later.use_count() == 2);
      const auto ordered = snapshot.Lookup(Query(tokens));
      const std::vector<IndexEntryId> ids{first_slot, generation1, generation2,
                                          early,      duplicate,   late};
      assert(ordered.size() == ids.size());
      for (std::size_t i = 0; i < ids.size(); ++i)
        assert(ordered[i].entry == ids[i]);
      // Later index availability mutations do not alter copied observations.
      index.SetLiveAvailable(first_slot, false);
      index.SetAvailability(early, {});
      assert(snapshot.Lookup(Query(tokens)).size() == ids.size());
      // The snapshot is not a payload owner. Expiration removes only the
      // affected checkpoint entries, even while the snapshot remains alive.
      const std::weak_ptr<const Checkpoint> weak = later;
      index.Erase(late);
      later.reset();
      assert(weak.expired());
      assert(snapshot.Lookup(Query(tokens)).size() == ids.size() - 1);
      retained = earlier;
    }
    // No pointer back to the destroyed index is needed for surviving sources.
    assert(snapshot.Lookup(Query(tokens)).size() == 5);
    retained.reset();
    assert(snapshot.Lookup(Query(tokens)).size() == 3);
    assert(ledger.Snapshot().total_bytes == 0);
  }
  // Destruction of the index AND ledger leaves standalone live metadata safe.
  const auto lives = snapshot.Lookup(Query(tokens));
  assert(lives.size() == 3);
  assert(std::ranges::all_of(lives, [](const auto& c) { return c.live; }));
  const auto copy = snapshot;
  snapshot = {};
  assert(copy.Lookup(Query(tokens)).size() == 3);
  std::vector<std::jthread> readers;
  for (unsigned i = 0; i < 4; ++i)
    readers.emplace_back([&] {
      for (unsigned iteration = 0; iteration < 100; ++iteration)
        assert(copy.Lookup(Query(tokens)).size() == 3);
    });
}
void RepeatedReadOnlyPlan() {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  index.Register({1}, kIndexComponents);
  const Tokens shallow{1, 2}, deep{1, 2, 3, 4}, prompt{1, 2, 3, 4, 5};
  const auto root =
      index.Insert(IndexCheckpoint(ledger, shallow), ResidentComponents());
  const auto selected =
      index.Insert(IndexCheckpoint(ledger, deep), ResidentComponents());
  const auto metadata = index.MetadataBytes();
  const auto bytes = ledger.Snapshot().total_bytes;
  for (int i = 0; i < 100; ++i) {
    {
      const auto plan = index.Lookup(Query(prompt, 2));
      assert(plan.selected->entry == selected && plan.selected->boundary == 4);
    }
    assert(index.MetadataBytes() == metadata);
    assert(ledger.Snapshot().total_bytes == bytes);
  }
  // A common token prefix is not a restorable arbitrary hybrid boundary.
  const Tokens branch{1, 2, 3, 9};
  assert(index.Lookup(Query(branch, 2)).selected->entry == root);
  assert(!index.Lookup(Query(prompt, 2, {9})).selected);
  auto disabled = Query(prompt, 2);
  disabled.reuse = false;
  assert(!index.Lookup(disabled).selected);
}
}  // namespace
int main() {
  InvalidationTelemetryDoesNotOwnOrBlockEntries();
  SnapshotDiskCandidates();
  SnapshotMixedValidity();
  SnapshotBorrowedInvalidation();
  SnapshotIdentityAndReadiness();
  SnapshotOrderingAndLifetime();
  RepeatedReadOnlyPlan();
  SelectionTable();
  CheckpointPresence();
  ImagesAndLearning();
  CoherenceAndFallback();
  CheckpointPrivateBoundary();
  TiesAndIsolation();
  LongPromptAndMemory();
  PrefixChurn();
  DeferredCompaction();
  AdmissionBeforeAllocation();
  AdmissionAndValidation();
  std::cout << "prefix index tests passed\n";
}
