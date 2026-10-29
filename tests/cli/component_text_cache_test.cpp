#include "src/cli/serve/component_text_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "src/cache/retention.hpp"
#include "src/cli/serve/cache_lookup_support.hpp"
#include "src/cli/serve/component_cache_metrics.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/qfn_kv_ingress_validation.hpp"
#include "src/cli/serve/text_generation_scheduler.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/env.hpp"
#include "src/core/platform/runtime_fingerprint.hpp"
#include "tests/cache/fake_adapter.hpp"
#include "tests/support/temporary_directory.hpp"

using namespace gufo;
using namespace gufo::server;
using namespace gufo::cache::testing;
namespace {

struct State final : TextRunnerState {
  FakeAdapter& adapter;
  cache::Slot& slot;
  State(FakeAdapter& adapter, cache::Slot& slot)
      : adapter(adapter), slot(slot) {}
  void Invalidate() noexcept override { (void)adapter.Invalidate(slot); }
};
class Runner : public TextModelRunner {
public:
  mutable FakeAdapter* adapter{};
  mutable std::size_t captures{};
  bool defer_idle{};
  bool in_pass{};
  bool partial{};
  std::size_t private_blocks{128};
  std::size_t row_blocks{2048};
  std::uint32_t max_context{4096};
  cache::Rows rows_per_chunk{4};
  mutable std::size_t through_calls{};
  std::function<void(std::string_view)> barrier;
  bool UsesComponentCache() const override { return true; }
  bool LookupFullCheckpointReady() const override { return true; }
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "component-fake",
            .state_abi = "component-fake-v1",
            .max_context = max_context,
            .capabilities = {.incremental_prefill = true,
                             .snapshot = true,
                             .fork = true,
                             .final_token_advance_required = false,
                             .prefix_reuse = true,
                             .in_pass_checkpoint = in_pass,
                             .partial_prefill_retention = partial}};
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    return {.state_capacity_bytes = 1ULL << 30,
            .per_request_state_bytes = 256,
            .retained_snapshot_capacity_bytes = 1ULL << 22};
  }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{TextExecutionPlanKind::kSerial, 1},
            {TextExecutionPlanKind::kBatched, 2}};
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view) const override {
    return {1};
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest&) const override {
    return std::vector<TextRunnerToken>{1};
  }
  std::string Decode(std::span<const TextRunnerToken>) const override {
    return "";
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error("legacy state must never be constructed");
  }
  // Image prompts carry their cache identity in the context.
  void SetPromptContext(
      TextRunnerState&,
      std::shared_ptr<const TextPromptContext>) const override {}
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    struct Block {
      cache::ResourceCharge pool;
      std::span<std::byte> data;
      std::shared_ptr<const void> owner;
    };
    auto blocks = std::make_shared<std::vector<Block>>();
    // Physical backing commits once, before any request; assigned owner tokens
    // account for the entire block including unused bytes.
    auto data = std::make_shared<std::vector<std::byte>>(row_blocks * 16 +
                                                         private_blocks * 64);
    for (std::size_t i = 0; i < row_blocks + private_blocks; ++i) {
      const auto size = i < row_blocks ? 16ULL : 64ULL;
      const auto offset =
          i < row_blocks ? i * 16 : row_blocks * 16 + (i - row_blocks) * 64;
      auto reservation =
          ledger.Reserve(cache::ResourceCategory::kBackingFree, size);
      auto bytes = std::span(*data).subspan(offset, size);
      blocks->push_back({reservation.Convert(), bytes,
                         std::shared_ptr<const void>(data, bytes.data())});
    }
    auto resources = std::make_unique<ComponentCacheResources>();
    resources->adapter = std::make_unique<FakeAdapter>(rows_per_chunk);
    adapter = dynamic_cast<FakeAdapter*>(resources->adapter.get());
    resources->owner = blocks;
    resources->stream = [this, foreground = std::this_thread::get_id()]()
        -> std::unique_ptr<cache::Stream> {
      if (defer_idle && std::this_thread::get_id() != foreground)
        return {};
      return std::make_unique<FakeStream>(true);
    };
    resources->test_barrier = [this](std::string_view point) {
      if (barrier)
        barrier(point);
    };
    resources->transfer_batch_size = 4;
    resources->allocate = [this, blocks,
                           data](const cache::PayloadRequest& request) {
      for (const auto& block : *blocks) {
        if ((request.category == cache::ResourceCategory::kPrivateState) !=
            (block.data.size() == 64))
          continue;
        try {
          auto reservation = std::make_shared<cache::ResourceReservation>(
              block.pool.ReserveBacking(request.category));
          ++captures;
          return ComponentCaptureBuffer{
              block.data,
              [reservation, owner = block.owner] {
                return cache::Payload::Committed(reservation->Convert(), owner);
              },
              data.get(),
              [reservation, owner = block.owner, bytes = block.data, request](
                  cache::SlotLease& lease, auto source) {
                return cache::Payload::Borrowed(
                    lease.Borrow(request.component, request.first, request.end,
                                 std::move(*reservation), owner,
                                 bytes.first(request.bytes), source));
              }};
        } catch (const std::bad_alloc&) {
        }
      }
      throw cache::ResourceExhausted();
    };
    resources->staging = [](std::size_t size) {
      auto data = std::make_shared<std::vector<std::uint8_t>>(size);
      return cache::StagingAllocation{data, *data};
    };
    resources->copy_observed_rows = [](auto source, auto destination,
                                       cache::Stream& stream) {
      return dynamic_cast<FakeStream&>(stream).Submit([source, destination] {
        std::memcpy(destination.data(), source.data(), destination.size());
        return cache::TransferResult::kSucceeded;
      });
    };
    return resources;
  }
  std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    return std::make_unique<State>(dynamic_cast<FakeAdapter&>(adapter), slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    const auto& s = dynamic_cast<State&>(state);
    assert(s.adapter.Positions(s.slot).front().valid_rows == position);
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t budget) const override {
    auto& s = dynamic_cast<State&>(state);
    const auto count = std::min(budget, prompt.size() - offset);
    s.adapter.Append(s.slot, prompt.subspan(offset, count),
                     prompt.subspan(offset, count));
    return {count, offset + count == prompt.size()};
  }
#if defined(GUFO_ENABLE_PREFILL_PREREQUISITES)
  std::optional<std::size_t> SinglePassPrefillLimit(
      const TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget) const override {
    if (offset != CheckpointPosition(state) || offset >= prompt.size())
      return std::nullopt;
    return std::min(budget, prompt.size() - offset);
  }
#endif
  std::optional<std::size_t> PrefillCheckpointBytes(
      const TextRunnerState&, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::size_t boundary) const override {
    const auto end = offset + std::min(budget, prompt.size() - offset);
    return boundary > offset && boundary < end && end - boundary <= 8
               ? std::optional<std::size_t>(1)
               : std::nullopt;
  }
  TextPrefillStep PrefillThrough(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget, std::size_t boundary,
      std::unique_ptr<TextRunnerSnapshot>* checkpoint,
      const std::function<std::unique_ptr<TextRunnerSnapshot>()>& capture)
      const override {
    ++through_calls;
    auto& s = dynamic_cast<State&>(state);
    const auto count = std::min(budget, prompt.size() - offset);
    const auto head = prompt.subspan(offset, boundary - offset);
    s.adapter.Append(s.slot, head, head);
    try {
      *checkpoint = capture();
    } catch (...) {
      checkpoint->reset();
    }
    const auto tail = prompt.subspan(boundary, offset + count - boundary);
    s.adapter.Append(s.slot, tail, tail);
    return {count, offset + count == prompt.size()};
  }
  TextDecodeSelection SelectNext(TextRunnerState& state,
                                 sampling::SamplerState&) const override {
    auto& s = dynamic_cast<State&>(state);
    return {.token = static_cast<TextRunnerToken>(
                s.adapter.RecurrentHash(s.slot) % 100000),
            .piece = "x"};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& s = dynamic_cast<State&>(state);
    s.adapter.Append(s.slot, {&token, 1}, {&token, 1});
  }
  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    const auto& s = dynamic_cast<const State&>(state);
    return s.adapter.Positions(s.slot).front().valid_rows;
  }
};
void BudgetUsesAllocatedSlotsInsteadOfLazyStateClaim() {
  class BudgetRunner final : public Runner {
  public:
    mutable std::size_t committed{};
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = Runner::ResourceClaim();
      claim.state_capacity_bytes = 4ULL << 30;
      // Capacity includes image storage allocated only when requested. It is
      // much larger than remaining memory, but these text slots fit.
      claim.per_request_state_bytes = 1ULL << 30;
      claim.temporary_scratch_bytes = 0;
      claim.retained_snapshot_capacity_bytes = adapter ? 1 << 20 : 2 << 20;
      claim.retained_snapshot_ceiling_bytes = adapter ? 2 << 20 : 4 << 20;
      return claim;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this, &ledger](std::size_t budget) {
        assert(ledger.Limits().ram_bytes == budget);
        committed = budget;
      };
      return resources;
    }
  };
  for (const auto requested : {0ULL, 3ULL << 20}) {
    auto runner = std::make_shared<BudgetRunner>();
    TextRunnerPool pool(runner, 2, {}, {.capacity_bytes = requested});
    assert(runner->committed == (requested ? 2ULL << 20 : 1ULL << 20));
    std::vector<TextRunnerToken> prompt{10, 11, 12, 13};
    auto request = pool.Acquire(prompt);
    while (!request.prefill_complete())
      (void)request.Prefill(8);
    assert(request.SelectNext().token != 0);
  }
}
void Prefill(TextRunnerPool::Request& request) {
  while (!request.prefill_complete())
    (void)request.Prefill(8);
}
TextRunnerToken Generate(TextRunnerPool::Request& request) {
  const auto token = request.SelectNext().token;
  request.Advance();
  return token;
}
void InPassCapturePreservesBoundaryAndLiveTail() {
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto cold_runner = std::make_shared<Runner>();
  TextRunnerPool cold_pool(cold_runner, 1);
  auto cold =
      cold_pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, false);
  Prefill(cold);
  const auto expected = Generate(cold);
  cold.Invalidate();
  for (const bool fail : {false, true}) {
    auto runner = std::make_shared<Runner>();
    runner->in_pass = true;
    TextRunnerPool pool(runner, 2);
    auto seed =
        pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 35);
    if (fail)
      runner->adapter->FailNextTransfer();
    const auto step = seed.Prefill(64);
    assert(step.consumed_tokens == 40 && step.decode_ready);
    assert(runner->through_calls == 1);
    assert(Generate(seed) == expected);
    if (fail) {
      seed.Invalidate();
      continue;
    }
    (void)seed.Commit();
    auto branch = prompt;
    branch.back() = 999;
    auto restored =
        pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, true, 35);
    assert(restored.cached_prompt_tokens() == 35);
    Prefill(restored);
    const auto resumed = Generate(restored);
    restored.Invalidate();
    auto independent =
        cold_pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(independent);
    assert(Generate(independent) == resumed);
    independent.Invalidate();
  }
}
void ContextOverflowDiagnostics() {
  auto runner = std::make_shared<Runner>();
  runner->max_context = 32;
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> oversized(33, 1);
  for (const bool lookup : {false, true}) {
    std::string message;
    try {
      if (lookup)
        (void)pool.Lookup(oversized);
      else
        (void)pool.Acquire(oversized);
    } catch (const std::length_error& error) {
      message = error.what();
    }
    assert(message == "prompt of 33 tokens exceeds the 32-token model context");
  }
  assert(runner->captures == 0 && runner->through_calls == 0);
  const std::vector<TextRunnerToken> at_limit(32, 1);
  assert(pool.Lookup(at_limit).total_tokens == 32);
  auto request = pool.Acquire(at_limit);
  assert(request.prompt_tokens() == 32);
  request.Invalidate();
}

void ColdAcquireResetsOnce() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  const std::vector<TextRunnerToken> prompt{1, 2, 3};
  auto first = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(runner->adapter->Invalidations() == 1);
  auto& state = dynamic_cast<State&>(first->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  (void)first->Commit(prompt, {}, prompt);
  auto second = cache.Acquire(prompt, {}, {}, false, 0, true);
  assert(!second->cache_hit());
  assert(runner->adapter->Invalidations() == 2);
}
void ColdAssignmentAvoidsDrainingCancelledSlot(bool deep = false) {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(deep ? 4096 : 3);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto completed = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(completed->state());
  auto* ready_slot = &state.slot;
  runner->adapter->Append(state.slot, prompt, prompt);
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  if (deep) {
    assert(completed->TryReserveSnapshot(
        completed->CaptureBytes(), prompt.size(), false,
        SnapshotPurpose::kContinuation, prompt));
    assert(completed->PublishSnapshot(prompt, completed->Capture(), false));
    runner->adapter->GuardRows(state.slot, cache::testing::kTarget, 0,
                               prompt.size());
    runner->adapter->GuardRows(state.slot, cache::testing::kDraft, 0,
                               prompt.size());
  }
  (void)completed->Commit({}, {}, prompt);

  const std::vector<TextRunnerToken> long_prompt{11, 12, 13};
  auto cancelled = cache.Acquire(long_prompt, {}, {}, true, 0, true);
  auto* draining = &dynamic_cast<State&>(cancelled->state()).slot;
  assert(draining != ready_slot);
  runner->adapter->SetExecutionComplete(*draining, false);
  cancelled->Invalidate();

  const std::vector<TextRunnerToken> unrelated{21, 22, 23};
  auto peer = cache.Acquire(unrelated, {}, {}, true, 0, true);
  assert(&dynamic_cast<State&>(peer->state()).slot == ready_slot &&
         "an unrelated peer waited for a draining cancelled slot");
  // Readiness is a preference, not an admission condition. If the ready slot
  // is occupied, the remaining slot is still available for ordered reuse.
  auto fallback = cache.Acquire(unrelated, {}, {}, false, 0, true);
  assert(&dynamic_cast<State&>(fallback->state()).slot == draining);
  runner->adapter->SetExecutionComplete(*draining, true);
  peer->Invalidate();
  fallback->Invalidate();
  if (deep) {
    auto restored = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(restored->cached_tokens() == prompt.size());
    auto& restored_state = dynamic_cast<State&>(restored->state());
    assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
  }
}
void ColdAssignmentKeepsOneDuplicateDeepFrontier() {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 3, 1 << 22, {});
  std::vector<TextRunnerToken> conversation(4096);
  std::iota(conversation.begin(), conversation.end(), 10);
  const auto prefix = std::span(conversation).first(4088);
  const auto reply = std::span(conversation).last(8);
  auto first = cache.Acquire(conversation, {}, {}, true, 0, true);
  auto& first_state = dynamic_cast<State&>(first->state());
  runner->adapter->Append(first_state.slot, prefix, prefix);
  assert(first->TryReserveSnapshot(first->CaptureBytes(), prefix.size(), false,
                                   SnapshotPurpose::kContinuation, prefix));
  assert(first->PublishSnapshot({prefix.begin(), prefix.end()},
                                first->Capture(), false));
  runner->adapter->GuardRows(first_state.slot, cache::testing::kTarget, 0,
                             prefix.size());
  runner->adapter->GuardRows(first_state.slot, cache::testing::kDraft, 0,
                             prefix.size());
  // A sibling restores the same ancestor while the leader remains leased.
  auto second = cache.Acquire(conversation, {}, {}, true, 0, true);
  assert(second->cached_tokens() == prefix.size());
  auto& second_state = dynamic_cast<State&>(second->state());
  runner->adapter->Append(first_state.slot, reply, reply);
  runner->adapter->Append(second_state.slot, reply, reply);
  const auto hash = runner->adapter->RecurrentHash(first_state.slot);
  assert(runner->adapter->RecurrentHash(second_state.slot) == hash);
  (void)first->Commit({}, {}, conversation);
  (void)second->Commit({}, {}, conversation);

  const std::vector<TextRunnerToken> short_prompt{2, 3, 4};
  auto cancelled = cache.Acquire(short_prompt, {}, {}, true, 0, true);
  auto* draining = &dynamic_cast<State&>(cancelled->state()).slot;
  runner->adapter->SetExecutionComplete(*draining, false);
  cancelled->Invalidate();
  auto peer = cache.Acquire(short_prompt, {}, {}, false, 0, true);
  assert(&dynamic_cast<State&>(peer->state()).slot != draining);
  // Once one copy is reassigned, the other is unique and must stay live.
  auto fallback = cache.Acquire(short_prompt, {}, {}, false, 0, true);
  assert(&dynamic_cast<State&>(fallback->state()).slot == draining);
  auto retained = cache.Acquire(conversation, {}, {}, true, 0, true);
  assert(retained->cached_tokens() == conversation.size() &&
         !retained->restored_snapshot_bytes());
  assert(runner->adapter->RecurrentHash(
             dynamic_cast<State&>(retained->state()).slot) == hash);
  runner->adapter->SetExecutionComplete(*draining, true);
}
void FailedRestoreResetsOnce() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  const std::vector<TextRunnerToken> prompt{1, 2, 3};
  auto source = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(source->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), prompt.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  assert(source->PublishSnapshot(prompt, source->Capture(), false));
  const auto before = runner->adapter->Invalidations();
  runner->adapter->FailNextTransfer();
  auto failed = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(!failed->cache_hit());
  assert(runner->adapter->Invalidations() == before + 2);
  const auto metrics = failed->Metrics();
  assert(metrics.restore_attempts == 1 && metrics.restore_failures == 1);
  assert(metrics.restore_failure_reason == "ram_transfer");
  assert(metrics.reuse_reason == CacheReuseReason::kRestoreFailed);
  assert(metrics.indexed_common_tokens == prompt.size());
  assert(metrics.indexed_gap_tokens == prompt.size());
  assert(metrics.used_entry_id == 0);
  // A rejected source must not remain advertised as executable after fallback.
  assert(cache.Lookup(prompt, nullptr, true, 0)->matched_tokens == 0);
  auto& cold = dynamic_cast<State&>(failed->state());
  runner->adapter->Append(cold.slot, prompt, prompt);
  assert(runner->adapter->RecurrentHash(cold.slot) ==
         runner->adapter->RecurrentHash(state.slot));
}
void PartialReuseDiagnostics() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto source = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(source->Metrics().reuse_reason == CacheReuseReason::kNoIndexedPrefix);
  auto& state = dynamic_cast<State&>(source->state());
  const std::vector<TextRunnerToken> shallow(prompt.begin(),
                                             prompt.begin() + 10);
  runner->adapter->Append(state.slot, shallow, shallow);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), shallow.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  assert(source->PublishSnapshot(shallow, source->Capture(), false));
  const std::vector<TextRunnerToken> tail(prompt.begin() + 10, prompt.end());
  runner->adapter->Append(state.slot, tail, tail);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), prompt.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  assert(source->PublishSnapshot(prompt, source->Capture(), false));
  auto edited =
      std::vector<TextRunnerToken>(prompt.begin(), prompt.begin() + 20);
  edited.push_back(999);
  const auto reason_index =
      static_cast<std::size_t>(CacheReuseReason::kBoundaryGap);
  const auto before = CacheReuseReasonCounts()[reason_index].load();
  auto partial = cache.Acquire(edited, {}, {}, true, 0, true);
  const auto metrics = partial->Metrics();
  assert(partial->cached_tokens() == 10);
  assert(metrics.reuse_reason == CacheReuseReason::kBoundaryGap);
  assert(metrics.indexed_common_tokens == 20 &&
         metrics.indexed_gap_tokens == 10);
  assert(metrics.used_entry_id != 0);
  const auto json = ComponentCacheJson(metrics);
  assert(json.member_str("reuse_reason") == "boundary_gap");
  assert(json.member_size("indexed_gap_tokens") == 10);
  (void)partial->Metrics();
  (void)cache.Lookup(edited, nullptr, true, 0);
  assert(CacheReuseReasonCounts()[reason_index].load() == before + 1);
  assert(CacheReuseCountersJson().member_size("boundary_gap") == before + 1);
  partial->Invalidate();
  auto disabled = cache.Acquire(edited, {}, {}, false, 0, true);
  assert(disabled->Metrics().reuse_reason == CacheReuseReason::kDisabled);
  assert(disabled->Metrics().indexed_common_tokens == 0);
  disabled->Invalidate();

  // Cancellation before trying a found candidate is not a restore failure.
  unsigned cancellation_checks = 0;
  auto cancelled = cache.Acquire(
      prompt, [&] { return ++cancellation_checks > 1; }, {}, true, 0, true);
  assert(cancelled->Metrics().reuse_reason == CacheReuseReason::kCancelled);
  assert(cancelled->Metrics().restore_attempts == 0);
  cancelled->Invalidate();

  runner->adapter->FailNextTransfer();
  auto fallback = cache.Acquire(prompt, {}, {}, true, 0, true);
  const auto fallback_metrics = fallback->Metrics();
  assert(fallback->cached_tokens() == shallow.size());
  assert(fallback_metrics.reuse_reason == CacheReuseReason::kRestoreFailed);
  assert(fallback_metrics.restore_attempts == 2 &&
         fallback_metrics.restore_failures == 1);
  assert(fallback_metrics.used_entry_id != 0 &&
         fallback_metrics.used_entry_id != fallback_metrics.selected_entry_id);
  fallback->Invalidate();

  const auto cold_index =
      static_cast<std::size_t>(CacheReuseReason::kNoIndexedPrefix);
  const auto cold_before = CacheReuseReasonCounts()[cold_index].load();
  for (unsigned i = 0; i < 2; ++i) {
    ComponentTextCache fresh(std::make_shared<Runner>(), 1, 1 << 22, {});
    auto cold = fresh.Acquire(prompt, {}, {}, true, 0, true);
  }
  assert(CacheReuseReasonCounts()[cold_index].load() == cold_before + 2);
}
void RestoreFromBusyBorrowedSource() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto source = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(source->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), prompt.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  std::shared_ptr<const TextRunnerSnapshot> snapshot = source->Capture();
  assert(source->PublishSnapshot(prompt, snapshot, false));
  auto gauges = source->Metrics().ledger;
  assert(gauges.reserved_bytes[static_cast<unsigned>(
             cache::ResourceCategory::kBackingAssigned)] == 320);
  auto peer = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(peer->cached_tokens() == prompt.size());
  auto& restored = dynamic_cast<State&>(peer->state());
  assert(runner->adapter->RecurrentHash(restored.slot) == hash);
  // Reading the prefix neither takes the source execution lease nor forces
  // it to materialize. Subsequent overwrite is still guarded.
  gauges = source->Metrics().ledger;
  assert(gauges.reserved_bytes[static_cast<unsigned>(
             cache::ResourceCategory::kBackingAssigned)] == 320);
  const std::array<cache::Token, 1> next{999};
  runner->adapter->Append(state.slot, next, next);
  gauges = source->Metrics().ledger;
  assert(gauges.reserved_bytes[static_cast<unsigned>(
             cache::ResourceCategory::kBackingAssigned)] == 0);
  assert(runner->adapter->RecurrentHash(restored.slot) == hash);
}
void ColdAssignmentPrefersPreservedSlot() {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto source = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(source->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  assert(source->TryReserveSnapshot(source->CaptureBytes(), prompt.size(),
                                    false, SnapshotPurpose::kContinuation, {}));
  std::shared_ptr<const TextRunnerSnapshot> snapshot = source->Capture();
  assert(source->PublishSnapshot(prompt, snapshot, false));
  (void)source->Commit(prompt, {}, prompt);
  // Idle streams are unavailable. Replacing the first slot would require
  // foreground preservation; the untouched second slot needs no copy.
  auto unrelated = prompt;
  unrelated.front() = 999;
  auto replacement = cache.Acquire(unrelated, {}, {}, true, 0, false);
  assert(!replacement->cache_hit());
  const auto metrics = replacement->Metrics();
  assert(metrics.spill.foreground_bytes == 0);
  assert(metrics.ledger.reserved_bytes[static_cast<unsigned>(
             cache::ResourceCategory::kBackingAssigned)] == 320);
}
void ImageAfterCachedTextPrefixCaptures() {
  // #584: a request that appends the first image after a cached text-only
  // prefix restores that prefix, whose input identity is empty, and must
  // still capture past the image.
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> text(40);
  std::iota(text.begin(), text.end(), 10);
  {
    auto root = cache.Acquire(text, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(root->state());
    runner->adapter->Append(state.slot, text, text);
    assert(root->TryReserveSnapshot(root->CaptureBytes(), text.size(), false,
                                    SnapshotPurpose::kContinuation, {}));
    std::shared_ptr<const TextRunnerSnapshot> snapshot = root->Capture();
    assert(root->PublishSnapshot(text, snapshot, false));
  }
  auto image = text;
  image.insert(image.end(), {50, 51, 52, 53});
  image.insert(image.end(), 16, 7);
  image.insert(image.end(), {60, 61});
  auto context = std::make_shared<TextPromptContext>();
  context->cache_identity = {1, 2, 3};
  context->cache_prefixes.push_back({44, {}});
  auto cold_runner = std::make_shared<Runner>();
  ComponentTextCache cold_cache(cold_runner, 1, 1 << 22, {});
  std::size_t inherited = text.size();
  for (const bool second_image : {false, true}) {
    if (second_image) {
      // A second image must preserve the first image's prefix identity too.
      image.insert(image.end(), {62, 63, 64, 65});
      context = std::make_shared<TextPromptContext>(*context);
      context->cache_prefixes.push_back({image.size(), {1, 2, 3}});
      context->cache_identity = {4, 5, 6};
      image.insert(image.end(), 16, 8);
      image.insert(image.end(), {70, 71});
    }
    auto lease = cache.Acquire(image, {}, context, true, 0, true);
    assert(lease->cached_tokens() == inherited);
    auto& state = dynamic_cast<State&>(lease->state());
    const auto suffix = std::span(image).subspan(inherited);
    runner->adapter->Append(state.slot, suffix, suffix);
    const auto hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), image.size(), false,
                                     SnapshotPurpose::kContinuation, {}));
    std::shared_ptr<const TextRunnerSnapshot> snapshot = lease->Capture();
    assert(lease->PublishSnapshot(image, snapshot, false));

    // Model the retry after discarding a cancelled assistant: the image prompt
    // is unchanged and only a short text suffix is appended. Keep the source
    // lease active so the retry must restore a checkpoint in the other slot.
    auto retry_prompt = image;
    retry_prompt.push_back(99);
    auto retry = cache.Acquire(retry_prompt, {}, context, true, 0, true);
    assert(retry->cached_tokens() == image.size());
    assert(retry->restored_snapshot_bytes() > 0);
    auto& restored = dynamic_cast<State&>(retry->state());
    assert(runner->adapter->RecurrentHash(restored.slot) == hash);
    // Executing the suffix reads target/draft rows too; private state alone
    // cannot establish that those payloads were restored correctly.
    const auto tail = std::span(retry_prompt).last(1);
    runner->adapter->Append(state.slot, tail, tail);
    runner->adapter->Append(restored.slot, tail, tail);
    auto cold = cold_cache.Acquire(retry_prompt, {}, context, true, 0, true);
    assert(cold->cached_tokens() == 0);
    auto& cold_state = dynamic_cast<State&>(cold->state());
    cold_runner->adapter->Append(cold_state.slot, retry_prompt, retry_prompt);
    const auto expected = cold_runner->adapter->RecurrentHash(cold_state.slot);
    assert(runner->adapter->RecurrentHash(state.slot) == expected);
    assert(runner->adapter->RecurrentHash(restored.slot) == expected);
    inherited = image.size();
  }

  // Image placeholders alone cannot establish a reusable multimodal prefix.
  // Changing the first image permits only the original text checkpoint.
  auto changed = std::make_shared<TextPromptContext>();
  changed->cache_identity = {9, 9, 9};
  changed->cache_prefixes.push_back({44, {}});
  auto replacement = cache.Acquire(image, {}, changed, true, 0, true);
  assert(replacement->cached_tokens() == text.size());
}
void CancelledPrefillKeepsCompletedSteps() {
  // A prefill cancelled between steps keeps the frontier it reached as a live
  // prefix (#486, #491), including on the component path Flash-Next uses.
  for (const bool partial : {false, true}) {
    auto runner = std::make_shared<Runner>();
    runner->partial = partial;
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto request = pool.Acquire(prompt, {}, [] { return false; }, {}, true, 0);
    assert(request.Prefill(16).consumed_tokens == 16);
    (void)request.Cancel();
    prompt.push_back(99);
    auto resend = pool.Acquire(prompt, {}, {}, {}, true, 0);
    assert(resend.cached_prompt_tokens() == (partial ? 16 : 0));
    resend.Invalidate();
  }
}
void SharedPeerReusesTheCurrentStableCapture() {
  for (const bool settled : {false, true}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 2);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto leader = pool.Acquire(prompt, {}, {}, {}, true, 33);
    const auto first = leader.Prefill(64);
    assert(first.consumed_tokens == 33 && !first.decode_ready);
    if (settled)
      leader.CapturePromptSnapshot();
    assert(leader.ShareCheckpoint(prompt.size()) == 33 &&
           "sharing inserted an extra prefill after the stable boundary");
    const auto last = leader.Prefill(64);
    assert(last.consumed_tokens == 7 && last.decode_ready);
    auto peer = pool.Acquire(prompt, {}, {}, {}, true, 33);
    assert(peer.cached_prompt_tokens() == 33);
    assert(peer.Prefill(64).consumed_tokens == 7);
    assert(Generate(peer) == Generate(leader));
    peer.Invalidate();
    leader.Invalidate();
  }
  // A peer with the same prompt shares at most the leader's last prefill
  // position. When that is the stable boundary, the leader stops right there:
  // the pending or just published copy is the shared checkpoint.
  for (const bool settled : {false, true}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 2);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto leader = pool.Acquire(prompt, {}, {}, {}, true, 39);
    const auto first = leader.Prefill(64);
    assert(first.consumed_tokens == 39 && !first.decode_ready);
    if (settled)
      leader.CapturePromptSnapshot();
    assert(leader.ShareCheckpoint(prompt.size()) == 39 &&
           "a peer arriving at the leader's stable boundary was not parked");
    assert(leader.Prefill(64).consumed_tokens == 1);
    auto peer = pool.Acquire(prompt, {}, {}, {}, true, 39);
    assert(peer.cached_prompt_tokens() == 39);
    assert(peer.Prefill(64).consumed_tokens == 1);
    assert(Generate(peer) == Generate(leader));
    peer.Invalidate();
    leader.Invalidate();
  }
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> prompt(40, 9);
  auto leader = pool.Acquire(prompt);
  assert(leader.Prefill(16).consumed_tokens == 16);
  assert(leader.ShareCheckpoint(16) == 0 &&
         "an uncaptured equal frontier cannot be offered to a peer");
  assert(leader.ShareCheckpoint(15) == 0);
  assert(leader.ShareCheckpoint(prompt.size()) > 16 &&
         "an uncaptured partial pass cannot be offered to a peer");
  leader.Invalidate();
}
void SchedulerSharesAnExactPendingFrontier() {
  // Hold the first real component capture after allocation, outside the cache
  // mutex. An unrelated queued request releases it only after the scheduler
  // has considered the earlier follower: no arrival sleeps or race lottery.
  class PendingRunner final : public Runner {
  public:
    mutable std::binary_semaphore entered{0}, release{0};
    mutable std::atomic<bool> first{true}, resumed{false}, timed_out{false};
    mutable std::mutex mutex;
    mutable std::thread::id capture_thread;
    void Resume() const {
      if (!resumed.exchange(true))
        release.release();
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->allocate =
          [this, allocate = resources->allocate](const auto& request) {
            // The inherited fake allocator counts captures non-atomically.
            // Independent asynchronous captures must serialize that call.
            const std::lock_guard lock(mutex);
            auto buffer = allocate(request);
            if (first.exchange(false)) {
              capture_thread = std::this_thread::get_id();
            }
            return buffer;
          };
      resources->stream = [this, stream = resources->stream] {
        bool hold;
        {
          const std::lock_guard lock(mutex);
          hold = capture_thread == std::this_thread::get_id();
          if (hold)
            capture_thread = {};
        }
        if (hold) {
          entered.release();
          if (!release.try_acquire_for(std::chrono::seconds(5))) {
            timed_out = true;
            throw std::runtime_error("pending component capture timed out");
          }
        }
        return stream();
      };
      return resources;
    }
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      if (prompt.front() == 999)
        Resume();
      return Runner::Prefill(state, prompt, offset, budget);
    }
  };
  auto runner = std::make_shared<PendingRunner>();
  auto pool = std::make_shared<TextRunnerPool>(runner, 3);
  TextGenerationScheduler scheduler(pool);
  struct ReleaseOnExit {
    PendingRunner& runner;
    ~ReleaseOnExit() { runner.Resume(); }
  } cleanup{*runner};
  std::vector<TextRunnerToken> prompt(600);
  std::iota(prompt.begin(), prompt.end(), 10);
  TextRequestMetadata metadata;
  metadata.cache_prefix_tokens = prompt.size() - 1;
  auto leader = scheduler.Submit(prompt, 1, 0.0f, {}, false, metadata);
  if (!runner->entered.try_acquire_for(std::chrono::seconds(5)))
    throw std::runtime_error("leader did not enter pending capture window");
  auto follower = scheduler.Submit(prompt, 1, 0.0f, {}, false, metadata);
  auto unrelated = scheduler.Submit({999, 1000}, 1, 0.0f);
  const auto peer = unrelated.Wait();
  const auto a = leader.Wait();
  const auto b = follower.Wait();
  assert(!runner->timed_out && runner->resumed);
  assert(peer.tokens.size() == 1 && a.tokens.size() == 1 &&
         a.tokens == b.tokens);
  assert(a.prefill_tokens == prompt.size());
  assert(b.shared_prefix_wait_ms > 0);
  assert(b.cached_prompt_tokens == prompt.size() - 1 && b.prefill_tokens == 1);
}
void PrefillLookaheadReachesTheNextFrontier() {
  class LookaheadRunner final : public Runner {
  public:
    struct Step {
      std::size_t frontier, end;
    };
    mutable std::vector<Step> steps;
    mutable const TextRunnerState* state{};
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      const auto ahead = state.prefill_lookahead();
      assert(ahead.empty() || ahead.data() == prompt.data() + prompt.size());
      steps.push_back({prompt.size(), prompt.size() + ahead.size()});
      this->state = &state;
      return Runner::Prefill(state, prompt, offset, budget);
    }
  };
  auto runner = std::make_shared<LookaheadRunner>();
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt(4000);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto request = pool.Acquire(prompt);
  while (!request.Prefill(1024).decode_ready) {
  }
  assert(runner->state && runner->state->prefill_lookahead().empty());
  // A grid checkpoint splits the prompt; every step looks ahead exactly to
  // the frontier after its own.
  assert(runner->steps.front().frontier < prompt.size());
  for (std::size_t i = 0; i < runner->steps.size(); ++i) {
    const auto [frontier, end] = runner->steps[i];
    auto next = prompt.size();
    for (std::size_t j = i + 1; j < runner->steps.size(); ++j)
      if (runner->steps[j].frontier > frontier) {
        next = runner->steps[j].frontier;
        break;
      }
    assert(end == (frontier == prompt.size() ? frontier : next));
  }
}
void BackingPreparationFollowsCaptures() {
  class PrepareRunner final : public Runner {
  public:
    mutable std::size_t unfinished{}, preparations{}, next_tokens{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      const auto allocate = resources->allocate;
      resources->allocate = [this, allocate](const auto& request) {
        auto buffer = allocate(request);
        ++unfinished;
        buffer.finish = [this, finish = buffer.finish] {
          auto payload = finish();
          --unfinished;
          return payload;
        };
        buffer.borrow = [this, borrow = buffer.borrow](auto& lease, auto rows) {
          auto payload = borrow(lease, rows);
          --unfinished;
          return payload;
        };
        return buffer;
      };
      resources->prepare_prefill = [this](std::size_t tokens) {
        assert(unfinished == 0 && tokens != 0);
        next_tokens = tokens;
        ++preparations;
      };
      return resources;
    }
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      assert(next_tokens == std::min(prompt.size() - offset, budget));
      next_tokens = 0;
      return Runner::Prefill(state, prompt, offset, budget);
    }
  };
  auto runner = std::make_shared<PrepareRunner>();
  TextRunnerPool pool(runner, 1);
  for (const std::size_t size : {256, 600, 601}) {
    const std::vector<TextRunnerToken> prompt(size, 8);
    auto request = pool.Acquire(prompt);
    const auto before = runner->preparations;
    while (!request.Prefill(512).decode_ready) {
    }
    assert(runner->preparations > before);
    (void)request.Commit();
    const auto prepared = runner->preparations;
    auto cached = pool.Acquire(prompt);
    assert(cached.cached_prompt_tokens() == prompt.size());
    assert(cached.Prefill(512).consumed_tokens == 0);
    assert(runner->preparations == prepared);
    cached.Invalidate();
  }
}

void PartialFrontierDoesNotServeAnEqualPrompt() {
  // A prefill cancelled between steps keeps a frontier with no next token. A
  // prompt ending exactly there must recompute its tail instead of decoding
  // from a placeholder, whether the frontier is live or restored.
  class FrontierRunner final : public Runner {
  public:
    mutable std::vector<std::size_t> ready;
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      const auto step = Runner::Prefill(state, prompt, offset, budget);
      if (step.decode_ready)
        ready.push_back(CheckpointPosition(state));
      return step;
    }
    bool LookupFullCheckpointReady() const override { return false; }
    bool HasDecodeFrontier(const TextRunnerState& state) const override {
      return std::ranges::find(ready, CheckpointPosition(state)) != ready.end();
    }
  };
  for (const bool replace_slot : {false, true}) {
    auto runner = std::make_shared<FrontierRunner>();
    runner->partial = true;
    TextRunnerPool pool(runner, 1);
    std::vector<TextRunnerToken> prompt(40);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto request = pool.Acquire(prompt, {}, [] { return false; }, {}, true, 0);
    assert(request.Prefill(16).consumed_tokens == 16);
    (void)request.Cancel();
    if (replace_slot) {
      auto unrelated = prompt;
      unrelated.front() = 999;
      auto other = pool.Acquire(unrelated, {}, {}, {}, true, 0);
      Prefill(other);
      (void)other.Commit();
    }
    const std::vector<TextRunnerToken> equal(prompt.begin(),
                                             prompt.begin() + 16);
    const auto lookup = pool.Lookup(equal);
    assert(lookup.matched_tokens < equal.size());
    auto resend = pool.Acquire(equal, {}, {}, {}, true, 0);
    assert(resend.cached_prompt_tokens() == lookup.matched_tokens);
    assert(resend.cached_prompt_tokens() < equal.size());
    const auto metrics = resend.component_cache_metrics();
    if (!replace_slot) {
      assert(metrics->reuse_reason == CacheReuseReason::kFrontierNotReady);
      assert(metrics->unready_frontiers > 0);
      assert(metrics->indexed_common_tokens == equal.size());
    } else {
      // The displaced partial frontier was never checkpointed: absence now
      // cannot diagnose historical eviction from the resident index alone.
      assert(metrics->reuse_reason == CacheReuseReason::kNoIndexedPrefix);
      assert(metrics->unready_frontiers == 0);
    }
    Prefill(resend);
    (void)Generate(resend);
    resend.Invalidate();
  }
}
void CompletedCancelledPromptGetsARetrySnapshot() {
  auto runner = std::make_shared<Runner>();
  runner->partial = true;
  runner->in_pass = true;
  TextRunnerPool pool(runner, 1);
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  {
    auto cancelled = pool.Acquire(prompt, {}, {}, {}, true, 35);
    assert(cancelled.Prefill(64).decode_ready);
    // Cancellation keeps the live prompt at 40 and an immutable fallback at
    // 35, without starting a complete-prompt capture.
    (void)cancelled.Cancel();
  }
  auto resumed = pool.Acquire(prompt, {}, {}, {}, true, 35);
  assert(resumed.cached_prompt_tokens() == prompt.size());
  const auto expected = Generate(resumed);
  assert(resumed.Commit().snapshot_bytes > 0);
  // Replace the execution slot so only immutable checkpoints can serve retry.
  auto unrelated = prompt;
  unrelated.front() = 999;
  auto replacement = pool.Acquire(unrelated, {}, {}, {}, true, 35);
  Prefill(replacement);
  (void)replacement.Commit();
  auto replay = pool.Acquire(prompt, {}, {}, {}, true, 35);
  assert(replay.cached_prompt_tokens() == prompt.size());
  assert(Generate(replay) == expected);
  replay.Invalidate();
}

void ColdAssignmentKeepsLiveFrontiers() {
  // No checkpoints are borrowed, so every slot has completed preservation.
  // Requests that cannot continue a frontier take the empty slot, then the
  // least recently used one, so the generated reply stays live (#562).
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  const auto turn = [&](const std::vector<TextRunnerToken>& tokens) {
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    const auto live = lease->cache_hit() && !lease->restored_snapshot_bytes();
    const auto cached = lease->cached_tokens();
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(tokens).subspan(cached);
    runner->adapter->Append(state.slot, tail, tail);
    (void)lease->Commit({}, {}, tokens);
    return live ? cached : 0;
  };
  std::vector<TextRunnerToken> conversation(12, 7);
  (void)turn(conversation);
  for (const TextRunnerToken other : {100, 101}) {
    (void)turn(std::vector<TextRunnerToken>(12, other));
    const auto frontier = conversation.size();
    conversation.push_back(8);
    assert(turn(conversation) == frontier);
  }
}
void ColdSideRequestsKeepPreservedDeepFrontier() {
  auto runner = std::make_shared<Runner>();
  runner->defer_idle = true;
  ComponentTextCache cache(runner, 2, 1 << 22, {});
  std::vector<TextRunnerToken> conversation(4096);
  std::iota(conversation.begin(), conversation.end(), 10);
  {
    auto source = cache.Acquire(conversation, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(source->state());
    const auto prefix = std::span(conversation).first(4088);
    runner->adapter->Append(state.slot, prefix, prefix);
    assert(source->TryReserveSnapshot(source->CaptureBytes(), prefix.size(),
                                      false, SnapshotPurpose::kContinuation,
                                      prefix));
    auto snapshot = source->Capture();
    assert(snapshot);
    assert(source->PublishSnapshot({prefix.begin(), prefix.end()},
                                   std::move(snapshot), false));
    // Deterministically finish preservation. A faster idle worker must not
    // remove the only protection of this live reply against short traffic.
    runner->adapter->GuardRows(state.slot, cache::testing::kTarget, 0,
                               prefix.size());
    runner->adapter->GuardRows(state.slot, cache::testing::kDraft, 0,
                               prefix.size());
    assert(source->Metrics().ledger.reserved_bytes[static_cast<unsigned>(
               cache::ResourceCategory::kBackingAssigned)] == 0);
    const auto reply = std::span(conversation).last(8);
    runner->adapter->Append(state.slot, reply, reply);
    (void)source->Commit({}, {}, conversation);
  }
  for (const TextRunnerToken code : {100, 101, 102}) {
    const std::vector<TextRunnerToken> short_prompt(12, code);
    auto side = cache.Acquire(short_prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(side->state());
    runner->adapter->Append(state.slot, short_prompt, short_prompt);
    // Readiness must not let short cancelled traffic evict the deep reply,
    // even though its older checkpoint has finished preservation.
    runner->adapter->SetExecutionComplete(state.slot, false);
    (void)side->Commit({}, {}, short_prompt);
  }
  auto resumed = cache.Acquire(conversation, {}, {}, true, 0, true);
  assert(resumed->cached_tokens() == conversation.size() &&
         !resumed->restored_snapshot_bytes() &&
         "short side requests evicted a preserved deep live frontier");
}
void ReuseAndBranches() {
  auto runner = std::make_shared<Runner>();
  TextRunnerPool pool(runner, 2);
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto initial = pool.Acquire(prompt);
  Prefill(initial);
  const auto token = Generate(initial);
  assert(initial.Commit().snapshot_bytes > 0);
  auto retry = pool.Acquire(prompt);
  assert(retry.component_cache_metrics()->reuse_reason ==
         CacheReuseReason::kRam);
  assert(retry.cached_prompt_tokens() == prompt.size());
  assert(retry.cache_restore_bytes() > 0);
  assert(Generate(retry) == token);
  auto peer = pool.Acquire(prompt);
  assert(peer.cached_prompt_tokens() == prompt.size());
  assert(Generate(peer) == token);
  (void)peer.Commit();
  (void)retry.Commit();
  runner->adapter->FailNextTransfer();
  auto failed_restore = pool.Acquire(prompt);
  assert(!failed_restore.cache_hit());
  Prefill(failed_restore);
  assert(Generate(failed_restore) == token);
  // The rejected checkpoint is quarantined. Re-publish only after the cold
  // execution has rebuilt and verified this boundary, not by retrying poison.
  assert(failed_restore.component_cache_metrics()->restore_failures == 1);
  assert(failed_restore.Commit().snapshot_bytes > 0);
  auto history = prompt;
  history.push_back(token);
  auto continued = pool.Acquire(history);
  assert(continued.cached_prompt_tokens() == history.size());
  assert(continued.cache_restore_bytes() == 0);
  assert(continued.component_cache_metrics()->reuse_reason ==
         CacheReuseReason::kLive);
  assert(continued.component_cache_metrics()->indexed_gap_tokens == 0);
  (void)continued.Cancel();
  history.back() += 1;
  auto edit = pool.Acquire(history);
  assert(edit.cached_prompt_tokens() == prompt.size());
  Prefill(edit);
  const auto edited = Generate(edit);
  (void)edit.Commit();
  auto cold = pool.Acquire(history, sampling::SamplingConfig{}, {}, {}, false);
  assert(!cold.cache_hit());
  Prefill(cold);
  assert(Generate(cold) == edited);
  cold.Invalidate();
}
void EditBoundariesSurviveActualBackingPressure() {
  auto runner = std::make_shared<Runner>();
  runner->private_blocks = 4;
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  const auto capture = [](auto& lease, const auto& tokens,
                          SnapshotPurpose purpose) {
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), tokens.size(),
                                     false, purpose, {}));
    assert(lease->PublishSnapshot(tokens, lease->Capture(), false));
  };
  for (TextRunnerToken family = 100; family < 104; ++family) {
    std::vector<TextRunnerToken> tokens(12, family);
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, tokens, tokens);
    capture(lease, tokens, SnapshotPurpose::kContinuation);
    (void)lease->Commit({}, {}, tokens);
  }
  std::vector<TextRunnerToken> prompt(12);
  std::iota(prompt.begin(), prompt.end(), 1);
  {
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    for (std::size_t end : {4U, 8U, 12U}) {
      const auto piece = std::span(prompt).subspan(end - 4, 4);
      runner->adapter->Append(state.slot, piece, piece);
      std::vector<TextRunnerToken> prefix(prompt.begin(), prompt.begin() + end);
      capture(lease, prefix,
              end == prompt.size() ? SnapshotPurpose::kContinuation
                                   : SnapshotPurpose::kHistory);
    }
    (void)lease->Commit({}, {}, prompt);
  }
  for (std::size_t edit : {5U, 9U}) {
    auto branch = prompt;
    branch[edit] += 1000;
    auto lease = cache.Acquire(branch, {}, {}, true, 0, true);
    const auto cached = edit < 8 ? 4U : 8U;
    assert(lease->cached_tokens() == cached);
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(branch).subspan(cached);
    runner->adapter->Append(state.slot, tail, tail);
    const auto restored_hash = runner->adapter->RecurrentHash(state.slot);
    auto cold_runner = std::make_shared<Runner>();
    ComponentTextCache cold(cold_runner, 1, 1 << 22, {});
    auto control = cold.Acquire(branch, {}, {}, false, 0, true);
    auto& control_state = dynamic_cast<State&>(control->state());
    cold_runner->adapter->Append(control_state.slot, branch, branch);
    assert(restored_hash ==
           cold_runner->adapter->RecurrentHash(control_state.slot));
  }
}
void ReusedConversationOutlivesOneUseRequests() {
  // Backing holds three private states. A continued conversation keeps its
  // latest boundary while newer one-use requests replace each other.
  auto runner = std::make_shared<Runner>();
  runner->private_blocks = 3;
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  const auto turn = [&](const std::vector<TextRunnerToken>& tokens) {
    auto lease = cache.Acquire(tokens, {}, {}, true, 0, true);
    const auto cached = lease->cached_tokens();
    auto& state = dynamic_cast<State&>(lease->state());
    const auto tail = std::span(tokens).subspan(cached);
    runner->adapter->Append(state.slot, tail, tail);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), tokens.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     {}));
    assert(lease->PublishSnapshot(tokens, lease->Capture(), false));
    (void)lease->Commit({}, {}, tokens);
    return cached;
  };
  std::vector<TextRunnerToken> conversation(12, 7);
  assert(turn(conversation) == 0);
  conversation.resize(16, 8);
  assert(turn(conversation) == 12);
  for (TextRunnerToken other = 100; other < 103; ++other)
    assert(turn(std::vector<TextRunnerToken>(12, other)) == 0);
  conversation.push_back(9);
  assert(turn(conversation) == 16);
}
void RestartAndCorruption() {
  const TemporaryDirectory temp("gufo-component-serving-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  TextRunnerToken token;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = pool.Acquire(prompt);
    Prefill(first);
    token = Generate(first);
    (void)first.Commit();
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    const auto lookup = pool.Lookup(prompt);
    assert(lookup.tier == "disk" && lookup.matched_tokens == prompt.size());
    const auto captures = runner->captures;
    for (int i = 0; i < 20; ++i)
      assert(pool.Lookup(prompt).matched_tokens == prompt.size());
    assert(runner->captures == captures);
    auto retry = pool.Acquire(prompt);
    assert(retry.cached_prompt_tokens() == lookup.matched_tokens);
    assert(retry.cache_disk_hit());
    assert(retry.cached_prompt_tokens() == prompt.size());
    assert(Generate(retry) == token);
    (void)retry.Commit();
  }
  for (const bool enabled : {true, false}) {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache exporter(runner, 1, 1 << 22, disk, false, 0, {},
                                enabled);
    assert(exporter.Lookup(prompt, nullptr, true, 0)->tier == "disk");
    auto blob = exporter.Export(prompt, nullptr, true, 0, "disk-transfer-v20");
    std::vector<std::uint8_t> wire;
    blob->Write([&](auto bytes) {
      wire.insert(wire.end(), bytes.begin(), bytes.end());
      return true;
    });
    auto destination_runner = std::make_shared<Runner>();
    ComponentTextCache destination(destination_runner, 1, 1 << 22, {});
    std::size_t offset = 0;
    (void)destination.Import(
        [&](std::span<std::uint8_t> out) {
          const auto n = std::min(out.size(), wire.size() - offset);
          std::memcpy(out.data(), wire.data() + offset, n);
          offset += n;
          return n;
        },
        wire.size(), "disk-transfer-v20");
    assert(destination.Lookup(prompt, nullptr, true, 0)->tier == "ram");
    auto restored = destination.Acquire(prompt, {}, {}, true, 0, true);
    assert(restored->cached_tokens() == prompt.size());
  }
  for (const auto& kind : {"chunks", "private"}) {
    for (const auto& file :
         std::filesystem::directory_iterator(directory / "v2" / kind)) {
      std::fstream bytes(file.path(),
                         std::ios::in | std::ios::out | std::ios::binary);
      char byte;
      bytes.read(&byte, 1);
      byte ^= 0x5a;
      bytes.seekp(0);
      bytes.write(&byte, 1);
    }
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto corrupted = pool.Acquire(prompt);
    assert(!corrupted.cache_hit());
    Prefill(corrupted);
    assert(Generate(corrupted) == token);
    corrupted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void FailedCaptureKeepsPublishedBoundary() {
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(42);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(lease->state());
  runner->adapter->Append(state.slot, std::span(prompt).first(40),
                          std::span(prompt).first(40));
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 40, false,
                                   SnapshotPurpose::kContinuation, {}));
  std::shared_ptr<const TextRunnerSnapshot> snapshot = lease->Capture();
  assert(lease->PublishSnapshot({prompt.begin(), prompt.begin() + 40}, snapshot,
                                false));
  snapshot.reset();
  runner->adapter->Append(state.slot, std::span(prompt).last(2),
                          std::span(prompt).last(2));
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 42, false,
                                   SnapshotPurpose::kContinuation, {}));
  runner->adapter->FailNextTransfer();
  bool failed = false;
  try {
    (void)lease->Capture();
  } catch (const std::runtime_error&) {
    failed = true;
  }
  assert(failed);
  lease->Invalidate();
  auto restored =
      cache.Acquire(std::span(prompt).first(40), {}, {}, true, 0, true);
  assert(restored->cached_tokens() == 40);
  auto& restored_state = dynamic_cast<State&>(restored->state());
  assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
}
void BackingPressureKeepsSharedPrefix(bool capacity_check = false) {
  struct PressureRunner final : Runner {
    bool capacity_check{};
    mutable std::size_t inherited_boundary{}, recopied_prefixes{},
        rejected_plans{}, failed_allocations{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->allocate = [this, allocate = std::move(resources->allocate)](
                                const cache::PayloadRequest& request) {
        if (request.category == cache::ResourceCategory::kBackingAssigned &&
            request.first < inherited_boundary)
          ++recopied_prefixes;
        try {
          return allocate(request);
        } catch (const cache::ResourceExhausted&) {
          ++failed_allocations;
          throw;
        }
      };
      if (capacity_check)
        resources->can_capture = [this, &ledger](auto payloads) {
          const bool needs_private =
              std::ranges::any_of(payloads, [](const auto& payload) {
                return payload.category ==
                       cache::ResourceCategory::kPrivateState;
              });
          const bool fits = !needs_private ||
                            !ledger.Snapshot().bytes[static_cast<std::size_t>(
                                cache::ResourceCategory::kPrivateState)];
          rejected_plans += !fits;
          return fits;
        };
      return resources;
    }
  };
  auto runner = std::make_shared<PressureRunner>();
  runner->capacity_check = capacity_check;
  // The next private frontier cannot fit until the old checkpoint is evicted.
  runner->private_blocks = 1;
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  std::vector<TextRunnerToken> prompt(80);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(lease->state());
  runner->adapter->Append(state.slot, std::span(prompt).first(64),
                          std::span(prompt).first(64));
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 64, false,
                                   SnapshotPurpose::kContinuation, {}));
  assert(lease->PublishSnapshot({prompt.begin(), prompt.begin() + 64},
                                lease->Capture(), false));
  runner->inherited_boundary = 64;
  runner->adapter->Append(state.slot, std::span(prompt).subspan(64),
                          std::span(prompt).subspan(64));
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  assert(lease->PublishSnapshot(prompt, lease->Capture(), false));
  assert(!runner->recopied_prefixes);
  if (capacity_check)
    assert(runner->rejected_plans && !runner->failed_allocations);
  else
    assert(runner->failed_allocations);
  assert(!lease->HasSnapshotFor(std::span(prompt).first(64)));
  lease->Invalidate();
  auto restored = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(restored->cached_tokens() == prompt.size());
  auto& restored_state = dynamic_cast<State&>(restored->state());
  assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
}
void FragmentedBackingKeepsCompletePrompt() {
  struct FragmentedRunner final : Runner {
    mutable std::size_t failed_allocations{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      struct Pool {
        std::array<cache::ResourceCharge, 6> pages;
        std::array<cache::ResourceCharge, 2> occupied;
        std::shared_ptr<std::array<std::byte, 96>> data =
            std::make_shared<std::array<std::byte, 96>>();
      };
      auto pool = std::make_shared<Pool>();
      for (auto& page : pool->pages)
        page =
            ledger.Reserve(cache::ResourceCategory::kBackingFree, 16).Convert();
      for (std::size_t i = 0; i < pool->occupied.size(); ++i)
        pool->occupied[i] =
            pool->pages[2 + 2 * i]
                .ReserveBacking(cache::ResourceCategory::kPrivateState)
                .Convert();
      // Free runs have sizes 32, 16, 16. Two 16-byte row captures followed
      // by the 32-byte state cannot fit with first-fit allocation, despite
      // sufficient total space. The existing occupied pages cannot be evicted.
      resources->owner = pool;
      resources->allocate = [this, pool](const cache::PayloadRequest& request) {
        const auto count = (request.bytes + 15) / 16;
        const auto first =
            cache::ResourceCharge::FindFreeBacking(pool->pages, count);
        if (first == pool->pages.size()) {
          ++failed_allocations;
          throw cache::ResourceExhausted();
        }
        auto reservation = std::make_shared<cache::ResourceReservation>(
            cache::ResourceCharge::ReserveBacking(
                std::span(pool->pages).subspan(first, count),
                request.category));
        auto bytes = std::span(*pool->data).subspan(first * 16, count * 16);
        auto owner = std::shared_ptr<const void>(pool->data, bytes.data());
        return ComponentCaptureBuffer{bytes,
                                      [reservation, owner] {
                                        return cache::Payload::Committed(
                                            reservation->Convert(), owner);
                                      },
                                      pool->data.get(),
                                      {}};
      };
      return resources;
    }
  };
  auto runner = std::make_shared<FragmentedRunner>();
  ComponentTextCache cache(runner, 1, 1 << 22, {});
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13};
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  auto& state = dynamic_cast<State&>(lease->state());
  runner->adapter->Append(state.slot, prompt, prompt);
  const auto hash = runner->adapter->RecurrentHash(state.slot);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  std::unique_ptr<TextRunnerSnapshot> snapshot;
  try {
    snapshot = lease->Capture();
  } catch (const cache::ResourceExhausted&) {
  }
  assert(snapshot && "fragmented backing rejected a complete prompt");
  assert(lease->PublishSnapshot(prompt, std::move(snapshot), false));
  assert(!runner->failed_allocations);
  lease->Invalidate();
  auto restored = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(restored->cached_tokens() == prompt.size());
  auto& restored_state = dynamic_cast<State&>(restored->state());
  assert(runner->adapter->RecurrentHash(restored_state.slot) == hash);
  const TextRunnerToken next = 99;
  runner->adapter->Append(restored_state.slot, {&next, 1}, {&next, 1});
}
// Invocation-order fixture only: real q8 semantic faults use the real QFN
// runner test. This sentinel proves startup calls the existing model-owned
// finish before exposing a matching durable checkpoint, not on later restore.
void RotatedDiskAdmissionFinishesBeforePublication() {
  class AdmissionRunner final : public Runner {
  public:
    bool rotated{true};
    bool refuse{};
    mutable std::atomic<unsigned> checked{};
    TextRunnerDescriptor Descriptor() const override {
      auto d = Runner::Descriptor();
      if (rotated) {
        d.state_abi = "qwen38-flash-next-rocm-session-v1";
        d.persistence = TextRunnerPersistenceDescriptor{
            {3, 2, 3}, models::qwen38_flash_next::kKvRotatedNumericsEpoch};
      }
      return d;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto r = Runner::CreateComponentCacheResources(ledger, ram);
      auto allocate = std::move(r->allocate);
      r->allocate = [this,
                     allocate = std::move(allocate)](const auto& request) {
        auto b = allocate(request);
        if (request.category != cache::ResourceCategory::kPrivateState) {
          auto finish = std::move(b.finish);
          b.finish = [this, finish = std::move(finish)]() mutable {
            ++checked;
            if (refuse)
              throw std::invalid_argument("fixture packed semantic rejection");
            return finish();
          };
        }
        return b;
      };
      return r;
    }
  };
  const TemporaryDirectory temp("gufo-rotated-disk-admission-order");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path(),
                                        .capacity_bytes = 1 << 20};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  const auto publish = [&] {
    auto runner = std::make_shared<AdmissionRunner>();
    TextRunnerPool pool(runner, 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  };
  publish();
  {
    auto runner = std::make_shared<AdmissionRunner>();
    runner->refuse = true;
    TextRunnerPool pool(runner, 1, disk);
    // No restore or execution before checking that startup invoked the hook.
    assert(runner->checked.load() > 0);
    auto request = pool.Acquire(prompt);
    assert(!request.cache_hit());
    request.Invalidate();
  }
  publish();  // Valid replacement in the same root after malformed retirement.
  {
    auto runner = std::make_shared<AdmissionRunner>();
    TextRunnerPool pool(runner, 1, disk);
    assert(runner->checked.load() > 0);
    auto request = pool.Acquire(prompt);
    assert(request.cache_disk_hit());
    request.Invalidate();
  }
  {
    auto runner = std::make_shared<AdmissionRunner>();
    runner->rotated = false;
    runner->refuse = true;
    TextRunnerPool pool(runner, 1, disk);
    assert(runner->checked.load() == 0);  // Legacy startup fast path unchanged.
  }
}

// CPU admission fixture: real QFN layout/codec, finish wrapper, CRC-valid
// disk store and serving publication. No model state is
// executed/restored: private bytes are placeholders, not a native checkpoint.
namespace packed_ingress_cpu {
namespace qfn = models::qwen38_flash_next;
constexpr std::uint32_t kContext = 256, kElements = 512;
constexpr std::size_t kRam = 4ULL << 20;
constexpr auto kPlain = qfn::KvCacheType::kQ8_0;
qfn::Config Geometry() {
  qfn::Config c;
  c.num_layers = 48;
  c.full_attention_interval = 4;
  c.context_length = kContext;
  c.hidden_size = 8;
  c.hc_count = 1;
  c.vocab_size = 32;
  c.num_kv_heads = 2;
  c.num_heads = 24;
  c.head_dim = 256;
  c.rotary_dim = 64;
  c.indexer_head_dim = 8;
  c.indexer_top_k = 16;
  c.compress_ratio = 4;
  c.ssm_conv_kernel = 2;
  c.ssm_head_dim = 4;
  c.ssm_num_k_heads = c.ssm_num_v_heads = 1;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_conv_kernel = 4;
  return c;
}
bool Packed(qfn::ContinuationPart part) {
  return part == qfn::ContinuationPart::kKey ||
         part == qfn::ContinuationPart::kValue ||
         part == qfn::ContinuationPart::kDraftKey ||
         part == qfn::ContinuationPart::kDraftValue;
}
cache::CompatibilityDigest Digest(std::span<const std::uint8_t> bytes) {
  crypto::Sha256Hasher hash;
  hash.Update(bytes);
  return hash.Finish();
}
cache::Identity Identity(qfn::KvCacheType type) {
  const auto text =
      "cpu-admission-only;" + qfn::KvCacheCompatibilityIdentity(type);
  return {text.begin(), text.end()};
}
// Startup admission uses only registered descriptors. A slot/transfer call is
// a test error, not a simulated native restore.
class LayoutAdapter final : public cache::Adapter {
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  [[noreturn]] static void Unexpected() {
    throw std::logic_error(
        "CPU ingress fixture must not execute/restore a slot");
  }

public:
  explicit LayoutAdapter(qfn::KvCacheType type) : identity_(Identity(type)) {
    const qfn::ContinuationLayout layout(Geometry(), true, kContext, 1, type);
    for (const auto& c : layout.Components())
      descriptors_.push_back(c.descriptor);
  }
  cache::Capabilities GetCapabilities() const override { return {true, true}; }
  std::span<const cache::ComponentDescriptor> Components() const override {
    return descriptors_;
  }
  cache::Identity CompatibilityIdentity() const override { return identity_; }
  std::unique_ptr<cache::Slot> CreateSlot(cache::MutationGuard&) override {
    Unexpected();
  }
  std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override {
    Unexpected();
  }
  std::vector<cache::Rows> PlanPrefill(const cache::Slot&, cache::Rows,
                                       cache::Rows) const override {
    Unexpected();
  }
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override {
    Unexpected();
  }
  cache::Completion CapturePrivate(const cache::Slot&, cache::ComponentId,
                                   std::span<std::byte>,
                                   cache::Stream&) override {
    Unexpected();
  }
  cache::Completion CopyRowsOut(const cache::Slot&, cache::ComponentId,
                                cache::Rows, cache::Rows, std::span<std::byte>,
                                cache::Stream&) override {
    Unexpected();
  }
  cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId, cache::Rows,
                               cache::Rows, std::span<const std::byte>,
                               cache::Stream&) override {
    Unexpected();
  }
  cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                std::span<const std::byte>,
                                cache::Stream&) override {
    Unexpected();
  }
  bool Validate(cache::Slot&,
                std::span<const cache::ComponentPosition>) override {
    Unexpected();
  }
  bool Invalidate(cache::Slot&) noexcept override { std::terminate(); }
};
class AdmissionRunner final : public Runner {
public:
  qfn::KvCacheType type;
  bool fingerprint{true}, foreign_abi{};
  bool fail_resource{}, fail_operational{};
  mutable cache::ResourceLedger* ledger{};
  mutable std::size_t allocations{}, finishes{};
  explicit AdmissionRunner(qfn::KvCacheType type) : type(type) {}
  std::vector<TextRunnerToken> TokenizeControl(
      std::string_view) const override {
    return {10, 11};
  }
  TextRunnerDescriptor Descriptor() const override {
    auto d = Runner::Descriptor();
    d.max_context = kContext;
    d.state_abi = foreign_abi ? "foreign-session-v1"
                              : "qwen38-flash-next-rocm-session-v1";
    if (fingerprint) {
      d.persistence = TextRunnerPersistenceDescriptor{
          Identity(type), qfn::ExpectedKvPayloadVersion(type)};
    }
    return d;
  }
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    this->ledger = &ledger;
    auto r = std::make_unique<ComponentCacheResources>();
    r->adapter = std::make_unique<LayoutAdapter>(type);
    r->owner = std::make_shared<int>(1);
    r->stream = [] { return std::make_unique<FakeStream>(); };
    r->staging = [](std::size_t size) {
      auto data = std::make_shared<std::vector<std::uint8_t>>(size);
      return cache::StagingAllocation{data, *data};
    };
    r->allocate = [this, &ledger](const cache::PayloadRequest& request) {
      ++allocations;
      if (fail_resource)
        throw cache::ResourceExhausted{};
      if (fail_operational)
        throw std::runtime_error("admission allocation failed");
      assert(request.bytes && request.bytes < kRam);
      struct Backing {
        cache::ResourceCharge pool;
        std::vector<std::byte> bytes;
      };
      auto backing = std::make_shared<Backing>(Backing{
          ledger.Reserve(cache::ResourceCategory::kBackingFree, request.bytes)
              .Convert(),
          std::vector<std::byte>(request.bytes)});
      auto reservation = std::make_shared<cache::ResourceReservation>(
          backing->pool.ReserveBacking(request.category));
      auto owner = std::shared_ptr<const void>(backing, backing->bytes.data());
      return ComponentCaptureBuffer{backing->bytes,
                                    [this, reservation, owner] {
                                      ++finishes;
                                      return cache::Payload::Committed(
                                          reservation->Convert(), owner);
                                    },
                                    nullptr,
                                    {}};
    };
    detail::InstallQfnPackedKvIngressValidation(*r, Geometry(), true, kContext,
                                                type);
    return r;
  }
};
struct Artifact {
  struct Header {
    cache::Identity identity;
    cache::DiskManifest manifest;
  } header;
  std::vector<std::vector<std::uint8_t>> images;
  std::array<std::size_t, 4> kv_images{};
};
Artifact Canonical(qfn::KvCacheType type) {
  Artifact a;
  auto& h = a.header;
  h.identity = Identity(type);
  h.manifest.checkpoint = {1000};
  h.manifest.lineage = {1000};
  h.manifest.tokens = {10, 11};
  h.manifest.compatibility = Digest(h.identity);
  const qfn::ContinuationLayout layout(Geometry(), true, kContext, 1, type);
  const auto positions = layout.Positions(2, 2, 0, 0);
  std::array<bool, 4> found{};
  for (std::size_t i = 0; i < layout.Components().size(); ++i) {
    const auto& c = layout.Components()[i];
    const auto& d = c.descriptor;
    cache::DiskComponent component{.descriptor = d, .position = positions[i]};
    const auto bytes =
        d.row_bytes ? positions[i].valid_rows * d.row_bytes : d.state_bytes;
    if (bytes) {
      const auto image = a.images.size();
      a.images.emplace_back(bytes);
      if (Packed(c.part)) {
        std::vector<float> values(kElements);
        values[0] = 127;
        values[1] = 0.5F;
        values[2] = -0.5F;
        values[32] = 1e-7F;
        for (std::size_t off = 0; off < bytes; off += d.row_bytes)
          qfn::EncodeKvRow(values, type,
                           std::as_writable_bytes(std::span(a.images.back()))
                               .subspan(off, d.row_bytes));
        const auto part = c.part == qfn::ContinuationPart::kKey        ? 0U
                          : c.part == qfn::ContinuationPart::kValue    ? 1U
                          : c.part == qfn::ContinuationPart::kDraftKey ? 2U
                                                                       : 3U;
        if (!found[part]) {
          a.kv_images[part] = image;
          found[part] = true;
        }
      }
      cache::DiskFileId id{};
      id[0] = static_cast<std::uint8_t>(image + 1);
      const cache::DiskPayload payload{id, bytes,
                                       cache::DiskChecksum(a.images.back())};
      if (d.row_bytes)
        component.tail = payload;
      else
        component.private_state = payload;
    }
    h.manifest.components.push_back(std::move(component));
  }
  assert(std::ranges::all_of(found, [](bool b) { return b; }));
  return a;
}
void Poison(Artifact& a, unsigned part, unsigned fault) {
  auto& image = a.images[a.kv_images.at(part)];
  auto block = std::span(image).last(sizeof(qfn::KvBlockQ8_0));
  if (fault == 0) {
    block[offsetof(qfn::KvBlockQ8_0, values)] = 0x80;
  } else {
    const auto scale =
        std::array<std::uint16_t, 5>{0xbc00, 0x8000, 0x7c00, 0x7e00, 0x7bff}.at(
            fault - 1);
    block[0] = scale & 255;
    block[1] = scale >> 8;
    std::fill(block.begin() + 2, block.end(), 0);
    if (fault == 5)
      block[2] = 127;
  }
  // Both real codecs accept the metadata; the bytes' CRC is correct even
  // though their last block (in row two) is semantically inadmissible.
  std::size_t next = 0;
  for (auto& c : a.header.manifest.components) {
    if (c.tail)
      c.tail->checksum = cache::DiskChecksum(a.images[next++]);
    if (c.private_state)
      c.private_state->checksum = cache::DiskChecksum(a.images[next++]);
  }
  (void)cache::DecodeManifest(cache::EncodeManifest(a.header.manifest));
}
void Publish(const std::filesystem::path& root, const Artifact& a,
             std::uint8_t manifest_id = 255) {
  cache::ResourceLedger ledger({kRam, kRam, 0, 0});
  cache::DiskStore disk(ledger, root, kRam);
  std::vector<cache::DiskWriteBuffer> buffers;
  std::size_t next = 0;
  for (const auto& c : a.header.manifest.components) {
    if (c.tail)
      buffers.push_back({true, c.tail->file, a.images[next++]});
    if (c.private_state)
      buffers.push_back({true, c.private_state->file, a.images[next++]});
  }
  assert(next == a.images.size());
  cache::DiskFileId manifest{};
  manifest[0] = manifest_id;
  disk.Publish(manifest, a.header.manifest, buffers);
}
std::size_t Admit(const std::filesystem::path& root, const Artifact& artifact,
                  const std::shared_ptr<AdmissionRunner>& runner) {
  ComponentTextCache restarted(
      runner, 0, kRam,
      TextRunnerDiskCacheOptions{.directory = root, .capacity_bytes = kRam});
  // Query the admitted index before any slot restore can hide a missing
  // startup check.
  return restarted.CachedPrefixTokens(artifact.header.manifest.tokens, nullptr);
}
void RotatedWithoutFingerprintDisk() {
  const auto type = qfn::KvCacheType::kQ8_0Rotated;
  auto bad = Canonical(type);
  Poison(bad, 0, 0);
  const TemporaryDirectory temp("gufo-rotated-process-local-disk");
  Publish(temp.path(), bad);
  auto runner = std::make_shared<AdmissionRunner>(type);
  runner->fingerprint = false;
  assert(Admit(temp.path(), bad, runner) == 0);
  assert(runner->allocations > 0);
}
void CorruptedDisk() {
  for (auto type : {kPlain, qfn::KvCacheType::kQ8_0Rotated}) {
    for (bool fingerprint : {true, false}) {
      const TemporaryDirectory temp("gufo-packed-corrupt-disk");
      const auto bad = Canonical(type);
      auto good = bad;
      good.header.manifest.checkpoint = {3000};
      good.header.manifest.lineage = {3000};
      good.header.manifest.tokens = {30, 31};
      for (auto& c : good.header.manifest.components) {
        for (auto& p : c.chunks)
          p.file[1] = 1;
        for (auto* p : {&c.tail, &c.private_state})
          if (*p)
            (*p)->file[1] = 1;
      }
      Publish(temp.path(), bad);
      Publish(temp.path(), good, 253);
      cache::DiskFileId file{};
      file[0] = static_cast<std::uint8_t>(bad.kv_images[0] + 1);
      const auto path =
          temp.path() / "v2" / "private" / cache::DiskFileName(file);
      const auto bytes = std::filesystem::file_size(path);
      {
        std::fstream payload(path,
                             std::ios::in | std::ios::out | std::ios::binary);
        payload.seekg(offsetof(qfn::KvBlockQ8_0, values));
        char value{};
        assert(payload.read(&value, 1));
        value ^= 1;
        payload.seekp(offsetof(qfn::KvBlockQ8_0, values));
        assert(payload.write(&value, 1));
      }
      assert(std::filesystem::file_size(path) == bytes);
      {
        // Metadata recovery accepts the unchanged manifests and payload sizes.
        // Only the serving startup's payload read can detect this corruption.
        cache::ResourceLedger ledger({kRam, kRam, 0, 0});
        cache::DiskStore disk(ledger, temp.path(), kRam);
        assert(disk.Entries().size() == 2);
        for (const auto* artifact :
             std::array<const Artifact*, 2>{&bad, &good}) {
          auto description =
              disk.Describe(artifact->header.manifest.checkpoint);
          assert(description);
          assert(cache::EncodeManifest(description->Manifest()) ==
                 cache::EncodeManifest(artifact->header.manifest));
        }
      }
      for (unsigned restart = 0; restart < 2; ++restart) {
        auto runner = std::make_shared<AdmissionRunner>(type);
        runner->fingerprint = fingerprint;
        ComponentTextCache restarted(
            runner, 0, kRam,
            TextRunnerDiskCacheOptions{.directory = temp.path(),
                                       .capacity_bytes = kRam});
        assert(restarted.CachedPrefixTokens(bad.header.manifest.tokens,
                                            nullptr) == 0);
        assert(restarted.CachedPrefixTokens(good.header.manifest.tokens,
                                            nullptr) == 2);
        assert(runner->finishes > 0);
      }
      cache::ResourceLedger ledger({kRam, kRam, 0, 0});
      cache::DiskStore disk(ledger, temp.path(), kRam);
      assert(disk.Entries().size() == 1);
      assert(disk.Describe(good.header.manifest.checkpoint));
    }
  }
}
void AdmissionFailuresPreserveDisk() {
  const auto valid = Canonical(kPlain);
  for (bool resource : {true, false}) {
    const TemporaryDirectory temp("gufo-packed-admission-failure");
    Publish(temp.path(), valid);
    auto runner = std::make_shared<AdmissionRunner>(kPlain);
    runner->fail_resource = resource;
    runner->fail_operational = !resource;
    bool failed{};
    try {
      (void)Admit(temp.path(), valid, runner);
    } catch (const cache::ResourceExhausted&) {
      assert(resource);
      failed = true;
    } catch (const std::runtime_error& error) {
      assert(!resource);
      assert(std::string_view(error.what()) == "admission allocation failed");
      failed = true;
    }
    assert(failed);
    runner = std::make_shared<AdmissionRunner>(kPlain);
    assert(Admit(temp.path(), valid, runner) == 2);
  }
}
void MalformedDisk() {
  const std::array types{kPlain, qfn::KvCacheType::kQ8_0Rotated};
  for (auto type : types) {
    for (bool fingerprint : {true, false}) {
      const auto valid = Canonical(type);
      for (unsigned part = 0; part < 4; ++part) {
        for (unsigned fault = 0; fault < 6; ++fault) {
          auto bad = valid;
          Poison(bad, part, fault);
          const TemporaryDirectory temp("gufo-packed-semantic-disk");
          Publish(temp.path(), bad);  // Valid manifest and CRC, invalid codec.
          auto runner = std::make_shared<AdmissionRunner>(type);
          runner->fingerprint = fingerprint;
          assert(Admit(temp.path(), bad, runner) == 0);
          assert(runner->allocations > 0);
          // A malformed publication was retired, not deferred until restore.
          {
            cache::ResourceLedger ledger({kRam, kRam, 0, 0});
            cache::DiskStore disk(ledger, temp.path(), kRam);
            assert(disk.Entries().empty());
          }
          Publish(temp.path(), valid);  // Same-root repair/retry.
          runner = std::make_shared<AdmissionRunner>(type);
          runner->fingerprint = fingerprint;
          assert(Admit(temp.path(), valid, runner) == 2);
          assert(runner->finishes > 0);
        }
      }
    }
    const auto valid = Canonical(type);
    for (std::size_t i = 0; i < valid.header.manifest.components.size(); ++i) {
      if (valid.header.manifest.components[i].descriptor.kind !=
          cache::ComponentKind::kAppendRows)
        continue;
      auto mismatch = valid;
      ++mismatch.header.manifest.components[i].descriptor.layout_version;
      const TemporaryDirectory temp("gufo-packed-layout-mismatch");
      Publish(temp.path(), mismatch);
      auto runner = std::make_shared<AdmissionRunner>(type);
      runner->fingerprint = false;
      assert(Admit(temp.path(), mismatch, runner) == 0);
    }
    for (bool fingerprint : {true, false}) {
      const TemporaryDirectory temp("gufo-packed-foreign-abi");
      auto bad = valid;
      Poison(bad, 0, 0);
      Publish(temp.path(), bad);
      auto runner = std::make_shared<AdmissionRunner>(type);
      runner->foreign_abi = true;
      runner->fingerprint = fingerprint;
      assert(Admit(temp.path(), bad, runner) == 2);
      assert(runner->allocations == 0);
    }
  }
  for (bool fingerprint : {true, false}) {
    const TemporaryDirectory temp("gufo-f16-no-packed-admission");
    auto f16 = Canonical(qfn::KvCacheType::kF16);
    Poison(f16, 0, 3);  // Never restore these nonfinite F16 controls.
    Publish(temp.path(), f16);
    auto runner = std::make_shared<AdmissionRunner>(qfn::KvCacheType::kF16);
    runner->fingerprint = fingerprint;
    assert(Admit(temp.path(), f16, runner) == 2);
    assert(runner->allocations == 0);
  }
}
}  // namespace packed_ingress_cpu

void DiskBudgetEvictsAnUnpinnedConversation() {
  const TemporaryDirectory temp("gufo-component-budget-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1024,
                                        .staging_capacity_bytes = 7};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto retained = pool.Acquire(second);
    assert(retained.cache_disk_hit());
    retained.Invalidate();
    auto evicted = pool.Acquire(first);
    assert(!evicted.cache_hit());
    evicted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
std::uintmax_t ManagedBytes(const std::filesystem::path& directory) {
  std::uintmax_t bytes{};
  for (const auto& file :
       std::filesystem::recursive_directory_iterator(directory / "v2"))
    if (file.is_regular_file())
      bytes += file.file_size();
  return bytes;
}
void UnpersistedRequestStaysOffDisk() {
  const TemporaryDirectory temp("gufo-component-unpersisted-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  const auto acquire = [&](TextRunnerPool& pool, bool persist) {
    return pool.Acquire(prompt, {}, {}, {}, true, 0, true, persist);
  };
  TextRunnerToken token;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = acquire(pool, false);
    Prefill(first);
    token = Generate(first);
    assert(first.Commit().disk_queued_bytes == 0);
    // The RAM cache still serves the prompt within this process.
    auto again = acquire(pool, false);
    assert(again.cache_hit() && !again.cache_disk_hit());
    assert(again.cached_prompt_tokens() == prompt.size());
    again.Invalidate();
  }
  assert(ManagedBytes(directory) == 0);
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto cold = acquire(pool, true);
    assert(!cold.cache_hit());
    Prefill(cold);
    assert(Generate(cold) == token);
    assert(cold.Commit().disk_queued_bytes > 0);
  }
  assert(ManagedBytes(directory) > 0);
  {
    // Reading what other requests stored is allowed.
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto reader = acquire(pool, false);
    assert(reader.cache_disk_hit());
    assert(reader.cached_prompt_tokens() == prompt.size());
    reader.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void UnpersistedOverRamCaptureIsRefused() {
  // A checkpoint larger than the RAM budget would be staged for disk only.
  // A request that stays off disk refuses it instead.
  const TemporaryDirectory temp("gufo-component-unpersisted-over-ram-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  auto runner = std::make_shared<Runner>();
  runner->private_blocks = 2;
  runner->rows_per_chunk = 4096;
  constexpr std::size_t kRam = 448 << 10;
  {
    ComponentTextCache cache(runner, 1, kRam, disk);
    std::vector<TextRunnerToken> prompt(65536);
    std::iota(prompt.begin(), prompt.end(), 10);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true, false);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    const auto bytes = lease->CaptureBytes();
    assert(bytes > kRam);
    assert(lease->TryReserveSnapshot(bytes, prompt.size(), false,
                                     SnapshotPurpose::kContinuation, prompt));
    bool refused = false;
    try {
      (void)lease->Capture();
    } catch (const cache::ResourceExhausted&) {
      refused = true;
    }
    assert(refused);
    lease->Invalidate();
  }
  assert(ManagedBytes(directory) == 0);
  std::filesystem::remove_all(directory);
}
void LoweredDiskBudgetEvictsOldestAtStartup() {
  const TemporaryDirectory temp("gufo-component-lowered-budget-test");
  const std::filesystem::path directory = temp.path();
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20,
                                  .staging_capacity_bytes = 7};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  // Restarting below the retained bytes keeps the budget's meaning: the
  // oldest conversation leaves at startup instead of failing to start.
  disk.capacity_bytes = ManagedBytes(directory) - 1;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    assert(ManagedBytes(directory) <= disk.capacity_bytes);
    auto retained = pool.Acquire(second);
    assert(retained.cache_disk_hit());
    retained.Invalidate();
    auto evicted = pool.Acquire(first);
    assert(!evicted.cache_hit());
    evicted.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void FilesystemFloorBlocksWritesAndReclaimsAtStartup() {
  const TemporaryDirectory temp("gufo-component-floor-test");
  std::atomic<std::uint64_t> available{1ULL << 30};
  TextRunnerDiskCacheOptions disk{
      .directory = temp.path(),
      .capacity_bytes = 1 << 20,
      .space_policy = {1 << 20, [&](const auto&) {
                         return platform::FilesystemInfo{
                             1ULL << 30, available.load(), true, true};
                       }}};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  assert(ManagedBytes(temp.path()) > 0);
  available = 0;
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    assert(ManagedBytes(temp.path()) == 0);
    auto request = pool.Acquire(prompt);
    assert(!request.cache_hit());
    Prefill(request);
    (void)Generate(request);
    (void)request.Commit();
  }
  assert(ManagedBytes(temp.path()) == 0);
  available = 1ULL << 30;
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  assert(ManagedBytes(temp.path()) > 0);
}

void OrphansLeaveBeforeCheckpoints() {
  const TemporaryDirectory temp("gufo-component-orphan-test");
  const std::filesystem::path directory = temp.path();
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20,
                                  .staging_capacity_bytes = 7};
  const std::vector<TextRunnerToken> first{10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto second = first;
  second.front() = 999;
  for (const auto& prompt : {first, second}) {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  // A crash left an unreferenced chunk that pushes the directory over a budget
  // the referenced checkpoints still fit; reclaiming it must suffice.
  disk.capacity_bytes = ManagedBytes(directory);
  const auto orphan = directory / "v2/chunks" / (std::string(32, 'f') + ".bin");
  std::ofstream(orphan, std::ios::binary) << std::string(4096, 'x');
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    assert(!std::filesystem::exists(orphan));
    assert(ManagedBytes(directory) <= disk.capacity_bytes);
    for (const auto& prompt : {first, second}) {
      auto restored = pool.Acquire(prompt);
      assert(restored.cache_disk_hit());
      restored.Invalidate();
    }
  }
  std::filesystem::remove_all(directory);
}
void StagingIsInternal() {
  const TemporaryDirectory temp("gufo-component-staging-test");
  const std::filesystem::path directory = temp.path();
  // The legacy option is ignored: even a value below one four-byte row
  // still writes and restores through the internal staging buffer.
  TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                  .capacity_bytes = 1 << 20,
                                  .staging_capacity_bytes = 3};
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto request = pool.Acquire(prompt);
    Prefill(request);
    (void)Generate(request);
    assert(request.Commit().disk_queued_bytes > 0);
  }
  {
    TextRunnerPool pool(std::make_shared<Runner>(), 1, disk);
    auto restored = pool.Acquire(prompt);
    assert(restored.cache_disk_hit());
    restored.Invalidate();
  }
  std::filesystem::remove_all(directory);
}
void StagingCountsAgainstRamCeiling() {
  class CeilingRunner final : public Runner {
  public:
    mutable std::size_t committed{};
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = Runner::ResourceClaim();
      claim.retained_snapshot_capacity_bytes = 2 << 20;
      claim.retained_snapshot_ceiling_bytes = 4 << 20;
      return claim;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this](std::size_t budget) {
        committed = budget;
      };
      return resources;
    }
  };
  const TemporaryDirectory temp("gufo-component-ceiling-test");
  const std::filesystem::path directory = temp.path();
  // Pinned staging and RAM share the ceiling: an explicit 4 MiB beside the
  // internal 1 MiB of staging commits 3 MiB, whatever the ignored option says.
  for (const std::size_t option : {std::size_t{0}, std::size_t{4} << 20}) {
    TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                    .capacity_bytes = 1 << 20,
                                    .staging_capacity_bytes = option};
    auto runner = std::make_shared<CeilingRunner>();
    TextRunnerPool pool(runner, 1, disk, {.capacity_bytes = 4 << 20});
    assert(runner->committed == 3 << 20);
  }
  std::filesystem::remove_all(directory);
}
// Slots that pin host memory lower both measured budgets and are charged in
// the ledger. They count once, on top of the cache's budget: backing gets the
// whole budget unless the host ceiling binds.
void PinnedSlotBytesCountOnce() {
  constexpr std::size_t kPin = 512 << 10;
  class PinningRunner final : public Runner {
  public:
    mutable std::size_t committed{}, limit{};
    mutable std::vector<cache::ResourceCharge> pins;
    TextRunnerResourceClaim ResourceClaim() const override {
      auto claim = Runner::ResourceClaim();
      // Automatic is half the available RAM; the ceiling is all of it.
      claim.retained_snapshot_capacity_bytes =
          (2 << 20) - pins.size() * kPin / 2;
      claim.retained_snapshot_ceiling_bytes = (4 << 20) - pins.size() * kPin;
      return claim;
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      // Each slot takes a stream as it allocates; pin one buffer per slot.
      resources->stream = [this, &ledger,
                           stream = std::move(resources->stream)] {
        if (!committed)
          pins.push_back(
              ledger.Reserve(cache::ResourceCategory::kMetadata, kPin)
                  .Convert());
        return stream();
      };
      resources->pinned_bytes = [this] { return pins.size() * kPin; };
      resources->commit_backing = [this, &ledger](std::size_t budget) {
        committed = budget;
        limit = ledger.Limits().ram_bytes;
      };
      return resources;
    }
  };
  // Automatic, an explicit budget smaller than the pins, and one the 4 MiB
  // host ceiling cuts to make room for them.
  for (const auto& [requested, backing] :
       {std::pair{0ULL, 2ULL << 20}, std::pair{1ULL << 20, 1ULL << 20},
        std::pair{8ULL << 20, 3ULL << 20}}) {
    auto runner = std::make_shared<PinningRunner>();
    TextRunnerPool pool(runner, 2, {}, {.capacity_bytes = requested});
    assert(runner->pins.size() == 2);
    assert(runner->committed == backing);
    assert(runner->limit == backing + 2 * kPin);
  }
  // Pins never raise the limit past a budget the other charges exceed: the
  // fake backing alone holds 40 KiB.
  bool refused = false;
  try {
    TextRunnerPool pool(std::make_shared<PinningRunner>(), 2, {},
                        {.capacity_bytes = 16 << 10});
  } catch (const std::exception&) {
    refused = true;
  }
  assert(refused);
  // Staging shares the ceiling: the limit is 4 MiB less 1 MiB of staging.
  const TemporaryDirectory temp("gufo-component-pinned-test");
  const std::filesystem::path directory = temp.path();
  {
    const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                          .capacity_bytes = 1 << 20};
    auto runner = std::make_shared<PinningRunner>();
    TextRunnerPool pool(runner, 2, disk, {.capacity_bytes = 8 << 20});
    assert(runner->limit == 3 << 20);
    assert(runner->committed == 2 << 20);
  }
  std::filesystem::remove_all(directory);
}
// Without pins the limit ends at the requested budget, whether or not the
// runner commits backing, and a budget below what is already charged still
// fails at startup.
void UnpinnedRunnersKeepTheirLimit() {
  class LimitRunner final : public Runner {
  public:
    bool commit{};
    mutable cache::ResourceLedger* ledger{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& l, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(l, ram);
      ledger = &l;
      if (commit)
        resources->commit_backing = [](std::size_t) {};
      return resources;
    }
  };
  for (const bool commit : {false, true}) {
    auto runner = std::make_shared<LimitRunner>();
    runner->commit = commit;
    {
      TextRunnerPool pool(runner, 2, {}, {.capacity_bytes = 1 << 20});
      assert(runner->ledger->Limits().ram_bytes == 1 << 20);
    }
    // The fake backing alone holds 40 KiB.
    bool refused = false;
    try {
      TextRunnerPool pool(runner, 2, {}, {.capacity_bytes = 16 << 10});
    } catch (const std::exception&) {
      refused = true;
    }
    assert(refused);
  }
}
void DiskDirectoryHasOneOwner() {
  class CommitRunner final : public Runner {
  public:
    mutable bool committed{};
    mutable std::size_t streams{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this](std::size_t) { committed = true; };
      // Every execution slot takes a preservation stream when it allocates.
      resources->stream = [this, stream = std::move(resources->stream)] {
        ++streams;
        return stream();
      };
      return resources;
    }
  };
  const TemporaryDirectory temp("gufo-component-owner-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7};
  {
    TextRunnerPool owner(std::make_shared<CommitRunner>(), 1, disk);
    auto second_runner = std::make_shared<CommitRunner>();
    std::string error;
    try {
      TextRunnerPool second(second_runner, 1, disk);
    } catch (const std::exception& e) {
      error = e.what();
    }
    assert(error.find("ownership") != std::string::npos &&
           error.find(directory.string()) != std::string::npos);
    // The losing server fails before allocating slots or committing its RAM.
    assert(!second_runner->streams && !second_runner->committed);
  }
  // Ownership ends with the owning server.
  {
    auto runner = std::make_shared<CommitRunner>();
    TextRunnerPool reopened(runner, 1, disk);
    assert(runner->committed);
  }
  std::filesystem::remove_all(directory);
}
void GridDoesNotSuppressFirstDiskPrompt() {
  const TemporaryDirectory temp("gufo-component-spacing-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7};
  std::vector<TextRunnerToken> prompt(42);
  std::iota(prompt.begin(), prompt.end(), 10);
  std::uint64_t hash;
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 22, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, std::span(prompt).first(40),
                            std::span(prompt).first(40));
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 40, true,
                                     SnapshotPurpose::kHistory, {}));
    assert(lease->PublishSnapshot({prompt.begin(), prompt.begin() + 40},
                                  lease->Capture(), false));
    assert(lease->HasSnapshotFor(std::span(prompt).first(40)));
    assert(!lease->Metrics().disk_queued_bytes);
    runner->adapter->Append(state.slot, std::span(prompt).last(2),
                            std::span(prompt).last(2));
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), 42, false,
                                     SnapshotPurpose::kContinuation, {}));
    assert(lease->PublishSnapshot(prompt, lease->Capture(), false));
    assert(lease->Metrics().disk_queued_bytes > 0);
    lease->Invalidate();
  }
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 22, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() && lease->cached_tokens() == 42);
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
  }
  std::filesystem::remove_all(directory);
}
void PublishedResidentCheckpointKeepsDiskStep() {
  // A published checkpoint that is still resident restores from RAM, so its
  // lookup candidate carries no disk description. The next capture within
  // the step must still be refused.
  const TemporaryDirectory temp("gufo-component-step-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7,
                                        .min_checkpoint_step_tokens = 16};
  std::vector<TextRunnerToken> prompt(48);
  std::iota(prompt.begin(), prompt.end(), 10);
  const std::vector<TextRunnerToken> first(prompt.begin(), prompt.begin() + 40);
  auto runner = std::make_shared<Runner>();
  ComponentTextCache cache(runner, 1, 1 << 22, disk);
  {
    auto lease = cache.Acquire(first, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, first, first);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), first.size(), false,
                                     SnapshotPurpose::kContinuation, {}));
    assert(lease->PublishSnapshot(first, lease->Capture(), false));
    assert(lease->Metrics().disk_queued_bytes > 0);
    lease->Invalidate();
    // Invalidation releases the execution slot so the idle queue can publish.
    // Wait for completion rather than guessing when a created file is attached.
    for (int i = 0; i < 500 && lease->Metrics().persistence->completed == 0;
         ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(lease->Metrics().persistence->completed == 1);
  }
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(lease->cached_tokens() == first.size() &&
         !lease->restored_from_disk());
  auto& state = dynamic_cast<State&>(lease->state());
  const auto tail = std::span(prompt).subspan(first.size());
  runner->adapter->Append(state.slot, tail, tail);
  assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                   SnapshotPurpose::kContinuation, {}));
  assert(lease->PublishSnapshot(prompt, lease->Capture(), false));
  assert(lease->Metrics().disk_queued_bytes == 0);
  lease->Invalidate();
  std::filesystem::remove_all(directory);
}
void NearTailDiskBranchSurvivesRestart() {
  const TemporaryDirectory temp("gufo-component-branch-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 7};
  std::vector<TextRunnerToken> seed(1100);
  std::iota(seed.begin(), seed.end(), 10);
  auto branch = seed;
  branch.resize(800);
  branch.back() = 9999;
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    auto first = pool.Acquire(seed);
    Prefill(first);
    (void)Generate(first);
    (void)first.Commit();
    auto fork = pool.Acquire(branch);
    Prefill(fork);
    (void)Generate(fork);
    (void)fork.Commit();
  }
  {
    auto runner = std::make_shared<Runner>();
    TextRunnerPool pool(runner, 1, disk);
    branch.back() = 8888;
    auto restored = pool.Acquire(branch);
    assert(restored.cache_disk_hit());
    assert(restored.cached_prompt_tokens() == 799);
    Prefill(restored);
    const auto token = Generate(restored);
    restored.Invalidate();
    auto cold = pool.Acquire(branch, sampling::SamplingConfig{}, {}, {}, false);
    Prefill(cold);
    assert(Generate(cold) == token);
  }
  std::filesystem::remove_all(directory);
}
void UnbackedDeepRowsSurviveDiskRestart() {
  struct Gate {
    std::atomic<bool> entered{false};
    std::binary_semaphore reading{0}, release{0};
  };
  struct HeldRunner final : Runner {
    std::shared_ptr<Gate> gate = std::make_shared<Gate>();
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->copy_observed_rows =
          [gate = gate, copy = resources->copy_observed_rows](
              auto source, auto destination, cache::Stream& stream) {
            if (!gate->entered.exchange(true)) {
              gate->reading.release();
              gate->release.acquire();
            }
            return copy(source, destination, stream);
          };
      return resources;
    }
  };
  const TemporaryDirectory temp("gufo-component-deep-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .staging_capacity_bytes = 8,
                                        .min_checkpoint_step_tokens = 1};
  std::vector<TextRunnerToken> prompt(5001);
  std::iota(prompt.begin(), prompt.end(), 10);
  std::uint64_t hash;
  {
    auto runner = std::make_shared<HeldRunner>();
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    // Over 40,000 logical row bytes exceed the 32,768-byte row pool.
    auto snapshot = lease->Capture();
    assert(snapshot);
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 40000);
    assert(runner->gate->reading.try_acquire_for(std::chrono::seconds(5)));
    const auto pending = cache.PendingPublication(prompt, {});
    assert(pending && pending->checkpoint_id &&
           pending->pending_tokens == prompt.size() &&
           pending->available_tokens == 0 && pending->Gain() == prompt.size());
    assert(!cache.PendingPublication(prompt, {}, pending->checkpoint_id + 1));
    assert(!cache.PendingPublication(std::span(prompt).first(4096), {}));
    auto different = prompt;
    ++different.front();
    assert(!cache.PendingPublication(different, {}));
    TextPromptContext image;
    image.cache_identity = {99};
    assert(!cache.PendingPublication(prompt, &image));
    (void)lease->Commit(prompt, {}, prompt);
    // A newly usable live frontier makes the pending write unhelpful. The
    // tracked job remains identifiable so its existing deadline can shrink;
    // a fresh admission must not begin waiting for it.
    const auto tracked =
        cache.PendingPublication(prompt, {}, pending->checkpoint_id);
    assert(tracked && tracked->Gain() == 0);
    assert(!cache.PendingPublication(prompt, {}));
    runner->gate->release.release();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (cache.PendingPublication(prompt, {}, pending->checkpoint_id) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    assert(!cache.PendingPublication(prompt, {}, pending->checkpoint_id));
  }
  {
    const auto seed_size = prompt.size();
    std::vector<TextRunnerToken> appended(63);
    std::iota(appended.begin(), appended.end(), 8000);
    prompt.insert(prompt.end(), appended.begin(), appended.end());
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() && lease->cached_tokens() == seed_size);
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
    runner->adapter->Append(state.slot, appended, appended);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    auto snapshot = lease->Capture();
    assert(snapshot);
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 40000);
    (void)lease->Commit(prompt, {}, prompt);
  }
  {
    cache::ResourceLedger ledger({1 << 24, 1 << 24, 0, 0});
    cache::DiskStore store(ledger, directory, disk.capacity_bytes);
    assert(store.Entries().size() == 2);
    const cache::DiskManifest* parent = nullptr;
    const cache::DiskManifest* child = nullptr;
    for (const auto& entry : store.Entries()) {
      if (entry.Manifest().tokens.size() == 5001)
        parent = &entry.Manifest();
      else if (entry.Manifest().tokens.size() == prompt.size())
        child = &entry.Manifest();
    }
    assert(parent && child);
    for (std::size_t i = 0; i < parent->components.size(); ++i) {
      const auto& before = parent->components[i];
      const auto& after = child->components[i];
      for (std::size_t j = 0; j < before.chunks.size(); ++j)
        assert(before.chunks[j].file == after.chunks[j].file);
      if (before.private_state)
        assert(before.private_state->file != after.private_state->file);
      if (before.tail) {
        assert(after.chunks.size() > before.chunks.size());
        assert(before.tail->file != after.chunks[before.chunks.size()].file);
      }
    }
  }
  {
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 1, 1 << 24, disk);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(lease->restored_from_disk() &&
           lease->cached_tokens() == prompt.size());
    auto& state = dynamic_cast<State&>(lease->state());
    assert(runner->adapter->RecurrentHash(state.slot) == hash);
  }
  std::filesystem::remove_all(directory);
}
void RamRefusalDiskDirectRestartContract() {
  enum class Case {
    kRamOnly,
    kSuccess,
    kOptOut,
    kFloor,
    kQuota,
    kPrivateBacking
  };
  const std::vector<TextRunnerToken> suffix{9001, 9002, 9003};
  std::vector<TextRunnerToken> prompt(4096);
  std::iota(prompt.begin(), prompt.end(), 10);
  for (const auto test : {Case::kRamOnly, Case::kSuccess, Case::kOptOut,
                          Case::kFloor, Case::kQuota, Case::kPrivateBacking}) {
    const TemporaryDirectory temp("gufo-component-direct-contract");
    auto runner = std::make_shared<Runner>();
    // Keep RAM caching enabled, but deny append-row backing. A normal RAM
    // capture must fail; the existing fallback can only borrow live rows.
    runner->row_blocks = 0;
    runner->private_blocks = test == Case::kPrivateBacking ? 0 : 2;
    runner->rows_per_chunk = 256;
    const auto make_disk = [&] {
      return TextRunnerDiskCacheOptions{
          .directory = temp.path(),
          .capacity_bytes = test == Case::kQuota ? 1U : 1U << 20,
          .min_checkpoint_step_tokens = 1,
          .space_policy = {1 << 20, [test](const auto&) {
                             return platform::FilesystemInfo{
                                 1ULL << 30,
                                 test == Case::kFloor ? 0ULL : 1ULL << 30, true,
                                 true};
                           }}};
    };
    const auto previous_level = Logger::Level();
    Logger::SetLevel(LogLevel::kInfo);
    std::ostringstream output;
    auto* previous_sink = std::clog.rdbuf(output.rdbuf());
    std::uint64_t cold_hash{};
    {
      const std::optional<TextRunnerDiskCacheOptions> disk =
          test == Case::kRamOnly ? std::nullopt
                                 : std::make_optional(make_disk());
      ComponentTextCache cache(runner, 1, 1 << 20, disk);
      auto lease =
          cache.Acquire(prompt, {}, {}, true, 0, true, test != Case::kOptOut);
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, prompt, prompt);
      cold_hash = runner->adapter->RecurrentHash(state.slot);
      assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                       false, SnapshotPurpose::kContinuation,
                                       prompt));
      bool refused{};
      try {
        auto snapshot = lease->Capture();
        assert(snapshot);
        assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
      } catch (const cache::ResourceExhausted&) {
        refused = true;
      }
      const bool capture_denied = test == Case::kRamOnly ||
                                  test == Case::kOptOut ||
                                  test == Case::kPrivateBacking;
      assert(refused == capture_denied);
      assert((lease->Metrics().disk_queued_bytes == 0) == capture_denied);
      (void)lease->Commit(prompt, {}, prompt);
      if (!capture_denied) {
        // Submission is not durability. Wait for the actual completion result.
        for (int i = 0; i < 500; ++i) {
          const auto stats = *lease->Metrics().persistence;
          if (stats.completed + stats.failed)
            break;
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        const auto stats = *lease->Metrics().persistence;
        assert(stats.pending == 0);
        assert(stats.completed == (test == Case::kSuccess ? 1U : 0U));
        assert(stats.failed == (test == Case::kSuccess ? 0U : 1U));
      } else if (test != Case::kRamOnly) {
        const auto stats = *lease->Metrics().persistence;
        assert(stats.completed == 0 && stats.failed == 0 && stats.pending == 0);
      }
    }
    std::clog.rdbuf(previous_sink);
    Logger::SetLevel(previous_level);
    const auto log = output.str();
    if (test == Case::kSuccess) {
      assert(log.find("event=published") != std::string::npos);
      assert(ManagedBytes(temp.path()) > 0);
    } else {
      assert(log.find("event=published") == std::string::npos);
      assert(log.find(test == Case::kFloor || test == Case::kQuota
                          ? "event=persistence_incomplete"
                          : "event=capture_refused") != std::string::npos);
      if (test != Case::kRamOnly)
        assert(ManagedBytes(temp.path()) == 0);
    }
    // Remove the denial on restart: a miss must mean no durable publication,
    // not an unchanged floor, quota or private-backing limitation.
    auto restored_runner = std::make_shared<Runner>();
    restored_runner->row_blocks = 0;
    restored_runner->private_blocks = 2;
    restored_runner->rows_per_chunk = 256;
    const TextRunnerDiskCacheOptions restored_disk{
        .directory = temp.path(),
        .capacity_bytes = 1 << 20,
        .space_policy = {1 << 20, [](const auto&) {
                           return platform::FilesystemInfo{
                               1ULL << 30, 1ULL << 30, true, true};
                         }}};
    ComponentTextCache restarted(restored_runner, 1, 1 << 20, restored_disk);
    auto warm = restarted.Acquire(prompt, {}, {}, true, 0, true);
    assert(warm->restored_from_disk() == (test == Case::kSuccess));
    assert(warm->cached_tokens() ==
           (test == Case::kSuccess ? prompt.size() : 0));
    if (test == Case::kSuccess) {
      auto& state = dynamic_cast<State&>(warm->state());
      assert(restored_runner->adapter->RecurrentHash(state.slot) == cold_hash);
      restored_runner->adapter->Append(state.slot, suffix, suffix);
      const auto warm_hash =
          restored_runner->adapter->RecurrentHash(state.slot);
      warm->Invalidate();
      auto cold = restarted.Acquire(prompt, {}, {}, false, 0, true, false);
      assert(cold->cached_tokens() == 0);
      auto& cold_state = dynamic_cast<State&>(cold->state());
      restored_runner->adapter->Append(cold_state.slot, prompt, prompt);
      restored_runner->adapter->Append(cold_state.slot, suffix, suffix);
      assert(restored_runner->adapter->RecurrentHash(cold_state.slot) ==
             warm_hash);
      cold->Invalidate();
    } else {
      warm->Invalidate();
    }
  }
}

void OverRamCaptureStagesPrivateStateUnderPressure(bool protect) {
  // An over-RAM checkpoint is captured for disk only. Its rows are borrowed
  // from the live slot, but its private state is staged in RAM backing, and
  // two retained checkpoints fill that backing. Wide chunks keep the capture
  // metadata within RAM.
  const TemporaryDirectory temp("gufo-component-over-ram-test");
  const std::filesystem::path directory = temp.path();
  const TextRunnerDiskCacheOptions disk{.directory = directory.string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  auto runner = std::make_shared<Runner>();
  runner->private_blocks = 2;
  runner->rows_per_chunk = 4096;
  constexpr std::size_t kRam = 448 << 10;
  const auto previous_level = Logger::Level();
  Logger::SetLevel(LogLevel::kInfo);
  std::ostringstream output;
  auto* previous_sink = std::clog.rdbuf(output.rdbuf());
  {
    ComponentTextCache cache(runner, 1, kRam, disk);
    const auto history = [](auto& lease, const auto& tokens,
                            std::size_t boundary) {
      assert(lease->TryReserveSnapshot(lease->CaptureBytes(), boundary, false,
                                       SnapshotPurpose::kHistory, {}));
      assert(lease->PublishSnapshot({tokens.begin(), tokens.begin() + boundary},
                                    lease->Capture(), false));
    };
    const auto append = [&](auto& lease, std::span<const TextRunnerToken> t) {
      auto& state = dynamic_cast<State&>(lease->state());
      runner->adapter->Append(state.slot, t, t);
    };
    std::vector<TextRunnerToken> prompt(65536);
    std::iota(prompt.begin(), prompt.end(), 10);
    if (!protect) {
      // The cache may reclaim another conversation's history checkpoints.
      const std::vector<TextRunnerToken> other(4, 7);
      auto lease = cache.Acquire(other, {}, {}, true, 0, true);
      append(lease, std::span(other).first(2));
      history(lease, other, 2);
      append(lease, std::span(other).subspan(2));
      history(lease, other, 4);
      (void)lease->Commit({}, {}, other);
    }
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    if (protect) {
      // The cache keeps this turn's own edit boundaries until it commits.
      append(lease, std::span(prompt).first(2));
      history(lease, prompt, 2);
      append(lease, std::span(prompt).subspan(2, 2));
      history(lease, prompt, 4);
      append(lease, std::span(prompt).subspan(4));
    } else {
      append(lease, prompt);
    }
    const auto bytes = lease->CaptureBytes();
    assert(bytes > kRam);
    assert(lease->TryReserveSnapshot(bytes, prompt.size(), false,
                                     SnapshotPurpose::kContinuation, prompt));
    if (protect) {
      bool refused = false;
      try {
        (void)lease->Capture();
      } catch (const cache::ResourceExhausted&) {
        refused = true;
      }
      assert(refused);
    } else {
      // Persisting would charge the manifest's tokens again, and in this fake
      // they cost as many bytes as the rows. This test covers the capture.
      auto snapshot = lease->Capture();
      assert(snapshot && snapshot->PayloadBytes() == bytes);
    }
    lease->Invalidate();
  }
  std::clog.rdbuf(previous_sink);
  Logger::SetLevel(previous_level);
  const auto log = output.str();
  const auto refused =
      log.find("event=capture_refused reason=byte_capacity tokens=65536");
  assert(protect == (refused != std::string::npos));
  // Only the other conversation's history is reclaimed. Edit boundaries stay.
  const auto reclaimed =
      "event=retention action=" +
      std::to_string(static_cast<unsigned>(cache::RetentionAction::kRemoved)) +
      " reason=" +
      std::to_string(
          static_cast<unsigned>(cache::RetentionReason::kByteCapacity));
  assert(protect == (log.find(reclaimed) == std::string::npos));
  assert(log.find("event=capture_failed") == std::string::npos);
  std::filesystem::remove_all(directory);
}
// Registered pins intentionally retain the strict Linux runtime contract.
#if defined(__linux__)
class PinRunner final : public Runner {
public:
  mutable std::size_t prefilled{};
  TextRunnerDescriptor Descriptor() const override {
    auto descriptor = Runner::Descriptor();
    descriptor.persistence = TextRunnerPersistenceDescriptor{{1, 2, 3}, 1};
    return descriptor;
  }
  std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const override {
    std::istringstream input{std::string(text)};
    std::vector<TextRunnerToken> tokens;
    TextRunnerToken token;
    while (input >> token)
      tokens.push_back(token);
    return tokens;
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t budget) const override {
    const auto step = Runner::Prefill(state, prompt, offset, budget);
    prefilled += step.consumed_tokens;
    return step;
  }
  TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& s = dynamic_cast<State&>(state);
    const auto hash = s.adapter.RecurrentHash(s.slot);
    std::array<float, 8> logits;
    for (std::size_t i = 0; i < logits.size(); ++i)
      logits[i] = static_cast<float>((hash >> (8 * i)) & 255) / 64;
    return {.token = static_cast<TextRunnerToken>(sampler.Sample(logits)),
            .piece = "x"};
  }
};
struct PinFixture {
  TemporaryDirectory temporary{"gufo-pinned-prefix-test"};
  std::filesystem::path root{temporary.path()};
  TextRunnerPinnedPrefixOptions options;
  std::vector<TextRunnerToken> tokens{10, 11, 12, 13, 14, 15, 16, 17};
  PinFixture() {
    options.files = {root / "rendered.txt"};
    options.directory = root / "pins";
    options.capacity_bytes = 1 << 20;
    WriteTokens();
  }
  void WriteTokens() const {
    std::ofstream file(options.files.front());
    for (const auto token : tokens)
      file << token << ' ';
  }
  void AddTokens(std::span<const TextRunnerToken> prefix) {
    options.files.push_back(
        root / ("rendered-" + std::to_string(options.files.size()) + ".txt"));
    std::ofstream file(options.files.back());
    for (const auto token : prefix)
      file << token << ' ';
  }
  std::uintmax_t ManagedBytes() const {
    std::uintmax_t bytes = 0;
    for (const auto& file :
         std::filesystem::recursive_directory_iterator(options.directory))
      if (file.is_regular_file())
        bytes += file.file_size();
    return bytes;
  }
  auto Files() const {
    std::map<std::filesystem::path,
             std::pair<std::uintmax_t, std::filesystem::file_time_type>>
        result;
    for (const auto& file :
         std::filesystem::recursive_directory_iterator(options.directory))
      if (file.is_regular_file())
        result[file.path()] = {file.file_size(), file.last_write_time()};
    return result;
  }
};
void PinnedPrefixRestartReplayAndNoRewrite() {
  for (const bool ordinary_disk : {false, true}) {
    PinFixture fixture;
    std::optional<TextRunnerDiskCacheOptions> disk;
    if (ordinary_disk)
      disk = TextRunnerDiskCacheOptions{.directory = fixture.root / "ordinary",
                                        .capacity_bytes = 1 << 20};
    auto prompt = fixture.tokens;
    prompt.insert(prompt.end(), {100, 101, 102});
    std::array<std::vector<TextRunnerToken>, 2> expected;
    for (std::size_t mode = 0; mode < expected.size(); ++mode) {
      auto cold_runner = std::make_shared<PinRunner>();
      TextRunnerPool cold(cold_runner, 1);
      auto request = cold.Acquire(
          prompt, {.temperature = mode ? 0.8F : 0.0F, .seed = 1234});
      Prefill(request);
      for (int i = 0; i < 8; ++i)
        expected[mode].push_back(Generate(request));
      request.Invalidate();
    }
    {
      auto runner = std::make_shared<PinRunner>();
      TextRunnerPool pool(runner, 2, disk, {.pinned_prefix = fixture.options});
      assert(runner->prefilled == fixture.tokens.size());
      const auto lookup = pool.Lookup(prompt);
      assert(lookup.tier == "ram" &&
             lookup.matched_tokens == fixture.tokens.size());
      for (std::size_t mode = 0; mode < expected.size(); ++mode) {
        auto request = pool.Acquire(
            prompt, {.temperature = mode ? 0.8F : 0.0F, .seed = 1234});
        assert(request.cached_prompt_tokens() == fixture.tokens.size());
        assert(!request.cache_disk_hit());
        Prefill(request);
        for (const auto token : expected[mode])
          assert(Generate(request) == token);
        request.Invalidate();
      }
    }
    const auto published = fixture.Files();
    for (const bool lazy : {false, true}) {
      auto options = fixture.options;
      options.lazy = lazy;
      auto runner = std::make_shared<PinRunner>();
      TextRunnerPool pool(runner, 2, disk, {.pinned_prefix = options});
      assert(runner->prefilled == 0);  // restart must not recompute the root
      const auto lookup = pool.Lookup(prompt);
      assert(lookup.tier == (lazy ? "disk" : "ram"));
      assert(lookup.matched_tokens == fixture.tokens.size());
      assert(fixture.Files() == published);
      for (std::size_t mode = 0; mode < expected.size(); ++mode) {
        auto request = pool.Acquire(
            prompt, {.temperature = mode ? 0.8F : 0.0F, .seed = 1234});
        assert(request.cached_prompt_tokens() == fixture.tokens.size());
        assert(request.cache_disk_hit() == (lazy && mode == 0));
        Prefill(request);
        for (const auto token : expected[mode])
          assert(Generate(request) == token);
        request.Invalidate();
      }
      assert(fixture.Files() == published);
    }
  }
}
void UnpersistedRequestCanReuseRegisteredPins() {
  PinFixture fixture;
  const TextRunnerDiskCacheOptions disk{.directory = fixture.root / "ordinary",
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  {
    // Explicit operator registration is independent of request persistence.
    TextRunnerPool create(std::make_shared<PinRunner>(), 2, disk,
                          {.pinned_prefix = fixture.options});
  }
  const auto published = fixture.Files();
  auto prompt = fixture.tokens;
  prompt.insert(prompt.end(), {100, 101, 102});
  for (bool lazy : {false, true}) {
    auto options = fixture.options;
    options.lazy = lazy;
    {
      auto runner = std::make_shared<PinRunner>();
      TextRunnerPool pool(runner, 2, disk, {.pinned_prefix = options});
      assert(runner->prefilled == 0);
      auto request = pool.Acquire(prompt, {}, {}, {}, true, 0, true, false);
      assert(request.cached_prompt_tokens() == fixture.tokens.size());
      assert(request.cache_disk_hit() == lazy);
      Prefill(request);
      (void)Generate(request);
      assert(request.Commit().disk_queued_bytes == 0);
      auto replay = pool.Acquire(prompt, {}, {}, {}, true, 0, true, false);
      assert(replay.cached_prompt_tokens() == prompt.size());
      assert(!replay.cache_disk_hit());
      replay.Invalidate();
    }
    assert(ManagedBytes(disk.directory) == 0);
    assert(fixture.Files() == published);
  }
}
void PinnedPrefixesRespectOffAndAutomaticRam() {
  PinFixture fixture;
  // 0 is AUTO, not off: creating the registered RAM checkpoint still works.
  auto automatic = *ParseRamCacheOption("0", nullptr);
  automatic.pinned_prefix = fixture.options;
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, automatic);
    assert(runner->prefilled == fixture.tokens.size());
    assert(runner->captures != 0);
  }
  const auto published = fixture.Files();
  auto off = *ParseRamCacheOption("off", nullptr);
  off.pinned_prefix = fixture.options;
  // A request for mandatory RAM warmth is incompatible with off, even when
  // a durable pin already exists. Refusal must not change the registry.
  {
    bool refused = false;
    try {
      TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {}, off);
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find("RAM-warm") != std::string::npos;
    }
    assert(refused && fixture.Files() == published);
  }
  off.pinned_prefix.lazy = true;
  automatic.pinned_prefix.lazy = true;
  for (const auto& options : {off, automatic}) {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, options);
    assert(runner->prefilled == 0 && runner->captures == 0);
    for (int attempt = 0; attempt < 2; ++attempt) {
      auto request = pool.Acquire(fixture.tokens);
      assert(request.cached_prompt_tokens() == fixture.tokens.size());
      // Off always reads the disk pin; AUTO warms it on first use.
      assert(request.cache_disk_hit() == (!options.enabled || attempt == 0));
      request.Invalidate();  // Do not let live reuse hide a RAM promotion.
    }
    assert((runner->captures != 0) == options.enabled);
    assert(fixture.Files() == published);
  }
  // Incompatible/new registrations need capture. Check the complete set
  // before retiring the old identity, not after a failed attempt to warm it.
  fixture.tokens.front() = 999;
  fixture.WriteTokens();
  bool refused = false;
  try {
    TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {}, off);
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()).find("creating pinned prefixes") !=
              std::string::npos;
  }
  assert(refused && fixture.Files() == published);
}
void PinnedPrefixPressureIdentityAndAccounting(bool record_pressure = false) {
  PinFixture fixture;
  const std::vector<TextRunnerToken> second{300, 301, 302, 303, 304};
  fixture.AddTokens(second);
  auto runner = std::make_shared<PinRunner>();
  // Keep separate backing- and record-pressure cases. In the latter, all 128
  // ordinary records fit simultaneously, in addition to the permanent root.
  runner->private_blocks = record_pressure ? 256 : 4;
  ComponentTextCache cache(runner, 1, 1 << 22, {}, false, 0, fixture.options);
  const auto published = fixture.Files();
  {
    auto lease = cache.Acquire(fixture.tokens, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == fixture.tokens.size());
    assert(lease->Metrics().spill.foreground_bytes == 0);
  }
  for (TextRunnerToken family = 100; family < 240; ++family) {
    const std::vector<TextRunnerToken> prompt(8, family);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(!lease->cache_hit());
    auto& state = dynamic_cast<TextRunnerState&>(lease->state());
    (void)runner->Prefill(state, prompt, 0, prompt.size());
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    auto snapshot = lease->Capture();
    assert(lease->Commit(prompt, std::move(snapshot), prompt) > 0);
  }
  auto prompt = fixture.tokens;
  prompt.push_back(999);
  {
    auto lease = cache.Acquire(prompt, {}, {}, true, prompt.size(), true);
    assert(lease->cached_tokens() == fixture.tokens.size());
    assert(lease->HasSnapshotFor(fixture.tokens));
    const auto metrics = lease->Metrics();
    assert(metrics.ledger.ram_bytes > 0);
    assert(metrics.ledger.total_bytes == metrics.ledger.ram_bytes + (1 << 20));
    assert(metrics.ledger.bytes[static_cast<std::size_t>(
               cache::ResourceCategory::kPrivateState)] > 0);
    lease->Invalidate();
  }
  {
    auto lease = cache.Acquire(second, {}, {}, true, 0, true);
    assert(lease->cached_tokens() == second.size());
    assert(lease->HasSnapshotFor(second));
    assert(lease->Metrics().ledger.bytes[static_cast<std::size_t>(
               cache::ResourceCategory::kPrivateState)] >= 2 * 64);
    lease->Invalidate();
  }
  auto changed = prompt;
  changed.front() ^= 1;
  assert(!cache.Acquire(changed, {}, {}, true, 0, true)->cache_hit());
  auto image = std::make_shared<TextPromptContext>();
  image->cache_identity = {42};
  assert(!cache.Acquire(prompt, {}, image, true, 0, true)->cache_hit());
  image->cache_prefixes = {{fixture.tokens.size(), {}}};
  assert(cache.Acquire(prompt, {}, image, true, 0, true)->cached_tokens() ==
         fixture.tokens.size());
  assert(!cache.Acquire(prompt, {}, {}, false, 0, true)->cache_hit());
  assert(fixture.Files() == published);
}
void PinnedPrefixIdentityReplacementAndBudgets() {
  PinFixture fixture;
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = fixture.options});
  }
  // Relevant numerics environment changes cannot restore the old checkpoint.
  const char* previous = std::getenv("GUFO_TEST_PIN_NUMERICS");
  const std::optional<std::string> saved =
      previous ? std::optional<std::string>(previous) : std::nullopt;
  assert(platform::SetEnv("GUFO_TEST_PIN_NUMERICS", "new-build-lane"));
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = fixture.options});
    assert(runner->prefilled == fixture.tokens.size());
  }
  if (saved)
    assert(platform::SetEnv("GUFO_TEST_PIN_NUMERICS", *saved));
  else
    assert(platform::UnsetEnv("GUFO_TEST_PIN_NUMERICS"));
  fixture.tokens.back() = 77;
  fixture.WriteTokens();
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = fixture.options});
    assert(runner->prefilled == fixture.tokens.size());
  }
  auto options = fixture.options;
  options.capacity_bytes = 1;
  bool refused = false;
  try {
    TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                        {.pinned_prefix = options});
  } catch (const std::runtime_error&) {
    refused = true;
  }
  assert(refused);  // reduced budget never evicts a compatible pin
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = fixture.options});
    assert(runner->prefilled == 0);
  }
}
void MultiplePinsChooseLongestAndRestart() {
  for (const bool ordinary_disk : {false, true}) {
    PinFixture fixture;
    std::optional<TextRunnerDiskCacheOptions> disk;
    if (ordinary_disk)
      disk = TextRunnerDiskCacheOptions{.directory = fixture.root / "ordinary",
                                        .capacity_bytes = 1 << 20};
    {
      TextRunnerPool pool(std::make_shared<PinRunner>(), 1, disk,
                          {.pinned_prefix = fixture.options});
    }
    const auto single = fixture.Files();
    auto longer = fixture.tokens;
    longer.insert(longer.end(), {20, 21, 22, 23});
    const std::vector<TextRunnerToken> other{50, 51, 52, 53, 54};
    fixture.AddTokens(longer);
    fixture.AddTokens(other);
    {
      auto runner = std::make_shared<PinRunner>();
      TextRunnerPool pool(runner, 1, disk, {.pinned_prefix = fixture.options});
      // Extending the legacy single-manifest directory preserves that pin.
      assert(runner->prefilled == longer.size() + other.size());
      const auto added = fixture.Files();
      for (const auto& file : single)
        assert(added.at(file.first) == file.second);
    }
    const auto published = fixture.Files();
    // Option order and file names do not participate in identity.
    std::reverse(fixture.options.files.begin(), fixture.options.files.end());
    const auto renamed = fixture.root / "renamed.txt";
    std::filesystem::copy_file(fixture.options.files.front(), renamed);
    fixture.options.files.front() = renamed;
    for (const bool lazy : {false, true}) {
      auto options = fixture.options;
      options.lazy = lazy;
      auto runner = std::make_shared<PinRunner>();
      TextRunnerPool pool(runner, 1, disk, {.pinned_prefix = options});
      assert(runner->prefilled == 0);
      const std::array prefixes{fixture.tokens, longer, other};
      for (const auto& prefix : prefixes) {
        auto prompt = prefix;
        prompt.push_back(999);
        for (const float temperature : {0.0F, 0.8F}) {
          const sampling::SamplingConfig sampling{.temperature = temperature,
                                                  .seed = 1234};
          TextRunnerPool cold(std::make_shared<PinRunner>(), 1);
          auto control = cold.Acquire(prompt, sampling, {}, {}, false);
          Prefill(control);
          auto request = pool.Acquire(prompt, sampling);
          assert(request.cached_prompt_tokens() == prefix.size());
          assert(request.cache_disk_hit() == (lazy && temperature == 0.0F));
          Prefill(request);
          for (int i = 0; i < 8; ++i)
            assert(Generate(request) == Generate(control));
          request.Invalidate();
          control.Invalidate();
        }
      }
      auto changed = longer;
      changed.back() ^= 1;
      auto request = pool.Acquire(changed);
      assert(request.cached_prompt_tokens() == fixture.tokens.size());
      request.Invalidate();
      changed.front() ^= 1;
      request = pool.Acquire(changed);
      assert(!request.cache_hit());
      request.Invalidate();
      changed = fixture.tokens;
      changed.pop_back();  // a partial registration is not a checkpoint
      request = pool.Acquire(changed);
      assert(!request.cache_hit());
      request.Invalidate();
      auto image = std::make_shared<TextPromptContext>();
      image->cache_identity = {42};
      request = pool.Acquire(longer, {}, {}, image);
      assert(!request.cache_hit());
      request.Invalidate();
      image->cache_prefixes = {{fixture.tokens.size(), {}}};
      request = pool.Acquire(longer, {}, {}, image);
      assert(request.cached_prompt_tokens() == fixture.tokens.size());
      request.Invalidate();
      request = pool.Acquire(longer, {}, {}, {}, false);
      assert(!request.cache_hit());
      request.Invalidate();
      assert(fixture.Files() == published);
    }
  }
}
void MultiplePinsAggregateDiskBudget() {
  for (const bool ordinary_disk : {false, true}) {
    for (const bool lazy : {false, true}) {
      PinFixture fixture;
      std::optional<TextRunnerDiskCacheOptions> disk;
      if (ordinary_disk)
        disk = TextRunnerDiskCacheOptions{
            .directory = fixture.root / "ordinary", .capacity_bytes = 1 << 20};
      {
        TextRunnerPool pool(std::make_shared<PinRunner>(), 1, disk,
                            {.pinned_prefix = fixture.options});
      }
      const auto single_bytes = fixture.ManagedBytes();
      const auto single = fixture.Files();
      const std::vector<TextRunnerToken> other{50, 51, 52, 53, 54};
      fixture.AddTokens(other);
      auto options = fixture.options;
      options.lazy = lazy;
      options.capacity_bytes = single_bytes;
      bool refused = false;
      try {
        TextRunnerPool pool(std::make_shared<PinRunner>(), 1, disk,
                            {.pinned_prefix = options});
      } catch (const cache::DiskCapacityExhausted&) {
        refused = true;
      }
      assert(refused);  // not an independent byte budget for every manifest
      assert(fixture.Files() == single);
      {
        auto runner = std::make_shared<PinRunner>();
        TextRunnerPool pool(runner, 1, disk,
                            {.pinned_prefix = fixture.options});
        assert(runner->prefilled == other.size());
      }
      const auto total = fixture.ManagedBytes();
      assert(total > single_bytes && total < 2 * single_bytes);
      const auto published = fixture.Files();
      options.capacity_bytes = total - 1;
      refused = false;
      try {
        TextRunnerPool pool(std::make_shared<PinRunner>(), 1, disk,
                            {.pinned_prefix = options});
      } catch (const std::runtime_error&) {
        refused = true;
      }
      assert(refused);
      assert(fixture.Files() == published);  // neither compatible pin evicted
      options.capacity_bytes = total;
      {
        auto runner = std::make_shared<PinRunner>();
        ComponentTextCache cache(runner, 1, 1 << 22, disk, false, 0, options);
        assert(runner->prefilled == 0);
        auto lease = cache.Acquire(other, {}, {}, true, 0, true);
        const auto metrics = lease->Metrics();
        const auto staging = ordinary_disk ? 2 << 20 : 1 << 20;
        assert(metrics.ledger.total_bytes ==
               metrics.ledger.ram_bytes + staging);
      }
      assert(fixture.Files() == published);
      // Managed temporary/orphan bytes count too, even in lazy startup.
      const auto orphan = options.directory / "v2" / "tmp" /
                          "1234567890abcdef1234567890abcdef.bin";
      {
        std::ofstream file(orphan);
        assert(file);
        file << "orphan";
      }
      const auto with_orphan = fixture.Files();
      refused = false;
      try {
        TextRunnerPool pool(std::make_shared<PinRunner>(), 1, disk,
                            {.pinned_prefix = options});
      } catch (const std::runtime_error&) {
        refused = true;
      }
      assert(refused && fixture.Files() == with_orphan);
      std::filesystem::remove(orphan);
    }
  }
}
void MultiplePinsAggregateRamAndIndependentLazyWarming() {
  PinFixture fixture;
  const std::vector<TextRunnerToken> other{50, 51, 52, 53, 54};
  fixture.AddTokens(other);
  {
    TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                        {.pinned_prefix = fixture.options});
  }
  const auto published = fixture.Files();
  auto runner = std::make_shared<PinRunner>();
  runner->private_blocks = 1;  // either root fits, but not both simultaneously
  bool refused = false;
  try {
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = fixture.options});
  } catch (const std::exception&) {
    refused = true;
  }
  assert(refused && runner->prefilled == 0);
  assert(fixture.Files() == published);
  auto options = fixture.options;
  options.lazy = true;
  runner = std::make_shared<PinRunner>();
  runner->private_blocks = 1;
  TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = options});
  for (int pass = 0; pass < 3; ++pass) {
    for (const auto& prefix : {fixture.tokens, other}) {
      auto request = pool.Acquire(prefix);
      assert(request.cached_prompt_tokens() == prefix.size());
      assert(request.cache_disk_hit() == (pass == 0 || prefix == other));
      request.Invalidate();
    }
  }
  assert(runner->prefilled == 0 && fixture.Files() == published);
}
void MultiplePinsExplicitReplacementAndValidation() {
  PinFixture fixture;
  const std::vector<TextRunnerToken> other{50, 51, 52, 53, 54};
  fixture.AddTokens(other);
  {
    TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                        {.pinned_prefix = fixture.options});
  }
  const auto published = fixture.Files();
  const auto duplicate_file = fixture.root / "same-tokens.txt";
  std::filesystem::copy_file(fixture.options.files.front(), duplicate_file);
  for (const bool duplicate : {false, true}) {
    auto invalid = fixture.options;
    invalid.files.push_back(duplicate ? duplicate_file
                                      : fixture.root / "missing.txt");
    bool refused = false;
    try {
      TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                          {.pinned_prefix = invalid});
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused && fixture.Files() == published);
  }
  {
    auto invalid = fixture.options;
    invalid.files.assign(TextRunnerPinnedPrefixOptions::kMaxRegistrations + 1,
                         invalid.files.front());
    bool refused = false;
    try {
      TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                          {.pinned_prefix = invalid});
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    assert(refused && fixture.Files() == published);
  }
  // Only a changed registration is re-prefilled; the other remains durable.
  fixture.tokens.back() = 77;
  fixture.WriteTokens();
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = fixture.options});
    assert(runner->prefilled == fixture.tokens.size());
  }
  // Relevant numerics identity changes reject every registration, lazy too.
  const char* previous = std::getenv("GUFO_TEST_MULTI_PIN_NUMERICS");
  const std::optional<std::string> saved =
      previous ? std::optional<std::string>(previous) : std::nullopt;
  assert(platform::SetEnv("GUFO_TEST_MULTI_PIN_NUMERICS", "changed"));
  auto options = fixture.options;
  options.lazy = true;
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = options});
    assert(runner->prefilled == fixture.tokens.size() + other.size());
  }
  if (saved)
    assert(platform::SetEnv("GUFO_TEST_MULTI_PIN_NUMERICS", *saved));
  else
    assert(platform::UnsetEnv("GUFO_TEST_MULTI_PIN_NUMERICS"));
  // Shrinking the explicit set returns to a single-manifest legacy layout.
  options = fixture.options;
  options.files.resize(1);
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = options});
    assert(runner->prefilled == fixture.tokens.size());
  }
  {
    auto runner = std::make_shared<PinRunner>();
    TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = options});
    assert(runner->prefilled == 0);
    auto request = pool.Acquire(other);
    assert(!request.cache_hit());
    request.Invalidate();
  }
  const auto manifests = options.directory / "v2" / "manifests";
  assert(std::distance(std::filesystem::directory_iterator(manifests),
                       std::filesystem::directory_iterator{}) == 1);
}
void MultiplePinsLazyCorruptionPreservesOtherRegistrations() {
  PinFixture fixture;
  auto longer = fixture.tokens;
  longer.insert(longer.end(), {20, 21, 22, 23});
  const std::vector<TextRunnerToken> other{50, 51, 52, 53, 54};
  fixture.AddTokens(longer);
  fixture.AddTokens(other);
  {
    TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                        {.pinned_prefix = fixture.options});
  }
  std::filesystem::path damaged;
  {
    cache::ResourceLedger ledger({1 << 22, 1 << 22, 0, 1 << 22});
    cache::DiskStore disk(ledger, fixture.options.directory, 1 << 20);
    for (const auto& entry : disk.Entries()) {
      if (entry.Manifest().tokens != longer)
        continue;
      for (const auto& component : entry.Manifest().components)
        if (component.private_state)
          damaged = fixture.options.directory / "v2" / "private" /
                    cache::DiskFileName(component.private_state->file);
    }
  }
  assert(!damaged.empty());
  auto options = fixture.options;
  options.lazy = true;
  auto runner = std::make_shared<PinRunner>();
  TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = options});
  {
    // Corruption after lazy indexing must quarantine only affected roots.
    std::fstream file(damaged, std::ios::binary | std::ios::in | std::ios::out);
    assert(file);
    char byte;
    file.read(&byte, 1);
    byte ^= 0x5a;
    file.seekp(0);
    file.write(&byte, 1);
  }
  const auto corrupt_files = fixture.Files();
  for (const auto& prompt : {longer, other, longer}) {
    auto request = pool.Acquire(prompt, {.temperature = 0});
    const auto depth = prompt == longer ? fixture.tokens.size() : other.size();
    assert(request.cached_prompt_tokens() == depth);
    TextRunnerPool cold(std::make_shared<PinRunner>(), 1);
    auto control = cold.Acquire(prompt, {.temperature = 0}, {}, {}, false);
    Prefill(request);
    Prefill(control);
    for (int i = 0; i < 8; ++i)
      assert(Generate(request) == Generate(control));
    request.Invalidate();
    control.Invalidate();
  }
  assert(fixture.Files() ==
         corrupt_files);  // no repair/republication on demand
}
void PinnedPrefixCorruptionCannotHideBehindOrdinaryDisk() {
  PinFixture fixture;
  const TextRunnerDiskCacheOptions disk{.directory = fixture.root / "ordinary",
                                        .capacity_bytes = 1 << 20};
  {
    auto runner = std::make_shared<PinRunner>();
    ComponentTextCache cache(runner, 1, 1 << 22, disk, false, 0,
                             fixture.options);
    auto lease = cache.Acquire(fixture.tokens, {}, {}, false, 0, true);
    auto& state = dynamic_cast<TextRunnerState&>(lease->state());
    (void)runner->Prefill(state, fixture.tokens, 0, fixture.tokens.size());
    assert(lease->TryReserveSnapshot(
        lease->CaptureBytes(), fixture.tokens.size(), false,
        SnapshotPurpose::kContinuation, fixture.tokens));
    auto snapshot = lease->Capture();
    assert(lease->Commit(fixture.tokens, std::move(snapshot), fixture.tokens) >
           0);
  }  // drain a compatible equal-depth ordinary publication
  const auto path = std::filesystem::directory_iterator(
                        fixture.options.directory / "v2" / "private")
                        ->path();
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    char byte;
    file.read(&byte, 1);
    byte ^= 0x5a;
    file.seekp(0);
    file.write(&byte, 1);
  }
  bool refused = false;
  try {
    ComponentTextCache cache(std::make_shared<PinRunner>(), 1, 1 << 22, disk,
                             false, 0, fixture.options);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  assert(refused);
}
void PinnedPrefixLazySurvivesRamRefusal() {
  PinFixture fixture;
  {
    TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                        {.pinned_prefix = fixture.options});
  }
  const auto published = fixture.Files();
  auto options = fixture.options;
  options.lazy = true;
  auto runner = std::make_shared<PinRunner>();
  runner->private_blocks = 0;  // restore fits the slot, but RAM captures cannot
  TextRunnerPool pool(runner, 1, {}, {.pinned_prefix = options});
  assert(runner->prefilled == 0);
  for (const TextRunnerToken suffix : {100, 101}) {
    auto prompt = fixture.tokens;
    prompt.push_back(suffix);
    const auto lookup = pool.Lookup(prompt);
    assert(lookup.tier == "disk");
    assert(lookup.matched_tokens == fixture.tokens.size());
    assert(fixture.Files() == published);
    auto request = pool.Acquire(prompt);
    assert(request.cached_prompt_tokens() == lookup.matched_tokens);
    assert(request.cache_disk_hit());
    Prefill(request);
    const auto actual = Generate(request);
    request.Invalidate();
    TextRunnerPool cold(std::make_shared<PinRunner>(), 1);
    auto control = cold.Acquire(prompt);
    Prefill(control);
    assert(Generate(control) == actual);
    control.Invalidate();
  }
  assert(fixture.Files() == published);
}
void PinnedPrefixCorruptionFailsClosed() {
  for (const bool truncate : {false, true}) {
    PinFixture fixture;
    {
      TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                          {.pinned_prefix = fixture.options});
    }
    const auto path = std::filesystem::directory_iterator(
                          fixture.options.directory / "v2" / "private")
                          ->path();
    if (truncate) {
      std::filesystem::resize_file(path, 1);
    } else {
      std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
      char byte;
      file.read(&byte, 1);
      byte ^= 0x5a;
      file.seekp(0);
      file.write(&byte, 1);
    }
    bool refused = false;
    try {
      TextRunnerPool pool(std::make_shared<PinRunner>(), 1, {},
                          {.pinned_prefix = fixture.options});
    } catch (const std::runtime_error&) {
      refused = true;
    }
    assert(refused);  // no partial state execution or silent repeated writes
  }
}
#endif  // defined(__linux__)

void CheckpointImportRespectsRamOff() {
  for (const bool with_disk : {false, true}) {
    const TemporaryDirectory temp("gufo-component-import-off");
    std::optional<TextRunnerDiskCacheOptions> disk;
    if (with_disk)
      disk = TextRunnerDiskCacheOptions{.directory = temp.path(),
                                        .capacity_bytes = 1 << 20};
    auto runner = std::make_shared<Runner>();
    ComponentTextCache target(runner, 1, 1 << 22, disk, false, 0, {}, false);
    const std::vector<TextRunnerToken> prompt{10, 11, 12, 13};
    {
      auto live = target.Acquire(prompt, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(live->state());
      runner->adapter->Append(state.slot, prompt, prompt);
      (void)live->Commit({}, {}, prompt);
    }
    for (const std::uint64_t length : {0, 1024}) {
      bool read = false, refused = false;
      try {
        (void)target.Import(
            [&](std::span<std::uint8_t>) {
              read = true;
              return std::size_t{0};
            },
            length, "off-policy");
      } catch (const CheckpointTransferError& e) {
        refused = e.status == 507 &&
                  std::string_view(e.code) == "cache_transfer_budget";
      }
      assert(refused && !read && runner->captures == 0);
      const auto lookup = target.Lookup(prompt, nullptr, true, 0);
      assert(lookup && lookup->tier == "live" &&
             lookup->matched_tokens == prompt.size());
    }
  }
}
void CheckpointTransferRoundTripAndConcurrentServing() {
  class ImportRunner final : public Runner {
  public:
    bool refuse{};
    std::function<void()> preparing;
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->can_capture = [this](auto) {
        if (preparing)
          preparing();
        return !refuse;
      };
      return resources;
    }
  };
  auto source_runner = std::make_shared<Runner>();
  source_runner->defer_idle = true;
  ComponentTextCache source(source_runner, 1, 1 << 20, {});
  std::vector<TextRunnerToken> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 10);
  auto context = std::make_shared<TextPromptContext>();
  context->cache_identity = {3, 7, 20};
  auto seed = source.Acquire(prompt, {}, context, true, 0, true);
  auto& seed_state = dynamic_cast<State&>(seed->state());
  source_runner->adapter->Append(seed_state.slot, prompt, prompt);
  const auto expected_hash =
      source_runner->adapter->RecurrentHash(seed_state.slot);
  assert(seed->TryReserveSnapshot(seed->CaptureBytes(), prompt.size(), false,
                                  SnapshotPurpose::kContinuation, prompt));
  assert(seed->PublishSnapshot(prompt, seed->Capture(), false));
  // Exercise borrowed rows still in a busy slot, without capturing live state.
  auto blob =
      source.Export(prompt, context.get(), true, 0, "payload=20;build=A");
  std::vector<std::uint8_t> wire;
  blob->Write([&](auto bytes) {
    wire.insert(wire.end(), bytes.begin(), bytes.end());
    return true;
  });
  bool busy = false;
  try {
    (void)source.Export(prompt, context.get(), true, 0, "payload=20;build=A");
  } catch (const CheckpointTransferError& error) {
    busy = error.status == 503;
  }
  assert(busy);
  blob.reset();
  auto target_runner = std::make_shared<ImportRunner>();
  ComponentTextCache target(target_runner, 2, 1 << 20, {});
  const std::vector<TextRunnerToken> other{100, 101, 102, 103};
  auto serving = target.Acquire(other, {}, {}, false, 0, true);
  auto& live = dynamic_cast<State&>(serving->state());
  target_runner->adapter->Append(live.slot, other, other);
  const auto live_hash = target_runner->adapter->RecurrentHash(live.slot);
  auto reader = [&](const auto& bytes) -> CheckpointTransferReader {
    return
        [&bytes, offset = std::size_t{0}](std::span<std::uint8_t> out) mutable {
          const auto n = std::min(out.size(), bytes.size() - offset);
          std::memcpy(out.data(), bytes.data() + offset, n);
          offset += n;
          return n;
        };
  };
  const auto allocations = target_runner->captures;
  for (const auto bad : {0, 1, 2}) {
    auto invalid = wire;
    if (bad == 0)
      invalid.pop_back();
    if (bad == 1)
      invalid.back() ^= 1;
    bool rejected = false;
    try {
      (void)target.Import(
          reader(invalid), wire.size(),
          bad == 2 ? "payload=19;build=B" : "payload=20;build=A");
    } catch (const CheckpointTransferError& error) {
      rejected = error.status == (bad == 2 ? 409 : 400);
    }
    assert(rejected && target_runner->captures == allocations);
    assert(!target.Lookup(prompt, context.get(), true, 0)->matched_tokens);
  }
  target_runner->refuse = true;
  bool refused = false;
  try {
    (void)target.Import(reader(wire), wire.size(), "payload=20;build=A");
  } catch (const CheckpointTransferError& error) {
    refused = error.status == 507;
  }
  assert(refused);
  assert(target_runner->adapter->RecurrentHash(live.slot) == live_hash);
  target_runner->refuse = false;
  std::binary_semaphore reading(0), proceed(0), allocating(0), publish(0);
  target_runner->preparing = [&] {
    allocating.release();
    publish.acquire();
  };
  std::exception_ptr failure;
  std::jthread importing([&] {
    try {
      auto input = reader(wire);
      bool first = true;
      auto imported = target.Import(
          [&](auto out) {
            if (std::exchange(first, false)) {
              reading.release();
              proceed.acquire();
            }
            return input(out);
          },
          wire.size(), "payload=20;build=A");
      assert(imported->Header().manifest.tokens == prompt);
    } catch (...) {
      failure = std::current_exception();
    }
  });
  reading.acquire();
  for (int i = 0; i < 100; ++i)
    assert(!target.Lookup(prompt, context.get(), true, 0)->matched_tokens);
  // Generation advances while import is blocked on network input.
  const std::array<TextRunnerToken, 1> next{104};
  target_runner->adapter->Append(live.slot, next, next);
  proceed.release();
  allocating.acquire();  // Import holds the publication owner, not a GPU slot.
  for (int i = 0; i < 100; ++i)
    assert(!target.Lookup(prompt, context.get(), true, 0)->matched_tokens);
  const std::array<TextRunnerToken, 1> next_decode{105};
  target_runner->adapter->Append(live.slot, next_decode, next_decode);
  const auto advanced_hash = target_runner->adapter->RecurrentHash(live.slot);
  publish.release();
  importing.join();
  target_runner->preparing = {};
  if (failure)
    std::rethrow_exception(failure);
  assert(target_runner->adapter->RecurrentHash(live.slot) == advanced_hash);
  assert(target.Lookup(prompt, context.get(), true, 0)->matched_tokens ==
         prompt.size());
  auto changed = std::make_shared<TextPromptContext>();
  changed->cache_identity = {3, 7, 21};
  assert(!target.Lookup(prompt, changed.get(), true, 0)->matched_tokens);
  auto restored = target.Acquire(prompt, {}, context, true, 0, true);
  assert(restored->cached_tokens() == prompt.size());
  auto& restored_state = dynamic_cast<State&>(restored->state());
  assert(target_runner->adapter->RecurrentHash(restored_state.slot) ==
         expected_hash);
  auto again =
      target.Export(prompt, context.get(), true, 0, "payload=20;build=A");
  auto original = CheckpointTransferBlob::Import(reader(wire), wire.size(),
                                                 again->Header().identity,
                                                 "payload=20;build=A");
  const auto old_payloads =
      CheckpointTransferPayloads(original->Header().manifest);
  const auto new_payloads =
      CheckpointTransferPayloads(again->Header().manifest);
  assert(old_payloads.size() == new_payloads.size());
  for (std::size_t i = 0; i < old_payloads.size(); ++i) {
    std::vector<std::uint8_t> a(old_payloads[i]->bytes),
        b(new_payloads[i]->bytes);
    original->Read(i, 0, a);
    again->Read(i, 0, b);
    assert(a == b);
  }
}
void LookupRuntimeDefaultsAndDisabled() {
  class IdentityRunner final : public Runner {
  public:
    TextRunnerDescriptor Descriptor() const override {
      auto descriptor = Runner::Descriptor();
      descriptor.persistence = TextRunnerPersistenceDescriptor{{1, 2, 3}, 21};
      return descriptor;
    }
    mutable std::size_t identity_calls{};
    std::string LookupContextIdentity() const override {
      ++identity_calls;
      return "fake-sidecar";
    }
  };
  const std::vector<TextRunnerToken> prompt{10};
  auto disabled = std::make_shared<IdentityRunner>();
  TextRunnerPool no_lookup(disabled, 1, {}, {.publish_lookup = false});
  assert(disabled->identity_calls == 0);
  assert(no_lookup.Lookup(prompt).epoch.empty());
  auto enabled = std::make_shared<IdentityRunner>();
  TextRunnerPool default_pool(enabled, 1);
  assert(enabled->identity_calls == 1);
  assert(default_pool.Lookup(prompt).epoch.size() == 64);
  if (!platform::CheckpointTransferRuntimeSupported()) {
    bool unsupported = false;
    try {
      TextRunnerPool transfer(std::make_shared<IdentityRunner>(), 1, {},
                              {.transfer_api = true});
    } catch (const CheckpointTransferError& error) {
      unsupported = error.status == 501 && std::string_view(error.code) ==
                                               "cache_transfer_unsupported";
    }
    assert(unsupported);
    std::cout << "SKIP: executable-bound transfer requires Linux x86-64\n";
  } else {
    auto runner = std::make_shared<IdentityRunner>();
    TextRunnerPool transfer(runner, 1, {},
                            {.publish_lookup = false, .transfer_api = true});
    assert(runner->identity_calls == 1);
    assert(transfer.Lookup(prompt).epoch.empty());
  }
}
#include "tests/cli/cache_lookup_checks.hpp"
#include "tests/cli/cache_lookup_r3_checks.hpp"
#include "tests/cli/cache_packed_transfer_checks.hpp"
}  // namespace
// An off RAM cache commits no backing, grows none before prefill and takes no
// capture, yet a session keeps its own live frontier and its metrics stay
// whole.
void RamCacheOffCommitsNoBackingAndKeepsLiveFrontiers() {
  class CountingRunner final : public Runner {
  public:
    mutable std::size_t committed{}, prepared{}, claims{};
    TextRunnerResourceClaim ResourceClaim() const override {
      ++claims;
      return Runner::ResourceClaim();
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this](std::size_t) { ++committed; };
      resources->prepare_prefill = [this](std::size_t) { ++prepared; };
      return resources;
    }
  };
  for (const bool enabled : {true, false}) {
    auto runner = std::make_shared<CountingRunner>();
    ComponentTextCache cache(runner, 2, 1 << 22, {}, false, 0, {}, enabled);
    std::vector<TextRunnerToken> conversation(12, 7);
    for (int turn = 0; turn < 2; ++turn) {
      auto lease = cache.Acquire(conversation, {}, {}, true, 0, true);
      if (!enabled) {
        assert(lease->PlannedCaptures(conversation.size()).empty());
        // Off is a policy, independent of a caller's payload estimate.
        for (const auto purpose :
             {SnapshotPurpose::kContinuation, SnapshotPurpose::kRetry,
              SnapshotPurpose::kBranchPoint, SnapshotPurpose::kHistory})
          assert(!lease->TryReserveSnapshot(0, conversation.size(), false,
                                            purpose, conversation));
      }
      const auto cached = lease->cached_tokens();
      auto& state = dynamic_cast<State&>(lease->state());
      const auto tail = std::span(conversation).subspan(cached);
      lease->PreparePrefill(tail.size());
      runner->adapter->Append(state.slot, tail, tail);
      const auto reserved = lease->TryReserveSnapshot(
          lease->CaptureBytes(), conversation.size(), false,
          SnapshotPurpose::kContinuation, conversation);
      assert(reserved == enabled);
      if (reserved)
        assert(lease->PublishSnapshot(conversation, lease->Capture(), false));
      const auto metrics = lease->Metrics();
      assert(metrics.deepest_live == cached);
      const auto json = ComponentCacheJson(metrics);
      assert(json.member_str("schema") == "component-cache-v1");
      assert(json.contains("captures") && json.contains("global_ram_bytes"));
      // The second turn continues the first turn's live frontier.
      assert(cached == (turn ? 12 : 0));
      (void)lease->Commit({}, {}, conversation);
      conversation.push_back(8);
    }
    assert(runner->committed == (enabled ? 1 : 0));
    assert(runner->prepared == (enabled ? 2 : 0));
    // The budget is measured from the runner's claim, which off never reads.
    assert((runner->claims != 0) == enabled);
    assert((runner->captures != 0) == enabled);
  }
}
// With a disk cache the off RAM cache keeps no checkpoint in RAM and commits
// only the backing disk captures copy through.
void RamCacheOffWithDiskCommitsOnlyStagingBacking() {
  const TemporaryDirectory temp("gufo-component-off-disk-test");
  const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                        .capacity_bytes = 1 << 20,
                                        .min_checkpoint_step_tokens = 1};
  class CountingRunner final : public Runner {
  public:
    mutable std::size_t committed{};
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->commit_backing = [this](std::size_t bytes) {
        committed = bytes;
      };
      return resources;
    }
  };
  const std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::uint64_t hash;
  {
    auto runner = std::make_shared<CountingRunner>();
    ComponentTextCache cache(runner, 1, 1 << 24, disk, false, 0, {}, false);
    assert(runner->committed != 0 && runner->committed <= 512ULL << 20);
    auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
    assert(!lease->TryReserveSnapshot(0, prompt.size(), false,
                                      SnapshotPurpose::kHistory, prompt));
    auto& state = dynamic_cast<State&>(lease->state());
    runner->adapter->Append(state.slot, prompt, prompt);
    hash = runner->adapter->RecurrentHash(state.slot);
    assert(lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(),
                                     false, SnapshotPurpose::kContinuation,
                                     prompt));
    auto snapshot = lease->Capture();
    assert(snapshot);
    // Nothing is retained in RAM; the checkpoint goes to disk alone.
    assert(lease->PublishSnapshot(prompt, std::move(snapshot), false) == 0);
    assert(lease->Metrics().disk_queued_bytes > 0);
    (void)lease->Commit(prompt, {}, prompt);
  }
  auto runner = std::make_shared<CountingRunner>();
  ComponentTextCache cache(runner, 1, 1 << 24, disk, false, 0, {}, false);
  auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
  assert(lease->restored_from_disk() &&
         lease->cached_tokens() == prompt.size());
  auto& state = dynamic_cast<State&>(lease->state());
  assert(runner->adapter->RecurrentHash(state.slot) == hash);
}
// With the RAM cache off, a capture within the disk step of a published
// checkpoint has nowhere to go. The cache refuses it at reservation, so no
// caller logs a failed capture. With the RAM cache on it goes to RAM.
void RamCacheOffRefusesCapturesWithinTheDiskStep() {
  for (const bool enabled : {true, false}) {
    const TemporaryDirectory temp("gufo-component-off-step-test");
    const TextRunnerDiskCacheOptions disk{.directory = temp.path().string(),
                                          .capacity_bytes = 1 << 20};
    auto runner = std::make_shared<Runner>();
    ComponentTextCache cache(runner, 2, 1 << 22, disk, false, 0, {}, enabled);
    std::vector<TextRunnerToken> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
    for (int turn = 0; turn < 2; ++turn) {
      auto lease = cache.Acquire(prompt, {}, {}, true, 0, true);
      auto& state = dynamic_cast<State&>(lease->state());
      const auto tail = std::span(prompt).subspan(lease->cached_tokens());
      runner->adapter->Append(state.slot, tail, tail);
      // The first turn has no checkpoint within the step. The second is nine
      // tokens past one, far inside the default 2048-token step.
      const auto reserved =
          lease->TryReserveSnapshot(lease->CaptureBytes(), prompt.size(), false,
                                    SnapshotPurpose::kContinuation, prompt);
      assert(reserved == (turn == 0 || enabled));
      if (reserved) {
        // Only the RAM cache retains the capture; off sends it to disk alone.
        const auto retained =
            lease->PublishSnapshot(prompt, lease->Capture(), false);
        assert((retained != 0) == enabled);
      }
      (void)lease->Commit(prompt, {}, prompt);
      prompt.push_back(99);
    }
  }
}
#if defined(GUFO_ENABLE_PREFILL_PREREQUISITES)
namespace {
void PreparedSerialCutKeepsRealComponentBoundaries() {
  auto runner = std::make_shared<Runner>();
  runner->in_pass = true;
  TextRunnerPool pool(runner, 1);
  const std::vector<TextRunnerToken> prompt(40, 7);
  auto request =
      pool.Acquire(prompt, sampling::SamplingConfig{}, {}, {}, true, 16);
  // B1/B2: a large proposed range is clipped to the actual stable boundary,
  // not published at the proposed end or through an interior capture.
  auto cut = request.PrepareSinglePassPrefill(31);
  assert(cut && *cut == 16);
  assert(request.prefill_position() == 0);  // Planning did not execute rows.
  assert(request.Prefill(*cut).consumed_tokens == 16);
  assert(request.prefill_position() == 16 && runner->through_calls == 0);
  while (!request.prefill_complete()) {
    cut = request.PrepareSinglePassPrefill(7);
    if (!cut) {
      // Join the exact capture once. It may finish between the failed poll
      // and this call, so asserting a second transient Pending flag is racy.
      // This fake's valid model plan does not decline; no sleep/retry loop.
      const auto position = request.prefill_position();
      request.CapturePromptSnapshot();
      assert(request.prefill_position() == position);
      cut = request.PrepareSinglePassPrefill(7);
    }
    assert(cut && *cut > 0 && *cut <= 7);
    assert(request.Prefill(*cut).consumed_tokens == *cut);
  }
  const auto token = request.SelectNext().token;
  (void)request.Commit();
  auto warm = pool.Acquire(prompt);
  assert(warm.cached_prompt_tokens() == prompt.size());
  assert(warm.SelectNext().token == token);
  warm.Invalidate();

  // B3/B4: a frozen cut cannot be relabelled with a different budget and
  // execute/publish under that identity. Rejection precedes component mutation.
  auto changed = prompt;
  changed.front() = 8;
  auto other = pool.Acquire(changed);
  cut = other.PrepareSinglePassPrefill(9);
  assert(cut && *cut == 9);
  bool rejected = false;
  try {
    (void)other.Prefill(10);
  } catch (const std::logic_error&) {
    rejected = true;
  }
  assert(rejected && other.prefill_position() == 0);
  other.Invalidate();
}

void AdaptiveSchedulingUsesRealComponentCache(bool capturing) {
  class AdaptiveRunner final : public Runner {
  public:
    explicit AdaptiveRunner(bool capturing) : capturing(capturing) {}
    const bool capturing;
    mutable std::binary_semaphore entered{0};
    mutable std::binary_semaphore resume{0};
    mutable std::atomic<bool> resumed{false}, timed_out{false};
    mutable std::atomic<bool> arm_capture{false};
    mutable std::mutex capture_mutex;
    mutable std::thread::id capture_thread;
    mutable bool selected{false}, blocked{false};
    mutable std::size_t advances{};
    mutable std::vector<std::pair<TextRunnerToken, std::size_t>> cuts;
    void Resume() const {
      if (!resumed.exchange(true))
        resume.release();
    }
    void Block() const {
      entered.release();
      if (!resume.try_acquire_for(std::chrono::seconds(5))) {
        timed_out = true;
        throw std::runtime_error("adaptive component fixture barrier timeout");
      }
    }
    std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
        cache::ResourceLedger& ledger, std::size_t ram) const override {
      auto resources = Runner::CreateComponentCacheResources(ledger, ram);
      resources->allocate =
          [this, allocate = resources->allocate](const auto& request) {
            auto buffer = allocate(request);
            if (arm_capture.exchange(false)) {
              const std::lock_guard lock(capture_mutex);
              capture_thread = std::this_thread::get_id();
            }
            return buffer;
          };
      resources->stream = [this, stream = resources->stream] {
        bool hold;
        {
          const std::lock_guard lock(capture_mutex);
          hold = capture_thread == std::this_thread::get_id();
          if (hold)
            capture_thread = {};
        }
        // Only the decoder's post-selection capture worker is held. All its
        // buffers are allocated, and Lease::Capture's transfer factory holds
        // neither the cache metadata mutex nor this fixture's mutex. Admission
        // and the peer slot's prefill can proceed while it is pending.
        if (hold)
          Block();
        return stream();
      };
      return resources;
    }
    TextPrefillStep Prefill(TextRunnerState& state,
                            std::span<const TextRunnerToken> prompt,
                            std::size_t offset,
                            std::size_t budget) const override {
      assert(CheckpointPosition(state) == offset);
      cuts.emplace_back(prompt.front(), budget);
      const bool first_long = prompt.front() == 22 && offset == 1;
      if (first_long) {
        if (capturing) {
          // Do not release the capture at Submit: it must stay pending until
          // the scheduler has granted and the real pool has clipped this cut.
          assert(!resumed && !timed_out && advances == 0);
          Resume();
        } else {
          assert(resumed && advances != 0);
        }
      }
      const auto step = Runner::Prefill(state, prompt, offset, budget);
      assert(step.consumed_tokens == std::min(budget, prompt.size() - offset));
      if (first_long)
        assert(step.consumed_tokens == budget);  // No hidden frontier clipping.
      assert(CheckpointPosition(state) == offset + step.consumed_tokens);
      return step;
    }
    TextDecodeSelection SelectNext(
        TextRunnerState& state,
        sampling::SamplerState& sampler) const override {
      if (capturing && !std::exchange(selected, true))
        arm_capture = true;
      return Runner::SelectNext(state, sampler);
    }
    void Advance(TextRunnerState& state, TextRunnerToken token) const override {
      if (!capturing && !std::exchange(blocked, true)) {
        // Request::Advance is reached only AFTER PreparePromptSnapshot has
        // completed. Blocking SelectNext instead leaves an async capture and
        // correctly gives the next prompt the quiet budget, not active 64.
        Block();
      }
      Runner::Advance(state, token);
      ++advances;
    }
  };
  auto runner = std::make_shared<AdaptiveRunner>(capturing);
  runner->row_blocks = 4096;
  runner->max_context = 8192;
  assert(runner->UsesComponentCache());  // CreateState throws on wrong path.
  auto pool = std::make_shared<TextRunnerPool>(runner, 2);
  {
    const std::vector<TextRunnerToken> prefix{22};
    auto warm = pool->Acquire(prefix);
    Prefill(warm);
    (void)warm.Commit();
  }
  runner->cuts.clear();
  TextGenerationScheduler scheduler(
      pool,
      {.decode_active_tokens = 512,
       .prerequisites = {.mode = detail::PrefillPrerequisiteMode::kAdaptive,
                         .tokens = 64}});
  struct ResumeOnExit {
    const AdaptiveRunner& runner;
    ~ResumeOnExit() { runner.Resume(); }
  } cleanup{*runner};  // Release before scheduler joins, including on failure.
  auto decoder = scheduler.Submit({11}, 4, 0.0f);
  if (!runner->entered.try_acquire_for(std::chrono::seconds(5)))
    throw std::runtime_error(
        "adaptive component fixture did not enter barrier");
  // One reused row leaves exactly 4096 new rows. The 2048 grid is less than
  // 2048 rows from reuse, and 4096 is within the final 128-row tail, so neither
  // is a planned checkpoint. Both budgets reach the model without clipping;
  // context and physical row backing also fit the complete prompt plus output.
  std::vector<TextRunnerToken> prompt(4097, 22);
  auto prefill = scheduler.Submit(prompt, 1, 0.0f);
  if (!capturing)
    runner->Resume();
  const auto first = decoder.Wait();
  const auto second = prefill.Wait();
  assert(!runner->timed_out);
  assert(first.tokens.size() == 4 && second.tokens.size() == 1);
  assert(second.cached_prompt_tokens == 1);
  assert(second.prefill_tokens == 4096);
  const auto first_long =
      std::find_if(runner->cuts.begin(), runner->cuts.end(),
                   [](const auto& cut) { return cut.first == 22; });
  assert(first_long != runner->cuts.end());
  if (capturing)
    assert(first_long->second == 4096);
  else
    assert(first_long->second == 64);
  // Independent cold component pool: altered cuts preserve exact state/output.
  auto cold_runner = std::make_shared<Runner>();
  cold_runner->row_blocks = 4096;
  cold_runner->max_context = 8192;
  TextRunnerPool cold(cold_runner, 1);
  auto reference = cold.Acquire(prompt, sampling::SamplingConfig{});
  while (!reference.prefill_complete())
    (void)reference.Prefill(4096);
  assert(second.tokens.front() == reference.SelectNext().token);
}
}  // namespace
#endif
int main(int argc, char* argv[]) {
  std::string_view suite;
  if (argc != 1) {
    if (argc != 3 || std::string_view(argv[1]) != "--suite") {
      std::cerr << "usage: component_text_cache_test [--suite NAME]\n";
      return 2;
    }
    suite = argv[2];
    if (suite.empty())
      return 2;
  }
  if (argc == 1 &&
      std::getenv("GUFO_TEST_ROTATED_PROCESS_LOCAL_ONLY") != nullptr) {
    packed_ingress_cpu::RotatedWithoutFingerprintDisk();
    std::cout << "CPU fingerprint-absent rotated disk admission PASS; native "
                 "NOT RUN\n";
    return 0;
  }
  if (argc == 1 && std::getenv("GUFO_TEST_PACKED_INGRESS_ONLY") != nullptr) {
    packed_ingress_cpu::CorruptedDisk();
    packed_ingress_cpu::AdmissionFailuresPreserveDisk();
    packed_ingress_cpu::MalformedDisk();
    std::cout << "CPU packed disk admission PASS\n";
    return 0;
  }
  // CTest runs bounded suites; direct no-argument execution retains every case.
  bool selected = false;
  const auto run = [&](std::string_view name, auto body) {
    if (suite.empty() || suite == name) {
      selected = true;
      body();
    }
  };
#if defined(GUFO_ENABLE_PREFILL_PREREQUISITES)
  run("prerequisites", [] {
    PreparedSerialCutKeepsRealComponentBoundaries();
    AdaptiveSchedulingUsesRealComponentCache(false);
    AdaptiveSchedulingUsesRealComponentCache(true);
  });
#endif
  run("transfer_plain_identity", [] {
    packed_transfer_cpu::MalformedImportsAndDisk(
        models::qwen38_flash_next::KvCacheType::kQ8_0, true);
  });
  run("transfer_plain_process_local", [] {
    packed_transfer_cpu::MalformedImportsAndDisk(
        models::qwen38_flash_next::KvCacheType::kQ8_0, false);
  });
  run("transfer_rotated_identity", [] {
    packed_transfer_cpu::MalformedImportsAndDisk(
        models::qwen38_flash_next::KvCacheType::kQ8_0Rotated, true);
  });
  run("transfer_rotated_process_local", [] {
    packed_transfer_cpu::MalformedImportsAndDisk(
        models::qwen38_flash_next::KvCacheType::kQ8_0Rotated, false);
  });
  run("transfer_layout", [] {
    packed_transfer_cpu::IdentityAndImportLayouts();
    packed_transfer_cpu::IncompatibleDiskLayouts();
    packed_transfer_cpu::RotatedWithoutFingerprintImport();
  });
  run("packed_pins", [] {
#if defined(__linux__)
    packed_transfer_cpu::RotatedExistingPinsBeforeReady();
#else
    std::cout << "SKIP: registered pins require strict Linux fingerprinting\n";
#endif
  });
  run("lookup", [] {
    ContextOverflowDiagnostics();
    LookupRuntimeDefaultsAndDisabled();
    CheckpointTransferRoundTripAndConcurrentServing();
    CheckpointImportRespectsRamOff();
    LookupIsReadOnlyAndMatchesAcquire();
    LookupRespectsRamOffWithoutDisablingLiveReuse();
    LookupDoesNotWaitForAdmission();
    LookupWhileAllSlotsPrefillAndDecode();
    LookupInvalidatesBeforeLiveRetirement();
    LookupPublicationDisabled();
    LookupPublicationStress();
    LookupMeasuredRatesAndEpoch();
    LookupR3BarrierFixtures();
  });
  run("pins", [] {
#if defined(__linux__)
    MultiplePinsChooseLongestAndRestart();
    MultiplePinsAggregateDiskBudget();
    MultiplePinsAggregateRamAndIndependentLazyWarming();
    MultiplePinsExplicitReplacementAndValidation();
    MultiplePinsLazyCorruptionPreservesOtherRegistrations();
    PinnedPrefixRestartReplayAndNoRewrite();
    UnpersistedRequestCanReuseRegisteredPins();
    PinnedPrefixesRespectOffAndAutomaticRam();
    PinnedPrefixPressureIdentityAndAccounting();
    PinnedPrefixPressureIdentityAndAccounting(true);
    PinnedPrefixIdentityReplacementAndBudgets();
    PinnedPrefixCorruptionFailsClosed();
    PinnedPrefixCorruptionCannotHideBehindOrdinaryDisk();
    PinnedPrefixLazySurvivesRamRefusal();
#else
    std::cout << "SKIP: registered pins require strict Linux fingerprinting\n";
#endif
  });
  run("lifecycle", [] {
    PublishedResidentCheckpointKeepsDiskStep();
    CancelledPrefillKeepsCompletedSteps();
    SchedulerSharesAnExactPendingFrontier();
    SharedPeerReusesTheCurrentStableCapture();
    BackingPreparationFollowsCaptures();
    PrefillLookaheadReachesTheNextFrontier();
    PartialFrontierDoesNotServeAnEqualPrompt();
    CompletedCancelledPromptGetsARetrySnapshot();
    BudgetUsesAllocatedSlotsInsteadOfLazyStateClaim();
    InPassCapturePreservesBoundaryAndLiveTail();
    ColdAcquireResetsOnce();
    ColdAssignmentAvoidsDrainingCancelledSlot();
    ColdAssignmentAvoidsDrainingCancelledSlot(true);
    ColdAssignmentKeepsOneDuplicateDeepFrontier();
    FailedRestoreResetsOnce();
    PartialReuseDiagnostics();
    RestoreFromBusyBorrowedSource();
    ImageAfterCachedTextPrefixCaptures();
  });
  run("reuse", [] {
    ColdAssignmentPrefersPreservedSlot();
    ColdAssignmentKeepsLiveFrontiers();
    ColdSideRequestsKeepPreservedDeepFrontier();
    ReuseAndBranches();
    EditBoundariesSurviveActualBackingPressure();
    ReusedConversationOutlivesOneUseRequests();
    RestartAndCorruption();
    FailedCaptureKeepsPublishedBoundary();
    BackingPressureKeepsSharedPrefix();
    BackingPressureKeepsSharedPrefix(true);
    FragmentedBackingKeepsCompletePrompt();
  });
  run("disk", [] {
    UnpersistedRequestStaysOffDisk();
    UnpersistedOverRamCaptureIsRefused();
    RotatedDiskAdmissionFinishesBeforePublication();
    packed_ingress_cpu::CorruptedDisk();
    packed_ingress_cpu::AdmissionFailuresPreserveDisk();
    DiskBudgetEvictsAnUnpinnedConversation();
    LoweredDiskBudgetEvictsOldestAtStartup();
    FilesystemFloorBlocksWritesAndReclaimsAtStartup();
    OrphansLeaveBeforeCheckpoints();
    StagingIsInternal();
    StagingCountsAgainstRamCeiling();
    PinnedSlotBytesCountOnce();
    UnpinnedRunnersKeepTheirLimit();
    DiskDirectoryHasOneOwner();
    GridDoesNotSuppressFirstDiskPrompt();
    NearTailDiskBranchSurvivesRestart();
    UnbackedDeepRowsSurviveDiskRestart();
    RamRefusalDiskDirectRestartContract();
  });
  run("ram", [] {
    OverRamCaptureStagesPrivateStateUnderPressure(false);
    OverRamCaptureStagesPrivateStateUnderPressure(true);
    RamCacheOffCommitsNoBackingAndKeepsLiveFrontiers();
    RamCacheOffWithDiskCommitsOnlyStagingBacking();
    RamCacheOffRefusesCapturesWithinTheDiskStep();
  });
  if (!selected) {
    std::cerr << "unknown component cache suite: " << suite << '\n';
    return 2;
  }
}
