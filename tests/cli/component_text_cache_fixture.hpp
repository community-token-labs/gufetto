#ifndef GUFO_TESTS_CLI_COMPONENT_TEXT_CACHE_FIXTURE_HPP_
#define GUFO_TESTS_CLI_COMPONENT_TEXT_CACHE_FIXTURE_HPP_
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
#include <future>
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
#include "src/cli/serve/component_text_cache.hpp"
#include "src/cli/serve/diagnostic_memory.hpp"
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
  CancellationCheck* native_cancel{};
  std::size_t* destructions{};
  ~State() override {
    if (destructions)
      ++*destructions;
    if (native_cancel)
      *native_cancel = {};
  }
  void SetCancellationCheck(const CancellationCheck& check) override {
    if (native_cancel)
      *native_cancel = check;
  }
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
}  // namespace
#endif
