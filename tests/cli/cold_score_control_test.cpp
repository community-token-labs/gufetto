#include "src/cli/serve/cold_score.hpp"
#include "src/cli/serve/diagnostic_memory.hpp"
#include "tests/cli/component_text_cache_fixture.hpp"

namespace {
struct ColdControl {
  bool pending{}, drain_ok{true}, nonfinite{}, refuse_begin{}, keep_pending{};
  std::size_t drains{}, destroyed{}, bindings{};
};
struct ColdState final : ColdScoreState {
  FakeAdapter& adapter;
  cache::Slot& slot;
  std::shared_ptr<ColdControl> control;
  ColdState(FakeAdapter& a, cache::Slot& s, std::shared_ptr<ColdControl> c)
      : adapter(a), slot(s), control(std::move(c)) {}
  ~ColdState() override { ++control->destroyed; }
  void Invalidate() noexcept override { (void)adapter.Invalidate(slot); }
  std::uint32_t ColdVocab() const noexcept override { return 3; }
  bool DrainCold() noexcept override {
    ++control->drains;
    if (!control->drain_ok)
      return false;
    control->pending = control->keep_pending;
    return true;
  }
  void ScoreCold(std::span<const std::int32_t> input, std::uint32_t rows,
                 std::span<float> output, std::span<std::uint32_t> ends,
                 std::size_t& passes, bool& started,
                 const CancellationCheck& cancelled,
                 const std::function<void()>& progress) override {
    if (control->refuse_begin)
      throw std::invalid_argument("pre-ownership native validation refusal");
    started = true;
    struct Step {
      std::uint32_t pass_end, captured_rows;
      bool complete;
    };
    std::size_t first = 0;
    models::qwen38_flash_next::CollectColdScorePasses(
        input.size(), rows, ends, passes,
        [&] {
          if (cancelled && cancelled())
            throw std::runtime_error("cancelled");
          const auto count = std::min<std::size_t>(4, input.size() - first);
          std::vector<cache::Token> tokens(input.begin() + first,
                                           input.begin() + first + count);
          adapter.Append(slot, tokens, tokens);
          const auto captured =
              models::qwen38_flash_next::DiagnosticCaptureRows(
                  first, first + count, input.size() - rows);
          first += count;
          control->pending = first != input.size();
          return Step{std::uint32_t(first), std::uint32_t(captured),
                      first == input.size()};
        },
        progress);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t column = 0; column < 3; ++column)
        output[row * 3 + column] =
            float(input[input.size() - rows + row]) + float(column) / 4;
    if (control->nonfinite)
      output[0] = std::numeric_limits<float>::infinity();
  }
};
class ColdRunner final : public Runner {
public:
  std::shared_ptr<ColdControl> control = std::make_shared<ColdControl>();
  std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& a, cache::Slot& s) const override {
    ++control->bindings;
    return std::make_unique<ColdState>(dynamic_cast<FakeAdapter&>(a), s,
                                       control);
  }
  std::size_t CheckpointPosition(const TextRunnerState& s) const override {
    const auto& state = dynamic_cast<const ColdState&>(s);
    return state.adapter.Positions(state.slot).front().valid_rows;
  }
  void ReconcileComponentState(TextRunnerState& s,
                               std::size_t position) const override {
    assert(CheckpointPosition(s) == position);
  }
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t ram) const override {
    auto result = Runner::CreateComponentCacheResources(ledger, ram);
    result->diagnostic_ready = [c = control](const cache::Slot&) {
      return !c->pending;
    };
    result->diagnostic_reset = [this](cache::Slot& s, TextRunnerState&) {
      if (control->pending)
        return cache::DiagnosticDiscardResult::kPending;
      return adapter->Invalidate(s)
                 ? cache::DiagnosticDiscardResult::kComplete
                 : cache::DiagnosticDiscardResult::kQuarantined;
    };
    return result;
  }
};
template<class F>
void Reject(F run) {
  bool rejected = false;
  try {
    run();
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
}
}  // namespace
int main(int argc, char** argv) {
  const bool pending_discard =
      argc == 2 && std::string_view(argv[1]) == "--pending-discard";
  if (argc != 1 && !pending_discard)
    return 2;
  std::string error;
  const auto bytes = ColdScoreControlBytes() +
                     ComponentTextCache::DiagnosticRequestBytes() +
                     sizeof(ColdState) + 4096 * 4 + 8 * 3 * 4;
  assert(DiagnosticMemoryClaim::Process().Activate(
      bytes, {1ULL << 30, 1ULL << 30, 1ULL << 29}, &error));
  auto runner = std::make_shared<ColdRunner>();
  auto pool = std::make_shared<TextRunnerPool>(runner, 1);
  std::vector<std::int32_t> input{0, 1, 2, 0, 1, 2, 0, 1};
  unsigned heartbeats = 0;
  {
    auto result = ScoreColdOnWorker(pool, input, 8, {}, [&] { ++heartbeats; });
    assert(heartbeats == 2 && result.vocab() == 3 &&
           result.logits().size() == 24);
    assert(result.pass_ends().size() == 2 && result.pass_ends()[0] == 4 &&
           result.pass_ends()[1] == 8);
    assert(result.logits()[0] == 0 && result.logits()[23] == 1.5f);
    assert(runner->control->drains == 1 && runner->control->destroyed == 1);
    Reject([&] {
      (void)ScoreColdOnWorker(pool, input, 8);
    });  // Result/send owns exclusion.
  }
  // F2 regression: an ordinary exception between passes leaves lookahead
  // pending.
  Reject([&] {
    (void)ScoreColdOnWorker(pool, input, 8, {}, [] {
      throw std::runtime_error("non-cancel failure after non-final pass");
    });
  });
  assert(runner->control->drains == 2 && !runner->control->pending);
  {
    auto recovered = ScoreColdOnWorker(pool, input, 8);
    assert(recovered.logits().size() == 24);
  }
  bool cancel = false;
  Reject([&] {
    (void)ScoreColdOnWorker(
        pool, input, 8, [&] { return cancel; }, [&] { cancel = true; });
  });
  assert(!runner->control->pending);
  {
    auto recovered = ScoreColdOnWorker(pool, input, 8);
  }
  runner->control->nonfinite = true;
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  runner->control->nonfinite = false;
  {
    auto recovered = ScoreColdOnWorker(pool, input, 8);
  }
  const auto drains = runner->control->drains;
  // Not every pre-bind refusal is transient: pending native work must remain
  // untouched, with no facade bound and no cleanup/retry hiding the refusal.
  const auto pending_bindings = runner->control->bindings;
  runner->control->pending = true;
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  assert(runner->control->pending &&
         runner->control->bindings == pending_bindings &&
         runner->control->drains == drains);
  runner->control->pending = false;
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8, [] { return true; }); });
  assert(runner->control->drains ==
         drains);  // Never started, nothing borrowed.
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 0); });
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 9); });
  input[0] = 3;
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  input[0] = 0;
  // L1: pre-ownership native refusal must not drain or poison the slot/gate.
  runner->control->refuse_begin = true;
  const auto refused_bindings = runner->control->bindings;
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  assert(runner->control->drains == drains && pool->usable_capacity() == 1 &&
         runner->control->bindings == refused_bindings + 1);
  runner->control->refuse_begin = false;
  {
    auto recovered = ScoreColdOnWorker(pool, input, 8);
  }
  // L2: successful drain followed by no-mutation Pending discard quarantines,
  // just like an unknown drain. Each runs in its own process.
  runner->control->drain_ok = pending_discard;
  runner->control->keep_pending = pending_discard;
  const auto destroyed = runner->control->destroyed;
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  assert(runner->control->destroyed == destroyed &&
         pool->usable_capacity() == 0);
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  std::cout
      << "cold_score_control_test passed (real pool, fake native passes)\n";
}
