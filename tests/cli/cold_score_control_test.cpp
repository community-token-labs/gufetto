#include "cold_score_fixture.inc"
#include "src/cli/serve/cold_score.hpp"
#include "src/cli/serve/diagnostic_memory.hpp"

namespace {
struct ColdControl {
  bool pending{}, drain_ok{true}, nonfinite{};
  std::size_t drains{}, destroyed{};
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
    control->pending = false;
    return true;
  }
  void ScoreCold(std::span<const std::int32_t> input, std::uint32_t rows,
                 std::span<float> output, std::span<std::uint32_t> ends,
                 std::size_t& passes, const CancellationCheck& cancelled,
                 const std::function<void()>& progress) override {
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
int main() {
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
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8, [] { return true; }); });
  assert(runner->control->drains ==
         drains);  // Never started, nothing borrowed.
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 0); });
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 9); });
  input[0] = 3;
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  input[0] = 0;
  // Unknown drain retains actual buffers, native facade, pool and exclusion.
  runner->control->drain_ok = false;
  const auto destroyed = runner->control->destroyed;
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  assert(runner->control->destroyed == destroyed &&
         pool->usable_capacity() == 0);
  Reject([&] { (void)ScoreColdOnWorker(pool, input, 8); });
  std::cout
      << "cold_score_control_test passed (real pool, fake native passes)\n";
}
