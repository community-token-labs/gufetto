// Use the actual private serving factory/state, not a second runner or forward.
// This translation unit replaces (does not also link) inference_backend.cpp.
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

#include "src/cli/serve/inference_backend.cpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"

void LimitReportedDeviceFree(std::size_t);
namespace {
namespace qfn = gufo::models::qwen38_flash_next;
namespace server = gufo::server;
using gufo::json::Value;
constexpr std::uint32_t kContext = 4352;
void Require(bool ok, std::string_view error) {
  if (!ok)
    throw std::runtime_error(std::string(error));
}
bool Equal(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}
std::vector<std::int32_t> Input(const qfn::Model& model, std::size_t count) {
  const auto seed =
      model.Tokenize("Qualification fixture: red 17 blue 29 green 43.\n");
  Require(!seed.empty(), "empty fixture tokenization");
  std::vector<std::int32_t> ids(count);
  for (std::size_t i = 0; i < count; ++i)
    ids[i] = seed[i % seed.size()];
  return ids;
}
std::shared_ptr<server::TextRunnerPool> Pool(
    std::shared_ptr<qfn::Model> model) {
  auto runner = std::make_shared<server::QwenFlashNextTextRunner>(
      model, kContext, false, 0);
  server::TextRunnerRamCacheOptions ram;
  ram.enabled = false;
  return std::make_shared<server::TextRunnerPool>(runner, 1, std::nullopt, ram);
}
std::vector<float> Ordinary(const std::shared_ptr<qfn::Model>& model,
                            std::span<const std::int32_t> input) {
  std::string error;
  auto session = model->CreateSession(gufo::core::SessionMode::kAutoregressive,
                                      kContext, &error);
  Require(bool(session), error);
  Require(session->Sync(input, &error), error);
  return {session->Logits().begin(), session->Logits().end()};
}
std::vector<float> Capture(const std::shared_ptr<server::TextRunnerPool>& pool,
                           std::span<const std::int32_t> input, unsigned rows) {
  auto result = server::ScoreColdOnWorker(pool, input, rows);
  return {result.logits().begin(), result.logits().end()};
}
Value Difference(std::span<const float> candidate,
                 std::span<const float> reference) {
  Require(candidate.size() == reference.size(), "comparison shape mismatch");
  double absolute = 0, relative = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    Require(std::isfinite(candidate[i]) && std::isfinite(reference[i]),
            "nonfinite native logits");
    const double delta = std::abs(double(candidate[i]) - reference[i]);
    absolute = std::max(absolute, delta);
    // Explicit denominator floor, not a numerical acceptance threshold.
    relative = std::max(
        relative, delta / std::max(std::abs(double(reference[i])), 1e-30));
  }
  auto out = Value::object();
  out["max_abs"] = absolute;
  out["max_rel"] = relative;
  out["relative_denominator_floor"] = 1e-30;
  out["byte_equal"] = Equal(candidate, reference);
  out["argmax_equal"] =
      std::max_element(candidate.begin(), candidate.end()) -
          candidate.begin() ==
      std::max_element(reference.begin(), reference.end()) - reference.begin();
  return out;
}
struct EmptyGuard : gufo::cache::MutationGuard {
  void BeforeOverwrite(gufo::cache::ComponentId, gufo::cache::Rows,
                       gufo::cache::Rows) override {}
  void BeforeRelease(gufo::cache::ComponentId, gufo::cache::Rows,
                     gufo::cache::Rows) noexcept override {}
};
// Production ordinary budgets cannot put a <=8-row suffix across the 128-row
// tail allowance. Exercise the existing adapter's budget API directly, with
// identical consumed chunks in the controls. No alternate forward is used.
std::vector<float> Boundary(const std::shared_ptr<qfn::Model>& model,
                            std::span<const std::int32_t> input, unsigned rows,
                            std::vector<std::uint32_t>& ends) {
  qfn::ContinuationAdapter adapter(
      model, gufo::core::SessionMode::kAutoregressive, kContext, {0x71});
  EmptyGuard guard;
  auto slot = adapter.CreateSlot(guard);
  std::vector<float> output(std::size_t(rows) * model->VocabSize());
  bool started = false;
  std::exception_ptr failure;
  try {
    adapter.BeginDiagnostic(*slot, input, rows, output, started);
    for (const auto budget : {model->PrefillCapacity(), 128U, 4U}) {
      const auto step = adapter.DiagnosticPrefillStep(*slot, budget);
      ends.push_back(step.pass_end);
      if (step.complete)
        break;
    }
    Require(!ends.empty() && ends.back() == input.size(),
            "boundary capture incomplete");
  } catch (...) {
    failure = std::current_exception();
  }
  // Never unwind live GPU/PLE state through the fixture's local owners.
  if ((started && !adapter.DrainDiagnosticCancellation(*slot)) ||
      adapter.DiagnosticReset(*slot) !=
          gufo::cache::DiagnosticDiscardResult::kComplete) {
    std::cerr
        << "unsafe native boundary cleanup; exiting without owner teardown\n"
        << std::flush;
    std::_Exit(2);
  }
  if (failure)
    std::rethrow_exception(failure);
  return output;
}
Value RunCapture(const std::shared_ptr<qfn::Model>& model) {
  const auto short_input = Input(*model, 17);
  const auto reference =
      Ordinary(model, short_input);  // Separate session lifetime.
  std::vector<float> multi;
  {
    auto pool = Pool(model);
    const auto single = Capture(pool, short_input, 1);
    Require(Equal(single, reference),
            "single-row capture differs from ordinary frontier");
    multi = Capture(pool, short_input, 8);
  }
  auto out = Value::object();
  out["last_row_equal"] = true;
  out["l5"] = Difference(std::span(multi).last(model->VocabSize()), reference);
  // L5 is measured, not a silently invented tolerance/bit-identity claim.
  const auto input = Input(*model, model->PrefillThroughCapacity() + 4);
  {
    auto pool = Pool(model);
    const auto baseline = Capture(pool, input, 8);
    struct Cancellation {
      unsigned passes{}, checks{};
    };
    auto control = std::make_shared<Cancellation>();
    bool cancelled = false;
    try {
      (void)server::ScoreColdOnWorker(
          pool, input, 8,
          [control] { return control->passes && ++control->checks == 4; },
          [control] { ++control->passes; });
    } catch (const std::exception& e) {
      cancelled =
          std::string_view(e.what()).find("cancel") != std::string_view::npos;
    }
    Require(cancelled && control->passes == 1 && control->checks == 4 &&
                pool->usable_capacity() == 1,
            "did not cancel inside the next pass and safely drain/reset");
    Require(Equal(Capture(pool, input, 8), baseline),
            "post-cancel scoring differs");
    out["cancel_recovery_equal"] = true;
    out["cancel_completed_passes"] = control->passes;
    out["cancel_native_checks"] = control->checks;
  }
  std::vector<std::uint32_t> ends, first_ends, last_ends;
  const auto across = Boundary(model, input, 8, ends);
  const auto first =
      Boundary(model, std::span(input).first(input.size() - 4), 4, first_ends);
  const auto last = Boundary(model, input, 4, last_ends);
  const auto half = std::size_t{4} * model->VocabSize();
  Require(ends.size() == 3 && ends[1] == input.size() - 4 &&
              first_ends.size() == 2 && last_ends == ends,
          "suffix did not cross the registered pass boundary");
  Require(Equal(std::span(across).first(half), first) &&
              Equal(std::span(across).last(half), last),
          "captured rows are misplaced across the pass boundary");
  out["pass_boundary_equal"] = true;
  auto ledger = Value::array();
  for (auto end : ends)
    ledger.push_back(end);
  out["boundary_pass_ends"] = std::move(ledger);
  out["vocab_size"] = model->VocabSize();
  return out;
}
}  // namespace
int main(int argc, char** argv) {
  std::string path, mode;
  for (int i = 1; i < argc; ++i) {
    const std::string_view option(argv[i]);
    if ((option != "--model" && option != "--mode") || ++i == argc)
      return 2;
    (option == "--model" ? path : mode) = argv[i];
  }
  if (path.empty()) {
    if (const auto* value = std::getenv("GUFO_QFN_COLD_SCORE_MODEL"))
      path = value;
  }
  if (path.empty()) {
    std::cerr << "missing model; NOT QUALIFIED\n";
    return 77;
  }
  if (mode != "off" && mode != "capture" && mode != "refusal")
    return 2;
  std::cout << "{\"stage\":\"readiness\",\"mode\":\"" << mode << "\"}\n"
            << std::flush;
  try {
    qfn::ModelOptions options;
    options.max_context = kContext;
    options.enable_vision = false;
    options.kv_cache_type = qfn::KvCacheType::kQ8_0;
    options.enable_diagnostic_logits = mode != "off";
    options.approve_diagnostic_logits = [](const auto& geometry,
                                           std::string* error) {
      return server::ApproveColdScoreMemory(geometry, kContext, error);
    };
    std::string error;
    auto model = qfn::Model::Load(path, options, &error);
    Require(bool(model), error);
    auto result = Value::object();
    result["schema"] = "gufo-full-logits-core-native-v1";
    result["mode"] = mode;
    result["status"] = "pass";
    if (mode == "off") {
      const auto& geometry = model->LogitCapacity();
      Require(
          !model->DiagnosticLogitsEnabled() &&
              geometry.max_logit_rows == geometry.max_speculative &&
              geometry.widening_bytes == 0 &&
              geometry.head_bytes == std::size_t(geometry.max_speculative) *
                                         model->VocabSize() * sizeof(float) &&
              server::DiagnosticMemoryClaim::Process().Snapshot().bytes == 0,
          "OFF geometry or claim changed");
      (void)Ordinary(model, Input(*model, 17));
      result["off_geometry_equal"] = true;
      result["max_logit_rows"] = geometry.max_logit_rows;
      result["max_speculative"] = geometry.max_speculative;
    } else if (mode == "capture") {
      result["capture"] = RunCapture(model);
    } else {
      const auto bytes =
          server::DiagnosticMemoryClaim::Process().Snapshot().bytes;
      Require(bytes > 0, "missing ON startup claim");
      const auto ceiling = bytes - 1;
      // Reuse the existing test-only HIP observation cap AFTER real model load.
      // Real Flash-Next factory, adapter and measured-fit admission still run;
      // this is not a synthetic adapter or a claim about physical RAM pressure.
      LimitReportedDeviceFree(gufo::hip::kDispatchReserve + ceiling);
      const auto expected = "--full-logits-api needs " + std::to_string(bytes) +
                            " bytes; snapshot ceiling is " +
                            std::to_string(ceiling) +
                            " bytes; lower --context or --sessions, free "
                            "memory, or restart without --full-logits-api";
      bool refused = false;
      try {
        auto pool = Pool(model);
      } catch (const std::invalid_argument& e) {
        Require(e.what() == expected,
                "wrong real-component refusal text: " + std::string(e.what()));
        refused = true;
      }
      Require(refused, "real component startup unexpectedly admitted E > X");
      result["status"] = "expected_refusal";
      result["required_bytes"] = static_cast<std::uint64_t>(bytes);
      result["ceiling_bytes"] = static_cast<std::uint64_t>(ceiling);
      result["error"] = expected;
      result["complete"] = true;
      std::cout << result.dump() << '\n' << std::flush;
      std::cerr << expected << '\n';
      return 42;  // Refusal must propagate to a nonzero startup/driver exit.
    }
    result["complete"] = true;
    std::cout << result.dump() << '\n' << std::flush;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "full-logits native test INVALID: " << e.what() << '\n'
              << std::flush;
    return 1;
  }
}
