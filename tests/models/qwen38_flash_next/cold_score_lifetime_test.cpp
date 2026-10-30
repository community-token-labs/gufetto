#include <cassert>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "src/cli/serve/diagnostic_memory.hpp"
#include "src/models/qwen38_flash_next/cold_score.hpp"

namespace {
std::size_t freed_models{}, freed_executors{};
struct Executor {
  int magic{};
  ~Executor() { ++freed_executors; }
};
struct Model;
Model* newest{};
struct Model {
  Model() { newest = this; }
  bool diagnostic_approved_{};
  std::unique_ptr<Executor> executor_;
  ~Model() { ++freed_models; }
};
struct Options {
  bool enable_diagnostic_logits{};
  std::function<bool(
      const gufo::models::qwen38_flash_next::DiagnosticLogitCapacity&,
      std::string*)>
      approve_diagnostic_logits;
};
void AssignError(std::string* error, const char* message) {
  if (error)
    *error = message;
}
#include "cold_score_lifetime_body.inc"
}  // namespace
int main(int argc, char** argv) {
  assert(argc == 2);
  const bool partial = std::string_view(argv[1]) == "partial";
  std::string error;
  // OFF deletes normally, never invokes the approval, and creates no claim.
  unsigned calls{};
  Options options;
  options.approve_diagnostic_logits = [&](const auto&, std::string*) {
    ++calls;
    return true;
  };
  {
    auto off = Load(options, false, &error);
    assert(off && !off->diagnostic_approved_);
  }
  assert(freed_models == 1 && freed_executors == 1 && calls == 0);
  assert(gufo::server::DiagnosticMemoryClaim::Process().Snapshot().bytes == 0);
  options.enable_diagnostic_logits = true;
  options.approve_diagnostic_logits = [&](const auto&, std::string* out) {
    ++calls;
    *out = "exact startup refusal";
    return false;
  };
  assert(!Load(options, false, &error) && error == "exact startup refusal");
  assert(freed_models == 2 && freed_executors == 1);
  options.approve_diagnostic_logits = [&](const auto&, std::string* out) {
    ++calls;
    return gufo::server::DiagnosticMemoryClaim::Process().Activate(
        4096, {1ULL << 30, 1ULL << 30, 1ULL << 29}, out);
  };
  // Exact source approval, startup log, Create-prefix and deleter, with proxy
  // native allocation. Partial failure leaves the Executor in its Model.
  auto on = Load(options, partial, &error);
  auto* retained = newest;
  assert(bool(on) != partial);
  assert(retained->executor_ && retained->executor_->magic == 37);
  on.reset();
  assert(freed_models == 2 && freed_executors == 1);
  assert(retained->executor_ &&
         retained->executor_->magic == 37);  // detach untouched
  // A switch refuses; it neither changes nor partially destroys the old Model.
  assert(!Load(options, false, &error));
  assert(error.find("only once per process") != std::string::npos);
  assert(freed_models == 3 && freed_executors == 1 &&
         retained->executor_->magic == 37);
}
