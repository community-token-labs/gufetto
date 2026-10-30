// Explicit model-dependent native gate. Missing model is a skip, not a pass.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/measured_admission.hpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"
#include "tests/diagnostics/recovery_bounds.hpp"

using namespace gufo;
using namespace gufo::server;
namespace {
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}
std::uint64_t Threads() {
  return std::distance(std::filesystem::directory_iterator("/proc/self/task"),
                       std::filesystem::directory_iterator{});
}
RecoverySample Sample() {
  Hip(hipDeviceSynchronize());
  Hip(hipGetLastError());
  std::size_t free = 0, total = 0;
  Hip(hipMemGetInfo(&free, &total));
  std::ifstream status("/proc/self/status");
  std::string key, rest;
  while (status >> key) {
    if (key == "VmRSS:") {
      std::uint64_t kib = 0;
      Require(bool(status >> kib), "unavailable RSS sample");
      return {free, kib * 1024, Threads()};
    }
    std::getline(status, rest);
  }
  throw std::runtime_error("no RSS sample");
}
void Print(const char* phase, const RecoverySample& s) {
  std::cout << phase << " heap=" << s.heap << " rss=" << s.rss
            << " process_threads=" << s.threads << '\n';
}
enum class Fault {
  kNone,
  kRefuse,
  kRequestedThrow,
  kProbeThrow,
  kProbeTimeout
};
struct Trace {
  bool probing = false;
  std::size_t observations = 0, requested_slots = 0, probe_slots = 0,
              probes = 0;
  std::uint64_t requested_threads = 0, probe_threads = 0;
  std::chrono::steady_clock::duration offset{};
};
struct ClearHook {
  ~ClearHook() { server::testing::admission_resources = {}; }
};

void RealPoolCase(
    const std::shared_ptr<models::qwen38_flash_next::Model>& model,
    Fault fault) {
  const auto before = Sample();
  auto trace = std::make_shared<Trace>();
  ClearHook clear;
  server::testing::admission_resources =
      [&, trace](ComponentCacheResources& resources) {
        Require(model->HasComponentBackend(),
                "fault hook ran before model binding");
        auto& startup = *resources.startup_admission;
        const auto observe = startup.observe;
        const auto now = startup.now;
        const auto make_probe = startup.make_probe;
        startup.now = [trace, now] { return now() + trace->offset; };
        startup.make_probe = [trace, make_probe](std::size_t context) {
          trace->probing = true;
          trace->observations = 0;
          ++trace->probes;
          auto adapter = make_probe(context);
          Require(dynamic_cast<models::qwen38_flash_next::ContinuationAdapter*>(
                      adapter.get()),
                  "probe replaced the real model adapter");
          return adapter;
        };
        startup.observe = [&, trace, observe, fault] {
          auto capacity =
              observe();  // Real device/host query, after real slot allocation.
          Require(model->HasComponentBackend(), "model binding was not staged");
          const auto slots = trace->observations++;
          auto& maximum =
              trace->probing ? trace->probe_slots : trace->requested_slots;
          maximum = std::max(maximum, slots);
          auto& threads =
              trace->probing ? trace->probe_threads : trace->requested_threads;
          threads = std::max(threads, Threads());
          if (!trace->probing && slots >= 2 && fault != Fault::kNone) {
            if (fault == Fault::kRequestedThrow)
              throw std::runtime_error(
                  "injected after TWO real requested slots/idle workers");
            // Fault injection lowers capacity only AFTER two real entries
            // exist. No OOM hunting, extra allocation, or fabricated positive
            // fit.
            return core::ExecutionCapacityObservation{0, 0, 0};
          }
          if (trace->probing) {
            if (fault == Fault::kProbeThrow && slots >= 1)
              throw std::runtime_error(
                  "injected after real probe adapter + staged trial slot");
            if (fault == Fault::kProbeTimeout && slots >= 2)
              trace->offset += std::chrono::seconds(31);
            if (fault == Fault::kRefuse && slots >= 2)
              return core::ExecutionCapacityObservation{0, 0, 0};
          }
          return capacity;
        };
      };
  bool accepted = false, capacity_refused = false;
  std::string error;
  {
    InferenceBackend backend(InferenceBackend::StartupMode::kAdmissionOnly);
    accepted = backend.load(model, &error, 32768, 3);
    capacity_refused = backend.measured_capacity_refused();
    Require(!backend.scheduler_started() && !backend.ready(),
            "scheduler started");
    Require(
        backend.session_states().empty() && backend.count_tokens("unused") == 0,
        "admission-only published a generation backend");
    Require(!model->HasComponentBackend(),
            "binding survived admission-only return");
  }
  const auto after = Sample();
  Print("case_before", before);
  Print("case_after", after);
  std::cout << "fault=" << int(fault)
            << " requested_slots=" << trace->requested_slots
            << " probe_slots=" << trace->probe_slots
            << " real_probe_adapters=" << trace->probes
            << " requested_threads_peak=" << trace->requested_threads
            << " probe_threads_peak=" << trace->probe_threads
            << " refusal=" << error << '\n';
  Require(accepted == (fault == Fault::kNone),
          "unexpected native admission outcome: " + error);
  Require(capacity_refused ==
              (fault == Fault::kRefuse || fault == Fault::kProbeTimeout),
          "capacity refusal was misclassified");
  Require(trace->requested_slots >= 2 &&
              trace->requested_threads >= before.threads + 2,
          "test did not stage two real requested slots AND idle workers");
  if (fault == Fault::kRefuse || fault == Fault::kProbeThrow ||
      fault == Fault::kProbeTimeout) {
    const auto required = fault == Fault::kProbeThrow ? 1U : 2U;
    Require(trace->probes > 0 && trace->probe_slots >= required &&
                trace->probe_threads >= before.threads + required,
            "test did not stage real probe adapter/slots/workers");
  }
  Require(after.threads <= before.threads, "idle workers survived teardown");
}
RecoverySample FullCycle(const char* path) {
  {
    std::string error;
    auto model = models::qwen38_flash_next::Model::Load(
        path,
        {.max_context = 32768, .decode_concurrency = 3, .enable_vision = false},
        &error);
    Require(bool(model), error);
    for (const auto fault :
         {Fault::kNone, Fault::kRefuse, Fault::kRequestedThrow,
          Fault::kProbeThrow, Fault::kProbeTimeout})
      RealPoolCase(model, fault);
  }
  return Sample();
}
}  // namespace

int main() {
  const char* path = std::getenv("GUFO_ADMISSION_MODEL");
  if (!path || !*path || !std::filesystem::exists("/proc/self/task")) {
    std::cout << "SKIP: set GUFO_ADMISSION_MODEL to the pinned Flash-Next GGUF "
                 "on Linux\n";
    return 77;
  }
  try {
    Hip(hipFree(nullptr));
    const auto cold = Sample();
    Print("cold_runtime_only", cold);
    Print("primed_load_unload",
          FullCycle(path));  // Prime ALL real paths, not just HIP init.
    std::vector<RecoverySample> controls;
    for (int sample = 0; sample < 8; ++sample) {
      controls.push_back(Sample());
      Print("post_prime_control", controls.back());
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const RecoveryBounds bounds(controls);
    std::cout << "observed_heap_variance=" << bounds.heap_max - bounds.heap_min
              << " observed_rss_variance=" << bounds.rss_max - bounds.rss_min
              << " cold-to-prime drift retained for attribution, not a leak "
                 "verdict\n";
    for (int cycle = 0; cycle < 2; ++cycle) {
      const auto recovered = FullCycle(path);
      Print("model_unloaded", recovered);
      Require(bounds.Contains(recovered),
              "heap/RSS/threads outside measured warm controls: unresolved, do "
              "not widen margins");
    }
    std::cout << "Real-pool ownership/heap/RSS recovery PASS; not workload "
                 "qualification\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
