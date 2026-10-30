// Explicit model-dependent native gate. Missing model is a skip, not a pass.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/measured_admission.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/models/qwen38_flash_next/continuation_adapter.hpp"
#include "tests/diagnostics/recovery_bounds.hpp"
#include "tests/diagnostics/rss_slope.hpp"

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
    const std::shared_ptr<models::qwen38_flash_next::Model>& model, Fault fault,
    bool check_recovery) {
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
  if (check_recovery)
    Require(after.threads <= before.threads, "idle workers survived teardown");
}
RecoverySample FullCycle(const char* path, bool check_recovery = true) {
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
      RealPoolCase(model, fault, check_recovery);
  }
  return Sample();
}
struct RssParts {
  std::uint64_t total = 0, anon = 0, file = 0, shmem = 0;
};
RssParts SampleRssParts() {
  std::ifstream status("/proc/self/status");
  RssParts result;
  unsigned fields = 0;
  std::string key, rest;
  while (status >> key) {
    std::uint64_t kib = 0;
    auto read = [&](std::uint64_t& value, unsigned flag) {
      Require(bool(status >> kib), "unavailable split RSS sample");
      value = kib * 1024;
      fields |= flag;
    };
    if (key == "VmRSS:")
      read(result.total, 1);
    else if (key == "RssAnon:")
      read(result.anon, 2);
    else if (key == "RssFile:")
      read(result.file, 4);
    else if (key == "RssShmem:")
      read(result.shmem, 8);
    std::getline(status, rest);
  }
  Require(fields == 15, "missing /proc/self/status RSS categories");
  return result;
}
int RunRssDiscriminator(const char* path) {
  std::vector<std::uint64_t> measured;
  measured.reserve(20);
  const auto started = std::chrono::steady_clock::now();
  const auto cycle = [&](const char* phase, int index, bool check = true) {
    const auto begin = std::chrono::steady_clock::now();
    const auto recovered = FullCycle(path, check);
    // Model is unloaded; FullCycle synchronized before this exact point.
    const auto rss = SampleRssParts();
    std::cout << "rss_cycle=" << index << " phase=" << phase
              << " rss_total=" << rss.total << " rss_anon=" << rss.anon
              << " rss_file=" << rss.file << " rss_shmem=" << rss.shmem
              << " heap=" << recovered.heap << " threads=" << recovered.threads
              << " wall_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - begin)
                     .count()
              << '\n'
              << std::flush;
    return rss.anon;
  };
  cycle("prime", 0, false);
  for (int i = 0; i < 3; ++i)
    cycle("warm", i);
  for (int i = 0; i < 20; ++i)
    measured.push_back(cycle("measured", i));
  const auto result = ClassifyRssAnon(measured);
  std::cout << std::setprecision(18)
            << "rss_window=last15 measured_cycles=20 slope_bytes_per_cycle="
            << result.slope_bytes_per_cycle
            << " growth_bytes=" << result.growth_bytes
            << " rss_file=advisory total_wall_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - started)
                   .count()
            << '\n';
  using Verdict = RssDiscriminatorResult::Verdict;
  if (result.verdict == Verdict::kFail) {
    std::cout << "FAIL: RssAnon slope > 256 KiB/cycle; CPU ASan/LSan "
                 "investigation before more host time\n";
    return 1;
  }
  if (result.verdict == Verdict::kBetween) {
    std::cout << "BETWEEN: bounded, not leak-free; low-priority follow-up\n";
    return 4;
  }
  std::cout << "PASS: last15 RssAnon slope <= 64 KiB/cycle AND growth <= 2 "
               "MiB; RssFile advisory\n";
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const bool discriminator =
      argc == 2 && std::string_view(argv[1]) == "--rss-discriminator";
  if (argc != 1 && !discriminator) {
    std::cerr << "ERROR: unknown native test arguments\n";
    return 2;
  }
  const char* path = std::getenv("GUFO_ADMISSION_MODEL");
  if (!path || !*path || !std::filesystem::exists("/proc/self/task")) {
    std::cout << "SKIP: set GUFO_ADMISSION_MODEL to the pinned Flash-Next GGUF "
                 "on Linux\n";
    return 77;
  }
  try {
    Hip(hipFree(nullptr));
    if (discriminator)
      return RunRssDiscriminator(path);
    const auto cold = Sample();
    Print("cold_runtime_only", cold);
    const auto started = std::chrono::steady_clock::now();
    std::vector<RecoverySample> all_cycles, controls, measured;
    all_cycles.reserve(6);
    controls.reserve(3);
    measured.reserve(2);
    const auto run_cycle = [&](const char* phase, bool check_recovery = true) {
      const auto begin = std::chrono::steady_clock::now();
      const auto recovered = FullCycle(path, check_recovery);
      const auto wall_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - begin)
              .count();
      std::cout << "cycle=" << all_cycles.size() << " phase=" << phase
                << " wall_ms=" << wall_ms << '\n';
      Print("model_unloaded", recovered);
      all_cycles.push_back(recovered);
      std::cout << std::flush;
      return recovered;
    };
    run_cycle("prime", false);  // Prime ALL paths before recovery assertions.
    for (int cycle = 0; cycle < 3; ++cycle)
      controls.push_back(run_cycle("warm_control"));
    // Four system pages (16 KiB on the pinned 4 KiB-page Linux host) cover
    // page-granular RSS/heap accounting and small allocator/logger tails.
    // This is a preregistered slack, not a leak allowance or retry-time knob.
    const auto page_bytes = platform::SystemPageSize();
    const RecoveryBounds bounds(controls, 4 * std::uint64_t(page_bytes));
    std::cout << "observed_heap_variance=" << bounds.heap_max - bounds.heap_min
              << " observed_rss_variance=" << bounds.rss_max - bounds.rss_min
              << " system_page_bytes=" << page_bytes
              << " tolerance_pages=4 tolerance_bytes=" << bounds.tolerance_bytes
              << " cold-to-prime drift retained for attribution, not a leak "
                 "verdict\n";
    for (int cycle = 0; cycle < 2; ++cycle)
      measured.push_back(run_cycle("measured"));
    // Never abort collection on a band miss: retain BOTH measured cycles.
    // A structural/HIP/ownership failure still throws ERROR immediately; do
    // not allocate another cycle after an unsafe ownership failure.
    const RecoveryTrend trend(all_cycles);
    const auto outside = bounds.OutsideCount(measured);
    std::cout << "cycles=" << all_cycles.size() << " out_of_band=" << outside
              << " rss_monotonic_growth=" << trend.rss_monotonic_growth
              << " heap_monotonic_loss=" << trend.heap_monotonic_loss
              << " total_wall_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - started)
                     .count()
              << '\n';
    if (outside) {
      std::cout << "INCONCLUSIVE: ownership checks passed; recovery outside "
                   "warm post-cycle band; retain all cycles, do not widen\n";
      return kRecoveryInconclusiveExit;
    }
    std::cout
        << "PASS: real-pool ownership checks; recovery inside finite warm "
           "post-cycle band; not leak-free/workload qualification\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return discriminator ? 2 : 1;
  }
}
