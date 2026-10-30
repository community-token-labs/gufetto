// Explicit native target only. Never part of hosted model-free CTest.
#include <hip/hip_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/cli/serve/measured_admission.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/cache/fake_adapter.hpp"

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
std::size_t Free() {
  Hip(hipDeviceSynchronize());
  std::size_t free = 0, total = 0;
  Hip(hipMemGetInfo(&free, &total));
  return free;
}
struct NativeSlot {
  void* allocation{};
  std::size_t& live;
  explicit NativeSlot(std::size_t& live) : live(live) {
    Hip(hipMalloc(&allocation, 1ULL << 20));
    ++live;
  }
  ~NativeSlot() {
    if (hipFree(allocation) != hipSuccess)
      std::terminate();
    --live;
  }
};
// Synthetic capacity/clock, REAL HIP allocations: verifies all shared helper
// unwind paths without OOM hunting or forcing a huge native model context.
void NativeUnwind(int mode) {
  std::size_t live = 0;
  auto now = std::chrono::steady_clock::time_point{};
  bool probing = false;
  ComponentCacheResources resources;
  resources.adapter = std::make_unique<cache::testing::FakeAdapter>();
  resources.freeze_execution_liability = [] { return std::size_t{0}; };
  resources.startup_admission = std::make_unique<ComponentStartupAdmission>();
  auto& startup = *resources.startup_admission;
  startup.context = 32768;
  startup.now = [&] { return now; };
  startup.slot_bytes = [](std::size_t) { return std::size_t{1} << 20; };
  startup.observe = [&] {
    const std::size_t bytes = live && !probing ? 0 : (512ULL << 20);
    return core::ExecutionCapacityObservation{bytes, bytes, bytes / 2};
  };
  startup.pending = [](const cache::Adapter&, std::size_t) {
    return ComponentStartupPending{};
  };
  startup.make_probe = [&](std::size_t) {
    probing = true;
    return std::make_unique<cache::testing::FakeAdapter>();
  };
  startup.freeze_capacity = [](auto) {};
  bool failed = false;
  {
    std::vector<std::unique_ptr<NativeSlot>> slots;
    try {
      (void)admission::AdmitMeasuredExecution(
          resources, 32768, 2, 0, 0, slots, [&](cache::Adapter&, std::size_t) {
            auto slot = std::make_unique<NativeSlot>(live);
            if (mode == 1)
              throw std::runtime_error("injected after native allocation");
            if (probing && mode == 2)
              now += std::chrono::seconds(31);
            return slot;
          });
    } catch (const std::exception&) {
      failed = true;
    }
  }
  Require(failed && live == 0, "native failure/timeout leaked a slot");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr
        << "Usage: admission_native_test <pinned Flash-Next target GGUF>\n";
    return 2;
  }
  try {
    Hip(hipFree(nullptr));  // Runtime initialization precedes the baseline.
    const auto baseline = Free();
    for (int mode = 0; mode < 3; ++mode) {
      NativeUnwind(mode);
      Require(Free() >= baseline, "native unwind heap did not recover");
    }
    std::string error;
    {
      auto model = models::qwen38_flash_next::Model::Load(
          argv[1], {.max_context = 32768, .enable_vision = false}, &error);
      Require(bool(model), error);
      std::size_t previous = 0;
      for (int cycle = 0; cycle < 3; ++cycle) {
        {
          InferenceBackend backend(
              InferenceBackend::StartupMode::kAdmissionOnly);
          Require(backend.load(model, &error, 32768, 1), error);
          Require(!backend.scheduler_started() && !backend.ready(),
                  "scheduler started in admission-only mode");
          Hip(hipDeviceSynchronize());
        }
        Require(!model->HasComponentBackend(),
                "model binding survived pool teardown");
        const auto free = Free();
        std::cout << "cycle=" << cycle << " heap_free=" << free << '\n';
        // First admission may materialize Model-owned shared scratch; later
        // cycles must not leak. The final model destruction is checked too.
        Require(cycle == 0 || free >= previous,
                "repeated pool teardown lost heap bytes");
        previous = free;
      }
    }
    Require(Free() >= baseline,
            "model teardown heap drift: investigate, not a no-leak pass");
    // Demonstrate subsequent allocation in this same process still succeeds.
    std::size_t live = 0;
    {
      NativeSlot subsequent(live);
    }
    Require(live == 0 && Free() >= baseline,
            "post-teardown allocation failed recovery");
    std::cout
        << "Native ownership/heap recovery PASS (not workload qualification)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
