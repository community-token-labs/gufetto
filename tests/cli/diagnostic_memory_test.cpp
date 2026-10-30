#include "src/cli/serve/diagnostic_memory.hpp"

#include <atomic>
#include <barrier>
#include <cassert>
#include <iostream>
#include <sstream>
#include <thread>

#include "src/core/platform/env.hpp"
#include "src/core/reclaimable_memory.hpp"

using gufo::server::DiagnosticMemoryClaim;
using gufo::server::Logger;
using gufo::server::LogLevel;

namespace {
constexpr std::size_t kEnvelope = 1 << 20;
constexpr gufo::core::ExecutionCapacityObservation kCapacity{
    4 * kEnvelope, 4 * kEnvelope, 2 * kEnvelope};
constexpr std::string_view kDuplicate =
    "full-logits diagnostics can be activated only once per process; "
    "restart Gufo to load another diagnostic Model";
constexpr std::string_view kLate =
    "full-logits diagnostics must be activated before cache admission; "
    "restart Gufo with --full-logits-api";

struct CaptureLog {
  std::ostringstream text;
  std::streambuf* previous = std::clog.rdbuf(text.rdbuf());
  ~CaptureLog() { std::clog.rdbuf(previous); }
};

void RefusalLogs() {
  for (const auto level : {LogLevel::kInfo, LogLevel::kError}) {
    Logger::SetLevel(level);
    DiagnosticMemoryClaim first;
    assert(first.Activate(kEnvelope, kCapacity, nullptr));
    {
      CaptureLog log;
      std::string error;
      assert(!first.Activate(kEnvelope, kCapacity, &error));
      assert(error == kDuplicate);
      assert(log.text.str().ends_with("[ERROR] [memory] " + error + "\n"));
    }
    DiagnosticMemoryClaim late;
    assert(late.SealStartup() == 0);
    {
      CaptureLog log;
      std::string error;
      assert(!late.Activate(kEnvelope, kCapacity, &error));
      assert(error == kLate);
      assert(log.text.str().ends_with("[ERROR] [memory] " + error + "\n"));
      assert(late.Snapshot().bytes == 0);
    }
  }
}

void BoundsAndRetention() {
  DiagnosticMemoryClaim claim;
  std::string error;
  assert(!claim.Activate(0, kCapacity, &error));
  assert(error == "full-logits workspace omits its static claim record");
  assert(!claim.Activate(kEnvelope, {kEnvelope - 1, kEnvelope, 0}, &error));
  assert(error ==
         "full-logits diagnostic workspace exceeds available startup memory");
  assert(!claim.Activate(kEnvelope, {kEnvelope, kEnvelope - 1, 0}, &error));
  assert(claim.Snapshot().bytes == 0);
  assert(claim.Activate(kEnvelope, {kEnvelope, kEnvelope, 0}, &error));
  // Neither repeated binding/admission nor unknown retirement releases bytes.
  assert(claim.SealStartup() == kEnvelope);
  assert(claim.SealStartup() == kEnvelope);
  CaptureLog log;
  claim.ReportUnknown("mock_hip_free");
  assert(claim.Snapshot().bytes == kEnvelope);
  assert(claim.Snapshot().unknown_outcomes == 1);
  assert(log.text.str().ends_with(
      "[ERROR] [memory] event=full_logits_memory_unknown claim_bytes=1048576 "
      "reason=mock_hip_free\n"));
  assert(!claim.Activate(kEnvelope, kCapacity, &error));
  assert(error == kDuplicate);
}

void GrowthAndOffParity() {
  using namespace gufo::core;
  DiagnosticMemoryClaim off;
  assert(off.SealStartup() == 0);
  for (const std::size_t available : {0, 8, 64, 128})
    for (const std::size_t ordinary : {0, 4, 64})
      for (const std::size_t bytes : {0, 1, 32, 128}) {
        // Independent old memory-null predicate, including insufficient
        // ordinary headroom and exact boundaries. No new resolver as oracle.
        const bool legacy =
            available >= ordinary && bytes <= available - ordinary;
        assert(FitsMemoryBudget(available, bytes, off.PendingBytes(ordinary)) ==
               legacy);
        const auto memory = std::make_shared<ReclaimableMemory>();
        memory->Configure(0, [&] {
          const auto protected_bytes = off.PendingBytes(ordinary);
          return available > protected_bytes ? available - protected_bytes : 0;
        });
        const auto old_headroom =
            available > ordinary ? available - ordinary : 0;
        assert(bool(memory->TryAcquireGrowth(bytes)) ==
               (bytes <= old_headroom));
      }
  DiagnosticMemoryClaim on;
  assert(on.Activate(kEnvelope, kCapacity, nullptr));
  const auto protected_bytes = on.PendingBytes(32);
  assert(protected_bytes == kEnvelope + 32);
  assert(FitsMemoryBudget(kEnvelope + 100, 68, protected_bytes));
  assert(!FitsMemoryBudget(kEnvelope + 100, 69, protected_bytes));
  auto memory = std::make_shared<ReclaimableMemory>();
  memory->Configure(0, [&] { return kEnvelope + 100 - on.PendingBytes(32); });
  auto first = memory->TryAcquireGrowth(64);
  assert(first && !memory->TryAcquireGrowth(5));
  first.reset();
  assert(memory->TryAcquireGrowth(68));
  assert(on.Snapshot().bytes == kEnvelope);
  try {
    (void)on.PendingBytes(std::numeric_limits<std::size_t>::max());
    assert(false);
  } catch (const std::overflow_error&) {
  }
}

void StartupRace() {
  for (int attempt = 0; attempt < 100; ++attempt) {
    DiagnosticMemoryClaim claim;
    std::barrier start(2);
    bool activated = false;
    std::size_t admitted = 0;
    std::jthread activation([&] {
      start.arrive_and_wait();
      activated = claim.Activate(kEnvelope, kCapacity, nullptr);
    });
    std::jthread admission([&] {
      start.arrive_and_wait();
      admitted = claim.SealStartup();
    });
    activation.join();
    admission.join();
    assert(admitted == (activated ? kEnvelope : 0));
    assert(claim.Snapshot().bytes == admitted);
    assert(!claim.Activate(kEnvelope, kCapacity, nullptr));
  }
}
}  // namespace

int main() {
  gufo::platform::SetEnv("NO_COLOR", "1");
  RefusalLogs();
  BoundsAndRetention();
  GrowthAndOffParity();
  StartupRace();
  assert(DiagnosticMemoryClaim::Process().Snapshot().bytes == 0);
}
