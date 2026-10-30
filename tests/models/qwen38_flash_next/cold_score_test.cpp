#include "src/models/qwen38_flash_next/cold_score.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
using namespace gufo::models::qwen38_flash_next;
void Check(bool ok, const char* message) {
  if (!ok) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}
}  // namespace

int main() {
  const auto on = DiagnosticLogitCapacity::Plan(248320, 4224, 1, true);
  const auto off = DiagnosticLogitCapacity::Plan(248320, 4224, 1, false);
  Check(on && off, "supported geometry");
  Check(on->device_bytes == 7946240 && on->pinned_bytes == 7946240 &&
            on->device_delta == 6952960 && on->pinned_delta == 6952960,
        "independent full-vocabulary two-pool arithmetic");
  Check(off->max_logit_rows == 1 && off->device_bytes == 993280 &&
            !off->device_delta && !off->pinned_delta &&
            off->batch_device_bytes == on->batch_device_bytes,
        "OFF and ordinary batch stay at S");
  constexpr std::size_t controls = 20000;  // Explicit sizing fixture, not ABI.
  const auto total = ColdScoreWorkspaceBytes(*on, 131072, controls);
  Check(total && *total == 13905920 + 524288 + 7946240 + controls,
        "persistent native + one lazy phase; capture counted once");
  Check(!ColdScoreWorkspaceBytes(*on, 0, controls) &&
            !ColdScoreWorkspaceBytes(*on, 8192, 0),
        "missing context/control bound refuses");
  Check(ColdScoreWorkspaceBytes(*on, 200000, controls) == total,
        "typed request cap stays bounded with a larger loaded context");
  auto forged = *on;
  ++forged.batch_device_bytes;
  Check(!ColdScoreWorkspaceBytes(forged, 8192, controls),
        "inconsistent native facts refuse, not a second sizing formula");
  const auto small = DiagnosticLogitCapacity::Plan(248320, 4, 1, true);
  Check(small && small->max_logit_rows == 4 && small->device_delta == 2979840,
        "native batch below eight");
  Check(ColdScoreWorkspaceBytes(*small, 2, controls) ==
            5959680 + 8 + 1986560 + controls,
        "short histories bound both input and capture");
  const auto speculative = DiagnosticLogitCapacity::Plan(248320, 32, 9, true);
  Check(speculative && speculative->max_logit_rows == 9 &&
            !speculative->device_delta && !speculative->pinned_delta,
        "existing larger speculative pool is not shrunk or widened");
  Check(!DiagnosticLogitCapacity::Plan(0, 32, 1, true) &&
            !DiagnosticLogitCapacity::Plan(248320, 0, 1, true) &&
            !DiagnosticLogitCapacity::Plan(248320, 4, 5, true),
        "invalid native geometry refuses");
  const auto limit = std::numeric_limits<std::size_t>::max();
  const auto payload = *total - controls;
  Check(ColdScoreWorkspaceBytes(*on, 131072, limit - payload) == limit,
        "exact representable boundary");
  Check(!ColdScoreWorkspaceBytes(*on, 131072, limit - payload + 1),
        "overflow refuses before reservation/allocation");
  std::cout << "cold_score_test passed (CPU sizing only)\n";
}
