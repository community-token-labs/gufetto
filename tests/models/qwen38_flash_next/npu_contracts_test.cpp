#include "src/models/qwen38_flash_next/npu_contracts.hpp"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace npu = gufo::models::qwen38_flash_next::npu;
using npu::kMiB;

namespace {
struct Admission final : npu::Admission {
  bool fits{true}, held{false};
  int reserves{}, releases{}, updates{};
  npu::ReservationRequest request;
  npu::BackingState backing;
  bool TryReserve(const npu::ReservationRequest& r) override {
    ++reserves;
    assert(!held);
    if (!fits)
      return false;
    request = r;
    held = true;
    return true;
  }
  void Update(const npu::BackingState& s) noexcept override {
    assert(held);
    assert(s.committed_bytes <= s.envelope_bytes);
    assert(s.pending_bytes <= s.envelope_bytes - s.committed_bytes);
    backing = s;
    ++updates;
  }
  void Release() noexcept override {
    assert(held);
    held = false;
    ++releases;
  }
};

template<typename F>
void Refused(F&& f, const char* fragment) {
  bool refused = false;
  try {
    f();
  } catch (const std::runtime_error& e) {
    const std::string message(e.what());
    assert(message.starts_with("NPU start refused: "));
    assert(message.find(fragment) != std::string::npos);
    refused = true;
  }
  assert(refused);
}

npu::BackingPlan Research() {
  return {npu::Stage::kResearch,
          45 * kMiB / 8,
          81 * kMiB / 2,
          4 * kMiB,
          8 * kMiB,
          0,
          true};
}

void Plans() {
  auto p = Research();
  auto r = npu::PlanReservation(p);
  assert(r.envelope_bytes == 128 * kMiB);
  assert(r.planned_peak_bytes == 465 * kMiB / 8);
  assert(npu::ReservationLog(r) ==
         "NPU path: 128 MiB (cache budget reduced by 128)");
  p.stage = npu::Stage::kShared;
  p.weights = 270 * kMiB;
  r = npu::PlanReservation(p);
  assert(p.weights + p.scratch == 621 * kMiB / 2);
  assert(r.envelope_bytes == 384 * kMiB);
  assert(r.planned_peak_bytes == 645 * kMiB / 2);
  p.stage = npu::Stage::kResearch;
  Refused([&] { (void)npu::PlanReservation(p); }, "exceeds");
  p = Research();
  p.duplicates = 1;
  Refused([&] { (void)npu::PlanReservation(p); }, "duplicate");
  p.duplicates = 0;
  p.overhead_bounded = false;
  Refused([&] { (void)npu::PlanReservation(p); }, "qualified bound");
  p.overhead_bounded = true;
  p.weights = std::numeric_limits<std::size_t>::max();
  Refused([&] { (void)npu::PlanReservation(p); }, "overflow");
  p = {npu::Stage::kResearch, 128 * kMiB, 0, 0, 0, 0, true};
  assert(npu::PlanReservation(p).planned_peak_bytes == 128 * kMiB);
  ++p.weights;
  Refused([&] { (void)npu::PlanReservation(p); }, "exceeds");
  p.weights = 0;
  Refused([&] { (void)npu::PlanReservation(p); }, "empty");
  p.stage = static_cast<npu::Stage>(999);
  Refused([&] { (void)npu::PlanReservation(p); }, "invalid");
}

void OffAndAdmission() {
  Admission owner;
  auto p = Research();
  p.stage = npu::Stage::kOff;
  p.weights = std::numeric_limits<std::size_t>::max();
  p.duplicates = 1;
  p.overhead_bounded = false;
  {
    npu::Lifetime off(p, owner);
    off.Reserve();
    assert(off.state().envelope_bytes == 0);
    Refused([&] { off.BeginAllocation(1); }, "not idle");
    assert(off.CanCheckpoint());
    off.Close();
  }
  assert(owner.reserves == 0 && owner.updates == 0 && owner.releases == 0);
  assert(npu::ReservationLog(npu::PlanReservation(p)) ==
         "NPU path: 0 MiB (cache budget reduced by 0)");
  owner.fits = false;
  {
    npu::Lifetime rejected(Research(), owner);
    Refused([&] { rejected.BeginAllocation(1); }, "not idle");
    Refused([&] { rejected.Reserve(); }, "headroom");
    assert(rejected.state().envelope_bytes == 0);
  }
  assert(owner.reserves == 1 && !owner.held && owner.releases == 0);
}

void Lifetimes() {
  static_assert(!std::is_copy_constructible_v<npu::Lifetime>);
  static_assert(!std::is_move_constructible_v<npu::Lifetime>);
  Admission owner;
  for (int cycle = 0; cycle < 32; ++cycle) {
    npu::Lifetime life(Research(), owner);
    life.Reserve();
    Refused([&] { life.Reserve(); }, "already");
    assert(life.state().unmaterialized_bytes() == 128 * kMiB);
    Refused([&] { life.Submit(); }, "no backing");
    Refused([&] { life.BeginAllocation(0); }, "exceeds");
    Refused(
        [&] { life.BeginAllocation(std::numeric_limits<std::size_t>::max()); },
        "exceeds");
    // Failure at each allocation position; earlier confirmed pieces remain
    // charged, and only a proven no-allocation failure rolls back pending
    // bytes.
    std::size_t live{};
    for (auto bytes : {kMiB, 2 * kMiB, 3 * kMiB}) {
      life.BeginAllocation(bytes);
      assert(!life.CanCheckpoint());
      assert(owner.backing.pending_bytes == bytes);
      assert(life.state().unmaterialized_bytes() == 128 * kMiB - live);
      Refused([&] { life.BeginAllocation(1); }, "not idle");
      life.AllocationFailed();
      assert(life.state().committed_bytes == live);
      life.BeginAllocation(bytes);
      life.AllocationSucceeded();
      live += bytes;
      assert(life.state().unmaterialized_bytes() == 128 * kMiB - live);
    }
    assert(life.CanCheckpoint());
    Refused([&] { life.Close(); }, "not been confirmed freed");
    life.Submit();
    assert(!life.CanCheckpoint());
    Refused([&] { life.Submit(); }, "not idle");
    Refused([&] { life.ReleaseConfirmed(live); }, "not idle");
    life.ConsumersCompleted();
    assert(life.CanCheckpoint());
    Refused([&] { life.ConsumersCompleted(); }, "no healthy dispatch");
    Refused([&] { life.ReleaseConfirmed(live + 1); }, "exceeds committed");
    life.ReleaseConfirmed(live);
    assert(life.CanCheckpoint());
    life.Close();
    assert(life.CanCheckpoint());
    life.Close();
    assert(!owner.held);
    assert(life.state().envelope_bytes == 0);
  }
  assert(owner.reserves == 32 && owner.releases == 32);
  {
    npu::Lifetime empty(Research(), owner);
    empty.Reserve();
  }
  assert(owner.releases == 33 && !owner.held);
}

void PeakBoundary() {
  Admission owner;
  const auto plan = Research();
  const auto peak = npu::PlanReservation(plan).planned_peak_bytes;
  npu::Lifetime life(plan, owner);
  assert(life.CanCheckpoint());
  life.Reserve();
  assert(life.CanCheckpoint());
  life.BeginAllocation(kMiB);
  life.AllocationSucceeded();
  // Both requests fit under the envelope, but only one fits the declared peak.
  Refused([&] { life.BeginAllocation(peak - kMiB + 1); }, "exceeds");
  assert(life.state().committed_bytes == kMiB);
  assert(life.state().pending_bytes == 0);
  life.BeginAllocation(peak - kMiB);
  life.AllocationSucceeded();
  assert(life.state().committed_bytes == peak);
  Refused([&] { life.BeginAllocation(1); }, "exceeds");
  life.ReleaseConfirmed(peak);
  life.Close();
  assert(!owner.held && owner.releases == 1);
}

void Quarantine() {
  for (int scenario = 0; scenario < 3; ++scenario) {
    Admission owner;
    {
      npu::Lifetime life(Research(), owner);
      life.Reserve();
      life.BeginAllocation(kMiB);
      if (scenario != 0)
        life.AllocationSucceeded();
      if (scenario == 2)
        life.Submit();
      life.Quarantine();
      life.Quarantine();
      assert(life.state().committed_bytes == kMiB);
      assert(life.state().pending_bytes == 0);
      assert(life.state().quarantined);
      assert(!life.CanCheckpoint());
      Refused([&] { life.Close(); }, "not idle");
      Refused([&] { life.ReleaseConfirmed(kMiB); }, "not idle");
      Refused([&] { life.BeginAllocation(1); }, "not idle");
      Refused([&] { life.ConsumersCompleted(); }, "no healthy dispatch");
    }
    assert(owner.held && owner.releases == 0 && owner.backing.quarantined);
  }
  Admission owner;
  {
    npu::Lifetime abandoned(Research(), owner);
    abandoned.Reserve();
    abandoned.BeginAllocation(kMiB);
  }
  assert(owner.held && owner.backing.quarantined);
  assert(owner.backing.committed_bytes == kMiB && owner.releases == 0);
}

void Identity() {
  npu::ArithmeticPlan plan{true,
                           "model/quant",
                           "two-pass",
                           "program-sha256",
                           "instructions-sha256",
                           "abi-1",
                           "layers/gate-up",
                           "residual-f16",
                           "4096+gpu-tail"};
  const auto key = npu::CacheIdentity(plan);
  assert(key == npu::CacheIdentity(plan));
  for (auto member :
       {&npu::ArithmeticPlan::model_quant, &npu::ArithmeticPlan::packing_passes,
        &npu::ArithmeticPlan::program_hash,
        &npu::ArithmeticPlan::instruction_hash, &npu::ArithmeticPlan::abi,
        &npu::ArithmeticPlan::layers_ops,
        &npu::ArithmeticPlan::residual_consumer,
        &npu::ArithmeticPlan::chunk_tail_routing}) {
    auto changed = plan;
    changed.*member += ":changed";
    assert(npu::CacheIdentity(changed) != key);
    (changed.*member).clear();
    Refused([&] { (void)npu::CacheIdentity(changed); }, "incomplete");
  }
  auto other = plan;
  plan.model_quant = "a:b";
  plan.packing_passes = "c";
  other.model_quant = "a";
  other.packing_passes = "b:c";
  assert(npu::CacheIdentity(plan) != npu::CacheIdentity(other));
  other.model_quant = std::string("a\0b", 3);
  assert(npu::CacheIdentity(plan) != npu::CacheIdentity(other));
  plan.enabled = false;
  assert(npu::CacheIdentity(plan) == "npu-arithmetic-v1:off");
  assert(npu::CacheIdentity(plan) != key);
}
}  // namespace

int main() {
  Plans();
  OffAndAdmission();
  Lifetimes();
  PeakBoundary();
  Quarantine();
  Identity();
}
