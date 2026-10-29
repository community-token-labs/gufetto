#include "src/models/qwen38_flash_next/npu_contracts.hpp"

#include <limits>
#include <stdexcept>
#include <string_view>

namespace gufo::models::qwen38_flash_next::npu {
namespace {
[[noreturn]] void Refuse(std::string_view reason) {
  throw std::runtime_error("NPU start refused: " + std::string(reason));
}
}  // namespace

ReservationRequest PlanReservation(const BackingPlan& plan) {
  // OFF is deliberately independent of stale ON configuration. No reservation,
  // validation of optional artifacts, runtime discovery or dormant allocation.
  if (plan.stage == Stage::kOff)
    return {};
  std::size_t cap{};
  switch (plan.stage) {
    case Stage::kResearch:
      cap = 128 * kMiB;
      break;
    case Stage::kShared:
      cap = 384 * kMiB;
      break;
    default:
      Refuse("invalid envelope stage");
  }
  if (!plan.overhead_bounded)
    Refuse("runtime overhead has no qualified bound");
  if (plan.duplicates)
    Refuse("duplicate backing requires separate lead approval");
  std::size_t peak{};
  for (auto bytes : {plan.weights, plan.scratch, plan.runtime, plan.staging}) {
    if (bytes > std::numeric_limits<std::size_t>::max() - peak)
      Refuse("backing byte arithmetic overflow");
    peak += bytes;
  }
  if (!peak || peak > cap)
    Refuse("planned backing exceeds envelope or is empty");
  return {cap, peak};
}

std::string ReservationLog(const ReservationRequest& request) {
  const auto mib = std::to_string(request.envelope_bytes / kMiB);
  return "NPU path: " + mib + " MiB (cache budget reduced by " + mib + ")";
}

Lifetime::Lifetime(const BackingPlan& plan, Admission& admission)
    : request_(PlanReservation(plan)), admission_(admission) {}

Lifetime::~Lifetime() {
  if (!reserved_)
    return;
  if (state_.committed_bytes || state_.pending_bytes || in_flight_ ||
      state_.quarantined) {
    Quarantine();
  } else {
    admission_.Release();
  }
}

void Lifetime::Reserve() {
  if (reserved_ || closed_)
    Refuse("reservation already acquired or closed");
  if (!request_.envelope_bytes) {
    closed_ = true;
    return;
  }
  if (!admission_.TryReserve(request_))
    Refuse("insufficient admission headroom for envelope");
  reserved_ = true;
  state_.envelope_bytes = request_.envelope_bytes;
  admission_.Update(state_);
}

void Lifetime::RequireIdle() const {
  if (!reserved_ || closed_ || in_flight_ || state_.quarantined ||
      state_.pending_bytes)
    Refuse("lifetime is not idle and admitted");
}

void Lifetime::BeginAllocation(std::size_t bytes) {
  RequireIdle();
  // Use the declared peak, not spare envelope capacity, so a changed plan must
  // be validated before start. Subtraction avoids wraparound at SIZE_MAX.
  if (!bytes || bytes > request_.planned_peak_bytes - state_.committed_bytes)
    Refuse("allocation exceeds declared peak/envelope");
  state_.pending_bytes = bytes;
  admission_.Update(state_);
}

void Lifetime::AllocationSucceeded() {
  if (!state_.pending_bytes || state_.quarantined)
    Refuse("no pending allocation to commit");
  state_.committed_bytes += state_.pending_bytes;
  state_.pending_bytes = 0;
  admission_.Update(state_);
}

void Lifetime::AllocationFailed() {
  if (!state_.pending_bytes || state_.quarantined)
    Refuse("no confirmed failed allocation to roll back");
  state_.pending_bytes = 0;
  admission_.Update(state_);
}

void Lifetime::ReleaseConfirmed(std::size_t bytes) {
  RequireIdle();
  if (bytes > state_.committed_bytes)
    Refuse("release exceeds committed backing");
  state_.committed_bytes -= bytes;
  admission_.Update(state_);
}

void Lifetime::Submit() {
  RequireIdle();
  if (!state_.committed_bytes)
    Refuse("dispatch has no backing");
  in_flight_ = true;
}

void Lifetime::ConsumersCompleted() {
  if (!in_flight_ || state_.quarantined)
    Refuse("no healthy dispatch awaiting consumer completion");
  in_flight_ = false;
}

void Lifetime::Quarantine() noexcept {
  if (!reserved_)
    return;
  state_.quarantined = true;
  // A pending allocation may already exist physically. Charge it as committed
  // on uncertainty and keep the entire reservation; never retry or reuse it.
  state_.committed_bytes += state_.pending_bytes;
  state_.pending_bytes = 0;
  admission_.Update(state_);
}

void Lifetime::Close() {
  if (closed_)
    return;
  RequireIdle();
  if (state_.committed_bytes)
    Refuse("backing has not been confirmed freed");
  admission_.Release();
  state_ = {};
  reserved_ = false;
  closed_ = true;
}

bool Lifetime::CanCheckpoint() const noexcept {
  return !in_flight_ && !state_.pending_bytes && !state_.quarantined;
}

std::string CacheIdentity(const ArithmeticPlan& plan) {
  if (!plan.enabled)
    return "npu-arithmetic-v1:off";
  std::string key = "npu-arithmetic-v1:on:";
  for (const auto* field :
       {&plan.model_quant, &plan.packing_passes, &plan.program_hash,
        &plan.instruction_hash, &plan.abi, &plan.layers_ops,
        &plan.residual_consumer, &plan.chunk_tail_routing}) {
    if (field->empty())
      Refuse("incomplete arithmetic cache identity");
    // Length framing, not delimiter concatenation: punctuation/NULs cannot
    // alias another field partition. Outer cache identity owns hashing.
    key += std::to_string(field->size()) + ":" + *field;
  }
  return key;
}

}  // namespace gufo::models::qwen38_flash_next::npu
