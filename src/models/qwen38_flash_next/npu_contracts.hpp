#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_NPU_CONTRACTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_NPU_CONTRACTS_HPP_

#include <cstddef>
#include <cstdint>
#include <string>

namespace gufo::models::qwen38_flash_next::npu {

inline constexpr std::size_t kMiB = 1024 * 1024;
enum class Stage { kOff, kResearch, kShared };

// All fields count unique physical backing, including alignment. Imported
// aliases add no bytes. Unknown driver/runtime overhead is not a zero estimate:
// the caller must supply a qualified bound before requesting admission.
struct BackingPlan {
  Stage stage{Stage::kOff};
  std::size_t weights{}, scratch{}, runtime{}, staging{}, duplicates{};
  bool overhead_bounded{false};
};

struct ReservationRequest {
  std::size_t envelope_bytes{}, planned_peak_bytes{};
};

// Pure CPU validation; no allocation, device access or admission side effect.
// The full cap, not just the planned peak, is reserved before cache budgeting.
[[nodiscard]] ReservationRequest PlanReservation(const BackingPlan&);
[[nodiscard]] std::string ReservationLog(const ReservationRequest&);

struct BackingState {
  std::size_t envelope_bytes{}, committed_bytes{}, pending_bytes{};
  bool quarantined{false};
  [[nodiscard]] std::size_t unmaterialized_bytes() const noexcept {
    return envelope_bytes - committed_bytes;
  }
};

// Implemented by the admission owner, not this model package. One instance per
// model reservation; it must outlive its Lifetime. Serialize calls with cache
// resampling/allocation under the owner's lock. TryReserve is all-or-nothing,
// reserves the full envelope before cache sizing, and has no effect on failure.
// TryReserve must refuse if any process-level NPU quarantine is still held.
// Update transfers reserved -> committed; it must not subtract resident bytes
// from sampled availability twice. Reported committed bytes are an upper bound,
// not proof that pages are reflected in availability: any policy add-back needs
// independently substantiated materialization. Quarantine stays charged until
// process exit.
class Admission {
public:
  virtual ~Admission() = default;
  virtual bool TryReserve(const ReservationRequest&) = 0;
  virtual void Update(const BackingState&) noexcept = 0;
  virtual void Release() noexcept = 0;
};

// CPU protocol only: this does not allocate/free backing or wait on fences.
// Callers must report confirmed allocation/free/completion under the same owner
// lock. Never infer completion from a timeout. At most one allocation/dispatch
// is outstanding. Destruction with live/uncertain backing quarantines, not
// frees.
class Lifetime {
public:
  Lifetime(const BackingPlan&, Admission&);
  ~Lifetime();
  Lifetime(const Lifetime&) = delete;
  Lifetime& operator=(const Lifetime&) = delete;
  Lifetime(Lifetime&&) = delete;
  Lifetime& operator=(Lifetime&&) = delete;

  void Reserve();
  void BeginAllocation(std::size_t bytes);  // charge BEFORE allocator call
  void AllocationSucceeded();
  void AllocationFailed();                   // only when no backing was created
  void ReleaseConfirmed(std::size_t bytes);  // after actual free, idle only
  void Submit();
  void ConsumersCompleted();   // both NPU producer AND GPU consumer fences
  void Quarantine() noexcept;  // timeout/cancel with uncertain completion
  void Close();                // only after all backing is confirmed freed
  // True means no NPU-side hazard, not that NPU is active. OFF must not block
  // ordinary checkpoints; the caller still validates its other state/fences.
  [[nodiscard]] bool CanCheckpoint() const noexcept;
  [[nodiscard]] BackingState state() const noexcept { return state_; }

private:
  void RequireIdle() const;
  ReservationRequest request_;
  Admission& admission_;
  BackingState state_;
  bool reserved_{false}, in_flight_{false}, closed_{false};
};

// Arithmetic identity is composed with the existing model/session identity in
// live, RAM, SSD and pinned-prefix paths. No pointer/ticket/runtime address is
// a numerical identity. Different routing/packing/program/consumer policy must
// produce different keys; ON/OFF never share until separately proven
// equivalent.
struct ArithmeticPlan {
  bool enabled{false};
  std::string model_quant, packing_passes, program_hash, instruction_hash;
  std::string abi, layers_ops, residual_consumer, chunk_tail_routing;
};
[[nodiscard]] std::string CacheIdentity(const ArithmeticPlan&);

}  // namespace gufo::models::qwen38_flash_next::npu

#endif
