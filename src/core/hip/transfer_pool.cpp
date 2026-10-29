#include "src/core/hip/transfer_pool.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/core/perf_trace.hpp"
#include "src/core/platform/shutdown_signal.hpp"

namespace gufo::hip {
namespace {
std::atomic<FatalCopyHandler> fatal_handler{};
std::atomic<bool> fault_unsettled{};
void CheckTransfer(hipError_t status, const char* operation,
                   FatalCopyInfo info = {}) noexcept {
  if (status != hipSuccess && !fault_unsettled.exchange(true)) {
    info.operation = operation;
    info.error = static_cast<int>(status);
    try {
      if (const auto handler = fatal_handler.load())
        handler(info);
    } catch (...) {
    }
    // Missing, throwing or returning notification cannot release any owners.
    platform::ExitAfterDeviceLoss();
  }
  // A different transfer already faulted. Retain this caller's owners too;
  // only the serving watchdog/process exit may end this wait, not HIP success.
  while (fault_unsettled.load())
    fault_unsettled.wait(true);
}
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
}  // namespace
void SetFatalCopyHandler(FatalCopyHandler handler) noexcept {
  fatal_handler.store(handler);
}
namespace detail {
struct TransferPoolState {
  struct Slot {
    hipStream_t stream{};
    hipEvent_t event{};
    bool busy{};
  };
  std::mutex mutex;
  std::vector<Slot> slots;
  ~TransferPoolState() {
    CheckTransfer(hipSuccess, "hipStreamSynchronize");
    for (auto& slot : slots) {
      if (slot.stream) {
        CheckTransfer(hipStreamSynchronize(slot.stream),
                      "hipStreamSynchronize");
        CheckTransfer(hipStreamDestroy(slot.stream), "hipStreamDestroy");
      }
      if (slot.event)
        CheckTransfer(hipEventDestroy(slot.event), "hipEventDestroy");
    }
  }
};
struct TransferLease {
  std::shared_ptr<TransferPoolState> state;
  std::size_t index;
  bool sealed{};
  bool claimed{};
  bool native_used{};
  hipStream_t bound_stream{};
  std::shared_ptr<void> bound_owner;
  FatalCopyInfo context{};
  void Check(hipError_t status, const char* operation) const noexcept {
    CheckTransfer(status, operation, context);
  }
  TransferLease(std::shared_ptr<TransferPoolState> state, std::size_t index)
      : state(std::move(state)), index(index) {}
  auto& Slot() const { return state->slots[index]; }
  hipStream_t Stream() const {
    return bound_stream ? bound_stream : Slot().stream;
  }
  cache::TransferResult Drain(const char* phase_name) noexcept {
    Check(hipSuccess, "hipStreamSynchronize");
    const auto status = [&] {
      perf::HostPhase phase(phase_name);
      return hipStreamSynchronize(Stream());
    }();
    Check(status, "hipStreamSynchronize");
    return cache::TransferResult::kSucceeded;
  }
  ~TransferLease() {
    // Metadata allocation may fail before any slot is claimed. Such a lease
    // must neither synchronize nor release another borrower's stream.
    if (!claimed)
      return;
    (void)Drain(bound_stream ? "hip_transfer_bound_lease_destructor_drain"
                             : "hip_transfer_owned_lease_destructor_drain");
    const std::lock_guard lock(state->mutex);
    Slot().busy = false;
  }
};
}  // namespace detail
namespace {
class TransferSignal final : public cache::CompletionSignal {
public:
  explicit TransferSignal(std::shared_ptr<detail::TransferLease> lease)
      : lease_(std::move(lease)) {}
  bool Ready() const noexcept override {
    lease_->Check(hipSuccess, "hipEventQuery");
    const auto status = hipEventQuery(lease_->Slot().event);
    lease_->Check(status == hipErrorNotReady ? hipSuccess : status,
                  "hipEventQuery");
    return status == hipSuccess;
  }
  cache::TransferResult Wait() noexcept override {
    lease_->Check(hipSuccess, "hipEventSynchronize");
    const auto status = [&] {
      perf::HostPhase phase(lease_->bound_stream
                                ? "hip_transfer_bound_event_wait"
                                : "hip_transfer_owned_event_wait");
      return hipEventSynchronize(lease_->Slot().event);
    }();
    lease_->Check(status, "hipEventSynchronize");
    // Healthy event paths retain the existing drain for external Native work.
    return lease_->Drain(lease_->bound_stream
                             ? "hip_transfer_bound_wait_drain"
                             : "hip_transfer_owned_wait_drain");
  }

private:
  std::shared_ptr<detail::TransferLease> lease_;
};
}  // namespace

TransferStream::TransferStream(std::shared_ptr<detail::TransferLease> lease)
    : lease_(std::move(lease)) {}
TransferStream::~TransferStream() {
  (void)lease_->Drain(lease_->bound_stream
                          ? "hip_transfer_bound_stream_destructor_drain"
                          : "hip_transfer_owned_stream_destructor_drain");
}
void TransferStream::BindTo(hipStream_t stream, std::shared_ptr<void> owner) {
  if (!stream || !owner)
    throw std::invalid_argument("bound transfer requires a stream owner");
  if (lease_->sealed || lease_->native_used || lease_->bound_stream)
    throw std::logic_error("transfer stream was already used");
  lease_->bound_stream = stream;
  lease_->bound_owner = std::move(owner);
}
void TransferStream::SetPersistenceContext(std::uint64_t entry,
                                           std::uint64_t bytes) noexcept {
  lease_->context.has_entry_id = true;
  lease_->context.entry_id = entry;
  lease_->context.bytes = bytes;
}
hipStream_t TransferStream::Native() const {
  lease_->Check(hipSuccess, "hipStreamSynchronize");
  if (lease_->sealed)
    throw std::logic_error("transfer stream is sealed");
  lease_->native_used = true;
  return lease_->Stream();
}
cache::TransferResult TransferStream::Synchronize() noexcept {
  return lease_->Drain(lease_->bound_stream
                           ? "hip_transfer_bound_synchronize_drain"
                           : "hip_transfer_owned_synchronize_drain");
}
cache::Completion TransferStream::Complete() {
  return Finish(std::make_unique<TransferSignal>(lease_));
}
cache::Completion TransferStream::Finish(
    std::unique_ptr<cache::CompletionSignal> signal) {
  const auto stream = Native();
  const auto status = [&] {
    perf::HostPhase phase(lease_->bound_stream
                              ? "hip_transfer_bound_event_record"
                              : "hip_transfer_owned_event_record");
    return hipEventRecord(lease_->Slot().event, stream);
  }();
  lease_->sealed = true;
  lease_->Check(status, "hipEventRecord");
  return cache::Completion(std::move(signal));
}
cache::Completion TransferStream::Copy(void* destination, const void* source,
                                       std::size_t bytes,
                                       CopyDirection direction,
                                       std::size_t piece_bytes) {
  if (!piece_bytes || (bytes && (!destination || !source)) ||
      (direction != CopyDirection::kDeviceToHost &&
       direction != CopyDirection::kHostToDevice))
    throw std::invalid_argument("invalid bounded transfer");
  // Host bookkeeping failure precedes the first possible DMA submission.
  auto signal = std::make_unique<TransferSignal>(lease_);
  const auto stream = Native();
  lease_->context.bytes = bytes;
  const auto kind = direction == CopyDirection::kDeviceToHost
                        ? hipMemcpyDeviceToHost
                        : hipMemcpyHostToDevice;
  try {
    for (std::size_t offset = 0; offset < bytes;) {
      const auto piece = std::min(piece_bytes, bytes - offset);
      const auto status = [&] {
        perf::HostPhase phase(
            direction == CopyDirection::kDeviceToHost
                ? (lease_->bound_stream ? "hip_transfer_bound_d2h_enqueue"
                                        : "hip_transfer_owned_d2h_enqueue")
                : (lease_->bound_stream ? "hip_transfer_bound_h2d_enqueue"
                                        : "hip_transfer_owned_h2d_enqueue"));
        return hipMemcpyAsync(static_cast<std::byte*>(destination) + offset,
                              static_cast<const std::byte*>(source) + offset,
                              piece, kind, stream);
      }();
      lease_->Check(status, "hipMemcpyAsync");
      offset += piece;
    }
    return Finish(std::move(signal));
  } catch (...) {
    // Includes host metadata failure after some copies were already queued.
    (void)Synchronize();
    throw;
  }
}
TransferPool::TransferPool(std::size_t capacity) {
  if (!capacity)
    throw std::invalid_argument("transfer pool requires a nonzero capacity");
  auto state = std::make_shared<detail::TransferPoolState>();
  state->slots.resize(capacity);
  for (auto& slot : state->slots) {
    Check(hipStreamCreateWithFlags(&slot.stream, hipStreamNonBlocking));
    Check(hipEventCreateWithFlags(&slot.event, hipEventDisableTiming));
  }
  state_ = std::move(state);
}
std::unique_ptr<TransferStream> TransferPool::TryAcquire() {
  CheckTransfer(hipSuccess, "hipStreamSynchronize");
  const std::lock_guard lock(state_->mutex);
  for (std::size_t i = 0; i < state_->slots.size(); ++i) {
    auto& slot = state_->slots[i];
    if (slot.busy)
      continue;
    auto lease = std::make_shared<detail::TransferLease>(state_, i);
    // An unclaimed lease has no return-to-pool action on allocation failure.
    auto stream = std::unique_ptr<TransferStream>(new TransferStream(lease));
    slot.busy = true;
    lease->claimed = true;
    return stream;
  }
  return nullptr;
}

}  // namespace gufo::hip
