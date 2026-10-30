#ifndef GUFO_CORE_HIP_TRANSFER_POOL_HPP_
#define GUFO_CORE_HIP_TRANSFER_POOL_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>

#include "src/cache/stream.hpp"

namespace gufo::hip {
namespace detail {
struct TransferPoolState;
struct TransferLease;
}  // namespace detail

enum class CopyDirection { kDeviceToHost, kHostToDevice };

struct FatalCopyInfo {
  const char*
      operation{};  // Static HIP API literal, never an error-name lookup.
  int error{};
  bool has_entry_id{};
  std::uint64_t entry_id{}, bytes{};
};
using FatalCopyHandler = void (*)(const FatalCopyInfo&);
// Install before backend load; target/state must live until process exit.
// The handler must arrange bounded exit without releasing transfer owners.
void SetFatalCopyHandler(FatalCopyHandler) noexcept;

// One stream/event pair per transfer. Caller orders access to this object;
// different leases may run concurrently. Sources must already be ready and
// stay immutable; retain all buffers until completion settles. A lease accepts
// one completion only. Complete seals Native submissions too.
class TransferStream final : public cache::Stream {
public:
  ~TransferStream() override;
  // Before any Native submission, use an existing execution stream instead of
  // the pool's stream. Retain its owner until both this lease and completion
  // are released; synchronization and error draining use the bound stream.
  void BindTo(hipStream_t, std::shared_ptr<void> owner);
  [[nodiscard]] hipStream_t Native() const;
  void SetPersistenceContext(std::uint64_t, std::uint64_t) noexcept override;
  [[nodiscard]] cache::TransferResult Synchronize() noexcept override;
  [[nodiscard]] cache::Completion Complete();
  [[nodiscard]] cache::Completion Copy(void* destination, const void* source,
                                       std::size_t bytes, CopyDirection,
                                       std::size_t piece_bytes);

private:
  friend class TransferPool;
  explicit TransferStream(std::shared_ptr<detail::TransferLease>);
  cache::Completion Finish(std::unique_ptr<cache::CompletionSignal>);
  std::shared_ptr<detail::TransferLease> lease_;
};

class TransferPool {
public:
  // Creates all nonblocking streams and timing-disabled events at startup.
  explicit TransferPool(std::size_t capacity);
  // Returns null on exhaustion. No HIP allocation/stream/event creation here.
  // A pair returns only after BOTH stream and completion release their lease.
  // An unproven transfer fault terminates without releasing/reusing its owners.
  [[nodiscard]] std::unique_ptr<TransferStream> TryAcquire();

private:
  std::shared_ptr<detail::TransferPoolState> state_;
};

}  // namespace gufo::hip
#endif
