#ifndef GUFO_CACHE_BACKING_HPP_
#define GUFO_CACHE_BACKING_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>

#include "src/cache/checkpoint.hpp"
#include "src/cache/ledger.hpp"
#include "src/cache/slot.hpp"

namespace gufo::cache {
namespace detail {
struct BackingState;
}

// Retains physical storage as well as accounting while a job holds its pin.
class BackingPin {
public:
  BackingPin(const BackingPin&) = default;
  BackingPin(BackingPin&&) noexcept = default;
  BackingPin& operator=(const BackingPin&);
  BackingPin& operator=(BackingPin&&) noexcept;

private:
  friend class BackingBlock;
  BackingPin(std::shared_ptr<detail::BackingState>, PersistencePin);
  std::shared_ptr<detail::BackingState> state_;
  PersistencePin pin_;
};

// A whole block is charged, including any unused payload capacity. Destroying
// an unconverted block cancels capture. Convert only after the copy succeeds.
// Keep this handle (or a pin) alive until all users of Bytes have drained.
class BackingBlock {
public:
  BackingBlock(BackingBlock&&) noexcept = default;
  BackingBlock& operator=(BackingBlock&&) noexcept;
  BackingBlock(const BackingBlock&) = delete;
  BackingBlock& operator=(const BackingBlock&) = delete;
  [[nodiscard]] std::span<std::byte> Bytes() const;
  [[nodiscard]] ResourceAllocationInfo Info() const;
  void Convert();
  [[nodiscard]] Payload TakePayload() &&;
  // Consume an unconverted assigned block as borrowed checkpoint backing.
  // Payload is a prefix of the block; accounting still charges full capacity.
  // The row handle retains the slab after this block/pool facade is destroyed.
  [[nodiscard]] std::shared_ptr<BorrowedRows> Borrow(
      SlotLease&, ComponentId, Rows first, Rows end, std::size_t payload_bytes,
      std::span<const std::byte> source = {}) &&;
  [[nodiscard]] BackingPin PinPersistence() const;

private:
  friend class CommittedBackingPool;
  BackingBlock(std::shared_ptr<detail::BackingState>, std::size_t, std::size_t,
               ResourceReservation);
  std::shared_ptr<detail::BackingState> state_;
  std::size_t index_;
  std::size_t blocks_;
  ResourceReservation reservation_;
  ResourceCharge charge_;
};

// Opaque allocator state is updated by release even on failure (for example,
// after unregister succeeds but unmapping fails). Failed release must leave
// data nonnull and retryable; the pool quarantines it, never reassigning it.
struct BackingPages {
  void* data{};
  std::shared_ptr<void> token;
  bool long_term_pinned{};
  explicit operator bool() const noexcept { return data != nullptr; }
};
struct BackingAllocator {
  // Commit either returns fully usable storage or no storage. Exceptions must
  // not leak physical memory. Release returns true ONLY after physical release.
  // Both run under the topology lock, never under the ledger lock, and must
  // not reenter the pool. Release must not throw. A release failure during
  // final state destruction terminates rather than crediting leaked storage;
  // ordinary trim/rollback failures remain quarantined and retryable.
  std::function<BackingPages(std::size_t)> commit;
  std::function<bool(BackingPages&, std::size_t)> release;
};

struct BackingPoolConfig {
  // A share of D4's RAM budget, including metadata headroom. This pool does
  // not consume the rest of the ledger budget. Block sizes are page multiples.
  std::size_t ram_bytes;
  std::size_t block_bytes;
  std::size_t metadata_headroom_bytes;
  // Zero commits every block at construction. Otherwise TryAcquire commits
  // blocks on demand, one segment of this many bytes (a block multiple) at a
  // time, until ram_bytes is reached. A payload never spans two segments.
  std::size_t segment_bytes{};
  // Asked before each on-demand segment with its bytes; false keeps the pool
  // at its current size, for example to leave host memory to the OS.
  std::function<bool(std::size_t)> admit_segment;
  // Optional authority ticket held through physical commit AND publication.
  // Empty denies growth. Also applies to eager construction; admit_segment
  // retains its legacy lazy-growth-only behavior. If both exist, both gate
  // lazy growth. Called under the topology lock: neither this callback nor
  // the ticket deleter may reclaim/reenter this pool or wait for its worker.
  std::function<std::shared_ptr<void>(std::size_t)> acquire_segment;
};

class CommittedBackingPool {
public:
  // Reserves metadata for every block first. Without segment_bytes it admits,
  // allocates and touches every payload page here; otherwise none until a
  // TryAcquire finds no free run. Only wholly free segments may be trimmed.
  // The ledger must outlive this facade.
  CommittedBackingPool(ResourceLedger&, BackingPoolConfig, BackingAllocator);
  // Thread safe. Races/bad_alloc try other blocks, then commit another
  // segment if the pool may grow, then decline capture. Physical commitment
  // runs on the calling thread (the device adapter may pin/register pages).
  [[nodiscard]] std::optional<BackingBlock> TryAcquire(
      ResourceCategory, std::size_t payload_bytes = 0);
  // Reject a whole capture only when rounded payloads exceed free plus
  // uncommitted capacity. Does not allocate or change the search cursor.
  // Fragmentation, growth limits and races still need TryAcquire's checks.
  [[nodiscard]] bool CanFit(std::span<const PayloadRequest>) const;
  // Prepares free backing on a background thread so a later TryAcquire finds
  // it ready. Reserve bytes are independent of the physical segment size;
  // zero does nothing and oversized reserves are bounded by capacity.
  void GrowAhead(std::size_t reserve_bytes);
  void WaitForGrowth();
  // Reclaims whole free segments, including interior holes. May exceed the
  // requested target by one segment. Only successful physical releases count.
  // Failed releases remain charged and quarantined until a successful retry.
  [[nodiscard]] std::size_t TrimFreeSegments(std::size_t target_bytes);
  // Synchronized conservative wholly-free segment capacity, not a promise
  // that a later release will succeed or that these bytes remain free.
  // Known failed releases are excluded; TrimFreeSegments still retries them.
  [[nodiscard]] std::size_t ImmediatelyTrimmableBytes() const;
  // Current physical blocks/bytes, including failed-release quarantine.
  // PageCommitAllocations is the current segment count, not cumulative.
  [[nodiscard]] std::size_t BlockCount() const noexcept;
  [[nodiscard]] std::size_t CommittedBytes() const noexcept;
  [[nodiscard]] std::size_t PageCommitAllocations() const noexcept;
  // Current bytes marked long-term pinned by the physical adapter. This is
  // distinct from the ledger's assigned-payload persistence pins.
  [[nodiscard]] std::size_t LongTermPinnedBytes() const noexcept;

private:
  std::shared_ptr<detail::BackingState> state_;
  std::mutex ahead_mutex_;
  // Joined before state_ is released; it keeps its own state reference.
  std::jthread ahead_;
};

}  // namespace gufo::cache
#endif
