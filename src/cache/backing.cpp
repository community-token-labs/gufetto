#include "src/cache/backing.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gufo::cache {
namespace detail {
struct BackingSegment {
  BackingPages pages;
  bool ready{};
  // Retain unconverted admissions if rollback cannot physically release.
  std::vector<ResourceReservation> pending;
};
struct BackingState {
  ResourceCharge metadata;
  std::unique_ptr<ResourceCharge[]> blocks;
  std::unique_ptr<BackingSegment[]> segments;
  std::size_t capacity{}, segment_blocks{}, block_bytes{}, segment_count{};
  std::atomic<std::size_t> committed{}, allocations{}, pinned_bytes{};
  std::size_t next{}, next_private{std::numeric_limits<std::size_t>::max()};
  ResourceLedger* ledger{};
  BackingAllocator allocator;
  std::function<bool(std::size_t)> admit_segment;
  std::function<std::shared_ptr<void>(std::size_t)> acquire_segment;
  // Original block handles never escape this state. Serialize search THROUGH
  // ReserveBacking with publication/trim; owners may only release or pin an
  // already assigned token, so a wholly-free segment cannot become busy here.
  mutable std::mutex topology;
  std::atomic<bool> growing_ahead{};

  std::byte* Address(std::size_t index) const {
    return static_cast<std::byte*>(
               segments[index / segment_blocks].pages.data) +
           index % segment_blocks * block_bytes;
  }
  std::size_t SegmentBlocks(std::size_t index) const {
    return std::min(segment_blocks, capacity - index * segment_blocks);
  }
  std::span<ResourceCharge> SegmentCharges(std::size_t index) const {
    return {blocks.get() + index * segment_blocks, SegmentBlocks(index)};
  }
  bool IsFree(std::size_t index) const {
    const auto& segment = segments[index];
    if (!segment.pages)
      return false;
    // Failed construction/rollback never publishes assignments.
    if (!segment.ready)
      return true;
    const auto range = SegmentCharges(index);
    return ResourceCharge::FindFreeBacking(range, range.size()) == 0;
  }
  bool Release(std::size_t index) {
    auto& segment = segments[index];
    const auto count = SegmentBlocks(index);
    const auto bytes = count * block_bytes;
    const bool pinned = segment.pages.long_term_pinned;
    segment.ready = false;
    const bool released = allocator.release(segment.pages, bytes);
    if (pinned && (released || !segment.pages.long_term_pinned))
      pinned_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    if (!released)
      return false;
    segment.pages = {};
    // Physical callback finishes before any accounting is credited.
    for (auto& charge : SegmentCharges(index))
      charge = {};
    segment.pending.clear();
    committed.fetch_sub(count, std::memory_order_release);
    allocations.fetch_sub(1, std::memory_order_relaxed);
    return true;
  }
  // Caller holds topology. Fill an interior hole before extending capacity.
  bool Grow(bool eager = false) {
    std::size_t index = 0;
    while (index < segment_count && segments[index].pages)
      ++index;
    if (index == segment_count)
      return false;
    const auto count = SegmentBlocks(index);
    const auto bytes = count * block_bytes;
    if (!eager && admit_segment && !admit_segment(bytes))
      return false;
    auto ticket = acquire_segment ? acquire_segment(bytes) : nullptr;
    if (acquire_segment && !ticket)
      return false;
    auto& segment = segments[index];
    std::vector<ResourceReservation> reservations;
    try {
      reservations.reserve(count);
      for (std::size_t i = 0; i < count; ++i)
        reservations.push_back(
            ledger->Reserve(ResourceCategory::kBackingFree, block_bytes));
      segment.pages = allocator.commit(bytes);
    } catch (const std::bad_alloc&) {
      if (eager)
        throw;
      return false;
    }
    if (!segment.pages)
      return false;
    committed.fetch_add(count, std::memory_order_release);
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (segment.pages.long_term_pinned)
      pinned_bytes.fetch_add(bytes, std::memory_order_relaxed);
    // Move cannot allocate. All admissions survive failed conversion/release.
    segment.pending = std::move(reservations);
    try {
      auto range = SegmentCharges(index);
      for (std::size_t i = 0; i < count; ++i)
        range[i] = segment.pending[i].Convert();
    } catch (const std::bad_alloc&) {
      (void)Release(index);
      if (eager)
        throw;
      return false;
    }
    segment.pending.clear();
    segment.ready = true;
    // The ticket's last owner releases only after this publication.
    return true;
  }
  ~BackingState() {
    if (!segments)
      return;
    for (std::size_t i = 0; i < segment_count; ++i) {
      if (segments[i].pages && !Release(i)) {
        // No caller remains to retry. Do not silently credit leaked physical
        // storage or let a failed unregister leave a dangling registration.
        std::terminate();
      }
    }
  }
};
}  // namespace detail
BackingPin::BackingPin(std::shared_ptr<detail::BackingState> state,
                       PersistencePin pin)
    : state_(std::move(state)), pin_(std::move(pin)) {}
BackingPin& BackingPin::operator=(const BackingPin& other) {
  if (this != &other) {
    pin_ = {};
    state_ = other.state_;
    pin_ = other.pin_;
  }
  return *this;
}
BackingPin& BackingPin::operator=(BackingPin&& other) noexcept {
  if (this != &other) {
    pin_ = {};
    state_ = std::move(other.state_);
    pin_ = std::move(other.pin_);
  }
  return *this;
}
BackingBlock::BackingBlock(std::shared_ptr<detail::BackingState> state,
                           std::size_t index, std::size_t blocks,
                           ResourceReservation reservation)
    : state_(std::move(state)),
      index_(index),
      blocks_(blocks),
      reservation_(std::move(reservation)) {}
BackingBlock& BackingBlock::operator=(BackingBlock&& other) noexcept {
  if (this != &other) {
    reservation_ = {};
    charge_ = {};
    state_ = std::move(other.state_);
    index_ = other.index_;
    blocks_ = other.blocks_;
    reservation_ = std::move(other.reservation_);
    charge_ = std::move(other.charge_);
  }
  return *this;
}
std::span<std::byte> BackingBlock::Bytes() const {
  if (!state_)
    throw std::logic_error("empty backing block");
  return {state_->Address(index_), blocks_ * state_->block_bytes};
}
ResourceAllocationInfo BackingBlock::Info() const {
  return reservation_ ? reservation_.Info() : charge_.Info();
}
void BackingBlock::Convert() {
  charge_ = reservation_.Convert();
}
Payload BackingBlock::TakePayload() && {
  if (!charge_ || reservation_)
    throw std::logic_error("backing capture has not completed");
  const auto bytes = Bytes();
  auto owner = std::shared_ptr<const void>(std::move(state_), bytes.data());
  return Payload::Committed(std::move(charge_), std::move(owner));
}
std::shared_ptr<BorrowedRows> BackingBlock::Borrow(
    SlotLease& lease, ComponentId component, Rows first, Rows end,
    std::size_t payload_bytes, std::span<const std::byte> source) && {
  auto buffer = Bytes();
  if (!reservation_ || payload_bytes > buffer.size())
    throw std::invalid_argument("invalid borrowed backing block");
  auto owner = std::shared_ptr<const void>(std::move(state_), buffer.data());
  return lease.Borrow(component, first, end, std::move(reservation_),
                      std::move(owner), buffer.first(payload_bytes), source);
}
BackingPin BackingBlock::PinPersistence() const {
  return BackingPin(state_, reservation_ ? reservation_.PinPersistence()
                                         : charge_.PinPersistence());
}

CommittedBackingPool::CommittedBackingPool(ResourceLedger& ledger,
                                           BackingPoolConfig config,
                                           BackingAllocator allocator) {
  if (!allocator.commit || !allocator.release)
    throw std::invalid_argument("missing physical backing allocator");
  constexpr std::size_t page = 4096;
  if (!config.block_bytes || config.block_bytes % page ||
      config.metadata_headroom_bytes > config.ram_bytes ||
      config.segment_bytes % config.block_bytes ||
      (config.segment_bytes && config.segment_bytes < config.block_bytes))
    throw std::invalid_argument("invalid committed backing budget/block size");
  const auto count =
      (config.ram_bytes - config.metadata_headroom_bytes) / config.block_bytes;
  // Headroom also covers the ledger's per-block allocations, borrower tokens,
  // pins and allocator overhead. Leave ample space rather than count payload
  // bytes twice. Future store/index metadata needs its own budget share.
  constexpr std::size_t metadata_per_block = 1024;
  if (config.metadata_headroom_bytes < sizeof(detail::BackingState) ||
      count > (config.metadata_headroom_bytes - sizeof(detail::BackingState)) /
                  metadata_per_block)
    throw std::invalid_argument("insufficient backing metadata headroom");
  auto metadata = ledger.Reserve(ResourceCategory::kMetadata,
                                 config.metadata_headroom_bytes);
  auto state = std::make_shared<detail::BackingState>();
  state->metadata = metadata.Convert();
  state->block_bytes = config.block_bytes;
  state->capacity = count;
  state->segment_blocks = config.segment_bytes
                              ? config.segment_bytes / config.block_bytes
                              : std::max<std::size_t>(count, 1);
  state->blocks = std::make_unique<ResourceCharge[]>(count);
  state->segment_count =
      count / state->segment_blocks + (count % state->segment_blocks != 0);
  state->segments =
      std::make_unique<detail::BackingSegment[]>(state->segment_count);
  state->ledger = &ledger;
  state->allocator = std::move(allocator);
  state->admit_segment = std::move(config.admit_segment);
  state->acquire_segment = std::move(config.acquire_segment);
  if (!config.segment_bytes && count && !state->Grow(true))
    throw std::runtime_error("cannot commit cache backing");
  state_ = std::move(state);
}
bool CommittedBackingPool::CanFit(
    std::span<const PayloadRequest> payloads) const {
  const std::lock_guard lock(state_->topology);
  std::size_t needed = 0;
  for (const auto& payload : payloads) {
    const auto blocks =
        payload.bytes ? 1 + (payload.bytes - 1) / state_->block_bytes : 1;
    if (blocks > std::min(state_->segment_blocks, state_->capacity) ||
        blocks > state_->capacity - needed)
      return false;
    needed += blocks;
  }
  // Holes remain configured capacity. Final admission still checks segment
  // fragmentation, quarantined releases and the shared growth authority.
  const auto uncommitted =
      state_->capacity - state_->committed.load(std::memory_order_acquire);
  if (needed <= uncommitted)
    return true;
  // Other pools on this ledger can also make this bound optimistic. It must
  // never reject a capture whose physical allocations could fit.
  const auto free =
      state_->ledger->Snapshot()
          .bytes[static_cast<std::size_t>(ResourceCategory::kBackingFree)] /
      state_->block_bytes;
  return needed - uncommitted <= free;
}
std::optional<BackingBlock> CommittedBackingPool::TryAcquire(
    ResourceCategory category, std::size_t payload_bytes) {
  if (category != ResourceCategory::kBackingAssigned &&
      category != ResourceCategory::kPrivateState &&
      category != ResourceCategory::kPrivateTail)
    throw std::invalid_argument("invalid backing assignment category");
  const auto blocks =
      payload_bytes ? 1 + (payload_bytes - 1) / state_->block_bytes : 1;
  const auto segment = state_->segment_blocks;
  if (blocks > std::min(segment, state_->capacity))
    return std::nullopt;
  const std::lock_guard lock(state_->topology);
  const bool from_end = category == ResourceCategory::kPrivateState;
  auto& cursor = from_end ? state_->next_private : state_->next;
  while (true) {
    const auto count = state_->capacity;
    auto i = cursor;
    if (i >= count)
      i = from_end && count ? count - 1 : 0;
    // A growing pool is usually full when a capture misses. The ledger's free
    // backing bounds this pool's, so skip a scan that cannot succeed: it
    // costs one ledger lock per committed block.
    const auto free_bytes =
        state_->ledger->Snapshot()
            .bytes[static_cast<std::size_t>(ResourceCategory::kBackingFree)];
    const bool may_fit = free_bytes >= blocks * state_->block_bytes;
    for (std::size_t tried = 0; may_fit && blocks <= count && tried < count;) {
      // Search one physical segment under one ledger lock instead of locking
      // separately for each block in a large, fragmented cache.
      auto begin = from_end ? i / segment * segment : i;
      auto limit =
          from_end ? i + 1 : std::min(count, (i / segment + 1) * segment);
      // After wrapping, inspect each possible start/end only once. Include
      // the rest of a grouped allocation, but do not retry a reservation
      // that already failed earlier in this scan.
      const auto candidates = std::min(count - tried, limit - begin);
      const auto length =
          candidates + std::min(blocks - 1, limit - begin - candidates);
      if (from_end)
        begin = limit - length;
      else
        limit = begin + length;
      // Holes and failed releases have no allocatable original handles.
      if (!state_->segments[i / segment].ready) {
        const auto step = limit - begin;
        tried += step;
        i = from_end ? (step > i ? count - 1 : i - step)
                     : (i + step >= count ? 0 : i + step);
        continue;
      }
      const auto range =
          std::span(state_->blocks.get(), count).subspan(begin, limit - begin);
      const auto offset =
          ResourceCharge::FindFreeBacking(range, blocks, from_end);
      if (offset < range.size()) {
        try {
          auto reservation = ResourceCharge::ReserveBacking(
              range.subspan(offset, blocks), category);
          const auto first = begin + offset;
          cursor = from_end ? (first ? first - 1
                                     : std::numeric_limits<std::size_t>::max())
                            : (first + blocks == count ? 0 : first + blocks);
          return BackingBlock(state_, first, blocks, std::move(reservation));
        } catch (const std::bad_alloc&) {
          // A racing assignment or injected bookkeeping failure is atomic.
        }
      }
      const auto step = offset == range.size() ? range.size()
                        : from_end ? range.size() - (offset + blocks) + 1
                                   : offset + 1;
      tried += step;
      i = from_end ? (step > i ? count - 1 : i - step)
                   : (i + step >= count ? 0 : i + step);
    }
    if (!state_->Grow())
      return std::nullopt;
  }
}
void CommittedBackingPool::GrowAhead(std::size_t reserve_bytes) {
  const auto committed = state_->committed.load(std::memory_order_acquire);
  if (!reserve_bytes || committed == state_->capacity ||
      state_->growing_ahead.load(std::memory_order_acquire))
    return;
  const auto free_bytes =
      state_->ledger->Snapshot()
          .bytes[static_cast<std::size_t>(ResourceCategory::kBackingFree)];
  const auto target_bytes =
      std::min(reserve_bytes, state_->capacity * state_->block_bytes);
  if (free_bytes >= target_bytes)
    return;
  const std::lock_guard lock(ahead_mutex_);
  if (state_->growing_ahead.exchange(true))
    return;
  if (ahead_.joinable())
    ahead_.join();
  try {
    ahead_ = std::jthread([state = state_, target_bytes] {
      try {
        // Bound one worker even when concurrent trim keeps opening holes.
        for (std::size_t attempt = 0; attempt < state->segment_count;
             ++attempt) {
          const std::lock_guard topology(state->topology);
          if (state->ledger->Snapshot().bytes[static_cast<std::size_t>(
                  ResourceCategory::kBackingFree)] >= target_bytes ||
              !state->Grow())
            break;
        }
      } catch (...) {
      }
      state->growing_ahead.store(false);
    });
  } catch (...) {
    state_->growing_ahead.store(false);
  }
}
void CommittedBackingPool::WaitForGrowth() {
  const std::lock_guard lock(ahead_mutex_);
  if (ahead_.joinable())
    ahead_.join();
}
std::size_t CommittedBackingPool::ImmediatelyTrimmableBytes() const {
  const std::lock_guard lock(state_->topology);
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < state_->segment_count; ++i)
    if (state_->segments[i].ready && state_->IsFree(i))
      bytes += state_->SegmentBlocks(i) * state_->block_bytes;
  return bytes;
}
std::size_t CommittedBackingPool::TrimFreeSegments(std::size_t target_bytes) {
  const std::lock_guard lock(state_->topology);
  std::size_t released = 0;
  for (std::size_t i = 0; i < state_->segment_count && released < target_bytes;
       ++i) {
    if (state_->IsFree(i) && state_->Release(i))
      released += state_->SegmentBlocks(i) * state_->block_bytes;
  }
  return released;
}
std::size_t CommittedBackingPool::BlockCount() const noexcept {
  return state_->committed.load(std::memory_order_acquire);
}
std::size_t CommittedBackingPool::CommittedBytes() const noexcept {
  return BlockCount() * state_->block_bytes;
}
std::size_t CommittedBackingPool::PageCommitAllocations() const noexcept {
  return state_->allocations.load(std::memory_order_relaxed);
}
std::size_t CommittedBackingPool::LongTermPinnedBytes() const noexcept {
  return state_->pinned_bytes.load(std::memory_order_relaxed);
}

}  // namespace gufo::cache
