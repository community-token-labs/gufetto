#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "src/cache/backing.hpp"
#include "tests/cache/fake_adapter.hpp"

namespace {
using namespace gufo::cache;
using Category = ResourceCategory;
constexpr std::size_t page = 4096;
constexpr std::size_t headroom = 64 << 10;
constexpr auto all = std::numeric_limits<std::size_t>::max();
ResourceLimits Limits(std::size_t blocks) {
  const auto bytes = headroom + blocks * page;
  return {bytes, bytes, 0, bytes};
}
BackingPoolConfig Config(std::size_t blocks, std::size_t segment = 2) {
  return {headroom + blocks * page, page, headroom, segment * page, {}, {}};
}

// Real independently-owned heap allocations, with exact address/length
// tracking. Callbacks consult the ledger: this also detects callbacks under its
// mutex.
struct Physical {
  explicit Physical(ResourceLedger& ledger) : ledger(ledger) {}
  ResourceLedger& ledger;
  std::map<void*, std::unique_ptr<std::byte[]>> live;
  std::map<void*, std::size_t> sizes;
  std::atomic<std::size_t> bytes{}, commits{}, releases{};
  std::atomic<bool> deny_commit{}, deny_release{}, deny_unpin{};
  std::function<void()> before_commit, before_release;
  BackingAllocator Allocator() {
    return {[this](std::size_t size) -> BackingPages {
              if (before_commit)
                before_commit();
              assert(ledger.Snapshot().total_bytes >= headroom + bytes + size);
              if (deny_commit)
                return {};
              auto storage = std::make_unique<std::byte[]>(size);
              auto* address = storage.get();
              assert(!live.contains(address));
              sizes.emplace(address, size);
              live.emplace(address, std::move(storage));
              bytes += size;
              ++commits;
              return {address, {}, true};
            },
            [this](BackingPages& storage, std::size_t size) {
              if (before_release)
                before_release();
              assert(ledger.Snapshot().total_bytes >= headroom + bytes);
              assert(live.contains(storage.data));
              assert(sizes.at(storage.data) == size);
              if (deny_unpin)
                return false;
              // Simulate unregister/unpin success followed by failed OS
              // release.
              storage.long_term_pinned = false;
              if (deny_release)
                return false;
              live.erase(storage.data);
              sizes.erase(storage.data);
              bytes -= size;
              ++releases;
              return true;
            }};
  }
  ~Physical() { assert(live.empty() && sizes.empty() && bytes == 0); }
};
void Fill(BackingBlock& block, std::byte value) {
  std::ranges::fill(block.Bytes(), value);
}
void Check(const BackingBlock& block, const std::byte* address,
           std::byte value) {
  assert(block.Bytes().data() == address);
  assert(std::ranges::all_of(block.Bytes(),
                             [=](auto byte) { return byte == value; }));
}

void InteriorHolesAndPartialEdge() {
  ResourceLedger ledger(Limits(5));
  Physical physical{ledger};
  CommittedBackingPool pool(ledger, Config(5), physical.Allocator());
  auto first = pool.TryAcquire(Category::kBackingAssigned, 2 * page);
  auto middle = pool.TryAcquire(Category::kPrivateState, 2 * page);
  auto last = pool.TryAcquire(Category::kPrivateTail);
  assert(first && middle && last && pool.BlockCount() == 5);
  const auto* survivor = last->Bytes().data();
  Fill(*last, std::byte{37});
  assert(pool.ImmediatelyTrimmableBytes() == 0);
  assert(pool.TrimFreeSegments(all) == 0);  // unconverted capture reservations
  middle.reset();
  assert(physical.bytes == 5 * page);  // logical release is NOT physical
  assert(pool.ImmediatelyTrimmableBytes() == 2 * page);
  const auto before = ledger.Snapshot().total_bytes;
  assert(pool.TrimFreeSegments(1) == 2 * page);
  assert(physical.bytes == 3 * page && pool.CommittedBytes() == 3 * page);
  assert(ledger.Snapshot().total_bytes == before - 2 * page);
  assert(pool.PageCommitAllocations() == 2);
  Check(*last, survivor, std::byte{37});
  auto recommitted = pool.TryAcquire(Category::kPrivateState, 2 * page);
  assert(recommitted && physical.bytes == 5 * page && physical.commits == 4);
  assert(std::ranges::all_of(recommitted->Bytes(),
                             [](auto b) { return b == std::byte{}; }));
  Check(*last, survivor, std::byte{37});
  first.reset();
  recommitted.reset();
  assert(pool.TrimFreeSegments(all) == 4 * page);
  assert(pool.CommittedBytes() == page && pool.PageCommitAllocations() == 1);
  Check(*last, survivor, std::byte{37});
  last.reset();
  assert(pool.TrimFreeSegments(0) == 0);
  assert(pool.TrimFreeSegments(all) == page);
  assert(pool.BlockCount() == 0 && pool.LongTermPinnedBytes() == 0);
  assert(ledger.Snapshot().total_bytes == headroom);
  auto again = pool.TryAcquire(Category::kPrivateTail, 2 * page);
  assert(again && pool.BlockCount() == 2);
}

void SharedOwnersPinsAndFacadeLifetime() {
  ResourceLedger ledger(Limits(2));
  Physical physical{ledger};
  auto pool = std::make_unique<CommittedBackingPool>(ledger, Config(2),
                                                     physical.Allocator());
  auto block = pool->TryAcquire(Category::kBackingAssigned, 2 * page);
  assert(block);
  Fill(*block, std::byte{19});
  auto persistence = std::make_optional(block->PinPersistence());
  // Queued persistence retains a cancelled, unconverted capture.
  block.reset();
  assert(pool->ImmediatelyTrimmableBytes() == 0);
  assert(pool->TrimFreeSegments(all) == 0);
  persistence.reset();
  assert(pool->ImmediatelyTrimmableBytes() == 2 * page);
  block = pool->TryAcquire(Category::kBackingAssigned, 2 * page);
  assert(block);
  block->Convert();
  auto payload = std::make_shared<Payload>(std::move(*block).TakePayload());
  block.reset();
  auto shared = payload;
  auto rows = payload->PinRows();
  payload.reset();
  assert(pool->TrimFreeSegments(all) == 0);
  shared.reset();
  assert(pool->TrimFreeSegments(all) == 0);  // direct in-flight RowPin
  assert(rows.Bytes().front() == std::byte{19});
  rows = {};
  assert(pool->TrimFreeSegments(all) == 2 * page);

  block = pool->TryAcquire(Category::kPrivateTail, 2 * page);
  assert(block);
  Fill(*block, std::byte{23});
  block->Convert();
  payload = std::make_shared<Payload>(std::move(*block).TakePayload());
  block.reset();
  pool.reset();
  assert(physical.bytes == 2 * page);
  assert(static_cast<const std::byte*>(payload->Owner().get())[page] ==
         std::byte{23});
  payload.reset();
  assert(physical.bytes == 0 && ledger.Snapshot().total_bytes == 0);
}

void BorrowedReservationsAndPreservation() {
  using namespace gufo::cache::testing;
  ResourceLedger ledger(Limits(16));  // Additional room for slot/row metadata.
  Physical physical{ledger};
  FakeAdapter adapter;
  FakeStream stream;
  LeasedSlot slot(ledger, adapter, stream, SlotId{1});
  auto lease = slot.Acquire();
  const std::array<Token, 5> tokens{1, 2, 3, 4, 5};
  adapter.Append(lease.Execution(), tokens, tokens);
  auto pool = std::make_unique<CommittedBackingPool>(ledger, Config(2),
                                                     physical.Allocator());
  auto block = pool->TryAcquire(Category::kBackingAssigned);
  assert(block);
  auto rows = std::move(*block).Borrow(lease, kTarget, 0, tokens.size(),
                                       sizeof(tokens));
  block.reset();
  assert(pool->TrimFreeSegments(all) == 0);
  auto pin = rows->Pin();
  rows.reset();
  assert(pool->ImmediatelyTrimmableBytes() == 0);
  assert(pool->TrimFreeSegments(all) == 0);  // in-flight borrowed source reader
  pin = {};
  assert(pool->TrimFreeSegments(all) == 2 * page);

  block = pool->TryAcquire(Category::kBackingAssigned);
  assert(block);
  rows = std::move(*block).Borrow(lease, kTarget, 0, tokens.size(),
                                  sizeof(tokens));
  block.reset();
  lease.Commit();
  assert(pool->TrimFreeSegments(all) == 0);
  pool.reset();
  assert(physical.bytes == 2 * page);
  lease = slot.Acquire();  // Preserve after facade destruction.
  assert(!rows->Location());
  pin = rows->Pin();
  assert(std::memcmp(pin.Owner().get(), tokens.data(), sizeof(tokens)) == 0);
  rows.reset();
  assert(physical.bytes == 2 * page);
  pin = {};
  assert(physical.bytes == 0);
}

void EagerAndAdmissionTickets() {
  ResourceLedger ledger(Limits(4));
  Physical physical{ledger};
  std::atomic<int> tickets{};
  std::atomic<bool> allow{true};
  auto config = Config(4, 0);
  config.admit_segment = [](std::size_t) { return false; };
  config.acquire_segment = [&](std::size_t size) -> std::shared_ptr<void> {
    assert(size == 4 * page);
    if (!allow)
      return {};
    ++tickets;
    return std::shared_ptr<void>(new int, [&](void* p) {
      // Ticket lasts beyond physical commit and accounting publication.
      assert(ledger.Snapshot().reserved_bytes[0] == 0);
      --tickets;
      delete static_cast<int*>(p);
    });
  };
  physical.before_commit = [&] { assert(tickets == 1); };
  CommittedBackingPool pool(ledger, config, physical.Allocator());
  assert(tickets == 0 && pool.BlockCount() == 4);
  auto block = pool.TryAcquire(Category::kPrivateTail);
  assert(block && pool.TrimFreeSegments(all) == 0);
  block.reset();
  assert(pool.TrimFreeSegments(1) == 4 * page);
  // Old boolean gate still controls on-demand growth, including recommit.
  assert(!pool.TryAcquire(Category::kPrivateTail));

  ResourceLedger lazy_ledger(Limits(2));
  Physical lazy{lazy_ledger};
  config = Config(2);
  bool legacy = true;
  config.admit_segment = [&](std::size_t) { return legacy; };
  config.acquire_segment = [&](std::size_t) -> std::shared_ptr<void> {
    return allow ? std::make_shared<int>(1) : nullptr;
  };
  CommittedBackingPool lazy_pool(lazy_ledger, config, lazy.Allocator());
  allow = false;
  assert(!lazy_pool.TryAcquire(Category::kPrivateTail));
  lazy_pool.GrowAhead(all);
  lazy_pool.WaitForGrowth();
  assert(lazy.bytes == 0);
  allow = true;
  legacy = false;
  assert(!lazy_pool.TryAcquire(Category::kPrivateTail));
  legacy = true;
  block = lazy_pool.TryAcquire(Category::kPrivateTail);
  assert(block && lazy.bytes == 2 * page);
  block.reset();
  assert(lazy_pool.TrimFreeSegments(all) == 2 * page);
  allow = false;
  assert(!lazy_pool.TryAcquire(Category::kPrivateTail));
}

void FailuresAndQuarantine() {
  // Preserve eager-mode exception behavior as well as rollback accounting.
  {
    ResourceLedger eager_ledger(Limits(2));
    Physical eager{eager_ledger};
    eager_ledger.FailAfter(LedgerStep::kConvert, 2);
    bool failed = false;
    try {
      CommittedBackingPool pool(eager_ledger, Config(2, 0), eager.Allocator());
    } catch (const std::bad_alloc&) {
      failed = true;
    }
    assert(failed && eager.bytes == 0 &&
           eager_ledger.Snapshot().total_bytes == 0);
  }
  ResourceLedger ledger(Limits(2));
  Physical physical{ledger};
  CommittedBackingPool pool(ledger, Config(2), physical.Allocator());
  for (const auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    for (std::size_t after = 0; after != 2; ++after) {
      ledger.FailAfter(step, after);
      assert(!pool.TryAcquire(Category::kPrivateTail));
      assert(physical.bytes == 0 && pool.BlockCount() == 0);
      assert(ledger.Snapshot().total_bytes == headroom);
    }
  }
  physical.deny_commit = true;
  assert(!pool.TryAcquire(Category::kPrivateTail));
  assert(ledger.Snapshot().total_bytes == headroom);
  physical.deny_commit = false;
  auto block = pool.TryAcquire(Category::kPrivateTail);
  assert(block);
  block.reset();
  physical.deny_unpin = true;
  const auto before = ledger.Snapshot().total_bytes;
  assert(pool.TrimFreeSegments(all) == 0);
  assert(pool.LongTermPinnedBytes() == 2 * page);
  assert(pool.ImmediatelyTrimmableBytes() == 0);
  assert(ledger.Snapshot().total_bytes == before);
  physical.deny_unpin = false;
  physical.deny_release = true;
  assert(pool.TrimFreeSegments(all) == 0);
  assert(physical.bytes == 2 * page && pool.BlockCount() == 2);
  assert(ledger.Snapshot().total_bytes == before);
  assert(pool.LongTermPinnedBytes() == 0);  // partial teardown recorded
  assert(pool.ImmediatelyTrimmableBytes() ==
         0);  // known failure is not promised
  assert(!pool.TryAcquire(Category::kPrivateTail));  // not device usable
  physical.deny_release = false;
  assert(pool.TrimFreeSegments(all) == 2 * page);
  assert(ledger.Snapshot().total_bytes == headroom);

  // A conversion failure followed by a rollback-release failure retains BOTH
  // already converted charges and pending admissions until a physical retry.
  ledger.FailAfter(LedgerStep::kConvert, 1);
  physical.deny_release = true;
  assert(!pool.TryAcquire(Category::kPrivateTail));
  assert(pool.CommittedBytes() == 2 * page && physical.bytes == 2 * page);
  assert(ledger.Snapshot().total_bytes == headroom + 2 * page);
  assert(ledger.Snapshot().reserved_bytes[0] == page);
  assert(!pool.TryAcquire(Category::kPrivateTail));
  physical.deny_release = false;
  assert(pool.TrimFreeSegments(all) == 2 * page);
  assert(ledger.Snapshot().total_bytes == headroom);
  block = pool.TryAcquire(Category::kPrivateTail, 2 * page);
  assert(block && pool.LongTermPinnedBytes() == 2 * page);
}

void GroupedSegmentEdgeChurn() {
  ResourceLedger ledger(Limits(11));
  Physical physical{ledger};
  CommittedBackingPool pool(ledger, Config(11, 3), physical.Allocator());
  for (int iteration = 0; iteration < 40; ++iteration) {
    std::vector<BackingBlock> held;
    for (int i = 0; i < 3; ++i) {
      auto block = pool.TryAcquire(
          i % 2 ? Category::kPrivateState : Category::kBackingAssigned,
          3 * page);
      assert(block);
      Fill(*block, std::byte{41});
      held.push_back(std::move(*block));
    }
    auto edge = pool.TryAcquire(Category::kPrivateTail, 2 * page);
    assert(edge && pool.BlockCount() == 11);
    Fill(*edge, std::byte{11});
    const auto* address = edge->Bytes().data();
    assert(!pool.TryAcquire(Category::kPrivateTail));
    held.erase(held.begin() + 1);
    assert(pool.TrimFreeSegments(1) == 3 * page);
    Check(*edge, address, std::byte{11});
    auto group = pool.TryAcquire(Category::kPrivateState, 3 * page);
    assert(group);
    for (auto& block : held)
      Check(block, block.Bytes().data(), std::byte{41});
    held.clear();
    group.reset();
    edge.reset();
    assert(pool.TrimFreeSegments(all) == 11 * page);
    assert(pool.PageCommitAllocations() == 0);
  }
}

struct Pause {
  std::mutex mutex;
  std::condition_variable condition;
  bool entered{}, proceed{};
  void Callback() {
    std::unique_lock lock(mutex);
    entered = true;
    condition.notify_all();
    condition.wait(lock, [&] { return proceed; });
  }
  void Wait() {
    std::unique_lock lock(mutex);
    condition.wait(lock, [&] { return entered; });
  }
  void Resume() {
    const std::lock_guard lock(mutex);
    proceed = true;
    condition.notify_all();
  }
};
void SerializedCommitAndTrim() {
  ResourceLedger ledger(Limits(2));
  Physical physical{ledger};
  Pause commit;
  std::atomic<int> tickets{};
  auto config = Config(2);
  config.acquire_segment = [&](std::size_t) {
    ++tickets;
    return std::shared_ptr<void>(new int, [&](void* p) {
      --tickets;
      delete static_cast<int*>(p);
    });
  };
  physical.before_commit = [&] {
    assert(tickets == 1);
    commit.Callback();
  };
  CommittedBackingPool pool(ledger, config, physical.Allocator());
  std::optional<BackingBlock> held;
  std::jthread acquiring(
      [&] { held = pool.TryAcquire(Category::kBackingAssigned, 2 * page); });
  commit.Wait();
  assert(tickets == 1);
  std::barrier start(2);
  std::size_t released = all;
  std::jthread trimming([&] {
    start.arrive_and_wait();
    released = pool.TrimFreeSegments(all);
  });
  start.arrive_and_wait();
  commit.Resume();
  acquiring.join();
  trimming.join();
  assert(held && released == 0 && tickets == 0);
  held.reset();

  Pause release;
  physical.before_release = [&] { release.Callback(); };
  trimming = std::jthread([&] { released = pool.TrimFreeSegments(all); });
  release.Wait();
  assert(pool.CommittedBytes() == 2 * page);
  assert(ledger.Snapshot().total_bytes == headroom + 2 * page);
  acquiring = std::jthread(
      [&] { held = pool.TryAcquire(Category::kPrivateState, 2 * page); });
  release.Resume();
  trimming.join();
  acquiring.join();
  assert(released == 2 * page && held && physical.commits == 2);
  physical.before_release = {};
  physical.before_commit = {};
}

void ConcurrentAcquireTrimAndGrowAhead() {
  ResourceLedger ledger(Limits(32));
  Physical physical{ledger};
  CommittedBackingPool pool(ledger, Config(32, 4), physical.Allocator());
  std::barrier start(5);
  std::mutex owned_mutex;
  std::vector<std::pair<std::byte*, std::size_t>> owned;
  std::vector<std::jthread> threads;
  for (int worker = 0; worker < 3; ++worker) {
    threads.emplace_back([&, worker] {
      start.arrive_and_wait();
      for (int i = 0; i < 150; ++i) {
        auto block = pool.TryAcquire(
            worker == 1 ? Category::kPrivateState : Category::kBackingAssigned,
            (1 + i % 4) * page);
        if (!block)
          continue;
        const auto data = block->Bytes();
        {
          const std::lock_guard lock(owned_mutex);
          const auto begin = reinterpret_cast<std::uintptr_t>(data.data());
          for (const auto& [address, size] : owned) {
            const auto other = reinterpret_cast<std::uintptr_t>(address);
            assert(begin + data.size() <= other || other + size <= begin);
          }
          owned.emplace_back(data.data(), data.size());
        }
        Fill(*block, std::byte(worker + 1));
        block->Convert();
        auto pin = block->PinPersistence();
        std::this_thread::yield();
        Check(*block, data.data(), std::byte(worker + 1));
        block.reset();
        // Exercise the pin-only interval against trim and other assignments.
        std::this_thread::yield();
        assert(std::ranges::all_of(
            data, [&](auto byte) { return byte == std::byte(worker + 1); }));
        {
          const std::lock_guard lock(owned_mutex);
          std::erase(owned, std::pair(data.data(), data.size()));
        }
      }
    });
  }
  threads.emplace_back([&] {
    start.arrive_and_wait();
    for (int i = 0; i < 150; ++i) {
      (void)pool.ImmediatelyTrimmableBytes();
      (void)pool.TrimFreeSegments(4 * page);
      pool.GrowAhead(8 * page);
      pool.WaitForGrowth();
    }
  });
  start.arrive_and_wait();
  threads.clear();
  pool.WaitForGrowth();
  assert(owned.empty());
  const auto bytes = physical.bytes.load();
  assert(pool.TrimFreeSegments(all) == bytes);
  assert(pool.CommittedBytes() == 0 &&
         ledger.Snapshot().total_bytes == headroom);
}
}  // namespace

int main() {
  InteriorHolesAndPartialEdge();
  SharedOwnersPinsAndFacadeLifetime();
  BorrowedReservationsAndPreservation();
  EagerAndAdmissionTickets();
  FailuresAndQuarantine();
  GroupedSegmentEdgeChurn();
  SerializedCommitAndTrim();
  ConcurrentAcquireTrimAndGrowAhead();
  std::cout << "cache physical backing CPU tests passed\n";
}
