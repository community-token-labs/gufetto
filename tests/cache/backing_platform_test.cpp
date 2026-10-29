#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>

#include "src/cache/backing.hpp"
#include "src/core/platform/page_pin.hpp"
#include "src/core/platform/pages.hpp"

namespace {
constexpr std::size_t page = 4096;
void* tracked_mapping{};
bool refuse_unmap{}, refuse_population{};
std::size_t unmap_attempts{};
// The pin-only ring is simulated at the syscall boundary, so failure paths do
// not depend on host io_uring permission/memlock or acquire real long-term
// pins.
constexpr int ring = 1000000;
bool ring_open{}, buffers_registered{};
bool refuse_setup{}, refuse_register{}, close_interrupted{};
int unregister_error{};
std::size_t unregister_attempts{}, close_attempts{};
}  // namespace

extern "C" {
int __real_madvise(void*, std::size_t, int);
int __wrap_madvise(void* data, std::size_t bytes, int advice) {
  if (refuse_population && advice == MADV_POPULATE_WRITE) {
    tracked_mapping = data;
    errno = ENOMEM;
    return -1;
  }
  return __real_madvise(data, bytes, advice);
}
int __real_munmap(void*, std::size_t);
int __wrap_munmap(void* data, std::size_t bytes) {
  if (data == tracked_mapping) {
    ++unmap_attempts;
    if (refuse_unmap) {
      errno = EBUSY;
      return -1;
    }
  }
  return __real_munmap(data, bytes);
}
int __real_close(int);
int __wrap_close(int fd) {
  if (fd != ring)
    return __real_close(fd);
  ++close_attempts;
  assert(ring_open);
  ring_open = false;
  buffers_registered = false;
  if (close_interrupted) {
    // Linux has consumed the descriptor even when reporting EINTR.
    errno = EINTR;
    return -1;
  }
  return 0;
}
long __wrap_syscall(long number, ...) {
  if (number == SYS_io_uring_setup) {
    assert(!ring_open);
    if (refuse_setup) {
      errno = EPERM;
      return -1;
    }
    ring_open = true;
    return ring;
  }
  // These are the only syscall() operations exercised by this focused test.
  // Do not read unused varargs: setup/register have different signatures.
  assert(number == SYS_io_uring_register);
  if (number != SYS_io_uring_register) {
    errno = ENOSYS;
    return -1;
  }
  va_list args;
  va_start(args, number);
  const int fd = va_arg(args, int);
  const int operation = va_arg(args, int);
  va_end(args);
  assert(fd == ring && ring_open);
  if (operation == IORING_REGISTER_BUFFERS) {
    assert(!buffers_registered);
    if (refuse_register) {
      errno = ENOMEM;
      return -1;
    }
    buffers_registered = true;
    return 0;
  }
  assert(operation == IORING_UNREGISTER_BUFFERS && buffers_registered);
  ++unregister_attempts;
  if (unregister_error) {
    errno = unregister_error;
    return -1;
  }
  buffers_registered = false;
  return 0;
}
}  // extern "C"

namespace {
void CheckUnmapped(void* data) {
  unsigned char resident{};
  errno = 0;
  assert(mincore(data, page, &resident) == -1 && errno == ENOMEM);
}
void CheckedPageRelease() {
  using namespace gufo::platform;
  auto* data = static_cast<std::byte*>(CommitZeroedPages(2 * page));
  assert(data);
  std::fill_n(data, 2 * page, std::byte{43});
  tracked_mapping = data;
  unmap_attempts = 0;
  assert(TryReleaseCommittedPages(nullptr, 0));
  assert(!TryReleaseCommittedPages(data, 0));
  assert(unmap_attempts == 0);
  refuse_unmap = true;
  assert(!TryReleaseCommittedPages(data, 2 * page));
  assert(unmap_attempts == 1 && data[page] == std::byte{43});
  ReleaseCommittedPages(data, 2 * page);  // Legacy best-effort API preserved.
  assert(unmap_attempts == 2 && data[page] == std::byte{43});
  refuse_unmap = false;
  assert(TryReleaseCommittedPages(data, 2 * page));
  assert(unmap_attempts == 3);
  CheckUnmapped(data);
  tracked_mapping = nullptr;
}
void FailedCommitRollback() {
  refuse_population = true;
  assert(gufo::platform::CommitZeroedPages(page) == nullptr);
  assert(tracked_mapping);
  CheckUnmapped(tracked_mapping);
  refuse_population = false;
  tracked_mapping = nullptr;

  // A second physical failure during rollback cannot return nullptr and let
  // the caller drop its admission charge. Verify the fail-closed boundary.
  const auto child = fork();
  assert(child >= 0);
  if (child == 0) {
    std::set_terminate([] { _exit(86); });
    refuse_population = true;
    refuse_unmap = true;
    (void)gufo::platform::CommitZeroedPages(page);
    _exit(87);
  }
  int status{};
  assert(waitpid(child, &status, 0) == child);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 86);
}
void CheckedUnpin() {
  using namespace gufo::platform;
  std::byte data{};
  const auto before = close_attempts;
  assert(TryUnpinHostPages(-1));
  assert(close_attempts == before);
  assert(PinHostPages(&data, 1) == ring);
  for (const int error : {EBUSY, EINTR}) {
    unregister_error = error;
    const auto attempts = unregister_attempts;
    assert(!TryUnpinHostPages(ring));
    assert(unregister_attempts == attempts + 1);  // bounded, caller retries
    assert(ring_open && buffers_registered && close_attempts == before);
  }
  unregister_error = 0;
  assert(TryUnpinHostPages(ring));
  assert(!ring_open && !buffers_registered && close_attempts == before + 1);

  assert(PinHostPages(&data, 1) == ring);
  close_interrupted = true;
  assert(
      TryUnpinHostPages(ring));  // Pins already released; do NOT retry close.
  assert(!ring_open && !buffers_registered && close_attempts == before + 2);
  close_interrupted = false;

  // Legacy callers still close their descriptor on unregister failure.
  assert(PinHostPages(&data, 1) == ring);
  unregister_error = EBUSY;
  UnpinHostPages(ring);
  assert(!ring_open && close_attempts == before + 3);
  unregister_error = 0;
  refuse_register = true;
  assert(PinHostPages(&data, 1) == -1);
  assert(!ring_open && close_attempts == before + 4);
  refuse_register = false;
  refuse_setup = true;
  assert(PinHostPages(&data, 1) == -1);
  assert(!ring_open && close_attempts == before + 4);
  refuse_setup = false;
}
void PoolReleaseFailureKeepsRealPagesAndCharge() {
  using namespace gufo;
  constexpr std::size_t metadata = 16 << 10;
  constexpr std::size_t budget = metadata + 2 * page;
  cache::ResourceLedger ledger({budget, budget, 0, budget});
  cache::CommittedBackingPool pool(
      ledger, {budget, page, metadata, 2 * page, {}, {}},
      {[](std::size_t bytes) {
         return cache::BackingPages{
             platform::CommitZeroedPages(bytes), {}, false};
       },
       [](cache::BackingPages& pages, std::size_t bytes) {
         return platform::TryReleaseCommittedPages(pages.data, bytes);
       }});
  auto block =
      pool.TryAcquire(cache::ResourceCategory::kBackingAssigned, 2 * page);
  assert(block);
  auto* data = block->Bytes().data();
  std::ranges::fill(block->Bytes(), std::byte{29});
  block.reset();
  tracked_mapping = data;
  refuse_unmap = true;
  assert(pool.TrimFreeSegments(2 * page) == 0);
  assert(pool.CommittedBytes() == 2 * page);
  assert(ledger.Snapshot().total_bytes == budget);
  assert(data[page] == std::byte{29});
  // A partially retired segment is quarantined, not handed out as usable.
  assert(pool.ImmediatelyTrimmableBytes() == 0);
  assert(!pool.TryAcquire(cache::ResourceCategory::kPrivateTail));
  refuse_unmap = false;
  assert(pool.TrimFreeSegments(2 * page) == 2 * page);
  assert(pool.CommittedBytes() == 0);
  assert(ledger.Snapshot().total_bytes == metadata);
  CheckUnmapped(data);
  tracked_mapping = nullptr;
  block = pool.TryAcquire(cache::ResourceCategory::kPrivateTail, 2 * page);
  assert(block);
  assert(std::ranges::all_of(block->Bytes(),
                             [](auto b) { return b == std::byte{}; }));
  block.reset();
  assert(pool.TrimFreeSegments(std::numeric_limits<std::size_t>::max()) ==
         2 * page);
}
}  // namespace

int main() {
  CheckedPageRelease();
  FailedCommitRollback();
  CheckedUnpin();
  PoolReleaseFailureKeepsRealPagesAndCharge();
  std::cout << "cache checked platform release tests passed\n";
}
