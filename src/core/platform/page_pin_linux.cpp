#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <new>
#include <vector>

#include "src/core/platform/page_pin.hpp"

namespace gufo::platform {

int PinHostPages(void* data, std::size_t bytes) noexcept {
  if (data == nullptr || bytes == 0)
    return -1;
  // Bound each kernel pinning range so request-side CPU page faults can
  // progress between ranges. One ring still owns the complete allocation.
  constexpr std::size_t max_buffer = 8ULL << 20;
  std::vector<iovec> ranges;
  try {
    for (std::size_t offset = 0; offset < bytes;) {
      const auto length = std::min(max_buffer, bytes - offset);
      ranges.push_back({static_cast<std::byte*>(data) + offset, length});
      offset += length;
    }
  } catch (const std::bad_alloc&) {
    return -1;
  }
  io_uring_params parameters{};
  const int ring = syscall(SYS_io_uring_setup, 1, &parameters);
  if (ring < 0)
    return -1;
  if (syscall(SYS_io_uring_register, ring, IORING_REGISTER_BUFFERS,
              ranges.data(), ranges.size()) == 0)
    return ring;
  close(ring);
  return -1;
}

bool TryUnpinHostPages(int pin) noexcept {
  if (pin < 0)
    return true;
  // Unlike the legacy best-effort release, do not close on unregister failure:
  // close alone may defer ring cleanup, so it cannot establish released bytes.
  if (syscall(SYS_io_uring_register, pin, IORING_UNREGISTER_BUFFERS, nullptr,
              0) != 0)
    return false;
  // Successful unregister has synchronously released the registered buffers.
  // Linux consumes the descriptor even when close reports a late error/EINTR;
  // retrying close could target an unrelated, reused descriptor.
  (void)close(pin);
  return true;
}

void UnpinHostPages(int pin) noexcept {
  if (pin >= 0) {
    (void)syscall(SYS_io_uring_register, pin, IORING_UNREGISTER_BUFFERS,
                  nullptr, 0);
    close(pin);
  }
}

}  // namespace gufo::platform
