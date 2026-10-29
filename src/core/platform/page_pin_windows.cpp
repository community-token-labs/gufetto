#include "src/core/platform/page_pin.hpp"

namespace gufo::platform {

// Windows exposes no io_uring-style register-pinned-buffer facility; HIP
// registration alone retains the allocation.
int PinHostPages(void*, std::size_t) noexcept {
  return -1;
}

void UnpinHostPages(int) noexcept {}

bool TryUnpinHostPages(int pin) noexcept {
  // This leaf never issues a nonnegative token. Do not pretend to release a
  // foreign/invalid token; the ordinary unavailable-pin sentinel is a no-op.
  return pin < 0;
}

}  // namespace gufo::platform
