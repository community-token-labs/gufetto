#include "src/core/hip/committed_backing.hpp"

#include <hip/hip_runtime.h>

#include <exception>
#include <memory>
#include <utility>

#include "src/core/platform/page_pin.hpp"
#include "src/core/platform/pages.hpp"

namespace gufo::hip {
namespace {
struct Registration {
  int pin{-1};
  bool registered{};
};

cache::BackingPages CommitPages(std::size_t bytes) {
  // Allocate bookkeeping before physical storage so failure cannot leak pages.
  auto registration = std::make_shared<Registration>();
  void* data = platform::CommitZeroedPages(bytes);
  if (data == nullptr)
    return {};
  registration->pin = platform::PinHostPages(data, bytes);
  if (hipHostRegister(data, bytes, hipHostRegisterDefault) != hipSuccess) {
    // A failed commit cannot return unaccounted resident storage.
    if (!platform::TryUnpinHostPages(registration->pin) ||
        !platform::TryReleaseCommittedPages(data, bytes))
      std::terminate();
    return {};
  }
  registration->registered = true;
  return {data, registration, registration->pin >= 0};
}

bool ReleasePages(cache::BackingPages& pages, std::size_t bytes) {
  auto& registration = *static_cast<Registration*>(pages.token.get());
  if (registration.registered) {
    if (hipHostUnregister(pages.data) != hipSuccess)
      return false;
    registration.registered = false;
  }
  // Preserve partial teardown state for retries, never reassign this segment.
  if (!platform::TryUnpinHostPages(registration.pin))
    return false;
  registration.pin = -1;
  pages.long_term_pinned = false;
  return platform::TryReleaseCommittedPages(pages.data, bytes);
}
}  // namespace

CommittedBackingPool::CommittedBackingPool(cache::ResourceLedger& ledger,
                                           BackingPoolConfig config)
    : cache::CommittedBackingPool(ledger, std::move(config),
                                  {CommitPages, ReleasePages}) {}

}  // namespace gufo::hip
