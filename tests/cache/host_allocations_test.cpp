#include "src/core/hip/host_allocations.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <map>
#include <new>

#include "src/core/platform/page_pin.hpp"

namespace {
std::map<void*, std::size_t> live;
std::map<int, void*> pins;
void* refuse_free{};
int refuse_unpin{-1}, next_pin{1};
bool refuse_allocation{}, empty_allocation{}, unavailable_pins{};
bool fail_bookkeeping{};
std::size_t allocations{}, frees{}, checked_unpins{}, legacy_unpins{};
}  // namespace

void* operator new(std::size_t bytes) {
  if (fail_bookkeeping) {
    fail_bookkeeping = false;
    throw std::bad_alloc();
  }
  if (void* data = std::malloc(bytes ? bytes : 1))
    return data;
  throw std::bad_alloc();
}
void operator delete(void* data) noexcept {
  std::free(data);
}
void operator delete(void* data, std::size_t) noexcept {
  std::free(data);
}

extern "C" hipError_t hipHostMalloc(void** data, std::size_t bytes, unsigned) {
  ++allocations;
  *data = nullptr;
  if (refuse_allocation)
    return hipErrorOutOfMemory;
  if (empty_allocation)
    return hipSuccess;
  *data = std::malloc(bytes ? bytes : 1);
  assert(*data && !live.contains(*data));
  live.emplace(*data, bytes);
  return hipSuccess;
}
extern "C" hipError_t hipHostFree(void* data) {
  ++frees;
  assert(live.contains(data));
  assert(std::ranges::none_of(
      pins, [=](const auto& pin) { return pin.second == data; }));
  if (data == refuse_free)
    return hipErrorInvalidValue;
  live.erase(data);
  std::free(data);
  return hipSuccess;
}
namespace gufo::platform {
int PinHostPages(void* data, std::size_t bytes) noexcept {
  assert(live.at(data) == bytes);
  if (unavailable_pins)
    return -1;
  const int pin = next_pin++;
  pins.emplace(pin, data);
  return pin;
}
bool TryUnpinHostPages(int pin) noexcept {
  if (pin < 0)
    return true;
  ++checked_unpins;
  assert(pins.contains(pin));
  if (pin == refuse_unpin)
    return false;
  pins.erase(pin);
  return true;
}
void UnpinHostPages(int pin) noexcept {
  if (pin < 0)
    return;
  ++legacy_unpins;
  assert(pins.erase(pin) == 1);
}
}  // namespace gufo::platform

namespace {
void CheckedSingleUpload() {
  gufo::hip::HostAllocations owner;
  assert(owner.TryReset());
  void* data{};
  assert(owner.Allocate(&data, 4096) == hipSuccess);
  assert(live.at(data) == 4096 && owner.PinnedBytes() == 4096);
  const auto before = frees;
  refuse_unpin = pins.begin()->first;
  assert(!owner.TryReset());
  assert(live.contains(data) && pins.contains(refuse_unpin));
  assert(owner.PinnedBytes() == 4096 && frees == before);
  refuse_unpin = -1;
  assert(owner.TryReset());
  assert(live.empty() && pins.empty() && owner.PinnedBytes() == 0);
  assert(frees == before + 1);
  assert(owner.TryReset() && frees == before + 1);
}
void PartialReleaseAndRetry() {
  gufo::hip::HostAllocations owner;
  void *first{}, *second{};
  assert(owner.Allocate(&first, 4096) == hipSuccess);
  assert(owner.Allocate(&second, 8192) == hipSuccess);
  refuse_free = first;
  const auto before_frees = frees;
  const auto before_unpins = checked_unpins;
  assert(!owner.TryReset());
  assert(live.contains(first) && !live.contains(second) && pins.empty());
  assert(frees == before_frees + 2 && checked_unpins == before_unpins + 2);
  // Zero pin bytes is NOT physical release: HIP still owns the failed buffer.
  assert(owner.PinnedBytes() == 0);
  refuse_free = nullptr;
  assert(owner.TryReset());
  assert(live.empty() && frees == before_frees + 3);
  assert(checked_unpins == before_unpins + 2);  // no consumed-token reuse
}
void LegacyResetAndUnavailablePins() {
  gufo::hip::HostAllocations owner;
  void* data{};
  assert(owner.Allocate(&data, 1024) == hipSuccess);
  const auto before_legacy = legacy_unpins;
  const auto before_checked = checked_unpins;
  owner.Reset();
  assert(live.empty() && pins.empty());
  assert(legacy_unpins == before_legacy + 1 &&
         checked_unpins == before_checked);
  assert(owner.TryReset());
  unavailable_pins = true;
  assert(owner.Allocate(&data, 1024) == hipSuccess);
  assert(owner.PinnedBytes() == 0 && live.contains(data));
  assert(owner.TryReset() && live.empty());
  unavailable_pins = false;
}
void AdmissionBeforePhysicalAllocation() {
  gufo::hip::HostAllocations owner;
  void* data = reinterpret_cast<void*>(1);
  const auto before = allocations;
  fail_bookkeeping = true;
  assert(owner.Allocate(&data, 1024) == hipErrorOutOfMemory);
  assert(!fail_bookkeeping && data == nullptr && allocations == before);
  assert(owner.TryReset() && live.empty());
  refuse_allocation = true;
  assert(owner.Allocate(&data, 1024) == hipErrorOutOfMemory);
  assert(data == nullptr && owner.TryReset());
  refuse_allocation = false;
  empty_allocation = true;
  assert(owner.Allocate(&data, 0) == hipSuccess);
  assert(data == nullptr && owner.TryReset());
  empty_allocation = false;
  assert(owner.Allocate(&data, 1024) == hipSuccess);
  assert(owner.TryReset() && live.empty() && pins.empty());
}
}  // namespace

int main() {
  CheckedSingleUpload();
  PartialReleaseAndRetry();
  LegacyResetAndUnavailablePins();
  AdmissionBeforePhysicalAllocation();
  assert(live.empty() && pins.empty());
  std::cout << "cache checked host allocation retirement tests passed\n";
}
