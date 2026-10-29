#include "src/cache/disk_policy.hpp"

#include <array>
#include <cassert>
#include <limits>

#include "src/cache/disk_catalog.hpp"
#include "tests/support/temporary_directory.hpp"

using namespace gufo::cache;
using gufo::platform::FilesystemInfo;
namespace {
constexpr std::uint64_t GiB = 1ULL << 30;
DiskFileId File(unsigned n) {
  DiskFileId result{};
  result[0] = n;
  return result;
}
DiskManifest Manifest(unsigned id, std::size_t bytes) {
  DiskManifest m;
  m.checkpoint = {id};
  m.lineage = {id};
  m.compatibility.fill(42);
  m.tokens = {id, 2, 3};
  m.components.push_back({{{1}, 1, ComponentKind::kPrivateState, 0, 0, bytes},
                          {{1}, 3},
                          {},
                          {},
                          DiskPayload{File(id + 100), bytes, 0}});
  return m;
}
void Publish(DiskStore& disk, unsigned id) {
  std::array<std::uint8_t, 4096> payload{};
  auto m = Manifest(id, payload.size());
  m.components[0].private_state->checksum = DiskChecksum(payload);
  const std::array buffers{DiskWriteBuffer{true, File(id + 100), payload}};
  disk.Publish(File(id), m, buffers);
}
void SizingAndPaths() {
  FilesystemInfo info{1024 * GiB, 130 * GiB, true, true};
  assert(AutomaticDiskCapacity(info) == 13 * GiB);
  assert(AutomaticDiskReserve(info) == info.capacity_bytes / 50);
  info.available_bytes = 500 * GiB;
  assert(AutomaticDiskCapacity(info) == 50 * GiB);
  info.available_bytes = 800 * GiB;
  assert(AutomaticDiskCapacity(info) == 64 * GiB);
  info.capacity_bytes = 20 * GiB;
  assert(AutomaticDiskReserve(info) == 8 * GiB);
  info.capacity_bytes = info.available_bytes = UINT64_MAX;
  assert(AutomaticDiskCapacity(info) == 64 * GiB);
  assert(AutomaticDiskReserve(info) == UINT64_MAX / 50);

  TemporaryDirectory dir("gufo-disk-policy");
  const auto probe = [&](const auto&) { return info; };
  auto plan = PlanDiskCache("auto", dir.path(), 0, 0, probe);
  assert(plan.Enabled() && plan.directory == dir.path() / "gufo/continuations");
  assert(!std::filesystem::exists(plan.directory));
  const auto target = dir.path() / "target";
  const auto link = dir.path() / "link";
  std::filesystem::create_directory(target);
  std::filesystem::create_directory_symlink(target, link);
  assert(PlanDiskCache(link / "new", {}, 0, 0, probe).Enabled());
  assert(PlanDiskCache("auto", link, 0, 0, probe).Enabled());
  assert(!PlanDiskCache(link, {}, 0, 0, probe).Enabled());
  plan = PlanDiskCache(dir.path() / "explicit", {}, 1234, 5678, probe);
  assert(plan.Enabled() && plan.capacity_bytes == 1234 &&
         plan.minimum_free_bytes == 5678);
  assert(!PlanDiskCache("off", dir.path(), 0, 0, probe).Enabled());
  assert(!PlanDiskCache("", dir.path(), 0, 0, probe).Enabled());
  assert(!PlanDiskCache("auto", "relative", 0, 0, probe).Enabled());
  info.writable = false;
  assert(PlanDiskCache("auto", dir.path(), 0, 0, probe).reason == "read_only");
  info.supported = false;
  assert(PlanDiskCache(dir.path(), {}, 0, 0, probe).reason ==
         "unsupported_filesystem");
  info = {};
  assert(!PlanDiskCache("auto", dir.path(), 0, 0, probe).Enabled());

  gufo::platform::Directory handle;
  std::error_code ec;
  assert(handle.Open(dir.path(), ec));
  const auto actual = gufo::platform::InspectFilesystem(handle);
  assert(actual.capacity_bytes > 0 &&
         actual.available_bytes <= actual.capacity_bytes);
  info = {GiB, 100 + kDiskAllocationSlack + 10, true, true};
  DiskSpacePolicy policy{100, probe};
  assert(policy.Check(handle, 10) == DiskSpaceState::kReady);
  assert(policy.Check(handle, 11) == DiskSpaceState::kPressure);
  assert(policy.Check(handle, UINT64_MAX) == DiskSpaceState::kPressure);
  info.available_bytes = 99;
  assert(policy.Check(handle) == DiskSpaceState::kPressure);
  info.writable = false;
  assert(policy.Check(handle) == DiskSpaceState::kUnavailable);
}
void PressureAndRestart() {
  TemporaryDirectory dir("gufo-disk-pressure");
  ResourceLedger ledger{{1 << 24, 1 << 23, 0, 0}};
  FilesystemInfo info{GiB, GiB, true, true};
  DiskSpacePolicy policy{1 << 20, [&](const auto&) { return info; }};
  std::uint64_t one{};
  {
    DiskStore disk(ledger, dir.path(), 1 << 20, policy);
    Publish(disk, 1);
    one = disk.Stats().managed_bytes;
    Publish(disk, 2);
    PrefixIndex index(ledger);
    DiskCatalog catalog(ledger, disk, index);
    auto first = disk.Open({1});
    auto second = disk.Open({2});
    info.available_bytes = 0;
    assert(disk.SpacePressure());
    assert(!catalog.EvictOne());  // Read pins are not sacrificed for space.
    bool refused{};
    try {
      Publish(disk, 3);
    } catch (const DiskCapacityExhausted&) {
      refused = true;
    }
    assert(refused && disk.Stats().managed_bytes == 2 * one);
    // Free-space pressure does not disable verified reads.
    std::array<std::uint8_t, 16> staging;
    const auto p = *first->Manifest().components[0].private_state;
    const auto read =
        disk.ReadPayload(*first, true, p, staging, [](auto, auto) {});
    assert(read.bytes == p.bytes);
    first.reset();
    assert(catalog.EvictOne());
    // No optimistic deletion credit: the injected filesystem still reports
    // full.
    assert(disk.SpacePressure());
    second.reset();
    info.available_bytes = GiB;
    Publish(disk, 3);
  }
  {
    DiskStore disk(ledger, dir.path(), one, policy);
    PrefixIndex index(ledger);
    DiskCatalog catalog(ledger, disk, index);
    assert(disk.Entries().size() == 2);
    while ((disk.Stats().managed_bytes > one || disk.SpacePressure()) &&
           catalog.EvictOne()) {
    }
    assert(disk.Stats().managed_bytes == one && disk.Open({3}));
    info.available_bytes = 0;
    while (disk.SpacePressure() && catalog.EvictOne()) {
    }
    assert(disk.Stats().managed_bytes == 0);
  }
}
void MidWritePressureCanRetry() {
  TemporaryDirectory dir("gufo-disk-midwrite");
  ResourceLedger ledger{{1 << 24, 1 << 23, 0, 0}};
  FilesystemInfo info{GiB, GiB, true, true};
  DiskSpacePolicy policy{1 << 20, [&](const auto&) { return info; }};
  DiskStore disk(ledger, dir.path(), 1 << 20, policy);
  auto m = Manifest(1, 64);
  std::array<std::uint8_t, 16> staging;
  bool inject = true;
  const std::array sources{
      DiskWriteSource{true, File(101), 64,
                      [&](std::uint64_t offset, std::span<std::uint8_t> bytes) {
                        std::ranges::fill(bytes, 7);
                        if (inject && offset == 16)
                          info.available_bytes = 0;
                      }}};
  bool refused{};
  try {
    (void)disk.PublishStream(File(1), m, sources, staging);
  } catch (const DiskCapacityExhausted&) {
    refused = true;
  }
  assert(refused && !disk.Open({1}) && disk.Stats().managed_bytes == 0);
  inject = false;
  info.available_bytes = GiB;
  (void)disk.PublishStream(File(1), m, sources, staging);
  assert(disk.Open({1}));
  info.writable = false;
  bool unavailable{};
  try {
    Publish(disk, 2);
  } catch (const std::runtime_error&) {
    unavailable = true;
  }
  assert(unavailable && disk.Open({1}) && !disk.SpacePressure());
}
}  // namespace
int main() {
  SizingAndPaths();
  PressureAndRestart();
  MidWritePressureCanRetry();
}
