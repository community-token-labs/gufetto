#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <thread>

#include "src/cache/disk.hpp"

using namespace gufo::cache;
namespace fs = std::filesystem;
namespace {
std::atomic<int> fail_sync{-1}, fail_write{-1}, kill_sync{-1};
std::atomic<bool> interrupt_sync{}, short_writes{}, record_io{}, fail_unlink{},
    unlink_effect_error{};
std::vector<std::string> events;
std::mutex events_mutex, sync_mutex;
std::condition_variable sync_ready;
std::atomic<bool> gate_sync{};
std::atomic<unsigned> sync_arrivals{}, sync_attempts{}, sync_completions{};
std::string FdPath(int fd) {
  char path[4096];
  auto n = readlink(("/proc/self/fd/" + std::to_string(fd)).c_str(), path,
                    sizeof(path));
  assert(n > 0);
  return std::string(path, n);
}
ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
struct Directory {
  fs::path path;
  Directory() {
    auto name =
        (fs::temp_directory_path() / "gufo-publication-test-XXXXXX").string();
    path = mkdtemp(name.data());
  }
  ~Directory() { fs::remove_all(path); }
};
DiskFileId Id(unsigned n) {
  DiskFileId id{};
  id[0] = n;
  id[1] = n >> 8;
  return id;
}
struct Fixture {
  std::vector<std::uint8_t> chunk = std::vector<std::uint8_t>(8, 42);
  std::vector<std::uint8_t> tail = std::vector<std::uint8_t>(4, 17);
  std::vector<std::uint8_t> state = std::vector<std::uint8_t>(16, 9);
  DiskManifest manifest;
  std::vector<DiskWriteBuffer> buffers;
  explicit Fixture(unsigned n = 1, unsigned chunk_count = 1) {
    manifest.checkpoint = {n};
    manifest.lineage = {1};
    manifest.tokens = {1, 2, 3};
    manifest.components.push_back(
        {{{1}, 1, ComponentKind::kAppendRows, 4, 2, 0},
         {{1}, 3},
         {{Id(1), 8, DiskChecksum(chunk)}},
         DiskPayload{Id(10 + n), 4, DiskChecksum(tail)},
         {}});
    manifest.components.push_back(
        {{{2}, 1, ComponentKind::kPrivateState, 0, 0, 16},
         {{2}, 3},
         {},
         {},
         DiskPayload{Id(20 + n), 16, DiskChecksum(state)}});
    buffers = {{false, Id(1), chunk},
               {true, Id(10 + n), tail},
               {true, Id(20 + n), state}};
    for (unsigned i = 1; i < chunk_count; ++i) {
      auto id = Id(256 + i);
      manifest.components[0].chunks.push_back({id, 8, DiskChecksum(chunk)});
      buffers.push_back({false, id, chunk});
    }
    manifest.tokens.resize(2 * chunk_count + 1, 3);
    for (auto& component : manifest.components)
      component.position.valid_rows = manifest.tokens.size();
  }
};
template<class F>
void Reject(F f) {
  bool failed{};
  try {
    f();
  } catch (const std::exception&) {
    failed = true;
  }
  assert(failed);
}
std::uint64_t Managed(const fs::path& root) {
  std::uint64_t bytes{};
  for (auto name : {"chunks", "private", "manifests", "tmp"})
    for (auto& entry : fs::directory_iterator(root / "v2" / name))
      if (entry.path().extension() == ".bin")
        bytes += entry.file_size();
  return bytes;
}
void CheckPublished(DiskStore& store, const fs::path& root) {
  for (auto& entry : store.Entries()) {
    assert(entry.Durable());
    for (auto& c : entry.Manifest().components) {
      auto check = [&](const DiskPayload& p, const char* name) {
        std::ifstream file(root / "v2" / name / DiskFileName(p.file),
                           std::ios::binary);
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)),
                                        {});
        assert(bytes.size() == p.bytes && DiskChecksum(bytes) == p.checksum);
      };
      for (auto& p : c.chunks)
        check(p, "chunks");
      for (auto p : {c.tail, c.private_state})
        if (p)
          check(*p, "private");
    }
  }
}
void CheckExact(DiskStore& store, const Fixture& fixture) {
  auto snapshot = store.Open(fixture.manifest.checkpoint);
  assert(snapshot);
  assert(EncodeManifest(snapshot->Manifest()) ==
         EncodeManifest(fixture.manifest));
  std::array<std::uint8_t, 3> staging{};
  for (const auto& buffer : fixture.buffers) {
    DiskPayload payload{buffer.file, buffer.bytes.size(),
                        DiskChecksum(buffer.bytes)};
    std::vector<std::uint8_t> restored(buffer.bytes.size());
    const auto read = store.ReadPayload(*snapshot, buffer.private_file, payload,
                                        staging, [&](auto offset, auto bytes) {
                                          std::copy(bytes.begin(), bytes.end(),
                                                    restored.begin() + offset);
                                        });
    assert(read.bytes == restored.size());
    assert(std::equal(restored.begin(), restored.end(), buffer.bytes.begin()));
  }
}
void WriteAccountingIncludesRollbackAndReuse() {
  Directory dir;
  DiskStore store(ledger, dir.path, 100000);
  Fixture first(1);
  short_writes = true;
  store.Publish(Id(100), first.manifest, first.buffers);
  short_writes = false;
  const auto initial = store.PublicationStats();
  assert(initial.shared_write_bytes == 8 && initial.private_write_bytes == 20);
  assert(initial.manifest_write_bytes == EncodeManifest(first.manifest).size());
  assert(initial.inherited_bytes == 0 && initial.write_calls > 4);
  Fixture child(2);
  store.Publish(Id(101), child.manifest, std::span(child.buffers).subspan(1));
  const auto inherited = store.PublicationStats();
  assert(inherited.shared_write_bytes == initial.shared_write_bytes);
  assert(inherited.private_write_bytes == initial.private_write_bytes + 20);
  assert(inherited.inherited_bytes == 8);
  assert(inherited.manifest_write_bytes ==
         initial.manifest_write_bytes + EncodeManifest(child.manifest).size());
  const auto retained = Managed(dir.path);
  Fixture failed(3);
  short_writes = true;
  fail_write = 1;  // Accept two private bytes, then fail the next write.
  Reject([&] {
    store.Publish(Id(102), failed.manifest,
                  std::span(failed.buffers).subspan(1));
  });
  short_writes = false;
  fail_write = -1;
  const auto rolled_back = store.PublicationStats();
  assert(rolled_back.private_write_bytes == inherited.private_write_bytes + 2);
  assert(rolled_back.shared_write_bytes == inherited.shared_write_bytes);
  assert(rolled_back.manifest_write_bytes == inherited.manifest_write_bytes);
  assert(rolled_back.inherited_bytes == inherited.inherited_bytes + 8);
  assert(rolled_back.write_calls == inherited.write_calls + 2);
  assert(Managed(dir.path) == retained + 2 && !store.Open({3}));
  store.ReclaimOrphans();
  store.WaitForReclamation();
  assert(Managed(dir.path) == retained);
  assert(store.PublicationStats().private_write_bytes ==
         rolled_back.private_write_bytes);
}
void Crashes(unsigned chunk_count = 1, int crash_signal = SIGABRT) {
  Fixture expected(1, chunk_count);
  const auto occurrences = chunk_count == 1
                               ? std::vector<int>{0, 1, 2}
                               : std::vector<int>{0, 31, 32, 63, 64, 66};
  for (int step = 0; step <= int(DiskPublicationStep::kIndexed); ++step) {
    // Cover both sides of each 32-payload batch boundary, plus the final
    // partial batch. Non-repeated hooks also exercise successful completion.
    for (int occurrence : occurrences) {
      if (chunk_count > 1 && step > int(DiskPublicationStep::kPayloadRenamed) &&
          occurrence != 0)
        continue;
      Directory dir;
      auto pid = fork();
      assert(pid >= 0);
      if (!pid) {
        DiskStore store(ledger, dir.path, 100000);
        Fixture f(1, chunk_count);
        int count{};
        const bool during_sync =
            crash_signal == SIGKILL &&
            step == int(DiskPublicationStep::kPayloadSynced);
        // Also kill from inside an actual sync call, with other workers
        // potentially still running, rather than only after the batch joins.
        if (during_sync)
          kill_sync = occurrence;
        store.SetCrashHook([&](auto s) {
          if (!during_sync && int(s) == step && count++ == occurrence)
            raise(crash_signal);
        });
        store.Publish(Id(100), f.manifest, f.buffers);
        _exit(0);
      }
      int status{};
      assert(waitpid(pid, &status, 0) == pid);
      bool repeated = step <= int(DiskPublicationStep::kPayloadRenamed);
      if (repeated || occurrence == 0)
        assert(WIFSIGNALED(status) && WTERMSIG(status) == crash_signal);
      else
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
      DiskStore reopened(ledger, dir.path, 100000);
      CheckPublished(reopened, dir.path);
      if (step < int(DiskPublicationStep::kManifestRenamed) &&
          (repeated || occurrence == 0))
        assert(reopened.Entries().empty());
      else
        assert(reopened.Entries().size() == 1);
      // Startup did not reclaim; worker preserves complete indexed checkpoints.
      auto before = Managed(dir.path);
      assert(reopened.Stats().managed_bytes == before);
      reopened.ReclaimOrphans();
      reopened.WaitForReclamation();
      CheckPublished(reopened, dir.path);
      assert(reopened.Stats().managed_bytes == Managed(dir.path));
      if (reopened.Entries().empty())
        assert(Managed(dir.path) == 0);
      else
        CheckExact(reopened, expected);
      assert(fs::is_empty(dir.path / "v2/tmp"));
    }
  }
}
void BatchedSyncs() {
  Directory dir;
  Fixture fixture(1, 65);  // 67 payloads: two full groups and a partial group.
  DiskStore store(ledger, dir.path, 100000);
  std::vector<DiskWriteSource> sources;
  for (const auto& buffer : fixture.buffers)
    sources.push_back({buffer.private_file, buffer.file, buffer.bytes.size(),
                       [buffer](auto offset, auto bytes) {
                         std::copy_n(buffer.bytes.begin() + offset,
                                     bytes.size(), bytes.begin());
                       }});
  std::array<std::uint8_t, 3> staging{};
  unsigned written{}, synced{}, renamed{};
  const auto attempts = sync_attempts.load();
  const auto completed = sync_completions.load();
  const auto before = store.PublicationStats();
  store.SetCrashHook([&](auto step) {
    if (step == DiskPublicationStep::kPayloadWritten) {
      ++written;
      assert(written - renamed <= 32 && synced == renamed);
    } else if (step == DiskPublicationStep::kPayloadSynced) {
      ++synced;
      assert(written == std::min((renamed / 32 + 1) * 32, 67U));
      // All calls have returned before the first synced hook or rename.
      assert(sync_attempts - attempts == written);
      assert(sync_completions - completed == written);
    } else if (step == DiskPublicationStep::kPayloadRenamed) {
      ++renamed;
      assert(synced == written);
    } else if (step == DiskPublicationStep::kTemporaryDirectorySynced) {
      assert(written == 67 && synced == written && renamed == written);
    }
  });
  sync_arrivals = 0;
  gate_sync = true;  // Require four simultaneous calls, not just async syntax.
  const auto stats =
      store.PublishStream(Id(100), fixture.manifest, sources, staging);
  gate_sync = false;
  store.SetCrashHook({});
  assert(stats.fsync_calls == 67 + 6);
  assert(sync_attempts - attempts == stats.fsync_calls);
  assert(sync_completions - completed == stats.fsync_calls);
  assert(store.PublicationStats().fsync_calls - before.fsync_calls ==
         stats.fsync_calls);
  assert(stats.shared_write_bytes == 65 * 8 && stats.private_write_bytes == 20);
  assert(fs::is_empty(dir.path / "v2/tmp"));
  CheckPublished(store, dir.path);
  CheckExact(store, fixture);
  assert(store.Retire({1}));
  // Corrupt the 65th chunk after two complete groups. Its failed streaming
  // CRC must prevent syncing/renaming the partial group or committing it.
  sources.back().read = [](auto, auto bytes) {
    std::fill(bytes.begin(), bytes.end(), 0);
  };
  const auto before_crc = store.PublicationStats();
  Reject([&] {
    (void)store.PublishStream(Id(100), fixture.manifest, sources, staging);
  });
  assert(store.PublicationStats().fsync_calls - before_crc.fsync_calls == 64);
  assert(store.Entries().empty() && !store.Open({1}));
  store.ReclaimOrphans();
  store.WaitForReclamation();
  assert(Managed(dir.path) == 0 && fs::is_empty(dir.path / "v2/tmp"));
}
void BatchedSyncFailures() {
  for (int failure : {0, 31, 32, 63, 64, 66}) {
    Directory dir;
    Fixture fixture(1, 65);
    DiskStore store(ledger, dir.path, 100000);
    unsigned renamed{};
    store.SetCrashHook([&](auto step) {
      if (step == DiskPublicationStep::kPayloadRenamed)
        ++renamed;
    });
    const auto before = store.PublicationStats();
    const auto attempts = sync_attempts.load();
    const auto completed = sync_completions.load();
    sync_arrivals = 0;
    gate_sync = true;
    fail_sync = failure;
    Reject([&] { store.Publish(Id(100), fixture.manifest, fixture.buffers); });
    gate_sync = false;
    fail_sync = -1;
    store.SetCrashHook({});
    // Even after one failure, every scheduled file sync settles. No payload
    // in the failed group is renamed; earlier groups are only orphans.
    const auto calls = std::min((failure / 32 + 1) * 32, 67);
    assert(sync_attempts - attempts == unsigned(calls));
    assert(sync_completions - completed == unsigned(calls));
    assert(store.PublicationStats().fsync_calls - before.fsync_calls ==
           unsigned(calls));
    assert(renamed == unsigned(failure / 32 * 32));
    assert(store.Entries().empty() && !store.Open({1}));
    assert(fs::is_empty(dir.path / "v2/manifests"));
    assert(store.Stats().managed_bytes == Managed(dir.path));
    store.ReclaimOrphans();
    store.WaitForReclamation();
    assert(Managed(dir.path) == 0 && store.Stats().managed_bytes == 0);
    assert(fs::is_empty(dir.path / "v2/tmp"));
    store.Publish(Id(100), fixture.manifest, fixture.buffers);
    CheckExact(store, fixture);
  }
}
void PayloadNamespaceOverlap() {
  Directory dir;
  Fixture fixture;
  fixture.manifest.components[0].tail->file = Id(1);
  fixture.buffers[1].file = Id(1);
  DiskStore store(ledger, dir.path, 100000);
  store.Publish(Id(100), fixture.manifest, fixture.buffers);
  assert(fs::is_empty(dir.path / "v2/tmp"));
  CheckExact(store, fixture);
}
void SharingAndPins() {
  Directory dir;
  DiskStore store(ledger, dir.path, 100000);
  Fixture a, b(2);
  store.Publish(Id(100), a.manifest, a.buffers);
  auto pin = store.Pin({1});
  assert(pin);
  std::thread eviction([&] { assert(!store.Retire({1})); });
  eviction.join();
  b.buffers.erase(b.buffers.begin());
  store.Publish(Id(101), b.manifest, b.buffers);
  pin = {};
  assert(store.Retire({1}));
  assert(fs::exists(dir.path / "v2/chunks" / DiskFileName(Id(1))));
  assert(!fs::exists(dir.path / "v2/private" / DiskFileName(Id(11))));
  CheckPublished(store, dir.path);
  assert(store.Retire({2}));
  assert(Managed(dir.path) == 0);
}
void IoOrderingAndShortWrites() {
  Directory dir;
  Fixture f;
  DiskStore store(ledger, dir.path, 100000);
  short_writes = true;
  interrupt_sync = true;
  events.clear();
  record_io = true;
  store.Publish(Id(100), f.manifest, f.buffers);
  record_io = false;
  short_writes = false;
  // Independently observe the actual fsync descriptors. Process abort tests
  // alone cannot model loss of the kernel page cache on a power failure.
  std::vector<std::string> expected;
  for (auto id : {Id(1), Id(11), Id(21)})
    expected.push_back("sync:" +
                       (dir.path / "v2/tmp" / DiskFileName(id)).string());
  for (auto name : {"tmp", "chunks", "private"})
    expected.push_back("sync:" + (dir.path / "v2" / name).string());
  expected.push_back("sync:" +
                     (dir.path / "v2/tmp" / DiskFileName(Id(100))).string());
  expected.push_back("sync:" + (dir.path / "v2/manifests").string());
  expected.push_back("sync:" + (dir.path / "v2/tmp").string());
  // Payload syncs may complete in any order, but all precede the existing
  // directory and manifest barriers, whose order is unchanged.
  assert(events.size() == expected.size());
  std::sort(events.begin(), events.begin() + 3);
  std::sort(expected.begin(), expected.begin() + 3);
  assert(events == expected);
  CheckPublished(store, dir.path);
  events.clear();
  record_io = true;
  assert(store.Retire({1}));
  record_io = false;
  assert(events[0] ==
         "unlink:" +
             (dir.path / "v2/manifests" / DiskFileName(Id(100))).string());
  assert(events[1] == "sync:" + (dir.path / "v2/manifests").string());
  assert(events[2].starts_with("unlink:"));
  // Partial writes are counted, retained as orphans, and reclaimed later.
  short_writes = true;
  fail_write = 1;
  Reject([&] { store.Publish(Id(100), f.manifest, f.buffers); });
  short_writes = false;
  fail_write = -1;
  assert(store.Stats().managed_bytes == 2);
  assert(Managed(dir.path) == 2);
  store.ReclaimOrphans();
  store.WaitForReclamation();
  assert(store.Stats().managed_bytes == 0);
}
void RejectedStartupBarrier() {
  Directory dir;
  Fixture f;
  {
    DiskStore layout(ledger, dir.path, 100000);
  }
  // Seed a checksum-valid manifest with a missing chunk. Its private files
  // exist but cannot be adopted by the new publication.
  auto rejected = f.manifest;
  rejected.components[0].tail->file = Id(31);
  rejected.components[1].private_state->file = Id(41);
  auto encoded = EncodeManifest(rejected);
  {
    std::ofstream manifest(dir.path / "v2/manifests" / DiskFileName(Id(100)),
                           std::ios::binary);
    manifest.write(reinterpret_cast<const char*>(encoded.data()),
                   encoded.size());
    std::ofstream tail(dir.path / "v2/private" / DiskFileName(Id(31)),
                       std::ios::binary);
    tail.write(reinterpret_cast<const char*>(f.tail.data()), f.tail.size());
    std::ofstream state(dir.path / "v2/private" / DiskFileName(Id(41)),
                        std::ios::binary);
    state.write(reinterpret_cast<const char*>(f.state.data()), f.state.size());
  }
  {
    DiskStore store(ledger, dir.path, 100000);
    assert(store.Entries().empty() && store.Stats().rejected_manifests == 1);
    Reject([&] { store.Publish(Id(101), f.manifest, f.buffers); });
    store.ReclaimOrphans();
    store.WaitForReclamation();
    assert(Managed(dir.path) == 0);
    store.Publish(Id(101), f.manifest, f.buffers);
  }
  DiskStore reopened(ledger, dir.path, 100000);
  assert(reopened.Entries().size() == 1 &&
         reopened.Stats().rejected_manifests == 0);
  CheckPublished(reopened, dir.path);
}
void UncertainManifestBarrier() {
  Directory dir;
  Fixture a, b(2);
  {
    DiskStore store(ledger, dir.path, 100000);
    store.Publish(Id(100), a.manifest, a.buffers);
    b.buffers.erase(b.buffers.begin());
    // Two new private payloads: fail the final temporary-directory sync, after
    // B's manifest rename has already become durable.
    fail_sync = 7;
    Reject([&] { store.Publish(Id(101), b.manifest, b.buffers); });
    fail_sync = -1;
    assert(fs::exists(dir.path / "v2/manifests" / DiskFileName(Id(101))));
    assert(store.Entries().size() == 1);
    Reject([&] { (void)store.Retire({1}); });
    assert(fs::exists(dir.path / "v2/chunks" / DiskFileName(Id(1))));
    Fixture retry(2);
    retry.manifest.components[0].tail->file = Id(32);
    retry.manifest.components[1].private_state->file = Id(42);
    retry.buffers[1].file = Id(32);
    retry.buffers[2].file = Id(42);
    Reject([&] { store.Publish(Id(102), retry.manifest, retry.buffers); });
    // A failed recovery barrier must keep publication and retirement blocked.
    fail_sync = 0;
    store.ReclaimOrphans();
    Reject([&] { store.WaitForReclamation(); });
    fail_sync = -1;
    Reject([&] { (void)store.Retire({1}); });
    Reject([&] { store.Publish(Id(102), retry.manifest, retry.buffers); });
    store.ReclaimOrphans();
    store.WaitForReclamation();
    store.Publish(Id(102), retry.manifest, retry.buffers);
    assert(store.Retire({1}));
    assert(store.Entries().size() == 1);
    CheckPublished(store, dir.path);
  }
  DiskStore reopened(ledger, dir.path, 100000);
  assert(reopened.Entries().size() == 1);
  assert(reopened.Entries()[0].Manifest().checkpoint.value == 2);
  CheckPublished(reopened, dir.path);
}
void RetirementFailure() {
  for (int sync : {0, 1, 2}) {
    Directory dir;
    Fixture f;
    DiskStore store(ledger, dir.path, 100000);
    store.Publish(Id(100), f.manifest, f.buffers);
    fail_sync = sync;
    Reject([&] { (void)store.Retire({1}); });
    fail_sync = -1;
    assert(store.Entries().empty());
    assert(Managed(dir.path) == (sync == 0 ? 28 : 0));
    assert(store.Stats().managed_bytes == Managed(dir.path));
    Fixture retry;
    retry.manifest.components[0].chunks[0].file = Id(51);
    retry.manifest.components[0].tail->file = Id(52);
    retry.manifest.components[1].private_state->file = Id(53);
    retry.buffers[0].file = Id(51);
    retry.buffers[1].file = Id(52);
    retry.buffers[2].file = Id(53);
    Reject([&] { store.Publish(Id(101), retry.manifest, retry.buffers); });
    fail_sync = 0;
    store.ReclaimOrphans();
    Reject([&] { store.WaitForReclamation(); });
    fail_sync = -1;
    Reject([&] { store.Publish(Id(101), retry.manifest, retry.buffers); });
    store.ReclaimOrphans();
    store.WaitForReclamation();
    assert(Managed(dir.path) == 0);
    store.Publish(Id(101), retry.manifest, retry.buffers);
    CheckPublished(store, dir.path);
  }
}
void RetirementUnlinkFailure() {
  for (bool applied : {true, false}) {
    Directory dir;
    Fixture a, b(2);
    DiskStore store(ledger, dir.path, 100000);
    store.Publish(Id(100), a.manifest, a.buffers);
    if (applied)
      unlink_effect_error = true;
    else
      fail_unlink = true;
    Reject([&] { (void)store.Retire({1}); });
    assert(store.Entries().empty());
    assert(!store.Pin({1}));
    Reject([&] { store.Publish(Id(101), b.manifest, b.buffers); });
    store.ReclaimOrphans();
    store.WaitForReclamation();
    assert(Managed(dir.path) == 0 && store.Stats().managed_bytes == 0);
    store.Publish(Id(101), b.manifest, b.buffers);
    CheckPublished(store, dir.path);
  }
}
void ReclamationSyncFailures() {
  for (int sync : {0, 1, 2, 3}) {
    Directory dir;
    DiskStore store(ledger, dir.path, 100000);
    Fixture a, fresh(2);
    short_writes = true;
    fail_write = 1;
    Reject([&] { store.Publish(Id(100), a.manifest, a.buffers); });
    short_writes = false;
    fail_write = -1;
    fresh.manifest.components[0].chunks[0].file = Id(51);
    fresh.buffers[0].file = Id(51);
    fail_sync = sync;
    store.ReclaimOrphans();
    Reject([&] { store.WaitForReclamation(); });
    fail_sync = -1;
    // Fresh IDs avoid incidental name collisions with the partial write.
    Reject([&] { store.Publish(Id(101), fresh.manifest, fresh.buffers); });
    store.ReclaimOrphans();
    store.WaitForReclamation();
    assert(Managed(dir.path) == 0 && store.Stats().managed_bytes == 0);
    store.Publish(Id(101), fresh.manifest, fresh.buffers);
    CheckPublished(store, dir.path);
  }
}
void RestartDeletionCredits() {
  for (int retirement_sync : {0, 1, 2}) {
    Directory dir;
    Fixture f;
    const auto budget = EncodeManifest(f.manifest).size() + 28;
    {
      DiskStore store(ledger, dir.path, budget);
      store.Publish(Id(100), f.manifest, f.buffers);
      fail_sync = retirement_sync;
      Reject([&] { (void)store.Retire({1}); });
      fail_sync = -1;
    }
    events.clear();
    record_io = true;
    {
      DiskStore reopened(ledger, dir.path, budget);
      assert(reopened.Entries().empty());
      assert(reopened.Stats().managed_bytes == Managed(dir.path));
    }
    record_io = false;
    const auto count = events.size();
    assert(count >= 4);
    std::size_t i = count - 4;
    for (auto directory : {"manifests", "chunks", "private", "tmp"})
      assert(events[i++] == "sync:" + (dir.path / "v2" / directory).string());
    // Failure to persist any namespace must prevent a usable store, including
    // after the previous process reported deletion and released its budget.
    for (std::size_t step = count - 4; step < count; ++step) {
      fail_sync = step;
      Reject([&] { DiskStore reopened(ledger, dir.path, budget); });
      fail_sync = -1;
    }
    DiskStore reopened(ledger, dir.path, budget);
    reopened.ReclaimOrphans();
    reopened.WaitForReclamation();
    reopened.Publish(Id(101), f.manifest, f.buffers);
    CheckPublished(reopened, dir.path);
  }
}
void RecoveryAndUnknowns() {
  Directory dir;
  Fixture f;
  {
    DiskStore store(ledger, dir.path, 100000);
    store.Publish(Id(100), f.manifest, f.buffers);
  }
  auto chunks = dir.path / "v2/chunks";
  std::ofstream(chunks / "notes") << "unknown";
  fs::create_directory(chunks / DiskFileName(Id(200)));
  fs::create_symlink(chunks / "notes", chunks / DiskFileName(Id(201)));
  std::ofstream(chunks / DiskFileName(Id(202))) << "hard link";
  fs::create_hard_link(chunks / DiskFileName(Id(202)), chunks / "hard-link");
  std::ofstream(dir.path / "v2/manifests" / DiskFileName(Id(203))) << "invalid";
  std::ofstream(dir.path / "v2/tmp" / DiskFileName(Id(204))) << "temporary";
  DiskStore store(ledger, dir.path, 100000);
  auto pin = store.Pin({1});
  assert(pin);
  store.ReclaimOrphans();
  store.WaitForReclamation();
  CheckPublished(store, dir.path);
  assert(fs::exists(chunks / "notes") &&
         fs::is_directory(chunks / DiskFileName(Id(200))));
  assert(fs::is_symlink(chunks / DiskFileName(Id(201))));
  assert(fs::exists(chunks / DiskFileName(Id(202))));
  assert(!fs::exists(dir.path / "v2/manifests" / DiskFileName(Id(203))));
  assert(fs::is_empty(dir.path / "v2/tmp"));
}
void LedgerFailures() {
  for (auto step : {LedgerStep::kReserve, LedgerStep::kConvert}) {
    bool completed{};
    for (std::size_t attempt = 0; attempt < 20; ++attempt) {
      Directory dir;
      ResourceLedger local{{1ULL << 32, 1ULL << 32, 0, 0}};
      {
        DiskStore store(local, dir.path, 100000);
        Fixture f;
        local.FailAfter(step, attempt);
        try {
          store.Publish(Id(100), f.manifest, f.buffers);
          completed = true;
        } catch (const std::bad_alloc&) {
          assert(store.Entries().empty());
          assert(Managed(dir.path) == 0);
        }
        local.ClearFaults();
      }
      assert(local.Snapshot().ram_bytes == 0);
      if (completed)
        break;
    }
    assert(completed);
  }
}
void WriterEviction() {
  Directory dir;
  DiskStore store(ledger, dir.path, 100000);
  Fixture a, b(2);
  store.Publish(Id(100), a.manifest, a.buffers);
  b.buffers.erase(b.buffers.begin());
  std::promise<void> reached, release, attempted;
  auto released = release.get_future();
  bool held{};
  store.SetCrashHook([&](auto step) {
    if (step == DiskPublicationStep::kPayloadWritten && !held) {
      held = true;
      reached.set_value();
      released.wait();
    }
  });
  std::thread writer([&] { store.Publish(Id(101), b.manifest, b.buffers); });
  reached.get_future().wait();
  std::thread eviction([&] {
    attempted.set_value();
    assert(store.Retire({1}));
  });
  attempted.get_future().wait();
  release.set_value();
  writer.join();
  eviction.join();
  assert(store.Entries().size() == 1 &&
         store.Entries()[0].Manifest().checkpoint.value == 2);
  CheckPublished(store, dir.path);
}
void Failures() {
  Fixture f;
  auto total = EncodeManifest(f.manifest).size() + 28;
  {
    Directory dir;
    DiskStore store(ledger, dir.path, total - 1);
    Reject([&] { store.Publish(Id(100), f.manifest, f.buffers); });
    assert(Managed(dir.path) == 0);
  }
  for (int sync = 0; sync < 9; ++sync) {
    Directory dir;
    {
      DiskStore store(ledger, dir.path, total);
      fail_sync = sync;
      Reject([&] { store.Publish(Id(100), f.manifest, f.buffers); });
      fail_sync = -1;
      assert(store.Entries().empty());
      assert(store.Stats().managed_bytes == Managed(dir.path));
    }
    DiskStore store(ledger, dir.path, total);
    CheckPublished(store, dir.path);
    store.ReclaimOrphans();
    store.WaitForReclamation();
    assert(store.Stats().managed_bytes == Managed(dir.path));
  }
  Directory dir;
  DiskStore store(ledger, dir.path, total * 4);
  auto corrupt = f.buffers;
  std::vector<std::uint8_t> bad(8);
  corrupt[0].bytes = bad;
  Reject([&] { store.Publish(Id(100), f.manifest, corrupt); });
  assert(Managed(dir.path) == 0);
  store.Publish(Id(100), f.manifest, f.buffers);
  Reject([&] { store.Publish(Id(100), f.manifest, f.buffers); });
  Fixture conflict(2);
  conflict.manifest.lineage = {2};
  Reject([&] { store.Publish(Id(101), conflict.manifest, conflict.buffers); });
  assert(store.Entries().size() == 1);
  CheckPublished(store, dir.path);
}
}  // namespace
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
  ++sync_attempts;
  struct Complete {
    ~Complete() { ++sync_completions; }
  } complete;
  if (gate_sync && sync_arrivals.fetch_add(1) < 4) {
    std::unique_lock lock(sync_mutex);
    sync_ready.notify_all();
    assert(sync_ready.wait_for(lock, std::chrono::seconds(5),
                               [] { return sync_arrivals >= 4; }));
  }
  if (kill_sync >= 0 && kill_sync.fetch_sub(1) == 0)
    raise(SIGKILL);
  auto count = fail_sync.load();
  if (count >= 0 && fail_sync.fetch_sub(1) == 0) {
    errno = EIO;
    return -1;
  }
  if (interrupt_sync.exchange(false)) {
    errno = EINTR;
    return -1;
  }
  if (record_io) {
    std::lock_guard lock(events_mutex);
    events.push_back("sync:" + FdPath(fd));
  }
  return __real_fsync(fd);
}
extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" ssize_t __wrap_write(int fd, const void* buffer, size_t bytes) {
  if (fail_write >= 0 && fail_write.fetch_sub(1) == 0) {
    errno = ENOSPC;
    return -1;
  }
  if (short_writes)
    bytes = std::min(bytes, std::size_t{2});
  return __real_write(fd, buffer, bytes);
}
extern "C" int __real_unlinkat(int, const char*, int);
extern "C" int __wrap_unlinkat(int fd, const char* name, int flags) {
  if (fail_unlink.exchange(false)) {
    errno = EIO;
    return -1;
  }
  if (record_io) {
    std::lock_guard lock(events_mutex);
    events.push_back("unlink:" + FdPath(fd) + "/" + name);
  }
  auto result = __real_unlinkat(fd, name, flags);
  if (unlink_effect_error.exchange(false) && result == 0) {
    errno = EIO;
    return -1;
  }
  return result;
}
int main() {
  const rlimit no_core{0, 0};
  assert(setrlimit(RLIMIT_CORE, &no_core) == 0);
  WriteAccountingIncludesRollbackAndReuse();
  Crashes();
  Crashes(65, SIGKILL);
  BatchedSyncs();
  BatchedSyncFailures();
  PayloadNamespaceOverlap();
  SharingAndPins();
  WriterEviction();
  Failures();
  RetirementFailure();
  RetirementUnlinkFailure();
  ReclamationSyncFailures();
  RestartDeletionCredits();
  UncertainManifestBarrier();
  RejectedStartupBarrier();
  RecoveryAndUnknowns();
  LedgerFailures();
  IoOrderingAndShortWrites();
}
