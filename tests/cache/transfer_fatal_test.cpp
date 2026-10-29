#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "src/cache/slot.hpp"
#include "src/cache/streaming.hpp"
#include "src/core/hip/transfer_pool.hpp"
#include "src/core/platform/shutdown_signal.hpp"
#include "tests/cache/fake_adapter.hpp"

using namespace gufo;
using namespace gufo::cache;
using namespace gufo::cache::testing;
namespace fs = std::filesystem;
struct HipTestStream {};
struct HipTestEvent {};
namespace {
const char* fail_api{};
unsigned skip_calls{}, copies{}, destroyed{};
bool pending{}, not_ready{};
int evidence = -1;
enum class Hook { kReturn, kThrow, kMissing, kBlocked };
Hook hook{};
std::function<void()> inspect_fault;
void Record(const char* text) {
  const auto size = std::strlen(text);
  assert(::write(evidence, text, size) == static_cast<ssize_t>(size));
}
struct Owner {
  ~Owner() { Record("OWNER_RELEASED\n"); }
};
hipError_t Hit(const char* api) {
  if (!fail_api || std::strcmp(fail_api, api) != 0)
    return hipSuccess;
  if (skip_calls) {
    --skip_calls;
    return hipSuccess;
  }
  fail_api = nullptr;  // Every subsequent HIP call would report success.
  if (pending) {
    if (inspect_fault)
      inspect_fault();
    Record("PENDING\n");
    for (;;)
      ::pause();
  }
  Record("FIRST_ERROR\n");
  return hipErrorUnknown;
}
void Fatal(const hip::FatalCopyInfo& info) {
  assert(info.error == hipErrorUnknown && info.operation);
  char line[192];
  std::snprintf(line, sizeof(line), "FATAL %s entry=%llu bytes=%llu\n",
                info.operation, static_cast<unsigned long long>(info.entry_id),
                static_cast<unsigned long long>(info.bytes));
  Record(line);
  if (inspect_fault)
    inspect_fault();
  if (hook == Hook::kThrow)
    throw std::runtime_error("notification failed");
  if (hook == Hook::kBlocked) {
    // Test handler's watchdog, NOT the real serving watchdog implementation.
    std::thread([] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      platform::ExitAfterDeviceLoss();
    }).detach();
    for (;;)
      ::pause();
  }
}
std::string Child(const std::function<void()>& body, int expected = 75,
                  bool kill_pending = false) {
  int pipe[2];
  assert(::pipe(pipe) == 0);
  const auto pid = ::fork();
  assert(pid >= 0);
  if (pid == 0) {
    ::close(pipe[0]);
    evidence = pipe[1];
    ::alarm(8);  // A deadlock fails this test, never hangs the suite.
    body();
    ::_exit(0);
  }
  ::close(pipe[1]);
  std::string output;
  char buffer[256];
  for (;;) {
    const auto n = ::read(pipe[0], buffer, sizeof(buffer));
    if (n == 0)
      break;
    assert(n > 0);
    output.append(buffer, n);
    if (kill_pending && output.find("PENDING\n") != std::string::npos) {
      assert(::kill(pid, SIGKILL) == 0);
      kill_pending = false;
    }
  }
  ::close(pipe[0]);
  int status{};
  assert(::waitpid(pid, &status, 0) == pid);
  if (!(expected < 0 ? WIFSIGNALED(status) && WTERMSIG(status) == -expected
                     : WIFEXITED(status) && WEXITSTATUS(status) == expected))
    std::fprintf(stderr, "child status=%d expected=%d evidence=%s\n", status,
                 expected, output.c_str());
  if (expected < 0)
    assert(WIFSIGNALED(status) && WTERMSIG(status) == -expected);
  else
    assert(WIFEXITED(status) && WEXITSTATUS(status) == expected);
  assert(output.find("OWNER_RELEASED") == std::string::npos);
  return output;
}
void FaultsNeverReturn() {
  for (const char* api : {"hipMemcpyAsync", "hipEventRecord", "hipEventQuery",
                          "hipEventSynchronize", "hipStreamSynchronize"}) {
    for (unsigned skip : {0U, 1U}) {
      if (skip && std::strcmp(api, "hipMemcpyAsync") != 0)
        continue;
      for (const auto mode :
           {Hook::kReturn, Hook::kThrow, Hook::kMissing, Hook::kBlocked}) {
        const auto output = Child([&] {
          Owner source;
          hook = mode;
          hip::SetFatalCopyHandler(mode == Hook::kMissing ? nullptr : Fatal);
          hip::TransferPool pool(1);
          auto stream = pool.TryAcquire();
          // Exercise bound ownership as well as an owned pool stream.
          auto owner = std::make_shared<Owner>();
          HipTestStream external;
          if (skip || std::strcmp(api, "hipEventRecord") == 0) {
            stream->BindTo(&external, owner);
            owner.reset();
          }
          stream->SetPersistenceContext(31, 8);
          std::array<std::byte, 8> from{}, to{};
          fail_api = api;
          skip_calls = skip;
          inspect_fault = [&] { assert(destroyed == 0); };
          auto done = stream->Copy(to.data(), from.data(), from.size(),
                                   hip::CopyDirection::kDeviceToHost, 4);
          (void)done.Ready();
          (void)done.Wait();
          Record("RETURNED\n");
        });
        assert(output.find("FIRST_ERROR") != std::string::npos);
        assert(output.find("RETURNED") == std::string::npos);
        if (mode != Hook::kMissing) {
          assert(output.find(std::string("FATAL ") + api) != std::string::npos);
          assert(output.find("entry=31 bytes=8") != std::string::npos);
        }
      }
    }
  }
}
void SharedFaultAndDestructor() {
  Child([] {
    Owner owner;
    hip::SetFatalCopyHandler(Fatal);
    hip::TransferPool pool(2);
    auto first = pool.TryAcquire(), peer = pool.TryAcquire();
    std::atomic<bool> entered_peer{}, returned{};
    inspect_fault = [&] {
      std::thread([&] {
        entered_peer = true;
        (void)peer->Synchronize();
        returned = true;
      }).detach();
      while (!entered_peer)
        std::this_thread::yield();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      assert(!returned && destroyed == 0);
    };
    fail_api = "hipEventRecord";
    (void)first->Complete();
  });
  Child([] {
    Owner owner;
    hip::SetFatalCopyHandler(Fatal);
    hip::TransferPool pool(1);
    auto stream = pool.TryAcquire();
    auto done = stream->Complete();
    assert(done.Wait() == TransferResult::kSucceeded);
    fail_api = "hipStreamSynchronize";
    stream.reset();
  });
}
void HealthyControls() {
  hip::TransferPool pool(1);
  std::array<std::byte, 9> source{}, target{};
  source.fill(std::byte{42});
  auto stream = pool.TryAcquire();
  bool rejected{};
  try {
    (void)stream->Copy(nullptr, source.data(), 1,
                       hip::CopyDirection::kDeviceToHost, 1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected && copies == 0);
  std::optional<Completion> done(
      stream->Copy(target.data(), source.data(), source.size(),
                   hip::CopyDirection::kDeviceToHost, 4));
  not_ready = true;
  assert(!done->Ready());
  assert(done->Ready());
  assert(done->Wait() == TransferResult::kSucceeded && source == target);
  assert(copies == 3);
  stream.reset();
  assert(!pool.TryAcquire());
  done.reset();
  stream = pool.TryAcquire();
  assert(stream);
  auto empty =
      stream->Copy(nullptr, nullptr, 0, hip::CopyDirection::kHostToDevice, 1);
  assert(empty.Wait() == TransferResult::kSucceeded);
  hip::TransferPool external_pool(1);
  auto bound = external_pool.TryAcquire();
  auto owner = std::make_shared<HipTestStream>();
  bound->BindTo(owner.get(), owner);
  assert(bound->Native() == owner.get());
  auto event = bound->Complete();
  assert(event.Wait() == TransferResult::kSucceeded);
}
DiskFileId Id(unsigned n) {
  DiskFileId id{};
  id[0] = n;
  return id;
}
struct Directory {
  fs::path path;
  Directory() {
    auto pattern =
        (fs::temp_directory_path() / "gufo-transfer-fatal-XXXXXX").string();
    const auto* result = ::mkdtemp(pattern.data());
    assert(result);
    path = result;
  }
  ~Directory() { fs::remove_all(path); }
};
DiskManifest Manifest(unsigned checkpoint,
                      std::span<const std::uint8_t> parent) {
  DiskManifest m;
  m.checkpoint = {checkpoint};
  m.lineage = {1};
  m.tokens = {1, 2};
  m.components.push_back({{{1}, 1, ComponentKind::kAppendRows, 4, 2, 0},
                          {{1}, 2},
                          {{Id(1), 8, DiskChecksum(parent)}},
                          {},
                          {}});
  if (checkpoint == 2) {
    m.tokens = {1, 2, 3, 4};
    m.components[0].position.valid_rows = 4;
    m.components[0].chunks.push_back({Id(2), 8, 0});
  }
  return m;
}
void StreamedBoundaryAndRestart(const char* api, unsigned skip = 0,
                                bool kill_pending = false) {
  Directory dir;
  const std::array<std::uint8_t, 8> original{1, 2, 3, 4, 5, 6, 7, 8};
  ResourceLedger ledger{{1 << 22, 1 << 21, 10, 4096}};
  {
    DiskStore disk(ledger, dir.path, 1 << 20);
    const std::array<DiskWriteBuffer, 1> buffers{{{false, Id(1), original}}};
    disk.Publish(Id(100), Manifest(1, original), buffers);
  }
  const auto output = Child(
      [&] {
        Owner frame;
        hip::SetFatalCopyHandler(Fatal);
        DiskStore disk(ledger, dir.path, 1 << 20);
        hip::TransferPool pool(1);
        std::weak_ptr<void> staging_owner;
        StreamedStore writer(
            ledger, disk, 4,
            [&](std::size_t bytes) {
              auto data = std::shared_ptr<std::uint8_t[]>(
                  new std::uint8_t[bytes], [](auto* p) {
                    Record("OWNER_RELEASED staging\n");
                    delete[] p;
                  });
              std::shared_ptr<void> owner(data, data.get());
              staging_owner = owner;
              return StagingAllocation{owner, {data.get(), bytes}};
            },
            [&]() -> std::unique_ptr<Stream> { return pool.TryAcquire(); });
        FakeAdapter adapter(2);
        FakeStream execution;
        LeasedSlot slot(ledger, adapter, execution, {1});
        auto lease = slot.Acquire();
        const std::array<Token, 4> tokens{1, 2, 3, 4};
        adapter.Append(lease.Execution(), tokens, tokens);
        auto rows = lease.Observe(
            kTarget, 2, 4,
            adapter.BorrowRowsView(lease.Execution(), kTarget, 2, 4));
        std::atomic<bool> mutating{}, mutated{};
        inspect_fault = [&] {
          assert(!staging_owner.expired() && rows->Bytes() == 8);
          assert(fs::file_size(dir.path / "v2/tmp" / DiskFileName(Id(2))) == 4);
          assert(
              !fs::exists(dir.path / "v2/manifests" / DiskFileName(Id(101))));
          // Mutation must still wait for the production PersistenceSource's
          // RowPin.
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          assert(mutating && !mutated);
        };
        unsigned pieces{};
        std::vector<PersistenceSource> sources;
        sources.emplace_back(
            false, Id(2), 8, 4, rows, 0,
            [&](auto source, auto destination, Stream& stream) {
              if (++pieces == 2) {
                assert(fs::file_size(dir.path / "v2/tmp" /
                                     DiskFileName(Id(2))) == 4);
                std::thread([&] {
                  mutating = true;
                  adapter.GuardRows(lease.Execution(), kTarget, 2, 4);
                  mutated = true;
                }).detach();
                while (!mutating)
                  std::this_thread::yield();
                fail_api = api;
                skip_calls = skip;
                pending = kill_pending;
              }
              auto done = dynamic_cast<hip::TransferStream&>(stream).Copy(
                  destination.data(), source.data(), destination.size(),
                  hip::CopyDirection::kDeviceToHost, 2);
              if (std::strcmp(api, "hipEventQuery") == 0)
                (void)done.Ready();
              return done;
            });
        TransferTiming timing;
        writer.Write(Id(101), Manifest(2, original), sources, timing);
        Record("RETURNED\n");
      },
      kill_pending ? -SIGKILL : 75, kill_pending);
  if (!kill_pending) {
    assert(output.find(std::string("FATAL ") + api) != std::string::npos);
    assert(output.find("entry=2 bytes=4") != std::string::npos);
  }
  assert(output.find("RETURNED") == std::string::npos);
  // A new reader process after writer death: no partial child visibility or
  // writer-process metadata can survive the restart.
  Child(
      [&] {
        DiskStore recovered(ledger, dir.path, 1 << 20);
        assert(!recovered.Open({2}));
        auto parent = recovered.Open({1});
        assert(parent && recovered.Entries().size() == 1);
        std::array<std::uint8_t, 4> staging{};
        std::vector<std::uint8_t> restored;
        const auto payload = parent->Manifest().components[0].chunks[0];
        (void)recovered.ReadPayload(
            *parent, false, payload, staging, [&](auto, auto bytes) {
              restored.insert(restored.end(), bytes.begin(), bytes.end());
            });
        assert(std::equal(original.begin(), original.end(), restored.begin(),
                          restored.end()));
        recovered.ReclaimOrphans();
        recovered.WaitForReclamation();
        assert(!fs::exists(dir.path / "v2/chunks" / DiskFileName(Id(2))));
        assert(fs::is_empty(dir.path / "v2/tmp"));
        // Inherited parent bytes are still content-checked, even with restored
        // mtime.
        const auto path = dir.path / "v2/chunks" / DiskFileName(Id(1));
        const auto time = fs::last_write_time(path);
        {
          std::fstream file(path,
                            std::ios::binary | std::ios::in | std::ios::out);
          file.put('x');
        }
        fs::last_write_time(path, time);
        bool rejected{};
        try {
          (void)recovered.ReadPayload(*parent, false, payload, staging,
                                      [](auto, auto) {});
        } catch (const std::exception&) {
          rejected = true;
        }
        assert(rejected);
      },
      0);
}
std::atomic<int> requested{};
std::atomic<bool> entered{}, release_handler{};
void Request(int signal) {
  requested.store(signal);
}
void PausedRequest(int signal) {
  entered = true;
  while (!release_handler)
    std::this_thread::yield();
  requested = signal;
}
void PlatformRequests() {
  Child([] {
    Owner owner;
    platform::RequestDeviceLossShutdown();
  });
  Child([] {
    Owner owner;
    {
      platform::ShutdownSignalGuard guard(Request);
    }
    platform::RequestDeviceLossShutdown();
  });
  Child([] {
    Owner owner;
    platform::ShutdownSignalGuard guard([](int) { throw 1; });
    platform::RequestDeviceLossShutdown();
  });
  Child(
      [] {
        platform::ShutdownSignalGuard original(Request),
            moved(std::move(original));
        platform::ShutdownSignalGuard assigned;
        assigned = std::move(moved);
        platform::RequestDeviceLossShutdown();
        assert(requested == SIGTERM);
      },
      0);
  Child(
      [] {
        auto guard =
            std::make_unique<platform::ShutdownSignalGuard>(PausedRequest);
        std::thread request(platform::RequestDeviceLossShutdown);
        while (!entered)
          std::this_thread::yield();
        guard.reset();
        release_handler = true;
        request.join();
        assert(requested == SIGTERM);
      },
      0);
}
}  // namespace
extern "C" {
const char* hipGetErrorString(hipError_t) {
  return "fake HIP failure";
}
hipError_t hipStreamCreateWithFlags(hipStream_t* p, unsigned) {
  *p = new HipTestStream;
  return hipSuccess;
}
hipError_t hipStreamDestroy(hipStream_t p) {
  ++destroyed;
  delete p;
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  return Hit("hipStreamSynchronize");
}
hipError_t hipEventCreateWithFlags(hipEvent_t* p, unsigned) {
  *p = new HipTestEvent;
  return hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t p) {
  ++destroyed;
  delete p;
  return hipSuccess;
}
hipError_t hipEventRecord(hipEvent_t, hipStream_t) {
  return Hit("hipEventRecord");
}
hipError_t hipEventQuery(hipEvent_t) {
  if (std::exchange(not_ready, false))
    return hipErrorNotReady;
  return Hit("hipEventQuery");
}
hipError_t hipEventSynchronize(hipEvent_t) {
  return Hit("hipEventSynchronize");
}
hipError_t hipMemcpyAsync(void* dst, const void* src, std::size_t n,
                          hipMemcpyKind, hipStream_t) {
  ++copies;
  std::memcpy(dst, src, n);
  return Hit("hipMemcpyAsync");
}
}
int main() {
  FaultsNeverReturn();
  SharedFaultAndDestructor();
  for (const char* api : {"hipMemcpyAsync", "hipEventRecord", "hipEventQuery",
                          "hipEventSynchronize", "hipStreamSynchronize"})
    StreamedBoundaryAndRestart(api);
  StreamedBoundaryAndRestart("hipMemcpyAsync", 1);
  StreamedBoundaryAndRestart("hipEventSynchronize", 0, true);
  PlatformRequests();
  HealthyControls();
}
