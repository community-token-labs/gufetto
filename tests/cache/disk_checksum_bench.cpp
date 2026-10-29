// CPU-only, one-pass checksum overhead; no global page-cache manipulation.
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "src/cache/disk.hpp"

namespace {
using Clock = std::chrono::steady_clock;
void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("disk checksum benchmark failed");
}
struct Scratch {
  std::filesystem::path directory;
  ~Scratch() { std::filesystem::remove_all(directory); }
};
}  // namespace
int main(int argc, char** argv) {
  if (argc != 2)
    throw std::invalid_argument(
        "usage: cache_disk_checksum_bench EXT4_DIRECTORY");
  auto pattern =
      (std::filesystem::path(argv[1]) / "gufo-crc-bench-XXXXXX").string();
  Require(mkdtemp(pattern.data()) != nullptr);
  Scratch scratch{pattern};
  const auto path = scratch.directory / "payload";
  constexpr std::size_t bytes = 2ULL << 30, piece = 1ULL << 20;
  std::vector<std::uint8_t> buffer(piece);
  std::uint64_t rng = 0x123456789abcdefULL;
  for (auto& value : buffer) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    value = rng;
  }
  int fd = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
  Require(fd >= 0);
  for (std::size_t n = 0; n < bytes / piece; ++n)
    Require(write(fd, buffer.data(), piece) == static_cast<ssize_t>(piece));
  Require(fsync(fd) == 0 && close(fd) == 0);
  std::uint64_t expected{};
  for (std::size_t n = 0; n < bytes / piece; ++n)
    expected = gufo::cache::DiskChecksum(buffer, expected);
  std::puts("cache_state,round,crc,bytes,wall_ms,checksum_ms,GiB_s");
  // Both arms use the same executable, buffer, read loop and full-byte count.
  // Warm is explicitly primed; advised is an own-file hint, not physical-I/O
  // proof.
  for (bool advised : {false, true}) {
    for (int round = 0; round < 3; ++round) {
      for (bool verify : {bool(round % 2), !bool(round % 2)}) {
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        Require(fd >= 0);
        if (advised) {
          Require(posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0);
        } else {
          for (std::size_t n = 0; n < bytes / piece; ++n)
            Require(read(fd, buffer.data(), piece) ==
                    static_cast<ssize_t>(piece));
          Require(lseek(fd, 0, SEEK_SET) == 0);
        }
        std::uint64_t crc{};
        double checksum_ms{};
        const auto start = Clock::now();
        for (std::size_t n = 0; n < bytes / piece; ++n) {
          Require(read(fd, buffer.data(), piece) ==
                  static_cast<ssize_t>(piece));
          if (verify) {
            const auto check_start = Clock::now();
            crc = gufo::cache::DiskChecksum(buffer, crc);
            checksum_ms += std::chrono::duration<double, std::milli>(
                               Clock::now() - check_start)
                               .count();
          }
        }
        const auto elapsed =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
        Require(close(fd) == 0 && (!verify || crc == expected));
        std::printf("%s,%d,%d,%zu,%.3f,%.3f,%.3f\n",
                    advised ? "advised" : "warm", round + 1, verify, bytes,
                    elapsed, checksum_ms, 2000 / elapsed);
      }
    }
  }
  std::filesystem::remove(path);
  // Actual inherited-publication path: one 2-GiB immutable row chunk, with
  // distinct tiny private state for each dependent checkpoint. No full host
  // copy.
  gufo::cache::ResourceLedger ledger{{64ULL << 20, 64ULL << 20, 0, 0}};
  gufo::cache::DiskStore store(ledger, scratch.directory / "store", 3ULL << 30);
  const auto id = [](unsigned value) {
    gufo::cache::DiskFileId result{};
    result[0] = value;
    return result;
  };
  gufo::cache::DiskManifest manifest;
  manifest.lineage = {1};
  manifest.tokens.assign(2048, 42);
  manifest.components.push_back(
      {{{1}, 1, gufo::cache::ComponentKind::kAppendRows, piece, 2048, 0},
       {{1}, 2048},
       {{id(1), bytes, expected}},
       {},
       {}});
  manifest.components.push_back(
      {{{2}, 1, gufo::cache::ComponentKind::kPrivateState, 0, 0, 16},
       {{2}, 2048},
       {},
       {},
       {}});
  std::vector<std::uint8_t> staging(piece);
  std::puts(
      "publication,shared_bytes,wall_ms,filesystem_ms,checksum_ms,fsync_ms");
  for (unsigned n = 1; n <= 4; ++n) {
    manifest.checkpoint = {n};
    manifest.components[1].private_state =
        gufo::cache::DiskPayload{id(n + 10), 16, 0};
    std::vector<gufo::cache::DiskWriteSource> sources;
    if (n == 1)
      sources.push_back({false, id(1), bytes, [&](auto, auto target) {
                           std::copy_n(buffer.begin(), target.size(),
                                       target.begin());
                         }});
    sources.push_back({true, id(n + 10), 16, [n](auto, auto target) {
                         std::fill(target.begin(), target.end(), n);
                       }});
    const auto before = store.PublicationStats();
    const auto start = Clock::now();
    (void)store.PublishStream(id(n + 20), manifest, sources, staging);
    const auto elapsed =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const auto after = store.PublicationStats();
    std::printf("%u,%zu,%.3f,%.3f,%.3f,%.3f\n", n, n == 1 ? 0 : bytes, elapsed,
                (after.filesystem_ns - before.filesystem_ns) / 1e6,
                (after.checksum_ns - before.checksum_ns) / 1e6,
                (after.fsync_ns - before.fsync_ns) / 1e6);
  }
}
