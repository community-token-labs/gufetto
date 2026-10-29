#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string_view>

#include "src/cache/disk.hpp"
using namespace gufo::cache;
DiskFileId Id(unsigned n) {
  DiskFileId id{};
  id[0] = n;
  id[1] = n >> 8;
  return id;
}
// Synthetic publication geometry, not a model or R61 replay: many 128-KiB
// files, bounded source storage, and incompressible deterministic bytes.
void Geometry(const std::filesystem::path& parent) {
  constexpr std::size_t tokens = 150000, components = 16, row_bytes = 512,
                        chunk_rows = 256, chunk_bytes = row_bytes * chunk_rows;
  auto pattern = (parent / "gufo-publication-geometry-XXXXXX").string();
  const auto* created = mkdtemp(pattern.data());
  assert(created);
  const auto root = std::filesystem::path(created);
  std::vector<std::uint8_t> data(chunk_bytes);
  std::uint32_t random = 0x12345678;
  for (auto& byte : data) {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    byte = static_cast<std::uint8_t>(random);
  }
  const auto full = std::span<const std::uint8_t>(data);
  const auto tail = full.first(row_bytes * (tokens % chunk_rows));
  const auto state = full.first(16);
  const auto full_crc = DiskChecksum(full), tail_crc = DiskChecksum(tail);
  DiskManifest manifest;
  manifest.checkpoint = {1};
  manifest.lineage = {1};
  manifest.tokens.assign(tokens, 17);
  std::vector<DiskWriteBuffer> buffers;
  unsigned file = 1;
  for (unsigned component = 1; component <= components; ++component) {
    DiskComponent c{
        {{component}, 1, ComponentKind::kAppendRows, row_bytes, chunk_rows, 0},
        {{component}, tokens},
        {},
        {},
        {}};
    for (std::size_t i = 0; i < tokens / chunk_rows; ++i) {
      const auto id = Id(file++);
      c.chunks.push_back({id, full.size(), full_crc});
      buffers.push_back({false, id, full});
    }
    const auto id = Id(file++);
    c.tail = DiskPayload{id, tail.size(), tail_crc};
    buffers.push_back({true, id, tail});
    manifest.components.push_back(std::move(c));
  }
  const auto private_id = Id(file++);
  manifest.components.push_back(
      {{{components + 1}, 1, ComponentKind::kPrivateState, 0, 0, state.size()},
       {{components + 1}, tokens},
       {},
       {},
       DiskPayload{private_id, state.size(), DiskChecksum(state)}});
  buffers.push_back({true, private_id, state});
  const auto encoded = EncodeManifest(manifest);
  const auto manifest_id = Id(file++);
  ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
  DiskPublicationStats stats;
  std::uint64_t managed{}, file_count{};
  double publication_ms{};
  const auto elapsed_ms = [](auto start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  {
    DiskStore store(ledger, root, 1ULL << 32);
    const auto start = std::chrono::steady_clock::now();
    store.Publish(manifest_id, manifest, buffers);
    publication_ms = elapsed_ms(start);
    stats = store.PublicationStats();
    managed = store.Stats().managed_bytes;
    assert(store.Entries().size() == 1);
  }
  const auto version = "v" + std::to_string(kDiskFormatVersion);
  for (const auto* folder : {"chunks", "private", "manifests"})
    for (const auto& entry :
         std::filesystem::directory_iterator(root / version / folder))
      file_count += entry.is_regular_file();
  double startup_ms{}, restore_ms{};
  std::uint64_t read_bytes{};
  {
    const auto startup = std::chrono::steady_clock::now();
    DiskStore reopened(ledger, root, 1ULL << 32);
    startup_ms = elapsed_ms(startup);
    auto snapshot = reopened.Open({1});
    assert(snapshot && EncodeManifest(snapshot->Manifest()) == encoded);
    std::vector<std::uint8_t> staging(chunk_bytes);
    const auto start = std::chrono::steady_clock::now();
    for (const auto& buffer : buffers) {
      const DiskPayload payload{buffer.file, buffer.bytes.size(),
                                buffer.bytes.size() == full.size() ? full_crc
                                : buffer.bytes.size() == tail.size()
                                    ? tail_crc
                                    : DiskChecksum(state)};
      const auto read = reopened.ReadPayload(
          *snapshot, buffer.private_file, payload, staging,
          [&](auto offset, auto bytes) {
            assert(offset + bytes.size() <= buffer.bytes.size());
            assert(std::equal(bytes.begin(), bytes.end(),
                              buffer.bytes.begin() + offset));
          });
      read_bytes += read.bytes;
      assert(read.bytes == buffer.bytes.size());
    }
    restore_ms = elapsed_ms(start);
  }
  const auto write_bytes = stats.private_write_bytes +
                           stats.shared_write_bytes +
                           stats.manifest_write_bytes;
  assert(read_bytes == components * tokens * row_bytes + state.size());
  assert(write_bytes == read_bytes + encoded.size() && write_bytes == managed);
  assert(file_count == buffers.size() + 1 && stats.inherited_bytes == 0);
  std::cout << "geometry=150k-16x512 tokens=" << tokens
            << " payload_files=" << buffers.size()
            << " file_count=" << file_count << " bytes=" << write_bytes
            << " publication_ms=" << publication_ms
            << " write_MBps=" << write_bytes / (publication_ms * 1000)
            << " fsync_calls=" << stats.fsync_calls
            << " fsync_ms=" << stats.fsync_ns / 1e6
            << " startup_ms=" << startup_ms << " restore_ms=" << restore_ms
            << " read_bytes=" << read_bytes << " exact=1\n";
  std::filesystem::remove_all(root);
}
int main(int argc, char** argv) {
  std::filesystem::path parent = argc > 1 ? argv[1] : "/tmp";
  if (argc == 3 && std::string_view(argv[2]) == "--geometry") {
    Geometry(parent);
    return 0;
  }
  assert(argc <= 2);
  for (std::size_t tokens : {128, 100000}) {
    auto pattern = (parent / "gufo-publication-bench-XXXXXX").string();
    auto root = std::filesystem::path(mkdtemp(pattern.data()));
    {
      ResourceLedger ledger{{1ULL << 32, 1ULL << 32, 0, 0}};
      DiskStore store(ledger, root, 1ULL << 32);
      DiskManifest m;
      m.checkpoint = {1};
      m.lineage = {1};
      m.tokens.assign(tokens, 17);
      DiskComponent c{{{1}, 1, ComponentKind::kAppendRows, 4, 2048, 0},
                      {{1}, tokens},
                      {},
                      {},
                      {}};
      std::vector<std::vector<std::uint8_t>> data(tokens / 2048 + 2);
      std::vector<DiskWriteBuffer> buffers;
      for (std::size_t i = 0; i < tokens / 2048; ++i) {
        data[i].assign(8192, 42);
        c.chunks.push_back({Id(i + 1), 8192, DiskChecksum(data[i])});
        buffers.push_back({false, Id(i + 1), data[i]});
      }
      auto i = tokens / 2048;
      data[i].assign(4 * (tokens % 2048), 42);
      c.tail = DiskPayload{Id(1000), data[i].size(), DiskChecksum(data[i])};
      buffers.push_back({true, Id(1000), data[i]});
      m.components.push_back(std::move(c));
      data[i + 1].assign(16, 9);
      m.components.push_back(
          {{{2}, 1, ComponentKind::kPrivateState, 0, 0, 16},
           {{2}, tokens},
           {},
           {},
           DiskPayload{Id(1001), 16, DiskChecksum(data[i + 1])}});
      buffers.push_back({true, Id(1001), data[i + 1]});
      auto begin = std::chrono::steady_clock::now();
      store.Publish(Id(2000), m, buffers);
      auto ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin)
                    .count();
      assert(store.Entries().size() == 1);
      auto stats = store.PublicationStats();
      std::cout << "tokens=" << tokens
                << " bytes=" << store.Stats().managed_bytes
                << " publication_ms=" << ms
                << " fsync_calls=" << stats.fsync_calls
                << " fsync_ms=" << stats.fsync_ns / 1e6 << '\n';
    }
    std::filesystem::remove_all(root);
  }
}
