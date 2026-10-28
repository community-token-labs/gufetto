#include "src/cli/serve/checkpoint_transfer.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include "src/cli/serve/cache_lookup_support.hpp"
#include "src/core/crypto/sha256.hpp"

using namespace gufo;
using namespace gufo::server;
namespace {
std::vector<std::uint8_t> Wire(const CheckpointTransferBlob& blob) {
  std::vector<std::uint8_t> out;
  blob.Write([&](auto bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
    return true;
  });
  assert(out.size() == blob.Bytes());
  return out;
}
CheckpointTransferReader Reader(const std::vector<std::uint8_t>& data) {
  return
      [&data, offset = std::size_t{0}](std::span<std::uint8_t> bytes) mutable {
        // Exercise short reads, including header fields split across
        // boundaries.
        const auto n =
            std::min({bytes.size(), data.size() - offset, std::size_t{17}});
        std::memcpy(bytes.data(), data.data() + offset, n);
        offset += n;
        return n;
      };
}
void Check(bool expected, const std::function<void()>& action, int status = 400,
           std::string_view code = {}) {
  bool rejected = false;
  try {
    action();
  } catch (const CheckpointTransferError& error) {
    assert(error.status == status);
    if (!code.empty())
      assert(error.code == code);
    rejected = true;
  }
  assert(rejected == expected);
}
#include "tests/cli/checkpoint_identity_checks.hpp"
}  // namespace
int main() {
  cache::Identity identity{1, 2, 20, 4};
  crypto::Sha256Hasher hash;
  hash.Update(identity);
  CheckpointTransferHeader header;
  header.configuration =
      "payload_version=20\ncontext=200000\nexecutable_sha256=test\n";
  header.identity = identity;
  header.manifest = {
      .checkpoint = {12},
      .lineage = {9},
      .compatibility = hash.Finish(),
      .input = {7, 8, 9},
      .tokens = {10, 11, 12, 13, 14},
      .components = {
          {.descriptor = {{1}, 1, cache::ComponentKind::kAppendRows, 4, 4, 0},
           .position = {{1}, 5},
           .chunks = {{{1}, 16, 0}},
           .tail = cache::DiskPayload{{2}, 4, 0}},
          {.descriptor =
               {{2}, 1, cache::ComponentKind::kPrivateState, 0, 0, 64},
           .position = {{2}, 5},
           .private_state = cache::DiskPayload{{3}, 64, 0}}}};
  header.image_prefixes = {{2, {}}, {4, {7, 8}}};
  header.stable_tokens = 3;
  header.decode_ready = true;
  std::vector<std::vector<std::uint8_t>> payloads{
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
      {16, 17, 18, 19},
      std::vector<std::uint8_t>(64, 87)};
  CheckpointIdentityChecks(header, payloads);
  auto source = CheckpointTransferBlob::Export(
      header, [&](auto i, const auto& consume) { consume(payloads[i]); });
  const auto wire = Wire(*source);
  auto restored = CheckpointTransferBlob::Import(
      Reader(wire), wire.size(), identity, header.configuration);
  assert(Wire(*restored) == wire);
  assert(restored->Header().epoch == source->Header().epoch);
  assert(restored->Header().image_prefixes.size() == 2);
  assert(restored->Header().image_prefixes[1].identity ==
         cache::Identity({7, 8}));
  assert(restored->Header().stable_tokens == 3 &&
         restored->Header().decode_ready);
  for (std::size_t i = 0; i < payloads.size(); ++i) {
    std::vector<std::uint8_t> bytes(payloads[i].size());
    restored->Load(i, bytes);
    assert(bytes == payloads[i]);
  }
  // Random-access reads must not depend on a shared spool file position.
  std::vector<std::thread> readers;
  for (std::size_t t = 0; t < 4; ++t) {
    readers.emplace_back([&, t] {
      for (std::size_t n = 0; n < 64; ++n) {
        const auto i = (n + t) % payloads.size();
        const auto offset = n % payloads[i].size();
        std::vector<std::uint8_t> bytes(payloads[i].size() - offset);
        restored->Read(i, offset, bytes);
        assert(
            std::ranges::equal(bytes, std::span(payloads[i]).subspan(offset)));
      }
    });
  }
  for (auto& reader : readers)
    reader.join();
  Check(true, [&] {
    std::array<std::uint8_t, 1> byte{};
    restored->Read(0, payloads[0].size(), byte);
  });
  // Incompatible model/quant/MTP/ABI and executable/numerics/render policies.
  auto changed = identity;
  changed[2] = 19;
  Check(
      true,
      [&] {
        (void)CheckpointTransferBlob::Import(Reader(wire), wire.size(), changed,
                                             header.configuration);
      },
      409);
  Check(
      true,
      [&] {
        (void)CheckpointTransferBlob::Import(
            Reader(wire), wire.size(), identity,
            header.configuration + "different-build");
      },
      409);
  // Every possible truncation, corrupt header, each payload and trailing bytes.
  for (std::size_t n = 0; n < wire.size(); ++n) {
    auto short_wire = std::vector<std::uint8_t>(wire.begin(), wire.begin() + n);
    Check(true, [&] {
      (void)CheckpointTransferBlob::Import(Reader(short_wire), wire.size(),
                                           identity, header.configuration);
    });
  }
  for (auto index : {std::size_t{0}, std::size_t{8}, wire.size() - 1,
                     wire.size() - 65, wire.size() - 69}) {
    auto corrupt = wire;
    corrupt[index] ^= 0x80;
    Check(true, [&] {
      (void)CheckpointTransferBlob::Import(Reader(corrupt), corrupt.size(),
                                           identity, header.configuration);
    });
  }
  auto trailing = wire;
  trailing.push_back(0);
  Check(true, [&] {
    (void)CheckpointTransferBlob::Import(Reader(trailing), trailing.size(),
                                         identity, header.configuration);
  });
  bool read = false;
  Check(
      true,
      [&] {
        (void)CheckpointTransferBlob::Import(
            [&](auto) {
              read = true;
              return 0;
            },
            kMaxCheckpointTransferBytes + 1, identity, header.configuration);
      },
      413);
  assert(!read);
  // Reject advertised header limits before reading/allocating the header.
  // Impossible lengths remain malformed even if also above the size limit.
  const auto advertised = [&](std::uint64_t size, std::uint64_t total,
                              int status, std::string_view code) {
    auto prefix = std::vector<std::uint8_t>(wire.begin(), wire.begin() + 16);
    for (unsigned i = 0; i < 8; ++i)
      prefix[8 + i] = static_cast<std::uint8_t>(size >> (8 * i));
    std::size_t received = 0;
    Check(
        true,
        [&] {
          (void)CheckpointTransferBlob::Import(
              [&](std::span<std::uint8_t> out) {
                assert(received + out.size() <= prefix.size());
                std::ranges::copy(
                    std::span(prefix).subspan(received, out.size()),
                    out.begin());
                received += out.size();
                return out.size();
              },
              total, identity, header.configuration);
        },
        status, code);
    assert(received == prefix.size());
  };
  advertised(cache::kMaxManifestBytes + 1, cache::kMaxManifestBytes + 1, 413,
             "cache_transfer_too_large");
  advertised(47, 48, 400, "cache_transfer_invalid");
  advertised(49, 48, 400, "cache_transfer_invalid");
  advertised(cache::kMaxManifestBytes + 1, cache::kMaxManifestBytes, 400,
             "cache_transfer_invalid");
  // Disconnect stops a streamed response immediately, without leaking a spool.
  std::size_t calls = 0;
  source->Write([&](auto) {
    ++calls;
    return false;
  });
  assert(calls == 1);
  // Cross the bounded staging-buffer boundary and feed the export in short
  // binary pieces, including CR/LF/zero bytes (no Windows text translation).
  auto large_header = header;
  auto large_payloads = payloads;
  auto& large = large_payloads.back();
  large.resize(kCheckpointTransferPieceBytes + 37);
  for (std::size_t i = 0; i < large.size(); ++i)
    large[i] = static_cast<std::uint8_t>(i);
  large_header.manifest.components.back().descriptor.state_bytes = large.size();
  large_header.manifest.components.back().private_state->bytes = large.size();
  const auto large_source = CheckpointTransferBlob::Export(
      large_header, [&](auto i, const auto& consume) {
        auto bytes = std::span<const std::uint8_t>(large_payloads[i]);
        while (!bytes.empty()) {
          const auto piece =
              bytes.first(std::min(bytes.size(), std::size_t{8191}));
          consume(piece);
          bytes = bytes.subspan(piece.size());
        }
      });
  const auto large_wire = Wire(*large_source);
  auto large_restored = CheckpointTransferBlob::Import(
      Reader(large_wire), large_wire.size(), identity, header.configuration);
  std::vector<std::uint8_t> loaded(large.size());
  large_restored->Load(large_payloads.size() - 1, loaded);
  assert(loaded == large);
  assert(Wire(*large_restored) == large_wire);
  std::cout << "checkpoint transfer envelope checks passed\n";
}
