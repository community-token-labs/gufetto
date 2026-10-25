#include "src/cli/serve/checkpoint_transfer.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

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
void Check(bool expected, const std::function<void()>& action,
           int status = 400) {
  bool rejected = false;
  try {
    action();
  } catch (const CheckpointTransferError& error) {
    assert(error.status == status);
    rejected = true;
  }
  assert(rejected == expected);
}
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
  // Disconnect stops a streamed response immediately, without leaking a spool.
  std::size_t calls = 0;
  source->Write([&](auto) {
    ++calls;
    return false;
  });
  assert(calls == 1);
  std::cout << "checkpoint transfer envelope checks passed\n";
}
