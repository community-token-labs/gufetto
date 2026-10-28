#include "src/cli/serve/checkpoint_transfer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/temporary_file.hpp"

namespace gufo::server {
namespace {
using Hash = std::array<std::uint8_t, 32>;
constexpr std::array<std::uint8_t, 8> kMagic{'G', 'U', 'F', 'O',
                                             'C', 'K', '0', '1'};
[[noreturn]] void Invalid() {
  throw CheckpointTransferError(400, "cache_transfer_invalid");
}
[[noreturn]] void IoError() {
  throw CheckpointTransferError(507, "cache_transfer_spool_io");
}
std::uint64_t Add(std::uint64_t a, std::uint64_t b) {
  if (b > kMaxCheckpointTransferBytes || a > kMaxCheckpointTransferBytes - b)
    throw CheckpointTransferError(413, "cache_transfer_too_large");
  return a + b;
}
void Put(std::vector<std::uint8_t>& out, std::uint64_t n) {
  for (unsigned i = 0; i < 8; ++i)
    out.push_back(static_cast<std::uint8_t>(n >> (8 * i)));
}
void Append(std::vector<std::uint8_t>& out,
            std::span<const std::uint8_t> bytes) {
  out.insert(out.end(), bytes.begin(), bytes.end());
}
void Field(std::vector<std::uint8_t>& out,
           std::span<const std::uint8_t> bytes) {
  Put(out, bytes.size());
  Append(out, bytes);
}
std::span<const std::uint8_t> Raw(std::string_view text) {
  return {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()};
}
struct Cursor {
  std::span<const std::uint8_t> data;
  std::span<const std::uint8_t> Take(std::uint64_t n) {
    if (n > data.size())
      Invalid();
    auto result = data.first(n);
    data = data.subspan(n);
    return result;
  }
  std::uint64_t Number() {
    auto bytes = Take(8);
    std::uint64_t n = 0;
    for (unsigned i = 0; i < 8; ++i)
      n |= std::uint64_t(bytes[i]) << (8 * i);
    return n;
  }
  std::span<const std::uint8_t> Field() { return Take(Number()); }
};
Hash Digest(std::span<const std::uint8_t> bytes) {
  crypto::Sha256Hasher hash;
  hash.Update(bytes);
  return hash.Finish();
}
Hash Epoch(const CheckpointTransferHeader& header) {
  std::vector<std::uint8_t> material;
  Field(material, Raw("gufo-checkpoint-transfer-v1"));
  Field(material, header.identity);
  Field(material, Raw(header.configuration));
  return Digest(material);
}
std::string Hex(std::span<const std::uint8_t> bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (auto b : bytes) {
    out.push_back(digits[b >> 4]);
    out.push_back(digits[b & 15]);
  }
  return out;
}
Hash Tokens(const cache::DiskManifest& manifest) {
  crypto::Sha256Hasher hash;
  for (auto token : manifest.tokens) {
    std::array<std::uint8_t, 4> bytes;
    for (unsigned i = 0; i < 4; ++i)
      bytes[i] = static_cast<std::uint8_t>(token >> (8 * i));
    hash.Update(bytes);
  }
  return hash.Finish();
}
void Exact(const CheckpointTransferReader& reader,
           std::span<std::uint8_t> out) {
  while (!out.empty()) {
    const auto n = reader(out);
    if (!n || n > out.size())
      Invalid();
    out = out.subspan(n);
  }
}
void FileWrite(platform::PlatformFile fd, std::uint64_t offset,
               std::span<const std::uint8_t> data) {
  std::error_code ec;
  while (!platform::SeekTo(fd, offset, ec)) {
    if (ec != std::errc::interrupted)
      IoError();
  }
  while (!data.empty()) {
    const auto n = platform::Write(fd, data.data(), data.size(), ec);
    if (ec == std::errc::interrupted)
      continue;
    if (ec || n == 0)
      IoError();
    data = data.subspan(n);
  }
}
void FileRead(platform::PlatformFile fd, std::uint64_t offset,
              std::span<std::uint8_t> data) {
  std::error_code ec;
  while (!data.empty()) {
    const auto n = platform::ReadAt(fd, offset, data.data(), data.size(), ec);
    if (ec == std::errc::interrupted)
      continue;
    if (ec || n == 0)
      IoError();
    offset += n;
    data = data.subspan(n);
  }
}
std::vector<cache::DiskPayload*> MutablePayloads(cache::DiskManifest& m) {
  std::vector<cache::DiskPayload*> out;
  for (auto& c : m.components) {
    for (auto& p : c.chunks)
      out.push_back(&p);
    if (c.tail)
      out.push_back(&*c.tail);
    if (c.private_state)
      out.push_back(&*c.private_state);
  }
  return out;
}
}  // namespace
std::vector<const cache::DiskPayload*> CheckpointTransferPayloads(
    const cache::DiskManifest& m) {
  std::vector<const cache::DiskPayload*> out;
  for (const auto& c : m.components) {
    for (const auto& p : c.chunks)
      out.push_back(&p);
    if (c.tail)
      out.push_back(&*c.tail);
    if (c.private_state)
      out.push_back(&*c.private_state);
  }
  return out;
}
struct CheckpointTransferBlob::Impl {
  platform::PlatformFile fd{platform::kInvalidFile};
#if defined(_WIN32)
  mutable std::mutex read_mutex;
#endif
  CheckpointTransferHeader header;
  std::vector<std::uint8_t> encoded;
  std::vector<std::uint64_t> offsets{0};
  std::vector<Hash> hashes;
  Impl() {
    std::error_code ec;
    fd = platform::OpenTemporaryFile(ec);
    if (!platform::FileValid(fd))
      IoError();
  }
  ~Impl() { platform::CloseFile(fd); }
  void Read(std::uint64_t offset, std::span<std::uint8_t> out) const {
#if defined(_WIN32)
    // Windows ReadAt may move the handle's file position. POSIX pread is
    // position-independent and keeps its existing concurrent read path.
    const std::lock_guard lock(read_mutex);
#endif
    FileRead(fd, offset, out);
  }
  void Layout() {
    offsets.assign(1, 0);
    for (const auto* p : CheckpointTransferPayloads(header.manifest)) {
      if (!p->bytes)
        Invalid();
      offsets.push_back(Add(offsets.back(), p->bytes));
    }
    if (header.identity.empty() || header.configuration.empty() ||
        header.manifest.tokens.empty() ||
        header.stable_tokens > header.manifest.tokens.size() ||
        header.manifest.compatibility != Digest(header.identity))
      Invalid();
    cache::InputIdentity images(header.manifest.tokens.size(),
                                header.manifest.input, header.image_prefixes);
    if (!std::ranges::equal(images.At(header.manifest.tokens.size()),
                            header.manifest.input))
      Invalid();
    // Both codec directions validate exact geometry and bounded metadata.
    (void)cache::DecodeManifest(cache::EncodeManifest(header.manifest));
  }
};
CheckpointTransferBlob::CheckpointTransferBlob(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CheckpointTransferBlob::~CheckpointTransferBlob() = default;
const CheckpointTransferHeader& CheckpointTransferBlob::Header() const {
  return impl_->header;
}
std::uint64_t CheckpointTransferBlob::PayloadBytes() const {
  return impl_->offsets.back();
}
std::uint64_t CheckpointTransferBlob::Bytes() const {
  return Add(impl_->encoded.size(), PayloadBytes());
}
void CheckpointTransferBlob::Read(std::size_t payload, std::uint64_t offset,
                                  std::span<std::uint8_t> out) const {
  if (payload >= impl_->offsets.size() - 1 ||
      offset > impl_->offsets[payload + 1] - impl_->offsets[payload] ||
      out.size() >
          impl_->offsets[payload + 1] - impl_->offsets[payload] - offset)
    Invalid();
  impl_->Read(impl_->offsets[payload] + offset, out);
}
std::shared_ptr<CheckpointTransferBlob> CheckpointTransferBlob::Export(
    CheckpointTransferHeader header, const ReadPayload& read) {
  auto impl = std::make_unique<Impl>();
  impl->header = std::move(header);
  impl->Layout();
  std::vector<Hash> hashes;
  auto payloads = MutablePayloads(impl->header.manifest);
  for (std::size_t i = 0; i < payloads.size(); ++i) {
    crypto::Sha256Hasher hash;
    std::uint64_t checksum = 0;
    std::uint64_t offset = 0;
    read(i, [&](std::span<const std::uint8_t> piece) {
      if (piece.size() > payloads[i]->bytes - offset)
        Invalid();
      hash.Update(piece);
      checksum = cache::DiskChecksum(piece, checksum);
      FileWrite(impl->fd, impl->offsets[i] + offset, piece);
      offset += piece.size();
    });
    if (offset != payloads[i]->bytes)
      Invalid();
    payloads[i]->checksum = checksum;
    hashes.push_back(hash.Finish());
  }
  auto& h = impl->header;
  auto& out = impl->encoded;
  Append(out, kMagic);
  Put(out, 0);  // Filled after the bounded header has been encoded.
  Put(out, h.stable_tokens);
  Put(out, h.decode_ready);
  Put(out, impl->offsets.back());
  const auto epoch = Epoch(h);
  Append(out, epoch);
  Append(out, Tokens(h.manifest));
  Field(out, h.identity);
  Field(out, Raw(h.configuration));
  Put(out, h.image_prefixes.size());
  for (const auto& prefix : h.image_prefixes) {
    Put(out, prefix.token_count);
    Field(out, prefix.identity);
  }
  Field(out, cache::EncodeManifest(h.manifest));
  Put(out, hashes.size());
  for (const auto& hash : hashes)
    Append(out, hash);
  const auto size = out.size() + Hash{}.size();
  if (size > cache::kMaxManifestBytes)
    throw CheckpointTransferError(413, "cache_transfer_too_large");
  for (unsigned i = 0; i < 8; ++i)
    out[8 + i] = static_cast<std::uint8_t>(size >> (8 * i));
  Append(out, Digest(out));
  h.epoch = Hex(epoch);
  impl->hashes = std::move(hashes);
  auto result = std::shared_ptr<CheckpointTransferBlob>(
      new CheckpointTransferBlob(std::move(impl)));
  (void)result->Bytes();
  return result;
}
std::shared_ptr<CheckpointTransferBlob> CheckpointTransferBlob::Import(
    const CheckpointTransferReader& reader, std::uint64_t length,
    std::span<const std::uint8_t> identity, std::string_view configuration) {
  if (length > kMaxCheckpointTransferBytes)
    throw CheckpointTransferError(413, "cache_transfer_too_large");
  if (length < 16 + 32)
    Invalid();
  std::array<std::uint8_t, 16> prefix;
  Exact(reader, prefix);
  Cursor p{prefix};
  if (!std::ranges::equal(p.Take(8), kMagic))
    Invalid();
  const auto header_bytes = p.Number();
  if (header_bytes < prefix.size() + 32 || header_bytes > length)
    Invalid();
  if (header_bytes > cache::kMaxManifestBytes)
    throw CheckpointTransferError(413, "cache_transfer_too_large");
  std::vector<std::uint8_t> encoded(header_bytes);
  std::ranges::copy(prefix, encoded.begin());
  Exact(reader, std::span(encoded).subspan(prefix.size()));
  auto bytes = std::span<const std::uint8_t>(encoded);
  if (!std::ranges::equal(Digest(bytes.first(bytes.size() - 32)),
                          bytes.last(32)))
    Invalid();
  Cursor c{bytes.subspan(16, bytes.size() - 16 - 32)};
  CheckpointTransferHeader h;
  h.stable_tokens = c.Number();
  const auto ready = c.Number();
  if (ready > 1)
    Invalid();
  h.decode_ready = ready;
  const auto payload_bytes = c.Number();
  const auto epoch = c.Take(32);
  const auto token_hash = c.Take(32);
  const auto wire_identity = c.Field();
  h.identity.assign(wire_identity.begin(), wire_identity.end());
  const auto config = c.Field();
  h.configuration.assign(reinterpret_cast<const char*>(config.data()),
                         config.size());
  if (!std::ranges::equal(identity, h.identity) ||
      configuration != h.configuration || !std::ranges::equal(epoch, Epoch(h)))
    throw CheckpointTransferError(409, "cache_transfer_epoch_mismatch");
  const auto images = c.Number();
  if (images > c.data.size() / 16)
    Invalid();
  h.image_prefixes.reserve(images);
  for (std::uint64_t i = 0; i < images; ++i) {
    const auto boundary = c.Number();
    const auto identity = c.Field();
    h.image_prefixes.push_back({boundary, {identity.begin(), identity.end()}});
  }
  try {
    h.manifest = cache::DecodeManifest(c.Field());
  } catch (const std::exception&) {
    Invalid();
  }
  if (!std::ranges::equal(Tokens(h.manifest), token_hash))
    Invalid();
  const auto payloads = CheckpointTransferPayloads(h.manifest);
  const auto count = c.Number();
  if (count != payloads.size() || count > c.data.size() / 32)
    Invalid();
  auto hashes = c.Take(count * 32);
  if (!c.data.empty())
    Invalid();
  auto impl = std::make_unique<Impl>();
  impl->header = std::move(h);
  try {
    impl->Layout();
  } catch (const CheckpointTransferError&) {
    throw;
  } catch (const std::exception&) {
    Invalid();
  }
  if (impl->offsets.back() != payload_bytes ||
      Add(payload_bytes, header_bytes) != length)
    Invalid();
  impl->encoded = std::move(encoded);
  impl->header.epoch = Hex(epoch);
  for (std::size_t i = 0; i < count; ++i) {
    Hash hash;
    std::ranges::copy(hashes.subspan(i * 32, 32), hash.begin());
    impl->hashes.push_back(hash);
  }
  std::vector<std::uint8_t> staging(kCheckpointTransferPieceBytes);
  for (std::size_t i = 0; i < payloads.size(); ++i) {
    crypto::Sha256Hasher hash;
    std::uint64_t checksum = 0;
    for (std::uint64_t offset = 0; offset < payloads[i]->bytes;) {
      auto piece = std::span(staging).first(
          std::min<std::uint64_t>(staging.size(), payloads[i]->bytes - offset));
      Exact(reader, piece);
      hash.Update(piece);
      checksum = cache::DiskChecksum(piece, checksum);
      FileWrite(impl->fd, impl->offsets[i] + offset, piece);
      offset += piece.size();
    }
    if (!std::ranges::equal(hash.Finish(), hashes.subspan(i * 32, 32)) ||
        checksum != payloads[i]->checksum)
      Invalid();
  }
  return std::shared_ptr<CheckpointTransferBlob>(
      new CheckpointTransferBlob(std::move(impl)));
}
void CheckpointTransferBlob::Load(std::size_t payload,
                                  std::span<std::uint8_t> destination) const {
  if (payload >= impl_->offsets.size() - 1 ||
      destination.size() !=
          impl_->offsets[payload + 1] - impl_->offsets[payload])
    Invalid();
  Read(payload, 0, destination);
  if (Digest(destination) != impl_->hashes.at(payload))
    Invalid();
}
void CheckpointTransferBlob::Write(
    const CheckpointTransferWriter& writer) const {
  if (!writer(impl_->encoded))
    return;
  std::vector<std::uint8_t> staging(kCheckpointTransferPieceBytes);
  for (std::uint64_t offset = 0; offset < PayloadBytes();) {
    auto piece = std::span(staging).first(
        std::min<std::uint64_t>(staging.size(), PayloadBytes() - offset));
    impl_->Read(offset, piece);
    if (!writer(piece))
      return;
    offset += piece.size();
  }
}
}  // namespace gufo::server
