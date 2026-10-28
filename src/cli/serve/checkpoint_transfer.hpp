#ifndef GUFO_SERVER_CHECKPOINT_TRANSFER_HPP_
#define GUFO_SERVER_CHECKPOINT_TRANSFER_HPP_

#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "src/cache/disk.hpp"

namespace gufo::server {
// Independent of the native disk and model payload ABIs. V1 is a complete,
// non-resumable checkpoint, not a native Session::SaveSnapshot struct image.
inline constexpr std::uint64_t kMaxCheckpointTransferBytes = 8ULL << 30;
inline constexpr std::size_t kCheckpointTransferPieceBytes = 1ULL << 20;
using CheckpointTransferReader =
    std::function<std::size_t(std::span<std::uint8_t>)>;
using CheckpointTransferWriter =
    std::function<bool(std::span<const std::uint8_t>)>;
class CheckpointTransferError : public std::runtime_error {
public:
  CheckpointTransferError(int status, const char* code)
      : std::runtime_error(code), status(status), code(code) {}
  int status;
  const char* code;
};
struct CheckpointTransferHeader {
  // Human-inspectable build/runtime/render-policy material, with no paths.
  // Model file hashes, payload version and mode/context are in identity.
  std::string configuration;
  cache::Identity identity;
  cache::DiskManifest manifest;
  // Cumulative native image/input identities at exact token boundaries.
  std::vector<cache::InputPrefix> image_prefixes;
  std::uint64_t stable_tokens{};
  bool decode_ready{};
  std::string epoch;
};
// Private, unlinked 0600 spool; never holds the entire payload in RAM. Read
// verifies the complete envelope, exact length, geometry and every SHA/CRC
// before returning an object that may be admitted. No caller-supplied paths.
class CheckpointTransferBlob {
public:
  using Consume = std::function<void(std::span<const std::uint8_t>)>;
  using ReadPayload = std::function<void(std::size_t, const Consume&)>;
  static std::shared_ptr<CheckpointTransferBlob> Export(
      CheckpointTransferHeader, const ReadPayload&);
  static std::shared_ptr<CheckpointTransferBlob> Import(
      const CheckpointTransferReader&, std::uint64_t content_length,
      std::span<const std::uint8_t> expected_identity,
      std::string_view expected_configuration);
  ~CheckpointTransferBlob();
  const CheckpointTransferHeader& Header() const;
  std::uint64_t Bytes() const;
  std::uint64_t PayloadBytes() const;
  void Read(std::size_t payload, std::uint64_t offset,
            std::span<std::uint8_t>) const;
  // Final destination copy rechecks the actual bytes read from the private
  // spool; filesystem corruption after receipt must not enter the RAM tier.
  void Load(std::size_t payload, std::span<std::uint8_t> destination) const;
  void Write(const CheckpointTransferWriter&) const;
  // Keeps the single-transfer quota through slow response streaming. Its
  // owner is independent of model/slot lifetime.
  std::shared_ptr<void> quota;

private:
  struct Impl;
  explicit CheckpointTransferBlob(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};
// Payload order is adapter component order, chunks then tail/private state.
std::vector<const cache::DiskPayload*> CheckpointTransferPayloads(
    const cache::DiskManifest&);
}  // namespace gufo::server
#endif
