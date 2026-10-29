#include "src/core/gguf_identity.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/fs.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::core {
namespace {

class FileDescriptor {
public:
  explicit FileDescriptor(platform::PlatformFile file) : file_(file) {}
  ~FileDescriptor() { platform::CloseFile(file_); }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  [[nodiscard]] platform::PlatformFile get() const { return file_; }

private:
  platform::PlatformFile file_;
};

void HashString(crypto::Sha256Hasher& hash, std::string_view value) {
  hash.Update(
      {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
}

std::string FileStamp(platform::PlatformFile file, std::size_t size) {
  platform::FileInfo info{};
  std::error_code ec;
  if (!platform::Stat(file, &info, ec) || !info.regular || info.size != size) {
    throw std::runtime_error("GGUF file changed or cannot be inspected");
  }
  return std::to_string(info.device) + ':' + std::to_string(info.inode) + ':' +
         std::to_string(info.size) + ':' + std::to_string(info.modified_ns) +
         ':' + std::to_string(info.changed_ns);
}

platform::Directory OpenCacheDirectory() {
  auto base = platform::UserCacheDirectory();
  if (base.empty())
    return {};
  std::error_code error;
  base /= "gufo";
  std::filesystem::create_directories(base, error);
  if (error)
    return {};
  base /= "gguf-sha256-memo-v2";
  (void)platform::MakeDirectory(base, error);
  platform::Directory directory;
  if (!directory.Open(base, error))
    return {};
  platform::FileInfo info{};
  if (!platform::StatAt(directory, ".", false, &info, error) ||
      !platform::OwnerOnly(info, 0077))
    return {};
  return directory;
}

std::string ReadDigest(const platform::Directory& directory,
                       const std::string& key) {
  if (!directory.valid())
    return {};
  std::error_code ec;
  FileDescriptor file(
      platform::OpenAt(directory, key, platform::FileMode::kRead, ec));
  platform::FileInfo info{};
  if (!platform::FileValid(file.get()) ||
      !platform::Stat(file.get(), &info, ec) || !info.regular ||
      !platform::OwnerOnly(info, 0022) || info.size != 65)
    return {};
  std::array<char, 65> bytes{};
  if (platform::Read(file.get(), bytes.data(), bytes.size(), ec) !=
          bytes.size() ||
      bytes.back() != '\n' ||
      !std::all_of(bytes.begin(), bytes.end() - 1, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    return {};
  return {bytes.data(), bytes.size() - 1};
}

void StoreDigest(const platform::Directory& directory, const std::string& key,
                 const std::string& digest) {
  if (!directory.valid())
    return;
  static std::atomic<unsigned long> serial{0};
  const std::string temp = "." + key + "." +
                           std::to_string(platform::ProcessId()) + "." +
                           std::to_string(serial.fetch_add(1));
  std::error_code ec;
  FileDescriptor file(
      platform::OpenAt(directory, temp, platform::FileMode::kCreateNew, ec));
  if (!platform::FileValid(file.get()))
    return;
  const std::string contents = digest + '\n';
  if (platform::Write(file.get(), contents.data(), contents.size(), ec) ==
      contents.size()) {
    (void)platform::Rename(directory, temp, directory, key, ec);
  }
  (void)platform::Unlink(directory, temp, ec);
}

void HashFile(crypto::Sha256Hasher& hash, platform::PlatformFile file,
              std::size_t size) {
  constexpr std::size_t kChunk = 8 * 1024 * 1024;
  constexpr std::size_t kAlignment = 4096;
  constexpr std::size_t kReaders = 4;
  // Preserve the open inode and avoid filling the remaining UMA with a second
  // copy of weights after GPU loading. Hashing stays in exact file order.
  FileDescriptor direct(size >= kChunk ? platform::ReopenUnbuffered(file)
                                       : platform::kInvalidFile);
  struct AlignedDeleter {
    std::size_t alignment;
    void operator()(std::uint8_t* pointer) const noexcept {
      ::operator delete(pointer, std::align_val_t(alignment));
    }
  };
  struct Slot {
    std::unique_ptr<std::uint8_t, AlignedDeleter> buffer{
        nullptr, AlignedDeleter{kAlignment}};
    std::future<void> read;
    std::size_t size{0};
  };
  std::array<Slot, kReaders> slots;
  std::size_t next = 0;
  const auto queue = [&](Slot& slot) {
    if (next == size)
      return;
    if (!slot.buffer) {
      slot.buffer.reset(static_cast<std::uint8_t*>(
          ::operator new(kChunk, std::align_val_t(kAlignment))));
    }
    const auto offset = next;
    slot.size = std::min(kChunk, size - next);
    next += slot.size;
    slot.read = std::async(std::launch::async, [&, offset] {
      std::size_t done = 0;
      bool buffered = !platform::FileValid(direct.get());
      while (done < slot.size) {
        const auto remaining = slot.size - done;
        const auto requested =
            buffered ? remaining
                     : (remaining + kAlignment - 1) / kAlignment * kAlignment;
        std::error_code ec;
        const auto count =
            platform::ReadAt(buffered ? file : direct.get(), offset + done,
                             slot.buffer.get() + done, requested, ec);
        if (count == 0 && ec == std::errc::interrupted)
          continue;
        if (count == 0 && !buffered &&
            (ec == std::errc::invalid_argument ||
             ec == std::errc::operation_not_supported)) {
          buffered = true;
          continue;
        }
        if (count == 0)
          throw std::runtime_error("cannot read complete GGUF for identity");
        done += std::min(count, remaining);
        if (done % kAlignment != 0)
          buffered = true;
      }
    });
  };
  for (auto& slot : slots)
    queue(slot);
  for (std::size_t index = 0; slots[index].read.valid();
       index = (index + 1) % slots.size()) {
    auto& slot = slots[index];
    slot.read.get();
    hash.Update({slot.buffer.get(), slot.size});
    queue(slot);
  }
}

// This is a bounded change detector, not a replacement for the full digest.
// Keep the layout version in the memo namespace when changing these windows.
std::string ContentFingerprint(const GgufMappedRegion& region) {
  constexpr std::size_t kBlock = 64 * 1024;
  constexpr std::size_t kWindows = 256;
  if (region.header_size > region.size)
    throw std::runtime_error("invalid GGUF header extent for identity");
  crypto::Sha256Hasher hash;
  std::array<std::uint8_t, kBlock> buffer{};
  const auto range = [&](std::size_t offset, std::size_t length) {
    HashString(hash, ":" + std::to_string(offset) + ":" +
                         std::to_string(length) + ":");
    while (length) {
      const auto requested = std::min(length, buffer.size());
      std::error_code ec;
      const auto count = platform::ReadAt(region.file_descriptor, offset,
                                          buffer.data(), requested, ec);
      if (count == 0 && ec == std::errc::interrupted)
        continue;
      if (count == 0)
        throw std::runtime_error("cannot read GGUF identity fingerprint");
      hash.Update({buffer.data(), count});
      offset += count;
      length -= count;
    }
  };
  range(0, region.header_size);
  const auto payload = region.size - region.header_size;
  if (payload <= kWindows * kBlock) {
    range(region.header_size, payload);
  } else {
    const auto span = payload - kBlock;
    for (std::size_t i = 0; i < kWindows; ++i) {
      // Distribute inclusive first/last windows without overflowing size_t.
      const auto offset = (span / (kWindows - 1)) * i +
                          (span % (kWindows - 1)) * i / (kWindows - 1);
      range(region.header_size + offset, kBlock);
    }
  }
  return hash.FinishHex();
}

std::string Probe(const GgufMappedRegion& region) {
  if (!platform::FileValid(region.file_descriptor))
    return {};
  const auto stamp = FileStamp(region.file_descriptor, region.size);
  const auto immutable = platform::ImmutableFileDigest(region.file_descriptor);
  const auto fingerprint = ContentFingerprint(region);
  if (FileStamp(region.file_descriptor, region.size) != stamp)
    throw std::runtime_error("GGUF file changed during identity probe");
  crypto::Sha256Hasher hash;
  HashString(hash, stamp + ":" + immutable + ":" + fingerprint);
  return hash.FinishHex();
}

std::string RegionDigest(const GgufMappedRegion& region) {
  crypto::Sha256Hasher hash;
  constexpr std::size_t chunk = 8 * 1024 * 1024;
  if (platform::FileValid(region.file_descriptor)) {
    HashFile(hash, region.file_descriptor, region.size);
  } else {
    const auto* bytes = static_cast<const std::uint8_t*>(region.data);
    for (std::size_t offset = 0; offset < region.size;) {
      const auto length = std::min(chunk, region.size - offset);
      hash.Update({bytes + offset, length});
      offset += length;
    }
  }
  return hash.FinishHex();
}

}  // namespace

std::string GgufIdentityHex(const GgufReader& reader) {
  const auto regions = reader.GetMappedRegions();
  std::vector<std::string> probes;
  for (const auto& region : regions)
    probes.push_back(Probe(region));
  auto cache = std::any_of(probes.begin(), probes.end(),
                           [](const auto& probe) { return !probe.empty(); })
                   ? OpenCacheDirectory()
                   : platform::Directory{};
  std::vector<std::string> digests;
  std::vector<bool> misses;
  crypto::Sha256Hasher hash;
  HashString(hash, kGgufIdentityScheme);
  for (std::size_t i = 0; i < regions.size(); ++i) {
    auto digest =
        probes[i].empty() ? std::string{} : ReadDigest(cache, probes[i]);
    misses.push_back(digest.empty());
    if (digest.empty())
      digest = RegionDigest(regions[i]);
    HashString(hash, ":" + std::to_string(regions[i].size) + ":");
    HashString(hash, digest);
    digests.push_back(std::move(digest));
  }
  // Recheck every shard, including those completed before another shard's
  // full scan. Never publish a memo for a lookup that failed verification.
  for (std::size_t i = 0; i < regions.size(); ++i) {
    if (Probe(regions[i]) != probes[i])
      throw std::runtime_error("GGUF artifact changed during identity lookup");
  }
  for (std::size_t i = 0; i < regions.size(); ++i) {
    if (misses[i] && !probes[i].empty())
      StoreDigest(cache, probes[i], digests[i]);
  }
  return hash.FinishHex();
}

}  // namespace gufo::core
