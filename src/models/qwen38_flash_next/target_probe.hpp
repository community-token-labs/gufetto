#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_TARGET_PROBE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_TARGET_PROBE_HPP_

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "src/core/platform/os_info.hpp"

// Private qualification instrumentation, not a model or HTTP API. No I/O when
// the environment variable is absent. The operator arms a <=32-position window
// only after priming/import; the environment stays identical for the epoch.
namespace gufo::models::qwen38_flash_next::detail {

class TargetProbeFile {
public:
  explicit TargetProbeFile(platform::PlatformFile fd) : fd_(fd) {}
  ~TargetProbeFile() { platform::CloseFile(fd_); }
  TargetProbeFile(const TargetProbeFile&) = delete;
  TargetProbeFile& operator=(const TargetProbeFile&) = delete;
  platform::PlatformFile get() const { return fd_; }

private:
  platform::PlatformFile fd_;
};

[[noreturn]] inline void TargetProbeError() {
  // Never include paths, tokens, logits or prompt content in server errors.
  throw std::runtime_error("private target probe configuration or I/O failed");
}

inline void TargetProbePrivate(platform::PlatformFile fd) {
  platform::FileInfo info{};
  std::error_code ec;
  if (!platform::Stat(fd, &info, ec) || !info.regular ||
      !platform::OwnerOnly(info, 077))
    TargetProbeError();
}

inline platform::Directory OpenTargetProbeDirectory(const char* path) {
  // The Windows filesystem leaf has no ACL/owner validation yet. Do not
  // silently weaken the private evidence directory's security contract.
  if (std::string_view(platform::HostOsName()) != "Linux")
    throw std::runtime_error(
        "private target probe is unsupported on this host");
  if (*path != '/')
    TargetProbeError();
  platform::Directory directory;
  platform::FileInfo info{};
  std::error_code ec;
  if (!directory.Open(path, ec) ||
      !platform::StatAt(directory, ".", false, &info, ec) ||
      !platform::OwnerOnly(info, 077))
    TargetProbeError();
  return directory;
}

inline const char* TargetProbeDirectory() {
  return std::getenv("GUFO_FLASH_NEXT_TARGET_PROBE");
}

// [first,end) uses next-token positions: prompt length is the first decode.
inline std::pair<std::uint32_t, std::uint32_t> TargetProbeRange(
    std::uint32_t first, std::uint32_t count) {
  const char* path = TargetProbeDirectory();
  if (path == nullptr || count == 0)
    return {};
  auto directory = OpenTargetProbeDirectory(path);
  std::error_code ec;
  TargetProbeFile arm(
      platform::OpenAt(directory, "arm", platform::FileMode::kRead, ec));
  if (!platform::FileValid(arm.get()) &&
      ec == std::errc::no_such_file_or_directory)
    return {};
  TargetProbePrivate(arm.get());
  std::array<char, 64> text{};
  const auto size = platform::Read(arm.get(), text.data(), text.size(), ec);
  if (ec || size == 0 || size == text.size())
    TargetProbeError();
  std::string_view bounds(text.data(), size);
  if (bounds.ends_with('\n'))
    bounds.remove_suffix(1);
  const auto colon = bounds.find(':');
  if (colon == std::string_view::npos)
    TargetProbeError();
  std::uint32_t begin = 0, end = 0;
  const auto a = std::from_chars(bounds.data(), bounds.data() + colon, begin);
  const auto b = std::from_chars(bounds.data() + colon + 1,
                                 bounds.data() + bounds.size(), end);
  if (a.ec != std::errc{} || a.ptr != bounds.data() + colon ||
      b.ec != std::errc{} || b.ptr != bounds.data() + bounds.size() ||
      end <= begin || end - begin > 32)
    TargetProbeError();
  const auto lo = std::max(first, begin);
  const auto hi = std::min<std::uint64_t>(std::uint64_t{first} + count, end);
  return lo < hi ? std::pair{lo, static_cast<std::uint32_t>(hi)}
                 : std::pair<std::uint32_t, std::uint32_t>{};
}

inline void WriteTargetProbe(std::uint32_t position, const char* suffix,
                             std::span<const std::uint8_t> bytes) {
  if (TargetProbeRange(position, 1).second == 0)
    return;
  auto directory = OpenTargetProbeDirectory(TargetProbeDirectory());
  std::error_code ec;
  const auto name = std::to_string(position) + suffix;
  if (std::string_view(suffix) == ".token" ||
      std::string_view(suffix) == ".stop") {
    const auto opposite =
        std::to_string(position) +
        (std::string_view(suffix) == ".token" ? ".stop" : ".token");
    platform::FileInfo info{};
    if (platform::StatAt(directory, opposite, false, &info, ec) ||
        ec != std::errc::no_such_file_or_directory)
      TargetProbeError();
  }
  const auto created =
      platform::OpenAt(directory, name, platform::FileMode::kCreateNew, ec);
  if (!platform::FileValid(created)) {
    if (ec != std::errc::file_exists)
      TargetProbeError();
    // Preview/anchor observations can repeat a row. Never silently overwrite
    // or hide a different second observation (or an incomplete prior file).
    TargetProbeFile existing(
        platform::OpenAt(directory, name, platform::FileMode::kRead, ec));
    TargetProbePrivate(existing.get());
    std::vector<std::uint8_t> previous(bytes.size() + 1);
    std::size_t offset = 0;
    while (offset < previous.size()) {
      const auto n = platform::Read(existing.get(), previous.data() + offset,
                                    previous.size() - offset, ec);
      if (ec) {
        if (ec == std::errc::interrupted)
          continue;
        TargetProbeError();
      }
      if (n == 0)
        break;
      offset += n;
    }
    if (offset != bytes.size() ||
        !std::equal(bytes.begin(), bytes.end(), previous.begin()))
      TargetProbeError();
    return;
  }
  TargetProbeFile file(created);
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto n = platform::Write(file.get(), bytes.data() + offset,
                                   bytes.size() - offset, ec);
    if (ec == std::errc::interrupted)
      continue;
    if (ec || n == 0)
      TargetProbeError();
    offset += n;
  }
}

inline void ProbeTargetLogits(std::uint32_t position,
                              std::span<const float> logits) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float) == 4);
  WriteTargetProbe(position, ".logits",
                   {reinterpret_cast<const std::uint8_t*>(logits.data()),
                    logits.size_bytes()});
}

inline void ProbeTargetToken(std::uint32_t position, std::int32_t token,
                             bool stop = false) {
  // .token contains emitted IDs only; terminal selections get .stop instead.
  WriteTargetProbe(
      position, stop ? ".stop" : ".token",
      stop ? std::span<const std::uint8_t>{}
           : std::span<const std::uint8_t>{
                 reinterpret_cast<const std::uint8_t*>(&token), sizeof(token)});
}

}  // namespace gufo::models::qwen38_flash_next::detail
#endif
