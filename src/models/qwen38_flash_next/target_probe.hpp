#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_TARGET_PROBE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_TARGET_PROBE_HPP_

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
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

// Private qualification instrumentation, not a model or HTTP API. No I/O when
// the environment variable is absent. The operator arms a <=32-position window
// only after priming/import; the environment stays identical for the epoch.
namespace gufo::models::qwen38_flash_next::detail {

class TargetProbeFile {
public:
  explicit TargetProbeFile(int fd) : fd_(fd) {}
  ~TargetProbeFile() {
    if (fd_ >= 0)
      (void)::close(fd_);
  }
  TargetProbeFile(const TargetProbeFile&) = delete;
  TargetProbeFile& operator=(const TargetProbeFile&) = delete;
  int get() const { return fd_; }

private:
  int fd_;
};

[[noreturn]] inline void TargetProbeError() {
  // Never include paths, tokens, logits or prompt content in server errors.
  throw std::runtime_error("private target probe configuration or I/O failed");
}

inline void TargetProbePrivate(int fd, bool directory) {
  struct stat info{};
  if (fd < 0 || ::fstat(fd, &info) != 0 || info.st_uid != ::geteuid() ||
      (info.st_mode & 077) != 0 ||
      (directory ? !S_ISDIR(info.st_mode) : !S_ISREG(info.st_mode)))
    TargetProbeError();
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
  if (*path != '/')
    TargetProbeError();
  TargetProbeFile directory(
      ::open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  TargetProbePrivate(directory.get(), true);
  TargetProbeFile arm(::openat(directory.get(), "arm",
                               O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
  if (arm.get() < 0 && errno == ENOENT)
    return {};
  TargetProbePrivate(arm.get(), false);
  std::array<char, 64> text{};
  const auto size = ::read(arm.get(), text.data(), text.size());
  if (size <= 0 || size == static_cast<ssize_t>(text.size()))
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
  TargetProbeFile directory(::open(
      TargetProbeDirectory(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  TargetProbePrivate(directory.get(), true);
  const auto name = std::to_string(position) + suffix;
  if (std::string_view(suffix) == ".token" ||
      std::string_view(suffix) == ".stop") {
    const auto opposite =
        std::to_string(position) +
        (std::string_view(suffix) == ".token" ? ".stop" : ".token");
    struct stat info{};
    if (::fstatat(directory.get(), opposite.c_str(), &info,
                  AT_SYMLINK_NOFOLLOW) == 0 ||
        errno != ENOENT)
      TargetProbeError();
  }
  const int created =
      ::openat(directory.get(), name.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (created < 0) {
    if (errno != EEXIST)
      TargetProbeError();
    // Preview/anchor observations can repeat a row. Never silently overwrite
    // or hide a different second observation (or an incomplete prior file).
    TargetProbeFile existing(
        ::openat(directory.get(), name.c_str(),
                 O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    TargetProbePrivate(existing.get(), false);
    std::vector<std::uint8_t> previous(bytes.size() + 1);
    std::size_t offset = 0;
    while (offset < previous.size()) {
      const auto n = ::read(existing.get(), previous.data() + offset,
                            previous.size() - offset);
      if (n < 0) {
        if (errno == EINTR)
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
    const auto n =
        ::write(file.get(), bytes.data() + offset, bytes.size() - offset);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
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
