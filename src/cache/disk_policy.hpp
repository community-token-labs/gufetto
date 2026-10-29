#ifndef GUFO_CACHE_DISK_POLICY_HPP_
#define GUFO_CACHE_DISK_POLICY_HPP_

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

#include "src/core/platform/fs.hpp"

namespace gufo::cache {
inline constexpr std::uint64_t kDiskAutomaticCap = 64ULL << 30;
inline constexpr std::uint64_t kDiskMinimumReserve = 8ULL << 30;
// Allocation/metadata slack per admission, additional to the configured floor.
inline constexpr std::uint64_t kDiskAllocationSlack = 64ULL << 10;
using DiskSpaceProbe =
    std::function<platform::FilesystemInfo(const platform::Directory&)>;
enum class DiskSpaceState { kReady, kPressure, kUnavailable };

struct DiskSpacePolicy {
  // Zero disables the guard for internal callers; CLI always resolves a floor.
  std::uint64_t minimum_free_bytes{};
  DiskSpaceProbe probe{platform::InspectFilesystem};
  [[nodiscard]] DiskSpaceState Check(const platform::Directory& directory,
                                     std::uint64_t additional = 0) const {
    if (!minimum_free_bytes)
      return DiskSpaceState::kReady;
    const auto info = probe(directory);
    if (!info.supported || !info.writable)
      return DiskSpaceState::kUnavailable;
    if (info.available_bytes < minimum_free_bytes)
      return DiskSpaceState::kPressure;
    const auto headroom = info.available_bytes - minimum_free_bytes;
    return headroom < kDiskAllocationSlack ||
                   additional > headroom - kDiskAllocationSlack
               ? DiskSpaceState::kPressure
               : DiskSpaceState::kReady;
  }
};

struct DiskCachePlan {
  std::filesystem::path directory;
  std::uint64_t capacity_bytes{}, minimum_free_bytes{};
  std::string reason;
  [[nodiscard]] bool Enabled() const { return !directory.empty(); }
};
[[nodiscard]] inline std::uint64_t AutomaticDiskCapacity(
    const platform::FilesystemInfo& info) {
  return std::min(kDiskAutomaticCap, info.available_bytes / 10);
}
[[nodiscard]] inline std::uint64_t AutomaticDiskReserve(
    const platform::FilesystemInfo& info) {
  return std::max(kDiskMinimumReserve, info.capacity_bytes / 50);
}
[[nodiscard]] inline std::filesystem::path AutomaticDiskDirectory(
    const std::filesystem::path& user_cache) {
  return user_cache.is_absolute() ? user_cache / "gufo" / "continuations"
                                  : std::filesystem::path{};
}
// Empty/off disables; auto uses an absolute per-user cache base. A failed plan
// is a downgrade for auto and a caller-visible error for an explicit directory.
// Probe the nearest existing ancestor without creating directories.
[[nodiscard]] DiskCachePlan PlanDiskCache(
    const std::filesystem::path& option,
    const std::filesystem::path& user_cache, std::uint64_t capacity = 0,
    std::uint64_t minimum_free = 0,
    const DiskSpaceProbe& probe = platform::InspectFilesystem);
}  // namespace gufo::cache
#endif
