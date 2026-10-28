#ifndef GUFO_SERVER_PINNED_PREFIX_HPP_
#define GUFO_SERVER_PINNED_PREFIX_HPP_

#include <cstddef>
#include <filesystem>
#include <vector>

namespace gufo::server {
// An explicit operator-owned registry. Never populated by requests.
struct TextRunnerPinnedPrefixOptions {
  static constexpr std::size_t kMaxRegistrations = 64;
  std::vector<std::filesystem::path> files;
  std::filesystem::path directory;
  std::size_t capacity_bytes{std::size_t{1} << 30};
  // Existing pins remain disk-backed instead of requiring startup RAM warmth.
  bool lazy{false};
  [[nodiscard]] bool enabled() const { return !directory.empty(); }
};
}  // namespace gufo::server
#endif
