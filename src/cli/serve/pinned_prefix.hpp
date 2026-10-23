#ifndef GUFO_SERVER_PINNED_PREFIX_HPP_
#define GUFO_SERVER_PINNED_PREFIX_HPP_

#include <cstddef>
#include <filesystem>

namespace gufo::server {
// One operator-owned text prefix per directory. Never populated by requests.
struct TextRunnerPinnedPrefixOptions {
  std::filesystem::path file, directory;
  std::size_t capacity_bytes{std::size_t{1} << 30};
  // Existing pins remain disk-backed instead of requiring startup RAM warmth.
  bool lazy{false};
  [[nodiscard]] bool enabled() const { return !directory.empty(); }
};
}  // namespace gufo::server
#endif
