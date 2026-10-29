#include "src/cache/disk_policy.hpp"

namespace gufo::cache {
DiskCachePlan PlanDiskCache(const std::filesystem::path& option,
                            const std::filesystem::path& user_cache,
                            std::uint64_t capacity, std::uint64_t minimum_free,
                            const DiskSpaceProbe& probe) {
  DiskCachePlan result;
  if (option.empty() || option == "off") {
    result.reason = "disabled";
    return result;
  }
  const bool automatic = option == "auto";
  if (automatic && !user_cache.is_absolute()) {
    result.reason = "no_absolute_user_cache";
    return result;
  }
  const auto path = automatic ? AutomaticDiskDirectory(user_cache) : option;
  std::error_code ec;
  const auto absolute = std::filesystem::absolute(path, ec);
  auto ancestor = absolute;
  if (!ec) {
    while (!std::filesystem::exists(ancestor, ec) && !ec &&
           ancestor != ancestor.parent_path())
      ancestor = ancestor.parent_path();
  }
  // Intermediate symlinks are valid in the configured path (as in OpenRoot).
  // Resolve only a probe ancestor; never relax refusal of a symlink cache leaf.
  if (!ec && ancestor != absolute)
    ancestor = std::filesystem::canonical(ancestor, ec);
  platform::Directory directory;
  if (ec || !directory.Open(ancestor, ec)) {
    result.reason = "inaccessible_path";
    return result;
  }
  const auto info = probe(directory);
  if (!info.supported || !info.writable) {
    result.reason = info.supported ? "read_only" : "unsupported_filesystem";
    return result;
  }
  result.capacity_bytes = capacity ? capacity : AutomaticDiskCapacity(info);
  result.minimum_free_bytes =
      minimum_free ? minimum_free : AutomaticDiskReserve(info);
  if (!result.capacity_bytes) {
    result.reason = "no_capacity";
    return result;
  }
  result.directory = absolute;
  result.reason = "enabled";
  return result;
}
}  // namespace gufo::cache
