#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cache/backing.hpp"
#include "src/core/platform/pages.hpp"

namespace {
// Linux-only test registration. Read current RSS, not the monotonic ru_maxrss
// high-water mark. smaps_rollup avoids statm's batched per-thread RSS counters.
std::size_t RssBytes() {
  std::ifstream input("/proc/self/smaps_rollup");
  std::string line;
  while (std::getline(input, line)) {
    if (line.starts_with("Rss:"))
      return std::stoull(line.substr(4)) * 1024;
  }
  throw std::runtime_error("current RSS unavailable: /proc/self/smaps_rollup");
}
}  // namespace

int main() {
  using namespace gufo;
  constexpr std::size_t segment = 8ULL << 20;
  constexpr std::size_t metadata = 64ULL << 10;
  constexpr std::size_t budget = 4 * segment + metadata;
  cache::ResourceLedger ledger({budget, budget, 0, budget});
  cache::CommittedBackingPool pool(
      ledger, {budget, segment, metadata, segment, {}, {}},
      {[](std::size_t bytes) {
         return cache::BackingPages{
             platform::CommitZeroedPages(bytes), {}, false};
       },
       [](cache::BackingPages& pages, std::size_t bytes) {
         return platform::TryReleaseCommittedPages(pages.data, bytes);
       }});
  // Warm parsing/iostream allocations before the baseline.
  (void)RssBytes();
  std::cout << "physical backing RSS probe (bytes)\n";
  const auto before = RssBytes();
  std::vector<cache::BackingBlock> held;
  held.reserve(4);
  for (int i = 0; i < 4; ++i) {
    auto block = pool.TryAcquire(cache::ResourceCategory::kBackingAssigned);
    assert(block);
    std::ranges::fill(block->Bytes(), std::byte(i + 1));
    block->Convert();
    held.push_back(std::move(*block));
  }
  const auto committed = RssBytes();
  auto* survivor = held.back().Bytes().data();
  held.erase(held.begin(), held.begin() + 3);
  const auto logically_free = RssBytes();
  const auto committed_before_trim = pool.CommittedBytes();
  assert(committed_before_trim == 4 * segment);
  const auto released = pool.TrimFreeSegments(3 * segment);
  const auto committed_after_trim = pool.CommittedBytes();
  const auto trimmed = RssBytes();
  assert(released == 3 * segment && committed_after_trim == segment);
  assert(held.front().Bytes().data() == survivor);
  assert(std::ranges::all_of(held.front().Bytes(),
                             [](auto b) { return b == std::byte{4}; }));
  std::cout << "rss_before_bytes=" << before
            << " rss_after_commit_bytes=" << committed
            << " rss_after_logical_release_bytes=" << logically_free
            << " rss_after_trim_bytes=" << trimmed
            << " pool_committed_before_trim_bytes=" << committed_before_trim
            << " pool_committed_after_trim_bytes=" << committed_after_trim
            << " actual_released_bytes=" << released << '\n';
  // Allow bounded runtime bookkeeping noise, not enough to pass a fake counter
  // or logical-only reclamation. The release must change measured resident RAM.
  constexpr std::size_t noise = 4ULL << 20;
  assert(committed >= before + 4 * segment - noise);
  assert(logically_free + noise >= committed);
  assert(logically_free >= trimmed + released - noise);
  held.clear();
  assert(pool.TrimFreeSegments(std::numeric_limits<std::size_t>::max()) ==
         segment);
  assert(ledger.Snapshot().total_bytes == metadata);
  auto recommitted = pool.TryAcquire(cache::ResourceCategory::kPrivateTail);
  assert(recommitted && pool.CommittedBytes() == segment);
  assert(std::ranges::all_of(recommitted->Bytes(),
                             [](auto b) { return b == std::byte{}; }));
}
