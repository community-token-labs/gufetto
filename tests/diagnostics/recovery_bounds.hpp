#ifndef GUFO_TESTS_RECOVERY_BOUNDS_HPP_
#define GUFO_TESTS_RECOVERY_BOUNDS_HPP_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

struct RecoverySample {
  std::uint64_t heap, rss, threads;
};
struct RecoveryBounds {
  std::uint64_t heap_min = std::numeric_limits<std::uint64_t>::max(),
                heap_max = 0;
  std::uint64_t rss_min = std::numeric_limits<std::uint64_t>::max(),
                rss_max = 0;
  std::uint64_t threads_max = 0;
  explicit RecoveryBounds(const std::vector<RecoverySample>& samples) {
    for (const auto& s : samples) {
      heap_min = std::min(heap_min, s.heap);
      heap_max = std::max(heap_max, s.heap);
      rss_min = std::min(rss_min, s.rss);
      rss_max = std::max(rss_max, s.rss);
      threads_max = std::max(threads_max, s.threads);
    }
  }
  // Only the observed post-prime control interval is allowed: no fixed byte
  // allowance or retry-time widening. Extra free heap / lower RSS is harmless.
  bool Contains(const RecoverySample& s) const {
    return s.heap >= heap_min && s.rss <= rss_max && s.threads <= threads_max;
  }
};
#endif
