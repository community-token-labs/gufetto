#ifndef GUFO_TESTS_RECOVERY_BOUNDS_HPP_
#define GUFO_TESTS_RECOVERY_BOUNDS_HPP_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

inline constexpr int kRecoveryInconclusiveExit = 3;

struct RecoverySample {
  std::uint64_t heap, rss, threads;
};
struct RecoveryBounds {
  std::uint64_t heap_min = std::numeric_limits<std::uint64_t>::max(),
                heap_max = 0;
  std::uint64_t rss_min = std::numeric_limits<std::uint64_t>::max(),
                rss_max = 0;
  std::uint64_t threads_max = 0;
  std::uint64_t tolerance_bytes;
  explicit RecoveryBounds(const std::vector<RecoverySample>& samples,
                          std::uint64_t tolerance = 0)
      : tolerance_bytes(tolerance) {
    if (samples.empty())
      throw std::invalid_argument("no warm post-cycle recovery controls");
    for (const auto& s : samples) {
      heap_min = std::min(heap_min, s.heap);
      heap_max = std::max(heap_max, s.heap);
      rss_min = std::min(rss_min, s.rss);
      rss_max = std::max(rss_max, s.rss);
      threads_max = std::max(threads_max, s.threads);
    }
  }
  // Fixed before the run: observed warm post-cycle range plus a small number
  // of system pages, never widened after seeing a measured result. Difference
  // comparisons avoid overflow at either uint64 boundary.
  bool Contains(const RecoverySample& s) const {
    return (s.heap >= heap_min || heap_min - s.heap <= tolerance_bytes) &&
           (s.rss <= rss_max || s.rss - rss_max <= tolerance_bytes) &&
           s.threads <= threads_max;
  }
  std::size_t OutsideCount(const std::vector<RecoverySample>& samples) const {
    return std::count_if(samples.begin(), samples.end(),
                         [&](const auto& s) { return !Contains(s); });
  }
};
struct RecoveryTrend {
  bool rss_monotonic_growth = false, heap_monotonic_loss = false;
  explicit RecoveryTrend(const std::vector<RecoverySample>& samples) {
    if (samples.size() < 2)
      return;
    rss_monotonic_growth = samples.back().rss > samples.front().rss;
    heap_monotonic_loss = samples.back().heap < samples.front().heap;
    for (std::size_t i = 1; i < samples.size(); ++i) {
      rss_monotonic_growth &= samples[i].rss >= samples[i - 1].rss;
      heap_monotonic_loss &= samples[i].heap <= samples[i - 1].heap;
    }
  }
};
#endif
