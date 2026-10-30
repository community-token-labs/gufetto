#ifndef GUFO_TESTS_RSS_SLOPE_HPP_
#define GUFO_TESTS_RSS_SLOPE_HPP_

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Lead preregistration: 20 measured points; fit ONLY the last 15 by cycle
// index.
struct RssDiscriminatorResult {
  long double slope_bytes_per_cycle, growth_bytes;
  enum class Verdict { kPass, kFail, kBetween } verdict;
};
inline RssDiscriminatorResult ClassifyRssAnon(
    const std::vector<std::uint64_t>& measured) {
  if (measured.size() != 20)
    throw std::invalid_argument(
        "RSS discriminator requires all 20 measured cycles");
  constexpr std::size_t first = 5;
  const long double origin = measured[first];
  long double numerator = 0;
  // For x=0..14, mean(x)=7 and sum((x-7)^2)=280. Subtract an
  // origin before multiplication to avoid loss of precision on large RSS.
  for (int x = 0; x < 15; ++x)
    numerator +=
        (x - 7) * (static_cast<long double>(measured[first + x]) - origin);
  const auto slope = numerator / 280;
  const auto growth =
      std::max(0.0L, static_cast<long double>(measured.back()) - origin);
  using Verdict = RssDiscriminatorResult::Verdict;
  const auto verdict = slope > 256 * 1024 ? Verdict::kFail
                       : slope <= 64 * 1024 && growth <= 2 * 1024 * 1024
                           ? Verdict::kPass
                           : Verdict::kBetween;
  return {slope, growth, verdict};
}
#endif
