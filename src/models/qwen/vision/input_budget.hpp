#ifndef GUFO_MODELS_QWEN_VISION_INPUT_BUDGET_HPP_
#define GUFO_MODELS_QWEN_VISION_INPUT_BUDGET_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace gufo::models::qwen::vision {

/// Zero preserves the full-context bound; a larger cap cannot expand context.
[[nodiscard]] constexpr std::uint32_t ImageTokenLimit(
    std::uint32_t context, std::uint32_t configured = 0) noexcept {
  return configured == 0 ? context : std::min(context, configured);
}

/// One budget for the entire history, including images before a cache hit.
class ImageTokenBudget {
public:
  explicit ImageTokenBudget(std::uint32_t limit) : limit_(limit) {}
  void Add(std::uint64_t rows) {
    if (rows > limit_ - used_)
      throw std::length_error(
          "aggregate image tokens exceed --vision-max-tokens-per-slot (" +
          std::to_string(limit_) + "); reduce images or increase the limit");
    used_ += rows;
  }

private:
  std::uint64_t limit_;
  std::uint64_t used_{0};
};

inline constexpr std::uint32_t kDefaultOutputPoolRows = 65536;

[[nodiscard]] inline std::size_t OutputReservationBytes(std::uint32_t rows,
                                                        std::uint32_t hidden) {
  const auto elements = std::uint64_t{rows} * hidden;
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float))
    throw std::overflow_error("vision output pool byte size overflows");
  return static_cast<std::size_t>(elements) * sizeof(float);
}

[[nodiscard]] inline std::size_t InputPositionBytes(std::uint32_t context) {
  const auto bytes = std::uint64_t{context} * 12 + 64;
  if (bytes > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("vision position reservation overflows");
  return static_cast<std::size_t>(bytes);
}

/// Embeddings are bounded by image rows, but positions span preceding text
/// through the last image and retain their context-sized high-water allowance.
[[nodiscard]] inline std::size_t InputReservationBytes(
    std::uint32_t context, std::uint32_t hidden, std::uint32_t configured = 0) {
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  const auto rows = ImageTokenLimit(context, configured);
  if (std::uint64_t{context} * 12 + 64 > max ||
      std::uint64_t{hidden} * sizeof(float) > max)
    throw std::overflow_error("vision input reservation overflows");
  const auto positions = std::size_t{context} * 12 + 64;
  const auto row_bytes = std::size_t{hidden} * sizeof(float);
  if (rows != 0 && row_bytes > (max - positions) / rows)
    throw std::overflow_error("vision input reservation overflows");
  return rows * row_bytes + positions;
}

}  // namespace gufo::models::qwen::vision
#endif
