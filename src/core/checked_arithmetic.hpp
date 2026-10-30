#ifndef GUFO_CORE_CHECKED_ARITHMETIC_HPP_
#define GUFO_CORE_CHECKED_ARITHMETIC_HPP_

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>

namespace gufo::core {

/// Overflow-checked 64-bit arithmetic for sizes derived from file metadata.
/// Callers refuse the file when a result is absent; they never clamp.
[[nodiscard]] constexpr std::optional<std::uint64_t> CheckedAdd(
    std::uint64_t lhs, std::uint64_t rhs) noexcept {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

[[nodiscard]] constexpr std::optional<std::uint64_t> CheckedMul(
    std::uint64_t lhs, std::uint64_t rhs) noexcept {
  if (rhs != 0 && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
    return std::nullopt;
  }
  return lhs * rhs;
}

/// Product of every factor, or nullopt when any partial product overflows.
[[nodiscard]] constexpr std::optional<std::uint64_t> CheckedProduct(
    std::initializer_list<std::uint64_t> factors) noexcept {
  std::uint64_t product = 1;
  for (const std::uint64_t factor : factors) {
    const auto next = CheckedMul(product, factor);
    if (!next.has_value()) {
      return std::nullopt;
    }
    product = *next;
  }
  return product;
}

/// `value` when it fits a 32-bit width, otherwise nullopt.
[[nodiscard]] constexpr std::optional<std::uint32_t> CheckedUint32(
    std::optional<std::uint64_t> value) noexcept {
  if (!value.has_value() ||
      *value > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(*value);
}

}  // namespace gufo::core

#endif  // GUFO_CORE_CHECKED_ARITHMETIC_HPP_
