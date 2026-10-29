#include "src/models/qwen38_flash_next/kv_cache_validation.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

#include "src/models/qwen38_flash_next/kv_cache.hpp"

namespace gufo::models::qwen38_flash_next::detail {
namespace {
bool IsFiniteHalf(std::uint16_t bits) {
  return (bits & 0x7c00) != 0x7c00;
}
std::uint16_t ReadHalf(const std::byte* bytes) {
  return std::to_integer<std::uint16_t>(bytes[0]) |
         (std::to_integer<std::uint16_t>(bytes[1]) << 8);
}
}  // namespace

// Independent functions keep the q4 loop out of q8's register allocation and
// inlining decisions. The scalar q8 checks are unchanged.
void ValidateQ8Rows(std::span<const std::byte> input) {
  for (std::size_t i = 0; i < input.size(); i += sizeof(KvBlockQ8_0)) {
    const auto scale_bits = std::to_integer<std::uint16_t>(input[i]) |
                            (std::to_integer<std::uint16_t>(input[i + 1]) << 8);
    if ((scale_bits & 0x8000) != 0 || !IsFiniteHalf(scale_bits))
      throw std::invalid_argument("invalid Flash-Next KV scale");
    const float scale = KvHalfBitsToFloat(scale_bits);
    int largest = 0;
    for (std::size_t j = 0; j < kKvQ8BlockElements; ++j) {
      const auto q = std::bit_cast<std::int8_t>(
          std::to_integer<std::uint8_t>(input[i + 2 + j]));
      if (q == -128)
        throw std::invalid_argument("invalid Flash-Next KV reconstruction");
      largest = std::max(largest, std::abs(static_cast<int>(q)));
    }
    // One overflow check per block; finite nonnegative scales make the
    // maximum code magnitude sufficient for every reconstructed operand.
    if (!IsFiniteHalf(KvFloatToHalfBits(largest * scale)))
      throw std::invalid_argument("invalid Flash-Next KV reconstruction");
  }
}

void ValidateQ4Rows(std::span<const std::byte> input) {
  for (std::size_t i = 0; i < input.size(); i += sizeof(KvBlockQ4_0)) {
    const auto bits = ReadHalf(input.data() + i);
    if (!IsFiniteHalf(bits))
      throw std::invalid_argument("invalid Flash-Next q4 scale");
    const float scale = KvHalfBitsToFloat(bits);
    int largest = 0;
    for (std::size_t j = 0; j < 16; ++j) {
      const auto byte = std::to_integer<unsigned>(input[i + 2 + j]);
      const int low = static_cast<int>(byte & 15) - 8;
      const int high = static_cast<int>(byte >> 4) - 8;
      largest = std::max(largest, std::max(std::abs(low), std::abs(high)));
    }
    // Half overflow is monotone in magnitude and symmetric in sign, even
    // for signed q4 scales. One check covers every nibble in this block.
    if (!IsFiniteHalf(KvFloatToHalfBits(largest * scale)))
      throw std::invalid_argument("invalid Flash-Next KV reconstruction");
  }
}
}  // namespace gufo::models::qwen38_flash_next::detail
