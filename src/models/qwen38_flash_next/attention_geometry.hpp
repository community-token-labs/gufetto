#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_ATTENTION_GEOMETRY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_ATTENTION_GEOMETRY_HPP_

#include <cstdint>

namespace gufo::models::qwen38_flash_next {

inline constexpr char kAttentionGeometryError[] =
    "unsupported Flash-Next attention geometry: requires head_dim=256, "
    "positive compression ratio and Q/KV head grouping divisible by 4";
inline constexpr char kIndexerGeometryError[] =
    "unsupported Flash-Next indexer geometry: requires 4 heads, dimension=128 "
    "and positive compression ratio";

// Low-level kernel contracts, not admission of a new executable model profile.
inline constexpr bool IsSupportedAttentionGeometry(
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim,
    std::uint32_t ratio) noexcept {
  return dim == 256 && ratio > 0 && heads > 0 && kv_heads > 0 &&
         heads % kv_heads == 0 && (heads / kv_heads) % 4 == 0;
}

inline constexpr bool IsSupportedIndexerGeometry(std::uint32_t heads,
                                                 std::uint32_t dim,
                                                 std::uint32_t ratio) noexcept {
  return heads == 4 && dim == 128 && ratio > 0;
}

}  // namespace gufo::models::qwen38_flash_next

#endif
