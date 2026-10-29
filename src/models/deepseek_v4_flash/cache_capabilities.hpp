#ifndef GUFO_MODELS_DEEPSEEK_V4_FLASH_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_DEEPSEEK_V4_FLASH_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::deepseek_v4_flash {

// DeepSeekTextRunner still uses legacy snapshots, not a component adapter.
// Its prefix-reuse and persistence declarations consume this record. Encoding
// is supported; enabling it for a loaded model also requires its fingerprint.
inline constexpr cache::Capabilities kCacheCapabilities{
    .continuation = true, .persistent_encoding = true};

}  // namespace gufo::models::deepseek_v4_flash

#endif  // GUFO_MODELS_DEEPSEEK_V4_FLASH_CACHE_CAPABILITIES_HPP_
