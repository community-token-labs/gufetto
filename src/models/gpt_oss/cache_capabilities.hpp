#ifndef GUFO_MODELS_GPT_OSS_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_GPT_OSS_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::gpt_oss {

// No runtime yet, so no reusable token-prefix state. The HIP runtime topic
// enables continuation once its SWA ring and full-attention KV can persist.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::models::gpt_oss

#endif  // GUFO_MODELS_GPT_OSS_CACHE_CAPABILITIES_HPP_
