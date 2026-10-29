#ifndef GUFO_MODELS_QWEN_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::qwen {

// Served through the continuation cache. ContinuationAdapter::GetCapabilities()
// returns this record, including the persistent encoding used by AR and DFlash.
inline constexpr cache::Capabilities kCacheCapabilities{
    .continuation = true, .persistent_encoding = true};

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_CACHE_CAPABILITIES_HPP_
