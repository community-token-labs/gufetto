#ifndef GUFO_MODELS_GEMMA4_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_GEMMA4_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::gemma4 {

// Only the config contract exists so far; there is no runtime state to
// continue until the continuation adapter lands.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CACHE_CAPABILITIES_HPP_
