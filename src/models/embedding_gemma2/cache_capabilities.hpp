#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::embedding_gemma2 {

// Embedding requests encode complete inputs, not reusable token-prefix state.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::models::embedding_gemma2

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_CACHE_CAPABILITIES_HPP_
