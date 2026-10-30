#ifndef GUFO_MODELS_QWEN3_RERANKER_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN3_RERANKER_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::qwen3_reranker {

// Rerank requests score complete query/document pairs, not reusable
// token-prefix state.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::models::qwen3_reranker

#endif  // GUFO_MODELS_QWEN3_RERANKER_CACHE_CAPABILITIES_HPP_
