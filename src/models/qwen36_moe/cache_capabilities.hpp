#ifndef GUFO_MODELS_QWEN36_MOE_CACHE_CAPABILITIES_HPP_
#define GUFO_MODELS_QWEN36_MOE_CACHE_CAPABILITIES_HPP_

#include "src/cache/types.hpp"

namespace gufo::models::qwen36_moe {

// Loader only: nothing is served yet. The continuation adapter, and with it
// continuation and persistent encoding, arrives with the runtime.
inline constexpr cache::Capabilities kCacheCapabilities{.continuation = false};

}  // namespace gufo::models::qwen36_moe

#endif  // GUFO_MODELS_QWEN36_MOE_CACHE_CAPABILITIES_HPP_
