#ifndef GUFO_MODELS_QWEN_ARCHITECTURE_HPP_
#define GUFO_MODELS_QWEN_ARCHITECTURE_HPP_

#include <string_view>

namespace gufo::models::qwen {

// Only the currently supported preview architecture. Future release names
// require explicit qualification; this is not a prefix or marketing-name test.
inline constexpr bool IsQwen4Family(std::string_view architecture) noexcept {
  return architecture == "qwen4exp";
}

}  // namespace gufo::models::qwen

#endif
