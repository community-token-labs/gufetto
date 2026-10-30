#ifndef GUFO_MODELS_QWEN_ARCHITECTURE_HPP_
#define GUFO_MODELS_QWEN_ARCHITECTURE_HPP_

#include <string_view>

namespace gufo::models::qwen {

inline constexpr std::string_view kQwen4PreviewArchitecture = "qwen4exp";

// Only the currently supported preview architecture. Future release names
// require explicit qualification; this is not a prefix or marketing-name test.
inline constexpr bool IsQwen4PreviewArch(
    std::string_view architecture) noexcept {
  return architecture == kQwen4PreviewArchitecture;
}

inline constexpr std::string_view kQwen35Architecture = "qwen35";

// Architectures whose hyperparameters use the hybrid Qwen key layout read by
// GgufReader::ExtractModelConfig: the Qwen3.5/3.8 loader and the Flash-Next
// preview. Every other family has its own loader or is unsupported.
inline constexpr bool IsQwenHybridConfigArch(
    std::string_view architecture) noexcept {
  return architecture == kQwen35Architecture ||
         IsQwen4PreviewArch(architecture);
}

}  // namespace gufo::models::qwen

#endif
