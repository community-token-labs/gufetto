#ifndef GUFO_MODELS_GEMMA4_CONTROL_TOKENS_HPP_
#define GUFO_MODELS_GEMMA4_CONTROL_TOKENS_HPP_

#include <array>
#include <string_view>

namespace gufo::models::gemma4 {

/// Gemma 4 control-token spellings, confirmed against the pinned artifact
/// (unsloth gemma-4-26B-A4B-it-GGUF c099eb48). Ids always come from the GGUF
/// vocabulary; these constants only name the text. Qwen's shared framing
/// lives in src/models/qwen/control_tokens.hpp; Gemma's stays here.
inline constexpr std::string_view kPad = "<pad>";
inline constexpr std::string_view kEos = "<eos>";
inline constexpr std::string_view kBos = "<bos>";
inline constexpr std::string_view kTurnStart = "<|turn>";
inline constexpr std::string_view kTurnEnd = "<turn|>";
inline constexpr std::string_view kThink = "<|think|>";
inline constexpr std::string_view kChannelStart = "<|channel>";
inline constexpr std::string_view kChannelEnd = "<channel|>";
inline constexpr std::string_view kToolStart = "<|tool>";
inline constexpr std::string_view kToolEnd = "<tool|>";
inline constexpr std::string_view kToolCallStart = "<|tool_call>";
inline constexpr std::string_view kToolCallEnd = "<tool_call|>";
inline constexpr std::string_view kToolResponseStart = "<|tool_response>";
inline constexpr std::string_view kToolResponseEnd = "<tool_response|>";
/// Quotes string values inside tool declarations, calls and responses.
inline constexpr std::string_view kQuote = "<|\"|>";
inline constexpr std::string_view kImageStart = "<|image>";
inline constexpr std::string_view kImage = "<|image|>";
inline constexpr std::string_view kImageEnd = "<image|>";
inline constexpr std::string_view kAudioStart = "<|audio>";
inline constexpr std::string_view kAudio = "<|audio|>";
inline constexpr std::string_view kAudioEnd = "<audio|>";
inline constexpr std::string_view kVideo = "<|video|>";

/// Texts that end generation besides the GGUF eos (`<turn|>` in the
/// artifact), as in llama.cpp's gemma4 vocabulary. They are treated as
/// control tokens even when the file types them otherwise.
inline constexpr std::array<std::string_view, 3> kEndOfGeneration = {
    kEos, kTurnEnd, kToolResponseStart};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CONTROL_TOKENS_HPP_
