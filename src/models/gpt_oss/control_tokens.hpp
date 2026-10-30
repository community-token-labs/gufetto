#ifndef GUFO_MODELS_GPT_OSS_CONTROL_TOKENS_HPP_
#define GUFO_MODELS_GPT_OSS_CONTROL_TOKENS_HPP_

#include <array>
#include <cstdint>
#include <string_view>

namespace gufo::models::gpt_oss {

/// Harmony framing of the o200k_harmony vocabulary. Production rendering and
/// parsing use these names; see README.md for guidance.
inline constexpr std::string_view kStartOfText = "<|startoftext|>";
inline constexpr std::string_view kEndOfText = "<|endoftext|>";
inline constexpr std::string_view kReturn = "<|return|>";
inline constexpr std::string_view kConstrain = "<|constrain|>";
inline constexpr std::string_view kChannel = "<|channel|>";
inline constexpr std::string_view kStart = "<|start|>";
inline constexpr std::string_view kEnd = "<|end|>";
inline constexpr std::string_view kMessage = "<|message|>";
inline constexpr std::string_view kCall = "<|call|>";
inline constexpr std::string_view kEndOfPrompt = "<|endofprompt|>";

struct ControlTokenId {
  std::string_view text;
  std::int32_t id;
};

/// Ids pinned from ggml-org/gpt-oss-120b-GGUF@7200b74b (tokenizer.ggml.tokens).
/// `kStartOfText` is BOS, `kEndOfText` padding and `kReturn` EOS; generation
/// also stops at `kCall`.
inline constexpr std::array<ControlTokenId, 10> kControlTokenIds{{
    {kStartOfText, 199998},
    {kEndOfText, 199999},
    {kReturn, 200002},
    {kConstrain, 200003},
    {kChannel, 200005},
    {kStart, 200006},
    {kEnd, 200007},
    {kMessage, 200008},
    {kCall, 200012},
    {kEndOfPrompt, 200018},
}};

}  // namespace gufo::models::gpt_oss

#endif  // GUFO_MODELS_GPT_OSS_CONTROL_TOKENS_HPP_
