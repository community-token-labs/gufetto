#ifndef GUFO_MODELS_QWEN3_RERANKER_PROMPT_HPP_
#define GUFO_MODELS_QWEN3_RERANKER_PROMPT_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen3_reranker {

/// The model card's default task, used when a request names none.
inline constexpr std::string_view kDefaultInstruction =
    "Given a web search query, retrieve relevant passages that answer the "
    "query";

/// The model card's sequence limit: prefix, pair and suffix together.
inline constexpr std::size_t kDefaultMaxLength = 8192;

/// The fixed chat frame before the pair: the system turn and the opening of
/// the user turn.
[[nodiscard]] std::string PromptPrefix();
/// The fixed frame after the pair: the user turn's end and an assistant turn
/// whose thinking is already empty, so the next token is the answer.
[[nodiscard]] std::string PromptSuffix();
/// The user text for one pair, as the model card's `format_instruction`.
[[nodiscard]] std::string FormatPair(std::string_view instruction,
                                     std::string_view query,
                                     std::string_view document);

/// Tokenizes text. `parse_special` matches control-token spellings; the
/// frame uses it, request text does not.
using Encoder = std::function<std::vector<std::uint32_t>(std::string_view text,
                                                         bool parse_special)>;

/// The frame's token IDs and the answer tokens, computed once per model.
struct PromptTokens {
  std::vector<std::uint32_t> prefix;
  std::vector<std::uint32_t> suffix;
  std::uint32_t yes{0};
  std::uint32_t no{0};
};

/// Tokenizes the frame and checks that "yes" and "no" are single tokens.
/// Returns false and sets `error` otherwise.
[[nodiscard]] bool TokenizeFrame(const Encoder& encode, PromptTokens* tokens,
                                 std::string* error);

/// `prefix + pair + suffix`, as the model card's `process_inputs`. A pair
/// longer than `max_length` leaves for it keeps its first tokens when
/// `truncate` is true, as the card's right-side truncation does; otherwise,
/// or when the frame alone does not fit, it returns false.
[[nodiscard]] bool FramePair(const PromptTokens& frame,
                             const std::vector<std::uint32_t>& pair,
                             std::size_t max_length, bool truncate,
                             std::vector<std::uint32_t>* tokens);

}  // namespace gufo::models::qwen3_reranker

#endif  // GUFO_MODELS_QWEN3_RERANKER_PROMPT_HPP_
