#ifndef GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/qwen/chat_template.hpp"

namespace gufo::models::gemma4 {

struct ChatOptions {
  bool add_generation_prompt{true};
  /// Gemma 4 thinking is opt-in: the template default is off.
  bool enable_thinking{false};
  bool preserve_thinking{false};
};

/// Resolves provider-neutral controls against the Gemma 4 template defaults.
/// Effort levels have no Gemma equivalent; any level other than minimal
/// enables thinking when it was not set explicitly.
[[nodiscard]] ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning);

struct RenderedPrompt {
  std::string text;
  /// Byte offset where the generation prompt starts; equals text.size()
  /// without one. Everything before it is the stable conversation prefix.
  std::size_t generation_prompt_offset{0};
  /// Byte offset of the prefix the next request is expected to share: the
  /// generation prompt, or, when the prompt ends with a text-only user turn
  /// after a tool result or another user turn, the end of the turn before
  /// it. Agent clients send per-turn context there and replace it on the
  /// next request. A tool result stays inside its model turn, which the next
  /// request continues, so the boundary precedes that turn's closing marker.
  std::size_t stable_prefix_offset{0};
};

/// Compiled renderer of Unsloth's Gemma 4 chat template. The GGUF template
/// is only accepted when its SHA-256 matches the qualified revision.
///
/// It renders system or developer instructions, user and assistant turns,
/// thinking on or off, replayed reasoning, tool declarations, tool calls
/// (arguments as a mapping, as llama.cpp passes them) and OpenAI `tool`
/// results, and the generation prompt. Media is rejected until the vision
/// topics land.
///
/// Known approximations of the Jinja source: message content arrives as one
/// string, so content-part lists are not representable; gufo::json keeps
/// numbers as doubles, so an integral number prints as a Python int (Python
/// would print 5.0 for a float literal) and integers beyond 2^53 lose
/// precision.
class ChatTemplate {
public:
  [[nodiscard]] static constexpr std::string_view
  UnslothTemplateSha256() noexcept {
    return "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b";
  }

  [[nodiscard]] static bool ValidateGgufTemplate(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  [[nodiscard]] static std::optional<RenderedPrompt> Render(
      std::span<const tokenization::ChatMessage> messages,
      std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
      std::string* error_msg = nullptr);
};

/// Reasoning channel markup. A replayed thought renders as
/// kThoughtStart + thought + "\n" + kThoughtEnd, as the template does.
inline constexpr std::string_view kThoughtStart = "<|channel>thought\n";
inline constexpr std::string_view kThoughtEnd = "<channel|>";

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_CHAT_TEMPLATE_HPP_
