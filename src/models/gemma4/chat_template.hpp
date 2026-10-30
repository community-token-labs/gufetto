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
  /// generation prompt, or, when the prompt ends with a user turn that
  /// follows another user turn, the end of the turn before it. Agent clients
  /// send per-turn context there and replace it on the next request.
  std::size_t stable_prefix_offset{0};
};

/// Compiled renderer of Unsloth's Gemma 4 chat template. The GGUF template
/// is only accepted when its SHA-256 matches the qualified revision.
///
/// This renders text conversations: system or developer instructions,
/// user and assistant turns, thinking on or off, replayed reasoning after
/// the last user turn, and the generation prompt. Tools, tool calls, tool
/// results and media are rejected until their renderer lands. Message
/// content arrives as one string, so a system message's content-part list
/// (which the template joins with spaces) is not representable here.
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
