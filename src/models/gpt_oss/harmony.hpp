#ifndef GUFO_MODELS_GPT_OSS_HARMONY_HPP_
#define GUFO_MODELS_GPT_OSS_HARMONY_HPP_

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::gpt_oss {

enum class HarmonyEffort : unsigned char { kLow, kMedium, kHigh };

struct HarmonyToolCall {
  /// Function name without the `functions.` namespace.
  std::string name;
  /// JSON text of the arguments, as the model wrote it.
  std::string arguments;
};

struct HarmonyMessage {
  /// system, developer, user, assistant or tool.
  std::string role;
  std::string content;
  /// Assistant analysis-channel text; empty means none.
  std::string thinking;
  /// Only the first call is rendered, as in the pinned template.
  std::vector<HarmonyToolCall> tool_calls;
};

struct HarmonyTool {
  std::string name;
  std::string description;
  /// JSON Schema object; empty means no parameters.
  std::string parameters_json;
};

struct HarmonyRenderOptions {
  std::string model_identity{
      "You are ChatGPT, a large language model trained by OpenAI."};
  /// YYYY-MM-DD, as the template's strftime_now("%Y-%m-%d").
  std::string current_date;
  HarmonyEffort effort{HarmonyEffort::kMedium};
  bool add_generation_prompt{true};
};

/// SHA-256 of the chat template the renderer is compiled from: the GGUF's
/// tokenizer.chat_template, identical to llama.cpp's
/// models/templates/openai-gpt-oss-120b.jinja.
[[nodiscard]] constexpr std::string_view HarmonyTemplateSha256() noexcept {
  return "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146";
}

/// Renders a conversation as the pinned template does. Returns nullopt with
/// `error` set where the template raises, for invalid tool-call arguments and
/// for schema constructs not compiled yet (oneOf).
[[nodiscard]] std::optional<std::string> RenderHarmony(
    std::span<const HarmonyMessage> messages,
    std::span<const HarmonyTool> tools, const HarmonyRenderOptions& options,
    std::string* error = nullptr);

struct HarmonyCompletion {
  /// Analysis-channel text, concatenated in order.
  std::string reasoning;
  /// Final-channel text and commentary preambles, concatenated in order.
  std::string content;
  /// Messages addressed to `functions.*`.
  std::vector<HarmonyToolCall> tool_calls;
  /// Messages addressed to other recipients (built-in browser/python tools),
  /// which the API surface does not execute.
  std::size_t dropped_messages{0};
  /// Every header was well formed.
  bool valid{true};
  /// The last message ended with `<|return|>` or `<|call|>`.
  bool complete{false};
};

/// Parses text generated after the `<|start|>assistant` generation prompt.
/// Messages end at `<|end|>`, `<|call|>` or `<|return|>`; a missing terminator
/// leaves `complete` false. Both recipient positions are accepted:
///   <|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>
///   <|channel|>commentary to=functions.NAME <|constrain|>json<|message|>
[[nodiscard]] HarmonyCompletion ParseHarmonyCompletion(std::string_view text);

}  // namespace gufo::models::gpt_oss

#endif  // GUFO_MODELS_GPT_OSS_HARMONY_HPP_
