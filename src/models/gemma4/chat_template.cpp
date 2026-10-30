#include "src/models/gemma4/chat_template.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "src/core/crypto/sha256.hpp"
#include "src/models/gemma4/control_tokens.hpp"

namespace gufo::models::gemma4 {
namespace {

using tokenization::ChatMessage;
using tokenization::ChatRole;

/// Python's str.isspace() code points, which Jinja's `trim` strips.
bool IsPythonSpace(std::uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 ||
         c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

/// Decodes the UTF-8 character starting at `i`; invalid bytes are not
/// whitespace, so their exact value does not matter here.
std::uint32_t CodePointAt(std::string_view s, std::size_t i, std::size_t* n) {
  const auto c = static_cast<unsigned char>(s[i]);
  std::size_t len = c < 0x80    ? 1
                    : c >= 0xF0 ? 4
                    : c >= 0xE0 ? 3
                    : c >= 0xC0 ? 2
                                : 1;
  if (i + len > s.size()) {
    len = 1;
  }
  std::uint32_t cp = len == 1 ? c : c & (0xFFU >> (len + 1));
  for (std::size_t k = 1; k < len; ++k) {
    cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3FU);
  }
  *n = len;
  return cp;
}

/// Python str.strip() with no arguments.
std::string PyTrim(std::string_view s) {
  std::size_t begin = 0;
  while (begin < s.size()) {
    std::size_t n = 0;
    if (!IsPythonSpace(CodePointAt(s, begin, &n))) {
      break;
    }
    begin += n;
  }
  std::size_t end = s.size();
  while (end > begin) {
    std::size_t start = end - 1;
    while (start > begin &&
           (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) {
      --start;
    }
    std::size_t n = 0;
    if (!IsPythonSpace(CodePointAt(s, start, &n)) || start + n != end) {
      break;
    }
    end = start;
  }
  return std::string(s.substr(begin, end - begin));
}

/// The template's strip_thinking macro: drops every `<|channel>…<channel|>`
/// span (and an unterminated `<|channel>` tail), then trims.
std::string StripThinking(std::string_view text) {
  std::string result;
  std::size_t start = 0;
  while (true) {
    const std::size_t split = text.find(kThoughtEnd, start);
    const std::string_view part = text.substr(
        start, split == std::string_view::npos ? std::string_view::npos
                                               : split - start);
    const std::size_t open = part.find(kChannelStart);
    result += open == std::string_view::npos ? part : part.substr(0, open);
    if (split == std::string_view::npos) {
      break;
    }
    start = split + kThoughtEnd.size();
  }
  return PyTrim(result);
}

/// The message role as the template prints it (`developer` stays itself).
std::string_view RoleName(ChatRole role) {
  switch (role) {
    case ChatRole::kSystem:
      return "system";
    case ChatRole::kDeveloper:
      return "developer";
    case ChatRole::kUser:
      return "user";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
  }
  return "user";
}

}  // namespace

ChatOptions ResolveChatOptions(const ReasoningOptions& reasoning) {
  ChatOptions options;
  if (reasoning.enabled.has_value()) {
    options.enable_thinking = *reasoning.enabled;
  } else if (reasoning.effort.has_value()) {
    options.enable_thinking = *reasoning.effort != ReasoningEffort::kMinimal;
  }
  options.preserve_thinking = reasoning.preserve_thinking.value_or(false);
  return options;
}

bool ChatTemplate::ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg) {
  const auto source = reader.GetMetadataString("tokenizer.chat_template");
  if (!source) {
    if (error_msg != nullptr) {
      *error_msg = "gemma4 GGUF has no tokenizer.chat_template";
    }
    return false;
  }
  const std::string digest = crypto::Sha256Hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t*>(source->data()), source->size()));
  if (digest != UnslothTemplateSha256()) {
    if (error_msg != nullptr) {
      *error_msg = "unqualified gemma4 chat template (sha256 " + digest + ")";
    }
    return false;
  }
  return true;
}

std::optional<RenderedPrompt> ChatTemplate::Render(
    std::span<const ChatMessage> messages,
    std::span<const tokenization::ChatTool> tools, const ChatOptions& options,
    std::string* error_msg) {
  if (!tools.empty() || std::ranges::any_of(messages, [](const ChatMessage& m) {
        return m.role == ChatRole::kTool || !m.tool_calls.empty() ||
               !m.images.empty();
      })) {
    if (error_msg != nullptr) {
      *error_msg = "gemma4 tools, tool results and media are not supported";
    }
    return std::nullopt;
  }

  std::string out(kBos);
  std::span<const ChatMessage> loop = messages;
  const bool first_is_system =
      !messages.empty() && (messages[0].role == ChatRole::kSystem ||
                            messages[0].role == ChatRole::kDeveloper);
  if (options.enable_thinking || first_is_system) {
    out += kTurnStart;
    out += "system\n";
    if (options.enable_thinking) {
      out += kThink;
      out += '\n';
    }
    if (first_is_system) {
      // Server-authored instructions follow the client's system text.
      out += PyTrim(messages[0].content + messages[0].framing_suffix);
      loop = messages.subspan(1);
    }
    out += kTurnEnd;
    out += '\n';
  }

  std::ptrdiff_t last_user = -1;
  for (std::size_t i = 0; i < loop.size(); ++i) {
    if (loop[i].role == ChatRole::kUser) {
      last_user = static_cast<std::ptrdiff_t>(i);
    }
  }
  // Per-turn context an agent replaces on every request follows another
  // user turn, never an assistant reply (that is ordinary chat, which the
  // next request keeps).
  const bool final_user_turn =
      last_user >= 1 &&
      static_cast<std::size_t>(last_user) + 1 == loop.size() &&
      loop[static_cast<std::size_t>(last_user) - 1].role == ChatRole::kUser;
  bool seen_assistant = false;
  std::optional<std::size_t> final_user_start;

  std::string_view prev_role;
  for (std::size_t index = 0; index < loop.size(); ++index) {
    const ChatMessage& message = loop[index];
    if (message.role == ChatRole::kAssistant) {
      seen_assistant = true;
    } else if (final_user_turn && seen_assistant &&
               static_cast<std::ptrdiff_t>(index) == last_user) {
      final_user_start = out.size();
    }
    const std::string_view raw_role = RoleName(message.role);
    const bool model = message.role == ChatRole::kAssistant;
    // Consecutive assistant messages share one model turn.
    if (!(model && prev_role == "assistant")) {
      out += kTurnStart;
      out += model ? std::string_view("model") : raw_role;
      out += '\n';
    }
    // Reasoning replays only after the last user turn (tool-call history,
    // the other case the template preserves, is not rendered here).
    if (!message.thought.empty() &&
        static_cast<std::ptrdiff_t>(index) > last_user) {
      out += kThoughtStart;
      out += message.thought;
      out += '\n';
      out += kThoughtEnd;
    }
    out += model ? StripThinking(message.content)
                 : PyTrim(message.content + message.framing_suffix);
    const bool continues_into_next =
        model && index + 1 < loop.size() &&
        loop[index + 1].role == ChatRole::kAssistant;
    if (!continues_into_next) {
      out += kTurnEnd;
      out += '\n';
    }
    prev_role = raw_role;
  }

  RenderedPrompt rendered;
  rendered.generation_prompt_offset = out.size();
  rendered.stable_prefix_offset = final_user_start.value_or(out.size());
  if (options.add_generation_prompt) {
    out += kTurnStart;
    out += "model\n";
    if (!options.enable_thinking) {
      out += kThoughtStart;
      out += kThoughtEnd;
    }
  }
  rendered.text = std::move(out);
  return rendered;
}

}  // namespace gufo::models::gemma4
