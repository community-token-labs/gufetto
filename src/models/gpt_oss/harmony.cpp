#include "src/models/gpt_oss/harmony.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <string>

#include "src/core/json.hpp"
#include "src/models/gpt_oss/control_tokens.hpp"

namespace gufo::models::gpt_oss {
namespace {

using json::Value;

constexpr std::string_view kToolsNamespace = "functions";
// The template's nested-object rendering keeps the source indentation of its
// `{{ render_typescript_type(...) }}` line.
constexpr std::string_view kNestedTypeIndent = "\n                ";

struct RenderError {
  std::string message;
};

std::string_view EffortName(HarmonyEffort effort) {
  switch (effort) {
    case HarmonyEffort::kLow:
      return "low";
    case HarmonyEffort::kHigh:
      return "high";
    case HarmonyEffort::kMedium:
      break;
  }
  return "medium";
}

// Jinja truthiness of an optional member.
bool Truthy(const Value* value) {
  if (value == nullptr) {
    return false;
  }
  switch (value->type()) {
    case Value::Type::kNull:
      return false;
    case Value::Type::kBool:
      return value->as_bool();
    case Value::Type::kNumber:
      return value->as_double() != 0.0;
    default:
      return !value->empty() || !value->str().empty();
  }
}

bool Contains(const Value* list, const std::string& name) {
  if (list == nullptr || !list->is_array()) {
    return false;
  }
  return std::ranges::any_of(list->items(), [&](const Value& item) {
    return item.is_string() && item.str() == name;
  });
}

std::string_view TypeName(const Value& spec) {
  const Value* type = spec.find("type");
  return type != nullptr ? std::string_view(type->str()) : std::string_view{};
}

std::string NullableSuffix(const Value& spec) {
  return Truthy(spec.find("nullable")) ? " | null" : "";
}

// render_typescript_type from the pinned template.
// NOLINTNEXTLINE(misc-no-recursion)
std::string TypeScriptType(const Value& spec) {
  const std::string_view type = TypeName(spec);
  const Value* type_value = spec.find("type");
  if (type == "array") {
    const Value* items = spec.find("items");
    if (!Truthy(items)) {
      return "any[]" + NullableSuffix(spec);
    }
    const std::string_view item_type = TypeName(*items);
    std::string rendered;
    if (item_type == "string") {
      rendered = "string[]";
    } else if (item_type == "number" || item_type == "integer") {
      rendered = "number[]";
    } else if (item_type == "boolean") {
      rendered = "boolean[]";
    } else {
      const std::string inner = TypeScriptType(*items);
      rendered = inner == "object | object" || inner.size() > 50 ? "any[]"
                                                                 : inner + "[]";
    }
    return rendered + NullableSuffix(spec);
  }
  if (type_value != nullptr && type_value->is_array() && !type_value->empty()) {
    std::string joined;
    for (const Value& entry : type_value->items()) {
      if (!joined.empty()) {
        joined += " | ";
      }
      joined += entry.is_string() ? entry.str() : entry.dump();
    }
    return joined;
  }
  if (spec.find("oneOf") != nullptr && Truthy(spec.find("oneOf"))) {
    throw RenderError{"oneOf tool parameters are not supported yet"};
  }
  if (type == "string") {
    const Value* values = spec.find("enum");
    if (!Truthy(values) || !values->is_array()) {
      return "string" + NullableSuffix(spec);
    }
    std::string rendered = "\"";
    for (std::size_t i = 0; i < values->items().size(); ++i) {
      const Value& entry = values->items()[i];
      rendered += i == 0 ? "" : "\" | \"";
      rendered += entry.is_string() ? entry.str() : entry.dump();
    }
    return rendered + "\"";
  }
  if (type == "number" || type == "integer") {
    return "number";
  }
  if (type == "boolean") {
    return "boolean";
  }
  if (type == "object") {
    const Value* properties = spec.find("properties");
    if (!Truthy(properties) || !properties->is_object()) {
      return "object";
    }
    const Value* required = spec.find("required");
    std::string rendered = "{\n";
    const auto& members = properties->members();
    for (std::size_t i = 0; i < members.size(); ++i) {
      rendered += members[i].first;
      rendered += Contains(required, members[i].first) ? ": " : "?: ";
      rendered += kNestedTypeIndent;
      rendered += TypeScriptType(members[i].second);
      rendered += i + 1 < members.size() ? ", " : "";
    }
    return rendered + "}";
  }
  return "any";
}

void AppendToolNamespace(std::string& out, std::span<const HarmonyTool> tools) {
  out += "## ";
  out += kToolsNamespace;
  out += "\n\nnamespace ";
  out += kToolsNamespace;
  out += " {\n\n";
  for (const HarmonyTool& tool : tools) {
    out += "// " + tool.description + "\n";
    out += "type " + tool.name + " = ";
    const Value parameters = tool.parameters_json.empty()
                                 ? Value::object()
                                 : json::parse(tool.parameters_json);
    const Value* properties = parameters.find("properties");
    if (!Truthy(properties) || !properties->is_object()) {
      out += "() => any;\n\n";
      continue;
    }
    const Value* required = parameters.find("required");
    out += "(_: {\n";
    for (const auto& [name, spec] : properties->members()) {
      const Value* description = spec.find("description");
      if (Truthy(description)) {
        out += "// " + description->str() + "\n";
      }
      out += name;
      out += Contains(required, name) ? ": " : "?: ";
      out += TypeScriptType(spec);
      if (const Value* fallback = spec.find("default"); fallback != nullptr) {
        // Enum defaults are concatenated as text; others go through tojson.
        out += ", // default: ";
        out += Truthy(spec.find("enum")) && fallback->is_string()
                   ? fallback->str()
                   : fallback->tojson();
      }
      out += ",\n";
    }
    out += "}) => any;\n\n";
  }
  out += "} // namespace ";
  out += kToolsNamespace;
}

void AppendMessage(std::string& out, std::string_view header,
                   std::string_view body, std::string_view terminator) {
  out += kStart;
  out += header;
  out += kMessage;
  out += body;
  out += terminator;
}

std::string AnalysisHeader() {
  std::string header = "assistant";
  header += kChannel;
  header += "analysis";
  return header;
}

std::string FinalHeader() {
  std::string header = "assistant";
  header += kChannel;
  header += "final";
  return header;
}

bool HasChannelTags(std::string_view text) {
  const std::string analysis =
      std::string(kChannel) + "analysis" + std::string(kMessage);
  const std::string final_tag =
      std::string(kChannel) + "final" + std::string(kMessage);
  return text.find(analysis) != std::string_view::npos ||
         text.find(final_tag) != std::string_view::npos;
}

std::string Render(std::span<const HarmonyMessage> messages,
                   std::span<const HarmonyTool> tools,
                   const HarmonyRenderOptions& options) {
  std::string system = options.model_identity + "\n";
  system += "Knowledge cutoff: 2024-06\n";
  system += "Current date: " + options.current_date + "\n\n";
  system += "Reasoning: " + std::string(EffortName(options.effort)) + "\n\n";
  system +=
      "# Valid channels: analysis, commentary, final. Channel must be "
      "included for every message.";
  if (!tools.empty()) {
    system +=
        "\nCalls to these tools must go to the commentary channel: "
        "'functions'.";
  }
  std::string out;
  AppendMessage(out, "system", system, kEnd);

  std::size_t first = 0;
  std::string developer;
  if (!messages.empty() && (messages.front().role == "developer" ||
                            messages.front().role == "system")) {
    developer = messages.front().content;
    first = 1;
  }
  if (!developer.empty() || !tools.empty()) {
    std::string body;
    if (!developer.empty()) {
      body += "# Instructions\n\n" + developer + "\n\n";
    }
    if (!tools.empty()) {
      body += "# Tools\n\n";
      AppendToolNamespace(body, tools);
    }
    AppendMessage(out, "developer", body, kEnd);
  }

  std::string last_tool_call;
  for (std::size_t i = first; i < messages.size(); ++i) {
    const HarmonyMessage& message = messages[i];
    if (message.role == "assistant") {
      if (HasChannelTags(message.content) || HasChannelTags(message.thinking)) {
        throw RenderError{"assistant text contains Harmony channel tags"};
      }
      if (!message.tool_calls.empty()) {
        const bool future_final = std::any_of(
            messages.begin() + static_cast<std::ptrdiff_t>(i) + 1,
            messages.end(), [](const HarmonyMessage& later) {
              return later.role == "assistant" && later.tool_calls.empty();
            });
        if (!message.content.empty() && !message.thinking.empty()) {
          throw RenderError{
              "a tool-call message cannot carry both content and thinking"};
        }
        const std::string& analysis =
            !message.content.empty() ? message.content : message.thinking;
        if (!analysis.empty() && !future_final) {
          AppendMessage(out, AnalysisHeader(), analysis, kEnd);
        }
        const HarmonyToolCall& call = message.tool_calls.front();
        std::string header = "assistant to=";
        header += kToolsNamespace;
        header += "." + call.name;
        header += kChannel;
        header += "commentary json";
        AppendMessage(out, header, json::parse(call.arguments).tojson(), kCall);
        last_tool_call = call.name;
      } else if (i + 1 == messages.size() && !options.add_generation_prompt) {
        if (!message.thinking.empty()) {
          AppendMessage(out, AnalysisHeader(), message.thinking, kEnd);
        }
        AppendMessage(out, FinalHeader(), message.content, kReturn);
      } else {
        AppendMessage(out, FinalHeader(), message.content, kEnd);
        last_tool_call.clear();
      }
    } else if (message.role == "tool") {
      if (last_tool_call.empty()) {
        throw RenderError{"tool message without a preceding tool call"};
      }
      std::string header(kToolsNamespace);
      header += "." + last_tool_call + " to=assistant";
      header += kChannel;
      header += "commentary";
      AppendMessage(out, header, Value(message.content).tojson(), kEnd);
    } else if (message.role == "user") {
      AppendMessage(out, "user", message.content, kEnd);
    }
  }
  if (options.add_generation_prompt) {
    out += kStart;
    out += "assistant";
  }
  return out;
}

// Splits one message header after `<|start|>assistant` (or the generation
// prompt) into channel, recipient and constraint.
struct Header {
  std::string channel;
  std::string recipient;
  bool valid{true};
};

std::string_view TakeWord(std::string_view& text) {
  const std::size_t end = std::min(text.find(' '), text.find('<'));
  const std::string_view word = text.substr(0, end);
  text.remove_prefix(word.size());
  return word;
}

bool KnownChannel(std::string_view channel) {
  return channel == "analysis" || channel == "commentary" || channel == "final";
}

// `name` or `namespace.name[.more]`: non-empty segments of [A-Za-z0-9_-].
bool ValidRecipient(std::string_view recipient) {
  if (recipient.empty() || recipient == kToolsNamespace) {
    return false;  // no recipient, or a function namespace without a name
  }
  std::size_t segment = 0;
  for (const char c : recipient) {
    if (c == '.') {
      if (segment == 0) {
        return false;
      }
      segment = 0;
    } else if (std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' ||
               c == '-') {
      ++segment;
    } else {
      return false;
    }
  }
  return segment != 0;
}

Header ParseHeader(std::string_view text) {
  constexpr std::string_view kRecipient = " to=";
  Header header;
  while (!text.empty()) {
    if (text.starts_with(kRecipient)) {
      text.remove_prefix(kRecipient.size());
      const std::string_view recipient = TakeWord(text);
      if (!ValidRecipient(recipient)) {
        header.valid = false;
        return header;
      }
      // A stray ` to=assistant` addresses the user-facing stream.
      if (recipient != "assistant") {
        header.recipient = recipient;
      }
    } else if (text.starts_with(kChannel)) {
      text.remove_prefix(kChannel.size());
      // Later channels win: gpt-oss sometimes prefixes a stray commentary.
      header.channel = TakeWord(text);
      if (!KnownChannel(header.channel)) {
        header.valid = false;
        return header;
      }
    } else if (text.starts_with(' ') || text.starts_with(kConstrain)) {
      text.remove_prefix(text.starts_with(' ') ? 1 : kConstrain.size());
      if (text.starts_with(kConstrain)) {
        text.remove_prefix(kConstrain.size());
      }
      if (TakeWord(text).empty()) {
        header.valid = false;
        return header;
      }
    } else {
      header.valid = false;
      return header;
    }
  }
  // Calls go to the commentary channel, or analysis for built-in tools.
  header.valid = !header.channel.empty() &&
                 (header.recipient.empty() || header.channel != "final");
  return header;
}

}  // namespace

std::optional<std::string> RenderHarmony(
    std::span<const HarmonyMessage> messages,
    std::span<const HarmonyTool> tools, const HarmonyRenderOptions& options,
    std::string* error) {
  try {
    return Render(messages, tools, options);
  } catch (const RenderError& failure) {
    if (error != nullptr) {
      *error = failure.message;
    }
  } catch (const std::exception& failure) {
    if (error != nullptr) {
      *error = std::string("invalid JSON: ") + failure.what();
    }
  }
  return std::nullopt;
}

HarmonyCompletion ParseHarmonyCompletion(std::string_view text) {
  constexpr std::string_view kAssistant = "assistant";
  constexpr std::array<std::string_view, 3> kTerminators{kEnd, kCall, kReturn};
  HarmonyCompletion result;
  std::size_t pos = 0;
  for (bool first = true; pos < text.size(); first = false) {
    if (!first) {
      if (!text.substr(pos).starts_with(kStart)) {
        result.valid = false;
        return result;
      }
      pos += kStart.size();
      if (!text.substr(pos).starts_with(kAssistant)) {
        result.valid = false;
        return result;
      }
      pos += kAssistant.size();
    }
    const std::size_t message = text.find(kMessage, pos);
    if (message == std::string_view::npos) {
      result.complete = false;
      return result;  // the header is still being generated
    }
    const Header header = ParseHeader(text.substr(pos, message - pos));
    if (!header.valid) {
      result.valid = false;
      return result;
    }
    const std::size_t body_start = message + kMessage.size();
    std::size_t body_end = text.size();
    std::string_view terminator;
    for (const std::string_view candidate : kTerminators) {
      const std::size_t found = text.find(candidate, body_start);
      if (found < body_end) {
        body_end = found;
        terminator = candidate;
      }
    }
    const std::string_view body =
        text.substr(body_start, body_end - body_start);
    const std::string prefix = std::string(kToolsNamespace) + ".";
    if (header.recipient.starts_with(prefix)) {
      result.tool_calls.push_back(
          {header.recipient.substr(prefix.size()), std::string(body)});
    } else if (!header.recipient.empty()) {
      ++result.dropped_messages;
    } else if (header.channel == "analysis") {
      result.reasoning += body;
    } else if (header.channel == "final" || header.channel == "commentary") {
      result.content += body;
    } else {
      result.valid = false;
      return result;
    }
    result.complete = terminator == kCall || terminator == kReturn;
    if (terminator.empty() || result.complete) {
      return result;
    }
    pos = body_end + terminator.size();
  }
  return result;
}

}  // namespace gufo::models::gpt_oss
