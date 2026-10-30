// Contract of the compiled Gemma 4 chat template.
// fixtures/template_goldens.json holds the prompts rendered by Jinja2
// from the artifact's own template (tools/gemma4/template_goldens.py); every
// case must render byte-identically. With GUFO_GEMMA4_MODEL set, the
// artifact's template must also pass the qualification check.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/gemma4/chat_template.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/gguf_builder.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;
using gufo::json::Value;
using gufo::tokenization::ChatMessage;
using gufo::tokenization::ChatRole;

namespace {

Value ReadFixture(const std::string& name) {
  std::ifstream file(std::string(GUFO_GEMMA4_FIXTURES) + "/" + name);
  Require(file.good(), "missing fixture " + name);
  std::stringstream text;
  text << file.rdbuf();
  return gufo::json::parse(text.str());
}

ChatRole Role(const std::string& role) {
  if (role == "system") {
    return ChatRole::kSystem;
  }
  if (role == "developer") {
    return ChatRole::kDeveloper;
  }
  if (role == "assistant") {
    return ChatRole::kAssistant;
  }
  if (role == "tool") {
    return ChatRole::kTool;
  }
  Require(role == "user", "unexpected role " + role);
  return ChatRole::kUser;
}

std::vector<ChatMessage> Messages(const Value& messages) {
  std::vector<ChatMessage> out;
  for (const auto& m : messages.items()) {
    ChatMessage& message = out.emplace_back(
        Role(m.member_str("role")), m.member_str("content"),
        m.member_str("name"), m.member_str("reasoning_content"));
    message.tool_call_id = m.member_str("tool_call_id");
    const Value* calls = m.find("tool_calls");
    if (calls == nullptr) {
      continue;
    }
    for (const auto& c : calls->items()) {
      ChatMessage::ToolCall& call = message.tool_calls.emplace_back();
      call.id = c.member_str("id");
      call.name = c.member_str("name");
      for (const auto& [key, value] : c.find("arguments")->members()) {
        call.arguments.push_back(
            {key, value.is_string() ? value.str() : value.dump(),
             value.is_string()});
      }
    }
  }
  return out;
}

std::vector<gufo::tokenization::ChatTool> Tools(const Value& definitions,
                                                const Value* names) {
  std::vector<gufo::tokenization::ChatTool> out;
  if (names == nullptr) {
    return out;
  }
  for (const auto& name : names->items()) {
    const Value* tool = definitions.find(name.str());
    Require(tool != nullptr, "unknown tool " + name.str());
    out.push_back(
        {tool->find("function")->member_str("name"), "", "{}", tool->dump()});
  }
  return out;
}

void CheckGoldens() {
  const Value cases = ReadFixture("template_cases.json");
  const Value goldens = ReadFixture("template_goldens.json");
  const Value* template_info = goldens.find("template");
  Require(
      template_info != nullptr && template_info->member_str("sha256") ==
                                      g4::ChatTemplate::UnslothTemplateSha256(),
      "goldens were rendered from another template");
  const Value* rendered = goldens.find("cases");
  std::size_t count = 0;
  for (const auto& [name, c] : cases.find("cases")->members()) {
    const Value* expected = rendered->find(name);
    Require(expected != nullptr, "no golden for " + name);
    g4::ChatOptions options;
    if (const Value* v = c.find("add_generation_prompt")) {
      options.add_generation_prompt = v->as_bool();
    }
    if (const Value* v = c.find("enable_thinking")) {
      options.enable_thinking = v->as_bool();
    }
    if (const Value* v = c.find("preserve_thinking")) {
      options.preserve_thinking = v->as_bool();
    }
    const auto messages = Messages(*c.find("messages"));
    const auto tools = Tools(*cases.find("tools"), c.find("tools"));
    std::string error;
    const auto prompt =
        g4::ChatTemplate::Render(messages, tools, options, &error);
    Require(prompt.has_value(), name + ": " + error);
    Require(prompt->text == expected->member_str("text"),
            name + ": rendered\n" + prompt->text + "\nexpected\n" +
                expected->member_str("text"));
    ++count;
  }
  Require(count >= 40, "golden cases");
}

void CheckOffsets() {
  const std::vector<ChatMessage> messages = {{ChatRole::kUser, "first"},
                                             {ChatRole::kAssistant, "ok"},
                                             {ChatRole::kUser, "context"},
                                             {ChatRole::kUser, "question"}};
  const auto prompt = g4::ChatTemplate::Render(messages, {}, {});
  Require(prompt.has_value(), "render");
  const std::string& text = prompt->text;
  Require(text.substr(prompt->generation_prompt_offset) ==
              "<|turn>model\n<|channel>thought\n<channel|>",
          "generation prompt offset");
  Require(text.substr(prompt->stable_prefix_offset)
                  .starts_with("<|turn>user\nquestion") &&
              text.substr(0, prompt->stable_prefix_offset)
                  .ends_with("context<turn|>\n"),
          "per-turn context boundary precedes the final user turn");
  const std::vector<ChatMessage> chat = {{ChatRole::kUser, "a"},
                                         {ChatRole::kAssistant, "b"},
                                         {ChatRole::kUser, "c"}};
  const auto plain = g4::ChatTemplate::Render(chat, {}, {});
  Require(plain->stable_prefix_offset == plain->generation_prompt_offset,
          "ordinary chat is stable up to the generation prompt");
}

void CheckRejections() {
  std::string error;
  std::vector<ChatMessage> messages = {{ChatRole::kUser, "hi"}};
  messages[0].images.push_back({});
  Require(!g4::ChatTemplate::Render(messages, {}, {}, &error) && !error.empty(),
          "image accepted");
  messages[0].images.clear();
  // "call:NAME{" and "declaration:NAME{" end a name at its first brace.
  gufo::tokenization::ChatTool tool{"f{x", "", "{}", R"({"function":{}})"};
  Require(!g4::ChatTemplate::Render(messages, {&tool, 1}, {}, &error),
          "tool name with a brace accepted");
  tool = {"f", "", "{}", "[1]"};
  Require(!g4::ChatTemplate::Render(messages, {&tool, 1}, {}, &error),
          "non-object tool definition accepted");
  tool.definition_json = "{";
  Require(!g4::ChatTemplate::Render(messages, {&tool, 1}, {}, &error),
          "malformed tool definition accepted");
  messages.push_back({ChatRole::kAssistant, ""});
  messages.back().tool_calls.push_back({"id", "g{", {}});
  Require(!g4::ChatTemplate::Render(messages, {}, {}, &error),
          "call name with a brace accepted");
}

void CheckOptions() {
  gufo::ReasoningOptions reasoning;
  Require(!g4::ResolveChatOptions(reasoning).enable_thinking,
          "thinking is off by default");
  reasoning.effort = gufo::ReasoningEffort::kMinimal;
  Require(!g4::ResolveChatOptions(reasoning).enable_thinking,
          "minimal effort keeps thinking off");
  reasoning.effort = gufo::ReasoningEffort::kHigh;
  Require(g4::ResolveChatOptions(reasoning).enable_thinking,
          "an effort level enables thinking");
  reasoning.enabled = false;
  Require(!g4::ResolveChatOptions(reasoning).enable_thinking,
          "explicit enable wins over effort");
  reasoning.preserve_thinking = true;
  Require(g4::ResolveChatOptions(reasoning).preserve_thinking,
          "preserve thinking");
}

void CheckQualification() {
  gemma4_test::GgufImage other(
      {{"tokenizer.chat_template", std::string("{{ messages }}")}});
  std::string error;
  Require(!g4::ChatTemplate::ValidateGgufTemplate(*other.reader(), &error) &&
              error.find("unqualified") != std::string::npos,
          "another template accepted");
  gemma4_test::GgufImage none({{"general.architecture", std::string("x")}});
  Require(!g4::ChatTemplate::ValidateGgufTemplate(*none.reader(), &error),
          "missing template accepted");
  const char* model = std::getenv("GUFO_GEMMA4_MODEL");
  if (model == nullptr || !std::filesystem::exists(model)) {
    std::cout << "artifact template check skipped (GUFO_GEMMA4_MODEL)\n";
    return;
  }
  const auto gguf = gufo::core::GgufReader::OpenFile(model, &error);
  Require(gguf != nullptr, error);
  Require(g4::ChatTemplate::ValidateGgufTemplate(*gguf, &error),
          "artifact template rejected: " + error);
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckGoldens();
    CheckOffsets();
    CheckRejections();
    CheckOptions();
    CheckQualification();
  });
}
