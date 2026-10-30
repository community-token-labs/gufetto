// Harmony rendering against Jinja2 goldens of the pinned template, and
// completion parsing against hand-written model outputs. Expected values are
// literals, never derived from the control-token constants.

#include "src/models/gpt_oss/harmony.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/gpt_oss/control_tokens.hpp"

namespace {

namespace gpt_oss = gufo::models::gpt_oss;
namespace json = gufo::json;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

json::Value LoadGoldens() {
  std::ifstream file(GUFO_GPT_OSS_HARMONY_GOLDENS, std::ios::binary);
  Expect(file.good(), "golden fixture opens");
  std::stringstream text;
  text << file.rdbuf();
  return json::parse(text.str());
}

std::string Member(const json::Value& object, const std::string& key) {
  const json::Value* value = object.find(key);
  return value != nullptr ? value->str() : std::string{};
}

std::vector<gpt_oss::HarmonyMessage> Messages(const json::Value& input) {
  std::vector<gpt_oss::HarmonyMessage> messages;
  for (const json::Value& entry : input.find("messages")->items()) {
    gpt_oss::HarmonyMessage message{Member(entry, "role"),
                                    Member(entry, "content"),
                                    Member(entry, "thinking"),
                                    {}};
    if (const json::Value* calls = entry.find("tool_calls")) {
      for (const json::Value& call : calls->items()) {
        const json::Value& function = *call.find("function");
        // Compact on input: the renderer must restore tojson spacing.
        message.tool_calls.push_back(
            {Member(function, "name"), function.find("arguments")->dump()});
      }
    }
    messages.push_back(std::move(message));
  }
  return messages;
}

std::vector<gpt_oss::HarmonyTool> Tools(const json::Value& input) {
  std::vector<gpt_oss::HarmonyTool> tools;
  if (const json::Value* list = input.find("tools")) {
    for (const json::Value& entry : list->items()) {
      const json::Value& function = *entry.find("function");
      const json::Value* parameters = function.find("parameters");
      tools.push_back({Member(function, "name"),
                       Member(function, "description"),
                       parameters != nullptr ? parameters->dump() : ""});
    }
  }
  return tools;
}

gpt_oss::HarmonyRenderOptions Options(const json::Value& input,
                                      const std::string& date) {
  gpt_oss::HarmonyRenderOptions options;
  options.current_date = date;
  if (const json::Value* identity = input.find("model_identity")) {
    options.model_identity = identity->str();
  }
  const std::string effort = Member(input, "reasoning_effort");
  if (effort == "low") {
    options.effort = gpt_oss::HarmonyEffort::kLow;
  } else if (effort == "high") {
    options.effort = gpt_oss::HarmonyEffort::kHigh;
  }
  if (const json::Value* prompt = input.find("add_generation_prompt")) {
    options.add_generation_prompt = prompt->as_bool();
  }
  return options;
}

void TestRenderGoldens(const json::Value& goldens) {
  Expect(Member(goldens, "template_sha256") == gpt_oss::HarmonyTemplateSha256(),
         "goldens come from the compiled template");
  const std::string date = Member(goldens, "date");
  std::size_t cases = 0;
  for (const auto& [name, golden] : goldens.find("cases")->members()) {
    const json::Value& input = *golden.find("input");
    const auto messages = Messages(input);
    const auto tools = Tools(input);
    std::string error;
    const auto rendered =
        gpt_oss::RenderHarmony(messages, tools, Options(input, date), &error);
    Expect(rendered.has_value(), name + " renders: " + error);
    if (*rendered != Member(golden, "rendered")) {
      std::cerr << name << " rendered:\n"
                << *rendered << "\nexpected:\n"
                << Member(golden, "rendered") << '\n';
      std::exit(1);
    }
    ++cases;
  }
  Expect(cases == 8, "every golden case ran");
}

void TestRenderErrors() {
  const gpt_oss::HarmonyRenderOptions options;
  std::string error;
  const std::vector<gpt_oss::HarmonyMessage> orphan_tool{
      {"user", "hi", "", {}}, {"tool", "result", "", {}}};
  Expect(!gpt_oss::RenderHarmony(orphan_tool, {}, options, &error),
         "tool result without a call is rejected");
  const std::vector<gpt_oss::HarmonyMessage> tagged{
      {"assistant", "<|channel|>final<|message|>x", "", {}}};
  Expect(!gpt_oss::RenderHarmony(tagged, {}, options, &error),
         "channel tags in content are rejected");
  const std::vector<gpt_oss::HarmonyMessage> bad_arguments{
      {"assistant", "", "", {{"f", "{not json"}}}};
  Expect(!gpt_oss::RenderHarmony(bad_arguments, {}, options, &error),
         "invalid tool arguments are rejected");
  const std::vector<gpt_oss::HarmonyTool> one_of{
      {"f", "d", R"({"properties":{"a":{"oneOf":[{"type":"string"}]}}})"}};
  Expect(!gpt_oss::RenderHarmony({}, one_of, options, &error) &&
             error.find("oneOf") != std::string::npos,
         "oneOf schemas are rejected until supported");
}

void TestNestedSchema() {
  // Jinja2 output of the pinned template for a nested object parameter,
  // including the indentation the template leaks into nested types.
  const std::vector<gpt_oss::HarmonyTool> tools{
      {"f", "d",
       R"({"type":"object","properties":{"o":{"type":"object","properties":)"
       R"({"a":{"type":"string"},"b":{"type":"array","items":{"type":"object",)"
       R"("properties":{"x":{"type":"number"}}}}},"required":["a"]}}})"}};
  const std::vector<gpt_oss::HarmonyMessage> messages{{"user", "x", "", {}}};
  gpt_oss::HarmonyRenderOptions options;
  options.current_date = "2026-10-30";
  const auto rendered = gpt_oss::RenderHarmony(messages, tools, options);
  Expect(rendered.has_value(), "nested schema renders");
  constexpr std::string_view kExpected =
      "type f = (_: {\no?: {\na: \n                string, b?: \n"
      "                {\nx?: \n                number}[]},\n}) => any;\n\n"
      "} // namespace functions<|end|>";
  Expect(rendered->find(kExpected) != std::string::npos,
         "nested object and array types match the template");
}

void TestControlTokens() {
  struct Literal {
    std::string_view text;
    std::int32_t id;
  };
  // From tokenizer.ggml.tokens of the pinned GGUF.
  constexpr std::array<Literal, 10> kPinned{{
      {"<|startoftext|>", 199998},
      {"<|endoftext|>", 199999},
      {"<|return|>", 200002},
      {"<|constrain|>", 200003},
      {"<|channel|>", 200005},
      {"<|start|>", 200006},
      {"<|end|>", 200007},
      {"<|message|>", 200008},
      {"<|call|>", 200012},
      {"<|endofprompt|>", 200018},
  }};
  Expect(gpt_oss::kControlTokenIds.size() == kPinned.size(), "catalogue size");
  for (std::size_t i = 0; i < kPinned.size(); ++i) {
    Expect(gpt_oss::kControlTokenIds[i].text == kPinned[i].text &&
               gpt_oss::kControlTokenIds[i].id == kPinned[i].id,
           "control token matches the pinned vocabulary");
  }
}

void ExpectCall(const gpt_oss::HarmonyCompletion& parsed, std::size_t index,
                std::string_view name, std::string_view arguments) {
  Expect(parsed.tool_calls.size() > index, "tool call present");
  Expect(parsed.tool_calls[index].name == name, "tool call name");
  Expect(parsed.tool_calls[index].arguments == arguments, "tool arguments");
}

void TestParseCompletions(const json::Value& goldens) {
  {
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>analysis<|message|>User says hi.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello!<|return|>");
    Expect(parsed.valid && parsed.complete, "final answer is complete");
    Expect(parsed.reasoning == "User says hi.", "analysis becomes reasoning");
    Expect(parsed.content == "Hello!", "final becomes content");
    Expect(parsed.tool_calls.empty(), "no tool calls");
  }
  {
    // Recipient in the role header, constraint after a space.
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>analysis<|message|>Need weather.<|end|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary "
        "<|constrain|>json<|message|>{\"location\":\"Paris\"}<|call|>");
    Expect(parsed.valid && parsed.complete, "role-header call is complete");
    Expect(parsed.reasoning == "Need weather.", "analysis before the call");
    ExpectCall(parsed, 0, "get_weather", "{\"location\":\"Paris\"}");
  }
  {
    // Recipient in the channel header.
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>commentary to=functions.get_weather <|constrain|>json"
        "<|message|>{}<|call|>");
    Expect(parsed.valid && parsed.complete, "channel-header call");
    ExpectCall(parsed, 0, "get_weather", "{}");
    Expect(parsed.content.empty(), "the call is not content");
  }
  {
    // Commentary preamble before a call.
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>commentary<|message|>Checking.<|end|>"
        "<|start|>assistant to=functions.ping<|channel|>commentary json"
        "<|message|>{}<|call|>");
    Expect(parsed.valid && parsed.complete, "preamble then call");
    Expect(parsed.content == "Checking.", "preamble becomes content");
    ExpectCall(parsed, 0, "ping", "{}");
  }
  {
    // Stray commentary prefixes, as gpt-oss-20b sometimes writes them; the
    // stop token was stripped, so the final message has no terminator.
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>commentary to=assistant<|channel|>analysis<|message|>a"
        "<|end|><|start|>assistant<|channel|>analysis<|message|>b<|end|>"
        "<|start|>assistant<|channel|>commentary<|channel|>final"
        "<|message|>done");
    Expect(parsed.valid && !parsed.complete, "stray prefixes are accepted");
    Expect(parsed.reasoning == "ab", "analysis blocks concatenate");
    Expect(parsed.content == "done", "final without terminator");
  }
  {
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>analysis to=browser.search<|message|>{\"query\":\"x\"}"
        "<|call|>");
    Expect(parsed.valid && parsed.complete, "built-in call parses");
    Expect(parsed.tool_calls.empty() && parsed.dropped_messages == 1,
           "built-in recipients are dropped");
  }
  {
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>analysis<|message|>abc<|end|><|start|>assistant<|chan");
    Expect(parsed.valid && !parsed.complete, "truncated header is pending");
    Expect(parsed.reasoning == "abc", "earlier messages survive truncation");
  }
  Expect(!gpt_oss::ParseHarmonyCompletion("<|channel|>bogus<|message|>x<|end|>")
              .valid,
         "unknown channel is invalid");
  Expect(!gpt_oss::ParseHarmonyCompletion(
              "<|channel|>analysis<|message|>a<|end|>junk<|start|>assistant")
              .valid,
         "text between messages is invalid");
  Expect(!gpt_oss::ParseHarmonyCompletion("<|start|>user<|message|>x").valid,
         "a missing channel is invalid");
  // Recipients never bypass channel or name validation.
  for (const std::string_view bad : {
           "<|channel|>bogus to=functions.f<|message|>{}<|call|>",
           "<|channel|>commentary to=functions.f<|channel|>bogus"
           "<|message|>{}<|call|>",
           " to=functions.f<|message|>{}<|call|>",
           "<|channel|>commentary to=<|message|>{}<|call|>",
           "<|channel|>commentary to= <|constrain|>json<|message|>{}<|call|>",
           "<|channel|>commentary to=functions.<|message|>{}<|call|>",
           "<|channel|>commentary to=functions<|message|>{}<|call|>",
           "<|channel|>commentary to=.f<|message|>{}<|call|>",
           "<|channel|>commentary to=functions..f<|message|>{}<|call|>",
           "<|channel|>commentary to=functions.f(x)<|message|>{}<|call|>",
           "<|channel|>final to=functions.f<|message|>{}<|call|>",
       }) {
    const auto parsed = gpt_oss::ParseHarmonyCompletion(bad);
    Expect(!parsed.valid && parsed.tool_calls.empty(),
           "malformed recipient or channel is rejected");
  }
  {
    // Stray commentary prefixes stay accepted in front of a call.
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>commentary to=assistant<|channel|>commentary "
        "to=functions.get-weather_2 <|constrain|>json<|message|>{}<|call|>");
    Expect(parsed.valid && parsed.complete, "stray prefix before a call");
    ExpectCall(parsed, 0, "get-weather_2", "{}");
  }
  {
    const auto parsed = gpt_oss::ParseHarmonyCompletion(
        "<|channel|>analysis to=python<|message|>print(1)<|call|>");
    Expect(parsed.valid && parsed.dropped_messages == 1,
           "a namespace-free built-in recipient is dropped, not called");
  }

  // Round trip: the rendered history of a tool call parses back.
  const std::string rendered =
      Member(*goldens.find("cases")->find("tool_round_trip"), "rendered");
  constexpr std::string_view kTurn =
      "Weather in Tokyo?<|end|><|start|>assistant";
  const std::size_t start = rendered.find(kTurn);
  const std::size_t end = rendered.find("<|call|>");
  Expect(start != std::string::npos && end != std::string::npos,
         "round-trip slice found");
  const auto parsed =
      gpt_oss::ParseHarmonyCompletion(std::string_view(rendered).substr(
          start + kTurn.size(), end + 8 - start - kTurn.size()));
  Expect(parsed.valid && parsed.complete, "rendered call parses");
  Expect(parsed.reasoning == "Need the weather tool.", "rendered analysis");
  ExpectCall(parsed, 0, "get_weather",
             "{\"location\": \"Tokyo, \xE6\x9D\xB1\xE4\xBA\xAC\", \"days\": 2, "
             "\"hourly\": false}");
}

}  // namespace

int main() {
  const json::Value goldens = LoadGoldens();
  TestRenderGoldens(goldens);
  TestRenderErrors();
  TestNestedSchema();
  TestControlTokens();
  TestParseCompletions(goldens);
  std::cout << "gpt_oss_harmony_test passed\n";
  return 0;
}
