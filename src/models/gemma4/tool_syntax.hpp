#ifndef GUFO_MODELS_GEMMA4_TOOL_SYNTAX_HPP_
#define GUFO_MODELS_GEMMA4_TOOL_SYNTAX_HPP_

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/json.hpp"

namespace gufo::models::gemma4 {

struct ToolCall {
  std::string name;
  json::Value arguments;  ///< Always a JSON object.
};

/// Parses the body of one `<|tool_call>…<tool_call|>` block:
/// `call:NAME{key:value,…}`. Values use the Gemma 4 argument syntax the chat
/// template renders: `<|"|>`-delimited strings, numbers, true/false/null,
/// `{…}` objects with bare or `<|"|>`-quoted keys, and `[…]` arrays. A bare
/// value that is not a JSON literal is kept as a string.
///
/// Rejected: a key repeated within one object, and an integer literal beyond
/// ±2^53, which gufo::json (doubles) cannot hold exactly. The name is not
/// checked against the declared tools; callers must do that.
[[nodiscard]] std::optional<ToolCall> ParseToolCall(
    std::string_view body, std::string* error_msg = nullptr);

/// The part of an unfinished call body that later bytes cannot change: the
/// top-level members whose value has ended (a delimiter follows it), in
/// order, then the text so far of a string value still open. A trailing
/// prefix of the string delimiter is held back.
struct PartialToolCall {
  std::string name;
  json::Value arguments;  ///< Always a JSON object.
  /// Key and text so far of an open top-level string value.
  std::optional<std::pair<std::string, std::string>> open_string;
};

/// Parses an unfinished call body, such as one still being generated. Empty
/// until `call:NAME{` has arrived. A member the complete parse would reject
/// (malformed, repeated at any depth, or an inexact integer) ends the result
/// before it, so the two parsers agree on every member they both report.
[[nodiscard]] std::optional<PartialToolCall> ParsePartialToolCall(
    std::string_view body);

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_TOOL_SYNTAX_HPP_
