#include "src/models/gemma4/tool_syntax.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>

namespace gufo::models::gemma4 {
namespace {

constexpr std::string_view kQuote = "<|\"|>";
constexpr std::size_t kMaxDepth = 64;
constexpr std::int64_t kMaxExactInteger = std::int64_t{1} << 53;

struct Parser {
  std::string_view text;
  std::size_t pos{0};
  std::string error;

  bool Fail(std::string message) {
    if (error.empty()) {
      error = std::move(message) + " at offset " + std::to_string(pos);
    }
    return false;
  }

  void SkipSpace() {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
      ++pos;
    }
  }

  bool Consume(std::string_view token) {
    if (text.substr(pos, token.size()) == token) {
      pos += token.size();
      return true;
    }
    return false;
  }

  bool QuotedString(std::string* out) {
    if (!Consume(kQuote)) {
      return Fail("expected string delimiter");
    }
    const std::size_t end = text.find(kQuote, pos);
    if (end == std::string_view::npos) {
      return Fail("unterminated string");
    }
    *out = std::string(text.substr(pos, end - pos));
    pos = end + kQuote.size();
    return true;
  }

  bool Key(std::string* out) {
    SkipSpace();
    if (text.substr(pos, kQuote.size()) == kQuote) {
      return QuotedString(out);
    }
    const std::size_t colon = text.find(':', pos);
    if (colon == std::string_view::npos) {
      return Fail("expected ':' after key");
    }
    std::string_view key = text.substr(pos, colon - pos);
    while (!key.empty() &&
           std::isspace(static_cast<unsigned char>(key.back())) != 0) {
      key.remove_suffix(1);
    }
    if (key.empty() || key.find_first_of("{}[],") != std::string_view::npos) {
      return Fail("invalid key");
    }
    *out = std::string(key);
    pos = colon;
    return true;
  }

  /// A bare scalar ends at the next structural delimiter. False when it is
  /// a number gufo::json cannot hold: an integer literal beyond 2^53 (checked
  /// on the digits, before any conversion, so overflow is caught too) or a
  /// numeric literal outside the double range. Other text is a string.
  bool Bare(json::Value* out) {
    const std::size_t start = pos;
    while (pos < text.size() && text[pos] != ',' && text[pos] != '}' &&
           text[pos] != ']') {
      ++pos;
    }
    std::string_view raw = text.substr(start, pos - start);
    while (!raw.empty() &&
           std::isspace(static_cast<unsigned char>(raw.back())) != 0) {
      raw.remove_suffix(1);
    }
    if (raw == "true" || raw == "false") {
      *out = json::Value(raw == "true");
      return true;
    }
    if (raw == "null") {
      *out = json::Value(nullptr);
      return true;
    }
    const char* first = raw.data();
    const char* last = raw.data() + raw.size();
    const std::string_view digits = raw.starts_with('-') ? raw.substr(1) : raw;
    if (!digits.empty() &&
        digits.find_first_not_of("0123456789") == std::string_view::npos) {
      // A plain integer literal: exact up to 2^53, rejected beyond it.
      std::int64_t integer = 0;
      const auto [end, ec] = std::from_chars(first, last, integer);
      if (ec != std::errc{} || end != last || integer > kMaxExactInteger ||
          integer < -kMaxExactInteger) {
        pos = start;
        return Fail("integer " + std::string(raw) +
                    " exceeds 2^53 and cannot be held exactly");
      }
      // A negative zero keeps its sign, as a double conversion would.
      *out = json::Value(integer == 0 && raw.starts_with('-')
                             ? -0.0
                             : static_cast<double>(integer));
      return true;
    }
    double number = 0.0;
    const auto [end, ec] = std::from_chars(first, last, number);
    if (end == last && !raw.empty()) {
      if (ec == std::errc{} && std::isfinite(number)) {
        *out = json::Value(number);
        return true;
      }
      if (ec == std::errc::result_out_of_range) {
        pos = start;
        return Fail("number " + std::string(raw) + " is out of range");
      }
    }
    *out = json::Value(std::string(raw));
    return true;
  }

  bool Value(json::Value* out, std::size_t depth) {
    if (depth > kMaxDepth) {
      return Fail("arguments nest too deeply");
    }
    SkipSpace();
    if (pos >= text.size()) {
      return Fail("expected a value");
    }
    if (text.substr(pos, kQuote.size()) == kQuote) {
      std::string s;
      if (!QuotedString(&s)) {
        return false;
      }
      *out = json::Value(std::move(s));
      return true;
    }
    if (text[pos] == '{') {
      return Object(out, depth + 1);
    }
    if (text[pos] == '[') {
      ++pos;
      *out = json::Value::array();
      SkipSpace();
      if (Consume("]")) {
        return true;
      }
      while (true) {
        json::Value item;
        if (!Value(&item, depth + 1)) {
          return false;
        }
        out->push_back(std::move(item));
        SkipSpace();
        if (Consume("]")) {
          return true;
        }
        if (!Consume(",")) {
          return Fail("expected ',' or ']'");
        }
      }
    }
    return Bare(out);
  }

  bool Object(json::Value* out, std::size_t depth) {
    if (!Consume("{")) {
      return Fail("expected '{'");
    }
    *out = json::Value::object();
    SkipSpace();
    if (Consume("}")) {
      return true;
    }
    // Keys seen in this object: a set keeps wide objects linear.
    std::unordered_set<std::string> seen;
    while (true) {
      std::string key;
      if (!Key(&key)) {
        return false;
      }
      SkipSpace();
      if (!Consume(":")) {
        return Fail("expected ':'");
      }
      if (!seen.insert(key).second) {
        return Fail("duplicate key " + key);
      }
      json::Value value;
      if (!Value(&value, depth)) {
        return false;
      }
      out->append_member(std::move(key), std::move(value));
      SkipSpace();
      if (Consume("}")) {
        return true;
      }
      if (!Consume(",")) {
        return Fail("expected ',' or '}'");
      }
    }
  }
};

}  // namespace

std::optional<ToolCall> ParseToolCall(std::string_view body,
                                      std::string* error_msg) {
  Parser p{body};
  p.SkipSpace();
  ToolCall call;
  bool ok = p.Consume("call:");
  if (!ok) {
    p.Fail("expected 'call:'");
  } else {
    const std::size_t brace = body.find('{', p.pos);
    if (brace == std::string_view::npos || brace == p.pos) {
      ok = p.Fail("expected a function name followed by '{'");
    } else {
      call.name = std::string(body.substr(p.pos, brace - p.pos));
      p.pos = brace;
      ok = p.Object(&call.arguments, 0);
      p.SkipSpace();
      if (ok && p.pos != body.size()) {
        ok = p.Fail("unexpected text after arguments");
      }
    }
  }
  if (!ok) {
    if (error_msg != nullptr) {
      *error_msg = p.error;
    }
    return std::nullopt;
  }
  return call;
}

std::optional<PartialToolCall> ParsePartialToolCall(std::string_view body) {
  Parser p{body};
  p.SkipSpace();
  if (!p.Consume("call:")) {
    return std::nullopt;
  }
  const std::size_t brace = body.find('{', p.pos);
  if (brace == std::string_view::npos || brace == p.pos) {
    return std::nullopt;
  }
  PartialToolCall call{.name = std::string(body.substr(p.pos, brace - p.pos)),
                       .arguments = json::Value::object()};
  p.pos = brace + 1;
  std::unordered_set<std::string> seen;
  while (true) {
    p.SkipSpace();
    std::string key;
    if (p.pos >= body.size() || body[p.pos] == '}' || !p.Key(&key)) {
      return call;
    }
    p.SkipSpace();
    if (!p.Consume(":") || seen.contains(key)) {
      return call;
    }
    p.SkipSpace();
    const std::string_view rest = body.substr(p.pos);
    if (rest.starts_with(kQuote) &&
        rest.find(kQuote, kQuote.size()) == std::string_view::npos) {
      std::string_view text = rest.substr(kQuote.size());
      for (std::size_t size = std::min(text.size(), kQuote.size() - 1);
           size > 0; --size) {
        if (kQuote.starts_with(text.substr(text.size() - size))) {
          text.remove_suffix(size);
          break;
        }
      }
      call.open_string.emplace(std::move(key), std::string(text));
      return call;
    }
    json::Value value;
    if (!p.Value(&value, 0)) {
      return call;
    }
    // A bare scalar may still grow; any value ends at its delimiter.
    p.SkipSpace();
    if (p.pos >= body.size() || (body[p.pos] != ',' && body[p.pos] != '}')) {
      return call;
    }
    seen.insert(key);
    call.arguments.append_member(std::move(key), std::move(value));
    if (!p.Consume(",")) {
      return call;
    }
  }
}

}  // namespace gufo::models::gemma4
