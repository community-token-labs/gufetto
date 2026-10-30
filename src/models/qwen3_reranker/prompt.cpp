#include "src/models/qwen3_reranker/prompt.hpp"

#include "src/models/qwen/control_tokens.hpp"

namespace gufo::models::qwen3_reranker {
namespace {

using tokenization::kImEnd;
using tokenization::kImStart;

constexpr std::string_view kSystem =
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or "
    "\"no\".";

}  // namespace

std::string PromptPrefix() {
  std::string text;
  text.append(kImStart).append("system\n").append(kSystem).append(kImEnd);
  text.append("\n").append(kImStart).append("user\n");
  return text;
}

std::string PromptSuffix() {
  std::string text;
  text.append(kImEnd).append("\n").append(kImStart);
  text.append("assistant\n<think>\n\n</think>\n\n");
  return text;
}

std::string FormatPair(std::string_view instruction, std::string_view query,
                       std::string_view document) {
  std::string text;
  text.append("<Instruct>: ").append(instruction);
  text.append("\n<Query>: ").append(query);
  text.append("\n<Document>: ").append(document);
  return text;
}

bool TokenizeFrame(const Encoder& encode, PromptTokens* tokens,
                   std::string* error) {
  PromptTokens frame;
  frame.prefix = encode(PromptPrefix(), true);
  frame.suffix = encode(PromptSuffix(), true);
  const auto yes = encode("yes", false);
  const auto no = encode("no", false);
  if (yes.size() != 1 || no.size() != 1 || yes[0] == no[0]) {
    if (error != nullptr)
      *error = "the tokenizer must spell \"yes\" and \"no\" as single tokens";
    return false;
  }
  if (frame.prefix.empty() || frame.suffix.empty()) {
    if (error != nullptr)
      *error = "the tokenizer produced an empty prompt frame";
    return false;
  }
  frame.yes = yes[0];
  frame.no = no[0];
  *tokens = std::move(frame);
  return true;
}

bool FramePair(const PromptTokens& frame,
               const std::vector<std::uint32_t>& pair, std::size_t max_length,
               bool truncate, std::vector<std::uint32_t>* tokens) {
  const std::size_t fixed = frame.prefix.size() + frame.suffix.size();
  if (fixed >= max_length)
    return false;
  std::size_t kept = pair.size();
  if (kept > max_length - fixed) {
    if (!truncate)
      return false;
    kept = max_length - fixed;
  }
  tokens->clear();
  tokens->reserve(fixed + kept);
  tokens->insert(tokens->end(), frame.prefix.begin(), frame.prefix.end());
  tokens->insert(tokens->end(), pair.begin(),
                 pair.begin() + static_cast<std::ptrdiff_t>(kept));
  tokens->insert(tokens->end(), frame.suffix.begin(), frame.suffix.end());
  return true;
}

}  // namespace gufo::models::qwen3_reranker
