// Encodes fixtures/tokenizer_goldens.json with the vocabulary of the pinned
// 26B-A4B target. The goldens come from tools/gemma4/tokenizer_goldens.py, a
// Python port of llama.cpp 6d5ac3c8's gemma4 tokenizer run over the same
// file's metadata, and agree with PR #426's llama.cpp b11069 ids on its
// corpus. Only GGUF headers are read. Skips (77) unless GUFO_GEMMA4_MODEL
// names the file.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/gemma4/control_tokens.hpp"
#include "src/models/gemma4/tokenizer.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using g4::TokenId;
using gemma4_test::Require;

namespace {

void Bytes(gufo::crypto::Sha256Hasher* h, const void* data, std::size_t n) {
  h->Update({static_cast<const std::uint8_t*>(data), n});
}

/// The generator's tokenizer_sha256: every token and merge as a u64 length
/// plus bytes, then the token types as i32, all little-endian.
std::string TokenizerSha256(const gufo::core::GgufReader& gguf) {
  gufo::crypto::Sha256Hasher h;
  for (const char* key : {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}) {
    for (const auto text : gguf.GetMetadataStringArray(key)) {
      const std::uint64_t size = text.size();
      Bytes(&h, &size, sizeof(size));
      Bytes(&h, text.data(), text.size());
    }
  }
  const auto* types = gguf.FindMetadata("tokenizer.ggml.token_type");
  Require(types != nullptr, "missing token types");
  const auto* values = std::get_if<std::vector<std::int64_t>>(&types->value);
  Require(values != nullptr, "token types are not signed integers");
  for (const auto value : *values) {
    const auto type = static_cast<std::int32_t>(value);
    Bytes(&h, &type, sizeof(type));
  }
  return h.FinishHex();
}

std::string Hex(const std::string& text) {
  std::string out;
  for (std::size_t i = 0; i + 1 < text.size(); i += 2) {
    out.push_back(static_cast<char>(std::stoi(text.substr(i, 2), nullptr, 16)));
  }
  return out;
}

std::string IdsSha256(const std::vector<TokenId>& ids) {
  std::vector<std::uint8_t> bytes;
  for (const auto id : ids) {
    const auto value = static_cast<std::uint32_t>(id);
    for (int shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    }
  }
  return gufo::crypto::Sha256Hex(bytes);
}

/// Ids recorded in the port plan's header dump of the same artifact,
/// independent of the goldens and of control_tokens.hpp's spellings.
void CheckControlTokens(const g4::Tokenizer& t) {
  const auto expect = [&](std::string_view text, TokenId id) {
    Require(t.FindToken(text) == id, "control token " + std::string(text) +
                                         " is not " + std::to_string(id));
  };
  expect("<pad>", 0);
  expect("<eos>", 1);
  expect("<bos>", 2);
  expect("<|tool>", 46);
  expect("<tool|>", 47);
  expect("<|tool_call>", 48);
  expect("<tool_call|>", 49);
  expect("<|tool_response>", 50);
  expect("<tool_response|>", 51);
  expect("<|\"|>", 52);
  expect("<|think|>", 98);
  expect("<|channel>", 100);
  expect("<channel|>", 101);
  expect("<|turn>", 105);
  expect("<turn|>", 106);
  expect("<|image>", 255999);
  expect("<|image|>", 258880);
  expect("<image|>", 258882);
  expect("<|video|>", 258884);
  // Every named constant exists in the vocabulary.
  for (const auto text : {g4::kPad,
                          g4::kEos,
                          g4::kBos,
                          g4::kTurnStart,
                          g4::kTurnEnd,
                          g4::kThink,
                          g4::kChannelStart,
                          g4::kChannelEnd,
                          g4::kToolStart,
                          g4::kToolEnd,
                          g4::kToolCallStart,
                          g4::kToolCallEnd,
                          g4::kToolResponseStart,
                          g4::kToolResponseEnd,
                          g4::kQuote,
                          g4::kImageStart,
                          g4::kImage,
                          g4::kImageEnd,
                          g4::kAudioStart,
                          g4::kAudio,
                          g4::kAudioEnd,
                          g4::kVideo}) {
    Require(t.FindToken(text).has_value(),
            "constant missing from the vocabulary: " + std::string(text));
  }
  Require(t.BosToken() == 2 && t.EosToken() == 106, "bos 2 and eos 106");
  const auto eog = t.EndOfGenerationTokens();
  Require(std::vector<TokenId>(eog.begin(), eog.end()) ==
              std::vector<TokenId>{1, 50, 106},
          "end of generation is <eos>, <|tool_response> and <turn|>");
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_GEMMA4_MODEL");
  if (model == nullptr || !std::filesystem::exists(model)) {
    std::cout << "SKIP: set GUFO_GEMMA4_MODEL\n";
    return 77;
  }
  return gemma4_test::Run([&] {
    std::string error;
    const auto gguf = gufo::core::GgufReader::OpenFile(model, &error);
    Require(gguf != nullptr, error);
    const auto tokenizer = g4::Tokenizer::CreateFromGguf(*gguf, &error);
    Require(tokenizer != nullptr, "tokenizer rejected: " + error);

    std::ifstream file(std::string(GUFO_GEMMA4_FIXTURES) +
                       "/tokenizer_goldens.json");
    std::stringstream text;
    text << file.rdbuf();
    const auto goldens = gufo::json::parse(text.str());
    const auto* meta = goldens.find("model");
    Require(meta != nullptr &&
                meta->member_str("tokenizer_sha256") == TokenizerSha256(*gguf),
            "the GGUF vocabulary differs from the goldens' vocabulary");
    CheckControlTokens(*tokenizer);

    const auto* entries = goldens.find("entries");
    Require(entries != nullptr && entries->items().size() >= 30,
            "golden entries");
    for (const auto& entry : entries->items()) {
      const std::string name = entry.member_str("name");
      const auto ids = tokenizer->Encode(
          Hex(entry.member_str("text_hex")), entry.find("add_bos")->as_bool(),
          entry.find("parse_special")->as_bool());
      if (const auto* expected = entry.find("ids")) {
        std::vector<TokenId> want;
        for (const auto& id : expected->items()) {
          want.push_back(static_cast<TokenId>(id.as_size()));
        }
        std::string got;
        for (const auto id : ids) {
          got += std::to_string(id) + " ";
        }
        Require(ids == want, name + ": got " + got);
      } else {
        Require(ids.size() == entry.member_size("count") &&
                    IdsSha256(ids) == entry.member_str("ids_sha256"),
                name + ": long encoding differs");
      }
    }
  });
}
