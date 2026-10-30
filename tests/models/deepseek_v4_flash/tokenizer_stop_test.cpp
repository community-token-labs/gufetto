#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/deepseek_v4_flash/runtime/native_internal.h"

// Only the metadata transport is fake. Compile the real CPU tokenizer without
// loading weights or linking the ROCm runtime. Spellings are independent
// goldens from this family's tokenizer and chat-template reference, not Qwen
// heuristics.
struct ds4_model {
  std::vector<std::string> tokens{"Neraz",
                                  "im",
                                  "|",
                                  ">",
                                  "<",
                                  "_end",
                                  "<|im_end|>",
                                  "<|endoftext|>",
                                  "<｜begin▁of▁sentence｜>",
                                  "<｜end▁of▁sentence｜>",
                                  "<｜User｜>",
                                  "<｜Assistant｜>",
                                  "<think>",
                                  "</think>",
                                  "｜DSML｜"};
};

bool ds4_model_string_array_begin(const ds4_model* model, const char* key,
                                  ds4_string_iterator* iterator) {
  const std::string_view name(key);
  if (name != "tokenizer.ggml.tokens" && name != "tokenizer.ggml.merges")
    return false;
  *iterator = {model, 0,
               name == "tokenizer.ggml.tokens" ? model->tokens.size() : 0};
  return true;
}

bool ds4_model_string_array_next(ds4_string_iterator* iterator,
                                 ds4_string_view* value) {
  if (iterator->remaining == 0)
    return false;
  const auto& token = iterator->model->tokens.at(iterator->pos++);
  *value = {token.data(), token.size()};
  --iterator->remaining;
  return true;
}

int main() {
  const ds4_model model;
  const std::unique_ptr<ds4_vocab, decltype(&ds4_vocab_destroy)> vocabulary(
      ds4_vocab_create(&model), ds4_vocab_destroy);
  if (!vocabulary || ds4_vocab_eos(vocabulary.get()) != 9)
    return 1;
  for (int id = 0; id < static_cast<int>(model.tokens.size()); ++id) {
    if (ds4_vocab_is_stop(vocabulary.get(), id) != (id == 9)) {
      std::cerr << "Only DeepSeek's atomic EOS may stop: " << model.tokens[id]
                << '\n';
      return 1;
    }
  }
  std::string text;
  for (const int id : {0, 1, 2, 3}) {
    std::size_t length = 0;
    std::unique_ptr<char, decltype(&std::free)> piece(
        ds4_vocab_token_text(vocabulary.get(), id, &length), std::free);
    text.append(piece.get(), length);
  }
  if (text != "Nerazim|>")
    return 1;
  std::cout << "DeepSeek EOS atomicity passed\n";
}
