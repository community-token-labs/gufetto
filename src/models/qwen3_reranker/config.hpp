#ifndef GUFO_MODELS_QWEN3_RERANKER_CONFIG_HPP_
#define GUFO_MODELS_QWEN3_RERANKER_CONFIG_HPP_

#include <cstdint>
#include <filesystem>
#include <string>

#include "src/core/json.hpp"

namespace gufo::models::qwen3_reranker {

/// The dense Qwen3 causal LM a Qwen3-Reranker checkpoint ships as
/// (`Qwen3ForCausalLM`). The defaults are the 0.6B reranker's config.json
/// at e61197ed. The attention
/// width is `num_attention_heads * head_dim`, which need not equal
/// `hidden_size` (2048 vs 1024 for 0.6B).
struct Config {
  std::uint32_t vocab_size{151669};
  std::uint32_t hidden_size{1024};
  std::uint32_t intermediate_size{3072};
  std::uint32_t num_hidden_layers{28};
  std::uint32_t num_attention_heads{16};
  std::uint32_t num_key_value_heads{8};
  std::uint32_t head_dim{128};
  std::uint32_t max_position_embeddings{40960};
  float rms_norm_eps{1.0e-6F};
  double rope_theta{1000000.0};
  /// The yes/no head reads rows of `embed_tokens` when true and of
  /// `lm_head` otherwise.
  bool tie_word_embeddings{true};
  std::string torch_dtype{"bfloat16"};
};

/// Reads a Hugging Face `config.json` object. Refuses anything this
/// reference does not compute exactly: other architectures, sliding windows,
/// RoPE scaling, attention biases, activations other than SiLU, and
/// inconsistent geometry. On failure it returns false and sets `error`.
[[nodiscard]] bool ParseConfig(const json::Value& root, Config* config,
                               std::string* error);

/// Reads `<model_dir>/config.json`.
[[nodiscard]] bool LoadConfig(const std::filesystem::path& model_dir,
                              Config* config, std::string* error);

}  // namespace gufo::models::qwen3_reranker

#endif  // GUFO_MODELS_QWEN3_RERANKER_CONFIG_HPP_
