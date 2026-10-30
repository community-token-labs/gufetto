#ifndef GUFO_MODELS_QWEN3_RERANKER_REFERENCE_CPU_SCORER_HPP_
#define GUFO_MODELS_QWEN3_RERANKER_REFERENCE_CPU_SCORER_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "src/models/qwen3_reranker/config.hpp"

namespace gufo::models::qwen3_reranker::reference {

/// One decoder layer in float32. Matrices are row-major `[out][in]`, as
/// PyTorch stores `nn.Linear.weight`, so `y = W x`.
struct LayerWeights {
  std::vector<float> input_layernorm;           // [hidden]
  std::vector<float> q_proj;                    // [heads * head_dim][hidden]
  std::vector<float> k_proj;                    // [kv_heads * head_dim][hidden]
  std::vector<float> v_proj;                    // [kv_heads * head_dim][hidden]
  std::vector<float> q_norm;                    // [head_dim]
  std::vector<float> k_norm;                    // [head_dim]
  std::vector<float> o_proj;                    // [hidden][heads * head_dim]
  std::vector<float> post_attention_layernorm;  // [hidden]
  std::vector<float> gate_proj;                 // [intermediate][hidden]
  std::vector<float> up_proj;                   // [intermediate][hidden]
  std::vector<float> down_proj;                 // [hidden][intermediate]
};

struct Weights {
  Config config;
  std::vector<float> embed_tokens;  // [vocab][hidden]
  std::vector<LayerWeights> layers;
  std::vector<float> norm;  // [hidden]
  /// `[vocab][hidden]`; empty when the head is tied to `embed_tokens`.
  std::vector<float> lm_head;
};

/// Checks every tensor size against `weights.config`.
[[nodiscard]] bool ValidateWeights(const Weights& weights, std::string* error);

/// Qwen3's RMSNorm: `x / sqrt(mean(x^2) + eps) * weight`.
void RmsNorm(std::span<const float> input, std::span<const float> weight,
             float eps, std::span<float> output);

/// Rotates one head in place with the rotate-half RoPE layout: element `i`
/// pairs with `i + head_dim / 2` at frequency `theta^(-2i / head_dim)`.
void ApplyRope(std::span<float> head, std::size_t position, double theta);

/// The final-norm hidden state of the last token of `tokens` after a full
/// causal forward pass. Scalar and unbatched: for small inputs and tests.
[[nodiscard]] std::vector<float> LastHiddenState(
    const Weights& weights, std::span<const std::uint32_t> tokens);

/// `P(yes) = softmax(logit_yes, logit_no)[yes]` at the last token, the model
/// card's relevance score.
[[nodiscard]] double ScoreTokens(const Weights& weights,
                                 std::span<const std::uint32_t> tokens,
                                 std::uint32_t yes, std::uint32_t no);

}  // namespace gufo::models::qwen3_reranker::reference

#endif  // GUFO_MODELS_QWEN3_RERANKER_REFERENCE_CPU_SCORER_HPP_
