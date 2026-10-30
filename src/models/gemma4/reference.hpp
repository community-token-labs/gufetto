#ifndef GUFO_MODELS_GEMMA4_REFERENCE_HPP_
#define GUFO_MODELS_GEMMA4_REFERENCE_HPP_

#include <cstdint>
#include <span>
#include <vector>

#include "src/models/gemma4/tokenizer.hpp"  // TokenId
#include "src/models/gemma4/weights.hpp"

namespace gufo::models::gemma4 {

/// Scalar reference of the Gemma 4 graph (llama.cpp 6d5ac3c8
/// `src/models/gemma4.cpp`), the judge for the device runtime. Weights are
/// dequantized on the fly, every
/// reduction accumulates in double, and every layer keeps full K/V for all
/// positions, so it only suits short prefixes. It pins operator semantics:
/// scaled embeddings, weighted q/k norms, the unweighted V norm and K=V on
/// global layers, NEOX rope with per-layer bases and frequency divisors,
/// window-masked attention with scale 1, sandwich norms, GeGLU, routed
/// experts (26B-A4B), the layer output scale, and the final logit softcap.
class Reference {
public:
  /// `weights` must have its host values read (ModelWeights::ReadHostValues)
  /// and must outlive the reference.
  explicit Reference(const ModelWeights& weights);

  /// Appends `tokens` after the current position. `logits` receives
  /// [tokens.size()][vocab] softcapped rows and `hidden` the post-norm
  /// [tokens.size()][hidden] rows fed to the LM head and to MTP; either may
  /// be null. Processing a prefix in several calls gives identical results.
  void Forward(std::span<const TokenId> tokens, std::vector<float>* logits,
               std::vector<float>* hidden);

  /// When enabled, Forward records the residual stream after every layer:
  /// [layer][row][hidden] for the rows of the latest call.
  void SetTrace(bool enabled) noexcept { trace_ = enabled; }
  [[nodiscard]] const std::vector<float>& LayerTrace() const noexcept {
    return layer_trace_;
  }

  [[nodiscard]] std::uint32_t Position() const noexcept { return position_; }
  void Reset();

private:
  /// out[r][o] = sum_c W[o][c] * x[r][c] for `rows` input rows.
  void MatMul(const TensorRef& weight, const float* x, std::size_t rows,
              float* out) const;
  /// Causal attention of `rows` new rows over every stored key; sliding
  /// layers see the last `sliding_window` keys, including their own.
  void Attention(std::uint32_t layer, const float* q, std::size_t rows,
                 float* out) const;
  /// Routed expert mixture of `rows` attention residual rows (26B-A4B):
  /// router, top-k softmax weights, per-expert scale, GeGLU experts.
  void Experts(const LayerWeights& w, const float* x, std::size_t rows,
               float* out) const;

  const ModelWeights& weights_;
  std::uint32_t position_{0};
  /// Per layer, [position][kv_heads * head_dim].
  std::vector<std::vector<float>> keys_;
  std::vector<std::vector<float>> values_;
  bool trace_{false};
  std::vector<float> layer_trace_;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_REFERENCE_HPP_
