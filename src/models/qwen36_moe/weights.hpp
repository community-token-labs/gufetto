#ifndef GUFO_MODELS_QWEN36_MOE_WEIGHTS_HPP_
#define GUFO_MODELS_QWEN36_MOE_WEIGHTS_HPP_

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen36_moe/config.hpp"

namespace gufo::models::qwen36_moe {

/// Storage contract of a mapped tensor. Each role names the formats the
/// planned runtime decodes for it.
enum class TensorRole : std::uint8_t {
  kEmbedding,   ///< Row lookup.
  kProjection,  ///< Dense GEMV/GEMM operand, including the LM head.
  kParameter,   ///< Norm gammas, SSM scalars, router and shared-expert gate.
  kExpert,      ///< Stacked routed-expert matrices.
};

/// One expected tensor in GGUF orientation: `cols` (ne[0]) is the contiguous
/// reduction dimension, `rows` (ne[1]) the output dimension and `experts`
/// (ne[2]) the stacked expert count.
struct TensorSpec {
  std::string name;
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::uint64_t experts{1};
  TensorRole role{TensorRole::kParameter};
  bool required{true};
};

/// Header-only description of one tensor present in an artifact.
struct TensorHeader {
  std::string_view name;
  std::span<const std::uint64_t> dimensions;
  core::GgmlType type{core::GgmlType::kF32};
};

[[nodiscard]] bool AcceptsType(TensorRole role, core::GgmlType type) noexcept;

/// Every tensor the trunk loader binds, in file order. `config.vocab_size`
/// must already be known.
[[nodiscard]] std::vector<TensorSpec> LoaderMap(const Config& config);

/// Checks an artifact's tensor table against `LoaderMap` without touching
/// payloads: required tensors present, shapes exact, formats accepted, and no
/// tensor left unmapped.
[[nodiscard]] bool CheckTensorTable(const Config& config,
                                    std::span<const TensorHeader> table,
                                    std::string* error_msg = nullptr);

/// Routed and shared experts of one layer; the attention/DeltaNet half is the
/// dense Qwen layer view with its FFN tensors left empty.
struct LayerWeights {
  QwenLayerWeights trunk;
  QwenTensorRef router;          ///< [hidden -> num_experts] F32
  QwenTensorRef ffn_gate_exps;   ///< [hidden -> expert_ff] x experts
  QwenTensorRef ffn_up_exps;     ///< [hidden -> expert_ff] x experts
  QwenTensorRef ffn_down_exps;   ///< [expert_ff -> hidden] x experts
  QwenTensorRef shexp_gate_inp;  ///< [hidden] -> scalar sigmoid gate
  QwenTensorRef shexp_gate;      ///< [hidden -> shared_ff]
  QwenTensorRef shexp_up;        ///< [hidden -> shared_ff]
  QwenTensorRef shexp_down;      ///< [shared_ff -> hidden]
};

struct ModelWeights {
  Config config;
  QwenTensorRef token_embd;
  QwenTensorRef output_norm;
  QwenTensorRef output;  ///< token_embd when the artifact ties the head.
  std::vector<LayerWeights> layers;

  /// Binds and validates the trunk artifact. Payloads stay mapped and
  /// untouched; only headers and extents are checked.
  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen36_moe

#endif  // GUFO_MODELS_QWEN36_MOE_WEIGHTS_HPP_
