#ifndef GUFO_MODELS_QWEN36_MOE_CONFIG_HPP_
#define GUFO_MODELS_QWEN36_MOE_CONFIG_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"
#include "src/core/model_config.hpp"

namespace gufo::models::qwen36_moe {

/// GGUF architecture of Qwen3.5/3.6 MoE checkpoints (llama.cpp `qwen35moe`).
inline constexpr std::string_view kArchitecture = "qwen35moe";

/// Architecture parameters of a `qwen35moe` GGUF artifact: the Qwen3.5 dense
/// trunk (Gated DeltaNet plus gated GQA every fourth layer) with every dense
/// FFN replaced by routed experts and one sigmoid-gated shared expert. Every
/// value is read from the file.
struct Config {
  std::uint32_t num_layers{0};      ///< Trunk layers (MTP excluded), 40.
  std::uint32_t hidden_size{0};     ///< 2048
  std::uint32_t vocab_size{0};      ///< token_embd rows, 248320.
  std::uint32_t context_length{0};  ///< 262144
  float rms_eps{1e-6F};

  // Gated GQA on every full_attention_interval-th layer.
  std::uint32_t full_attention_interval{0};      ///< 4
  std::uint32_t num_heads{0};                    ///< 16
  std::uint32_t num_kv_heads{0};                 ///< 2
  std::uint32_t head_dim{0};                     ///< 256
  std::uint32_t rotary_dim{0};                   ///< 64
  float rope_theta{0.0F};                        ///< 1e7
  std::array<std::uint32_t, 4> rope_sections{};  ///< mRoPE {11, 11, 10, 0}

  // Gated DeltaNet linear attention.
  std::uint32_t ssm_conv_kernel{0};  ///< 4
  std::uint32_t ssm_head_dim{0};     ///< 128 (state_size; key and value dim)
  std::uint32_t ssm_num_k_heads{0};  ///< 16 (group_count)
  std::uint32_t ssm_num_v_heads{0};  ///< 32 (time_step_rank)
  std::uint32_t ssm_inner_size{0};   ///< 4096

  // Mixture of experts with one always-on shared expert.
  std::uint32_t num_experts{0};       ///< 256
  std::uint32_t num_experts_used{0};  ///< 8
  std::uint32_t expert_ff{0};         ///< 512
  std::uint32_t shared_expert_ff{0};  ///< 512

  [[nodiscard]] std::uint32_t SsmKeyDim() const noexcept {
    return ssm_num_k_heads * ssm_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmValueDim() const noexcept {
    return ssm_num_v_heads * ssm_head_dim;
  }
  /// Channels of the fused q|k|v projection and its causal convolution.
  [[nodiscard]] std::uint32_t SsmConvChannels() const noexcept {
    return 2 * SsmKeyDim() + SsmValueDim();
  }
  [[nodiscard]] std::uint32_t AttentionQDim() const noexcept {
    return num_heads * head_dim;
  }
  [[nodiscard]] std::uint32_t AttentionKvDim() const noexcept {
    return num_kv_heads * head_dim;
  }
  [[nodiscard]] bool IsLinearLayer(std::uint32_t layer) const noexcept {
    return layer < num_layers && ((layer + 1) % full_attention_interval) != 0;
  }

  /// True only for the Qwen3.6-35B-A3B geometry. Other `qwen35moe` shapes
  /// parse and bind, but no runtime has been qualified for them.
  [[nodiscard]] bool IsQualifiedGeometry() const noexcept;

  /// The dense-trunk view consumed by the shared Qwen attention, Gated
  /// DeltaNet and state-sizing code. `intermediate_size` carries the shared
  /// expert width; the dense FFN tensors themselves are absent.
  [[nodiscard]] core::ModelConfig TrunkConfig(std::string model_name) const;

  /// Reads and validates the `qwen35moe.*` metadata. Rejects embedded MTP
  /// layers and vision tensors, which this loader does not map yet.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);
};

}  // namespace gufo::models::qwen36_moe

#endif  // GUFO_MODELS_QWEN36_MOE_CONFIG_HPP_
