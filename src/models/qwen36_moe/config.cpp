#include "src/models/qwen36_moe/config.hpp"

#include <cmath>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace gufo::models::qwen36_moe {
namespace {

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key) {
    const auto value = gguf.GetMetadataUint32(key);
    if (!value.has_value()) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return value.value_or(0);
  }

  float F32(std::string_view key) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value || !std::isfinite(*value)) {
      Fail("missing or invalid GGUF float " + std::string(key));
    }
    return value.value_or(0.0F);
  }

  /// Integer arrays of any width; GGUF writers pick the narrowest type.
  std::vector<std::uint64_t> U64Array(std::string_view key) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      Fail("missing GGUF array " + std::string(key));
    } else if (const auto* u =
                   std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

bool CheckedMul(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
  return !__builtin_mul_overflow(a, b, out);
}

bool CheckedAdd(std::uint64_t a, std::uint64_t b, std::uint64_t* out) {
  return !__builtin_add_overflow(a, b, out);
}

}  // namespace

bool Config::IsQualifiedGeometry() const noexcept {
  return num_layers == 40 && hidden_size == 2048 &&
         full_attention_interval == 4 && num_heads == 16 && num_kv_heads == 2 &&
         head_dim == 256 && rotary_dim == 64 &&
         rope_sections == std::array<std::uint32_t, 4>{11, 11, 10, 0} &&
         ssm_conv_kernel == 4 && ssm_head_dim == 128 && ssm_num_k_heads == 16 &&
         ssm_num_v_heads == 32 && ssm_inner_size == 4096 &&
         num_experts == 256 && num_experts_used == 8 && expert_ff == 512 &&
         shared_expert_ff == 512;
}

core::ModelConfig Config::TrunkConfig(std::string model_name) const {
  core::ModelConfig c;
  c.architecture = std::string(kArchitecture);
  c.model_name = std::move(model_name);
  c.num_layers = num_layers;
  c.hidden_size = hidden_size;
  c.intermediate_size = shared_expert_ff;
  c.num_attention_heads = num_heads;
  c.num_key_value_heads = num_kv_heads;
  c.head_dim = head_dim;
  c.vocab_size = vocab_size;
  c.context_length = context_length;
  c.full_attention_interval = full_attention_interval;
  c.mtp_num_layers = 0;
  c.ssm_conv_kernel = ssm_conv_kernel;
  c.ssm_state_size = ssm_head_dim;
  c.ssm_group_count = ssm_num_k_heads;
  c.ssm_time_step_rank = ssm_num_v_heads;
  c.ssm_inner_size = ssm_inner_size;
  c.rotary_dim = rotary_dim;
  c.rope_theta = rope_theta;
  c.is_text_only = true;
  return c;
}

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture").value_or("") !=
      kArchitecture) {
    r.Fail("GGUF architecture is not qwen35moe");
    return std::nullopt;
  }
  const std::string p = std::string(kArchitecture) + ".";
  auto key = [&](std::string_view suffix) { return p + std::string(suffix); };

  Config c;
  // Upstream converters add MTP layers to block_count and record them here.
  // No MTP tensors are mapped yet, so refuse rather than misnumber layers.
  if (gguf.FindMetadata(key("nextn_predict_layers")) != nullptr &&
      r.U32(key("nextn_predict_layers")) != 0) {
    r.Fail("qwen35moe MTP layers are not supported yet");
    return std::nullopt;
  }
  if (gguf.HasMtpTensors()) {
    r.Fail("qwen35moe MTP tensors are not supported yet");
    return std::nullopt;
  }
  if (gguf.HasVisionTensors()) {
    r.Fail("qwen35moe vision tensors belong in the --mmproj sidecar");
    return std::nullopt;
  }
  c.num_layers = r.U32(key("block_count"));
  c.hidden_size = r.U32(key("embedding_length"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"));

  c.full_attention_interval = r.U32(key("full_attention_interval"));
  c.num_heads = r.U32(key("attention.head_count"));
  c.num_kv_heads = r.U32(key("attention.head_count_kv"));
  c.head_dim = r.U32(key("attention.key_length"));
  if (r.U32(key("attention.value_length")) != c.head_dim) {
    r.Fail("attention value length must match the key length");
  }
  c.rotary_dim = r.U32(key("rope.dimension_count"));
  c.rope_theta = r.F32(key("rope.freq_base"));
  const auto sections = r.U64Array(key("rope.dimension_sections"));

  c.ssm_conv_kernel = r.U32(key("ssm.conv_kernel"));
  c.ssm_head_dim = r.U32(key("ssm.state_size"));
  c.ssm_num_k_heads = r.U32(key("ssm.group_count"));
  c.ssm_num_v_heads = r.U32(key("ssm.time_step_rank"));
  c.ssm_inner_size = r.U32(key("ssm.inner_size"));

  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  c.expert_ff = r.U32(key("expert_feed_forward_length"));
  c.shared_expert_ff = r.U32(key("expert_shared_feed_forward_length"));
  if (!r.ok) {
    return std::nullopt;
  }

  // Stage 1: every raw value is positive and representable before any value
  // takes part in arithmetic. Kernels index with 32-bit signed integers.
  for (const std::uint32_t value :
       {c.num_layers, c.hidden_size, c.full_attention_interval, c.num_heads,
        c.num_kv_heads, c.head_dim, c.rotary_dim, c.ssm_conv_kernel,
        c.ssm_head_dim, c.ssm_num_k_heads, c.ssm_num_v_heads, c.ssm_inner_size,
        c.num_experts, c.num_experts_used, c.expert_ff, c.shared_expert_ff}) {
    if (value == 0 || value > kMaxDimension) {
      r.Fail("qwen35moe dimensions must be positive and fit 32-bit indexing");
      return std::nullopt;
    }
  }
  if (c.context_length < 2 || c.context_length > kMaxDimension ||
      !(c.rms_eps > 0) || !(c.rope_theta > 0)) {
    r.Fail("qwen35moe metadata describes an unsupported shape");
    return std::nullopt;
  }

  // Stage 2: every derived dimension in checked 64-bit arithmetic, each
  // bounded like the raw values.
  std::uint64_t key_dim = 0;
  std::uint64_t value_dim = 0;
  std::uint64_t conv_channels = 0;
  std::uint64_t q_dim = 0;
  std::uint64_t q_gate_dim = 0;
  std::uint64_t kv_dim = 0;
  if (!CheckedMul(c.ssm_num_k_heads, c.ssm_head_dim, &key_dim) ||
      !CheckedMul(c.ssm_num_v_heads, c.ssm_head_dim, &value_dim) ||
      !CheckedMul(2, key_dim, &conv_channels) ||
      !CheckedAdd(conv_channels, value_dim, &conv_channels) ||
      !CheckedMul(c.num_heads, c.head_dim, &q_dim) ||
      !CheckedMul(2, q_dim, &q_gate_dim) ||
      !CheckedMul(c.num_kv_heads, c.head_dim, &kv_dim)) {
    r.Fail("qwen35moe derived dimensions overflow");
    return std::nullopt;
  }
  for (const std::uint64_t value :
       {key_dim, value_dim, conv_channels, q_dim, q_gate_dim, kv_dim}) {
    if (value > kMaxDimension) {
      r.Fail("qwen35moe derived dimensions exceed 32-bit indexing");
      return std::nullopt;
    }
  }

  // Stage 3: relations between validated values. Structural limits shared
  // with the dense Qwen3.5 trunk (see core::ModelConfig::IsValidQwen), not
  // limits of the format.
  if (c.full_attention_interval != 4 ||
      c.num_layers % c.full_attention_interval != 0 ||
      c.num_heads % c.num_kv_heads != 0 || c.rotary_dim % 2 != 0 ||
      c.rotary_dim > c.head_dim || c.ssm_conv_kernel != 4 ||
      c.ssm_head_dim != 128 || c.ssm_num_v_heads % c.ssm_num_k_heads != 0 ||
      c.ssm_inner_size != value_dim || c.num_experts_used > c.num_experts) {
    r.Fail("qwen35moe metadata describes an unsupported shape");
    return std::nullopt;
  }
  if (sections.size() != 4 || sections[3] != 0) {
    r.Fail("rope.dimension_sections must contain three axes and a zero pad");
    return std::nullopt;
  }
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    if (sections[i] == 0 || sections[i] > c.rotary_dim / 2) {
      r.Fail("invalid mRoPE axis dimension");
      return std::nullopt;
    }
    sum += sections[i];
    c.rope_sections[i] = static_cast<std::uint32_t>(sections[i]);
  }
  if (sum != c.rotary_dim / 2) {
    r.Fail("mRoPE sections must sum to half the rotary dimension");
    return std::nullopt;
  }
  return c;
}

}  // namespace gufo::models::qwen36_moe
