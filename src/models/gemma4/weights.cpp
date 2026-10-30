#include "src/models/gemma4/weights.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace gufo::models::gemma4 {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this runtime binds. Unknown types get a
/// zero block, which every validation below rejects.
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ5_1:
      return {32, 24};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    default:
      return {0, 0};
  }
}

// Accepted formats per role are exactly those of the supported artifact,
// unsloth gemma-4-26B-A4B-it UD-Q4_K_XL and its MTP drafter. Widening a role
// needs a kernel and a test for the new format first.
/// Dense attention, MLP and draft projections.
constexpr std::initializer_list<GgmlType> kProjection = {GgmlType::kQ8_0};
/// Token embedding, which is also the tied LM head.
constexpr std::initializer_list<GgmlType> kEmbedding = {GgmlType::kQ8_0};
/// Norm weights, scales, rope divisors and the expert router.
constexpr std::initializer_list<GgmlType> kVector = {GgmlType::kF32};
/// Fused routed gate/up experts, reducing over the hidden width.
constexpr std::initializer_list<GgmlType> kExpertGateUp = {GgmlType::kQ4_K,
                                                           GgmlType::kQ5_K};
/// Routed down experts, reducing over the (not 256-aligned) expert width.
constexpr std::initializer_list<GgmlType> kExpertDown = {GgmlType::kQ5_1,
                                                         GgmlType::kQ8_0};

/// `*value *= factor`, false (value unchanged) when the product wraps.
[[nodiscard]] bool MulChecked(std::uint64_t* value,
                              std::uint64_t factor) noexcept {
  if (factor != 0 && *value > UINT64_MAX / factor) {
    return false;
  }
  *value *= factor;
  return true;
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};
  std::unordered_set<std::string_view> bound;

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats;
  /// `experts` > 1 expects a stack of that many [cols, rows] matrices.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::initializer_list<GgmlType> types, bool required = true,
                std::uint64_t experts = 1) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    bool type_ok = false;
    std::string accepted;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
      accepted +=
          (accepted.empty() ? "" : ", ") + std::string(core::ToString(type));
    }
    const Format format = FormatOf(info->type);
    if (!type_ok || format.block == 0) {
      Fail("tensor " + name + " has format " +
           std::string(core::ToString(info->type)) + "; accepted: " + accepted);
      return t;
    }
    if (cols % format.block != 0) {
      Fail("tensor " + name + " has " + std::to_string(cols) +
           " columns, not a multiple of the " +
           std::string(core::ToString(info->type)) + " block of " +
           std::to_string(format.block));
      return t;
    }
    // The extent must not wrap, and must be what the reader indexed.
    std::uint64_t bytes = cols / format.block * format.bytes;
    if (!MulChecked(&bytes, rows) || !MulChecked(&bytes, experts) ||
        bytes != info->size_bytes) {
      Fail("tensor " + name + " has an inconsistent payload size");
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    bound.insert(info->name);
    // Locate the shard so disk readers can address the payload directly and
    // a payload running past its shard is caught here, not as a fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool mapped = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        mapped = true;
        break;
      }
    }
    if (!mapped) {
      Fail("tensor " + name + " lies outside the mapped file");
      return TensorRef{};
    }
    if (bytes > regions[t.shard].size - t.file_offset) {
      Fail("tensor " + name + " is truncated: " + std::to_string(bytes) +
           " bytes at offset " + std::to_string(t.file_offset) + " of " +
           std::to_string(regions[t.shard].size));
      return TensorRef{};
    }
    return t;
  }

  TensorRef RopeFreqs(const Config& c) {
    return Get("rope_freqs.weight", c.head_dim_global / 2, 1, kVector);
  }

  /// Every tensor in the file must have a role; an unknown name means a
  /// variant this binder does not understand.
  void RejectUnmapped() {
    for (const auto& info : reader.GetTensors()) {
      if (!ok) {
        return;
      }
      if (!bound.contains(info.name)) {
        Fail("unmapped tensor " + std::string(info.name));
      }
    }
  }

  /// Binds one block. `with_kv` is false for Q-only draft layers.
  LayerWeights Layer(const Config& c, std::uint32_t l, bool with_kv) {
    LayerWeights w;
    const std::string p = "blk." + std::to_string(l) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t head_dim = c.HeadDim(l);
    w.attn_norm = Get(p + "attn_norm.weight", hidden, 1, kVector);
    w.attn_q = Get(p + "attn_q.weight", hidden, c.QDim(l), kProjection);
    w.attn_q_norm = Get(p + "attn_q_norm.weight", head_dim, 1, kVector);
    if (with_kv) {
      w.attn_k = Get(p + "attn_k.weight", hidden, c.KvDim(l), kProjection);
      w.attn_v =
          Get(p + "attn_v.weight", hidden, c.KvDim(l), kProjection, false);
      w.attn_k_norm = Get(p + "attn_k_norm.weight", head_dim, 1, kVector);
    } else if (reader.FindTensor(p + "attn_k.weight") != nullptr ||
               reader.FindTensor(p + "attn_v.weight") != nullptr) {
      Fail("shared-KV draft layer " + std::to_string(l) +
           " must not carry K/V projections");
    }
    w.attn_output =
        Get(p + "attn_output.weight", c.QDim(l), hidden, kProjection);
    w.post_attn_norm =
        Get(p + "post_attention_norm.weight", hidden, 1, kVector);
    w.ffn_norm = Get(p + "ffn_norm.weight", hidden, 1, kVector);
    w.ffn_gate = Get(p + "ffn_gate.weight", hidden, c.ffn_size, kProjection);
    w.ffn_up = Get(p + "ffn_up.weight", hidden, c.ffn_size, kProjection);
    w.ffn_down = Get(p + "ffn_down.weight", c.ffn_size, hidden, kProjection);
    w.post_ffn_norm = Get(p + "post_ffw_norm.weight", hidden, 1, kVector);
    w.layer_output_scale =
        Get(p + "layer_output_scale.weight", 1, 1, kVector, false);
    if (c.HasExperts()) {
      Experts(c, p, &w);
    } else if (reader.FindTensor(p + "ffn_gate_inp.weight") != nullptr) {
      Fail("layer " + std::to_string(l) + " carries experts in a dense model");
    }
    return w;
  }

  /// The router and the routed experts of an expert layer. Only the fused
  /// gate/up layout is supported.
  void Experts(const Config& c, const std::string& p, LayerWeights* w) {
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t experts = c.num_experts;
    const std::uint64_t width = c.expert_ffn_size;
    w->router = Get(p + "ffn_gate_inp.weight", hidden, experts, kVector);
    w->router_scale = Get(p + "ffn_gate_inp.scale", hidden, 1, kVector);
    w->pre_ffn_norm_2 = Get(p + "pre_ffw_norm_2.weight", hidden, 1, kVector);
    w->post_ffn_norm_1 = Get(p + "post_ffw_norm_1.weight", hidden, 1, kVector);
    w->post_ffn_norm_2 = Get(p + "post_ffw_norm_2.weight", hidden, 1, kVector);
    if (ok && reader.FindTensor(p + "ffn_gate_up_exps.weight") == nullptr) {
      Fail("tensor " + p +
           "ffn_gate_up_exps.weight is missing; separate gate and up "
           "experts are not supported");
      return;
    }
    w->gate_up_exps = Get(p + "ffn_gate_up_exps.weight", hidden, 2 * width,
                          kExpertGateUp, true, experts);
    w->down_exps = Get(p + "ffn_down_exps.weight", width, hidden, kExpertDown,
                       true, experts);
    w->down_exps_scale = Get(p + "ffn_down_exps.scale", experts, 1, kVector);
  }
};

/// Reads the small F32 payloads the host needs: per-layer output scales and
/// the rope divisors.
bool ReadValues(const TensorRef& rope_freqs, std::vector<LayerWeights>* layers,
                RopeFactors* factors, std::string* error_msg) {
  const auto fail = [&](std::string message) {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return false;
  };
  if (rope_freqs.empty()) {
    return fail("rope_freqs.weight is not bound; call Bind first");
  }
  for (auto& layer : *layers) {
    if (layer.layer_output_scale.empty()) {
      continue;
    }
    float value = 0.0F;
    std::memcpy(&value, layer.layer_output_scale.data, sizeof(value));
    if (!std::isfinite(value)) {
      return fail("tensor " + std::string(layer.layer_output_scale.name) +
                  " is not finite");
    }
    layer.output_scale = value;
  }
  factors->values.resize(rope_freqs.cols);
  std::memcpy(factors->values.data(), rope_freqs.data,
              factors->values.size() * sizeof(float));
  for (float v : factors->values) {
    if (!std::isfinite(v) || !(v > 0.0F)) {
      factors->values.clear();
      return fail("rope_freqs.weight must contain positive finite divisors");
    }
  }
  return true;
}

/// Vocabulary size from the embedding tensor, which every other vocabulary
/// sized tensor must match.
std::uint64_t VocabRows(const core::GgufReader& reader) {
  const auto* info = reader.FindTensor("token_embd.weight");
  if (info == nullptr || info->dimensions.size() != 2) {
    return 0;
  }
  return info->dimensions[1];
}

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format f = FormatOf(type);
  return f.block == 0 ? 0 : static_cast<std::size_t>(cols / f.block) * f.bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  ModelWeights w;
  w.config = std::move(*config);
  const Config& c = w.config;
  const std::uint64_t vocab = VocabRows(reader);
  if (vocab == 0 || vocab > 0x7fffffffULL) {
    b.Fail("token_embd.weight must be a [hidden, vocab] matrix");
    return std::nullopt;
  }
  w.vocab_size = static_cast<std::uint32_t>(vocab);
  w.token_embd = b.Get("token_embd.weight", c.hidden_size, vocab, kEmbedding);
  w.output = b.Get("output.weight", c.hidden_size, vocab, kEmbedding, false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.output_norm = b.Get("output_norm.weight", c.hidden_size, 1, kVector);
  w.rope_freqs = b.RopeFreqs(c);
  for (std::uint32_t l = 0; b.ok && l < c.num_layers; ++l) {
    w.layers.push_back(b.Layer(c, l, true));
    // A missing V projection means V is the unnormalized K projection; that
    // is only defined when K and V share a width.
    if (b.ok && w.layers.back().attn_v.empty() && c.IsSliding(l)) {
      b.Fail("sliding layer " + std::to_string(l) + " lacks attn_v");
    }
  }
  b.RejectUnmapped();
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

std::optional<DraftWeights> DraftWeights::Bind(const core::GgufReader& reader,
                                               const ModelWeights& target,
                                               std::string* error_msg) {
  auto config = Config::DraftFromGguf(reader, target.config, error_msg);
  if (!config) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  DraftWeights w;
  w.config = std::move(*config);
  const Config& c = w.config;
  if (VocabRows(reader) != target.vocab_size) {
    b.Fail("gemma4-assistant vocabulary differs from the target");
    return std::nullopt;
  }
  const std::uint64_t target_hidden = c.target_hidden_size;
  w.pre_projection = b.Get("nextn.pre_projection.weight", 2 * target_hidden,
                           c.hidden_size, kProjection);
  w.post_projection = b.Get("nextn.post_projection.weight", c.hidden_size,
                            target_hidden, kProjection);
  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, target.vocab_size, kEmbedding);
  w.output_norm = b.Get("output_norm.weight", c.hidden_size, 1, kVector);
  w.rope_freqs = b.RopeFreqs(c);
  for (std::uint32_t l = 0; b.ok && l < c.num_layers; ++l) {
    w.layers.push_back(b.Layer(c, l, false));
  }
  b.RejectUnmapped();
  if (!b.ok) {
    return std::nullopt;
  }
  return w;
}

bool ModelWeights::ReadHostValues(std::string* error_msg) {
  return ReadValues(rope_freqs, &layers, &rope_factors, error_msg);
}

bool DraftWeights::ReadHostValues(std::string* error_msg) {
  return ReadValues(rope_freqs, &layers, &rope_factors, error_msg);
}

}  // namespace gufo::models::gemma4
