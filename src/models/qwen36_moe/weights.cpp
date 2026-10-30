#include "src/models/qwen36_moe/weights.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace gufo::models::qwen36_moe {
namespace {

using core::GgmlType;

std::string Shape(std::uint64_t cols, std::uint64_t rows,
                  std::uint64_t experts) {
  return "[" + std::to_string(cols) + ", " + std::to_string(rows) + ", " +
         std::to_string(experts) + "]";
}

QwenTensorRef Ref(const core::GgufReader& reader, std::string_view name) {
  const auto* tensor = reader.FindTensor(name);
  if (tensor == nullptr || tensor->data == nullptr) {
    return {};
  }
  const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
  std::size_t available = 0;
  for (const auto& region : reader.GetMappedRegions()) {
    const auto base = reinterpret_cast<std::uintptr_t>(region.data);
    if (address >= base && address - base < region.size) {
      available = region.size - (address - base);
      break;
    }
  }
  return {.data = tensor->data,
          .type = tensor->type,
          .num_elements = tensor->ElementCount(),
          .available_bytes = available};
}

}  // namespace

bool AcceptsType(TensorRole role, GgmlType type) noexcept {
  switch (role) {
    case TensorRole::kEmbedding:
      // The shared Qwen embedding lookup decoders.
      return type == GgmlType::kF32 || type == GgmlType::kBF16 ||
             type == GgmlType::kQ8_0 || type == GgmlType::kQ4_K ||
             type == GgmlType::kQ5_K || type == GgmlType::kQ6_K;
    case TensorRole::kProjection:
      // The shared Qwen GEMV/GEMM decoders used by the dense trunk.
      return type == GgmlType::kF32 || type == GgmlType::kBF16 ||
             type == GgmlType::kQ8_0 || type == GgmlType::kQ4_K ||
             type == GgmlType::kQ5_K || type == GgmlType::kQ6_K ||
             type == GgmlType::kQ3_K || type == GgmlType::kIQ4_NL ||
             type == GgmlType::kIQ4_XS || type == GgmlType::kIQ3_S;
    case TensorRole::kParameter:
      // Consumed as float pointers by norm, recurrence and routing kernels.
      return type == GgmlType::kF32;
    case TensorRole::kExpert:
      // The routed-expert formats of the Flash-Next MoE kernels.
      return type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
             type == GgmlType::kQ6_K || type == GgmlType::kQ5_1 ||
             type == GgmlType::kQ8_0;
  }
  return false;
}

std::vector<TensorSpec> LoaderMap(const Config& c) {
  using R = TensorRole;
  const std::uint64_t hidden = c.hidden_size;
  std::vector<TensorSpec> map;
  map.push_back({"token_embd.weight", hidden, c.vocab_size, 1, R::kEmbedding});
  map.push_back({"output_norm.weight", hidden, 1, 1, R::kParameter});
  // Absent when the artifact ties the LM head to the embedding.
  map.push_back(
      {"output.weight", hidden, c.vocab_size, 1, R::kProjection, false});
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    map.push_back({p + "attn_norm.weight", hidden, 1, 1, R::kParameter});
    map.push_back(
        {p + "post_attention_norm.weight", hidden, 1, 1, R::kParameter});
    if (c.IsLinearLayer(il)) {
      const std::uint64_t channels = c.SsmConvChannels();
      const std::uint64_t value_dim = c.SsmValueDim();
      const std::uint64_t v_heads = c.ssm_num_v_heads;
      map.push_back(
          {p + "attn_qkv.weight", hidden, channels, 1, R::kProjection});
      map.push_back(
          {p + "attn_gate.weight", hidden, value_dim, 1, R::kProjection});
      map.push_back({p + "ssm_conv1d.weight", c.ssm_conv_kernel, channels, 1,
                     R::kParameter});
      map.push_back(
          {p + "ssm_alpha.weight", hidden, v_heads, 1, R::kProjection});
      map.push_back(
          {p + "ssm_beta.weight", hidden, v_heads, 1, R::kProjection});
      map.push_back({p + "ssm_dt.bias", v_heads, 1, 1, R::kParameter});
      map.push_back({p + "ssm_a", v_heads, 1, 1, R::kParameter});
      map.push_back(
          {p + "ssm_norm.weight", c.ssm_head_dim, 1, 1, R::kParameter});
      map.push_back(
          {p + "ssm_out.weight", value_dim, hidden, 1, R::kProjection});
    } else {
      // q and its output gate are interleaved per head: [q_h | gate_h].
      map.push_back({p + "attn_q.weight", hidden, 2ULL * c.AttentionQDim(), 1,
                     R::kProjection});
      map.push_back(
          {p + "attn_k.weight", hidden, c.AttentionKvDim(), 1, R::kProjection});
      map.push_back(
          {p + "attn_v.weight", hidden, c.AttentionKvDim(), 1, R::kProjection});
      map.push_back({p + "attn_output.weight", c.AttentionQDim(), hidden, 1,
                     R::kProjection});
      map.push_back(
          {p + "attn_q_norm.weight", c.head_dim, 1, 1, R::kParameter});
      map.push_back(
          {p + "attn_k_norm.weight", c.head_dim, 1, 1, R::kParameter});
    }
    map.push_back(
        {p + "ffn_gate_inp.weight", hidden, c.num_experts, 1, R::kParameter});
    map.push_back({p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                   c.num_experts, R::kExpert});
    map.push_back({p + "ffn_up_exps.weight", hidden, c.expert_ff, c.num_experts,
                   R::kExpert});
    map.push_back({p + "ffn_down_exps.weight", c.expert_ff, hidden,
                   c.num_experts, R::kExpert});
    map.push_back(
        {p + "ffn_gate_inp_shexp.weight", hidden, 1, 1, R::kParameter});
    map.push_back({p + "ffn_gate_shexp.weight", hidden, c.shared_expert_ff, 1,
                   R::kProjection});
    map.push_back({p + "ffn_up_shexp.weight", hidden, c.shared_expert_ff, 1,
                   R::kProjection});
    map.push_back({p + "ffn_down_shexp.weight", c.shared_expert_ff, hidden, 1,
                   R::kProjection});
  }
  return map;
}

bool CheckTensorTable(const Config& config, std::span<const TensorHeader> table,
                      std::string* error_msg) {
  const auto fail = [&](std::string message) {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return false;
  };
  std::unordered_map<std::string_view, const TensorHeader*> by_name;
  for (const auto& entry : table) {
    if (!by_name.emplace(entry.name, &entry).second) {
      return fail("duplicate tensor " + std::string(entry.name));
    }
  }
  const auto map = LoaderMap(config);
  std::size_t matched = 0;
  for (const auto& spec : map) {
    const auto it = by_name.find(spec.name);
    if (it == by_name.end()) {
      if (spec.required) {
        return fail("missing tensor " + spec.name);
      }
      continue;
    }
    ++matched;
    const auto& d = it->second->dimensions;
    const std::uint64_t cols = !d.empty() ? d[0] : 0;
    const std::uint64_t rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t experts = d.size() > 2 ? d[2] : 1;
    if (d.empty() || d.size() > 3 || cols != spec.cols || rows != spec.rows ||
        experts != spec.experts) {
      return fail("tensor " + spec.name + " has shape " +
                  Shape(cols, rows, experts) + ", expected " +
                  Shape(spec.cols, spec.rows, spec.experts));
    }
    if (!AcceptsType(spec.role, it->second->type)) {
      return fail("tensor " + spec.name + " has unsupported format " +
                  std::string(core::ToString(it->second->type)));
    }
  }
  if (matched != table.size()) {
    // Ignoring an unknown tensor could silently drop part of the model.
    std::unordered_set<std::string_view> known;
    for (const auto& spec : map) {
      known.insert(spec.name);
    }
    for (const auto& entry : table) {
      if (!known.contains(entry.name)) {
        return fail("unmapped tensor " + std::string(entry.name));
      }
    }
  }
  return true;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  // The vocabulary is the embedding row count; no metadata key carries it.
  const auto* embd = reader.FindTensor("token_embd.weight");
  if (embd == nullptr || embd->dimensions.size() != 2 ||
      embd->dimensions[1] == 0 || embd->dimensions[1] > kMaxDimension) {
    if (error_msg != nullptr) {
      *error_msg = "missing or malformed token_embd.weight";
    }
    return std::nullopt;
  }
  config->vocab_size = static_cast<std::uint32_t>(embd->dimensions[1]);

  std::vector<TensorHeader> table;
  table.reserve(reader.GetTensors().size());
  for (const auto& t : reader.GetTensors()) {
    table.push_back({t.name, t.dimensions, t.type});
  }
  if (!CheckTensorTable(*config, table, error_msg)) {
    return std::nullopt;
  }

  ModelWeights w;
  w.config = *config;
  w.token_embd = Ref(reader, "token_embd.weight");
  w.output_norm = Ref(reader, "output_norm.weight");
  w.output = Ref(reader, "output.weight");
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.layers.resize(w.config.num_layers);
  for (std::uint32_t il = 0; il < w.config.num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const auto get = [&](std::string_view suffix) {
      return Ref(reader, p + std::string(suffix));
    };
    auto& l = w.layers[il];
    auto& t = l.trunk;
    t.is_full_attention = !w.config.IsLinearLayer(il);
    t.attn_norm = get("attn_norm.weight");
    t.ffn_norm = get("post_attention_norm.weight");
    if (t.is_full_attention) {
      t.attn_q = get("attn_q.weight");
      t.attn_k = get("attn_k.weight");
      t.attn_v = get("attn_v.weight");
      t.attn_output = get("attn_output.weight");
      t.attn_q_norm = get("attn_q_norm.weight");
      t.attn_k_norm = get("attn_k_norm.weight");
    } else {
      t.attn_qkv = get("attn_qkv.weight");
      t.attn_gate = get("attn_gate.weight");
      t.ssm_a = get("ssm_a");
      t.ssm_conv1d = get("ssm_conv1d.weight");
      t.ssm_dt = get("ssm_dt.bias");
      t.ssm_alpha = get("ssm_alpha.weight");
      t.ssm_beta = get("ssm_beta.weight");
      t.ssm_norm = get("ssm_norm.weight");
      t.ssm_out = get("ssm_out.weight");
    }
    l.router = get("ffn_gate_inp.weight");
    l.ffn_gate_exps = get("ffn_gate_exps.weight");
    l.ffn_up_exps = get("ffn_up_exps.weight");
    l.ffn_down_exps = get("ffn_down_exps.weight");
    l.shexp_gate_inp = get("ffn_gate_inp_shexp.weight");
    l.shexp_gate = get("ffn_gate_shexp.weight");
    l.shexp_up = get("ffn_up_shexp.weight");
    l.shexp_down = get("ffn_down_shexp.weight");
  }
  return w;
}

}  // namespace gufo::models::qwen36_moe
