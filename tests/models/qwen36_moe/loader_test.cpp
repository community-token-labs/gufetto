#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen36_moe/config.hpp"
#include "src/models/qwen36_moe/weights.hpp"

namespace q36 = gufo::models::qwen36_moe;
using gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

using Value = std::variant<std::uint32_t, std::int32_t, float, std::string,
                           std::vector<std::uint64_t>>;
using Metadata = std::map<std::string, Value>;

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  GgmlType type{GgmlType::kF32};
};

std::uint64_t EncodedBytes(const Tensor& t) {
  std::uint64_t n = 1;
  for (const auto d : t.dims)
    n *= d;
  switch (t.type) {
    case GgmlType::kF32:
      return n * 4;
    case GgmlType::kQ8_0:
      return n / 32 * 34;
    default:
      throw std::runtime_error("synthetic payload format not implemented");
  }
}

// GGUF v3 image. Payloads are zero; the loader reads headers and extents only.
std::vector<std::uint8_t> WriteGguf(const Metadata& fields,
                                    const std::vector<Tensor>& tensors) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<class T>(T value) {
    const auto begin = bytes.size();
    bytes.resize(begin + sizeof(value));
    std::memcpy(bytes.data() + begin, &value, sizeof(value));
  };
  const auto text = [&](const std::string& value) {
    pod(std::uint64_t{value.size()});
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{tensors.size()});
  pod(std::uint64_t{fields.size()});
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::int32_t>) {
            pod(std::uint32_t{5});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            pod(std::uint32_t{9});
            pod(std::uint32_t{10});
            pod(std::uint64_t{item.size()});
            for (const auto entry : item)
              pod(entry);
          }
        },
        value);
  }
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    text(t.name);
    pod(static_cast<std::uint32_t>(t.dims.size()));
    for (const auto d : t.dims)
      pod(d);
    pod(static_cast<std::uint32_t>(t.type));
    pod(offset);
    offset += (EncodedBytes(t) + 31) / 32 * 32;
  }
  bytes.resize((bytes.size() + 31) / 32 * 32 + offset);
  return bytes;
}

struct Fixture {
  Metadata metadata;
  std::vector<Tensor> tensors;
};

// The tensor-free table of the pinned UD-Q4_K_XL artifact.
Fixture LoadTable() {
  std::ifstream in(QWEN36_MOE_TABLE);
  Require(in.good(), "fixture table missing");
  Fixture f;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::vector<std::string> cols;
    std::stringstream ss(line);
    for (std::string col; std::getline(ss, col, '\t');)
      cols.push_back(col);
    Require(cols.size() == 4, "malformed fixture line: " + line);
    const auto numbers = [](const std::string& csv) {
      std::vector<std::uint64_t> out;
      std::stringstream items(csv);
      for (std::string item; std::getline(items, item, ',');)
        out.push_back(std::stoull(item));
      return out;
    };
    if (cols[0] == "tensor") {
      f.tensors.push_back({cols[1], numbers(cols[3]),
                           static_cast<GgmlType>(std::stoul(cols[2]))});
      continue;
    }
    const int kind = std::stoi(cols[2]);
    if (kind == 4) {
      f.metadata[cols[1]] = static_cast<std::uint32_t>(std::stoul(cols[3]));
    } else if (kind == 5) {
      f.metadata[cols[1]] = static_cast<std::int32_t>(std::stol(cols[3]));
    } else if (kind == 6) {
      f.metadata[cols[1]] = std::stof(cols[3]);
    } else if (kind == 8) {
      f.metadata[cols[1]] = cols[3];
    } else if (kind == 9 && cols[1].ends_with("dimension_sections")) {
      f.metadata[cols[1]] = numbers(cols[3]);
    }
  }
  return f;
}

std::optional<q36::Config> Parse(const Metadata& fields,
                                 std::string* error = nullptr) {
  const auto bytes = WriteGguf(fields, {});
  std::string open_error;
  const auto reader = gufo::core::GgufReader::OpenMemory(
      bytes.data(), bytes.size(), &open_error);
  Require(reader != nullptr, "metadata-only GGUF rejected: " + open_error);
  std::string local;
  auto config = q36::Config::FromGguf(*reader, error ? error : &local);
  Require(config.has_value() == (error ? error->empty() : local.empty()),
          "error text must accompany every rejection");
  return config;
}

std::vector<q36::TensorHeader> Headers(const std::vector<Tensor>& tensors) {
  std::vector<q36::TensorHeader> table;
  for (const auto& t : tensors)
    table.push_back({t.name, t.dims, t.type});
  return table;
}

void CheckPinnedMetadata(const Fixture& f) {
  const auto c = Parse(f.metadata);
  Require(c.has_value(), "pinned qwen35moe metadata rejected");
  Require(c->num_layers == 40 && c->hidden_size == 2048 &&
              c->context_length == 262144 && c->num_heads == 16 &&
              c->num_kv_heads == 2 && c->head_dim == 256 &&
              c->rotary_dim == 64 && c->rope_theta == 1e7F &&
              c->rope_sections == std::array<std::uint32_t, 4>{11, 11, 10, 0},
          "attention metadata changed");
  Require(c->ssm_conv_kernel == 4 && c->ssm_head_dim == 128 &&
              c->ssm_num_k_heads == 16 && c->ssm_num_v_heads == 32 &&
              c->ssm_inner_size == 4096 && c->SsmConvChannels() == 8192,
          "Gated DeltaNet metadata changed");
  Require(c->num_experts == 256 && c->num_experts_used == 8 &&
              c->expert_ff == 512 && c->shared_expert_ff == 512,
          "expert metadata changed");
  Require(c->rms_eps > 9.9e-7F && c->rms_eps < 1.01e-6F, "rms epsilon");
  Require(c->IsQualifiedGeometry(), "pinned geometry not qualified");
  Require(c->IsLinearLayer(0) && !c->IsLinearLayer(3) && c->IsLinearLayer(38) &&
              !c->IsLinearLayer(39) && !c->IsLinearLayer(40),
          "layer schedule changed");

  auto trunk = c->TrunkConfig("Qwen3.6-35B-A3B");
  trunk.vocab_size = 248320;
  Require(trunk.IsValidQwen(), "trunk view fails the dense Qwen contract");
  Require(trunk.SsmQkvSize() == c->SsmConvChannels() &&
              trunk.SsmValueSize() == c->ssm_head_dim &&
              trunk.FullAttentionLayerCount() == 10 &&
              trunk.SsmLayerCount() == 30,
          "trunk view disagrees with the MoE config");

  // The dense loader must refuse this family with an actionable message
  // instead of looking for dense FFN tensors.
  const auto bytes = WriteGguf(f.metadata, {});
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size());
  std::string error;
  const bool dense = reader->ExtractModelConfig(&error).has_value();
  Require(!dense && error.find("qwen35moe") != std::string::npos,
          "dense Qwen config accepted qwen35moe: " + error);
}

void CheckMalformedMetadata(const Fixture& f) {
  const auto rejects = [&](const char* what, auto mutate,
                           const std::string& expect) {
    auto fields = f.metadata;
    mutate(fields);
    std::string error;
    Require(!Parse(fields, &error).has_value(),
            std::string(what) + " accepted");
    Require(error.find(expect) != std::string::npos,
            std::string(what) + " gave: " + error);
  };
  const std::string p = "qwen35moe.";
  rejects(
      "dense architecture",
      [](Metadata& m) { m["general.architecture"] = std::string{"qwen35"}; },
      "not qwen35moe");
  rejects(
      "embedded MTP",
      [&](Metadata& m) { m[p + "nextn_predict_layers"] = std::uint32_t{1}; },
      "MTP");
  rejects(
      "missing expert count", [&](Metadata& m) { m.erase(p + "expert_count"); },
      "expert_count");
  rejects(
      "value length",
      [&](Metadata& m) {
        m[p + "attention.value_length"] = std::uint32_t{128};
      },
      "value length");
  rejects(
      "rope sections",
      [&](Metadata& m) {
        m[p + "rope.dimension_sections"] =
            std::vector<std::uint64_t>{11, 11, 11, 0};
      },
      "sum");
  rejects(
      "interval",
      [&](Metadata& m) { m[p + "full_attention_interval"] = std::uint32_t{3}; },
      "unsupported shape");
  rejects(
      "uneven value heads",
      [&](Metadata& m) { m[p + "ssm.time_step_rank"] = std::uint32_t{24}; },
      "unsupported shape");
  rejects(
      "too many active experts",
      [&](Metadata& m) { m[p + "expert_used_count"] = std::uint32_t{257}; },
      "unsupported shape");

  auto other = f.metadata;
  other[p + "expert_count"] = std::uint32_t{128};
  const auto c = Parse(other);
  Require(c.has_value() && !c->IsQualifiedGeometry(),
          "other qwen35moe shapes must parse but stay unqualified");
}

// Every raw dimension must be positive, and every derived dimension is
// computed with checked arithmetic and bounded before it is compared.
void CheckDimensionArithmetic(const Fixture& f) {
  const std::string p = "qwen35moe.";
  const auto rejects = [&](const std::string& what, const Metadata& extra,
                           const std::string& expect) {
    auto fields = f.metadata;
    for (const auto& [key, value] : extra)
      fields[p + key] = value;
    std::string error;
    Require(!Parse(fields, &error).has_value(), what + " accepted");
    Require(error.find(expect) != std::string::npos, what + " gave: " + error);
  };
  for (const char* key :
       {"block_count", "embedding_length", "full_attention_interval",
        "attention.head_count", "attention.head_count_kv",
        "rope.dimension_count", "ssm.conv_kernel", "ssm.state_size",
        "ssm.group_count", "ssm.time_step_rank", "ssm.inner_size",
        "expert_count", "expert_used_count", "expert_feed_forward_length",
        "expert_shared_feed_forward_length"}) {
    rejects(std::string("zero ") + key, {{key, std::uint32_t{0}}}, "positive");
    rejects(std::string("oversized ") + key, {{key, std::uint32_t{1U << 31}}},
            "positive");
  }
  rejects("zero head_dim",
          {{"attention.key_length", std::uint32_t{0}},
           {"attention.value_length", std::uint32_t{0}}},
          "positive");
  rejects("zero context", {{"context_length", std::uint32_t{0}}},
          "unsupported shape");

  // ssm.time_step_rank * 128 wraps to 4096 in 32 bits and would match
  // ssm.inner_size; in 64 bits it exceeds the bound.
  rejects("SSM value dim wrapping to inner size",
          {{"ssm.time_step_rank", std::uint32_t{33554464}}}, "exceed");
  rejects("SSM key dim", {{"ssm.group_count", std::uint32_t{1U << 24}}},
          "exceed");
  rejects("SSM conv channels", {{"ssm.group_count", std::uint32_t{1U << 23}}},
          "exceed");
  rejects("attention q dim",
          {{"attention.head_count", std::uint32_t{1U << 24}}}, "exceed");
  rejects("attention q|gate dim",
          {{"attention.head_count", std::uint32_t{1U << 22}}}, "exceed");
  rejects("attention kv dim",
          {{"attention.head_count_kv", std::uint32_t{1U << 24}}}, "exceed");
  rejects("head dim product",
          {{"attention.key_length", std::uint32_t{1U << 30}},
           {"attention.value_length", std::uint32_t{1U << 30}}},
          "exceed");
}

void CheckPinnedTable(const Fixture& f) {
  auto c = *Parse(f.metadata);
  c.vocab_size = 248320;
  const auto map = q36::LoaderMap(c);
  Require(map.size() == 733 && f.tensors.size() == 733,
          "loader map does not cover the 733 pinned tensors");
  std::string error;
  const bool pinned = q36::CheckTensorTable(c, Headers(f.tensors), &error);
  Require(pinned, "pinned tensor table rejected: " + error);

  const auto rejects = [&](const char* what, auto mutate,
                           const std::string& expect) {
    auto tensors = f.tensors;
    mutate(tensors);
    std::string why;
    Require(!q36::CheckTensorTable(c, Headers(tensors), &why),
            std::string(what) + " accepted");
    Require(why.find(expect) != std::string::npos,
            std::string(what) + " gave: " + why);
  };
  const auto find = [](std::vector<Tensor>& tensors, const std::string& name) {
    for (auto& t : tensors)
      if (t.name == name)
        return &t;
    throw std::runtime_error("fixture lacks " + name);
  };
  rejects(
      "missing router",
      [](std::vector<Tensor>& t) {
        std::erase_if(t, [](const Tensor& x) {
          return x.name == "blk.7.ffn_gate_inp.weight";
        });
      },
      "missing tensor blk.7.ffn_gate_inp.weight");
  rejects(
      "unknown tensor",
      [](std::vector<Tensor>& t) {
        t.push_back(
            {"blk.0.nextn.eh_proj.weight", {4096, 2048}, GgmlType::kF32});
      },
      "unmapped tensor blk.0.nextn.eh_proj.weight");
  rejects(
      "dense FFN on a MoE layer",
      [](std::vector<Tensor>& t) {
        t.push_back({"blk.1.ffn_up.weight", {2048, 512}, GgmlType::kQ8_0});
      },
      "unmapped tensor blk.1.ffn_up.weight");
  rejects(
      "expert count",
      [&](std::vector<Tensor>& t) {
        find(t, "blk.5.ffn_down_exps.weight")->dims = {512, 2048, 255};
      },
      "expected [512, 2048, 256]");
  rejects(
      "GDN layer at an attention slot",
      [&](std::vector<Tensor>& t) {
        find(t, "blk.3.attn_q.weight")->name = "blk.3.attn_qkv.weight";
      },
      "missing tensor blk.3.attn_q.weight");
  rejects(
      "unsupported expert format",
      [&](std::vector<Tensor>& t) {
        find(t, "blk.0.ffn_up_exps.weight")->type = GgmlType::kIQ4_XS;
      },
      "unsupported format IQ4_XS");
  rejects(
      "quantized router",
      [&](std::vector<Tensor>& t) {
        find(t, "blk.0.ffn_gate_inp.weight")->type = GgmlType::kQ8_0;
      },
      "unsupported format Q8_0");

  auto tied = f.tensors;
  std::erase_if(tied,
                [](const Tensor& x) { return x.name == "output.weight"; });
  const bool tied_ok = q36::CheckTensorTable(c, Headers(tied), &error);
  Require(tied_ok, "tied LM head rejected: " + error);
}

// A miniature qwen35moe with real payload extents, bound through GgufReader.
void CheckBindSynthetic() {
  const std::string p = "qwen35moe.";
  Metadata m{{"general.architecture", std::string{"qwen35moe"}},
             {"general.name", std::string{"synthetic"}}};
  for (const auto& [key, value] :
       std::initializer_list<std::pair<const char*, std::uint32_t>>{
           {"block_count", 4},
           {"embedding_length", 256},
           {"context_length", 64},
           {"full_attention_interval", 4},
           {"attention.head_count", 2},
           {"attention.head_count_kv", 1},
           {"attention.key_length", 64},
           {"attention.value_length", 64},
           {"rope.dimension_count", 32},
           {"ssm.conv_kernel", 4},
           {"ssm.state_size", 128},
           {"ssm.group_count", 1},
           {"ssm.time_step_rank", 2},
           {"ssm.inner_size", 256},
           {"expert_count", 4},
           {"expert_used_count", 2},
           {"expert_feed_forward_length", 64},
           {"expert_shared_feed_forward_length", 64}}) {
    m[p + key] = value;
  }
  m[p + "attention.layer_norm_rms_epsilon"] = 1e-6F;
  m[p + "rope.freq_base"] = 1e7F;
  m[p + "rope.dimension_sections"] = std::vector<std::uint64_t>{6, 6, 4, 0};

  auto c = *Parse(m);
  Require(!c.IsQualifiedGeometry(), "miniature counted as qualified");
  c.vocab_size = 64;
  std::vector<Tensor> tensors;
  for (const auto& spec : q36::LoaderMap(c)) {
    if (spec.name == "output.weight")
      continue;  // tied head
    std::vector<std::uint64_t> dims{spec.cols};
    if (spec.rows != 1 || spec.experts != 1)
      dims.push_back(spec.rows);
    if (spec.experts != 1)
      dims.push_back(spec.experts);
    const bool q8 = spec.role == q36::TensorRole::kExpert ||
                    spec.name == "token_embd.weight";
    tensors.push_back({spec.name, dims, q8 ? GgmlType::kQ8_0 : GgmlType::kF32});
  }
  const auto bytes = WriteGguf(m, tensors);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Require(reader != nullptr, "synthetic GGUF rejected: " + error);
  const auto w = q36::ModelWeights::Bind(*reader, &error);
  Require(w.has_value(), "synthetic bind failed: " + error);
  Require(w->config.vocab_size == 64 && w->layers.size() == 4,
          "bound geometry");
  Require(w->output.data == w->token_embd.data, "tied head not shared");
  for (std::uint32_t il = 0; il < 4; ++il) {
    const auto& l = w->layers[il];
    const bool full = il == 3;
    Require(l.trunk.is_full_attention == full, "layer kind");
    Require(full ? !l.trunk.attn_q.empty() && l.trunk.attn_qkv.empty()
                 : !l.trunk.attn_qkv.empty() && l.trunk.attn_q.empty(),
            "layer tensors bound to the wrong kind");
    Require(l.trunk.ffn_gate.empty() && l.trunk.ffn_up.empty() &&
                l.trunk.ffn_down.empty(),
            "dense FFN slots must stay empty");
    for (const auto* ref :
         {&l.trunk.attn_norm, &l.trunk.ffn_norm, &l.router, &l.ffn_gate_exps,
          &l.ffn_up_exps, &l.ffn_down_exps, &l.shexp_gate_inp, &l.shexp_gate,
          &l.shexp_up, &l.shexp_down}) {
      Require(!ref->empty() && ref->FitsAvailableStorage(), "unbound tensor");
    }
    Require(l.ffn_down_exps.num_elements == 64ULL * 256 * 4 &&
                l.ffn_down_exps.type == GgmlType::kQ8_0,
            "expert stack extent");
  }

  // Bind runs the same table check over the reader's headers.
  tensors.push_back({"blk.0.ffn_gate.weight", {256, 64}, GgmlType::kF32});
  const auto extra = WriteGguf(m, tensors);
  const auto extra_reader =
      gufo::core::GgufReader::OpenMemory(extra.data(), extra.size(), &error);
  Require(extra_reader != nullptr, "synthetic GGUF rejected: " + error);
  const bool bound = q36::ModelWeights::Bind(*extra_reader, &error).has_value();
  Require(!bound && error == "unmapped tensor blk.0.ffn_gate.weight",
          "bind ignored an unmapped tensor: " + error);
}

}  // namespace

int main() {
  try {
    const auto fixture = LoadTable();
    CheckPinnedMetadata(fixture);
    CheckMalformedMetadata(fixture);
    CheckDimensionArithmetic(fixture);
    CheckPinnedTable(fixture);
    CheckBindSynthetic();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "qwen36_moe loader tests passed\n";
  return 0;
}
