#include "src/models/gemma4/config.hpp"

#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "src/models/gemma4/weights.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/gguf_builder.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::BoolArray;
using gemma4_test::GgufImage;
using gemma4_test::I32Array;
using gemma4_test::Metadata;
using gemma4_test::Require;
using gemma4_test::Tensors;
using gemma4_test::TensorSpec;
using gufo::core::GgmlType;

namespace {

/// A two-layer miniature of the 31B layout: one sliding and one global
/// layer, with the per-layer KV head array and rope factors of the real file.
Metadata TargetMetadata() {
  const std::string p = "gemma4.";
  return {
      {"general.architecture", std::string("gemma4")},
      {p + "block_count", std::uint32_t{2}},
      {p + "context_length", std::uint32_t{4096}},
      {p + "embedding_length", std::uint32_t{64}},
      {p + "feed_forward_length", std::uint32_t{128}},
      {p + "attention.head_count", std::uint32_t{4}},
      {p + "attention.head_count_kv", I32Array{{2, 1}}},
      {p + "attention.sliding_window_pattern", BoolArray{{true, false}}},
      {p + "attention.key_length", std::uint32_t{128}},
      {p + "attention.value_length", std::uint32_t{128}},
      {p + "attention.key_length_swa", std::uint32_t{64}},
      {p + "attention.value_length_swa", std::uint32_t{64}},
      {p + "rope.dimension_count", std::uint32_t{128}},
      {p + "rope.dimension_count_swa", std::uint32_t{64}},
      {p + "rope.freq_base", 1000000.0F},
      {p + "rope.freq_base_swa", 10000.0F},
      {p + "attention.sliding_window", std::uint32_t{16}},
      {p + "attention.layer_norm_rms_epsilon", 1e-6F},
      {p + "final_logit_softcapping", 30.0F},
      {p + "attention.shared_kv_layers", std::uint32_t{0}},
      {p + "embedding_length_per_layer_input", std::uint32_t{0}},
  };
}

Metadata DraftMetadata() {
  const std::string p = "gemma4-assistant.";
  return {
      {"general.architecture", std::string("gemma4-assistant")},
      {p + "block_count", std::uint32_t{2}},
      {p + "context_length", std::uint32_t{4096}},
      {p + "embedding_length", std::uint32_t{32}},
      {p + "feed_forward_length", std::uint32_t{64}},
      {p + "attention.head_count", std::uint32_t{4}},
      {p + "attention.head_count_kv", I32Array{{2, 1}}},
      {p + "attention.sliding_window_pattern", BoolArray{{true, false}}},
      {p + "attention.key_length", std::uint32_t{128}},
      {p + "attention.value_length", std::uint32_t{128}},
      {p + "attention.key_length_swa", std::uint32_t{64}},
      {p + "attention.value_length_swa", std::uint32_t{64}},
      {p + "rope.dimension_count", std::uint32_t{128}},
      {p + "rope.dimension_count_swa", std::uint32_t{64}},
      {p + "rope.freq_base", 1000000.0F},
      {p + "rope.freq_base_swa", 10000.0F},
      {p + "attention.sliding_window", std::uint32_t{16}},
      {p + "attention.layer_norm_rms_epsilon", 1e-6F},
      {p + "attention.shared_kv_layers", std::uint32_t{2}},
      {p + "embedding_length_out", std::uint32_t{64}},
      {p + "nextn_predict_layers", std::uint32_t{2}},
  };
}

std::optional<g4::Config> Parse(const Metadata& metadata) {
  GgufImage image(metadata);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  std::string error;
  auto config = g4::Config::FromGguf(*image.reader(), &error);
  Require(config.has_value() == error.empty(),
          "error message must accompany every rejection");
  return config;
}

void CheckTarget() {
  const auto c = Parse(TargetMetadata());
  Require(c.has_value(), "valid gemma4 metadata rejected");
  Require(c->num_layers == 2 && c->hidden_size == 64 && c->ffn_size == 128,
          "dimensions");
  Require(c->IsSliding(0) && !c->IsSliding(1), "sliding pattern");
  Require(c->HeadDim(0) == 64 && c->HeadDim(1) == 128, "per-layer head dims");
  Require(c->kv_heads[0] == 2 && c->kv_heads[1] == 1, "per-layer KV heads");
  Require(c->QDim(1) == 512 && c->KvDim(1) == 128, "projection widths");
  Require(c->RopeTheta(0) == 10000.0F && c->RopeTheta(1) == 1000000.0F,
          "per-layer rope bases");
  Require(c->sliding_window == 16 && c->final_logit_softcap == 30.0F,
          "window and softcap");
  Require(c->GlobalLayerCount() == 1, "global layer count");
}

void CheckRejections() {
  const Metadata valid = TargetMetadata();
  const auto reject = [&](const std::string& key, gemma4_test::MetaValue value,
                          const char* what) {
    Metadata wrong = valid;
    wrong[key] = std::move(value);
    Require(!Parse(wrong).has_value(), what);
  };
  reject("general.architecture", std::string("qwen35"), "other architecture");
  reject("gemma4.attention.head_count_kv", I32Array{{2}}, "short KV array");
  reject("gemma4.attention.head_count_kv", I32Array{{3, 1}},
         "KV heads not dividing query heads");
  reject("gemma4.attention.head_count_kv", I32Array{{-2, 1}},
         "negative KV heads");
  // Scalars are validated like array entries; zero KV heads used to reach a
  // modulo by zero.
  reject("gemma4.attention.head_count_kv", std::uint32_t{0},
         "scalar zero KV heads");
  reject("gemma4.attention.head_count_kv", I32Array{{2, 0}},
         "zero KV heads in one layer");
  reject("gemma4.feed_forward_length", std::uint32_t{0}, "scalar zero FFN");
  {
    // The cap applies before any per-layer allocation, so a huge count with
    // scalar per-layer values fails fast with its own message.
    Metadata huge = valid;
    huge["gemma4.block_count"] = std::uint32_t{0xffffffffU};
    huge["gemma4.attention.head_count_kv"] = std::uint32_t{1};
    GgufImage image(huge);
    std::string error;
    Require(!g4::Config::FromGguf(*image.reader(), &error) &&
                error.find("block_count exceeds") != std::string::npos,
            "layer cap: " + error);
  }
  // 2^25 heads of 256 overflow a 32-bit query width.
  reject("gemma4.attention.head_count", std::uint32_t{1} << 25,
         "query width overflow");
  reject("gemma4.feed_forward_length", I32Array{{128, 0}},
         "zero FFN in one layer");
  reject("gemma4.attention.sliding_window_pattern", BoolArray{{true}},
         "short pattern");
  reject("gemma4.attention.sliding_window_pattern", BoolArray{{true, true}},
         "no global layer");
  reject("gemma4.attention.value_length", std::uint32_t{64},
         "key and value widths differ");
  reject("gemma4.attention.key_length_swa", std::uint32_t{48},
         "head dimension not a multiple of 64");
  reject("gemma4.rope.dimension_count", std::uint32_t{64}, "partial rope");
  reject("gemma4.embedding_length_per_layer_input", std::uint32_t{256},
         "per-layer embeddings");
  reject("gemma4.attention.shared_kv_layers", std::uint32_t{1},
         "KV-shared target layers");
  reject("gemma4.attention.sliding_window", std::uint32_t{0}, "zero window");
  reject("gemma4.rope.freq_base", -1.0F, "negative rope base");
  for (const char* key :
       {"gemma4.block_count", "gemma4.attention.head_count_kv",
        "gemma4.attention.sliding_window_pattern",
        "gemma4.attention.key_length_swa", "gemma4.rope.freq_base_swa",
        "gemma4.attention.layer_norm_rms_epsilon"}) {
    Metadata wrong = valid;
    wrong.erase(key);
    Require(!Parse(wrong).has_value(), std::string("missing ") + key);
  }
  // A scalar KV head count applies to every layer.
  Metadata scalar = valid;
  scalar["gemma4.attention.head_count_kv"] = std::uint32_t{1};
  const auto c = Parse(scalar);
  Require(c && c->kv_heads[0] == 1 && c->kv_heads[1] == 1,
          "scalar KV head count");
}

/// The miniature with routed experts beside the dense MLP (26B-A4B). Its
/// hidden width is 256 so that the K-quant gate/up experts can bind.
Metadata MoeMetadata() {
  Metadata m = TargetMetadata();
  m["gemma4.embedding_length"] = std::uint32_t{256};
  m["gemma4.expert_count"] = std::uint32_t{8};
  m["gemma4.expert_used_count"] = std::uint32_t{2};
  m["gemma4.expert_feed_forward_length"] = std::uint32_t{32};
  return m;
}

void CheckExpertConfig() {
  const auto c = Parse(MoeMetadata());
  Require(c && c->HasExperts() && c->num_experts == 8 && c->experts_used == 2 &&
              c->expert_ffn_size == 32,
          "expert metadata");
  Require(!Parse(TargetMetadata())->HasExperts(), "dense target has experts");
  const auto reject = [&](const std::string& key, gemma4_test::MetaValue value,
                          const char* what) {
    Metadata wrong = MoeMetadata();
    wrong[key] = std::move(value);
    Require(!Parse(wrong).has_value(), what);
  };
  reject("gemma4.expert_used_count", std::uint32_t{9},
         "more routed experts than the kernels support");
  reject("gemma4.expert_used_count", std::uint32_t{0}, "no routed experts");
  reject("gemma4.expert_count", std::uint32_t{512}, "too many experts");
  reject("gemma4.expert_feed_forward_length", std::uint32_t{0},
         "scalar zero expert width");
  reject("gemma4.expert_feed_forward_length", I32Array{{32, 0}},
         "zero expert width in one layer");
  Metadata missing = MoeMetadata();
  missing.erase("gemma4.expert_feed_forward_length");
  Require(!Parse(missing).has_value(), "missing expert width");
}

std::vector<std::uint8_t> Zeros(std::size_t n) {
  return std::vector<std::uint8_t>(n, 0);
}

/// Q8_0 bytes for a [cols, rows] matrix.
TensorSpec Q8(std::uint64_t cols, std::uint64_t rows) {
  return {{cols, rows}, GgmlType::kQ8_0, Zeros(cols / 32 * 34 * rows)};
}
/// Q4_K bytes for a stack of [cols, rows] matrices.
TensorSpec Q4K(std::uint64_t cols, std::uint64_t rows, std::uint64_t experts) {
  return {{cols, rows, experts},
          GgmlType::kQ4_K,
          Zeros(cols / 256 * 144 * rows * experts)};
}
TensorSpec F32(std::uint64_t n, float value = 1.0F) {
  std::vector<std::uint8_t> data(n * 4);
  for (std::uint64_t i = 0; i < n; ++i) {
    std::memcpy(data.data() + i * 4, &value, 4);
  }
  return {{n}, GgmlType::kF32, data};
}

Tensors TargetTensors(const g4::Config& c, std::uint64_t vocab) {
  Tensors t;
  t["token_embd.weight"] = Q8(c.hidden_size, vocab);
  t["output_norm.weight"] = F32(c.hidden_size);
  t["rope_freqs.weight"] = F32(c.head_dim_global / 2);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    t[p + "attn_norm.weight"] = F32(c.hidden_size);
    t[p + "attn_q.weight"] = Q8(c.hidden_size, c.QDim(l));
    t[p + "attn_k.weight"] = Q8(c.hidden_size, c.KvDim(l));
    if (c.IsSliding(l)) {
      t[p + "attn_v.weight"] = Q8(c.hidden_size, c.KvDim(l));
    }
    t[p + "attn_q_norm.weight"] = F32(c.HeadDim(l));
    t[p + "attn_k_norm.weight"] = F32(c.HeadDim(l));
    t[p + "attn_output.weight"] = Q8(c.QDim(l), c.hidden_size);
    t[p + "post_attention_norm.weight"] = F32(c.hidden_size);
    t[p + "ffn_norm.weight"] = F32(c.hidden_size);
    t[p + "ffn_gate.weight"] = Q8(c.hidden_size, c.ffn_size);
    t[p + "ffn_up.weight"] = Q8(c.hidden_size, c.ffn_size);
    t[p + "ffn_down.weight"] = Q8(c.ffn_size, c.hidden_size);
    t[p + "post_ffw_norm.weight"] = F32(c.hidden_size);
    t[p + "layer_output_scale.weight"] = F32(1, 0.5F + static_cast<float>(l));
  }
  return t;
}

/// Adds the router, norms and stacked experts of every layer.
void AddExperts(const g4::Config& c, Tensors* t) {
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    (*t)[p + "ffn_gate_inp.weight"] = {
        {c.hidden_size, c.num_experts},
        GgmlType::kF32,
        Zeros(c.hidden_size * c.num_experts * 4)};
    (*t)[p + "ffn_gate_inp.scale"] = F32(c.hidden_size);
    (*t)[p + "pre_ffw_norm_2.weight"] = F32(c.hidden_size);
    (*t)[p + "post_ffw_norm_1.weight"] = F32(c.hidden_size);
    (*t)[p + "post_ffw_norm_2.weight"] = F32(c.hidden_size);
    (*t)[p + "ffn_gate_up_exps.weight"] =
        Q4K(c.hidden_size, 2 * c.expert_ffn_size, c.num_experts);
    (*t)[p + "ffn_down_exps.weight"] = {
        {c.expert_ffn_size, c.hidden_size, c.num_experts},
        GgmlType::kQ5_1,
        Zeros(c.expert_ffn_size / 32 * 24 * c.hidden_size * c.num_experts)};
    (*t)[p + "ffn_down_exps.scale"] = F32(c.num_experts);
  }
}

void CheckExpertWeights() {
  const auto c = Parse(MoeMetadata());
  Tensors tensors = TargetTensors(*c, 96);
  AddExperts(*c, &tensors);
  std::string error;
  {
    GgufImage image(MoeMetadata(), tensors);
    const auto w = g4::ModelWeights::Bind(*image.reader(), &error);
    Require(w.has_value(), "valid expert weights rejected: " + error);
    const auto& l = w->layers[1];
    Require(l.gate_up_exps.experts == 8 && l.gate_up_exps.rows == 64 &&
                l.down_exps.type == GgmlType::kQ5_1 &&
                l.down_exps.SizeBytes() == 32 / 32 * 24 * 256 * 8,
            "stacked expert tensors");
  }
  const auto rejects = [&](const Tensors& wrong, const Metadata& meta,
                           const char* what) {
    GgufImage image(meta, wrong);
    Require(!g4::ModelWeights::Bind(*image.reader(), &error) && !error.empty(),
            what);
  };
  auto wrong = tensors;
  wrong.erase("blk.0.ffn_gate_up_exps.weight");
  rejects(wrong, MoeMetadata(), "separate gate/up experts accepted");
  wrong = tensors;
  wrong["blk.0.ffn_down_exps.weight"].dims[2] = 7;
  rejects(wrong, MoeMetadata(), "short expert stack accepted");
  wrong = tensors;
  wrong["blk.1.ffn_gate_up_exps.weight"].type = GgmlType::kQ5_1;
  rejects(wrong, MoeMetadata(), "unsupported expert format accepted");
  wrong = tensors;
  wrong["blk.1.ffn_gate_up_exps.weight"] = {
      {256, 64, 8}, GgmlType::kQ8_0, Zeros(256 / 32 * 34 * 64 * 8)};
  rejects(wrong, MoeMetadata(), "Q8_0 gate/up experts accepted");
  rejects(tensors, TargetMetadata(), "experts in a dense model accepted");
}

void CheckWeights() {
  const auto c = Parse(TargetMetadata());
  Tensors tensors = TargetTensors(*c, 96);
  {
    GgufImage image(TargetMetadata(), tensors);
    std::string error;
    auto w = g4::ModelWeights::Bind(*image.reader(), &error);
    Require(w.has_value(), "valid weights rejected: " + error);
    Require(w->vocab_size == 96 && w->TiedOutput(), "tied vocabulary head");
    Require(w->layers[1].output_scale == 1.0F && w->rope_factors.values.empty(),
            "Bind reads no payload");
    Require(w->ReadHostValues(&error), "host values rejected: " + error);
    Require(
        w->layers[0].output_scale == 0.5F && w->layers[1].output_scale == 1.5F,
        "layer output scales");
    Require(!w->layers[0].attn_v.empty() && w->layers[1].attn_v.empty(),
            "global layer uses K as V");
    Require(w->rope_factors.values.size() == 64, "rope factors");

    // The draft reads target KV and embeddings; its own head is separate.
    Tensors dt;
    const auto dc_meta = DraftMetadata();
    dt["token_embd.weight"] = Q8(32, 96);
    dt["output_norm.weight"] = F32(32);
    dt["rope_freqs.weight"] = F32(64);
    dt["nextn.pre_projection.weight"] = Q8(128, 32);
    dt["nextn.post_projection.weight"] = Q8(32, 64);
    for (std::uint32_t l = 0; l < 2; ++l) {
      const std::string p = "blk." + std::to_string(l) + ".";
      const std::uint64_t hd = l == 0 ? 64 : 128;
      dt[p + "attn_norm.weight"] = F32(32);
      dt[p + "attn_q.weight"] = Q8(32, 4 * hd);
      dt[p + "attn_q_norm.weight"] = F32(hd);
      dt[p + "attn_output.weight"] = Q8(4 * hd, 32);
      dt[p + "post_attention_norm.weight"] = F32(32);
      dt[p + "ffn_norm.weight"] = F32(32);
      dt[p + "ffn_gate.weight"] = Q8(32, 64);
      dt[p + "ffn_up.weight"] = Q8(32, 64);
      dt[p + "ffn_down.weight"] = Q8(64, 32);
      dt[p + "post_ffw_norm.weight"] = F32(32);
      dt[p + "layer_output_scale.weight"] = F32(1, 0.25F);
    }
    GgufImage draft(dc_meta, dt);
    auto d = g4::DraftWeights::Bind(*draft.reader(), *w, &error);
    Require(d.has_value(), "valid draft rejected: " + error);
    Require(d->ReadHostValues(&error) && d->layers[1].output_scale == 0.25F,
            "draft host values");
    Require(d->config.SharedKvSource(0, w->config) == 0 &&
                d->config.SharedKvSource(1, w->config) == 1,
            "draft KV sources");

    auto wrong_vocab = dt;
    wrong_vocab["token_embd.weight"] = Q8(32, 95);
    GgufImage bad_draft(dc_meta, wrong_vocab);
    Require(!g4::DraftWeights::Bind(*bad_draft.reader(), *w, &error),
            "draft with another vocabulary accepted");
    auto with_k = dt;
    with_k["blk.0.attn_k.weight"] = Q8(32, 128);
    GgufImage kv_draft(dc_meta, with_k);
    Require(!g4::DraftWeights::Bind(*kv_draft.reader(), *w, &error),
            "draft layer with its own K accepted");
    auto extra = dt;
    extra["blk.0.ffn_gate_inp.scale"] = F32(32);
    GgufImage extra_draft(dc_meta, extra);
    Require(!g4::DraftWeights::Bind(*extra_draft.reader(), *w, &error) &&
                error.find("unmapped") != std::string::npos,
            "draft with an unmapped tensor accepted");
  }
  const auto rejects = [&](Tensors wrong, const char* what) {
    GgufImage image(TargetMetadata(), wrong);
    std::string error;
    Require(!g4::ModelWeights::Bind(*image.reader(), &error) && !error.empty(),
            what);
  };
  auto wrong = tensors;
  wrong.erase("blk.0.attn_v.weight");
  rejects(wrong, "sliding layer without V accepted");
  wrong = tensors;
  wrong["blk.1.attn_q.weight"] = Q8(64, 256);
  rejects(wrong, "wrong Q width accepted");
  wrong = tensors;
  wrong["blk.0.ffn_up.weight"].type = GgmlType::kQ4_1;
  rejects(wrong, "unsupported projection format accepted");
  wrong = tensors;
  wrong["blk.0.ffn_down.weight"].type = GgmlType::kQ5_1;
  rejects(wrong, "expert-only format in a dense projection accepted");
  wrong = tensors;
  wrong["blk.1.attn_extra.weight"] = F32(64);
  rejects(wrong, "unmapped tensor accepted");
  {
    // Bad payloads pass the header-only bind and fail on the host read.
    wrong = tensors;
    wrong["rope_freqs.weight"] = F32(64, -1.0F);
    GgufImage image(TargetMetadata(), wrong);
    std::string error;
    auto w = g4::ModelWeights::Bind(*image.reader(), &error);
    Require(w.has_value(), "header-only bind read rope payload: " + error);
    Require(!w->ReadHostValues(&error) && !error.empty(),
            "non-positive rope factors accepted");
  }
  wrong = tensors;
  wrong.erase("output_norm.weight");
  rejects(wrong, "missing output norm accepted");
}

/// Expert tensors in a file without gemma4.expert_count are not a dense
/// target.
void CheckDenseExpertTensors() {
  std::vector<std::uint8_t> zeros(64 * 8 * 4, 0);
  Tensors tensors;
  tensors["blk.0.ffn_gate_inp.weight"] =
      TensorSpec{{64, 8}, GgmlType::kF32, std::move(zeros)};
  GgufImage image(TargetMetadata(), tensors);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  std::string error;
  Require(!g4::Config::FromGguf(*image.reader(), &error) && !error.empty(),
          "router tensors without expert metadata accepted");
}

std::optional<g4::Config> ParseDraft(const Metadata& metadata,
                                     const g4::Config& target) {
  GgufImage image(metadata);
  Require(image.reader() != nullptr,
          "in-memory GGUF rejected: " + image.error());
  std::string error;
  auto config = g4::Config::DraftFromGguf(*image.reader(), target, &error);
  Require(config.has_value() == error.empty(),
          "error message must accompany every rejection");
  return config;
}

void CheckDraft() {
  const auto target = Parse(TargetMetadata());
  Require(target.has_value(), "valid gemma4 metadata rejected");
  const auto d = ParseDraft(DraftMetadata(), *target);
  Require(d.has_value(), "valid gemma4-assistant metadata rejected");
  Require(d->num_layers == 2 && d->hidden_size == 32 &&
              d->target_hidden_size == 64 && d->shared_kv_layers == 2,
          "draft dimensions");
  Require(
      d->SharedKvSource(0, *target) == 0 && d->SharedKvSource(1, *target) == 1,
      "draft KV sources");

  // Each draft layer reads the last target layer of its attention kind.
  Metadata wide = TargetMetadata();
  wide["gemma4.block_count"] = std::uint32_t{3};
  wide["gemma4.attention.head_count_kv"] = I32Array{{2, 1, 2}};
  wide["gemma4.attention.sliding_window_pattern"] =
      BoolArray{{true, false, true}};
  const auto wide_target = Parse(wide);
  Require(wide_target.has_value(), "three-layer target rejected");
  const auto wide_draft = ParseDraft(DraftMetadata(), *wide_target);
  Require(wide_draft && wide_draft->SharedKvSource(0, *wide_target) == 2 &&
              wide_draft->SharedKvSource(1, *wide_target) == 1,
          "draft KV sources are the last target layer of each kind");

  const auto reject = [&](const std::string& key, gemma4_test::MetaValue value,
                          const char* what) {
    Metadata wrong = DraftMetadata();
    wrong[key] = std::move(value);
    Require(!ParseDraft(wrong, *target).has_value(), what);
  };
  const std::string p = "gemma4-assistant.";
  reject("general.architecture", std::string("gemma4"), "target as draft");
  reject(p + "embedding_length_out", std::uint32_t{32},
         "draft output width differs from the target");
  reject(p + "attention.shared_kv_layers", std::uint32_t{1},
         "draft layer with its own KV");
  reject(p + "nextn_predict_layers", std::uint32_t{1},
         "draft predicting fewer layers than it has");
  reject(p + "attention.head_count_kv", I32Array{{1, 1}},
         "draft KV heads differ from the source layer");
  reject(p + "rope.freq_base_swa", 20000.0F, "draft rope base differs");
  reject(p + "attention.sliding_window", std::uint32_t{32},
         "draft window differs");
  reject(p + "block_count", std::uint32_t{513}, "draft with too many layers");
  reject(p + "attention.head_count", std::uint32_t{1} << 25,
         "draft query width overflow");
  reject(p + "attention.head_count_kv", std::uint32_t{0},
         "draft scalar zero KV heads");
  reject(p + "attention.head_count_kv", I32Array{{2, 0}},
         "draft zero KV heads in one layer");
  reject(p + "feed_forward_length", std::uint32_t{0}, "draft scalar zero FFN");
  reject(p + "feed_forward_length", I32Array{{64, 0}},
         "draft zero FFN in one layer");
  reject(p + "expert_feed_forward_length", std::uint32_t{0},
         "draft zero expert width");
  reject(p + "expert_feed_forward_length", I32Array{{32, 0}},
         "draft zero expert width in one layer");
  reject(p + "expert_count", std::uint32_t{8}, "draft with routed experts");
  Metadata missing = DraftMetadata();
  missing.erase(p + "embedding_length_out");
  Require(!ParseDraft(missing, *target).has_value(),
          "missing draft output width");
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckTarget();
    CheckRejections();
    CheckExpertConfig();
    CheckDenseExpertTensors();
    CheckDraft();
    CheckWeights();
    CheckExpertWeights();
  });
}
