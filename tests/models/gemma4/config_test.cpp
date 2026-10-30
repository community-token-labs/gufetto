#include "src/models/gemma4/config.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

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

/// The miniature with routed experts beside the dense MLP (26B-A4B).
Metadata MoeMetadata() {
  Metadata m = TargetMetadata();
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
  Metadata missing = MoeMetadata();
  missing.erase("gemma4.expert_feed_forward_length");
  Require(!Parse(missing).has_value(), "missing expert width");
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
  });
}
