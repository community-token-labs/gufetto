// Reads the published 26B-A4B target and its MTP drafter (unsloth
// gemma-4-26B-A4B-it-GGUF c099eb48) through the production config and weight
// binder. Only the GGUF headers are parsed; no tensor data is touched. Skips
// (77) unless GUFO_GEMMA4_MODEL and GUFO_GEMMA4_MTP_MODEL name the two files.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/gemma4/config.hpp"
#include "src/models/gemma4/weights.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;

namespace {

std::unique_ptr<gufo::core::GgufReader> Open(const char* path) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(reader != nullptr, std::string(path) + ": " + error);
  return reader;
}

std::uint32_t Token(const gufo::core::GgufReader& gguf, const char* key) {
  const auto value = gguf.GetMetadataUint32(key);
  Require(value.has_value(), std::string("missing ") + key);
  return *value;
}

void CheckTarget(const gufo::core::GgufReader& gguf, const g4::Config& c) {
  Require(gguf.GetTensorCount() == 658, "target tensor count");
  Require(c.num_layers == 30 && c.hidden_size == 2816 && c.num_heads == 16 &&
              c.ffn_size == 2112 && c.context_length == 262144,
          "target dimensions");
  std::vector<std::uint32_t> global;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    if (!c.IsSliding(l)) {
      global.push_back(l);
    }
    Require(c.kv_heads[l] == (c.IsSliding(l) ? 8U : 2U),
            "KV heads in layer " + std::to_string(l));
  }
  Require(global == std::vector<std::uint32_t>{5, 11, 17, 23, 29},
          "global layers are 5, 11, 17, 23 and 29");
  Require(c.head_dim_sliding == 256 && c.head_dim_global == 512 &&
              c.rope_theta_sliding == 1e4F && c.rope_theta_global == 1e6F,
          "per-kind head dims and rope bases");
  Require(c.sliding_window == 1024 && c.final_logit_softcap == 30.0F &&
              c.rms_eps == 1e-6F,
          "window, softcap and norm epsilon");
  Require(
      c.num_experts == 128 && c.experts_used == 8 && c.expert_ffn_size == 704,
      "routed experts");
  Require(Token(gguf, "tokenizer.ggml.bos_token_id") == 2 &&
              Token(gguf, "tokenizer.ggml.eos_token_id") == 106,
          "target bos 2 and eos 106 (<turn|>)");
}

void CheckDraft(const gufo::core::GgufReader& gguf, const g4::Config& d,
                const g4::Config& target) {
  Require(gguf.GetTensorCount() == 49, "draft tensor count");
  Require(d.num_layers == 4 && d.hidden_size == 1024 && d.ffn_size == 8192 &&
              d.target_hidden_size == 2816 && d.shared_kv_layers == 4 &&
              !d.HasExperts(),
          "draft dimensions");
  Require(d.IsSliding(0) && d.IsSliding(1) && d.IsSliding(2) && !d.IsSliding(3),
          "draft pattern is three sliding layers, then one global");
  std::vector<std::uint32_t> sources;
  for (std::uint32_t l = 0; l < d.num_layers; ++l) {
    sources.push_back(d.SharedKvSource(l, target));
  }
  Require(sources == std::vector<std::uint32_t>{28, 28, 28, 29},
          "draft reads target layers 28 (sliding) and 29 (global)");
  Require(Token(gguf, "tokenizer.ggml.eos_token_id") == 1,
          "draft eos 1, unlike the target's 106");
}

/// Every tensor binds; global layers take V from K; the routed experts use
/// the artifact's mixed quantization.
void CheckTargetWeights(const g4::ModelWeights& w) {
  using gufo::core::GgmlType;
  Require(w.vocab_size == 262144 && w.TiedOutput() &&
              w.token_embd.type == GgmlType::kQ8_0,
          "tied Q8_0 vocabulary head");
  Require(w.rope_freqs.cols == 256, "rope divisors");
  std::uint32_t q4k = 0;
  std::uint32_t q5k = 0;
  std::uint32_t q5_1 = 0;
  std::uint32_t q8_0 = 0;
  for (std::uint32_t l = 0; l < w.layers.size(); ++l) {
    const auto& layer = w.layers[l];
    Require(layer.attn_v.empty() == !w.config.IsSliding(l),
            "attn_v only on sliding layers, layer " + std::to_string(l));
    Require(!layer.layer_output_scale.empty() && !layer.router.empty() &&
                layer.gate_up_exps.experts == 128 &&
                layer.gate_up_exps.rows == 1408 && layer.down_exps.cols == 704,
            "expert layout in layer " + std::to_string(l));
    q4k += layer.gate_up_exps.type == GgmlType::kQ4_K ? 1 : 0;
    q5k += layer.gate_up_exps.type == GgmlType::kQ5_K ? 1 : 0;
    q5_1 += layer.down_exps.type == GgmlType::kQ5_1 ? 1 : 0;
    q8_0 += layer.down_exps.type == GgmlType::kQ8_0 ? 1 : 0;
  }
  Require(q4k == 29 && q5k == 1 && q5_1 == 29 && q8_0 == 1,
          "routed expert quantization mix");
  Require(w.layers[29].gate_up_exps.type == GgmlType::kQ5_K,
          "layer 29 gate/up experts are Q5_K");
}

void CheckDraftWeights(const g4::DraftWeights& d) {
  Require(d.pre_projection.cols == 5632 && d.pre_projection.rows == 1024 &&
              d.post_projection.cols == 1024 &&
              d.post_projection.rows == 2816 && d.token_embd.rows == 262144,
          "draft projections and head");
  for (const auto& layer : d.layers) {
    Require(layer.attn_k.empty() && layer.attn_v.empty() &&
                !layer.attn_q.empty() && layer.router.empty(),
            "draft layers are Q-only and dense");
  }
}

}  // namespace

int main() {
  const char* model = std::getenv("GUFO_GEMMA4_MODEL");
  const char* mtp = std::getenv("GUFO_GEMMA4_MTP_MODEL");
  if (model == nullptr || mtp == nullptr || !std::filesystem::exists(model) ||
      !std::filesystem::exists(mtp)) {
    std::cout << "SKIP: set GUFO_GEMMA4_MODEL and GUFO_GEMMA4_MTP_MODEL\n";
    return 77;
  }
  return gemma4_test::Run([&] {
    const auto target_gguf = Open(model);
    std::string error;
    const auto target = g4::Config::FromGguf(*target_gguf, &error);
    Require(target.has_value(), "target config rejected: " + error);
    CheckTarget(*target_gguf, *target);

    const auto draft_gguf = Open(mtp);
    const auto draft = g4::Config::DraftFromGguf(*draft_gguf, *target, &error);
    Require(draft.has_value(), "draft config rejected: " + error);
    CheckDraft(*draft_gguf, *draft, *target);

    const auto weights = g4::ModelWeights::Bind(*target_gguf, &error);
    Require(weights.has_value(), "target weights rejected: " + error);
    CheckTargetWeights(*weights);
    const auto draft_weights =
        g4::DraftWeights::Bind(*draft_gguf, *weights, &error);
    Require(draft_weights.has_value(), "draft weights rejected: " + error);
    CheckDraftWeights(*draft_weights);
  });
}
