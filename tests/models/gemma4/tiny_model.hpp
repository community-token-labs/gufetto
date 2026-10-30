#ifndef GUFO_TESTS_MODELS_GEMMA4_TINY_MODEL_HPP_
#define GUFO_TESTS_MODELS_GEMMA4_TINY_MODEL_HPP_

// A tiny 26B-A4B-shaped gemma4 target with deterministic pseudo-random
// weights. tools/gemma4/reference_oracle.py builds the same bytes from this
// specification and computes the expected logits independently, so keep the
// two in step:
//
// - 3 layers: sliding, sliding, global. hidden 256, 2 query heads, KV heads
//   {1, 2, 1}, head dims 64 (sliding) and 128 (global), window 4, dense FFN
//   64, 4 experts with 2 used and expert FFN 32, vocabulary 32, eps 1e-6,
//   softcap 30, rope bases 1e4 (sliding) and 1e6 (global).
// - rope_freqs [64]: 1.0 for the first 16 pairs, 1e30 for the rest.
// - Every other tensor's bytes come from splitmix64 seeded with the FNV-1a
//   64-bit hash of its name, one draw per field:
//   * F32 norm/scale vectors and layer_output_scale: 0.5 + u, u = (r >> 40)
//     / 2^24; the F32 router weight: 2u - 1.
//   * Q8_0 blocks: d = binary16 bits 0x2000 + r % 0x400, then 32 int8 values
//     r % 255 - 127.
//   * Q4_K blocks: d and dmin = binary16 0x1C00 + r % 0x400, then 12 scale
//     bytes and 128 quant bytes, each r & 0xFF.
//   * Q5_1 blocks: d = binary16 0x2000 + r % 0x400, m = binary16 0xA000 +
//     r % 0x400 (negative), then 4 qh bytes and 16 quant bytes, each r & 0xFF.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "tests/models/gemma4/gguf_builder.hpp"

namespace gemma4_test {

class SplitMix64 {
public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
  std::uint64_t Next() {
    std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  float Unit() { return static_cast<float>(Next() >> 40) / 16777216.0F; }

private:
  std::uint64_t state_;
};

inline std::uint64_t Fnv1a(const std::string& text) {
  std::uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char c : text) {
    h = (h ^ c) * 0x100000001B3ULL;
  }
  return h;
}

namespace tiny {

inline void Put16(std::vector<std::uint8_t>* out, std::uint16_t v) {
  out->push_back(static_cast<std::uint8_t>(v));
  out->push_back(static_cast<std::uint8_t>(v >> 8));
}

inline void PutF32(std::vector<std::uint8_t>* out, float v) {
  std::uint8_t bytes[4];
  std::memcpy(bytes, &v, 4);
  out->insert(out->end(), bytes, bytes + 4);
}

inline TensorSpec Vector(const std::string& name, std::uint64_t n) {
  SplitMix64 rng(Fnv1a(name));
  TensorSpec t{{n}, gufo::core::GgmlType::kF32, {}};
  for (std::uint64_t i = 0; i < n; ++i) {
    PutF32(&t.data, 0.5F + rng.Unit());
  }
  return t;
}

/// [cols, rows, experts] in a quantized format.
inline TensorSpec Matrix(const std::string& name, gufo::core::GgmlType type,
                         std::uint64_t cols, std::uint64_t rows,
                         std::uint64_t experts = 1) {
  using gufo::core::GgmlType;
  SplitMix64 rng(Fnv1a(name));
  TensorSpec t{{cols, rows}, type, {}};
  if (experts > 1) {
    t.dims.push_back(experts);
  }
  const std::uint64_t count = cols * rows * experts;
  const auto bytes = [&](int n) {
    for (int i = 0; i < n; ++i) {
      t.data.push_back(static_cast<std::uint8_t>(rng.Next() & 0xFF));
    }
  };
  if (type == GgmlType::kF32) {
    for (std::uint64_t i = 0; i < count; ++i) {
      PutF32(&t.data, 2.0F * rng.Unit() - 1.0F);
    }
  } else if (type == GgmlType::kQ8_0) {
    for (std::uint64_t b = 0; b < count / 32; ++b) {
      Put16(&t.data, static_cast<std::uint16_t>(0x2000 + rng.Next() % 0x400));
      for (int i = 0; i < 32; ++i) {
        t.data.push_back(static_cast<std::uint8_t>(
            static_cast<int>(rng.Next() % 255) - 127));
      }
    }
  } else if (type == GgmlType::kQ4_K) {
    for (std::uint64_t b = 0; b < count / 256; ++b) {
      Put16(&t.data, static_cast<std::uint16_t>(0x1C00 + rng.Next() % 0x400));
      Put16(&t.data, static_cast<std::uint16_t>(0x1C00 + rng.Next() % 0x400));
      bytes(12 + 128);
    }
  } else {  // kQ5_1
    for (std::uint64_t b = 0; b < count / 32; ++b) {
      Put16(&t.data, static_cast<std::uint16_t>(0x2000 + rng.Next() % 0x400));
      Put16(&t.data, static_cast<std::uint16_t>(0xA000 + rng.Next() % 0x400));
      bytes(4 + 16);
    }
  }
  return t;
}

inline Metadata TinyMetadata() {
  const std::string p = "gemma4.";
  return {
      {"general.architecture", std::string("gemma4")},
      {p + "block_count", std::uint32_t{3}},
      {p + "context_length", std::uint32_t{64}},
      {p + "embedding_length", std::uint32_t{256}},
      {p + "feed_forward_length", std::uint32_t{64}},
      {p + "attention.head_count", std::uint32_t{2}},
      {p + "attention.head_count_kv", I32Array{{1, 2, 1}}},
      {p + "attention.sliding_window_pattern", BoolArray{{true, true, false}}},
      {p + "attention.key_length", std::uint32_t{128}},
      {p + "attention.value_length", std::uint32_t{128}},
      {p + "attention.key_length_swa", std::uint32_t{64}},
      {p + "attention.value_length_swa", std::uint32_t{64}},
      {p + "rope.dimension_count", std::uint32_t{128}},
      {p + "rope.dimension_count_swa", std::uint32_t{64}},
      {p + "rope.freq_base", 1000000.0F},
      {p + "rope.freq_base_swa", 10000.0F},
      {p + "attention.sliding_window", std::uint32_t{4}},
      {p + "attention.layer_norm_rms_epsilon", 1e-6F},
      {p + "final_logit_softcapping", 30.0F},
      {p + "expert_count", std::uint32_t{4}},
      {p + "expert_used_count", std::uint32_t{2}},
      {p + "expert_feed_forward_length", std::uint32_t{32}},
  };
}

inline Tensors TinyTensors() {
  using gufo::core::GgmlType;
  constexpr std::uint64_t kHidden = 256, kFfn = 64, kExperts = 4,
                          kExpertFfn = 32, kVocab = 32, kHeads = 2;
  const std::uint64_t kv_heads[3] = {1, 2, 1};
  Tensors t;
  t["token_embd.weight"] =
      Matrix("token_embd.weight", GgmlType::kQ8_0, kHidden, kVocab);
  t["output_norm.weight"] = Vector("output_norm.weight", kHidden);
  TensorSpec freqs{{64}, GgmlType::kF32, {}};
  for (int i = 0; i < 64; ++i) {
    PutF32(&freqs.data, i < 16 ? 1.0F : 1e30F);
  }
  t["rope_freqs.weight"] = std::move(freqs);
  for (int l = 0; l < 3; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    const std::uint64_t dim = l < 2 ? 64 : 128;
    const auto q8 = [&](const std::string& n, std::uint64_t c,
                        std::uint64_t r) {
      t[p + n] = Matrix(p + n, GgmlType::kQ8_0, c, r);
    };
    const auto vec = [&](const std::string& n, std::uint64_t size) {
      t[p + n] = Vector(p + n, size);
    };
    vec("attn_norm.weight", kHidden);
    q8("attn_q.weight", kHidden, kHeads * dim);
    q8("attn_k.weight", kHidden, kv_heads[l] * dim);
    if (l < 2) {
      q8("attn_v.weight", kHidden, kv_heads[l] * dim);
    }
    vec("attn_q_norm.weight", dim);
    vec("attn_k_norm.weight", dim);
    q8("attn_output.weight", kHeads * dim, kHidden);
    vec("post_attention_norm.weight", kHidden);
    vec("ffn_norm.weight", kHidden);
    q8("ffn_gate.weight", kHidden, kFfn);
    q8("ffn_up.weight", kHidden, kFfn);
    q8("ffn_down.weight", kFfn, kHidden);
    vec("post_ffw_norm.weight", kHidden);
    vec("layer_output_scale.weight", 1);
    t[p + "ffn_gate_inp.weight"] =
        Matrix(p + "ffn_gate_inp.weight", GgmlType::kF32, kHidden, kExperts);
    vec("ffn_gate_inp.scale", kHidden);
    vec("pre_ffw_norm_2.weight", kHidden);
    vec("post_ffw_norm_1.weight", kHidden);
    vec("post_ffw_norm_2.weight", kHidden);
    t[p + "ffn_gate_up_exps.weight"] =
        Matrix(p + "ffn_gate_up_exps.weight", GgmlType::kQ4_K, kHidden,
               2 * kExpertFfn, kExperts);
    t[p + "ffn_down_exps.weight"] =
        Matrix(p + "ffn_down_exps.weight", GgmlType::kQ5_1, kExpertFfn, kHidden,
               kExperts);
    vec("ffn_down_exps.scale", kExperts);
  }
  return t;
}

}  // namespace tiny
}  // namespace gemma4_test

#endif  // GUFO_TESTS_MODELS_GEMMA4_TINY_MODEL_HPP_
