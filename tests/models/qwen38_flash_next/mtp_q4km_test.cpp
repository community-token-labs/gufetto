#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/reference.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"
#include "tests/models/qwen38_flash_next/fixtures/unsloth_q4km_descriptors.hpp"

namespace q = gufo::models::qwen38_flash_next;
using gufo::core::GgmlType;
namespace {
void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

// Byte-level goldens, independent of the decoder's bit extraction. Exercise
// both nibble halves, every high bit, signed scales and the -16/+15 endpoints.
void CheckQ5() {
  for (const auto scale : {std::uint16_t{0x3800}, std::uint16_t{0xb800}}) {
    std::array<std::uint8_t, 44> bytes{};
    for (unsigned b = 0; b < 2; ++b) {
      std::memcpy(bytes.data() + b * 22, &scale, 2);
      for (unsigned i = 0; i < 32; ++i) {
        const unsigned code = b == 0 ? i : 31 - i;
        bytes[b * 22 + 2 + i / 8] |= ((code / 16) << (i % 8));
        bytes[b * 22 + 6 + i % 16] |= ((code % 16) << (4 * (i / 16)));
      }
    }
    std::array<float, 64> result{};
    gufo::quant::DequantizeQ5_0(bytes.data(), result.data(), result.size());
    for (unsigned i = 0; i < 64; ++i) {
      const int code = i < 32 ? static_cast<int>(i) : 63 - static_cast<int>(i);
      Require(result[i] == (scale == 0x3800 ? 0.5F : -0.5F) * (code - 16),
              "Q5_0 independent golden");
    }
  }
}

// Synthetic scalar/vector arithmetic contract. No GPU decoder, packing or
// model payload is used to compute the expected integer dot products.
void CheckRoutedArithmetic() {
  using Projection = q::reference::Projection;
  q::TensorRef t{.type = GgmlType::kQ4_K,
                 .cols = 512,
                 .rows = 2,
                 .experts = 2,
                 .name = {}};
  std::vector<std::uint8_t> bytes(t.SizeBytes());
  t.data = bytes.data();
  double old_error = 0, new_error = 0, norm = 0, sum_protocol_error = 0;
  bool reconstruction_matches = true;
  for (unsigned trial = 0; trial < 4; ++trial) {
    std::vector<float> x(t.cols), quantized(t.cols);
    std::array<float, 16> scales{}, stored_sums{};
    std::array<int, 512> codes{};
    for (unsigned b = 0; b < 16; ++b) {
      float peak = 0;
      for (unsigned j = 0; j < 32; ++j) {
        const unsigned col = b * 32 + j;
        x[col] = std::sin(float(col * 13 + trial * 31) * 0.017F) +
                 0.19F * float(trial + 1);
        peak = std::max(peak, std::abs(x[col]));
      }
      const float d = peak / 127.0F;
      scales[b] = static_cast<float>(static_cast<_Float16>(d));
      int sum = 0;
      for (unsigned j = 0; j < 32; ++j) {
        const unsigned col = b * 32 + j;
        codes[col] = static_cast<int>(std::round(x[col] / d));
        sum += codes[col];
        quantized[col] = scales[b] * codes[col];
      }
      stored_sums[b] = static_cast<float>(static_cast<_Float16>(d * sum));
    }
    for (unsigned e = 0; e < 2; ++e) {
      std::array<float, 2> expected{}, mmq_sum{}, old{}, actual{}, dense{};
      for (unsigned r = 0; r < 2; ++r) {
        double dot = 0, alternate = 0;
        for (unsigned b = 0; b < 2; ++b) {
          auto* p = bytes.data() + (e * 2 + r) * t.RowBytes() + b * 144;
          p[1] = 0x38;  // d = 1/2, dmin = 1/4: exact binary fractions.
          p[3] = 0x34;
          std::array<unsigned, 8> sc{}, mn{};
          for (unsigned s = 0; s < 8; ++s) {
            sc[s] = (s * 11 + e * 7 + r * 3 + b * 5) % 63 + 1;
            mn[s] = (s * 7 + e * 13 + r * 5 + b * 3) % 63 + 1;
          }
          // Inverse GGUF six-bit scale layout; expected values above do not
          // come from any decoder, and include the high two bits.
          for (unsigned s = 0; s < 4; ++s) {
            p[4 + s] = sc[s] | ((sc[s + 4] >> 4) << 6);
            p[8 + s] = mn[s] | ((mn[s + 4] >> 4) << 6);
            p[12 + s] = (sc[s + 4] & 15) | ((mn[s + 4] & 15) << 4);
          }
          for (unsigned s = 0; s < 8; ++s) {
            int product = 0, sum = 0;
            for (unsigned j = 0; j < 32; ++j) {
              const unsigned code = (j * 3 + s * 5 + e + r * 7 + b) % 16;
              const unsigned offset = (s / 2) * 32 + j;
              if (s % 2 == 0)
                p[16 + offset] = code;
              else
                p[16 + offset] |= code << 4;
              const auto col = b * 256 + s * 32 + j;
              product += static_cast<int>(code) * codes[col];
              sum += codes[col];
            }
            const auto group = b * 8 + s;
            dot += 0.5 * sc[s] * scales[group] * product -
                   0.25 * mn[s] * scales[group] * sum;
            alternate += 0.5 * sc[s] * scales[group] * product -
                         0.25 * mn[s] * stored_sums[group];
          }
        }
        expected[r] = static_cast<float>(dot);
        mmq_sum[r] = static_cast<float>(alternate);
      }
      q::cpu::MatVec(t, e, x, old);  // The old Q4 reference protocol.
      q::reference::DecodeMatVec(t, e, x, actual, Projection::kRoutedVector);
      q::reference::DecodeMatVec(t, e, x, dense, Projection::kDense);
      Require(dense == old, "dense Q4 must retain F32 activations");
      std::array<float, 2> reconstructed{};
      q::cpu::MatVec(t, e, quantized, reconstructed);
      reconstruction_matches =
          reconstruction_matches && actual == reconstructed;
      for (unsigned r = 0; r < 2; ++r) {
        old_error += std::pow(double(old[r]) - expected[r], 2);
        new_error += std::pow(double(actual[r]) - expected[r], 2);
        sum_protocol_error += std::pow(double(mmq_sum[r]) - expected[r], 2);
        norm += double(expected[r]) * expected[r];
      }
    }
  }
  std::cout << "Q4 routed synthetic relative_rms old="
            << std::sqrt(old_error / norm)
            << " new=" << std::sqrt(new_error / norm)
            << " wrong-stored-sum=" << std::sqrt(sum_protocol_error / norm)
            << '\n';
  Require(reconstruction_matches,
          "routed vector must use reconstructed Q8_1, not stored sum");
  Require(std::sqrt(old_error / norm) > 5e-5,
          "fixture must distinguish old F32 activation protocol");
  Require(sum_protocol_error > 0,
          "fixture must distinguish MMQ stored-sum protocol");
  Require(std::sqrt(new_error / norm) < 2e-6,
          "Q4 routed vector differs from independent integer formula");

  std::vector<float> zero(t.cols, 0.0F);
  for (unsigned e = 0; e < 2; ++e) {
    std::array<float, 2> actual{};
    q::reference::DecodeMatVec(t, e, zero, actual, Projection::kRoutedVector);
    Require(actual == std::array<float, 2>{}, "zero routed Q4 input changed");
  }

  // Q8_0 dense/routed paths are byte-identical to the pre-change protocol.
  t.type = GgmlType::kQ8_0;
  bytes.assign(t.SizeBytes(), 0);
  t.data = bytes.data();
  for (unsigned row = 0; row < 4; ++row)
    for (unsigned b = 0; b < 16; ++b) {
      auto* p = bytes.data() + row * t.RowBytes() + b * 34;
      p[1] = 0x38;
      for (unsigned j = 0; j < 32; ++j)
        p[2 + j] = static_cast<std::uint8_t>(int((j + b + row) % 255) - 127);
    }
  std::vector<float> x(t.cols), quantized(t.cols);
  for (unsigned j = 0; j < t.cols; ++j)
    x[j] = 0.3F + std::sin(float(j) * 0.031F);
  for (unsigned b = 0; b < 16; ++b) {
    float peak = 0;
    for (unsigned j = 0; j < 32; ++j)
      peak = std::max(peak, std::abs(x[b * 32 + j]));
    const float d = peak / 127.0F;
    const float stored = static_cast<float>(static_cast<_Float16>(d));
    for (unsigned j = 0; j < 32; ++j)
      quantized[b * 32 + j] = stored * std::round(x[b * 32 + j] / d);
  }
  for (unsigned e = 0; e < 2; ++e) {
    std::array<float, 2> expected{}, dense{}, routed{};
    q::cpu::MatVec(t, e, quantized, expected);
    q::reference::DecodeMatVec(t, e, x, dense, Projection::kDense);
    q::reference::DecodeMatVec(t, e, x, routed, Projection::kRoutedVector);
    Require(dense == expected && routed == expected,
            "Q8 dense/routed activation protocol regressed");
  }
  std::cout << "Q8 dense/routed synthetic: bit-identical legacy protocol\n";
}

void CheckRows() {
  // Two rows in each of two experts; no padding can hide a bad expert stride.
  for (auto type : {GgmlType::kQ4_K, GgmlType::kQ6_K, GgmlType::kQ5_0}) {
    q::TensorRef t{
        .type = type, .cols = 256, .rows = 2, .experts = 2, .name = {}};
    std::vector<std::uint8_t> bytes(t.SizeBytes());
    t.data = bytes.data();
    for (unsigned row = 0; row < 4; ++row) {
      auto* p = bytes.data() + row * t.RowBytes();
      if (type == GgmlType::kQ4_K) {
        // d=1/2, dmin=1/4, scale=1, min=2, q=row.
        p[1] = 0x38;
        p[3] = 0x34;
        for (unsigned s = 0; s < 4; ++s) {
          p[4 + s] = 1;
          p[8 + s] = 2;
          p[12 + s] = 0x21;
        }
        std::memset(p + 16, row * 17, 128);
      } else if (type == GgmlType::kQ6_K) {
        // All six-bit codes are 32+row; signed subscales alternate.
        std::memset(p, row * 17, 128);
        std::memset(p + 128, 0xaa, 64);
        for (unsigned s = 0; s < 16; ++s)
          p[192 + s] = s % 2 ? 0xff : 1;
        p[209] = 0x38;
      } else {
        for (unsigned k = 0; k < 8; ++k) {
          p[k * 22 + 1] = 0x38;
          std::memset(p + k * 22 + 6, row * 17, 16);
        }
      }
    }
    std::array<float, 256> out{};
    for (unsigned e = 0; e < 2; ++e)
      for (unsigned r = 0; r < 2; ++r) {
        q::cpu::DequantizeRow(t, e, r, out.data());
        const unsigned row = 2 * e + r;
        for (unsigned i = 0; i < 256; ++i) {
          float expected = type == GgmlType::kQ4_K ? 0.5F * row - 0.5F
                           : type == GgmlType::kQ5_0
                               ? 0.5F * (static_cast<int>(row) - 16)
                               : 0.5F * row * (i / 16 % 2 ? -1 : 1);
          Require(out[i] == expected, "independent packed expert-row golden");
        }
      }
    // HC/EH packed dense projections must not acquire Q8 activations.
    std::array<float, 256> x{};
    for (unsigned i = 0; i < x.size(); ++i)
      x[i] = std::sin(float(i) * 0.031F) + 0.27F;
    for (unsigned e = 0; e < 2; ++e) {
      std::array<float, 2> expected{}, actual{};
      q::cpu::MatVec(t, e, x, expected);
      q::reference::DecodeMatVec(t, e, x, actual,
                                 q::reference::Projection::kDense);
      Require(actual == expected, "packed dense lost F32 activation protocol");
    }
    t.cols = type == GgmlType::kQ5_0 ? 319 : 320;
    Require(t.RowBytes() == 0, "unaligned packed columns accepted");
  }
}

struct Tensor {
  std::string name;
  std::uint64_t cols, rows, experts;
  GgmlType type;
};

// Sparse synthetic GGUF, not model data. All payloads intentionally alias one
// zero-filled range: binding only inspects descriptors and mapped bounds.
struct Fixture {
  std::filesystem::path path;
  std::vector<std::uint8_t> header;
  std::vector<Tensor> tensors;
  std::uint64_t payload = 0;
  Fixture() {
    path = std::filesystem::temp_directory_path() /
           ("gufo-mtp-q4km-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            ".gguf");
  }
  ~Fixture() { std::filesystem::remove(path); }
  template<class T>
  void Pod(T value) {
    const auto size = header.size();
    header.resize(size + sizeof(value));
    std::memcpy(header.data() + size, &value, sizeof(value));
  }
  void Text(const std::string& s) {
    Pod(std::uint64_t{s.size()});
    header.insert(header.end(), s.begin(), s.end());
  }
  void Add(std::string name, std::uint64_t cols, std::uint64_t rows,
           GgmlType type, std::uint64_t experts = 1) {
    tensors.push_back({std::move(name), cols, rows, experts, type});
  }
  void Mixer(const std::string& p, GgmlType down, bool inject) {
    Add(p + "_norm.weight", 10240, 1, GgmlType::kF32);
    Add(p + "_down.weight", 10240, 320, down);
    Add(p + "_up.weight", 320, 10240, GgmlType::kQ5_0);
    if (inject)
      Add(p + "_inject.weight", 10240, 4, GgmlType::kQ4_K);
  }
  void Mtp() {
    const std::string p = "blk.48.";
    Mixer(p + "hc_attn", GgmlType::kQ4_K, true);
    Mixer(p + "hc_ffn", GgmlType::kQ6_K, true);
    Mixer(p + "nextn.hc_head", GgmlType::kQ4_K, false);
    for (auto name : {"nextn.enorm.weight", "nextn.hnorm.weight"})
      Add(p + name,
          std::string(name).find("hnorm") != std::string::npos ? 10240 : 2560,
          1, GgmlType::kF32);
    Add(p + "nextn.eh_proj.weight", 5120, 2560, GgmlType::kQ4_K);
    Add(p + "attn_q.weight", 2560, 12288, GgmlType::kQ4_K);
    Add(p + "attn_k.weight", 2560, 512, GgmlType::kQ4_K);
    Add(p + "attn_v.weight", 2560, 512, GgmlType::kQ6_K);
    Add(p + "attn_output.weight", 6144, 2560, GgmlType::kQ4_K);
    Add(p + "attn_q_norm.weight", 256, 1, GgmlType::kF32);
    Add(p + "attn_k_norm.weight", 256, 1, GgmlType::kF32);
    Add(p + "indexer.q_proj.weight", 2560, 512, GgmlType::kBF16);
    Add(p + "indexer.k_proj.weight", 2560, 128, GgmlType::kBF16);
    Add(p + "indexer.q_norm.weight", 128, 1, GgmlType::kF32);
    Add(p + "indexer.k_norm.weight", 128, 1, GgmlType::kF32);
    Add(p + "ffn_gate_inp.weight", 2560, 512, GgmlType::kF32);
    Add(p + "ffn_gate_inp_shexp.weight", 2560, 1, GgmlType::kF32);
    Add(p + "ffn_gate_exps.weight", 2560, 640, GgmlType::kQ4_K, 512);
    Add(p + "ffn_up_exps.weight", 2560, 640, GgmlType::kQ4_K, 512);
    Add(p + "ffn_down_exps.weight", 640, 2560, GgmlType::kQ8_0, 512);
    Add(p + "ffn_gate_shexp.weight", 2560, 640, GgmlType::kQ4_K);
    Add(p + "ffn_up_shexp.weight", 2560, 640, GgmlType::kQ4_K);
    Add(p + "ffn_down_shexp.weight", 640, 2560, GgmlType::kQ8_0);
  }
  std::unique_ptr<gufo::core::GgufReader> Open(bool truncated = false) {
    header.clear();
    payload = 0;
    std::map<std::string, std::uint32_t> ints{
        {"block_count", 49},
        {"nextn_predict_layers", 1},
        {"embedding_length", 2560},
        {"context_length", 262144},
        {"hyper_connection.count", 4},
        {"hyper_connection.low_rank", 320},
        {"full_attention_interval", 4},
        {"attention.head_count", 24},
        {"attention.head_count_kv", 2},
        {"attention.key_length", 256},
        {"rope.dimension_count", 64},
        {"attention.indexer.head_count", 4},
        {"attention.indexer.key_length", 128},
        {"attention.indexer.top_k", 2048},
        {"ssm.conv_kernel", 4},
        {"ssm.state_size", 128},
        {"ssm.group_count", 16},
        {"ssm.time_step_rank", 48},
        {"ssm.inner_size", 6144},
        {"expert_count", 512},
        {"expert_used_count", 10},
        {"expert_feed_forward_length", 640},
        {"expert_shared_feed_forward_length", 640}};
    Pod(std::uint32_t{0x46554747});
    Pod(std::uint32_t{3});
    Pod(std::uint64_t{tensors.size()});
    Pod(std::uint64_t{ints.size() + 5});
    Text("general.architecture");
    Pod(std::uint32_t{8});
    Text("qwen4exp");
    for (const auto& [name, value] : ints) {
      Text("qwen4exp." + name);
      Pod(std::uint32_t{4});
      Pod(value);
    }
    for (const auto& [name, value] : std::map<std::string, float>{
             {"attention.layer_norm_rms_epsilon", 1e-6F},
             {"rope.freq_base", 1e7F}}) {
      Text("qwen4exp." + name);
      Pod(std::uint32_t{6});
      Pod(value);
    }
    Text("qwen4exp.rope.dimension_sections");
    Pod(std::uint32_t{9});
    Pod(std::uint32_t{10});
    Pod(std::uint64_t{4});
    for (auto x : {11, 11, 10, 0})
      Pod(std::uint64_t(x));
    Text("qwen4exp.attention.compress_ratios");
    Pod(std::uint32_t{9});
    Pod(std::uint32_t{10});
    Pod(std::uint64_t{49});
    for (unsigned i = 0; i < 49; ++i)
      Pod(std::uint64_t{i % 4 == 3 ? 4U : 0U});
    for (const auto& t : tensors) {
      Text(t.name);
      Pod(std::uint32_t{t.experts != 1 ? 3U : t.rows != 1 ? 2U : 1U});
      Pod(t.cols);
      if (t.rows != 1 || t.experts != 1)
        Pod(t.rows);
      if (t.experts != 1)
        Pod(t.experts);
      Pod(static_cast<std::uint32_t>(t.type));
      Pod(std::uint64_t{0});
      payload = std::max(payload, std::uint64_t(gufo::quant::EncodedSizeBytes(
                                      t.type, t.cols * t.rows * t.experts)));
    }
    header.resize((header.size() + 31) / 32 * 32);
    {
      std::ofstream file(path, std::ios::binary | std::ios::trunc);
      file.write(reinterpret_cast<const char*>(header.data()), header.size());
    }
    std::filesystem::resize_file(path,
                                 header.size() + payload - (truncated ? 1 : 0));
    std::string error;
    return gufo::core::GgufReader::OpenFile(path.string(), &error);
  }
};

void CheckBinding() {
  Fixture f;
  f.Mtp();
  Require(f.tensors.size() == 32, "modern 32-tensor fixture");
  std::uint64_t end = 0;
  for (const auto& expected : unsloth_q4km_fixture::kDescriptors) {
    const auto it =
        std::find_if(f.tensors.begin(), f.tensors.end(),
                     [&](const auto& t) { return t.name == expected.name; });
    Require(it != f.tensors.end(), "captured descriptor missing");
    Require(it->cols == expected.ne[0] && it->rows == expected.ne[1] &&
                it->experts == expected.ne[2] &&
                static_cast<unsigned>(it->type) == expected.type,
            "synthetic shape/type differs from independently captured header");
    const unsigned rank = it->experts != 1 ? 3U : it->rows != 1 ? 2U : 1U;
    Require(rank == expected.rank, "captured tensor rank mismatch");
    end = std::max(end, expected.offset +
                            gufo::quant::EncodedSizeBytes(
                                it->type, it->cols * it->rows * it->experts));
  }
  Require((unsloth_q4km_fixture::kHeaderBytes + 31) / 32 * 32 + end ==
              unsloth_q4km_fixture::kFileBytes,
          "captured geometry/file size mismatch");
  q::Config trunk;
  {
    auto reader = f.Open();
    Require(bool(reader), "synthetic GGUF parse");
    auto config = q::Config::FromGguf(*reader, false);
    Require(bool(config), "fixture config");
    trunk = *config;
    trunk.nextn_layers = 0;
    std::string error;
    auto mtp = q::MtpWeights::Bind(*reader, trunk, &error);
    Require(bool(mtp), error.c_str());
    Require(mtp->block.hc_attn.up.RowBytes() == 220, "320-wide Q5_0 row");
    Require(mtp->block.nextn_eh_proj.RowBytes() == 2880, "split EH geometry");
  }
  for (const auto& [name, type] : std::vector<std::pair<std::string, GgmlType>>{
           {"hc_attn_up.weight", GgmlType::kQ4_K},
           {"indexer.k_proj.weight", GgmlType::kQ4_K},
           {"attn_q.weight", GgmlType::kQ6_K},
           {"hc_attn_inject.weight", GgmlType::kQ5_0},
           {"ffn_down_shexp.weight", GgmlType::kQ6_K}}) {
    auto it =
        std::find_if(f.tensors.begin(), f.tensors.end(),
                     [&](const auto& t) { return t.name == "blk.48." + name; });
    const auto old = it->type;
    it->type = type;
    {
      auto reader = f.Open();
      std::string error;
      Require(!reader || !q::MtpWeights::Bind(*reader, trunk, &error),
              "unsupported role admitted");
    }
    it->type = old;
  }
  {
    auto reader = f.Open(true);
    Require(!reader, "truncated payload admitted");
  }
  // Trunk allowlists are deliberately not broadened. Bind reaches each
  // protected role before reporting missing later trunk tensors.
  for (const auto& name : {"token_embd.weight", "output.weight",
                           "output_hc_down.weight", "output_hc_up.weight"}) {
    Fixture protected_role;
    protected_role.Add("token_embd.weight", 2560, 16, GgmlType::kQ8_0);
    protected_role.Add("output.weight", 2560, 16, GgmlType::kQ8_0);
    protected_role.Mixer("output_hc", GgmlType::kQ8_0, false);
    for (auto& t : protected_role.tensors) {
      if (t.type == GgmlType::kQ5_0)
        t.type = GgmlType::kQ8_0;
      if (t.name == name)
        t.type = t.cols == 320 ? GgmlType::kQ5_0 : GgmlType::kQ4_K;
    }
    auto reader = protected_role.Open();
    Require(bool(reader), "protected fixture parse");
    std::string error;
    Require(!q::ModelWeights::Bind(*reader, &error),
            "trunk packed role admitted");
    Require(error.find(name) != std::string::npos &&
                error.find("unsupported format") != std::string::npos,
            "trunk rejection did not identify protected role");
  }
  // Existing shared-Q8 sidecar remains accepted.
  for (auto& t : f.tensors)
    if (t.type == GgmlType::kQ4_K || t.type == GgmlType::kQ5_0 ||
        t.type == GgmlType::kQ6_K)
      t.type = GgmlType::kQ8_0;
  {
    auto reader = f.Open();
    std::string error;
    Require(reader && q::MtpWeights::Bind(*reader, trunk, &error).has_value(),
            "shared Q8 regressed");
  }
}
}  // namespace
int main() {
  try {
    CheckQ5();
    CheckRows();
    CheckRoutedArithmetic();
    CheckBinding();
    std::cout << "MTP Q4KM CPU checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
