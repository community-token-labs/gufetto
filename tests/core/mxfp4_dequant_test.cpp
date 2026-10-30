// MXFP4 dequantization: bit-exact against llama.cpp's reference over every
// exponent and code byte, plus hand-derived edge values.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace {

// Oracle copied from llama.cpp (MIT, Copyright (c) 2023-2026 The ggml
// authors) at 6d5ac3c8d5aae1831edf17c2d5cda11abe025bb3:
// kvalues_fp4 (ggml/src/ggml-common.h), ggml_e8m0_to_fp32_half
// (ggml/src/ggml-impl.h) and dequantize_row_mxfp4 (ggml/src/ggml-quants.c).
// Kept independent of the product constants so a table typo fails here.
namespace ggml_oracle {

constexpr std::int8_t kvalues_mxfp4[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                           0, -1, -2, -3, -4, -6, -8, -12};

struct block_mxfp4 {
  std::uint8_t e;
  std::uint8_t qs[16];
};

float ggml_e8m0_to_fp32_half(std::uint8_t x) {
  std::uint32_t bits;
  if (x < 2) {
    bits = 0x00200000 << x;
  } else {
    bits = (std::uint32_t)(x - 1) << 23;
  }
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

void dequantize_row_mxfp4(const block_mxfp4* x, float* y, std::int64_t k) {
  static const int qk = 32;
  const int nb = k / qk;
  for (int i = 0; i < nb; i++) {
    const float d = ggml_e8m0_to_fp32_half(x[i].e);
    for (int j = 0; j < qk / 2; ++j) {
      const std::int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
      const std::int8_t x1 = kvalues_mxfp4[x[i].qs[j] >> 4];
      y[i * qk + j + 0] = x0 * d;
      y[i * qk + j + qk / 2] = x1 * d;
    }
  }
}

}  // namespace ggml_oracle

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

std::uint32_t Bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

using Block = std::array<std::uint8_t, 17>;

std::vector<float> Dequantize(const std::vector<Block>& blocks) {
  std::vector<float> out(blocks.size() * 32);
  gufo::quant::DequantizeMXFP4(blocks.data(), out.data(), out.size());
  return out;
}

// Every exponent with every code byte: 16 blocks of 16 bytes per exponent.
void TestBitExactAgainstLlamaCpp() {
  std::vector<Block> blocks;
  for (int e = 0; e < 256; ++e) {
    for (int chunk = 0; chunk < 16; ++chunk) {
      Block block{};
      block[0] = static_cast<std::uint8_t>(e);
      for (int j = 0; j < 16; ++j) {
        block[1 + j] = static_cast<std::uint8_t>((chunk * 16) + j);
      }
      blocks.push_back(block);
    }
  }
  const auto actual = Dequantize(blocks);
  std::vector<float> expected(actual.size());
  ggml_oracle::dequantize_row_mxfp4(
      reinterpret_cast<const ggml_oracle::block_mxfp4*>(blocks.data()),
      expected.data(), static_cast<std::int64_t>(expected.size()));
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (Bits(actual[i]) != Bits(expected[i])) {
      std::cerr << "element " << i << " block exponent "
                << static_cast<int>(blocks[i / 32][0]) << " got 0x" << std::hex
                << Bits(actual[i]) << " expected 0x" << Bits(expected[i])
                << '\n';
      std::exit(1);
    }
  }
  for (int e = 0; e < 256; ++e) {
    Expect(Bits(gufo::quant::E8m0ToFp32Half(static_cast<std::uint8_t>(e))) ==
               Bits(ggml_oracle::ggml_e8m0_to_fp32_half(
                   static_cast<std::uint8_t>(e))),
           "E8M0 half scale matches llama.cpp");
  }
}

// Expected bit patterns derived by hand from the OCP MX spec.
void TestEdgeValues() {
  struct Case {
    std::uint8_t e;
    std::uint8_t nibble;
    std::uint32_t bits;
  };
  constexpr std::array<Case, 15> kCases{{
      {127, 0x1, 0x3F000000},  // scale 1: E2M1 0.5
      {127, 0x7, 0x40C00000},  // 6.0, the largest E2M1 magnitude
      {127, 0xF, 0xC0C00000},  // -6.0
      {127, 0x8, 0x00000000},  // E2M1 negative zero decodes to +0
      {127, 0x0, 0x00000000},
      {128, 0x3, 0x40400000},  // scale 2: 2 * 1.5
      {0, 0x1, 0x00200000},    // 2^-127 * 0.5 = 2^-128, subnormal
      {0, 0x9, 0x80200000},    // -2^-128
      {1, 0x1, 0x00400000},    // 2^-127, subnormal
      {0, 0x7, 0x01400000},    // 2^-127 * 6 = 1.5 * 2^-125, normal
      {2, 0x1, 0x00800000},    // 2^-126, the smallest normal
      {254, 0x3, 0x7F400000},  // 2^127 * 1.5
      {254, 0x7, 0x7F800000},  // 2^127 * 6 overflows to +inf
      {255, 0x1, 0x7F000000},  // reserved exponent decodes as 2^128 * 0.5
      {255, 0xA, 0xFF800000},  // reserved exponent: 2^128 * -1 is -inf
  }};
  for (const auto& c : kCases) {
    Block block{};
    block[0] = c.e;
    block[1] = c.nibble;                                    // element 0
    block[16] = static_cast<std::uint8_t>(c.nibble << 4U);  // element 31
    const auto out = Dequantize({block});
    Expect(Bits(out[0]) == c.bits, "low nibble decodes to element 0");
    Expect(Bits(out[31]) == c.bits, "high nibble decodes to element 31");
    Expect(Bits(out[16]) == 0 && Bits(out[15]) == 0,
           "unused nibbles decode to +0");
  }
}

void TestLayoutAndValidation() {
  Block block{};
  block[0] = 127;  // scale 2^0
  for (std::uint8_t j = 0; j < 16; ++j) {
    block[1 + j] = static_cast<std::uint8_t>(j | ((15U - j) << 4U));
  }
  const auto out = Dequantize({block});
  constexpr std::array<float, 16> kE2m1{
      0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F,  6.0F,
      -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
  for (std::size_t j = 0; j < 16; ++j) {
    Expect(out[j] == kE2m1[j], "first half comes from low nibbles");
    Expect(out[16 + j] == kE2m1[15 - j], "second half from high nibbles");
  }

  std::vector<Block> row(3);
  row[0][0] = 0;
  row[1][0] = 254;
  row[2][0] = 127;
  Expect(gufo::quant::ValidMxfp4Scales(row.data(), 96), "finite exponents");
  row[1][0] = 0xFF;
  Expect(!gufo::quant::ValidMxfp4Scales(row.data(), 96),
         "0xFF exponent is rejected");
  Expect(gufo::quant::ValidMxfp4Scales(row.data(), 32),
         "only the first k / 32 blocks are checked");
}

void TestStorageGeometry() {
  using gufo::core::GgmlType;
  Expect(static_cast<int>(GgmlType::kMXFP4) == 39, "ggml type id");
  Expect(gufo::core::ToString(GgmlType::kMXFP4) == "MXFP4", "type label");
  Expect(gufo::quant::QuantizedBlockElements(GgmlType::kMXFP4) == 32,
         "32 elements per block");
  // gpt-oss-120b expert rows are 2880 wide: 90 blocks of 17 bytes.
  Expect(gufo::quant::QuantizedRowBytes(GgmlType::kMXFP4, 2880) == 1530,
         "row bytes");
  Expect(gufo::quant::EncodedSizeBytes(GgmlType::kMXFP4, 2880 * 2880) ==
             1530U * 2880U,
         "tensor bytes");
  Expect(gufo::quant::QuantizedRowBytes(GgmlType::kMXFP4, 48) == 0,
         "misaligned rows are rejected");
}

}  // namespace

int main() {
  TestBitExactAgainstLlamaCpp();
  TestEdgeValues();
  TestLayoutAndValidation();
  TestStorageGeometry();
  std::cout << "mxfp4_dequant_test passed\n";
  return 0;
}
