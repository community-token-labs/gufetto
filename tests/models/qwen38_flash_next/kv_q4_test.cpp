#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/kv_cache.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using qfn::KvCacheType;

namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid q4 call accepted");
}
void CheckBlocks() {
  std::array<float, 32> input{};
  std::array<std::byte, 18> packed{};
  std::array<float, 32> output{};
  const auto encode = [&] {
    qfn::EncodeKvRow(input, KvCacheType::kQ4_0, packed);
  };
  encode();
  Require(packed[0] == std::byte{0} && packed[1] == std::byte{0}, "zero scale");
  for (std::size_t i = 2; i < packed.size(); ++i)
    Require(packed[i] == std::byte{0x88}, "zero nibbles");
  // Literal independent wire vector: d=1, coordinates -8..7 twice.
  for (unsigned i = 0; i < 32; ++i)
    input[i] = static_cast<float>(static_cast<int>(i % 16) - 8);
  encode();
  Require(packed[0] == std::byte{0} && packed[1] == std::byte{0x3c}, "d=1");
  for (unsigned i = 0; i < 16; ++i)
    Require(packed[2 + i] == std::byte(i | (i << 4)), "nibble order");
  qfn::DecodeKvRow(packed, KvCacheType::kQ4_0, output);
  Require(output == input, "exact grid round trip");
  // First maximal magnitude wins: +8 before -8 gives d=-1.
  input.fill(0);
  input[0] = 8;
  input[16] = -8;
  encode();
  Require(packed[1] == std::byte{0xbc} && packed[2] == std::byte{0xf0},
          "signed absmax tie or saturation");
  // Positive d: ties toward the upper offset code, not ties-away from zero.
  input.fill(0);
  input[0] = -8;
  input[1] = -0.5F;
  input[17] = 0.5F;
  encode();
  Require(packed[3] == std::byte{0x98}, "q4 half ties");
  for (const auto type : {KvCacheType::kQ4_0, KvCacheType::kQ4_0Rotated}) {
    std::array<std::byte, 18> other{};
    qfn::EncodeKvRow(input, type, other);
    Require(other == packed, "codec must not rotate coordinates");
  }
  // Half scale underflows, while F32 scale and reciprocal remain valid.
  input.fill(0);
  input[0] = std::ldexp(1.0F, -25);
  encode();
  Require(packed[0] == std::byte{0} && packed[1] == std::byte{0x80} &&
              packed[2] == std::byte{0x80},
          "negative-zero scale preserves codes");
  qfn::DecodeKvRow(packed, KvCacheType::kQ4_0, output);
  for (auto x : output)
    Require(x == 0, "underflow reconstruction");
}
void CheckValidation() {
  std::array<float, 64> input{};
  std::array<std::byte, 36> packed{};
  packed.fill(std::byte{0x55});
  const auto before = packed;
  for (const auto invalid : {std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::max()}) {
    input[63] = invalid;
    Reject([&] { qfn::EncodeKvRow(input, KvCacheType::kQ4_0, packed); });
    Require(packed == before, "partial encode on rejection");
  }
  input.fill(0);
  qfn::EncodeKvRow(input, KvCacheType::kQ4_0, packed);
  packed[18] = std::byte{0};
  packed[19] = std::byte{0x7c};
  std::array<float, 64> output{};
  output.fill(123);
  Reject([&] { qfn::DecodeKvRow(packed, KvCacheType::kQ4_0, output); });
  Require(std::ranges::all_of(output, [](float x) { return x == 123; }),
          "partial decode on rejection");
  Reject([&] { (void)qfn::KvRowBytes(31, KvCacheType::kQ4_0); });
  Reject([&] { (void)qfn::KvRowBytes(512, KvCacheType::kQ8_0Q4_0); });
  Reject([&] {
    qfn::EncodeKvRow(input, KvCacheType::kQ4_0,
                     std::as_writable_bytes(std::span(input)).first(36));
  });
  for (const auto type :
       {KvCacheType::kQ4_0, KvCacheType::kQ8_0Q4_0, KvCacheType::kQ4_0Rotated,
        KvCacheType::kQ8_0Q4_0Rotated}) {
    Require(qfn::KvRowBytes(512, qfn::KvValueType(type)) == 288, "V stride");
    Require(qfn::ExpectedKvPayloadVersion(type) == 24, "new numerical epoch");
  }
}
void CheckPopulation() {
  // Independently compute the scalar ggml convention and pack expected bytes.
  std::uint32_t state = 0x934681;
  std::array<float, 512> input{}, decoded{};
  std::array<std::byte, 288> actual{}, expected{};
  for (unsigned trial = 0; trial < 128; ++trial) {
    for (auto& x : input) {
      state = state * 1664525U + 1013904223U;
      x = static_cast<float>(static_cast<int>(state % 32767) - 16383) / 1024;
    }
    for (unsigned block = 0; block < 16; ++block) {
      const auto* x = input.data() + block * 32;
      unsigned maximum = 0;
      for (unsigned i = 1; i < 32; ++i)
        if (std::abs(x[i]) > std::abs(x[maximum]))
          maximum = i;
      volatile float d = x[maximum] / -8.0F;
      volatile float inverse = 1.0F / d;
      const auto bits = qfn::KvFloatToHalfBits(d);
      expected[block * 18] = std::byte(bits & 255);
      expected[block * 18 + 1] = std::byte(bits >> 8);
      for (unsigned j = 0; j < 16; ++j) {
        volatile float lo_product = x[j] * inverse;
        volatile float hi_product = x[j + 16] * inverse;
        const auto lo = std::clamp(static_cast<int>(lo_product + 8.5F), 0, 15);
        const auto hi = std::clamp(static_cast<int>(hi_product + 8.5F), 0, 15);
        expected[block * 18 + 2 + j] = std::byte(lo | (hi << 4));
      }
    }
    qfn::EncodeKvRow(input, KvCacheType::kQ4_0, actual);
    Require(actual == expected, "reference population packed bytes");
    qfn::DecodeKvRow(actual, KvCacheType::kQ4_0, decoded);
    for (unsigned i = 0; i < 512; ++i)
      Require(
          std::isfinite(decoded[i]) && std::abs(decoded[i] - input[i]) < 2.1F,
          "round-trip error bound");
  }
}
}  // namespace
int main() {
  try {
    CheckBlocks();
    CheckValidation();
    CheckPopulation();
    std::cout << "Flash-Next q4 codec: PASS\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
