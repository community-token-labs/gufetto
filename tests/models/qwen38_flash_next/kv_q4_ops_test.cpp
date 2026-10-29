#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kv_cache.hpp"
#include "src/models/qwen38_flash_next/kv_rotation.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace gpu = qfn::rocm;
using Type = qfn::KvCacheType;
namespace {
void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
void Hip(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}
template<class T>
struct Buffer {
  T* data{};
  std::size_t count;
  explicit Buffer(std::size_t n) : count(n) {
    Hip(hipMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    Hip(hipMemset(data, 0, n * sizeof(T)));
  }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void Put(std::span<const T> values) {
    Require(values.size() == count, "upload length");
    Hip(hipMemcpy(data, values.data(), count * sizeof(T),
                  hipMemcpyHostToDevice));
  }
  std::vector<T> Get() {
    Hip(hipGetLastError());
    std::vector<T> result(count);
    Hip(hipMemcpy(result.data(), data, count * sizeof(T),
                  hipMemcpyDeviceToHost));
    return result;
  }
};
std::vector<float> Values(std::size_t count, unsigned seed) {
  std::vector<float> result(count);
  for (auto& x : result) {
    seed = seed * 1664525U + 1013904223U;
    x = static_cast<float>(static_cast<int>(seed % 16384) - 8192) / 1024;
  }
  return result;
}
std::vector<std::byte> Encode(std::span<const float> values, Type codec) {
  std::vector<std::byte> result(qfn::KvRowBytes(values.size(), codec));
  qfn::EncodeKvRow(values, codec, result);
  return result;
}
std::vector<std::uint16_t> Decode(std::span<const std::byte> bytes, Type codec,
                                  std::size_t count) {
  std::vector<float> floats(count);
  qfn::DecodeKvRow(bytes, codec, floats);
  std::vector<std::uint16_t> result(count);
  std::transform(floats.begin(), floats.end(), result.begin(),
                 qfn::KvFloatToHalfBits);
  return result;
}
void HalfEqual(const std::vector<std::uint16_t>& actual,
               const std::vector<std::uint16_t>& expected) {
  Require(actual.size() == expected.size(), "half length");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i] != expected[i] && ((actual[i] | expected[i]) & 0x7fff) != 0)
      throw std::runtime_error("half operand mismatch at " + std::to_string(i));
}
void FloatEqual(const std::vector<float>& a, const std::vector<float>& b) {
  Require(a.size() == b.size(), "float length");
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
        (a[i] != b[i] &&
         std::bit_cast<unsigned>(a[i]) != std::bit_cast<unsigned>(b[i])))
      throw std::runtime_error("attention operand control mismatch at " +
                               std::to_string(i));
}
void CheckStore() {
  constexpr unsigned rows = 257, width = 512, start = 3;
  auto input = Values(rows * width, 91);
  // Signed ties, zero, subnormal half scales and upper finite boundaries.
  for (unsigned i = 0; i < 32; ++i)
    input[i] = static_cast<float>(static_cast<int>(i % 16) - 8);
  std::fill(input.begin() + 32, input.begin() + 64, 0);
  input[32] = 8;
  input[48] = -8;
  std::fill(input.begin() + 64, input.begin() + 96, 0);
  input[64] = std::ldexp(1.0F, -25);
  std::fill(input.begin() + 96, input.begin() + 128, 0);
  input[96] = -65504;
  Buffer<float> source(input.size());
  source.Put(input);
  Buffer<std::uint32_t> position(1);
  position.Put(std::vector<std::uint32_t>{start});
  const auto expected = Encode(input, Type::kQ4_0);
  constexpr auto row = qfn::KvRowBytes(width, Type::kQ4_0);
  Buffer<std::byte> cache((start + rows + 1) * row);
  Hip(hipMemset(cache.data, 0x55, cache.count));
  gpu::StoreKv(source.data, cache.data, rows, width, position.data, nullptr,
               Type::kQ4_0);
  auto actual = cache.Get();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto want = i < start * row || i >= (start + rows) * row
                          ? std::byte{0x55}
                          : expected[i - start * row];
    if (actual[i] != want)
      throw std::runtime_error("q4 store byte mismatch at " +
                               std::to_string(i));
  }
  // Invalid input must poison, not silently become finite packed state.
  for (const auto invalid :
       {std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::max(), std::ldexp(1.0F, -140)}) {
    std::fill(input.begin(), input.end(), 0);
    input[31] = invalid;
    source.Put(input);
    gpu::StoreKv(source.data, cache.data, 1, width, position.data, nullptr,
                 Type::kQ4_0);
    actual = cache.Get();
    const auto bits = std::to_integer<unsigned>(actual[start * row]) |
                      (std::to_integer<unsigned>(actual[start * row + 1]) << 8);
    Require((bits & 0x7c00) == 0x7c00, "invalid q4 block not poisoned");
  }
}
void CheckReaders(Type policy, unsigned n, bool sparse) {
  constexpr unsigned width = 512, qwidth = 6144, depth = 257;
  const unsigned start = depth - n, words = ((depth + 3) / 4 + 31) / 32;
  auto k = Values(depth * width, 134), v = Values(depth * width, 258);
  if (qfn::KvIsRotated(policy)) {
    for (std::size_t i = 0; i < k.size(); i += 256) {
      qfn::RotateKvHead(std::span(k).subspan(i, 256),
                        qfn::KvRotationKind::kKeyQuery);
      qfn::RotateKvHead(std::span(v).subspan(i, 256),
                        qfn::KvRotationKind::kValueOutput);
    }
  }
  const auto kt = qfn::KvKeyType(policy), vt = qfn::KvValueType(policy);
  const auto kb = Encode(k, kt), vb = Encode(v, vt);
  const auto kh = Decode(kb, kt, k.size()), vh = Decode(vb, vt, v.size());
  Buffer<std::byte> keys(kb.size()), values(vb.size());
  keys.Put(kb);
  values.Put(vb);
  Buffer<std::uint16_t> key_half(kh.size()), value_half(vh.size());
  key_half.Put(kh);
  value_half.Put(vh);
  Buffer<std::uint16_t> key_view(kh.size()), value_view(vh.size());
  Require(gpu::DequantizeQ8Kv(keys.data, values.data, key_view.data,
                              value_view.data, depth, width, nullptr, policy),
          "prefill view rejected");
  HalfEqual(key_view.Get(), kh);
  HalfEqual(value_view.Get(), vh);
  Buffer<float> queries(n * qwidth), out(n * qwidth);
  const auto query = Values(n * qwidth, 99);
  queries.Put(query);
  Buffer<std::uint32_t> position(1), masks(n * words);
  position.Put(std::vector<std::uint32_t>{start});
  std::vector<std::uint32_t> mask(n * words, 0x55555555U);
  masks.Put(mask);
  const auto run = [&](bool packed, bool wmma) {
    const void* pk = packed ? static_cast<void*>(keys.data) : key_half.data;
    const void* pv = packed ? static_cast<void*>(values.data) : value_half.data;
    const auto type = packed ? policy : Type::kF16;
    if (wmma)
      Require(gpu::WmmaCausalAttention(queries.data, nullptr, pk, pv,
                                       sparse ? masks.data : nullptr, words,
                                       out.data, n, start, 24, 2, 256, 4,
                                       nullptr, false, type),
              "WMMA rejected");
    else
      gpu::Attention(queries.data, pk, pv, sparse ? masks.data : nullptr, words,
                     out.data, nullptr, 1, n, position.data, 24, 2, 256, 4,
                     nullptr, nullptr, nullptr, 0, type);
    return out.Get();
  };
  FloatEqual(run(true, false), run(false, false));
  FloatEqual(run(true, true), run(false, true));
  // This is raw-byte round-trip/operator evidence, not model restore evidence.
  keys.Put(keys.Get());
  values.Put(values.Get());
  FloatEqual(run(true, false), run(false, false));
}
}  // namespace
int main() {
  try {
    CheckStore();
    for (const auto type : {Type::kQ4_0, Type::kQ8_0Q4_0, Type::kQ4_0Rotated,
                            Type::kQ8_0Q4_0Rotated})
      for (const auto n : {1U, 2U, 3U, 8U, 256U})
        for (bool sparse : {false, true}) {
          std::cout << "q4 reader policy=" << static_cast<unsigned>(type)
                    << " n=" << n << " sparse=" << sparse << std::endl;
          CheckReaders(type, n, sparse);
        }
    std::cout << "q4/mixed stores, view, dense/sparse scalar/MTP/WMMA "
                 "operands: PASS\n";
  } catch (const std::exception& e) {
    std::cerr << "q4 native FAIL: " << e.what() << '\n';
    return 1;
  }
}
