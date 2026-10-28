#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_PACKED_MTP_REFERENCE_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_PACKED_MTP_REFERENCE_HPP_

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

// Independent CPU decoding plus the declared packed projection arithmetic.
// Never calls RowElement/PackedElement or a second GPU projection as oracle.
namespace packed_mtp_test {
namespace rocm = gufo::models::qwen38_flash_next::rocm;
inline void Hip(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
template<class T>
struct Buffer {
  T* data = nullptr;
  explicit Buffer(std::size_t n) {
    Hip(hipMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
  }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
struct Weights {
  std::vector<std::uint8_t> raw;
  std::vector<float> values;
};
inline Weights Make(rocm::WeightType type, unsigned m, unsigned k) {
  const unsigned block = type == rocm::WeightType::kQ5_0 ? 32 : 256;
  const unsigned bytes = type == rocm::WeightType::kQ5_0   ? 22
                         : type == rocm::WeightType::kQ4_K ? 144
                                                           : 210;
  Weights w{std::vector<std::uint8_t>(std::size_t(m) * (k / block) * bytes),
            std::vector<float>(std::size_t(m) * k)};
  std::uint32_t seed = 0x73257;
  for (auto& v : w.raw) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    v = seed;
  }
  for (unsigned r = 0; r < m; ++r) {
    auto* row = w.raw.data() + std::size_t(r) * (k / block) * bytes;
    for (unsigned b = 0; b < k / block; ++b) {
      auto* p = row + b * bytes;
      const std::uint16_t d =
          b % 2 ? 0x9400 : 0x1400;  // +/- 1/1024, unsaturated
      if (type == rocm::WeightType::kQ6_K) {
        std::memcpy(p + 208, &d, 2);
        // Include signed endpoints without overflowing any integer code.
        for (unsigned s = 0; s < 16; ++s)
          p[192 + s] = (s % 2 ? -1 : 1) * (1 + s % 7);
      } else {
        std::memcpy(p, &d, 2);
        if (type == rocm::WeightType::kQ4_K) {
          const std::uint16_t mn = 0x1000;
          std::memcpy(p + 2, &mn, 2);
        }
      }
    }
    if (type == rocm::WeightType::kQ4_K)
      gufo::quant::DequantizeQ4_K(row, w.values.data() + std::size_t(r) * k, k);
    else if (type == rocm::WeightType::kQ6_K)
      gufo::quant::DequantizeQ6_K(row, w.values.data() + std::size_t(r) * k, k);
    else
      gufo::quant::DequantizeQ5_0(row, w.values.data() + std::size_t(r) * k, k);
  }
  return w;
}
inline std::vector<float> Input(unsigned rows, unsigned k) {
  std::vector<float> x(std::size_t(rows) * k);
  // Identical row at every group/remainder boundary: bitwise invariance.
  for (unsigned r = 0; r < rows; ++r)
    for (unsigned i = 0; i < k; ++i)
      x[std::size_t(r) * k + i] =
          static_cast<float>(static_cast<int>((i * 37) % 101) - 50) * 0.00013F;
  return x;
}
inline float Dot(const float* w, const float* x, unsigned k) {
  std::array<float, 32> lanes{};
  for (unsigned lane = 0; lane < 32; ++lane) {
    for (unsigned i = lane * 4; i < k; i += 128) {
      float dot = 0;
      for (unsigned j = 0; j < 4 && i + j < k; ++j)
        dot = std::fma(w[i + j], x[i + j], dot);
      lanes[lane] += dot;
    }
  }
  for (unsigned stride = 16; stride != 0; stride /= 2) {
    const auto prior = lanes;
    for (unsigned lane = 0; lane < 32; ++lane)
      lanes[lane] = prior[lane] + prior[lane ^ stride];
  }
  return lanes[0];
}
inline void Near(float actual, float expected) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) > 2e-6F + 2e-5F * std::abs(expected))
    throw std::runtime_error(
        "packed MTP independent arithmetic reference mismatch");
}
inline void Projection(rocm::WeightType type, unsigned m, unsigned k) {
  const auto w = Make(type, m, k);
  Buffer<std::uint8_t> dw(w.raw.size());
  Hip(hipMemcpy(dw.data, w.raw.data(), w.raw.size(), hipMemcpyHostToDevice));
  std::vector<float> scalar;
  const auto input = Input(1, k);
  std::vector<float> expected(m);
  for (unsigned j = 0; j < m; ++j)
    expected[j] = Dot(w.values.data() + std::size_t(j) * k, input.data(), k);
  for (unsigned rows :
       {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 16U, 17U, 32U, 33U}) {
    const auto x = Input(rows, k);
    Buffer<float> dx(x.size()), dy(std::size_t(rows) * m);
    Hip(hipMemcpy(dx.data, x.data(), x.size() * sizeof(float),
                  hipMemcpyHostToDevice));
    rocm::SmallGemm(dw.data, type, dx.data, dy.data, rows, m, k, nullptr);
    std::vector<float> y(std::size_t(rows) * m);
    Hip(hipMemcpy(y.data(), dy.data, y.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
    for (unsigned r = 0; r < rows; ++r)
      for (unsigned j = 0; j < m; ++j) {
        Near(y[std::size_t(r) * m + j], expected[j]);
        if (rows != 1 && y[std::size_t(r) * m + j] != scalar[j])
          throw std::runtime_error(
              "packed MTP projection depends on token group");
      }
    if (rows == 1)
      scalar = y;
  }
}

inline void Hc(unsigned rows) {
  constexpr unsigned hidden = 2560, dim = 10240, rank = 320;
  const auto xn = Input(rows, dim);
  const auto down = Make(rocm::WeightType::kQ4_K, rank, dim);
  const auto up = Make(rocm::WeightType::kQ5_0, dim, rank);
  const auto inject = Make(rocm::WeightType::kQ4_K, 4, dim);
  Buffer<std::uint8_t> dd(down.raw.size()), du(up.raw.size()),
      di(inject.raw.size());
  Hip(hipMemcpy(dd.data, down.raw.data(), down.raw.size(),
                hipMemcpyHostToDevice));
  Hip(hipMemcpy(du.data, up.raw.data(), up.raw.size(), hipMemcpyHostToDevice));
  Hip(hipMemcpy(di.data, inject.raw.data(), inject.raw.size(),
                hipMemcpyHostToDevice));
  Buffer<float> dx(xn.size()), lo(std::size_t(rows) * rank), gate(xn.size()),
      mixed(std::size_t(rows) * hidden), logits(std::size_t(rows) * 4);
  Hip(hipMemcpy(dx.data, xn.data(), xn.size() * sizeof(float),
                hipMemcpyHostToDevice));
  rocm::SmallGemm(dd.data, rocm::WeightType::kQ4_K, dx.data, lo.data, rows,
                  rank, dim, nullptr);
  rocm::SiluScale(lo.data, 0.25F, std::size_t(rows) * rank, nullptr);
  rocm::SmallGemm(du.data, rocm::WeightType::kQ5_0, lo.data, gate.data, rows,
                  dim, rank, nullptr);
  rocm::SmallGemm(di.data, rocm::WeightType::kQ4_K, dx.data, logits.data, rows,
                  4, dim, nullptr);
  rocm::HcMixEpilogue(dx.data, gate.data, nullptr, mixed.data, nullptr, rows,
                      hidden, 4, nullptr);
  std::vector<float> result(std::size_t(rows) * hidden),
      actual_inject(std::size_t(rows) * 4);
  Hip(hipMemcpy(result.data(), mixed.data, result.size() * sizeof(float),
                hipMemcpyDeviceToHost));
  Hip(hipMemcpy(actual_inject.data(), logits.data,
                actual_inject.size() * sizeof(float), hipMemcpyDeviceToHost));
  std::array<float, rank> low{};
  for (unsigned i = 0; i < rank; ++i) {
    const float v =
        Dot(down.values.data() + std::size_t(i) * dim, xn.data(), dim);
    low[i] = v / (1 + std::exp(-v)) * 0.25F;
  }
  std::array<float, hidden> expected{};
  for (unsigned i = 0; i < hidden; ++i) {
    double sum = 0;
    for (unsigned s = 0; s < 4; ++s) {
      const unsigned j = s * hidden + i;
      const float g =
          Dot(up.values.data() + std::size_t(j) * rank, low.data(), rank);
      sum += xn[j] / (1 + std::exp(-double(g)));
    }
    expected[i] = static_cast<float>(sum / 4);
  }
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned s = 0; s < 4; ++s)
      Near(actual_inject[r * 4 + s],
           Dot(inject.values.data() + std::size_t(s) * dim, xn.data(), dim));
    for (unsigned i = 0; i < hidden; ++i) {
      Near(result[std::size_t(r) * hidden + i], expected[i]);
      if (result[std::size_t(r) * hidden + i] != result[i])
        throw std::runtime_error("packed HC depends on row group");
    }
  }
}
}  // namespace packed_mtp_test
#endif
