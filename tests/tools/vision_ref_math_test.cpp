#include "tools/quality/vision_ref_math.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using vision_ref::Matrix;

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void near(float actual, long double expected, long double tolerance,
          const std::string& message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(static_cast<long double>(actual) - expected) > tolerance)
    throw std::runtime_error(
        message + ": actual=" + std::to_string(actual) +
        " expected=" + std::to_string(static_cast<double>(expected)));
}

template<typename F>
void rejects(F action, const std::string& message) {
  bool threw = false;
  try {
    action();
  } catch (const std::exception&) {
    threw = true;
  }
  require(threw, "missing exception: " + message);
}

// Test-side forward raster -> merge-block index, deliberately not the module's
// inverse row decoder. No production code or module-private helpers are used.
size_t block_row(size_t y, size_t x, size_t grid_width) {
  return ((y / 2) * (grid_width / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
}

void test_patchify() {
  constexpr size_t width = 64, height = 96;
  std::vector<unsigned char> rgb(width * height * 3);
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x)
      for (size_t c = 0; c < 3; ++c)
        rgb[(y * width + x) * 3 + c] =
            static_cast<unsigned char>((y * 7 + x * 13 + c * 59) % 256);
  const Matrix patches = vision_ref::patchify(rgb, width, height);
  require(patches.rows == 24 && patches.cols == 1536, "patch shape");
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x)
      for (size_t c = 0; c < 3; ++c) {
        const size_t row = block_row(y / 16, x / 16, width / 16);
        const size_t col = c * 512 + (y % 16) * 16 + x % 16;
        // Exact rational oracle: the fused subtraction is exact for every
        // byte, leaving one F32 rounding at division. This detects the old
        // pixel*(1/255) expression, not just endpoint normalization.
        const int pixel = rgb[(y * width + x) * 3 + c];
        const float expected = static_cast<float>((2 * pixel - 255) / 255.0L);
        near(patches(row, col), expected, 0,
             "patch C,Y,X ordering and normalization");
        near(patches(row, col + 256), expected, 0,
             "still-frame temporal duplication");
      }
  const Matrix black =
      vision_ref::patchify(std::vector<unsigned char>(32 * 32 * 3, 0), 32, 32);
  const Matrix white = vision_ref::patchify(
      std::vector<unsigned char>(32 * 32 * 3, 255), 32, 32);
  for (float value : black.values)
    near(value, -1, 0, "black endpoint");
  for (float value : white.values)
    near(value, 1, 0, "white endpoint");
  const Matrix middle = vision_ref::patchify(
      std::vector<unsigned char>(32 * 32 * 3, 127), 32, 32);
  near(middle(0, 0), static_cast<float>(-1.0L / 255), 0,
       "fused processor normalization at pixel 127");
  const float unfused = ((127.0f * (1.0f / 255.0f)) - 0.5f) / 0.5f;
  require(middle(0, 0) != unfused,
          "normalization regression discriminates unfused arithmetic");
}

void test_positions() {
  Matrix table(48 * 48, 3);
  for (size_t y = 0; y < 48; ++y)
    for (size_t x = 0; x < 48; ++x) {
      table(y * 48 + x, 0) = static_cast<float>(y);
      table(y * 48 + x, 1) = static_cast<float>(x);
      table(y * 48 + x, 2) = static_cast<float>(y * x);
    }
  // Bilinear interpolation preserves these three analytic surfaces.
  Matrix actual(24, 3);
  std::fill(actual.values.begin(), actual.values.end(), 2.0f);
  vision_ref::add_positions(actual, table, 64, 96);
  for (size_t y = 0; y < 6; ++y)
    for (size_t x = 0; x < 4; ++x) {
      const size_t row = block_row(y, x, 4);
      const long double sy = 47.0L * y / 5, sx = 47.0L * x / 3;
      near(actual(row, 0), sy + 2, 5e-6L, "position height align-corners");
      near(actual(row, 1), sx + 2, 5e-6L, "position width align-corners");
      near(actual(row, 2), sy * sx + 2, 3e-4L, "position four-tap surface");
    }
  near(actual(block_row(0, 0, 4), 2), 2, 0, "top-left corner");
  near(actual(block_row(0, 3, 4), 1), 49, 0, "top-right corner");
  near(actual(block_row(5, 0, 4), 0), 49, 0, "bottom-left corner");
  near(actual(block_row(5, 3, 4), 2), 2211, 0, "bottom-right corner");

  Matrix sparse(2304, 1), sample(24, 1);
  sparse(9 * 48 + 15, 0) = 2;
  sparse(9 * 48 + 16, 0) = 3;
  sparse(10 * 48 + 15, 0) = 5;
  sparse(10 * 48 + 16, 0) = 7;
  vision_ref::add_positions(sample, sparse, 64, 96);
  near(sample(block_row(1, 1, 4), 0),
       (3.0L / 5) * (2.0L / 3 + 2) + (2.0L / 5) * (5.0L / 3 + 14.0L / 3), 5e-6L,
       "independent four-tap weights");

  Matrix identity(2304, 3);
  vision_ref::add_positions(identity, table, 768, 768);
  for (size_t y = 0; y < 48; ++y)
    for (size_t x = 0; x < 48; ++x)
      for (size_t c = 0; c < 3; ++c)
        near(identity(block_row(y, x, 48), c), table(y * 48 + x, c), 0,
             "identity position grid");
}

void test_linear_norm_gelu_add() {
  Matrix x(2, 3), w(3, 2);
  x.values = {1, 2, 3, -1, 0, 2};
  w.values = {1, 4, 2, 5, 3, 6};
  const Matrix y = vision_ref::linear(x, w, {0.5f, -1.0f});
  const std::vector<float> expected = {14.5f, 31, 5.5f, 7};
  require(y.values == expected, "linear input-output weight layout and bias");

  Matrix cancellation(1, 3), ones(3, 1);
  cancellation.values = {16777216.0f, 1.0f, -16777216.0f};
  ones.values = {1, 1, 1};
  near(vision_ref::linear(cancellation, ones, {3})(0, 0), 3, 0,
       "F32 ordered linear; bias last");
  Matrix fused(1, 2), fw(2, 1);
  fused.values = {-1, 1.0f + 0x1p-23f};
  fw.values = {1, 1.0f - 0x1p-23f};
  near(vision_ref::linear(fused, fw, {0})(0, 0), 0, 0,
       "linear must not contract FMA");

  Matrix norm_input(2, 4);
  norm_input.values = {1, 2, 3, 4, 9, 9, 9, 9};
  const std::vector<float> gain = {1, 2, -1, 0.5f}, bias = {0, 1, 2, -3};
  const Matrix norm = vision_ref::layer_norm(norm_input, gain, bias);
  for (size_t c = 0; c < 4; ++c) {
    const long double want =
        ((static_cast<long double>(c) + 1 - 2.5L) /
         std::sqrt(1.25L + static_cast<long double>(1e-6f))) *
            gain[c] +
        bias[c];
    near(norm(0, c), want, 3e-7L, "population layer norm");
    near(norm(1, c), bias[c], 0, "constant layer norm");
  }
  const Matrix custom = vision_ref::layer_norm(norm_input, gain, bias, 0.75f);
  near(custom(0, 0), -1.5L / std::sqrt(2.0L), 1e-7L, "custom norm epsilon");

  const std::vector<float> samples = {-20, -3, -1, -0.25f, 0, 0.25f, 1, 3, 20};
  for (bool approximate : {false, true}) {
    Matrix activation(1, samples.size());
    activation.values = samples;
    vision_ref::gelu(activation, approximate);
    for (size_t i = 0; i < samples.size(); ++i) {
      const long double a = samples[i];
      const long double value =
          approximate ? a / 2 *
                            (1 + std::tanh(std::sqrt(2 / std::acos(-1.0L)) *
                                           (a + 0.044715L * a * a * a)))
                      : a / 2 * (1 + std::erf(a / std::sqrt(2.0L)));
      near(activation(0, i), value, 3e-7L,
           approximate ? "tanh GELU" : "erf GELU");
    }
    Matrix extreme(1, 2);
    extreme.values = {-std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max()};
    vision_ref::gelu(extreme, approximate);
    near(extreme(0, 0), 0, 0, "negative GELU saturation");
    near(extreme(0, 1), std::numeric_limits<float>::max(), 0,
         "positive GELU saturation");
  }
  Matrix residual = x;
  vision_ref::add_inplace(residual, x);
  vision_ref::add_inplace(residual, residual);
  for (size_t i = 0; i < x.values.size(); ++i)
    near(residual.values[i], 4 * x.values[i], 0,
         "residual including self alias");
}

void test_rope() {
  for (double theta : {10000.0, 16.0}) {
    Matrix q(8, 16 * 72), k(8, 16 * 72);
    for (size_t r = 0; r < q.rows; ++r)
      for (size_t c = 0; c < q.cols; ++c) {
        q(r, c) = static_cast<float>(static_cast<int>(c % 13) - 6) / 8;
        k(r, c) = static_cast<float>(static_cast<int>(c % 17) - 8) / 8;
      }
    const Matrix original_q = q, original_k = k;
    vision_ref::apply_rope(q, k, 64, 32, 16, theta);
    for (size_t y = 0; y < 2; ++y)
      for (size_t x = 0; x < 4; ++x) {
        const size_t row = block_row(y, x, 4);
        for (size_t head = 0; head < 16; ++head)
          for (size_t i = 0; i < 36; ++i) {
            const long double coord = i < 18 ? y : x;
            const long double angle =
                coord / std::pow(static_cast<long double>(theta),
                                 static_cast<long double>(i % 18) / 18);
            const size_t lo = head * 72 + i, hi = lo + 36;
            for (bool query : {false, true}) {
              const Matrix& before = query ? original_q : original_k;
              const Matrix& after = query ? q : k;
              const long double a = before(row, lo), b = before(row, hi);
              near(after(row, lo), a * std::cos(angle) - b * std::sin(angle),
                   4e-7L, "RoPE split-half low, H/W frequency layout");
              near(after(row, hi), b * std::cos(angle) + a * std::sin(angle),
                   4e-7L, "RoPE split-half high, head separation");
            }
          }
      }
  }
}

// Independent scalar long-double attention. Used only on small fixtures whose
// dyadic Q/K products sum exactly in F32; no reuse of module dot/softmax logic.
std::vector<long double> attention_oracle(const Matrix& q, const Matrix& k,
                                          const Matrix& v, size_t heads) {
  std::vector<long double> result(q.values.size());
  const size_t dim = q.cols / heads;
  for (size_t row = 0; row < q.rows; ++row)
    for (size_t head = 0; head < heads; ++head) {
      std::vector<long double> logits(k.rows);
      for (size_t key = 0; key < k.rows; ++key) {
        for (size_t d = 0; d < dim; ++d)
          logits[key] += static_cast<long double>(q(row, head * dim + d)) *
                         k(key, head * dim + d);
        logits[key] /= std::sqrt(static_cast<long double>(dim));
      }
      const long double maximum =
          *std::max_element(logits.begin(), logits.end());
      long double total = 0;
      for (long double& logit : logits) {
        logit = std::exp(logit - maximum);
        total += logit;
      }
      for (size_t d = 0; d < dim; ++d) {
        long double sum = 0;
        for (size_t key = 0; key < k.rows; ++key)
          sum += (logits[key] / total) * v(key, head * dim + d);
        result[row * q.cols + head * dim + d] = sum;
      }
    }
  return result;
}

void test_attention() {
  // Exercise the actual 16x72 layout and default head count independently of
  // the smaller multi-key oracle fixtures below. With one key, P is exactly 1.
  Matrix model_q(1, 1152), model_k(1, 1152), model_v(1, 1152);
  for (size_t c = 0; c < 1152; ++c) {
    model_q(0, c) = static_cast<float>(c % 5) / 8;
    model_k(0, c) = static_cast<float>(c % 7) / 8;
    model_v(0, c) = static_cast<float>(c) / 16;
  }
  require(
      vision_ref::attention(model_q, model_k, model_v).values == model_v.values,
      "default 16-head attention with head_dim=72");
  for (size_t n : {1, 8, 9, 32, 33, 129}) {
    Matrix q(n, 16), k(n, 16), v(n, 16);
    for (size_t r = 0; r < n; ++r)
      for (size_t c = 0; c < 16; ++c) {
        q(r, c) =
            static_cast<float>(static_cast<int>((r + 3 * c) % 9) - 4) / 128;
        k(r, c) =
            static_cast<float>(static_cast<int>((3 * r + c) % 7) - 3) / 128;
        v(r, c) =
            static_cast<float>(static_cast<int>((7 * r + 5 * c) % 19) - 9) / 4;
      }
    const auto expected = attention_oracle(q, k, v, 2);
    const Matrix actual = vision_ref::attention(q, k, v, 2);
    const Matrix small_block = vision_ref::attention(q, k, v, 2, 7);
    require(actual.values == small_block.values,
            "attention query blocking must be numerically invariant");
    for (size_t i = 0; i < actual.values.size(); ++i)
      near(actual.values[i], expected[i], 2e-7L,
           "near-uniform attention tail N=" + std::to_string(n));

    std::fill(q.values.begin(), q.values.end(), 0);
    std::fill(v.values.begin(), v.values.end(), 0);
    for (size_t c = 0; c < v.cols; ++c) {
      if (n == 1) {
        v(0, c) = static_cast<float>(c + 1);
      } else {
        v(0, c) = 100000000;
        v(1, c) = static_cast<float>(c + 1);
        v(2, c) = -100000000;
      }
    }
    const Matrix uniform = vision_ref::attention(q, k, v, 2);
    for (size_t r = 0; r < n; ++r)
      for (size_t c = 0; c < v.cols; ++c)
        near(uniform(r, c), static_cast<long double>(c + 1) / n, 2e-7L,
             "F64 uniform cancellation PV N=" + std::to_string(n));
  }

  Matrix q(2, 3), k(2, 3), v(2, 3);
  q.values = {16777216, 1, -16777216, 16777216, 1, -16777216};
  k.values = {1, 1, 1, 0, 0, 0};
  v.values = {3, 3, 3, -1, -1, -1};
  const Matrix ordered = vision_ref::attention(q, k, v, 1);
  for (float value : ordered.values)
    near(value, 1, 0, "QK must accumulate F32, not F64");

  Matrix fq(2, 2), fk(2, 2), fv(2, 2);
  fq.values = {-1, 1.0f + 0x1p-23f, -1, 1.0f + 0x1p-23f};
  fk.values = {1, 1.0f - 0x1p-23f, 0, 0};
  fv.values = {1e20f, 1e20f, -1e20f, -1e20f};
  for (float value : vision_ref::attention(fq, fk, fv, 1).values)
    near(value, 0, 0, "QK must not contract FMA");

  Matrix stable_q(3, 1), stable_k(3, 1), stable_v(3, 1);
  stable_q.values = {100, -100, 0};
  stable_k.values = {100, 99, -100};
  stable_v.values = {1, 2, -3};
  const auto stable_expected =
      attention_oracle(stable_q, stable_k, stable_v, 1);
  const Matrix stable = vision_ref::attention(stable_q, stable_k, stable_v, 1);
  for (size_t i = 0; i < stable.values.size(); ++i)
    near(stable.values[i], stable_expected[i], 1e-7L,
         "stable maximum-subtracted full-key softmax");

  // Cross the key tile boundary with a final-key-only signal. Analytic full
  // noncausal uniform attention gives one at every query, including row zero.
  Matrix tile_q(4097, 1), tile_k(4097, 1), tile_v(4097, 1);
  tile_v(4096, 0) = 4097;
  for (float value : vision_ref::attention(tile_q, tile_k, tile_v, 1).values)
    near(value, 1, 0, "4096-key tile tail and noncausal last-key contribution");
}

void test_rejections() {
  const size_t maximum = std::numeric_limits<size_t>::max();
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  rejects([] { Matrix x(0, 2); }, "empty matrix");
  rejects([&] { Matrix x(maximum, 2); }, "shape multiplication overflow");
  Matrix x(1, 2), w(2, 1), bad(1, 2);
  rejects([&] { (void)x(1, 0); }, "mutable index bounds");
  const Matrix& cx = x;
  rejects([&] { (void)cx(0, 2); }, "const index bounds");
  bad.values.pop_back();
  rejects([&] { vision_ref::gelu(bad, false); }, "mutated storage");
  bad = x;
  bad.rows = maximum;
  rejects([&] { vision_ref::gelu(bad, false); }, "mutated overflowing shape");
  bad = x;
  bad.values[0] = nan;
  rejects([&] { vision_ref::linear(bad, w, {0}); }, "nonfinite linear input");
  rejects([&] { vision_ref::linear(x, w, {infinity}); }, "nonfinite bias");
  rejects([&] { vision_ref::linear(x, x, {0, 0}); }, "linear shape");
  rejects([&] { vision_ref::linear(x, w, {}); }, "linear bias shape");
  rejects([&] { vision_ref::layer_norm(bad, {1, 1}, {0, 0}); },
          "nonfinite norm input");
  rejects([&] { vision_ref::layer_norm(x, {1, 1}, {0, 0}, 0); },
          "zero epsilon");
  rejects([&] { vision_ref::layer_norm(x, {1, 1}, {0, 0}, infinity); },
          "nonfinite epsilon");
  rejects([&] { vision_ref::layer_norm(x, {1}, {0, 0}); }, "gain shape");
  rejects([&] { vision_ref::gelu(bad, false); }, "nonfinite GELU");
  rejects([&] { vision_ref::add_inplace(x, bad); }, "nonfinite residual");
  rejects([&] { vision_ref::add_inplace(x, w); }, "residual shape");
  Matrix huge(1, 2);
  huge.values = {std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
  rejects([&] { vision_ref::layer_norm(huge, {1, 1}, {0, 0}); },
          "norm mean overflow");
  huge.values[1] = -huge.values[0];
  rejects([&] { vision_ref::layer_norm(huge, {1, 1}, {0, 0}); },
          "norm variance overflow");
  Matrix hw(2, 1);
  hw.values = {2, 0};
  rejects([&] { vision_ref::linear(huge, hw, {0}); }, "linear output overflow");
  rejects([&] { vision_ref::add_inplace(huge, huge); }, "residual overflow");

  const std::vector<unsigned char> rgb(32 * 32 * 3);
  rejects([&] { vision_ref::patchify(rgb, 16, 32); }, "too-small image");
  rejects([&] { vision_ref::patchify(rgb, 33, 32); }, "unaligned image");
  rejects([&] { vision_ref::patchify(rgb, 8192, 4096); }, "oversize image");
  rejects([&] { vision_ref::patchify(rgb, maximum - 31, 32); },
          "image product overflow");
  rejects([&] { vision_ref::patchify({}, 32, 32); }, "RGB exact byte count");
  auto extra = rgb;
  extra.push_back(0);
  rejects([&] { vision_ref::patchify(extra, 32, 32); }, "RGB trailing byte");
  Matrix grid(4, 72), other = grid, table(2304, 72);
  rejects([&] { vision_ref::add_positions(grid, table, 64, 32); },
          "position grid shape");
  rejects([&] { vision_ref::add_positions(grid, grid, 32, 32); },
          "position table shape/alias");
  table.values[0] = nan;
  rejects([&] { vision_ref::add_positions(grid, table, 32, 32); },
          "nonfinite positions");
  rejects([&] { vision_ref::apply_rope(grid, grid, 32, 32, 1); }, "RoPE alias");
  rejects([&] { vision_ref::apply_rope(grid, other, 32, 32, 0); },
          "RoPE zero heads");
  rejects([&] { vision_ref::apply_rope(grid, other, 32, 32, 2); },
          "RoPE head dimension");
  for (double theta : {0.0, -1.0, static_cast<double>(nan),
                       static_cast<double>(infinity), 1e-300})
    rejects([&] { vision_ref::apply_rope(grid, other, 32, 32, 1, theta); },
            "invalid theta");
  other.values[0] = infinity;
  rejects([&] { vision_ref::apply_rope(grid, other, 32, 32, 1); },
          "nonfinite RoPE");
  rejects([&] { vision_ref::attention(x, x, x, 0); }, "attention zero heads");
  rejects([&] { vision_ref::attention(x, x, x, 3); },
          "attention head divisibility");
  rejects([&] { vision_ref::attention(x, x, x, 1, 0); },
          "attention zero query block");
  rejects([&] { vision_ref::attention(x, w, x, 1); }, "attention shape");
  rejects([&] { vision_ref::attention(x, x, bad, 1); },
          "nonfinite attention values");
  Matrix overflow(1, 1);
  overflow.values[0] = std::numeric_limits<float>::max();
  rejects([&] { vision_ref::attention(overflow, overflow, overflow, 1); },
          "QK overflow");
}

}  // namespace

int main() {
  try {
    test_patchify();
    test_positions();
    test_linear_norm_gelu_add();
    test_rope();
    test_attention();
    test_rejections();
    std::cout << "vision_ref_math_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "vision_ref_math_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
