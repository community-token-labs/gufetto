#include "vision_ref_math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vision_ref {
namespace {

size_t product(size_t a, size_t b) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("vision_ref: shape product overflow");
  return a * b;
}

size_t matrix_size(size_t r, size_t c) {
  if (r == 0 || c == 0)
    throw std::invalid_argument("vision_ref: empty matrix");
  const size_t n = product(r, c);
  if (n > std::vector<float>().max_size())
    throw std::length_error("vision_ref: matrix too large");
  return n;
}

void finite(const std::vector<float>& values) {
  for (float value : values)
    if (!std::isfinite(value))
      throw std::domain_error("vision_ref: nonfinite input or result");
}

void validate(const Matrix& x) {
  if (matrix_size(x.rows, x.cols) != x.values.size())
    throw std::invalid_argument("vision_ref: storage/shape mismatch");
  finite(x.values);
}

void vector_shape(const std::vector<float>& x, size_t size) {
  if (x.size() != size)
    throw std::invalid_argument("vision_ref: vector shape mismatch");
  finite(x);
}

void same_shape(const Matrix& a, const Matrix& b) {
  if (a.rows != b.rows || a.cols != b.cols)
    throw std::invalid_argument("vision_ref: matrix shape mismatch");
}

size_t image_area(size_t width, size_t height) {
  if (width < 32 || height < 32 || width % 32 || height % 32)
    throw std::invalid_argument(
        "vision_ref: image dimensions must be >=32 and aligned to 32");
  const size_t area = product(width, height);
  if (area > 16777216)
    throw std::invalid_argument("vision_ref: image area exceeds 16777216");
  return area;
}

// Inverse of (block_y, block_x, inner_y, inner_x) flattening.
std::pair<size_t, size_t> patch_coordinates(size_t row, size_t grid_width) {
  return {2 * (row / (2 * grid_width)) + (row / 2) % 2,
          2 * ((row / 4) % (grid_width / 2)) + row % 2};
}

void image_shape(const Matrix& x, size_t width, size_t height) {
  if (x.rows != image_area(width, height) / 256)
    throw std::invalid_argument("vision_ref: image grid/row mismatch");
}

}  // namespace

Matrix::Matrix(size_t r, size_t c)
    : rows(r), cols(c), values(matrix_size(r, c), 0.0f) {}

float& Matrix::operator()(size_t r, size_t c) {
  if (r >= rows || c >= cols || matrix_size(rows, cols) != values.size())
    throw std::out_of_range("vision_ref: matrix index or storage");
  return values[r * cols + c];
}

const float& Matrix::operator()(size_t r, size_t c) const {
  if (r >= rows || c >= cols || matrix_size(rows, cols) != values.size())
    throw std::out_of_range("vision_ref: matrix index or storage");
  return values[r * cols + c];
}

Matrix linear(const Matrix& x, const Matrix& weight_in_out,
              const std::vector<float>& bias) {
  validate(x);
  validate(weight_in_out);
  if (x.cols != weight_in_out.rows)
    throw std::invalid_argument("vision_ref: linear input/weight mismatch");
  vector_shape(bias, weight_in_out.cols);
  Matrix out(x.rows, weight_in_out.cols);
  // Independent rows/columns only: never parallelize/reassociate a reduction.
#pragma omp parallel for num_threads(4) schedule(static)
  for (size_t row = 0; row < x.rows; ++row) {
    float* dst = out.values.data() + row * out.cols;
    for (size_t in = 0; in < x.cols; ++in) {
      const float a = x.values[row * x.cols + in];
      const float* w = weight_in_out.values.data() + in * out.cols;
#pragma omp simd
      for (size_t col = 0; col < out.cols; ++col)
        dst[col] += a * w[col];
    }
#pragma omp simd
    for (size_t col = 0; col < out.cols; ++col)
      dst[col] += bias[col];
  }
  finite(out.values);
  return out;
}

Matrix layer_norm(const Matrix& x, const std::vector<float>& gain,
                  const std::vector<float>& bias, float eps) {
  validate(x);
  vector_shape(gain, x.cols);
  vector_shape(bias, x.cols);
  if (!std::isfinite(eps) || eps <= 0)
    throw std::invalid_argument(
        "vision_ref: layer norm epsilon must be finite and positive");
  Matrix out(x.rows, x.cols);
  int invalid = 0;
#pragma omp parallel for num_threads(4) schedule(static) reduction(| : invalid)
  for (size_t row = 0; row < x.rows; ++row) {
    const float* src = x.values.data() + row * x.cols;
    float mean = 0.0f;
    for (size_t col = 0; col < x.cols; ++col)
      mean += src[col];
    mean /= static_cast<float>(x.cols);
    float variance = 0.0f;
    for (size_t col = 0; col < x.cols; ++col) {
      const float delta = src[col] - mean;
      variance += delta * delta;
    }
    variance /= static_cast<float>(x.cols);
    const float shifted_variance = variance + eps;
    if (!std::isfinite(mean) || !std::isfinite(shifted_variance)) {
      invalid = 1;
      continue;
    }
    const float inv = 1.0f / std::sqrt(shifted_variance);
#pragma omp simd
    for (size_t col = 0; col < x.cols; ++col)
      out.values[row * x.cols + col] =
          ((src[col] - mean) * inv) * gain[col] + bias[col];
  }
  if (invalid)
    throw std::domain_error("vision_ref: layer norm statistics overflow");
  finite(out.values);
  return out;
}

void gelu(Matrix& x, bool approximate) {
  validate(x);
#pragma omp parallel for num_threads(4) schedule(static)
  for (size_t i = 0; i < x.values.size(); ++i) {
    const float a = x.values[i];
    // Both F32 erf/tanh expressions saturate here. Avoid cubic overflow for
    // finite, extreme inputs without introducing an artificial NaN boundary.
    if (std::abs(a) > 10.0f) {
      x.values[i] = a > 0.0f ? a : -0.0f;
    } else if (approximate) {
      const float cube = (a * a) * a;
      const float inner = 0.7978845608028654f * (a + 0.044715f * cube);
      x.values[i] = (0.5f * a) * (1.0f + std::tanh(inner));
    } else {
      x.values[i] = (0.5f * a) * (1.0f + std::erf(a * 0.7071067811865475f));
    }
  }
  finite(x.values);
}

void add_inplace(Matrix& x, const Matrix& other) {
  validate(x);
  validate(other);
  same_shape(x, other);
#pragma omp parallel for simd num_threads(4) schedule(static)
  for (size_t i = 0; i < x.values.size(); ++i)
    x.values[i] += other.values[i];
  finite(x.values);
}

Matrix patchify(const std::vector<unsigned char>& rgb, size_t width,
                size_t height) {
  const size_t area = image_area(width, height);
  if (rgb.size() != product(area, 3))
    throw std::invalid_argument("vision_ref: RGB byte count mismatch");
  Matrix out(area / 256, 3 * 2 * 16 * 16);
#pragma omp parallel for num_threads(4) schedule(static)
  for (size_t row = 0; row < out.rows; ++row) {
    const auto [py, px] = patch_coordinates(row, width / 16);
    for (size_t c = 0; c < 3; ++c)
      for (size_t t = 0; t < 2; ++t)
        for (size_t y = 0; y < 16; ++y)
          for (size_t x = 0; x < 16; ++x) {
            const size_t source = ((py * 16 + y) * width + px * 16 + x) * 3 + c;
            const size_t target =
                row * out.cols + ((c * 2 + t) * 16 + y) * 16 + x;
            // Pinned TorchvisionBackend fuses rescale into mean/std:
            // (0.5 * inverse(1/255)) == 127.5, followed by F32 normalize.
            out.values[target] =
                (static_cast<float>(rgb[source]) - 127.5f) / 127.5f;
          }
  }
  return out;
}

void add_positions(Matrix& x, const Matrix& position_table, size_t width,
                   size_t height) {
  validate(x);
  validate(position_table);
  image_shape(x, width, height);
  if (&x == &position_table || position_table.rows != 48 * 48 ||
      position_table.cols != x.cols)
    throw std::invalid_argument("vision_ref: position table shape or alias");
  const size_t grid_h = height / 16, grid_w = width / 16;
#pragma omp parallel for num_threads(4) schedule(static)
  for (size_t row = 0; row < x.rows; ++row) {
    const auto [py, px] = patch_coordinates(row, grid_w);
    // Match the pinned utility's multiply-then-divide F32 coordinate order.
    const float sy =
        (static_cast<float>(py) * 47.0f) / static_cast<float>(grid_h - 1);
    const float sx =
        (static_cast<float>(px) * 47.0f) / static_cast<float>(grid_w - 1);
    const size_t y0 = static_cast<size_t>(std::floor(sy)),
                 x0 = static_cast<size_t>(std::floor(sx));
    const size_t y1 = std::min(y0 + 1, size_t(47)),
                 x1 = std::min(x0 + 1, size_t(47));
    const float dy = sy - static_cast<float>(y0),
                dx = sx - static_cast<float>(x0);
    const std::array<size_t, 4> taps = {y0 * 48 + x0, y0 * 48 + x1,
                                        y1 * 48 + x0, y1 * 48 + x1};
    const std::array<float, 4> weights = {
        (1.0f - dy) * (1.0f - dx), (1.0f - dy) * dx, dy * (1.0f - dx), dy * dx};
#pragma omp simd
    for (size_t col = 0; col < x.cols; ++col) {
      float position = 0.0f;
      for (size_t tap = 0; tap < 4; ++tap)
        position +=
            position_table.values[taps[tap] * x.cols + col] * weights[tap];
      x.values[row * x.cols + col] += position;
    }
  }
  finite(x.values);
}

void apply_rope(Matrix& q, Matrix& k, size_t width, size_t height, size_t heads,
                double theta) {
  validate(q);
  validate(k);
  same_shape(q, k);
  image_shape(q, width, height);
  const float base = static_cast<float>(theta);
  if (&q == &k || heads == 0 || q.cols % heads || q.cols / heads != 72 ||
      !std::isfinite(theta) || !std::isfinite(base) || base <= 0)
    throw std::invalid_argument(
        "vision_ref: RoPE requires distinct Q/K, 72 columns/head and positive "
        "F32 theta");
  std::array<float, 18> inverse{};
  for (size_t i = 0; i < inverse.size(); ++i) {
    // torch.arange(0,36,2,dtype=float) / 36, then F32 power and reciprocal.
    const float power = std::pow(base, static_cast<float>(2 * i) / 36.0f);
    inverse[i] = 1.0f / power;
    if (!std::isfinite(inverse[i]) || inverse[i] <= 0)
      throw std::domain_error("vision_ref: invalid RoPE frequency");
  }
  int invalid = 0;
#pragma omp parallel for num_threads(4) schedule(static) reduction(| : invalid)
  for (size_t row = 0; row < q.rows; ++row) {
    const auto [py, px] = patch_coordinates(row, width / 16);
    std::array<float, 36> cosine{}, sine{};
    bool valid = true;
    for (size_t i = 0; i < 36; ++i) {
      const float angle =
          static_cast<float>(i < 18 ? py : px) * inverse[i % 18];
      if (!std::isfinite(angle)) {
        valid = false;
        break;
      }
      cosine[i] = std::cos(angle);
      sine[i] = std::sin(angle);
    }
    if (!valid) {
      invalid = 1;
      continue;
    }
    for (Matrix* matrix : {&q, &k})
      for (size_t head = 0; head < heads; ++head) {
        float* dst = matrix->values.data() + row * q.cols + head * 72;
#pragma omp simd
        for (size_t i = 0; i < 36; ++i) {
          const float lo = dst[i], hi = dst[i + 36];
          dst[i] = lo * cosine[i] + (-hi) * sine[i];
          dst[i + 36] = hi * cosine[i] + lo * sine[i];
        }
      }
  }
  if (invalid)
    throw std::domain_error("vision_ref: RoPE angle overflow");
  finite(q.values);
  finite(k.values);
}

Matrix attention(const Matrix& q, const Matrix& k, const Matrix& v,
                 size_t heads, size_t query_block) {
  validate(q);
  validate(k);
  validate(v);
  same_shape(q, k);
  same_shape(q, v);
  if (heads == 0 || q.cols % heads || query_block == 0)
    throw std::invalid_argument(
        "vision_ref: attention head count or query block");
  const size_t n = q.rows, dim = q.cols / heads;
  const size_t block = std::min({query_block, size_t(128), n});
  const float scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim)));
  Matrix out(n, q.cols);
  std::vector<float> keys(product(dim, n));
  std::vector<float> scores(product(block, n));
  std::vector<double> probabilities(product(block, n));
  std::vector<double> values(product(n, dim));
  std::vector<double> accum(product(block, dim));
  // One head at a time: ~150 MiB maximum scratch, never N*N*heads.
  // Pack V once per head, avoiding a full-hidden-width stride for every query.
  for (size_t head = 0; head < heads; ++head) {
#pragma omp parallel for num_threads(4) schedule(static)
    for (size_t key = 0; key < n; ++key)
      for (size_t d = 0; d < dim; ++d) {
        keys[d * n + key] = k.values[key * k.cols + head * dim + d];
        values[key * dim + d] = v.values[key * v.cols + head * dim + d];
      }
    for (size_t begin = 0; begin < n; begin += block) {
      const size_t count = std::min(block, n - begin);
      std::fill(scores.begin(), scores.end(), 0.0f);
#pragma omp parallel for num_threads(4) schedule(static)
      for (size_t local = 0; local < count; ++local) {
        float* dst = scores.data() + local * n;
        const float* query =
            q.values.data() + (begin + local) * q.cols + head * dim;
        // Key-column tiling keeps the output tile hot across all d.
        for (size_t tile = 0; tile < n; tile += 4096) {
          const size_t end = tile + std::min(size_t(4096), n - tile);
          for (size_t d = 0; d < dim; ++d) {
            const float a = query[d];
            const float* key = keys.data() + d * n;
#pragma omp simd
            for (size_t j = tile; j < end; ++j)
              dst[j] += a * key[j];
          }
#pragma omp simd
          for (size_t j = tile; j < end; ++j)
            dst[j] *= scale;
        }
      }
      finite(scores);  // Outside OpenMP: exceptions must reach the caller.
      std::fill(accum.begin(), accum.end(), 0.0);
#pragma omp parallel for num_threads(4) schedule(static)
      for (size_t local = 0; local < count; ++local) {
        const float* logits = scores.data() + local * n;
        const double maximum = *std::max_element(logits, logits + n);
        double denominator = 0.0;
        double* probability_row = probabilities.data() + local * n;
        for (size_t key = 0; key < n; ++key) {
          probability_row[key] = std::exp(static_cast<double>(logits[key]) - maximum);
          denominator += probability_row[key];
        }
        double* sum = accum.data() + local * dim;
        for (size_t key = 0; key < n; ++key) {
          const double probability = probability_row[key] / denominator;
          const double* value = values.data() + key * dim;
#pragma omp simd
          for (size_t d = 0; d < dim; ++d)
            sum[d] += probability * value[d];
        }
        float* dst =
            out.values.data() + (begin + local) * out.cols + head * dim;
#pragma omp simd
        for (size_t d = 0; d < dim; ++d)
          dst[d] = static_cast<float>(sum[d]);
      }
    }
  }
  finite(out.values);
  return out;
}

}  // namespace vision_ref
