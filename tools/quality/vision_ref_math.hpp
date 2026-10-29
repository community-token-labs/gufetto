#pragma once

#include <cstddef>
#include <vector>

// Independent CPU arithmetic for the pinned Qwen4-Exp vision specification.
// No production encoder dependencies. Compile with -ffp-contract=off, without
// fast-math/reassociation: linear and QK reductions are ordered F32 operations.
namespace vision_ref {

// Nonempty row-major storage. Public metadata/storage are revalidated by each
// operation. Invalid shapes, nonfinite inputs/results and overflow throw.
struct Matrix {
  size_t rows, cols;
  std::vector<float> values;
  Matrix(size_t r, size_t c);
  float& operator()(size_t r, size_t c);
  const float& operator()(size_t r, size_t c) const;
};

// weight_in_out is [input, output], explicitly transposed from GGUF [out, in].
// Bias is added after the ordered F32 dot product, not used as its initial sum.
Matrix linear(const Matrix& x, const Matrix& weight_in_out,
              const std::vector<float>& bias);
// Ordered F32 two-pass population mean/variance, epsilon inside the sqrt.
Matrix layer_norm(const Matrix& x, const std::vector<float>& gain,
                  const std::vector<float>& bias, float eps = 1e-6f);
// approximate=true selects the F32 tanh formula; false selects F32 erf.
void gelu(Matrix& x, bool approximate);
void add_inplace(Matrix& x, const Matrix& other);

// Width/height are PIXELS for all three image-grid APIs. RGB is interleaved,
// already aligned to 32; dimensions >=32, area <=16777216. No resize is done.
// Rows are 2x2 merge-block-major; columns are C,T,Y,X (3,2,16,16), with a
// duplicated still frame. Pinned TorchvisionBackend fuses mean/std with the
// inverse rescale: normalization is (F32(pixel)-127.5f)/127.5f.
Matrix patchify(const std::vector<unsigned char>& rgb, size_t width,
                size_t height);
// A 48x48 row-major table; F32 bilinear align_corners=True, four taps in h,w
// order. x and position_table must not alias. No N*4*D gather temporary.
void add_positions(Matrix& x, const Matrix& position_table, size_t width,
                   size_t height);
// Exactly 72 columns/head: split-half rotation with [h18,w18,h18,w18].
// q and k must not alias. The vision config inherits theta=10000.0 from the
// pinned RotaryEmbeddingConfigMixin, not the text model's different theta.
void apply_rope(Matrix& q, Matrix& k, size_t width, size_t height,
                size_t heads = 16, double theta = 10000.0);
// Full, noncausal self-attention; equal Q/K/V shapes. Ordered F32 QK and F32
// scaling, stable full-key F64 softmax, ordered F64 PV, one final F32 cast.
// Heads are serial, query blocks capped at 128, QK key tiles capped at 4096.
// Scratch is O(min(query_block,128)*N + N*head_dim), never all-head N*N.
Matrix attention(const Matrix& q, const Matrix& k, const Matrix& v,
                 size_t heads = 16, size_t query_block = 128);

}  // namespace vision_ref
