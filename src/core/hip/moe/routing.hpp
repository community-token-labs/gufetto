#ifndef GUFO_CORE_HIP_MOE_ROUTING_HPP_
#define GUFO_CORE_HIP_MOE_ROUTING_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Expert routing and MoE epilogue launchers shared by MoE models.
/// Activations are row-major float32 [tokens][dim] unless noted; every launch
/// is asynchronous on `stream`.
namespace gufo::hip::moe {

// Routed expert GEMMs: assignment rows are compacted by expert with every
// bucket padded to a 16-row tile (`pad_bounds`), so a token tile never
// straddles experts; `rows_out` maps a compact row to its (token, slot)
// output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;

/// counts[e] = number of (token, slot) pairs routed to expert e.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);

/// Softmax over `n_experts` logits per token (rows `stride` apart), top-k
/// selection, renormalized weights. ids [t][k] int32, weights [t][k] f32.
void RouterTopK(const float* logits, std::uint32_t stride, std::int32_t* ids,
                float* weights, std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i]
///           + sigmoid(gate[t * gate_stride]) * shared[t][i].
void MoeEpilogue(const float* expert_out, const float* weights,
                 const float* shared, const float* gate,
                 std::uint32_t gate_stride, float* out, std::uint32_t n_tokens,
                 std::uint32_t k, std::uint32_t dim, hipStream_t stream);
/// Same contract, four adjacent lanes per thread (`dim % 4 == 0`, else the
/// scalar kernel runs).
void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     const float* shared, const float* gate,
                     std::uint32_t gate_stride, float* out,
                     std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                     hipStream_t stream);
/// MoeEpilogueVec4 over F16 expert rows.
void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        const float* shared, const float* gate,
                        std::uint32_t gate_stride, float* out,
                        std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream);

/// Routed expert GEMMs. RoutedCompact sorts the (token, slot) assignments
/// by expert into `rows_token`/`rows_slot` (RoutedCompactRows(slots,
/// experts) entries, -1 for padding) with every bucket padded to 16 rows
/// (`pad_bounds`, experts + 1 entries) from the per-expert `counts`.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

}  // namespace gufo::hip::moe

#endif  // GUFO_CORE_HIP_MOE_ROUTING_HPP_
