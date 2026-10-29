#ifndef GUFO_MODELS_QWEN_VISION_POLICY_HPP_
#define GUFO_MODELS_QWEN_VISION_POLICY_HPP_

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "src/core/memory_budget.hpp"
#include "src/models/qwen/vision/input_budget.hpp"
#include "src/models/qwen/vision/memory_budget.hpp"

namespace gufo::models::qwen::vision {

/// Qwen27 keeps pre-pool serving semantics unless a pool is explicitly chosen.
/// A cap alone only limits input; it does not opt into shared retention or its
/// admission claims. Qwen pool/cap options remain unqualified.
struct Policy {
  bool enabled;
  std::optional<std::uint32_t> pool_rows;
  std::uint32_t max_tokens_per_slot;

  [[nodiscard]] bool ProtectedExecution() const noexcept {
    return pool_rows.has_value();
  }
  [[nodiscard]] std::size_t PositionClaimBytes(std::uint32_t context) const {
    return ProtectedExecution() ? InputPositionBytes(context) : 0;
  }
  [[nodiscard]] std::size_t SharedClaimBytes(std::uint32_t context,
                                             std::uint32_t width) const {
    return ProtectedExecution()
               ? core::AddMemoryBytes(EncoderWorkspaceBytes(context),
                                      OutputReservationBytes(*pool_rows, width))
               : 0;
  }
  [[nodiscard]] std::optional<std::size_t> StateCapacityBytes(
      std::optional<std::size_t> heap_free, std::size_t host_ceiling,
      std::size_t dispatch_reserve) const {
    if (!ProtectedExecution())
      return heap_free;  // Legacy Qwen uses raw HIP free, even below reserves.
    if (!heap_free)
      throw std::runtime_error("cannot query device execution memory capacity");
    const auto heap =
        *heap_free > dispatch_reserve ? *heap_free - dispatch_reserve : 0;
    return std::min(heap, host_ceiling);
  }
  [[nodiscard]] std::optional<std::uint32_t> PromptLimit() const noexcept {
    if (pool_rows || max_tokens_per_slot != 0)
      return max_tokens_per_slot;
    return std::nullopt;
  }
  [[nodiscard]] std::optional<std::uint32_t> CountLimit(
      std::uint32_t context) const noexcept {
    return PromptLimit()
               ? std::optional{ImageTokenLimit(context, max_tokens_per_slot)}
               : std::nullopt;
  }
};

/// The shared encoder accepts only these model-specific projector widths.
/// Resolve here also for direct Encoder/Open callers and installed-model loads.
/// Absence is distinct from numeric zero: explicit zero is always invalid.
[[nodiscard]] inline Policy ResolvePolicy(
    std::uint32_t output_width,
    std::optional<std::uint32_t> pool_rows = std::nullopt,
    std::uint32_t max_tokens_per_slot = 0, bool enabled = true) {
  if (pool_rows == 0)
    throw std::invalid_argument("vision-pool-tokens must be positive");
  if (!pool_rows && output_width == 2560)
    pool_rows = kDefaultOutputPoolRows;
  return {enabled, pool_rows, max_tokens_per_slot};
}

}  // namespace gufo::models::qwen::vision
#endif
