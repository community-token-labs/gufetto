#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_COLD_SCORE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_COLD_SCORE_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace gufo::models::qwen38_flash_next {

inline constexpr std::size_t kColdScoreMaxTokens = 131072;
inline constexpr std::size_t kColdScoreMaxRows = 8;
inline constexpr std::size_t kColdScoreMaxPasses = 4096;
inline constexpr std::size_t kColdScoreMaxVocab = 248320;

// One executor, not per slot. S remains the ordinary speculative capacity;
// widening the two distinct head pools must not widen ordinary batch scratch.
struct DiagnosticLogitCapacity {
  std::uint32_t vocab{}, max_batch{}, max_speculative{}, max_logit_rows{};
  std::size_t head_bytes{}, widening_bytes{}, batch_device_bytes{};
  bool operator==(const DiagnosticLogitCapacity&) const = default;
  static constexpr std::optional<DiagnosticLogitCapacity> Plan(
      std::uint32_t vocab, std::uint32_t batch, std::uint32_t speculative,
      bool enabled) noexcept {
    if (!vocab || !batch || !speculative || speculative > batch)
      return std::nullopt;
    const auto rows =
        enabled ? std::max(speculative,
                           std::min<std::uint32_t>(kColdScoreMaxRows, batch))
                : speculative;
    const std::uint64_t row = std::uint64_t{vocab} * sizeof(float);
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (rows > limit / row || 8 * std::min(8U, speculative) > limit / row)
      return std::nullopt;
    const auto gross = static_cast<std::size_t>(row * rows);
    const auto delta = static_cast<std::size_t>(row * (rows - speculative));
    return DiagnosticLogitCapacity{
        vocab,
        batch,
        speculative,
        rows,
        gross,
        delta,
        static_cast<std::size_t>(row * 8 * std::min(8U, speculative))};
  }
};

/// Same per-step cap and lookahead cap as the ordinary isolated runner.
/// Does not select the separate 8192-row NPU paired route.
[[nodiscard]] constexpr std::size_t DiagnosticPrefillCount(
    std::size_t remaining, std::size_t budget, std::size_t capacity,
    std::size_t through_capacity) noexcept {
  return std::min(
      {remaining, budget,
       remaining <= through_capacity ? through_capacity : capacity});
}

/// Intersection of a completed pass [first,end) with terminal [suffix,N).
[[nodiscard]] constexpr std::size_t DiagnosticCaptureRows(
    std::size_t first, std::size_t end, std::size_t suffix) noexcept {
  return end > suffix ? end - std::max(first, suffix) : 0;
}

// Shared by native passes and host controls: keep progress and ledger checks
// in the production loop, not in a separately maintained fake algorithm.
template<class Step>
void CollectColdScorePasses(std::size_t tokens, std::uint32_t rows,
                            std::span<std::uint32_t> ends, std::size_t& passes,
                            Step step, const std::function<void()>& progress) {
  passes = 0;
  std::size_t captured = 0;
  for (;;) {
    if (passes == ends.size())
      throw std::runtime_error("cold score pass limit");
    const auto next = step();
    if (!next.pass_end || next.pass_end > tokens ||
        (passes && next.pass_end <= ends[passes - 1]) ||
        next.captured_rows > rows - captured)
      throw std::runtime_error("cold score capture progress");
    ends[passes++] = next.pass_end;
    captured += next.captured_rows;
    if (progress)
      progress();
    if (next.complete) {
      if (next.pass_end != tokens || captured != rows)
        throw std::runtime_error("cold score rows incomplete");
      return;
    }
  }
}

// Typed core only: no JSON body, wire identity or HTTP scratch. The one fixed
// service phase holds an exact input clone, <=8 full rows and concrete controls
// (including their embedded pass ledger and sizeof(DiagnosticMemoryClaim)).
// Native widening is persistent; request arrays stay lazy inside the promise.
// This arithmetic neither activates the process claim nor proves residency.
inline std::optional<std::size_t> ColdScoreWorkspaceBytes(
    const DiagnosticLogitCapacity& native, std::size_t context,
    std::size_t controls) noexcept {
  const auto checked = DiagnosticLogitCapacity::Plan(
      native.vocab, native.max_batch, native.max_speculative, true);
  if (!checked || !context || !controls || native.vocab > kColdScoreMaxVocab ||
      *checked != native)
    return std::nullopt;
  const auto rows = std::min(
      {kColdScoreMaxRows, context, std::size_t{native.max_logit_rows}});
  std::size_t bytes = controls;
  for (const auto term :
       {native.widening_bytes, native.widening_bytes,
        std::min(context, kColdScoreMaxTokens) * sizeof(std::int32_t),
        rows * native.vocab * sizeof(float)}) {
    if (term > std::numeric_limits<std::size_t>::max() - bytes)
      return std::nullopt;
    bytes += term;
  }
  return bytes;
}

}  // namespace gufo::models::qwen38_flash_next
#endif
