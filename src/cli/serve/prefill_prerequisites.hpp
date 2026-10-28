#ifndef GUFO_CLI_SERVE_PREFILL_PREREQUISITES_HPP_
#define GUFO_CLI_SERVE_PREFILL_PREREQUISITES_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace gufo::server::detail {

enum class PrefillPrerequisiteMode { kLegacy, kAdaptive, kSerialShadow };

struct PrefillPrerequisites {
  PrefillPrerequisiteMode mode{PrefillPrerequisiteMode::kLegacy};
  std::size_t tokens{0};

  [[nodiscard]] bool Enabled() const noexcept {
    return mode != PrefillPrerequisiteMode::kLegacy;
  }
  void Validate(std::size_t slots) const {
    if (mode != PrefillPrerequisiteMode::kLegacy &&
        mode != PrefillPrerequisiteMode::kAdaptive &&
        mode != PrefillPrerequisiteMode::kSerialShadow)
      throw std::invalid_argument("unknown experimental prefill policy");
    if ((!Enabled() && tokens != 0) ||
        (Enabled() && (tokens < 64 || tokens > 4224 || slots > 8)))
      throw std::invalid_argument("invalid experimental prefill policy/budget");
  }
};

struct PrefillPrerequisiteBudget {
  std::size_t prompt_tokens{0};
  std::size_t verifier_rows_upper{0};
};

/// No implicit checkpoint grid: the pool and model own legal boundaries.
[[nodiscard]] inline std::size_t ClipSinglePass(
    std::size_t offset, std::size_t frontier, std::size_t budget,
    std::size_t capacity, std::size_t next_boundary) noexcept {
  if (offset >= frontier)
    return 0;
  auto count = std::min({frontier - offset, budget, capacity});
  if (next_boundary > offset)
    count = std::min(count, next_boundary - offset);
  return count;
}

[[nodiscard]] inline PrefillPrerequisiteBudget SelectPrerequisiteBudget(
    const PrefillPrerequisites& config, std::size_t ready,
    std::span<const std::size_t> remaining) {
  if (!config.Enabled())
    throw std::logic_error("legacy policy has no prerequisite budget");
  if (ready == 0)
    return {4096, 0};
  if (config.mode == PrefillPrerequisiteMode::kAdaptive)
    return {config.tokens, 0};
  if (ready > 7 || remaining.size() != ready)
    throw std::invalid_argument("serial shadow needs a complete ready cohort");
  std::size_t rows = 0;
  for (const auto cap : remaining)
    rows += std::min(cap, std::size_t{8});
  if (rows >= config.tokens)
    throw std::invalid_argument("no positive serial-shadow prompt cut");
  return {config.tokens - rows, rows};
}

/// Backend route provenance is retained separately from model-filtered count.
enum class DecodeRoute {
  kNone,
  kGenericScalar,
  kGenericSequential,
  kEngineScalar,
  kEngineBatch
};

[[nodiscard]] inline DecodeRoute QfnBackendRoute(
    bool use_mtp, std::size_t backend_count, std::size_t max_tokens) noexcept {
  if (backend_count == 0)
    return DecodeRoute::kNone;
  if (backend_count >= 2)
    return use_mtp ? DecodeRoute::kEngineBatch
                   : DecodeRoute::kGenericSequential;
  return use_mtp && max_tokens != 1 ? DecodeRoute::kEngineScalar
                                    : DecodeRoute::kGenericScalar;
}

/// A finite roster, not a mutable due-bit counter. Newcomers join NEXT cohort.
class ReadyDecodeCohort {
public:
  void Freeze(std::span<const std::uint64_t> ids) {
    if (!Empty() || ids.size() > ids_.size())
      throw std::logic_error("invalid ready cohort freeze");
    for (std::size_t i = 0; i < ids.size(); ++i)
      if (std::find(ids.begin(), ids.begin() + i, ids[i]) != ids.begin() + i)
        throw std::invalid_argument("duplicate ready cohort member");
    std::copy(ids.begin(), ids.end(), ids_.begin());
    count_ = ids.size();
  }
  [[nodiscard]] bool Contains(std::uint64_t id) const noexcept {
    return std::find(ids_.begin(), ids_.begin() + count_, id) !=
           ids_.begin() + count_;
  }
  void ServedOrDeferred(std::uint64_t id) noexcept {
    const auto end = ids_.begin() + count_;
    const auto found = std::find(ids_.begin(), end, id);
    if (found != end) {
      std::move(found + 1, end, found);
      --count_;
    }
  }
  [[nodiscard]] bool Empty() const noexcept { return count_ == 0; }

private:
  std::array<std::uint64_t, 8> ids_{};
  std::size_t count_{0};
};

}  // namespace gufo::server::detail
#endif
