#ifndef GUFO_CORE_MEMORY_BUDGET_HPP_
#define GUFO_CORE_MEMORY_BUDGET_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace gufo::core {

/// Serve's post-loading host execution/cache reserve, not a weight-load gate.
inline constexpr std::uint64_t kHostExecutionReserveBytes = std::uint64_t{4}
                                                            << 30;

[[nodiscard]] inline std::size_t AddMemoryBytes(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("memory reservation overflows");
  return a + b;
}

/// Used while initialization is quiescent, never to sample a running session.
[[nodiscard]] inline std::size_t PendingMemoryBytes(std::size_t claimed,
                                                    std::size_t allocated) {
  if (allocated > claimed)
    throw std::logic_error("allocated execution memory exceeds its claim");
  return claimed - allocated;
}

/// Subtraction avoids overflow even with adversarial allocation sizes.
[[nodiscard]] constexpr bool FitsMemoryBudget(std::size_t available,
                                              std::size_t allocation,
                                              std::size_t protected_bytes) {
  return protected_bytes <= available &&
         allocation <= available - protected_bytes;
}

struct ProtectedCacheHeadroom {
  std::size_t automatic;
  std::size_t ceiling;
};

/// AUTO is half available memory. Deduct half the liability (rounded upward)
/// before its separate 32-GiB cap; protect the full amount from the ceiling.
/// Resident allocations were already removed by the provider's free sample.
[[nodiscard]] inline ProtectedCacheHeadroom ProtectCacheHeadroom(
    std::size_t automatic, std::size_t ceiling, std::size_t pending) {
  if (pending > ceiling)
    throw std::invalid_argument(
        "outstanding execution memory exceeds remaining capacity");
  const auto half = pending / 2 + pending % 2;
  return {automatic > half ? automatic - half : 0, ceiling - pending};
}

}  // namespace gufo::core
#endif
