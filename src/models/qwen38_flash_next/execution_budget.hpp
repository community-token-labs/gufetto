#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_EXECUTION_BUDGET_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_EXECUTION_BUDGET_HPP_

#include "src/core/memory_budget.hpp"

namespace gufo::models::qwen38_flash_next {

enum class SnapshotRoute { kNative, kComponentServing };
inline constexpr std::size_t kNativeSnapshotSpareBlocks = 2;

// Component capture copies private state into component backing; it never
// uses SaveSnapshot/AllocateSpare. Preserve the legacy allowance on all native
// routes, without manufacturing dormant spare allocations in serving.
[[nodiscard]] inline std::size_t SnapshotSpareBytes(std::size_t mutable_bytes,
                                                    SnapshotRoute route) {
  return route == SnapshotRoute::kNative
             ? core::AddMemoryBytes(mutable_bytes, mutable_bytes)
             : 0;
}

}  // namespace gufo::models::qwen38_flash_next
#endif
