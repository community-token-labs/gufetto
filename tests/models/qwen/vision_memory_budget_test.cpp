#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>

#include "src/core/memory_budget.hpp"
#include "src/models/qwen/vision/input_budget.hpp"
#include "src/models/qwen/vision/memory_budget.hpp"

int main() {
  using namespace gufo::core;
  using namespace gufo::models::qwen::vision;
  static_assert(EncoderWorkspaceBytes() == 3471376384ULL);
  static_assert(EncoderWorkspaceBytes(0) == 0);
  static_assert(EncoderWorkspaceBytes(UINT32_MAX) == EncoderWorkspaceBytes());
  // The 4096-patch kernel uses a smaller tile, but nearby shapes need128.
  for (const std::uint64_t patches : {4092ULL, 4096ULL, 4100ULL, 65536ULL}) {
    const auto tile = patches == 4096 ? 64ULL : 128ULL;
    const auto exact = 40416 * patches + 96 * tile * patches + 4608 * tile +
                       std::min<std::uint64_t>(16ULL << 20, 768 * patches);
    assert(EncoderWorkspaceBytes(patches / 4) >= exact);
  }
  assert(FullContextInputBytes(262144, 2560) == 2687500352ULL);
  assert(FullContextInputBytes(262144, 5120) == 5371854912ULL);
  // Integrating the cap must shrink only embeddings, not late-image positions.
  assert(InputReservationBytes(262144, 2560) ==
         FullContextInputBytes(262144, 2560));
  assert(InputReservationBytes(262144, 5120, 16384) ==
         16384ULL * 5120 * sizeof(float) + 262144ULL * 12 + 64);
  assert(PendingMemoryBytes(1000, 800) == 200);
  bool underclaim = false;
  try {
    (void)PendingMemoryBytes(800, 1000);
  } catch (const std::logic_error&) {
    underclaim = true;
  }
  assert(underclaim);
  for (const bool input : {false, true}) {
    bool overflow = false;
    try {
      if (input)
        (void)FullContextInputBytes(UINT32_MAX, UINT32_MAX);
      else
        (void)AddMemoryBytes(std::numeric_limits<std::size_t>::max(), 1);
    } catch (const std::overflow_error&) {
      overflow = true;
    }
    assert(overflow);
  }
  const auto protected_memory = ProtectCacheHeadroom(8000, 12000, 4000);
  assert(protected_memory.automatic == 6000 &&
         protected_memory.ceiling == 8000);
  const auto odd = ProtectCacheHeadroom(8000, 12000, 4001);
  assert(odd.automatic == 5999 && odd.ceiling == 7999);
  assert(ProtectCacheHeadroom(5, 10, 10).ceiling == 0);
  bool refused = false;
  try {
    (void)ProtectCacheHeadroom(8000, 12000, 12001);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  assert(refused);
  // A segment alone fits; the same segment plus promised allocations does not.
  assert(!FitsMemoryBudget(1000, 801, 200));
  assert(FitsMemoryBudget(1000, 800, 200));
  assert(!FitsMemoryBudget(1000, std::numeric_limits<std::size_t>::max(), 200));
  const auto frozen = PendingMemoryBytes(1000, 800);
  const auto grow = [frozen](std::size_t available, std::size_t bytes) {
    return FitsMemoryBudget(available, bytes, frozen);
  };
  // Allocation, failure, free and reallocation never publish fewer credits
  // against an older free-memory sample. The guard deliberately stays frozen.
  for (const auto available : {1000U, 850U, 1000U, 800U})
    assert(grow(available, 801) == false);

  SerializedEncoderWorkspace workspace;
  std::barrier start(2);
  std::atomic<unsigned> active{};
  const auto encode = [&] {
    start.arrive_and_wait();
    for (unsigned i = 0; i < 50; ++i) {
      const auto lease = workspace.Acquire();
      assert(lease.owns_lock());
      assert(active.fetch_add(1) == 0);
      std::this_thread::yield();
      assert(active.fetch_sub(1) == 1);
    }
  };
  std::thread first(encode), second(encode);
  first.join();
  second.join();
  try {
    const auto lease = workspace.Acquire();
    throw std::runtime_error("encode failure");
  } catch (const std::runtime_error&) {
  }
  const auto recovered = workspace.Acquire();
  assert(recovered.owns_lock());
}
