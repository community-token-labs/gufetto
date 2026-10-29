#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "src/core/memory_budget.hpp"
#include "src/core/reclaimable_memory.hpp"
#include "src/models/qwen/vision/input_budget.hpp"
#include "src/models/qwen/vision/memory_budget.hpp"
#include "src/models/qwen38_flash_next/execution_budget.hpp"

namespace {
void ScopedServingAuthority() {
  using gufo::core::ReclaimableMemory;
  using gufo::core::ReclaimableMemoryBinding;
  struct Model {
    std::shared_ptr<ReclaimableMemory> executor, encoder, allocator;
    unsigned native_calls{};
    void NativeImage() {
      if (encoder)
        encoder->AllocateImage(1, [] {});
      ++native_calls;
    }
  };
  auto model = std::make_shared<Model>();  // retained outside serving ownership
  const auto bind = [&] {
    return ReclaimableMemoryBinding::Bind(
        model->executor, std::make_shared<ReclaimableMemory>(),
        [model](auto budget) noexcept {
          model->executor = model->encoder = model->allocator =
              std::move(budget);
        });
  };
  model->NativeImage();
  {
    auto serving = bind();
    assert(model->executor && model->encoder && model->allocator);
    const auto original = model->executor;
    bool duplicate = false;
    try {
      (void)bind();
    } catch (const std::logic_error& error) {
      duplicate =
          std::string_view(error.what()) ==
          "a second component backend on the same Model is not supported "
          "(one backend per Model); release the first or load a separate Model";
    }
    assert(duplicate && model->executor == original &&
           model->encoder == original && model->allocator == original);
    model->executor->Configure(10, [] { return 10; });
    auto image = model->executor->TryAcquireImage();
    model->NativeImage();
    model->encoder->RetireImage(1, [] { return true; });
  }
  assert(!model->executor && !model->encoder && !model->allocator);
  model->NativeImage();  // successful backend teardown restores native route
  try {
    auto serving = bind();
    throw std::invalid_argument(
        "startup failed before authority configuration");
  } catch (const std::invalid_argument&) {
  }
  assert(!model->executor && !model->encoder && !model->allocator);
  model->NativeImage();  // failed initialization also restores native route
  assert(model->native_calls == 4);
}

void PermanentPinCushion() {
  using gufo::core::ReclaimableMemory;
  std::size_t free = 800;
  auto memory = std::make_shared<ReclaimableMemory>();
  memory->Configure(400, [&] { return free; });
  const auto [needed, available] = memory->ImageHeadroom(100);
  assert(needed == 500 && available == 800);
  auto pin = memory->TryAcquireImage(100);
  assert(pin && memory->GetStats().outstanding == 500);
  assert(!memory->TryAcquireGrowth(301));
  auto growth = memory->TryAcquireGrowth(300);
  assert(growth);
  free -= 300;  // segment-rounded permanent physical backing
  growth.reset();
  assert(pin->Complete() && !pin->Complete());
  assert(memory->GetStats().outstanding == 0);
  free -= 100;  // frozen hard liability becomes resident later
  auto image = memory->TryAcquireImage();
  assert(image && memory->GetStats().outstanding == 400);
  image.reset();

  // Without the cushion, permanent backing could spend those extra100 bytes.
  free = 800;
  pin = memory->TryAcquireImage();
  growth = memory->TryAcquireGrowth(400);
  assert(pin && growth);
  free -= 400;
  growth.reset();
  pin.reset();
  free -= 100;
  assert(!memory->TryAcquireImage());  // no reclaimable backing remains

  free = 800;
  image = memory->TryAcquireImage();
  memory->AllocateImage(100, [&] { free -= 100; });
  bool overflow = false;
  try {
    (void)memory->TryAcquireImage(std::numeric_limits<std::size_t>::max() -
                                  350);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  assert(overflow && memory->GetStats().images == 1);
  assert(!memory->GetStats().reclaiming);
  memory->RetireImage(100, [&] {
    free += 100;
    return true;
  });
  assert(memory->GetStats().outstanding == 400);
}

void ReclaimableClaims() {
  using gufo::core::ReclaimableMemory;
  std::size_t free = 200;
  auto memory = std::make_shared<ReclaimableMemory>();
  memory->Configure(400, [&] { return free; });
  unsigned attempts = 0;
  memory->SetReclaimer([&](std::size_t deficit) {
    ++attempts;
    assert(deficit == 200);
    // Logical metadata removal cannot be credited as physical headroom.
    return attempts == 1;
  });
  assert(!memory->TryAcquireImage());
  assert(attempts == 2 && memory->GetStats().outstanding == 0);
  auto growth = memory->TryAcquireGrowth(100);
  assert(growth && !memory->TryAcquireImage());
  growth.reset();
  bool pinned = true;
  memory->SetReclaimer([&](std::size_t deficit) {
    assert(!memory->TryAcquireGrowth(1));
    if (pinned)
      return false;
    free += deficit;  // successful physical release
    return true;
  });
  assert(!memory->TryAcquireImage());
  pinned = false;
  auto image = memory->TryAcquireImage();
  assert(image && free == 400);
  assert(memory->GetStats().outstanding == 400);
  auto peer = memory->TryAcquireImage();
  assert(peer && memory->GetStats().images == 2);
  assert(memory->GetStats().outstanding == 400);  // one shared envelope
  assert(!memory->TryAcquireGrowth(1));
  memory->AllocateImage(150, [&] { free -= 150; });
  assert(memory->GetStats().resident == 150);
  assert(memory->GetStats().outstanding == free);
  try {
    memory->AllocateImage(20, [] { throw std::bad_alloc(); });
    assert(false);
  } catch (const std::bad_alloc&) {
  }
  assert(memory->GetStats().resident == 150);
  memory->RetireImage(50, [] { return false; });
  assert(memory->GetStats().resident == 150);  // failure gives no credit
  memory->RetireImage(50, [&] {
    free += 50;
    return true;
  });
  assert(memory->GetStats().resident == 100);
  image->Complete();
  image->Complete();
  assert(memory->GetStats().images == 1);
  peer.reset();  // cancellation/normal destruction releases final claim
  assert(memory->GetStats().outstanding == 0);
  assert(memory->GetStats().resident ==
         100);  // outputs/positions stay resident
  growth = memory->TryAcquireGrowth(free);
  assert(growth && !memory->TryAcquireGrowth(1));
  growth.reset();
  image.reset();
  image = memory->TryAcquireImage();
  assert(image && memory->GetStats().outstanding == 300);
  memory->RetireImage(100, [&] {
    free += 100;
    return true;
  });
  assert(memory->GetStats().outstanding == 400 && free == 400);
  image.reset();
  bool unadmitted = false;
  try {
    memory->AllocateImage(1, [] {});
  } catch (const std::logic_error&) {
    unadmitted = true;
  }
  assert(unadmitted);
  memory->SetReclaimer({});

  // Hold a real commit ticket across a competing image admission, rather
  // than only checking free bytes before the allocation begins.
  std::barrier rendezvous(2);
  std::thread committer([&] {
    const auto pending = memory->TryAcquireGrowth(100);
    assert(pending);
    rendezvous.arrive_and_wait();
    rendezvous.arrive_and_wait();
  });
  rendezvous.arrive_and_wait();
  assert(!memory->TryAcquireImage());
  rendezvous.arrive_and_wait();
  committer.join();
  assert(memory->TryAcquireImage());
}
}  // namespace

int main() {
  ReclaimableClaims();
  ScopedServingAuthority();
  PermanentPinCushion();
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
  using gufo::models::qwen38_flash_next::SnapshotRoute;
  using gufo::models::qwen38_flash_next::SnapshotSpareBytes;
  assert(SnapshotSpareBytes(118038528, SnapshotRoute::kNative) == 236077056);
  assert(SnapshotSpareBytes(118038528, SnapshotRoute::kComponentServing) == 0);
  constexpr std::size_t hard = 615813120 + 63569920 + 118366208;
  static_assert(hard == 797749248);
  assert(ProtectCacheHeadroom(11925708800ULL, 19088683008ULL, hard).automatic ==
         11526834176ULL);
  assert(ProtectCacheHeadroom(12011968512ULL, 19728969728ULL, hard).automatic ==
         11613093888ULL);
  assert(4 * (InputPositionBytes(262144) - 16) == 12583104);
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
