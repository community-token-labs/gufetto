#include <cassert>
#include <limits>
#include <memory>
#include <stdexcept>

#include "src/models/qwen/vision/output_pool.hpp"
#include "src/models/qwen/vision/prompt.hpp"

using gufo::models::qwen::vision::OutputPool;

struct FakeStorage final : OutputPool::Storage {
  bool reader_complete{false};
  bool failed_reader{false};
  bool retained{true};
  bool ever_encoded{true};
  bool TryEvict() override {
    if (failed_reader)
      throw std::runtime_error("reader completion failed");
    if (!reader_complete)
      return false;
    retained = false;
    return true;
  }
};

int main() {
  assert(OutputPool{}.Capacity() == 65536);
  OutputPool pool(128);
  const auto a = pool.GetOwner("a", 64);
  const auto b = pool.GetOwner("b", 64);
  const auto c = pool.GetOwner("c", 64);
  assert(pool.GetStats().rows == 0);
  assert(pool.GetOwner("a", 64) == a);
  auto first = pool.TryAcquire(a);
  auto duplicate = pool.TryAcquire(a);
  auto second = pool.TryAcquire(b);
  assert(first && duplicate && second);
  assert(pool.GetStats().rows == 128);
  auto storage = std::make_shared<FakeStorage>();
  assert(first->GetStorage([&] { return storage; }) == storage);
  assert(duplicate->GetStorage([] { return nullptr; }) == storage);
  for (int i = 0; i < 100; ++i)
    assert(!pool.TryAcquire(c));
  assert(pool.GetStats().peak_rows == 128);
  assert(pool.GetStats().evictions == 0);
  first.reset();
  assert(!pool.TryAcquire(c));  // duplicate still pins A
  duplicate.reset();
  assert(!pool.TryAcquire(c));  // outstanding device reader still pins storage
  storage->failed_reader = true;
  bool reader_error = false;
  try {
    (void)pool.TryAcquire(c);
  } catch (const std::runtime_error&) {
    reader_error = true;
  }
  assert(reader_error && pool.GetStats().rows == 128 && storage->retained);
  storage->failed_reader = false;
  storage->reader_complete = true;
  auto third = pool.TryAcquire(c);
  assert(third && !storage->retained && storage->ever_encoded);
  assert(pool.GetStats().rows == 128 && pool.GetStats().evictions == 1);
  assert(!pool.TryAcquire(a));  // B/C active, replay queues instead of growing
  second.reset();
  auto replay = pool.TryAcquire(a);  // idle B evicted; metadata A survives
  assert(replay->GetStorage([] { return nullptr; }) == storage);
  replay->RecordEncode(storage->ever_encoded, 64);
  storage->retained = true;
  assert(pool.GetStats().encodes == 1 && pool.GetStats().reencodes == 1);
  assert(pool.GetStats().encoded_rows == 64 &&
         pool.GetStats().peak_encoded_rows == 64);

  // The last released/used idle owner survives an LRU eviction.
  third.reset();
  replay.reset();
  auto readmit_b = pool.TryAcquire(b);
  assert(readmit_b);
  const auto previous_evictions = pool.GetStats().evictions;
  auto retained_a = pool.TryAcquire(a);
  assert(retained_a && pool.GetStats().evictions == previous_evictions);

  bool rejected = false;
  try {
    (void)pool.GetOwner("oversized", 129);
  } catch (const std::length_error&) {
    rejected = true;
  }
  assert(rejected);
  bool wrong_identity = false;
  try {
    (void)pool.GetOwner("a", 32);
  } catch (const std::logic_error&) {
    wrong_identity = true;
  }
  assert(wrong_identity);
  bool wrong_pool = false;
  try {
    (void)OutputPool{}.TryAcquire(a);
  } catch (const std::invalid_argument&) {
    wrong_pool = true;
  }
  assert(wrong_pool);
  bool zero = false;
  try {
    OutputPool invalid(0);
  } catch (const std::invalid_argument&) {
    zero = true;
  }
  assert(zero);

  OutputPool large(std::numeric_limits<std::uint32_t>::max());
  auto full = large.GetOwner("full", std::numeric_limits<std::uint32_t>::max());
  auto full_lease = large.TryAcquire(full);
  assert(full_lease && !large.TryAcquire(large.GetOwner("extra", 1)));

  bool overflow = false;
  try {
    (void)gufo::models::qwen::vision::OutputReservationBytes(UINT32_MAX,
                                                             UINT32_MAX);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  assert(overflow);
  gufo::models::qwen::vision::Prompt prompt;
  prompt.images.resize(2);
  prompt.images[0].grid = {32768, 8, 8};
  prompt.images[1].grid = {40000, 8, 8};
  assert(!prompt.IntersectsImages(0, 32768));
  assert(prompt.IntersectsImages(32768, 1));
  assert(!prompt.IntersectsImages(32768, 0));
  assert(prompt.IntersectsImages(32831, 2));
  assert(!prompt.IntersectsImages(32832, 40000 - 32832));
  assert(!prompt.IntersectsImages(40064, 4096));
  assert(!prompt.IntersectsImages(UINT32_MAX, UINT32_MAX));
  prompt.images[0].grid = {0, 65535, 65535};
  prompt.images[1].grid = {0, 65535, 65535};
  bool rows_overflow = false;
  try {
    (void)prompt.ImageRows();
  } catch (const std::length_error&) {
    rows_overflow = true;
  }
  assert(rows_overflow);

  // RAII cancellation/failure returns pins even when storage creation throws.
  OutputPool failure(64);
  try {
    auto lease = failure.TryAcquire(failure.GetOwner("failed", 64));
    (void)lease->GetStorage([]() -> std::shared_ptr<OutputPool::Storage> {
      throw std::runtime_error("fixture encode failure");
    });
  } catch (const std::runtime_error&) {
  }
  assert(failure.TryAcquire(failure.GetOwner("recovered", 64)));
  // A surviving lease owns its accounting domain after the pool facade dies.
  std::shared_ptr<OutputPool::Lease> survivor;
  {
    OutputPool temporary(1);
    survivor = temporary.TryAcquire(temporary.GetOwner("survivor", 1));
  }
  survivor.reset();
}
