#include "src/core/memory_reservations.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

using gufo::core::NamedReservations;
using gufo::core::ReservationDomain;

static_assert(!std::is_copy_constructible_v<NamedReservations>);
static_assert(!std::is_move_constructible_v<NamedReservations>);
static_assert(!std::is_copy_constructible_v<NamedReservations::Binding>);

void TestDomainAndIdentity() {
  const auto domain = ReservationDomain::Create();
  const auto first = NamedReservations::Create(domain);
  const auto second = NamedReservations::Create(domain);
  assert(first->id() != 0 && second->id() != first->id());
  assert(first->SharesDomain(*second));
  const auto independent =
      NamedReservations::Create(ReservationDomain::Create());
  assert(!first->SharesDomain(*independent));
  assert(
      NamedReservations::Create()->SharesDomain(*NamedReservations::Create()));
  const auto old = second->id();
  auto transient = NamedReservations::Create(domain);
  const auto retired = transient->id();
  transient.reset();
  const auto replacement = NamedReservations::Create(domain);
  assert(replacement->id() > retired && retired > old);
  try {
    (void)NamedReservations::Create(nullptr);
    assert(false);
  } catch (const std::invalid_argument& error) {
    assert(std::string_view(error.what()) ==
           "missing memory reservation domain");
  }
}

void TestExclusiveBindingAndRebind() {
  const auto owner = NamedReservations::Create(ReservationDomain::Create());
  const auto id = owner->id();
  assert(!owner->HasBinding());
  auto binding = owner->Bind();
  assert(owner->HasBinding() && binding->owner() == owner);
  for (int attempt = 0; attempt != 3; ++attempt) {
    try {
      (void)owner->Bind();
      assert(false);
    } catch (const std::logic_error& error) {
      // Independent literal pins the established public refusal text.
      assert(std::string_view(error.what()) ==
             "a second component backend on the same Model is not supported "
             "(one backend per Model); release the first or load a separate "
             "Model");
    }
    // Destruction of a failed/inactive token cannot clear the real binding.
    assert(owner->HasBinding());
  }
  auto copied_handle = binding;
  binding.reset();
  assert(owner->HasBinding());
  copied_handle.reset();
  assert(!owner->HasBinding());
  const auto rebound = owner->Bind();
  assert(rebound->owner() == owner && owner->id() == id);
}

void TestOwnershipOutlivesCaller() {
  auto domain = ReservationDomain::Create();
  auto owner = NamedReservations::Create(domain);
  const std::weak_ptr<ReservationDomain> domain_lifetime = domain;
  const std::weak_ptr<NamedReservations> model_lifetime = owner;
  auto binding = owner->Bind();
  owner.reset();
  domain.reset();
  assert(!model_lifetime.expired() && !domain_lifetime.expired());
  binding.reset();
  assert(model_lifetime.expired() && domain_lifetime.expired());
}

void TestConcurrentBinding() {
  const auto owner = NamedReservations::Create(ReservationDomain::Create());
  std::barrier ready(2);
  std::barrier attempted(2);
  std::atomic<int> accepted{}, refused{};
  auto bind = [&] {
    ready.arrive_and_wait();
    std::shared_ptr<NamedReservations::Binding> binding;
    try {
      binding = owner->Bind();
      ++accepted;
    } catch (const std::logic_error&) {
      ++refused;
    }
    // Hold the winner until both attempts have finished.
    attempted.arrive_and_wait();
  };
  std::jthread first(bind), second(bind);
  first.join();
  second.join();
  assert(accepted == 1 && refused == 1 && !owner->HasBinding());
  assert(owner->Bind());
}

void TestConcurrentOwnerCreation() {
  const auto domain = ReservationDomain::Create();
  std::vector<std::shared_ptr<NamedReservations>> owners(32);
  std::vector<std::jthread> threads;
  for (std::size_t i = 0; i != owners.size(); ++i)
    threads.emplace_back(
        [&, i] { owners[i] = NamedReservations::Create(domain); });
  threads.clear();  // Join before inspecting the independent vector elements.
  std::set<std::uint64_t> ids;
  for (const auto& owner : owners) {
    assert(owner && owner->SharesDomain(*owners.front()));
    assert(ids.insert(owner->id()).second);
    assert(!owner->HasBinding());
  }
}

// Independent rendering of the existing component-cache RAM branch. No named
// helper is used to calculate the zero-reservation reference.
std::size_t LegacyBudget(gufo::core::ReservationObservation sample,
                         gufo::core::NamedCacheBudgetOptions options) {
  const auto sub = [](auto a, auto b) { return a > b ? a - b : 0; };
  const auto available =
      std::min(sample.host_available, sample.cgroup_available);
  auto ceiling = std::min(sub(available, sample.host_reserve),
                          sub(sample.device_available, sample.device_reserve));
  auto automatic = std::min(available / 2, ceiling);
  if (options.ordinary_pending > ceiling)
    throw std::invalid_argument(
        "outstanding execution memory exceeds remaining capacity");
  automatic = sub(automatic,
                  options.ordinary_pending / 2 + options.ordinary_pending % 2);
  ceiling -= options.ordinary_pending;
  if (options.staging > ceiling)
    throw std::invalid_argument(
        "cache staging exceeds capacity after execution reservations");
  automatic += options.pinned / 2;
  ceiling += options.pinned;
  const auto maximum = sub(ceiling, options.staging);
  auto ram = std::min(options.requested, maximum);
  if (options.automatic)
    ram = std::min(ram, automatic);
  const auto limit =
      std::min({ram + options.pinned, maximum, options.ledger_limit});
  return std::min(ram, sub(limit, options.pinned));
}

void TestEmptyBudgetParity() {
  using namespace gufo::core;
  for (const std::size_t host : {64, 127, 256})
    for (const auto group : {std::numeric_limits<std::size_t>::max(),
                             std::size_t{80}, std::size_t{192}})
      for (const std::size_t device : {48, 200, 300})
        for (const std::size_t pending : {0, 1, 32, 70})
          for (const std::size_t staging : {0, 8, 40})
            for (const std::size_t pinned : {0, 1, 16})
              for (const std::size_t requested : {16, 128})
                for (const bool automatic : {false, true})
                  for (const std::size_t ledger : {8, 200}) {
                    const ReservationObservation observation{host, device,
                                                             group, 4, 3};
                    const NamedCacheBudgetOptions options{
                        requested, ledger, pending, staging, pinned, automatic};
                    std::optional<std::size_t> expected;
                    std::string reason;
                    try {
                      expected = LegacyBudget(observation, options);
                    } catch (const std::invalid_argument& error) {
                      reason = error.what();
                    }
                    try {
                      const auto result = ResolveNamedCacheBudget(
                          observation, {}, {}, 0, options);
                      assert(expected && result.bytes == *expected);
                      assert(result.before_debit == result.after_debit);
                    } catch (const std::invalid_argument& error) {
                      assert(!expected && reason == error.what());
                    }
                  }
}

void TestNamedBudget() {
  using namespace gufo::core;
  const ReservationObservation raw{2048, 2048};
  NamedCacheBudgetOptions options{4096, 8192, 0, 0, 0, true};
  auto result = ResolveNamedCacheBudget(raw, {}, {128, 128}, 128, options);
  assert(result.before_debit == 1024 && result.after_debit == 896);
  assert(result.bytes == 896 && result.physical_ceiling == 1920);
  const ReservationObservation partial{1984, 1984};
  result =
      ResolveNamedCacheBudget(partial, {64, 64, 0}, {64, 64}, 128, options);
  assert(result.before_debit == 1024 && result.bytes == 896);
  options.requested = 512;  // Existing AUTO cap, before the full debit.
  result =
      ResolveNamedCacheBudget(partial, {64, 64, 0}, {64, 64}, 128, options);
  assert(result.before_debit == 512 && result.bytes == 384);
  options.automatic = false;
  options.requested = 768;
  assert(ResolveNamedCacheBudget(partial, {64, 64, 0}, {64, 64}, 128, options)
             .bytes == 640);
  options.requested = 128;
  assert(ResolveNamedCacheBudget(raw, {}, {128, 128}, 128, options).bytes == 0);
  options.requested = 127;
  try {
    (void)ResolveNamedCacheBudget(raw, {}, {128, 128}, 128, options);
    assert(false);
  } catch (const std::invalid_argument& error) {
    assert(std::string_view(error.what()) ==
           "named reservations exceed cache policy budget");
  }
  options = {4096, 8192, 0, 0, 0, true};
  result = ResolveNamedCacheBudget({4096, 4096, 1024}, {128, 0, 64}, {64, 0},
                                   128, options);
  assert(result.before_debit == 544 && result.bytes == 416);
  // Another owner affects physical capacity, not our baseline or policy debit.
  result = ResolveNamedCacheBudget({2048, 1024}, {64, 0, 0}, {64, 700}, 128,
                                   options);
  assert(result.before_debit == 1024 && result.after_debit == 896);
  assert(result.physical_ceiling == 324 && result.bytes == 324);
  try {
    (void)ResolveNamedCacheBudget(raw, {}, {0, 2049}, 128, options);
    assert(false);
  } catch (const std::invalid_argument& error) {
    assert(std::string_view(error.what()) ==
           "named reservations exceed physical capacity");
  }
  try {
    (void)ResolveNamedCacheBudget(raw, {129, 0, 0}, {}, 128, options);
    assert(false);
  } catch (const std::invalid_argument&) {
  }
  try {
    (void)ResolveNamedCacheBudget(
        {std::numeric_limits<std::size_t>::max(), 2048}, {1, 0, 0}, {}, 1,
        options);
    assert(false);
  } catch (const std::overflow_error&) {
  }
}

int main() {
  TestDomainAndIdentity();
  TestExclusiveBindingAndRebind();
  TestOwnershipOutlivesCaller();
  TestConcurrentBinding();
  TestConcurrentOwnerCreation();
  TestEmptyBudgetParity();
  TestNamedBudget();
}
