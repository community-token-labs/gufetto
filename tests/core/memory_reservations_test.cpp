#include "src/core/memory_reservations.hpp"

#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
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

int main() {
  TestDomainAndIdentity();
  TestExclusiveBindingAndRebind();
  TestOwnershipOutlivesCaller();
  TestConcurrentBinding();
  TestConcurrentOwnerCreation();
}
