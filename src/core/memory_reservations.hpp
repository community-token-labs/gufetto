#ifndef GUFO_CORE_MEMORY_RESERVATIONS_HPP_
#define GUFO_CORE_MEMORY_RESERVATIONS_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "src/core/memory_budget.hpp"

namespace gufo::core {

struct MemoryLanes {
  std::size_t host{}, device{};
};
struct ReservationObservation {
  std::size_t host_available{}, device_available{};
  std::size_t cgroup_available = std::numeric_limits<std::size_t>::max();
  std::size_t host_reserve{}, device_reserve{};
};
// Validated credits for this owner's incremental backing, not gross
// allocations. cgroup is credit substantiated for every constraint used in
// cgroup_available.
struct ReservationMaterialization {
  std::size_t host{}, device{}, cgroup{};
};
struct NamedCacheBudgetOptions {
  std::size_t requested{}, ledger_limit{}, ordinary_pending{}, staging{},
      pinned{};
  bool automatic{};
};
struct NamedCacheBudget {
  std::size_t before_debit{}, after_debit{}, physical_ceiling{}, bytes{};
};

// RAM-enabled policy only. requested already includes the existing AUTO cap.
// Other owners affect pending but NEVER this owner's debit or materialization.
// Caller validates backing/observation attribution; this function proves only
// arithmetic. Physical availability is never reconstructed from policy credit.
[[nodiscard]] inline NamedCacheBudget ResolveNamedCacheBudget(
    ReservationObservation sample, ReservationMaterialization materialized,
    MemoryLanes pending, std::size_t debit, NamedCacheBudgetOptions options) {
  if (std::max({materialized.host, materialized.device, materialized.cgroup}) >
      debit)
    throw std::invalid_argument("named materialization exceeds its envelope");
  const auto subtract = [](std::size_t a, std::size_t b) {
    return a > b ? a - b : 0;
  };
  const auto host = std::min(sample.host_available, sample.cgroup_available);
  const MemoryLanes physical{
      subtract(host, sample.host_reserve),
      subtract(sample.device_available, sample.device_reserve)};
  const auto policy_host = std::min(
      AddMemoryBytes(sample.host_available, materialized.host),
      sample.cgroup_available == std::numeric_limits<std::size_t>::max()
          ? sample.cgroup_available
          : AddMemoryBytes(sample.cgroup_available, materialized.cgroup));
  const auto policy_ceiling = std::min(
      subtract(policy_host, sample.host_reserve),
      subtract(AddMemoryBytes(sample.device_available, materialized.device),
               sample.device_reserve));
  const auto protected_policy =
      ProtectCacheHeadroom(std::min(policy_host / 2, policy_ceiling),
                           policy_ceiling, options.ordinary_pending);
  if (options.staging > protected_policy.ceiling)
    throw std::invalid_argument(
        "cache staging exceeds capacity after execution reservations");
  const auto automatic =
      AddMemoryBytes(protected_policy.automatic, options.pinned / 2);
  const auto ceiling = AddMemoryBytes(protected_policy.ceiling, options.pinned);
  const auto maximum = subtract(ceiling, options.staging);
  auto candidate = std::min(options.requested, maximum);
  if (options.automatic)
    candidate = std::min(candidate, automatic);
  const auto limit = std::min({AddMemoryBytes(candidate, options.pinned),
                               maximum, options.ledger_limit});
  const auto before = std::min(candidate, subtract(limit, options.pinned));
  if (debit > before)
    throw std::invalid_argument(
        "named reservations exceed cache policy budget");
  const auto fixed = AddMemoryBytes(options.ordinary_pending, options.staging);
  const auto host_pending = AddMemoryBytes(fixed, pending.host);
  const auto device_pending = AddMemoryBytes(fixed, pending.device);
  if (host_pending > physical.host || device_pending > physical.device)
    throw std::invalid_argument("named reservations exceed physical capacity");
  const auto actual =
      std::min(physical.host - host_pending, physical.device - device_pending);
  return {before, before - debit, actual, std::min(before - debit, actual)};
}

inline constexpr char kSecondComponentBackendError[] =
    "a second component backend on the same Model is not supported "
    "(one backend per Model); release the first or load a separate Model";

// A serving capacity domain is shared across model owners. Separate domains
// are useful for isolated CPU fixtures, not independent budgets for one device.
class ReservationDomain {
public:
  [[nodiscard]] static std::shared_ptr<ReservationDomain> Create() {
    return std::shared_ptr<ReservationDomain>(new ReservationDomain);
  }
  [[nodiscard]] static std::shared_ptr<ReservationDomain> Shared() {
    static const auto domain = Create();
    return domain;
  }

private:
  friend class NamedReservations;
  ReservationDomain() = default;
  std::mutex mutex_;
  std::uint64_t next_owner_{};
};

// Stable model identity and component-binding lifetime. Binding does not grant
// memory, install callbacks or imply that any feature reservation is admitted.
class NamedReservations
    : public std::enable_shared_from_this<NamedReservations> {
public:
  NamedReservations(const NamedReservations&) = delete;
  NamedReservations& operator=(const NamedReservations&) = delete;

  class Binding {
  public:
    ~Binding() {
      const std::lock_guard lock(owner_->domain_->mutex_);
      if (active_)
        owner_->bound_ = false;
    }
    Binding(const Binding&) = delete;
    Binding& operator=(const Binding&) = delete;
    [[nodiscard]] const std::shared_ptr<NamedReservations>& owner() const {
      return owner_;
    }

  private:
    friend class NamedReservations;
    explicit Binding(std::shared_ptr<NamedReservations> owner)
        : owner_(std::move(owner)) {}
    std::shared_ptr<NamedReservations> owner_;
    bool active_{};
  };

  [[nodiscard]] static std::shared_ptr<NamedReservations> Create(
      std::shared_ptr<ReservationDomain> domain = ReservationDomain::Shared()) {
    if (!domain)
      throw std::invalid_argument("missing memory reservation domain");
    auto owner = std::shared_ptr<NamedReservations>(
        new NamedReservations(std::move(domain)));
    const std::lock_guard lock(owner->domain_->mutex_);
    if (owner->domain_->next_owner_ ==
        std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("memory reservation owner identity exhausted");
    owner->id_ = ++owner->domain_->next_owner_;
    return owner;
  }

  [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
  [[nodiscard]] bool SharesDomain(
      const NamedReservations& other) const noexcept {
    return domain_ == other.domain_;
  }
  [[nodiscard]] bool HasBinding() const {
    const std::lock_guard lock(domain_->mutex_);
    return bound_;
  }
  [[nodiscard]] std::shared_ptr<Binding> Bind() {
    // Allocate before publishing the binding. Failure leaves the old binding
    // untouched, and an inactive token's destruction cannot clear it.
    auto binding = std::shared_ptr<Binding>(new Binding(shared_from_this()));
    const std::lock_guard lock(domain_->mutex_);
    if (bound_)
      throw std::logic_error(kSecondComponentBackendError);
    bound_ = true;
    binding->active_ = true;
    return binding;
  }

private:
  explicit NamedReservations(std::shared_ptr<ReservationDomain> domain)
      : domain_(std::move(domain)) {}
  std::shared_ptr<ReservationDomain> domain_;
  std::uint64_t id_{};
  bool bound_{};
};

}  // namespace gufo::core
#endif  // GUFO_CORE_MEMORY_RESERVATIONS_HPP_
