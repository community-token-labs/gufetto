#ifndef GUFO_CORE_MEMORY_RESERVATIONS_HPP_
#define GUFO_CORE_MEMORY_RESERVATIONS_HPP_

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace gufo::core {

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
