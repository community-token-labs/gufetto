#ifndef GUFO_CORE_RECLAIMABLE_MEMORY_HPP_
#define GUFO_CORE_RECLAIMABLE_MEMORY_HPP_

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "src/core/memory_budget.hpp"

namespace gufo::core {

// One authority for lazy image capacity and competing physical cache growth.
// Headroom is physical free memory AFTER hard execution/staging/OS reserves.
// A queued demand owns neither a claim nor an execution/source lease. Admission
// materializes reclamation before promising capacity; retained metadata is
// never collateral. Image capacity covers the shared workspace/output pool and
// every slot's positions, not one copy per admitted request.
class ReclaimableMemory
    : public std::enable_shared_from_this<ReclaimableMemory> {
public:
  class ImageLease {
  public:
    ~ImageLease() { Complete(); }
    ImageLease(const ImageLease&) = delete;
    ImageLease& operator=(const ImageLease&) = delete;
    // Call after the last image prefill (or a cache restore past all images).
    // Cancellation falls back to destruction after execution has unwound.
    bool Complete() noexcept {
      const std::lock_guard lock(owner_->mutex_);
      if (!active_)
        return false;
      --owner_->images_;
      owner_->pin_cushion_ -= cushion_;
      active_ = false;
      return true;
    }

  private:
    friend class ReclaimableMemory;
    explicit ImageLease(std::shared_ptr<ReclaimableMemory> owner)
        : owner_(std::move(owner)) {}
    std::shared_ptr<ReclaimableMemory> owner_;
    bool active_{};
    std::size_t cushion_{};
  };

  struct Stats {
    std::size_t capacity{}, resident{}, outstanding{}, growth{}, images{};
    bool reclaiming{};
  };

  // Quiescent initialization, after slots exist and before backing commits.
  void Configure(std::size_t image_capacity,
                 std::function<std::size_t()> headroom) {
    const std::lock_guard lock(mutex_);
    if (headroom_ || !headroom)
      throw std::logic_error("invalid reclaimable memory initialization");
    capacity_ = image_capacity;
    headroom_ = std::move(headroom);
  }

  // Reclaim attempts one bounded step. True permits remeasurement, not credit.
  // Clear at serving-cache teardown BEFORE destroying index/retention. This
  // mutex drains in-flight callbacks without holding the accounting mutex.
  void SetReclaimer(std::function<bool(std::size_t)> reclaim) {
    const std::lock_guard lock(reclaim_mutex_);
    reclaim_ = std::move(reclaim);
  }

  // Permanent backing additionally leaves room for every frozen hard/staging
  // liability to become resident later. Otherwise that later allocation could
  // make a permanently pinned pool fail our conservative frozen guard forever.
  [[nodiscard]] std::shared_ptr<ImageLease> TryAcquireImage(
      std::size_t cushion = 0) {
    auto lease =
        std::shared_ptr<ImageLease>(new ImageLease(shared_from_this()));
    std::unique_lock reclaim_lock(reclaim_mutex_, std::try_to_lock);
    if (!reclaim_lock.owns_lock())
      return {};
    {
      const std::lock_guard lock(mutex_);
      RequireConfigured();
      if (images_ != 0 && cushion == 0) {
        ++images_;
        lease->active_ = true;
        return lease;
      }
      // A growth ticket survives physical commit/publication. Do not wait on
      // it here: the scheduler must keep admitted work and pin holders moving.
      if (growth_ != 0)
        return {};
      reclaiming_ = true;
    }
    try {
      for (;;) {
        std::size_t deficit{};
        {
          const std::lock_guard lock(mutex_);
          const auto available = headroom_();
          const auto next_cushion = AddMemoryBytes(pin_cushion_, cushion);
          // Retiring resident images can rearm the entire capacity later.
          // Validate that sum before publishing any active lease state.
          (void)AddMemoryBytes(capacity_, next_cushion);
          const auto pending = capacity_ - resident_ + next_cushion;
          if (available >= pending) {
            ++images_;
            lease->active_ = true;
            lease->cushion_ = cushion;
            pin_cushion_ = next_cushion;
            reclaiming_ = false;
            return lease;
          }
          deficit = pending - available;
        }
        // No accounting lock while taking cache/pool locks. Grow's callback
        // may take our mutex under the pool topology lock, but cannot enter
        // this reclaimer; reclaiming_ denies new growth throughout the walk.
        if (!reclaim_ || !reclaim_(deficit))
          break;
      }
    } catch (...) {
      const std::lock_guard lock(mutex_);
      reclaiming_ = false;
      throw;
    }
    {
      const std::lock_guard lock(mutex_);
      reclaiming_ = false;
    }
    return {};
  }

  [[nodiscard]] std::shared_ptr<void> TryAcquireGrowth(std::size_t bytes) {
    auto ticket = std::make_shared<GrowthLease>(shared_from_this(), bytes);
    const std::lock_guard lock(mutex_);
    RequireConfigured();
    if (reclaiming_ || !FitsMemoryBudget(headroom_(), bytes,
                                         AddMemoryBytes(growth_, Pending())))
      return {};
    growth_ = AddMemoryBytes(growth_, bytes);
    ticket->active = true;
    return ticket;
  }

  // Allocate/release callbacks MUST NOT reenter this authority, cache or pool.
  // Serialize the physical operation with sampling/claim conversion so growth
  // never spends freed bytes before their image obligation has been rearmed.
  template<class Allocate>
  void AllocateImage(std::size_t bytes, Allocate allocate) {
    const std::lock_guard lock(mutex_);
    RequireConfigured();
    if (images_ == 0 || bytes > capacity_ - resident_)
      throw std::logic_error("image allocation has no admitted capacity");
    allocate();
    resident_ += bytes;
  }
  template<class Release>
  void RetireImage(std::size_t bytes, Release release) noexcept {
    const std::lock_guard lock(mutex_);
    if (bytes > resident_)
      std::terminate();
    if (release())
      resident_ -= bytes;
    // Failed physical release keeps the resident charge, even at teardown.
  }
  [[nodiscard]] std::pair<std::size_t, std::size_t> ImageHeadroom(
      std::size_t cushion = 0) const {
    const std::lock_guard lock(mutex_);
    RequireConfigured();
    const auto free = headroom_();
    return {AddMemoryBytes(capacity_ - resident_,
                           AddMemoryBytes(pin_cushion_, cushion)),
            free > growth_ ? free - growth_ : 0};
  }
  [[nodiscard]] Stats GetStats() const {
    const std::lock_guard lock(mutex_);
    return {capacity_, resident_, Pending(), growth_, images_, reclaiming_};
  }

private:
  struct GrowthLease {
    GrowthLease(std::shared_ptr<ReclaimableMemory> owner, std::size_t bytes)
        : owner(std::move(owner)), bytes(bytes) {}
    ~GrowthLease() {
      if (active) {
        const std::lock_guard lock(owner->mutex_);
        owner->growth_ -= bytes;
      }
    }
    std::shared_ptr<ReclaimableMemory> owner;
    std::size_t bytes;
    bool active{};
  };
  void RequireConfigured() const {
    if (!headroom_)
      throw std::logic_error("reclaimable memory is not configured");
  }
  [[nodiscard]] std::size_t Pending() const noexcept {
    return images_ ? capacity_ - resident_ + pin_cushion_ : 0;
  }
  mutable std::mutex mutex_;
  std::mutex reclaim_mutex_;
  std::function<std::size_t()> headroom_;
  std::function<bool(std::size_t)> reclaim_;
  std::size_t capacity_{}, resident_{}, growth_{}, images_{}, pin_cushion_{};
  bool reclaiming_{};
};

// Scoped installation into an externally owned model. The installer captures
// that model's lifetime and must be nonthrowing and quiescent. Keeping this
// token in component resources unwinds both failed initialization and normal
// teardown; native users must never inherit an unconfigured serving authority.
class ReclaimableMemoryBinding {
public:
  using Install = std::function<void(std::shared_ptr<ReclaimableMemory>)>;
  [[nodiscard]] static std::shared_ptr<ReclaimableMemoryBinding> Bind(
      const std::shared_ptr<ReclaimableMemory>& current,
      std::shared_ptr<ReclaimableMemory> next, Install install) {
    if (current || !next || !install)
      throw std::logic_error("model already has a component memory authority");
    auto result = std::shared_ptr<ReclaimableMemoryBinding>(
        new ReclaimableMemoryBinding(std::move(install)));
    result->install_(std::move(next));
    return result;
  }
  ~ReclaimableMemoryBinding() { install_({}); }
  ReclaimableMemoryBinding(const ReclaimableMemoryBinding&) = delete;
  ReclaimableMemoryBinding& operator=(const ReclaimableMemoryBinding&) = delete;

private:
  explicit ReclaimableMemoryBinding(Install install)
      : install_(std::move(install)) {}
  Install install_;
};

}  // namespace gufo::core
#endif
