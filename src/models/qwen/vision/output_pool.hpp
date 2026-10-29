#ifndef GUFO_MODELS_QWEN_VISION_OUTPUT_POOL_HPP_
#define GUFO_MODELS_QWEN_VISION_OUTPUT_POOL_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "src/models/qwen/vision/input_budget.hpp"

namespace gufo::models::qwen::vision {

/// Whole-history output credits, shared by one encoder. Idle tensors are a
/// replay optimization; admitted requests and unfinished device readers win.
class OutputPool {
  struct State;

public:
  static constexpr std::uint32_t kDefaultRows = kDefaultOutputPoolRows;

  struct Storage {
    virtual ~Storage() = default;
    /// Must release tensors only after every reader completed. False retains
    /// both tensors and credits. Called only with no admitted CPU borrowers.
    /// A hard reader failure may throw; it must not release unknown readers.
    virtual bool TryEvict() = 0;
  };

  class Owner {
  public:
    [[nodiscard]] std::uint32_t Rows() const noexcept { return rows_; }

  private:
    friend class OutputPool;
    friend struct State;
    explicit Owner(std::uint32_t rows) : rows_(rows) {}
    const std::uint32_t rows_;
    std::weak_ptr<State> pool_;
    std::shared_ptr<Storage> storage_;
    std::size_t pins_{};
    std::size_t encoded_rows_{};
    bool charged_{};
  };

  struct Stats {
    std::size_t rows{}, peak_rows{}, evictions{}, waits{}, encodes{},
        reencodes{};
    std::size_t encoded_rows{}, peak_encoded_rows{};
  };

private:
  struct State {
    explicit State(std::uint32_t rows) : capacity(rows) {
      if (rows == 0)
        throw std::invalid_argument(
            "vision output pool must have positive capacity");
    }
    void Release(const std::shared_ptr<Owner>& owner) noexcept {
      const std::lock_guard lock(mutex);
      if (--owner->pins_ == 0) {
        const auto it = std::find(resident.begin(), resident.end(), owner);
        resident.splice(resident.end(), resident, it);
      }
    }
    const std::uint32_t capacity;
    std::mutex mutex;
    std::map<std::string, std::weak_ptr<Owner>> identities;
    std::list<std::shared_ptr<Owner>> resident;
    Stats stats;
  };

public:
  class Lease {
  public:
    ~Lease() {
      if (active_)
        state_->Release(owner_);
    }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    /// Storage factories allocate CPU metadata only. Never retain a borrowed
    /// tensor outside this lease without recording a device-reader fence.
    [[nodiscard]] std::shared_ptr<Storage> GetStorage(
        const std::function<std::shared_ptr<Storage>()>& create) const {
      const std::lock_guard lock(state_->mutex);
      if (!owner_->storage_)
        owner_->storage_ = create();
      if (!owner_->storage_)
        throw std::logic_error("vision output storage factory returned null");
      return owner_->storage_;
    }
    void RecordEncode(bool reencode, std::uint32_t rows = 0) const {
      const std::lock_guard lock(state_->mutex);
      if (rows > owner_->rows_ - owner_->encoded_rows_)
        throw std::logic_error("encoded rows exceed reserved output credits");
      owner_->encoded_rows_ += rows;
      state_->stats.encoded_rows += rows;
      state_->stats.peak_encoded_rows =
          std::max(state_->stats.peak_encoded_rows, state_->stats.encoded_rows);
      ++state_->stats.encodes;
      state_->stats.reencodes += reencode;
    }

  private:
    friend class OutputPool;
    Lease(std::shared_ptr<State> state, std::shared_ptr<Owner> owner)
        : state_(std::move(state)), owner_(std::move(owner)) {}
    std::shared_ptr<State> state_;
    std::shared_ptr<Owner> owner_;
    bool active_{};
  };

  explicit OutputPool(std::uint32_t rows = kDefaultRows)
      : state_(std::make_shared<State>(rows)) {}
  [[nodiscard]] std::uint32_t Capacity() const noexcept {
    return state_->capacity;
  }

  void ValidateRows(std::uint64_t rows) const {
    if (rows > Capacity())
      throw std::length_error(
          "aggregate image tokens exceed --vision-pool-tokens (" +
          std::to_string(Capacity()) + "); reduce images or increase the pool");
  }

  /// Preparing a request creates metadata, not a reservation or GPU tensors.
  /// Identities include all image pixels/positions and belong to this encoder.
  [[nodiscard]] std::shared_ptr<Owner> GetOwner(std::string identity,
                                                std::uint32_t rows) {
    if (identity.empty() || rows == 0)
      throw std::invalid_argument(
          "vision output owner requires identity and rows");
    ValidateRows(rows);
    const std::lock_guard lock(state_->mutex);
    // Expired weak identities must not accumulate over an unbounded history.
    std::erase_if(state_->identities,
                  [](const auto& item) { return item.second.expired(); });
    if (const auto it = state_->identities.find(identity);
        it != state_->identities.end()) {
      if (auto owner = it->second.lock()) {
        if (owner->rows_ != rows)
          throw std::logic_error("vision output identity changed row count");
        return owner;
      }
    }
    auto owner = std::shared_ptr<Owner>(new Owner(rows));
    owner->pool_ = state_;
    state_->identities.emplace(std::move(identity), owner);
    return owner;
  }

  /// Nonblocking: nullptr means queue without holding a runner or partial
  /// credits. Successful leases reserve the whole history before encoding.
  [[nodiscard]] std::shared_ptr<Lease> TryAcquire(
      const std::shared_ptr<Owner>& owner) {
    if (!owner || owner->pool_.lock() != state_)
      throw std::invalid_argument(
          "vision output owner belongs to another pool");
    const std::lock_guard lock(state_->mutex);
    // Allocate the lease before changing accounting; allocation failure must
    // not leak a pin. Its destructor becomes responsible only on success.
    auto lease = std::shared_ptr<Lease>(new Lease(state_, owner));
    if (!owner->charged_) {
      for (auto it = state_->resident.begin();
           owner->rows_ > Capacity() - state_->stats.rows &&
           it != state_->resident.end();) {
        auto& victim = *it;
        if (victim->pins_ != 0 ||
            (victim->storage_ && !victim->storage_->TryEvict())) {
          ++it;
          continue;
        }
        state_->stats.rows -= victim->rows_;
        state_->stats.encoded_rows -= victim->encoded_rows_;
        victim->encoded_rows_ = 0;
        victim->charged_ = false;
        ++state_->stats.evictions;
        it = state_->resident.erase(it);
      }
      if (owner->rows_ > Capacity() - state_->stats.rows) {
        ++state_->stats.waits;
        return {};
      }
      state_->resident.push_back(owner);
      owner->charged_ = true;
      state_->stats.rows += owner->rows_;
      state_->stats.peak_rows =
          std::max(state_->stats.peak_rows, state_->stats.rows);
    }
    ++owner->pins_;
    lease->active_ = true;
    return lease;
  }

  [[nodiscard]] Stats GetStats() const {
    const std::lock_guard lock(state_->mutex);
    return state_->stats;
  }

private:
  std::shared_ptr<State> state_;
};

}  // namespace gufo::models::qwen::vision
#endif
