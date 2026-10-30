#ifndef GUFO_SERVER_COMPONENT_TEXT_CACHE_HPP_
#define GUFO_SERVER_COMPONENT_TEXT_CACHE_HPP_

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "src/cache/checkpoint.hpp"
#include "src/cache/streaming.hpp"
#include "src/cli/serve/checkpoint_transfer.hpp"
#include "src/cli/serve/component_cache_metrics.hpp"
#include "src/cli/serve/continuation_cache.hpp"
#include "src/cli/serve/pinned_prefix.hpp"
#include "src/cli/serve/text_publication_wait.hpp"
#include "src/core/startup_admission.hpp"

namespace gufo::server {
class TextModelRunner;
class TextRunnerState;
class TextRunnerSnapshot;
class TextDiagnosticLease;
struct TextRunnerDiskCacheOptions;
struct TextPromptContext;

// Model-independent serving orchestration; the model supplies physical storage
// and binds its ordinary execution facade to adapter-owned slots.
struct ComponentCaptureBuffer {
  std::span<std::byte> bytes;
  std::function<cache::Payload()> finish;
  // Identity of one underlying allocation, for joining adjacent byte spans.
  // Null providers always transfer each buffer separately.
  const void* allocation{};
  // Consume assigned backing without copying completed append rows. The
  // source view is backend-native and kept immutable by the slot guard.
  std::function<cache::Payload(cache::SlotLease&, std::span<const std::byte>)>
      borrow;
};
// Startup-only pending bytes. Keep shared and accumulated per-slot claims
// separate: materializing shared storage must not hide a new slot's liability.
struct ComponentStartupPending {
  std::size_t shared_execution{}, slot_execution{};
  std::size_t shared_images{}, slot_images{};
};

// Opt-in contract for homogeneous pools: every slot of one trial has identical
// model/mode/context/format settings. Mixed configurations must not opt in.
// Probe adapters share the loaded model, but NOT production pending counters.
// They never bind a second backend or publish cache state.
struct ComponentStartupAdmission {
  std::size_t context{};
  // Production uses the monotonic clock; CPU fixtures can advance it without
  // sleeping to exercise deadlines inside a staged probe.
  std::function<std::chrono::steady_clock::time_point()> now =
      std::chrono::steady_clock::now;
  // Complete logical bytes of ONE not-yet-created slot, including its lazy
  // obligations. Used only for the provisional first-slot precheck, never as
  // the measured debit or in addition to that slot's measured pending bytes.
  std::function<std::size_t(std::size_t)> slot_bytes;
  std::function<core::ExecutionCapacityObservation()> observe;
  std::function<ComponentStartupPending(const cache::Adapter&, std::size_t)>
      pending;
  std::function<std::unique_ptr<cache::Adapter>(std::size_t)> make_probe;
  // Supplies the exact successful observation to the provider's final guard.
  // Runtime backing growth must continue to remeasure, not use this snapshot.
  std::function<void(core::ExecutionCapacityObservation)> freeze_capacity;
};

struct ComponentCacheResources {
  std::shared_ptr<void> owner;
  std::unique_ptr<cache::Adapter> adapter;
  std::unique_ptr<ComponentStartupAdmission> startup_admission;
  cache::StreamLease stream;
  std::function<ComponentCaptureBuffer(const cache::PayloadRequest&)> allocate;
  cache::StagingAllocator staging;
  // Opt-in diagnostic witnesses; absent means immediate refusal. Called only
  // with exclusive slot ownership (ready also participates in atomic claim).
  // Ready is strong, nonblocking native quiescence +
  // healthy/materialized/empty, not Slot::ExecutionComplete's hint. Reset
  // returns pending without mutation, or clears native/bound-state residue and
  // fences before reporting complete.
  std::function<bool(const cache::Slot&)> diagnostic_ready;
  std::function<cache::DiagnosticDiscardResult(cache::Slot&, TextRunnerState&)>
      diagnostic_reset;
  // Prepare on-demand backing immediately before the next prefill chunk.
  std::function<void(std::size_t)> prepare_prefill;
  // Optional capacity bound before assigning a complete capture. False takes
  // the ordinary reclamation path; true still requires every allocation.
  std::function<bool(std::span<const cache::PayloadRequest>)> can_capture;
  // Optional initialization step after slots allocate. The provider leaves
  // payload backing uncommitted until serving selects the post-slot budget.
  std::function<void(std::size_t)> commit_backing;
  // Optional host bytes the slots pinned and charged to the ledger as they
  // allocated. Serving charges them on top of the cache's budget, and adds
  // them back to the runner's claim, which measures the host after them.
  std::function<std::size_t()> pinned_bytes;
  cache::PersistenceSource::CopyRows copy_observed_rows;
  // At most this many transfers of one capture/restore are in flight before
  // waiting (16 at most). When the shared pool is momentarily exhausted, the
  // batch settles its own leases and retries. One keeps generic providers
  // usable without a pool-capacity contract.
  std::size_t transfer_batch_size{1};
  std::size_t reassignment_batch_size{1};
  // Deterministic CPU fixtures only; empty in production providers. Hooks may
  // pause at lookup_observed, admission, selected, or mutation (owner locked).
  // They must not re-enter this cache from the same thread.
  std::function<void(std::string_view)> test_barrier;
  // Called once after all native slots exist, before backing/staging commit.
  // Return genuinely unallocated execution obligations, not resident memory.
  // The provider freezes the same value for asynchronous backing growth.
  std::function<std::size_t()> freeze_execution_liability;
  // Bind one bounded logical-eviction step after initialization; the provider
  // physically trims outside the publication lock and remeasures headroom.
  // Clearing the callback drains it before this cache's state is destroyed.
  std::function<void(std::function<bool()>)> bind_execution_reclaimer;
  // Protect the image execution envelope while RAM backing becomes permanent,
  // including entire shared segments rather than only the pin's payload bytes.
  // Called outside cache/publication locks and, at startup, before acquiring a
  // runner. A nonempty ticket must live through allocation and pin publication.
  // Empty refuses eager readiness or leaves lazy pins disk-backed. A lazy
  // caller already holds a runner: providers may trim free segments but must
  // refuse conservatively rather than wait for or evict active sources.
  std::function<std::shared_ptr<void>()> reserve_permanent_backing;
  // Optional refusal diagnostic: image envelope bytes needed and currently
  // available, respectively. Called outside cache/publication locks; it does
  // not reserve capacity and must not be used to authorize permanent backing.
  std::function<std::pair<std::size_t, std::size_t>()>
      permanent_backing_headroom;
};

class ComponentTextCache {
public:
  struct Impl;
  class Lease {
  public:
    struct Impl;
    explicit Lease(std::unique_ptr<Impl>);
    ~Lease();
    Lease(const Lease&) = delete;
    [[nodiscard]] ContinuationState& state() const;
    [[nodiscard]] bool cache_hit() const noexcept;
    [[nodiscard]] std::size_t cached_tokens() const noexcept;
    [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept;
    [[nodiscard]] double restore_ms() const noexcept;
    [[nodiscard]] bool restored_from_disk() const noexcept;
    [[nodiscard]] ContinuationLookup lookup() const noexcept;
    [[nodiscard]] bool HasSnapshotFor(std::span<const ContinuationToken>) const;
    [[nodiscard]] bool TryReserveSnapshot(std::size_t, std::size_t, bool,
                                          SnapshotPurpose,
                                          std::span<const ContinuationToken>);
    void SkipSnapshot(SnapshotEventReason, std::size_t, std::size_t) noexcept;
    std::size_t PublishSnapshot(std::vector<ContinuationToken>,
                                std::shared_ptr<const ContinuationSnapshot>,
                                bool);
    std::size_t Commit(std::vector<ContinuationToken>,
                       std::shared_ptr<const ContinuationSnapshot>,
                       std::vector<ContinuationToken>);
    void Invalidate() noexcept;
    [[nodiscard]] std::size_t CaptureBytes() const;
    [[nodiscard]] std::size_t CaptureBytes(std::size_t boundary) const;
    [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Capture();
    void PreparePrefill(std::size_t tokens);
    [[nodiscard]] ComponentCacheMetrics Metrics() const;
    [[nodiscard]] std::vector<std::size_t> PlannedCaptures(
        std::size_t stable) const;

  private:
    friend class ComponentTextCache;
    std::unique_ptr<Impl> impl_;
  };
  // Slots allocate under the larger of ram_bytes and ceiling_bytes, so a
  // smaller budget does not refuse what they pin. The RAM limit then becomes
  // the budget plus those pins, within the host ceiling. With `ram_enabled`
  // false, ram_bytes is ignored: no checkpoint stays in RAM and no backing is
  // committed, unless a disk cache needs some to stage captures.
  ComponentTextCache(std::shared_ptr<TextModelRunner>, std::size_t,
                     std::size_t ram_bytes,
                     const std::optional<TextRunnerDiskCacheOptions>&,
                     bool automatic_ram = false, std::size_t ceiling_bytes = 0,
                     const TextRunnerPinnedPrefixOptions& pins = {},
                     bool ram_enabled = true, bool publish_lookup = true);
  ~ComponentTextCache();
  [[nodiscard]] std::unique_ptr<Lease> Acquire(
      std::span<const ContinuationToken>,
      const ContinuationCache::CancellationCheck&,
      std::shared_ptr<const TextPromptContext>, bool, std::size_t, bool,
      bool persist = true);
  // Full concrete fixed lease object (not the interface base), with no
  // allocation. Provider-owned BindComponentState heap storage is separate and
  // must also be booked by the provider; this object owns no
  // vectors/strings/functions.
  [[nodiscard]] static std::size_t DiagnosticRequestBytes() noexcept;
  [[nodiscard]] std::unique_ptr<TextDiagnosticLease> TryAcquireDiagnostic(
      const ContinuationCache::CancellationCheck& = {});
  [[nodiscard]] std::optional<ContinuationReuse> Lookup(
      std::span<const ContinuationToken>, const TextPromptContext*, bool reuse,
      std::size_t stable) const;
  [[nodiscard]] std::span<const std::uint8_t> CompatibilityIdentity() const;
  // Transfer never acquires/replaces an execution slot. Export pins only
  // immutable checkpoints; Import publishes once after complete verification.
  [[nodiscard]] std::shared_ptr<CheckpointTransferBlob> Export(
      std::span<const ContinuationToken>, const TextPromptContext*, bool reuse,
      std::size_t stable, std::string configuration) const;
  [[nodiscard]] std::shared_ptr<CheckpointTransferBlob> Import(
      const CheckpointTransferReader&, std::uint64_t content_length,
      std::string_view configuration);

  // A disk job is writing a prefix of the prompt much deeper than any
  // restorable boundary; admission may wait for it instead of prefilling.
  // A nonzero checkpoint ID follows that one job, even if another deeper job
  // is submitted later. Its current gain may shrink as usable cache improves.
  [[nodiscard]] std::optional<TextPendingPublication> PendingPublication(
      std::span<const ContinuationToken> prompt,
      const TextPromptContext* context, std::uint64_t checkpoint_id = 0) const;
  [[nodiscard]] std::size_t CachedPrefixTokens(
      std::span<const ContinuationToken>, const TextPromptContext*,
      bool common = false) const;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t usable_capacity() const noexcept;
  // One scheduler observer; must not throw. Invoked outside publication locks.
  void SetAvailabilityCallback(std::function<void()>);

private:
  void InitializePins(const TextRunnerPinnedPrefixOptions&);
  std::unique_ptr<Impl> impl_;
};

// Only one implementation exists for a pool. Legacy models keep their old
// lease; converted models use the component store and never construct it.
class ServingCacheLease {
public:
  ServingCacheLease(ContinuationCache::Lease lease)
      : legacy_(std::move(lease)) {}
  ServingCacheLease(std::unique_ptr<ComponentTextCache::Lease> lease)
      : component_(std::move(lease)) {}
  ServingCacheLease(ServingCacheLease&&) noexcept = default;
  ServingCacheLease& operator=(ServingCacheLease&&) noexcept = default;
  [[nodiscard]] explicit operator bool() const noexcept {
    return component_ || bool(legacy_);
  }
  [[nodiscard]] bool component() const noexcept { return bool(component_); }
  [[nodiscard]] ContinuationState& state() const;
  [[nodiscard]] bool cache_hit() const noexcept;
  [[nodiscard]] std::size_t cached_tokens() const noexcept;
  [[nodiscard]] std::size_t restored_snapshot_bytes() const noexcept;
  [[nodiscard]] double restore_ms() const noexcept;
  [[nodiscard]] bool restored_from_disk() const noexcept;
  [[nodiscard]] ContinuationLookup lookup() const noexcept;
  [[nodiscard]] bool HasSnapshotFor(std::span<const ContinuationToken>) const;
  void AdoptRestoredPrefix(std::size_t, std::size_t, double);
  [[nodiscard]] bool TryReserveSnapshot(
      std::size_t, std::size_t, bool = false,
      SnapshotPurpose = SnapshotPurpose::kContinuation,
      std::span<const ContinuationToken> = {});
  void SkipSnapshot(SnapshotEventReason, std::size_t, std::size_t) noexcept;
  std::size_t PublishSnapshot(std::vector<ContinuationToken>,
                              std::shared_ptr<const ContinuationSnapshot>,
                              bool = false);
  std::size_t Commit(std::vector<ContinuationToken>,
                     std::shared_ptr<const ContinuationSnapshot> = {},
                     std::vector<ContinuationToken> = {});
  void Invalidate() noexcept;
  [[nodiscard]] std::size_t CaptureBytes() const;
  [[nodiscard]] std::size_t CaptureBytes(std::size_t boundary) const;
  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Capture();
  void PreparePrefill(std::size_t tokens);
  [[nodiscard]] std::optional<ComponentCacheMetrics> Metrics() const;
  [[nodiscard]] std::vector<std::size_t> PlannedCaptures(
      std::size_t stable) const;

private:
  ContinuationCache::Lease legacy_;
  std::unique_ptr<ComponentTextCache::Lease> component_;
};
}  // namespace gufo::server
#endif
