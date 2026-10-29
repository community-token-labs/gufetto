#include "src/cli/serve/component_text_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#include "src/cache/disk_catalog.hpp"
#include "src/cache/retention.hpp"
#include "src/cli/serve/logging.hpp"
#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/os_info.hpp"
#include "src/core/platform/runtime_fingerprint.hpp"
#include "src/models/qwen38_flash_next/kv_cache.hpp"

namespace gufo::server {
namespace {
using Clock = std::chrono::steady_clock;
// Pins are deliberately stricter than the ordinary ABI-compatible disk tier.
// Bind the executable and loaded runtime objects, not a potentially stale Git
// revision string. No paths, identities or environment values are logged.
cache::Identity PinnedCompatibility(cache::Identity identity, bool enabled) {
  if (!enabled)
    return identity;
  const auto suffix = platform::StrictRuntimeFingerprint(
      "gufo-registered-prefix-v1;prefill-chunk=4096");
  identity.insert(identity.end(), suffix.begin(), suffix.end());
  return identity;
}
std::size_t DiskStaging(bool ordinary, bool pinned) {
  return (std::size_t(ordinary) + std::size_t(pinned)) *
         cache::kDefaultDiskStagingBytes;
}
// Process-local runners may lack durable artifact fingerprints; their
// registered append layout must still activate packed row admission.
bool RotatedPackedIngress(
    const TextRunnerDescriptor& descriptor,
    std::span<const cache::ComponentDescriptor> components) {
  namespace qfn = models::qwen38_flash_next;
  if (descriptor.state_abi != "qwen38-flash-next-rocm-session-v1")
    return false;
  return (descriptor.persistence &&
          descriptor.persistence->payload_version ==
              qfn::ExpectedKvPayloadVersion(qfn::KvCacheType::kQ8_0Rotated)) ||
         std::ranges::any_of(components, [](const auto& component) {
           return component.kind == cache::ComponentKind::kAppendRows &&
                  component.layout_version == 3;
         });
}
bool PackedKvIngress(const TextRunnerDescriptor& descriptor,
                     std::span<const cache::ComponentDescriptor> components) {
  if (descriptor.state_abi != "qwen38-flash-next-rocm-session-v1")
    return false;
  // Inspect the registered append layout even without a durable fingerprint.
  if (std::ranges::any_of(components, [](const auto& c) {
        return c.kind == cache::ComponentKind::kAppendRows &&
               c.layout_version == 2;
      }))
    return true;
  return RotatedPackedIngress(descriptor, components);
}
class SettledSignal final : public cache::CompletionSignal {
public:
  bool Ready() const noexcept override { return true; }
  cache::TransferResult Wait() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};
cache::InputIdentity Inputs(std::size_t count,
                            const TextPromptContext* context) {
  std::vector<cache::InputPrefix> prefixes;
  if (context)
    for (const auto& p : context->cache_prefixes)
      prefixes.push_back({p.token_count, p.identity});
  return cache::InputIdentity(
      count, context ? context->cache_identity : cache::Identity{},
      std::move(prefixes));
}
cache::Identity At(const cache::InputIdentity& input, std::size_t boundary) {
  const auto bytes = input.At(boundary);
  return {bytes.begin(), bytes.end()};
}
struct TransferQuota {
  std::shared_ptr<std::atomic<bool>> gate;
  explicit TransferQuota(std::shared_ptr<std::atomic<bool>> gate)
      : gate(std::move(gate)) {
    if (this->gate->exchange(true, std::memory_order_acq_rel))
      throw CheckpointTransferError(503, "cache_transfer_busy");
  }
  ~TransferQuota() { gate->store(false, std::memory_order_release); }
};
std::size_t ResidentBytes(const cache::Checkpoint& checkpoint) {
  std::size_t bytes = 0;
  for (const auto& c : checkpoint.Components()) {
    for (const auto& chunk : c.chunks)
      bytes += chunk.Storage().Bytes();
    if (c.tail)
      bytes += c.tail->Bytes();
    if (c.private_state)
      bytes += c.private_state->Bytes();
  }
  return bytes;
}
std::size_t PayloadBytes(const cache::DiskManifest& manifest) {
  std::size_t bytes = 0;
  for (const auto& component : manifest.components) {
    for (const auto& chunk : component.chunks)
      bytes += chunk.bytes;
    if (component.tail)
      bytes += component.tail->bytes;
    if (component.private_state)
      bytes += component.private_state->bytes;
  }
  return bytes;
}
struct DurableCapture {
  cache::ResourceCharge metadata;
  std::shared_ptr<const void> branch;
  cache::DiskReadPin ancestor_pin;
  cache::DiskManifest manifest;
  std::vector<cache::PersistenceSource> sources;
};
class ComponentSnapshot final : public TextRunnerSnapshot {
public:
  std::shared_ptr<const cache::Checkpoint> checkpoint;
  std::shared_ptr<DurableCapture> durable;
  std::size_t bytes{};
  std::size_t PayloadBytes() const noexcept override { return bytes; }
};
cache::RetentionPurpose Purpose(SnapshotPurpose purpose) {
  switch (purpose) {
    case SnapshotPurpose::kRetry:
      return cache::RetentionPurpose::kRetry;
    case SnapshotPurpose::kHistory:
      return cache::RetentionPurpose::kHistory;
    case SnapshotPurpose::kBranchPoint:
      return cache::RetentionPurpose::kBranchPoint;
    case SnapshotPurpose::kContinuation:
      return cache::RetentionPurpose::kContinuation;
  }
  throw std::logic_error("unknown retention purpose");
}
void Check(cache::Completion completion) {
  if (completion.Wait() != cache::TransferResult::kSucceeded)
    throw std::runtime_error("component cache transfer failed");
}
// A published checkpoint that is still resident restores from RAM, so its
// candidate carries no disk description; its availability still marks every
// component durable.
bool Published(const cache::PrefixCandidate& candidate) {
  if (candidate.durable)
    return true;
  return candidate.availability &&
         !candidate.availability->components.empty() &&
         std::ranges::all_of(candidate.availability->components,
                             &cache::ComponentAvailability::durable);
}
bool SameOwner(const std::shared_ptr<const void>& a,
               const std::shared_ptr<const void>& b) {
  return !a.owner_before(b) && !b.owner_before(a);
}
class TransferBatch {
public:
  explicit TransferBatch(std::size_t capacity) : capacity_(capacity) {
    if (!capacity || capacity > jobs_.size())
      throw std::invalid_argument("invalid cache transfer batch size");
  }
  // Acquire returns null when the shared pool is momentarily exhausted. Settle
  // this batch's own leases before retrying, so a wide batch never fails a
  // transfer that a narrower one would have completed.
  template<class Acquire, class Submit>
  void Add(Acquire acquire, Submit submit) {
    if (size_ == capacity_)
      Finish();
    auto stream = acquire();
    if (!stream && size_) {
      Finish();
      stream = acquire();
    }
    if (!stream)
      throw std::runtime_error("component cache transfer pool exhausted");
    auto completion = submit(*stream);
    jobs_[size_++].emplace(Job{std::move(stream), std::move(completion)});
  }
  void Finish() {
    // Jobs keep their streams and completions together. On any exception,
    // destruction drains every remaining transfer before buffers unwind.
    for (std::size_t i = 0; i < size_; ++i) {
      Check(std::move(jobs_[i]->completion));
      jobs_[i].reset();
    }
    size_ = 0;
  }

private:
  struct Job {
    std::unique_ptr<cache::Stream> stream;
    cache::Completion completion;
  };
  std::array<std::optional<Job>, 16> jobs_;
  std::size_t capacity_, size_{};
};
class Events final : public cache::RetentionEventSink {
public:
  void Emit(const cache::RetentionEvent& e) noexcept override {
    try {
      Logger::Info(
          "cache",
          "schema=component-cache-v1 event=retention action=" +
              std::to_string(static_cast<unsigned>(e.action)) +
              " reason=" + std::to_string(static_cast<unsigned>(e.reason)) +
              " tokens=" + std::to_string(e.boundary) +
              " rank=" + std::to_string(e.rank) +
              " unique_bytes_freed=" + std::to_string(e.unique_bytes_freed) +
              " ledger_bytes=" + std::to_string(e.ledger_bytes));
    } catch (...) {
    }
  }
};
}  // namespace

struct ComponentTextCache::Impl {
  struct Entry {
    // Guard fallbacks outlive their slots. Idle pieces and capture/load work
    // obtain fresh one-completion leases from the shared transfer pool.
    std::unique_ptr<cache::Stream> preservation;
    std::unique_ptr<cache::LeasedSlot> slot;
    std::optional<cache::IndexEntryId> live;
    std::unique_ptr<cache::ExecutionHistory> history;
    // Identity of uninterrupted native execution; token equality alone does
    // not establish that a durable chunk belongs to this execution.
    std::shared_ptr<const void> branch;
    std::shared_ptr<const cache::DiskDescription> durable;
    // The live frontier's prompt, to tell whether any checkpoint covers it.
    std::vector<cache::Token> live_tokens;
    cache::Identity live_input;
    // Acquisition order; cold assignment replaces the oldest live frontier.
    std::uint64_t used{};
    bool busy{};
    bool decode_ready{};
  };
  std::shared_ptr<TextModelRunner> runner;
  cache::ResourceLedger ledger;
  std::unique_ptr<ComponentCacheResources> resources;
  cache::Identity identity;
  cache::CompatibilityDigest digest;
  cache::DiskFileId nonce{};
  Events events;
  // Durable descriptions in the index retain a Store facade until index and
  // retention teardown have finished using its publication pins.
  std::unique_ptr<cache::DiskStore> disk;
  // A different owned directory, never handed to DiskCatalog or its LRU.
  std::unique_ptr<cache::DiskStore> pin_disk;
  cache::PrefixIndex index;
  cache::RetentionPolicy retention;
  // Readers never take the serving mutex. Invalidate before a mutation and
  // publish its complete metadata before releasing the owner. A racing reader
  // returns a conservative miss, not busy and not a partially updated index.
  bool lookup_enabled;
  struct LookupView {
    cache::PrefixIndexSnapshot index;
    Clock::time_point published;
    std::uint64_t version{};
  };
  std::atomic<std::shared_ptr<const LookupView>> lookup_view;
  std::atomic<std::uint64_t> lookup_version{0};
  mutable std::atomic<std::uint64_t> next_observation{1};
  std::uint64_t next_acquisition{1};
  struct PublicationMutex {
    Impl& owner;
    std::mutex native;
    void lock() {
      native.lock();
      if (owner.lookup_enabled) {
        owner.lookup_version.fetch_add(1, std::memory_order_acq_rel);
        owner.lookup_view.store({}, std::memory_order_release);
      }
      if (owner.resources->test_barrier)
        owner.resources->test_barrier("mutation");
    }
    void unlock() noexcept {
      if (owner.lookup_enabled) {
        owner.PublishLookup();
        owner.lookup_version.fetch_add(1, std::memory_order_release);
      }
      native.unlock();
    }
  };
  mutable PublicationMutex mutex{*this};
  // One bounded spool/transfer per loaded pool, including slow HTTP writers.
  std::shared_ptr<std::atomic<bool>> transfer_gate =
      std::make_shared<std::atomic<bool>>(false);
  std::vector<std::unique_ptr<Entry>> entries;
  std::uint64_t clock{};
  struct Readiness {
    std::weak_ptr<const cache::Checkpoint> checkpoint;
    bool decode_ready{};
  };
  // Weak metadata follows retained payload lifetime; never keeps an evicted
  // checkpoint alive. Disk-only descriptions cannot attest decode readiness.
  std::vector<Readiness> readiness;

  void PublishLookup() noexcept {
    if (!lookup_enabled)
      return;
    try {
      auto view = std::make_shared<LookupView>();
      view->index =
          index.Snapshot([&](const cache::PrefixCandidate& candidate) {
            if (candidate.live) {
              const auto& entry = *entries.at(candidate.live->slot.value - 1);
              return !entry.busy && entry.decode_ready;
            }
            return runner->LookupFullCheckpointReady() ||
                   (candidate.checkpoint &&
                    std::ranges::any_of(readiness, [&](const auto& r) {
                      return r.decode_ready &&
                             r.checkpoint.lock() == candidate.checkpoint;
                    }));
          });
      view->published = Clock::now();
      const auto version = lookup_version.load(std::memory_order_relaxed);
      view->version = version + (version & 1);
      lookup_view.store(std::move(view), std::memory_order_release);
    } catch (...) {
      // Publication is observational: allocation/metadata pressure must not
      // fail generation or retain the previous, possibly evicted inventory.
      lookup_view.store({}, std::memory_order_release);
    }
  }

  cache::PrefixLookup Plan(std::span<const ContinuationToken> prompt,
                           const cache::InputIdentity& input,
                           std::size_t stable, bool reuse) const {
    return index.Lookup({identity, prompt, input, stable, reuse});
  }
  std::unique_ptr<cache::DiskCatalog> catalog;
  std::unique_ptr<cache::StreamedStore> streaming;
  std::unique_ptr<cache::PersistenceQueue> queue;
  std::unique_ptr<cache::StreamedStore> pin_streaming;
  // Not admitted to RetentionPolicy: neither RAM nor record pressure may
  // remove these roots. The vector and payloads use the common ledger.
  struct Pin {
    cache::CheckpointId durable;
    std::shared_ptr<const cache::Checkpoint> ram;
    std::optional<cache::IndexEntryId> entry;
    bool warming{};
  };
  cache::ResourceCharge pin_metadata;
  std::vector<Pin> pins;
  bool lazy_pins{};
  static constexpr std::size_t kPersistenceDepth = 8;
  // Backing that disk captures copy through when the RAM cache is off. It
  // equals the small explicit budget used before off existed, and it retains
  // nothing.
  static constexpr std::size_t kDiskOnlyBackingBytes = std::size_t{512} << 20;
  // Ledger room for index and request metadata when the RAM cache is off. It
  // caps accounting only; the memory is not reserved.
  static constexpr std::size_t kOffMetadataBytes = std::size_t{64} << 20;
  // Queue ownership plus transient submissions before superseded callbacks
  // release their serving handles. No callback runs under the serving mutex.
  std::array<std::shared_ptr<const cache::Checkpoint>, 2 * kPersistenceDepth>
      pending;
  std::array<std::shared_ptr<DurableCapture>, 2 * kPersistenceDepth>
      pending_durable;
  std::size_t disk_capacity{}, disk_step{};
  // The cache's own RAM budget. The ledger limit adds what the slots pinned.
  // Zero when the RAM cache is off.
  std::size_t budget{};
  // RAM backing committed only to copy captures on their way to disk.
  std::size_t disk_backing{};
  bool ram_enabled{true};
  cache::ResourceCharge metadata;

  // One admitted append chunk/tail at a time, no private-state or GPU staging
  // allocation. ReadPayload verifies the complete dependency CRC before finish
  // can validate model bytes and convert backing ownership. Drop that temporary
  // payload before the next chunk. No live slot is restored or mutated here.
  void PreflightDiskAppendRows(cache::DiskStore& store,
                               cache::CheckpointId id) {
    auto snapshot = store.Open(id);
    if (!snapshot)
      throw std::runtime_error("packed disk admission lost its publication");
    const auto descriptors = resources->adapter->Components();
    const auto& manifest = snapshot->Manifest();
    if (manifest.components.size() != descriptors.size())
      throw std::invalid_argument("packed disk component layout mismatch");
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
      const auto& a = descriptors[i];
      const auto& component = manifest.components[i];
      const auto& b = component.descriptor;
      if (a.id != b.id || a.kind != b.kind ||
          a.layout_version != b.layout_version || a.row_bytes != b.row_bytes ||
          a.rows_per_chunk != b.rows_per_chunk ||
          a.state_bytes != b.state_bytes)
        throw std::invalid_argument("packed disk component layout mismatch");
      if (a.kind != cache::ComponentKind::kAppendRows)
        continue;
      if (a.row_bytes == 0 || a.rows_per_chunk == 0)
        throw std::invalid_argument("packed disk row geometry mismatch");
      cache::Rows first = 0;
      const auto read = [&](const cache::DiskPayload& payload, bool tail) {
        if (payload.bytes == 0 || payload.bytes % a.row_bytes != 0 ||
            payload.bytes > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument("packed disk row geometry mismatch");
        const auto rows = payload.bytes / a.row_bytes;
        if (rows > a.rows_per_chunk || first > component.position.valid_rows ||
            rows > component.position.valid_rows - first)
          throw std::invalid_argument("packed disk row range mismatch");
        const auto bytes = static_cast<std::size_t>(payload.bytes);
        const cache::PayloadRequest request{
            a.id,
            tail ? cache::ResourceCategory::kPrivateTail
                 : cache::ResourceCategory::kBackingAssigned,
            first, first + rows, bytes};
        auto buffer = resources->allocate(request);
        if (!buffer.finish || buffer.bytes.size() < bytes)
          throw std::logic_error("incomplete packed disk admission buffer");
        const auto host = std::as_writable_bytes(buffer.bytes.first(bytes));
        (void)store.ReadPayload(
            *snapshot, tail, payload,
            {reinterpret_cast<std::uint8_t*>(host.data()), host.size()},
            [](std::uint64_t, std::span<const std::uint8_t>) {});
        // Never finish from the consume callback: final CRC/identity checks
        // happen after its final piece. Model rejection must precede exposure.
        auto verified = buffer.finish();
        if (!verified.IsValid())
          throw std::logic_error("invalid packed disk admission ownership");
        first += rows;
      };
      for (const auto& payload : component.chunks)
        read(payload, false);
      if (component.tail)
        read(*component.tail, true);
      if (first != component.position.valid_rows)
        throw std::invalid_argument("packed disk row coverage mismatch");
    }
  }

  void PreflightPackedDisk(cache::DiskStore& store) {
    if (!PackedKvIngress(runner->Descriptor(),
                         resources->adapter->Components()))
      return;
    // Retirement invalidates Entries(); retain bounded publication identities.
    struct Publication {
      cache::CheckpointId id;
      std::uint64_t epoch;
    };
    const auto entries = store.Entries();
    if (entries.empty())
      return;
    if (entries.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(Publication))
      throw std::length_error("packed disk admission inventory overflow");
    auto charge = ledger.Reserve(cache::ResourceCategory::kMetadata,
                                 entries.size() * sizeof(Publication));
    std::vector<Publication> publications;
    publications.reserve(entries.size());
    for (const auto& entry : entries)
      if (entry.Durable() && entry.Manifest().compatibility == digest)
        publications.push_back({entry.Manifest().checkpoint, entry.Epoch()});
    for (const auto& [id, epoch] : publications) {
      const auto retire = [&] {
        store.Invalidate(id, epoch);
        const auto retired = store.TryRetire(id, epoch);
        if (retired != cache::DiskRetireResult::kRetired &&
            retired != cache::DiskRetireResult::kMissing)
          throw std::runtime_error("cannot retire rejected packed checkpoint");
      };
      if (!store.Describe(id)) {
        // A previous payload read may quarantine other publications sharing
        // that dependency. Retire them too, without trying to read them again.
        retire();
        continue;
      }
      try {
        PreflightDiskAppendRows(store, id);
      } catch (const cache::DiskDependencyError&) {
        // The store already quarantined every owner of the invalid file.
        // Do not catch unrelated resource, allocation or operational failures.
        retire();
        Logger::Warn("cache", "event=packed_disk_dependency_rejected");
      } catch (const std::invalid_argument&) {
        retire();
        Logger::Warn("cache", "event=packed_disk_semantic_rejected");
      }
    }
  }

  Impl(std::shared_ptr<TextModelRunner> runner, std::size_t count,
       std::size_t ram, std::size_t slot_ceiling,
       const std::optional<TextRunnerDiskCacheOptions>& options,
       bool automatic_ram, const TextRunnerPinnedPrefixOptions& pins,
       bool ram_on, bool publish_lookup)
      : runner(std::move(runner)),
        // Disk transfers lease one fixed pinned staging buffer. Its size does
        // not change restore or write speed (#562), and a larger one only
        // takes pinned RAM and device heap from the cache and the sessions.
        ledger({slot_ceiling + DiskStaging(bool(options), pins.enabled()),
                slot_ceiling, DiskStaging(bool(options), pins.enabled()),
                slot_ceiling}),
        resources(this->runner->CreateComponentCacheResources(ledger, ram)),
        identity(PinnedCompatibility(
            resources->adapter->CompatibilityIdentity(), pins.enabled())),
        index(ledger),
        retention(ledger, index, &events),
        lookup_enabled(publish_lookup),
        ram_enabled(ram_on) {
    if (!ram_enabled && pins.enabled() && !pins.lazy)
      throw std::invalid_argument(
          "RAM-warm pinned prefixes require an enabled RAM cache; use "
          "pin-prefix-lazy for existing disk pins with cache-ram-bytes off");
    metadata = ledger.Reserve(cache::ResourceCategory::kMetadata, sizeof(Impl))
                   .Convert();
    crypto::Sha256Hasher hash;
    hash.Update(identity);
    digest = hash.Finish();
    if (!platform::FillRandomBytes(nonce))
      throw std::runtime_error("cannot obtain cache publication nonce");
    index.Register(identity, resources->adapter->Components(), digest);
    index.SetInvalidationObserver([this](auto id, auto boundary, auto reason) {
      if (!lookup_enabled)
        return;
      const auto version = lookup_version.load(std::memory_order_relaxed);
      Logger::Info(
          "cache",
          "schema=cache-lookup-v1 event=retire entry_id=" +
              std::to_string(id.value) + " tokens=" + std::to_string(boundary) +
              " retirement_version=" + std::to_string(version + (version & 1)) +
              " reason=" + std::string(reason));
    });
    if (options) {
      const auto row_bytes =
          cache::MinimumStagingBytes(resources->adapter->Components());
      if (ledger.Limits().staging_bytes < row_bytes)
        throw std::logic_error("a " + std::to_string(row_bytes) +
                               "-byte cache row exceeds the disk staging "
                               "buffer");
      if (options->staging_capacity_bytes)
        Logger::Warn("cache",
                     "schema=component-cache-v1 event=option_ignored "
                     "option=--cache-disk-staging-bytes requested_bytes=" +
                         std::to_string(options->staging_capacity_bytes) +
                         " staging_capacity_bytes=" +
                         std::to_string(ledger.Limits().staging_bytes));
      // Take directory ownership before slots and backing commit, so a
      // second server fails without claiming its RAM budget first.
      disk = std::make_unique<cache::DiskStore>(ledger, options->directory,
                                                options->capacity_bytes);
    }
    if (pins.enabled()) {
      if (pins.files.empty() ||
          pins.files.size() >
              TextRunnerPinnedPrefixOptions::kMaxRegistrations ||
          !pins.capacity_bytes)
        throw std::invalid_argument(
            "pinned prefixes require 1..64 files and a byte budget");
      if (!this->runner->Descriptor().persistence)
        throw std::invalid_argument(
            "pinned prefix requires a durable model identity");
      if (options && std::filesystem::weakly_canonical(options->directory) ==
                         std::filesystem::weakly_canonical(pins.directory))
        throw std::invalid_argument(
            "pin-prefix-dir must differ from cache-disk");
      pin_disk = std::make_unique<cache::DiskStore>(ledger, pins.directory,
                                                    pins.capacity_bytes);
    }
    for (std::size_t i = 0; i < count; ++i) {
      auto entry = std::make_unique<Entry>();
      entry->preservation = Stream();
      entry->slot = std::make_unique<cache::LeasedSlot>(
          ledger, *resources->adapter, *entry->preservation,
          cache::SlotId{i + 1},
          cache::IdleSpillConfig{1ULL << 20, resources->stream,
                                 [this] {
                                   const std::lock_guard lock(mutex.native);
                                   return std::ranges::none_of(
                                       entries,
                                       [](const auto& e) { return e->busy; });
                                 },
                                 resources->reassignment_batch_size});
      entries.push_back(std::move(entry));
    }
    std::size_t automatic_bytes{}, max_bytes{}, slot_pinned_bytes{};
    if (!ram_enabled) {
      // Only disk captures copy through backing, so without a disk cache no
      // pool exists. The ledger admits what the slots already charged, that
      // backing, and request metadata.
      ram = options ? kDiskOnlyBackingBytes : 0;
      disk_backing = ram;
      slot_pinned_bytes =
          resources->pinned_bytes ? resources->pinned_bytes() : 0;
      const auto staging = ledger.Limits().staging_bytes;
      const auto limit =
          std::min(ledger.Limits().ram_bytes,
                   ledger.Snapshot().ram_bytes + ram + kOffMetadataBytes);
      ledger.ConstrainLimits({limit + staging, limit, staging, limit});
      if (resources->commit_backing) {
        if (ram) {
          resources->commit_backing(ram);
        } else {
          // No pool exists, so a payload allocation must fail, not crash.
          resources->prepare_prefill = nullptr;
          resources->can_capture = nullptr;
          resources->allocate =
              [](const cache::PayloadRequest&) -> ComponentCaptureBuffer {
            throw cache::ResourceExhausted();
          };
        }
      }
    } else if (resources->commit_backing) {
      const auto claim = this->runner->ResourceClaim();
      // The claim measures the host after the slots allocated, so bytes they
      // pinned are missing from it. Add them back: in full to the ceiling,
      // and half to the automatic budget, which is half the available RAM.
      // The ledger charges them on top of the cache's own budget.
      const auto pinned =
          resources->pinned_bytes ? resources->pinned_bytes() : 0;
      const auto measured = claim.retained_snapshot_capacity_bytes.value_or(0);
      const auto automatic = measured + pinned / 2;
      const auto ceiling =
          claim.retained_snapshot_ceiling_bytes.value_or(measured) + pinned;
      automatic_bytes =
          std::min(TextRunnerRamCacheOptions::kAutomaticMaxBytes, automatic);
      // Staging is pinned host memory too, so the host keeps its reserve
      // only if both fit under the ceiling.
      const auto staging = ledger.Limits().staging_bytes;
      max_bytes = ceiling > staging ? ceiling - staging : 0;
      ram = std::min(ram, max_bytes);
      if (automatic_ram)
        ram = std::min(ram, automatic);
      // Within the host ceiling and the limit the slots allocated under. Other
      // charges above the budget are refused here, as without pins.
      const auto limit =
          std::min({ram + pinned, max_bytes, ledger.Limits().ram_bytes});
      ledger.ConstrainLimits({limit + staging, limit, staging, limit});
      ram = std::min(ram, limit > pinned ? limit - pinned : 0);
      resources->commit_backing(ram);
      slot_pinned_bytes = pinned;
    } else {
      // Slots allocated under the host ceiling; return to the budget. This
      // ignores pinned_bytes: a runner that pins must commit backing.
      const auto staging = ledger.Limits().staging_bytes;
      ledger.ConstrainLimits({ram + staging, ram, staging, ram});
    }
    budget = ram_enabled ? ram : 0;
    if (pins.enabled())
      pin_streaming = std::make_unique<cache::StreamedStore>(
          ledger, *pin_disk, cache::kDefaultDiskStagingBytes,
          resources->staging, resources->stream);
    if (options) {
      // Validate/retire before even provisional catalog membership or index
      // publication, not merely on the first later restore.
      PreflightPackedDisk(*disk);
      catalog = std::make_unique<cache::DiskCatalog>(ledger, *disk, index);
      for (const auto& durable : disk->Entries()) {
        cache::ExecutionHistory::ObserveDurableId(
            durable.Manifest().checkpoint);
        if (durable.Manifest().compatibility == digest)
          (void)index.Insert(identity,
                             catalog->Find(durable.Manifest().checkpoint));
      }
      streaming = std::make_unique<cache::StreamedStore>(
          ledger, *disk, cache::kDefaultDiskStagingBytes, resources->staging,
          resources->stream);
      disk_capacity = options->capacity_bytes;
      disk_step = options->min_checkpoint_step_tokens;
      disk->ReclaimOrphans();
      if (disk->Stats().managed_bytes > disk_capacity) {
        // A lowered --cache-disk-bytes keeps its meaning: reclaim orphans and
        // temporaries first, then evict the oldest checkpoints by creation
        // (startup has no use history) to fit.
        disk->WaitForReclamation();
        std::size_t evicted{};
        while (disk->Stats().managed_bytes > disk_capacity &&
               catalog->EvictOne())
          ++evicted;
        const auto managed = disk->Stats().managed_bytes;
        const auto fields = " managed_bytes=" + std::to_string(managed) +
                            " capacity_bytes=" + std::to_string(disk_capacity);
        if (evicted)
          Logger::Info("cache",
                       "schema=component-cache-v1 event=disk_evicted "
                       "reason=budget entries=" +
                           std::to_string(evicted) + fields);
        if (managed > disk_capacity)
          Logger::Warn(
              "cache",
              "schema=component-cache-v1 event=disk_over_budget" + fields);
      }
      queue = std::make_unique<cache::PersistenceQueue>(
          ledger, *streaming, kPersistenceDepth, ram,
          [this] {
            const std::lock_guard lock(mutex.native);
            return std::ranges::none_of(
                entries, [](const auto& entry) { return entry->busy; });
          },
          [this](cache::CheckpointId id, bool success,
                 const cache::TransferTiming& timing) {
            const std::lock_guard lock(mutex);
            for (auto& checkpoint : pending)
              if (checkpoint && checkpoint->Id() == id)
                checkpoint.reset();
            std::shared_ptr<DurableCapture> completed;
            for (auto& capture : pending_durable)
              if (capture && capture->manifest.checkpoint == id) {
                completed = capture;
                capture.reset();
              }
            if (!success) {
              Logger::Info(
                  "cache",
                  "schema=component-cache-v1 event=persistence_incomplete "
                  "checkpoint_id=" +
                      std::to_string(id.value));
              return;
            }
            catalog->Track(id);
            if (const auto description = catalog->Find(id)) {
              if (completed)
                for (auto& entry : entries)
                  if (entry->branch == completed->branch &&
                      (!entry->durable ||
                       entry->durable->Manifest().tokens.size() <
                           description->Manifest().tokens.size()))
                    entry->durable = description;
              auto found = index.Lookup(
                  {identity, description->Manifest().tokens,
                   cache::InputIdentity(description->Manifest().tokens.size(),
                                        description->Manifest().input)});
              bool attached = false;
              for (const auto& candidate : found.candidates) {
                if (candidate.checkpoint && candidate.checkpoint->Id() == id) {
                  index.AttachDurable(candidate.entry, description);
                  attached = true;
                }
              }
              if (!attached)
                (void)index.Insert(identity, description);
              Logger::Info(
                  "cache",
                  "schema=component-cache-v1 event=published tokens=" +
                      std::to_string(description->Manifest().tokens.size()) +
                      " checkpoint_id=" + std::to_string(id.value) +
                      " payload_bytes=" +
                      std::to_string(PayloadBytes(description->Manifest())) +
                      " yield_ms=" + std::to_string(timing.yield_ns / 1e6) +
                      " stream_wait_ms=" +
                      std::to_string(timing.stream_wait_ns / 1e6) +
                      " filesystem_ms=" +
                      std::to_string(timing.filesystem_ns / 1e6) +
                      " io_lock_ms=" + std::to_string(timing.io_lock_ns / 1e6) +
                      " checksum_ms=" +
                      std::to_string(timing.checksum_ns / 1e6) +
                      " fsync_ms=" + std::to_string(timing.fsync_ns / 1e6));
            }
          },
          [this](const cache::DiskManifest& manifest) {
            // A checkpoint larger than the entire disk budget cannot be
            // admitted by deleting other conversations.
            const auto bytes = PayloadBytes(manifest);
            if (bytes > disk_capacity)
              return false;
            // Startup orphan reclamation holds the store lock asynchronously
            // and eviction only tries that lock, so a write racing it would
            // be dropped. Only this worker joins it once the queue exists.
            try {
              disk->WaitForReclamation();
            } catch (const std::exception& error) {
              Logger::Warn("cache", std::string("schema=component-cache-v1 "
                                                "event=disk_reclamation_failed "
                                                "reason=") +
                                        error.what());
            }
            const std::lock_guard lock(mutex);
            return catalog->EvictOne();
          });
    }
    Logger::Info("cache",
                 "schema=component-cache-v1 event=configured sessions=" +
                     std::to_string(count) +
                     " snapshot_entries=128 capacity_bytes=" +
                     std::to_string(budget) + " staging_capacity_bytes=" +
                     std::to_string(ledger.Limits().staging_bytes) +
                     " automatic_bytes=" + std::to_string(automatic_bytes) +
                     " max_bytes=" + std::to_string(max_bytes) +
                     " disk_capacity_bytes=" + std::to_string(disk_capacity) +
                     " slot_pinned_bytes=" + std::to_string(slot_pinned_bytes) +
                     " ram_cache=" + (ram_enabled ? "on" : "off") +
                     " disk_backing_bytes=" + std::to_string(disk_backing));
  }
  ~Impl() {
    // Settle immutable-source readers before slots/resources/index unwind.
    if (queue)
      queue->Drain();
    for (auto& entry : entries)
      entry->slot->StopIdleSpill();
    // Only after callbacks drain: retention teardown later runs after the
    // publication fields have been destroyed.
    index.SetInvalidationObserver({});
  }
  std::unique_ptr<cache::Stream> Stream() {
    auto stream = resources->stream();
    if (!stream)
      throw std::runtime_error("component cache transfer pool exhausted");
    return stream;
  }
  cache::DiskFileId File(std::uint64_t kind, std::uint64_t id) const {
    crypto::Sha256Hasher hash;
    hash.Update(nonce);
    std::array<std::uint8_t, 16> suffix{};
    for (std::size_t i = 0; i < 8; ++i) {
      suffix[i] = static_cast<std::uint8_t>(kind >> (8 * i));
      suffix[8 + i] = static_cast<std::uint8_t>(id >> (8 * i));
    }
    hash.Update(suffix);
    const auto bytes = hash.Finish();
    cache::DiskFileId file;
    std::copy_n(bytes.begin(), file.size(), file.begin());
    return file;
  }
  std::size_t Persist(std::shared_ptr<const cache::Checkpoint> checkpoint,
                      bool pin = false) {
    // Grid captures support RAM edits during prefill. Persist semantic prompt
    // and learned boundaries: an early grid must not consume the spacing
    // allowance of the first complete turn.
    if ((!pin && !queue) ||
        checkpoint->Purpose() == cache::CheckpointPurpose::kGrid)
      return 0;
    if (!pin) {
      const std::lock_guard lock(mutex.native);
      auto compatible_prefix = [&](const auto& tokens, const auto& input) {
        return input == checkpoint->Input() &&
               tokens.size() <= checkpoint->Boundary() &&
               std::equal(tokens.begin(), tokens.end(),
                          checkpoint->Tokens().begin());
      };
      if (checkpoint->Purpose() != cache::CheckpointPurpose::kLearned) {
        auto found = index.Lookup({identity, checkpoint->Tokens(),
                                   cache::InputIdentity(checkpoint->Boundary(),
                                                        checkpoint->Input())});
        std::size_t prior = 0;
        for (const auto& candidate : found.candidates)
          if (Published(candidate))
            prior =
                std::max(prior, static_cast<std::size_t>(candidate.boundary));
        for (const auto& queued : pending)
          if (queued && compatible_prefix(queued->Tokens(), queued->Input()))
            prior =
                std::max(prior, static_cast<std::size_t>(queued->Boundary()));
        for (const auto& queued : pending_durable)
          if (queued && compatible_prefix(queued->manifest.tokens,
                                          queued->manifest.input))
            prior = std::max(prior, queued->manifest.tokens.size());
        if (prior && checkpoint->Boundary() - prior < disk_step) {
          Logger::Info(
              "cache",
              "schema=component-cache-v1 event=persistence_refused "
              "reason=min_step tokens=" +
                  std::to_string(checkpoint->Boundary()) +
                  " checkpoint_id=" + std::to_string(checkpoint->Id().value));
          return 0;
        }
      }
    }
    cache::DiskManifest manifest{
        .checkpoint = checkpoint->Id(),
        .lineage = checkpoint->Lineage(),
        .compatibility = digest,
        .input = checkpoint->Input(),
        .tokens = {checkpoint->Tokens().begin(), checkpoint->Tokens().end()},
        .purpose = checkpoint->Purpose(),
        .rank = checkpoint->Rank()};
    std::vector<cache::PersistenceSource> sources;
    std::uint64_t private_id = checkpoint->Id().value;
    auto append = [&](const cache::Payload& payload, std::size_t bytes,
                      bool private_file, cache::DiskFileId file) {
      if (auto rows = payload.BorrowedStorage()) {
        // Read rows still in a slot from there: idle preservation only runs
        // while no request is busy, and a waiting job keeps its pins.
        sources.emplace_back(private_file, file, bytes, 1, std::move(rows),
                             resources->copy_observed_rows);
        return cache::DiskPayload{file, bytes, 0};
      }
      auto owner = payload.Owner();
      if (!owner)
        throw std::logic_error("persistence source is not immutable");
      sources.emplace_back(
          private_file, file, bytes, 1, payload.CommittedCharge(), owner,
          [owner, checkpoint](std::uint64_t offset,
                              std::span<std::byte> destination,
                              cache::Stream&) {
            std::memcpy(destination.data(),
                        static_cast<const std::byte*>(owner.get()) + offset,
                        destination.size());
            return cache::Completion(std::make_unique<SettledSignal>());
          });
      return cache::DiskPayload{file, bytes, 0};
    };
    for (const auto& c : checkpoint->Components()) {
      cache::DiskComponent component{.descriptor = c.descriptor,
                                     .position = c.position};
      if (c.private_state)
        component.private_state =
            append(*c.private_state, c.descriptor.state_bytes, true,
                   File(2 + c.descriptor.id.value, private_id));
      for (const auto& chunk : c.chunks)
        component.chunks.push_back(
            append(chunk.Storage(),
                   (chunk.End() - chunk.First()) * c.descriptor.row_bytes,
                   false, File(0, chunk.Id().value)));
      if (c.tail)
        component.tail = append(
            *c.tail,
            (c.position.valid_rows % c.descriptor.rows_per_chunk) *
                c.descriptor.row_bytes,
            true, File(0x100000000ULL + c.descriptor.id.value, private_id));
      manifest.components.push_back(std::move(component));
    }
    std::size_t bytes = 0;
    for (const auto& c : manifest.components) {
      for (const auto& chunk : c.chunks)
        bytes += chunk.bytes;
      if (c.tail)
        bytes += c.tail->bytes;
      if (c.private_state)
        bytes += c.private_state->bytes;
    }
    if (pin) {
      cache::TransferTiming timing;
      // Synchronous required publication, including checksums, fsync and the
      // atomic manifest rename, before this server can become ready.
      pin_streaming->Write(File(1, private_id), manifest, sources, timing);
      Logger::Info("cache",
                   "schema=pinned-prefix-v1 event=published tokens=" +
                       std::to_string(checkpoint->Boundary()) +
                       " payload_bytes=" + std::to_string(timing.bytes) +
                       " managed_bytes=" +
                       std::to_string(pin_disk->Stats().managed_bytes));
      return bytes;
    }
    auto job = cache::PersistenceJob(ledger, File(1, private_id), manifest,
                                     std::move(sources));
    std::size_t pending_slot;
    {
      const std::lock_guard lock(mutex.native);
      auto position = std::find(pending.begin(), pending.end(), nullptr);
      if (position == pending.end())
        return 0;
      pending_slot = position - pending.begin();
      *position = checkpoint;
    }
    if (!queue->TrySubmit(std::move(job))) {
      const std::lock_guard lock(mutex.native);
      if (pending[pending_slot] == checkpoint)
        pending[pending_slot].reset();
      return 0;
    }
    return bytes;
  }
  bool WithinDiskStep(std::span<const cache::Token> tokens,
                      const cache::Identity& input) {
    const std::lock_guard lock(mutex.native);
    auto found = index.Lookup(
        {identity, tokens, cache::InputIdentity(tokens.size(), input)});
    std::size_t prior = 0;
    for (const auto& candidate : found.candidates)
      if (Published(candidate))
        prior = std::max(prior, static_cast<std::size_t>(candidate.boundary));
    const auto compatible = [&](auto prefix, const auto& prefix_input) {
      return prefix_input == input && prefix.size() <= tokens.size() &&
             std::equal(prefix.begin(), prefix.end(), tokens.begin());
    };
    for (const auto& checkpoint : pending)
      if (checkpoint && compatible(checkpoint->Tokens(), checkpoint->Input()))
        prior =
            std::max(prior, static_cast<std::size_t>(checkpoint->Boundary()));
    for (const auto& capture : pending_durable)
      if (capture &&
          compatible(capture->manifest.tokens, capture->manifest.input))
        prior = std::max(prior, capture->manifest.tokens.size());
    return prior && tokens.size() - prior < disk_step;
  }
  std::size_t Persist(std::shared_ptr<DurableCapture> capture) {
    if (capture->sources.empty())
      return 0;
    if (WithinDiskStep(capture->manifest.tokens, capture->manifest.input) &&
        capture->manifest.purpose != cache::CheckpointPurpose::kLearned)
      return 0;
    auto job = cache::PersistenceJob(
        ledger, File(1, capture->manifest.checkpoint.value), capture->manifest,
        std::move(capture->sources));
    std::size_t pending_slot;
    {
      const std::lock_guard lock(mutex.native);
      const auto position =
          std::find(pending_durable.begin(), pending_durable.end(), nullptr);
      if (position == pending_durable.end())
        return 0;
      pending_slot = position - pending_durable.begin();
      *position = capture;
    }
    if (!queue->TrySubmit(std::move(job))) {
      const std::lock_guard lock(mutex.native);
      if (pending_durable[pending_slot] == capture)
        pending_durable[pending_slot].reset();
      return 0;
    }
    return PayloadBytes(capture->manifest);
  }
};

struct ComponentTextCache::Lease::Impl {
  ComponentTextCache::Impl* cache;
  ComponentTextCache::Impl::Entry* entry;
  cache::SlotLease slot;
  std::unique_ptr<TextRunnerState> state;
  cache::InputIdentity input;
  std::vector<cache::Token> prompt;
  std::size_t cached{}, restored{}, stable{}, capture_boundary{};
  double restore_time{};
  bool disk_hit{}, active{true}, preserve_source{}, pin_capture{};
  ComponentTextCache::Impl::Pin* pin_hit{};
  cache::CheckpointId source{};
  std::array<cache::CheckpointId, cache::RetentionPolicy::kRecordLimit>
      edit_boundaries{};
  std::size_t edit_boundary_count{};
  cache::ResourceCharge edit_metadata;
  SnapshotPurpose purpose{SnapshotPurpose::kContinuation};
  std::vector<cache::Token> capture_tokens;
  ContinuationLookup lookup_result;
  ComponentCacheMetrics metrics;
  Impl(ComponentTextCache::Impl* cache, ComponentTextCache::Impl::Entry* entry,
       cache::SlotLease slot, std::span<const cache::Token> prompt,
       cache::InputIdentity input, std::size_t stable)
      : cache(cache),
        entry(entry),
        slot(std::move(slot)),
        state(cache->runner->BindComponentState(*cache->resources->adapter,
                                                this->slot.Execution())),
        input(std::move(input)),
        prompt(prompt.begin(), prompt.end()),
        stable(stable),
        edit_metadata(cache->ledger
                          .Reserve(cache::ResourceCategory::kMetadata,
                                   sizeof(edit_boundaries))
                          .Convert()) {}
  cache::RetentionRequest Retain(
      std::span<const cache::Token> tokens,
      std::optional<std::size_t> new_payload_bytes = {}) const {
    return {
        .tokens = tokens,
        .compatibility = cache->identity,
        .input = At(input, std::min(tokens.size(), prompt.size())),
        .purpose = Purpose(purpose),
        .stable = std::min(stable, tokens.size()),
        .source = source,
        .preserve_source = preserve_source,
        .new_payload_bytes =
            new_payload_bytes
                ? *new_payload_bytes
                : entry->history->NewPayloadBytes(
                      cache->resources->adapter->Positions(slot.Execution())),
        .protected_checkpoints =
            std::span(edit_boundaries).first(edit_boundary_count),
        .reused = cached != 0,
        .reused_tokens = cached};
  }
  std::unique_ptr<TextRunnerSnapshot> CaptureDurable();
};

std::unique_ptr<TextRunnerSnapshot>
ComponentTextCache::Lease::Impl::CaptureDurable() {
  if (!cache->queue || !cache->resources->copy_observed_rows ||
      purpose == SnapshotPurpose::kHistory)
    throw cache::ResourceExhausted();
  const auto capture_input = At(input, capture_tokens.size());
  if (purpose != SnapshotPurpose::kBranchPoint &&
      cache->WithinDiskStep(capture_tokens, capture_input))
    return {};
  const auto positions = cache->resources->adapter->Positions(slot.Execution());
  const auto descriptors = cache->resources->adapter->Components();
  struct Copy {
    cache::PayloadRequest request;
    ComponentCaptureBuffer buffer;
    cache::DiskFileId file;
    bool private_file;
  };
  std::size_t pieces = 0;
  for (const auto& d : descriptors) {
    const auto p =
        std::ranges::find(positions, d.id, &cache::ComponentPosition::id);
    if (p == positions.end())
      throw std::logic_error("incomplete durable capture positions");
    pieces += d.kind == cache::ComponentKind::kPrivateState
                  ? 1
                  : p->valid_rows / d.rows_per_chunk +
                        bool(p->valid_rows % d.rows_per_chunk);
  }
  // Admit metadata before vectors/owners. No append payload allocation is
  // admitted for the native rows: observations may be retired at any guard.
  auto admission = cache->ledger.Reserve(
      cache::ResourceCategory::kMetadata,
      sizeof(DurableCapture) + sizeof(ComponentSnapshot) +
          capture_tokens.size() * sizeof(cache::Token) + capture_input.size() +
          descriptors.size() * sizeof(cache::DiskComponent) +
          pieces * (sizeof(cache::DiskPayload) +
                    sizeof(cache::PersistenceSource) + sizeof(Copy) + 256));
  auto durable = std::make_shared<DurableCapture>();
  durable->metadata = admission.Convert();
  std::shared_ptr<const cache::DiskDescription> ancestor;
  {
    const std::lock_guard lock(cache->mutex.native);
    durable->branch = entry->branch;
    ancestor = entry->durable;
  }
  // Component dependencies are owned by a particular store. The separate
  // pin directory is not an ordinary-disk ancestor, even for identical rows.
  if (ancestor && &ancestor->Store() != cache->disk.get())
    ancestor.reset();
  if (ancestor) {
    const auto& previous = ancestor->Manifest();
    if (previous.tokens.size() > capture_tokens.size() ||
        !std::equal(previous.tokens.begin(), previous.tokens.end(),
                    capture_tokens.begin()) ||
        previous.input != At(input, previous.tokens.size()))
      ancestor.reset();
    else {
      durable->ancestor_pin = ancestor->Pin();
      if (!durable->ancestor_pin)
        ancestor.reset();
    }
  }
  auto& manifest = durable->manifest;
  manifest.checkpoint = cache::ExecutionHistory::AllocateCheckpointId();
  manifest.lineage =
      ancestor ? ancestor->Manifest().lineage : entry->history->Lineage();
  manifest.compatibility = cache->digest;
  manifest.input = capture_input;
  manifest.tokens = capture_tokens;
  manifest.purpose = purpose == SnapshotPurpose::kBranchPoint
                         ? cache::CheckpointPurpose::kLearned
                         : cache::CheckpointPurpose::kPrompt;
  manifest.components.reserve(descriptors.size());
  durable->sources.reserve(pieces);
  std::vector<Copy> copies;
  copies.reserve(pieces);
  for (const auto& d : descriptors) {
    const auto p =
        std::ranges::find(positions, d.id, &cache::ComponentPosition::id);
    cache::DiskComponent component{.descriptor = d, .position = *p};
    const auto completed =
        d.kind == cache::ComponentKind::kAppendRows
            ? p->valid_rows / d.rows_per_chunk * d.rows_per_chunk
            : 0;
    cache::Rows inherited = 0;
    if (completed && ancestor) {
      const auto& components = ancestor->Manifest().components;
      const auto previous = std::ranges::find_if(
          components, [&](const auto& c) { return c.descriptor.id == d.id; });
      if (previous != components.end()) {
        const auto chunks = std::min<std::size_t>(previous->chunks.size(),
                                                  completed / d.rows_per_chunk);
        component.chunks.assign(previous->chunks.begin(),
                                previous->chunks.begin() + chunks);
        inherited = chunks * d.rows_per_chunk;
      }
    }
    std::shared_ptr<cache::BorrowedRows> rows;
    if (completed) {
      if (inherited != completed) {
        auto view = cache->resources->adapter->BorrowRowsView(
            slot.Execution(), d.id, inherited, completed);
        if (!view.empty())
          rows = slot.Observe(d.id, inherited, completed, view);
      }
      component.chunks.reserve(completed / d.rows_per_chunk);
    }
    const auto capture = [&](cache::Rows first, cache::Rows end,
                             bool private_file) {
      const auto bytes = d.kind == cache::ComponentKind::kPrivateState
                             ? d.state_bytes
                             : (end - first) * d.row_bytes;
      const auto file =
          cache->File(0, cache::ExecutionHistory::AllocateCheckpointId().value);
      if (rows && !private_file) {
        durable->sources.emplace_back(false, file, bytes, d.row_bytes, rows,
                                      (first - inherited) * d.row_bytes,
                                      cache->resources->copy_observed_rows);
      } else {
        const cache::PayloadRequest request{
            d.id,
            d.kind == cache::ComponentKind::kPrivateState
                ? cache::ResourceCategory::kPrivateState
            : private_file ? cache::ResourceCategory::kPrivateTail
                           : cache::ResourceCategory::kBackingAssigned,
            first, end, bytes};
        copies.push_back({request, {}, file, private_file});
      }
      return cache::DiskPayload{file, bytes, 0};
    };
    if (d.kind == cache::ComponentKind::kPrivateState)
      component.private_state = capture(0, 0, true);
    else {
      for (cache::Rows first = inherited; first < completed;
           first += d.rows_per_chunk)
        component.chunks.push_back(
            capture(first, first + d.rows_per_chunk, false));
      if (completed != p->valid_rows)
        component.tail = capture(completed, p->valid_rows, true);
    }
    manifest.components.push_back(std::move(component));
  }
  // Private states, tails, and rows the slot cannot lend are staged in RAM
  // backing. Allocate them largest first, as Capture does. Reclaim for the
  // staged bytes only, because Reclaim refuses a request larger than RAM. The
  // disk checkpoint does not replace a retained one, so keep the source.
  std::vector<cache::PayloadRequest> requests;
  requests.reserve(copies.size());
  std::size_t staged = 0;
  for (const auto& copy : copies) {
    requests.push_back(copy.request);
    staged += copy.request.bytes;
  }
  std::vector<std::size_t> allocation_order(copies.size());
  std::iota(allocation_order.begin(), allocation_order.end(), 0);
  std::ranges::sort(allocation_order, [&](auto a, auto b) {
    return requests[a].bytes != requests[b].bytes
               ? requests[a].bytes > requests[b].bytes
               : a < b;
  });
  auto retain = Retain(capture_tokens, staged);
  retain.preserve_source = true;
  // Nearby retry copies may replace their source even with preserve_source.
  // Disk staging publishes no RAM replacement, so protect it explicitly along
  // with this turn's edit boundaries.
  std::array<cache::CheckpointId, cache::RetentionPolicy::kRecordLimit + 1>
      protected_checkpoints{};
  auto protected_count = retain.protected_checkpoints.size();
  std::ranges::copy(retain.protected_checkpoints,
                    protected_checkpoints.begin());
  if (retain.source.value)
    protected_checkpoints[protected_count++] = retain.source;
  retain.protected_checkpoints =
      std::span(protected_checkpoints).first(protected_count);
  while (!copies.empty()) {
    try {
      if (cache->resources->can_capture &&
          !cache->resources->can_capture(requests))
        throw cache::ResourceExhausted();
      for (const auto i : allocation_order)
        copies[i].buffer = cache->resources->allocate(requests[i]);
      break;
    } catch (const cache::ResourceExhausted&) {
      for (auto& copy : copies)
        copy.buffer = {};
      const std::lock_guard lock(cache->mutex);
      if (!cache->retention.Reclaim(retain))
        throw;
    }
  }
  TransferBatch transfers(cache->resources->transfer_batch_size);
  for (auto& copy : copies) {
    const auto& r = copy.request;
    transfers.Add([&] { return cache->resources->stream(); },
                  [&](cache::Stream& stream) {
                    return r.category == cache::ResourceCategory::kPrivateState
                               ? cache->resources->adapter->CapturePrivate(
                                     slot.Execution(), r.component,
                                     copy.buffer.bytes.first(r.bytes), stream)
                               : cache->resources->adapter->CopyRowsOut(
                                     slot.Execution(), r.component, r.first,
                                     r.end, copy.buffer.bytes.first(r.bytes),
                                     stream);
                  });
  }
  transfers.Finish();
  for (auto& copy : copies) {
    auto payload = copy.buffer.finish();
    auto owner = payload.Owner();
    metrics.captured_bytes += payload.Bytes();
    metrics.copied_payload_bytes += copy.request.bytes;
    durable->sources.emplace_back(
        copy.private_file, copy.file, copy.request.bytes,
        copy.request.category == cache::ResourceCategory::kPrivateState
            ? 1
            : std::ranges::find(descriptors, copy.request.component,
                                &cache::ComponentDescriptor::id)
                  ->row_bytes,
        payload.CommittedCharge(), owner,
        [owner](auto offset, auto destination, cache::Stream&) {
          std::memcpy(destination.data(),
                      static_cast<const std::byte*>(owner.get()) + offset,
                      destination.size());
          return cache::Completion(std::make_unique<SettledSignal>());
        });
  }
  ++metrics.captures;
  metrics.captured_checkpoint_ids.push_back(durable->manifest.checkpoint.value);
  auto snapshot = std::make_unique<ComponentSnapshot>();
  snapshot->bytes = entry->history->NewPayloadBytes(positions);
  snapshot->durable = std::move(durable);
  return snapshot;
}

ComponentTextCache::ComponentTextCache(
    std::shared_ptr<TextModelRunner> runner, std::size_t count, std::size_t ram,
    const std::optional<TextRunnerDiskCacheOptions>& disk, bool automatic_ram,
    std::size_t ceiling, const TextRunnerPinnedPrefixOptions& pins,
    bool ram_enabled, bool publish_lookup)
    : impl_(std::make_unique<Impl>(std::move(runner), count, ram,
                                   std::max(ram, ceiling), disk, automatic_ram,
                                   pins, ram_enabled, publish_lookup)) {
  if (pins.enabled())
    InitializePins(pins);
  // Publish startup disk/pin inventory before the HTTP service becomes ready.
  const std::lock_guard lock(impl_->mutex);
}
void ComponentTextCache::InitializePins(
    const TextRunnerPinnedPrefixOptions& options) {
  // Validate the complete operator-supplied set before retiring anything.
  // File names and option order are not identity; exact tokens, supplemental
  // input and the strong executable/numerics digest remain the authority.
  struct Registration {
    cache::ResourceCharge metadata;
    std::vector<cache::Token> tokens;
    cache::CheckpointId existing;
  };
  auto registration_metadata =
      impl_->ledger
          .Reserve(cache::ResourceCategory::kMetadata,
                   options.files.size() * sizeof(Registration))
          .Convert();
  std::vector<Registration> registrations;
  registrations.reserve(options.files.size());
  if (registrations.capacity() != options.files.size())
    throw std::logic_error("unexpected pin registration capacity");
  constexpr std::size_t kMaxPrefixFileBytes = 16ULL << 20;
  for (const auto& path : options.files) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 ||
        file.tellg() > std::streamoff(kMaxPrefixFileBytes))
      throw std::invalid_argument(
          "pin prefix file must contain 1..16 MiB of rendered text");
    std::string rendered(static_cast<std::size_t>(file.tellg()), '\0');
    file.seekg(0);
    if (!file.read(rendered.data(),
                   static_cast<std::streamsize>(rendered.size())))
      throw std::runtime_error("cannot read pin prefix file");
    auto tokens = impl_->runner->TokenizeControl(rendered);
    if (tokens.empty() ||
        tokens.size() >= impl_->runner->Descriptor().max_context)
      throw std::invalid_argument(
          "pin prefix must be nonempty and leave context for a tail");
    if (std::ranges::any_of(registrations, [&](const auto& registration) {
          return registration.tokens == tokens;
        }))
      throw std::invalid_argument("duplicate exact pinned prefix registration");
    auto metadata = impl_->ledger
                        .Reserve(cache::ResourceCategory::kMetadata,
                                 tokens.capacity() * sizeof(cache::Token))
                        .Convert();
    registrations.push_back({std::move(metadata), std::move(tokens), {}});
  }
  auto& disk = *impl_->pin_disk;
  if (disk.Stats().rejected_manifests)
    throw std::runtime_error(
        "pinned prefix is corrupt or truncated; repair its directory "
        "explicitly");
  // Off may read existing lazy disk pins, but creating a pin needs a RAM
  // checkpoint. Check the whole set before retiring any old registration.
  if (!impl_->ram_enabled) {
    for (const auto& registration : registrations) {
      const auto existing =
          std::ranges::any_of(disk.Entries(), [&](const auto& e) {
            const auto& manifest = e.Manifest();
            return manifest.compatibility == impl_->digest &&
                   manifest.input.empty() &&
                   manifest.tokens == registration.tokens;
          });
      if (!existing)
        throw std::invalid_argument(
            "creating pinned prefixes requires an enabled RAM cache");
    }
  }
  // The existing v2 manifests ARE the registry: no migration or per-pin
  // subdirectories/budgets. One owned DiskStore bounds the sum, including
  // temporaries and orphans, and uses one shared staging buffer. The explicit
  // configured set replaces removed/incompatible identities, never via LRU.
  // Retire outside Entries() iteration: retirement invalidates that view.
  for (;;) {
    cache::CheckpointId replace{};
    for (const auto& entry : disk.Entries()) {
      const auto& manifest = entry.Manifest();
      cache::ExecutionHistory::ObserveDurableId(manifest.checkpoint);
      auto registration = std::ranges::find(registrations, manifest.tokens,
                                            &Registration::tokens);
      if (manifest.compatibility == impl_->digest && manifest.input.empty() &&
          registration != registrations.end()) {
        if (registration->existing.value &&
            registration->existing != manifest.checkpoint)
          throw std::runtime_error("duplicate durable pinned prefix identity");
        registration->existing = manifest.checkpoint;
      } else {
        replace = manifest.checkpoint;
      }
    }
    if (!replace.value)
      break;
    if (!disk.Retire(replace))
      throw std::runtime_error("cannot replace incompatible pinned prefix");
    Logger::Info("cache", "schema=pinned-prefix-v1 event=identity_changed");
  }
  const bool creating = std::ranges::any_of(
      registrations, [](const auto& r) { return !r.existing.value; });
  if (creating) {
    disk.ReclaimOrphans();
    disk.WaitForReclamation();
  }
  if (disk.Stats().managed_bytes > options.capacity_bytes)
    throw std::runtime_error(
        "pin registry exceeds its reduced byte budget; not evicted");
  impl_->pin_metadata = impl_->ledger
                            .Reserve(cache::ResourceCategory::kMetadata,
                                     registrations.size() * sizeof(Impl::Pin))
                            .Convert();
  impl_->pins.reserve(registrations.size());
  if (impl_->pins.capacity() != registrations.size())
    throw std::logic_error("unexpected pin registry capacity");
  impl_->lazy_pins = options.lazy;
  for (const auto& registration : registrations) {
    const auto& tokens = registration.tokens;
    const auto existing = registration.existing;
    if (existing.value &&
        PackedKvIngress(impl_->runner->Descriptor(),
                        impl_->resources->adapter->Components())) {
      // Both eager and lazy existing pins are external input. Preserve the
      // explicit-repair policy for a bad pinned root: refuse readiness rather
      // than silently replace the operator's registered publication.
      impl_->PreflightDiskAppendRows(disk, existing);
    }
    auto& pin = impl_->pins.emplace_back();
    pin.durable = existing;
    if (existing.value && options.lazy) {
      pin.entry = impl_->index.Insert(impl_->identity, disk.Describe(existing));
      Logger::Info("cache",
                   "schema=pinned-prefix-v1 event=ready mode=lazy tokens=" +
                       std::to_string(tokens.size()) +
                       " ram_payload_bytes=0 managed_bytes=" +
                       std::to_string(disk.Stats().managed_bytes));
      continue;
    }
    const auto started = Clock::now();
    // Lease a cold destination, then restore this particular pin publication.
    // A general lookup could fall back to an equal/deeper ordinary checkpoint
    // and accidentally claim warm readiness while the actual pin is corrupt.
    auto lease = Acquire(tokens, {}, {}, false, 0, true);
    if (!lease)
      throw std::runtime_error("cannot lease pinned-prefix startup slot");
    auto& state = dynamic_cast<TextRunnerState&>(lease->state());
    if (existing.value) {
      cache::TransferTiming timing;
      if (!impl_->pin_streaming->Restore(
              existing, impl_->digest, *impl_->resources->adapter,
              lease->impl_->slot.Execution(), timing, {}, {},
              [&](bool private_file, cache::DiskFileId file) {
                disk.InvalidateDependency(private_file, file);
              }))
        throw std::runtime_error(
            "pinned prefix restore failed; refusing warm readiness");
      lease->impl_->entry->history = std::make_unique<cache::ExecutionHistory>(
          cache::ExecutionHistory::RestoredUnshared(
              impl_->ledger, impl_->resources->adapter->Components(),
              impl_->identity));
      impl_->runner->SetPromptContext(state, {});
      impl_->runner->ReconcileComponentState(state, tokens.size());
      if (!impl_->runner->HasDecodeFrontier(state))
        throw std::runtime_error(
            "pinned prefix has no complete decode frontier");
      lease->impl_->cached = tokens.size();
    }
    auto offset = lease->cached_tokens();
    while (offset < tokens.size()) {
      const auto count = std::min<std::size_t>(4096, tokens.size() - offset);
      lease->PreparePrefill(count);
      const auto step = impl_->runner->Prefill(state, tokens, offset, count);
      if (!step.consumed_tokens || step.consumed_tokens > count)
        throw std::runtime_error("pinned prefix prefill made invalid progress");
      offset += step.consumed_tokens;
    }
    if (impl_->runner->CheckpointPosition(state) != tokens.size())
      throw std::runtime_error(
          "pinned prefix checkpoint is not at the exact boundary");
    const auto bytes = lease->CaptureBytes();
    if (bytes > impl_->budget ||
        !lease->TryReserveSnapshot(bytes, tokens.size(), true,
                                   SnapshotPurpose::kBranchPoint, tokens))
      throw std::runtime_error(
          "pin prefix cannot fit RAM; use lazy mode for an existing pin");
    // Required RAM warmth means materialized host backing, not native rows that
    // will need spilling when the temporary startup slot is reassigned.
    lease->impl_->pin_capture = true;
    auto snapshot = lease->Capture();
    const auto* component =
        dynamic_cast<const ComponentSnapshot*>(snapshot.get());
    if (!component || !component->checkpoint)
      throw std::runtime_error("pin prefix did not materialize in RAM");
    const auto checkpoint = component->checkpoint;
    if (!existing.value) {
      (void)impl_->Persist(checkpoint, true);
      pin.durable = checkpoint->Id();
    }
    const auto payload_bytes = ResidentBytes(*checkpoint);
    if (options.lazy) {
      pin.entry =
          impl_->index.Insert(impl_->identity, disk.Describe(pin.durable));
    } else {
      std::vector<cache::ComponentAvailability> available;
      for (const auto& c : checkpoint->Components())
        available.push_back({c.descriptor.id, true, false});
      pin.entry = impl_->index.Insert(checkpoint, std::move(available));
      pin.ram = checkpoint;
    }
    lease->Invalidate();
    Logger::Info(
        "cache",
        "schema=pinned-prefix-v1 event=ready mode=" +
            std::string(options.lazy ? "lazy" : "ram") +
            " restored=" + std::to_string(bool(existing.value)) +
            " tokens=" + std::to_string(tokens.size()) + " ram_payload_bytes=" +
            std::to_string(options.lazy ? 0 : payload_bytes) +
            " managed_bytes=" + std::to_string(disk.Stats().managed_bytes) +
            " startup_ms=" +
            std::to_string(std::chrono::duration<double, std::milli>(
                               Clock::now() - started)
                               .count()));
  }
}
ComponentTextCache::~ComponentTextCache() = default;
std::size_t ComponentTextCache::capacity() const noexcept {
  return impl_->entries.size();
}

std::unique_ptr<ComponentTextCache::Lease> ComponentTextCache::Acquire(
    std::span<const ContinuationToken> prompt,
    const ContinuationCache::CancellationCheck& cancelled,
    std::shared_ptr<const TextPromptContext> context, bool reuse,
    std::size_t stable, bool stop_at_eos) {
  if (cancelled && cancelled())
    return {};
  if (impl_->resources->test_barrier)
    impl_->resources->test_barrier("admission");
  auto input = Inputs(prompt.size(), context.get());
  cache::PrefixLookup found;
  Impl::Entry* entry = nullptr;
  std::optional<cache::BorrowedLocation> live;
  std::size_t common = 0;
  std::uint64_t admission_version{}, acquisition_id{};
  {
    const std::lock_guard lock(impl_->mutex);
    // Last completed publication BEFORE this admission's own invalidation.
    // Counting that invalidation as a later version would excuse every race.
    const auto version = impl_->lookup_version.load(std::memory_order_relaxed);
    admission_version = version - (version & 1);
    acquisition_id = impl_->next_acquisition++;
    if (impl_->catalog)
      impl_->catalog->Reconcile();
    found = impl_->Plan(prompt, input, stable, reuse);
    // A miss reports how far a stored prompt agreed, so an empty cache can
    // be told apart from one holding only incompatible boundaries.
    if (reuse && !found.selected)
      common =
          impl_->index.CommonPrefixTokens({impl_->identity, prompt, input});
    if (found.selected && found.selected->live) {
      const auto location = *found.selected->live;
      auto& selected_entry = *impl_->entries.at(location.slot.value - 1);
      if (!selected_entry.busy) {
        entry = &selected_entry;
        live = location;
      }
    }
    if (!entry) {
      // A frontier far beyond every checkpoint of its prompt is the only
      // copy of that work, as for a conversation larger than the RAM budget
      // whose captures are refused: take it last. Then completed
      // preservation first: it needs no foreground copy. Among those, a live
      // frontier holds generated tokens no checkpoint covers, such as the
      // tool call the next turn replays. Take a slot without one, then a
      // shallow frontier before a deep one, then the least recently used.
      // Faster idle preservation must not make short side requests evict a
      // deep conversation merely because its backing copy finished earlier.
      constexpr std::size_t kUncoveredTokens = 4096;
      const auto uncovered = [&](const Impl::Entry& e) {
        if (!e.live || e.live_tokens.size() <= kUncoveredTokens)
          return false;
        return !impl_->index.HasCheckpoint(
            {impl_->identity, e.live_tokens,
             cache::InputIdentity(e.live_tokens.size(), e.live_input)},
            e.live_tokens.size() - kUncoveredTokens);
      };
      using Rank = std::tuple<bool, bool, bool, bool, std::uint64_t>;
      const auto rank = [&](const Impl::Entry& e) {
        return Rank{uncovered(e), !e.slot->PreservationComplete(), bool(e.live),
                    e.live_tokens.size() >= kUncoveredTokens, e.used};
      };
      std::optional<Rank> selected_rank;
      for (auto& candidate : impl_->entries) {
        if (candidate->busy)
          continue;
        if (!entry) {
          entry = candidate.get();
          continue;
        }
        if (!selected_rank)
          selected_rank = rank(*entry);
        const auto candidate_rank = rank(*candidate);
        if (candidate_rank < *selected_rank) {
          entry = candidate.get();
          selected_rank = candidate_rank;
        }
      }
      // Keep ordinary cold selection unchanged. Only a cancelled executor
      // with outstanding work needs an alternative; do not query every
      // stream or look up exact checkpoint coverage on normal admissions.
      if (entry && !entry->slot->ExecutionComplete()) {
        if (!selected_rank)
          selected_rank = rank(*entry);
        Impl::Entry* ready = nullptr;
        std::optional<Rank> ready_rank;
        for (auto& candidate : impl_->entries) {
          if (candidate.get() == entry || candidate->busy ||
              !candidate->slot->ExecutionComplete())
            continue;
          const auto candidate_rank = rank(*candidate);
          // A ready slot must not sacrifice uncovered work or introduce
          // foreground preservation. Keep a deep generated tail unless an
          // exact checkpoint or another idle live copy still covers it.
          const auto live_copy = [&] {
            return candidate->history &&
                   std::ranges::any_of(impl_->entries, [&](const auto& other) {
                     return other != candidate && !other->busy && other->live &&
                            other->history &&
                            other->history->Lineage() ==
                                candidate->history->Lineage() &&
                            other->live_input == candidate->live_input &&
                            other->live_tokens == candidate->live_tokens;
                   });
          };
          if (std::tie(std::get<0>(candidate_rank),
                       std::get<1>(candidate_rank)) >
                  std::tie(std::get<0>(*selected_rank),
                           std::get<1>(*selected_rank)) ||
              (candidate->live &&
               candidate->live_tokens.size() >= kUncoveredTokens &&
               !impl_->index.HasCheckpoint(
                   {impl_->identity, candidate->live_tokens,
                    cache::InputIdentity(candidate->live_tokens.size(),
                                         candidate->live_input)},
                   candidate->live_tokens.size()) &&
               !live_copy()))
            continue;
          if (!ready || candidate_rank < *ready_rank) {
            ready = candidate.get();
            ready_rank = candidate_rank;
          }
        }
        if (ready)
          entry = ready;
      }
    }
    if (!entry)
      throw cache::SlotBusy();
    entry->busy = true;
    entry->used = ++impl_->clock;
    if (!live) {
      entry->branch = std::make_shared<const std::uint8_t>(0);
      entry->durable.reset();
    }
    if (entry->live) {
      impl_->index.Erase(*entry->live);
      entry->live.reset();
      entry->live_tokens.clear();
    }
  }
  std::unique_ptr<Lease> lease;
  try {
    auto slot = entry->slot->Acquire(live);
    lease = std::make_unique<Lease>(std::make_unique<Lease::Impl>(
        impl_.get(), entry, std::move(slot), prompt, input, stable));
    auto& request = *lease->impl_;
    request.metrics.acquisition_id = acquisition_id;
    request.metrics.admission_index_version = admission_version;
    request.metrics.selected_entry_id =
        found.selected ? found.selected->entry.value : 0;
    if (impl_->resources->test_barrier)
      impl_->resources->test_barrier("selected");
    request.state->SetCancellationCheck(cancelled);
    request.state->SetStopAtEos(stop_at_eos);
    impl_->runner->SetPromptContext(*request.state, context);
    const auto started = Clock::now();
    // Cold Acquire has already reset the slot. Only an attempted restore (or
    // an unusable live frontier) can require another reset before cold use.
    bool clean = !live;
    // A frontier captured between prefill passes cannot select the next token
    // of a prompt that ends there; serve that prompt from a shallower boundary.
    const auto decodable = [&](std::size_t boundary) {
      return boundary < prompt.size() ||
             impl_->runner->HasDecodeFrontier(*request.state);
    };
    if (live && !decodable(found.selected->boundary)) {
      live.reset();
      const std::lock_guard lock(impl_->mutex);
      entry->branch = std::make_shared<const std::uint8_t>(0);
      entry->durable.reset();
    }
    bool restored = bool(live);
    if (live)
      request.cached = found.selected->boundary;
    for (const auto& candidate : found.candidates) {
      if (restored || (cancelled && cancelled()))
        break;
      if (candidate.live)
        continue;
      ++request.metrics.restore_attempts;
      std::string_view failure_reason = "ram_begin_restore";
      const auto failed = [&](std::string_view reason) {
        ++request.metrics.restore_failures;
        request.metrics.restore_failure_reason = reason;
        Logger::Warn(
            "cache",
            "schema=cache-lookup-v1 event=restore_failure acquisition_id=" +
                std::to_string(acquisition_id) +
                " entry_id=" + std::to_string(candidate.entry.value) +
                " reason=" + std::string(reason));
        const std::lock_guard lock(impl_->mutex);
        // Cold fallback does not validate this source. Quarantine its index
        // entry rather than re-publish an executable claim after rejection.
        // Retention/payload ownership and already selected readers are intact.
        impl_->index.Erase(candidate.entry);
      };
      try {
        if (candidate.checkpoint) {
          const auto& checkpoint = *candidate.checkpoint;
          std::vector<cache::ComponentPosition> positions;
          for (const auto& c : checkpoint.Components())
            positions.push_back(c.position);
          clean = false;
          impl_->resources->adapter->BeginRestore(request.slot.Execution(),
                                                  positions);
          // Every pin outlives all submitted loads, including failure drain.
          // Borrowed views name immutable native rows; materialized views name
          // assigned host backing. No source slot execution lease is needed.
          failure_reason = "ram_pin";
          std::size_t pin_count = 0;
          for (const auto& c : checkpoint.Components())
            pin_count += c.chunks.size() + bool(c.tail) + bool(c.private_state);
          auto pin_metadata =
              impl_->ledger.Reserve(cache::ResourceCategory::kMetadata,
                                    pin_count * sizeof(cache::RowPin));
          std::vector<cache::RowPin> pins;
          pins.reserve(pin_count);
          for (const auto& c : checkpoint.Components()) {
            if (c.private_state)
              pins.push_back(c.private_state->PinRows());
            for (const auto& chunk : c.chunks)
              pins.push_back(chunk.Storage().PinRows());
            if (c.tail)
              pins.push_back(c.tail->PinRows());
          }
          failure_reason = "ram_transfer";
          TransferBatch transfers(impl_->resources->transfer_batch_size);
          std::size_t base = 0;
          for (const auto& c : checkpoint.Components()) {
            auto load = [&](const cache::RowPin& pin, cache::Rows first,
                            cache::Rows end) {
              const auto bytes =
                  c.descriptor.kind == cache::ComponentKind::kPrivateState
                      ? c.descriptor.state_bytes
                      : (end - first) * c.descriptor.row_bytes;
              if (pin.Bytes().empty())
                throw std::logic_error("RAM source has no readable view");
              const std::span<const std::byte> buffer(pin.Bytes().data(),
                                                      bytes);
              transfers.Add(
                  [&] { return impl_->resources->stream(); },
                  [&](cache::Stream& stream) {
                    return c.descriptor.kind ==
                                   cache::ComponentKind::kPrivateState
                               ? impl_->resources->adapter->LoadPrivate(
                                     request.slot.Execution(), c.descriptor.id,
                                     buffer, stream)
                               : impl_->resources->adapter->CopyRowsIn(
                                     request.slot.Execution(), c.descriptor.id,
                                     first, end, buffer, stream);
                  });
              request.restored += bytes;
            };
            if (c.private_state)
              load(pins[base++], 0, 0);
            auto same_source = [](const cache::RowPin& a,
                                  const cache::RowPin& b) {
              if (a.Location() || b.Location())
                return a.Location() == b.Location();
              return SameOwner(a.Owner(), b.Owner());
            };
            bool loaded_tail = false;
            for (std::size_t i = 0; i < c.chunks.size();) {
              const auto& chunk = c.chunks[i];
              auto end = chunk.End();
              const auto& pin = pins[base + i];
              const auto* data = pin.Bytes().data();
              auto next = i + 1;
              while (data && next < c.chunks.size() &&
                     c.chunks[next].First() == end &&
                     same_source(pin, pins[base + next]) &&
                     pins[base + next].Bytes().data() ==
                         data +
                             (end - chunk.First()) * c.descriptor.row_bytes) {
                end = c.chunks[next].End();
                ++next;
              }
              if (data && next == c.chunks.size() && c.tail &&
                  same_source(pin, pins[base + c.chunks.size()]) &&
                  end == c.position.valid_rows / c.descriptor.rows_per_chunk *
                             c.descriptor.rows_per_chunk &&
                  pins[base + c.chunks.size()].Bytes().data() ==
                      data + (end - chunk.First()) * c.descriptor.row_bytes) {
                end = c.position.valid_rows;
                loaded_tail = true;
              }
              load(pin, chunk.First(), end);
              i = next;
            }
            base += c.chunks.size();
            if (c.tail) {
              if (!loaded_tail)
                load(pins[base],
                     c.position.valid_rows / c.descriptor.rows_per_chunk *
                         c.descriptor.rows_per_chunk,
                     c.position.valid_rows);
              ++base;
            }
          }
          transfers.Finish();
          failure_reason = "ram_validate";
          if (!impl_->resources->adapter->Validate(request.slot.Execution(),
                                                   positions))
            throw std::runtime_error("component RAM restore validation failed");
          failure_reason = "ram_admission";
          entry->history = std::make_unique<cache::ExecutionHistory>(
              cache::ExecutionHistory::Restored(impl_->ledger, checkpoint));
          request.source = checkpoint.Id();
          {
            const std::lock_guard lock(impl_->mutex);
            if (std::ranges::none_of(impl_->pins, [&](const auto& pin) {
                  return pin.ram && pin.ram->Id() == checkpoint.Id();
                }))
              impl_->retention.Touch(checkpoint.Id());
          }
          restored = true;
        } else if (candidate.durable) {
          clean = false;
          cache::TransferTiming timing;
          const auto id = candidate.durable->Manifest().checkpoint;
          const bool pin = &candidate.durable->Store() == impl_->pin_disk.get();
          auto* streaming =
              pin ? impl_->pin_streaming.get() : impl_->streaming.get();
          failure_reason = "disk_restore_exception";
          if (!streaming) {
            failed("disk_streaming_unavailable");
            continue;
          }
          restored = streaming->Restore(
              id, impl_->digest, *impl_->resources->adapter,
              request.slot.Execution(), timing, {}, {},
              [this, pin](bool private_file, cache::DiskFileId file) {
                const std::lock_guard lock(impl_->mutex);
                if (pin) {
                  impl_->pin_disk->InvalidateDependency(private_file, file);
                  for (auto& registration : impl_->pins) {
                    if (registration.entry &&
                        !impl_->pin_disk->Pin(registration.durable)) {
                      impl_->index.Erase(*registration.entry);
                      registration.entry.reset();
                    }
                  }
                  Logger::Warn("cache",
                               "schema=pinned-prefix-v1 event=corrupt");
                } else {
                  impl_->catalog->Invalidate(private_file, file);
                }
              });
          if (!restored)
            failed("disk_restore_rejected");
          if (restored) {
            request.disk_hit = true;
            if (pin) {
              auto registration =
                  std::ranges::find(impl_->pins, id, &Impl::Pin::durable);
              if (registration != impl_->pins.end())
                request.pin_hit = &*registration;
            }
            request.restored = timing.bytes;
            Logger::Info(
                "cache",
                "schema=component-cache-v1 event=disk_restore tokens=" +
                    std::to_string(candidate.boundary) +
                    " bytes=" + std::to_string(timing.bytes) + " pieces=" +
                    std::to_string(timing.pieces) + " stream_acquire_ms=" +
                    std::to_string(timing.stream_acquire_ns / 1e6) +
                    " stream_wait_ms=" +
                    std::to_string(timing.stream_wait_ns / 1e6) +
                    " filesystem_ms=" +
                    std::to_string(timing.filesystem_ns / 1e6) +
                    " checksum_ms=" + std::to_string(timing.checksum_ns / 1e6) +
                    " staging_wait_ms=" +
                    std::to_string(timing.staging_wait_ns / 1e6));
            entry->history = std::make_unique<cache::ExecutionHistory>(
                cache::ExecutionHistory::RestoredUnshared(
                    impl_->ledger, impl_->resources->adapter->Components(),
                    impl_->identity));
            const std::lock_guard lock(impl_->mutex);
            if (!pin)
              impl_->catalog->Touch(id);
          }
        }
        if (restored && !decodable(candidate.boundary)) {
          // Intentional conservative full-boundary fallback, not a failed
          // transfer/restore. Lookup omits the same unready boundary.
          if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
            throw std::runtime_error("component restore reset failed");
          entry->history.reset();
          request.source = {};
          request.restored = 0;
          request.disk_hit = false;
          request.pin_hit = nullptr;
          restored = false;
          continue;
        }
        if (restored) {
          request.cached = candidate.boundary;
          const std::lock_guard lock(impl_->mutex);
          entry->durable = candidate.durable;
        }
      } catch (...) {
        failed(failure_reason);
        if (!impl_->resources->adapter->Invalidate(request.slot.Execution()))
          throw;
        clean = true;
        restored = false;
        request.restored = 0;
        request.disk_hit = false;
        request.pin_hit = nullptr;
        request.source = {};
        entry->history.reset();
      }
    }
    if (!restored) {
      if (!clean &&
          !impl_->resources->adapter->Invalidate(request.slot.Execution()))
        throw std::runtime_error("component cold initialization failed");
      entry->history = std::make_unique<cache::ExecutionHistory>(
          cache::ExecutionHistory::Cold(impl_->ledger,
                                        impl_->resources->adapter->Components(),
                                        impl_->identity));
    }
    // Restore/reset may clear the attachment; install this request again.
    impl_->runner->SetPromptContext(*request.state, context);
    impl_->runner->ReconcileComponentState(*request.state, request.cached);
    if (request.pin_hit && impl_->lazy_pins && impl_->ram_enabled) {
      auto& pin = *request.pin_hit;
      bool warm = false;
      {
        const std::lock_guard lock(impl_->mutex);
        if (!pin.ram && pin.entry && !pin.warming) {
          pin.warming = true;
          warm = true;
        }
      }
      if (warm) {
        try {
          const auto bytes = lease->CaptureBytes();
          if (bytes > impl_->budget ||
              !lease->TryReserveSnapshot(bytes, request.cached, true,
                                         SnapshotPurpose::kBranchPoint, {}))
            throw cache::ResourceExhausted();
          request.pin_capture = true;
          const auto snapshot = lease->Capture();
          const auto* component =
              dynamic_cast<const ComponentSnapshot*>(snapshot.get());
          if (!component || !component->checkpoint)
            throw cache::ResourceExhausted();
          const auto checkpoint = component->checkpoint;
          std::vector<cache::ComponentAvailability> available;
          for (const auto& c : checkpoint->Components())
            available.push_back({c.descriptor.id, true, false});
          {
            const std::lock_guard lock(impl_->mutex);
            const auto entry_id =
                impl_->index.Insert(checkpoint, std::move(available));
            if (pin.entry)
              impl_->index.Erase(*pin.entry);
            pin.entry = entry_id;
            pin.ram = checkpoint;
          }
          request.source = checkpoint->Id();
          Logger::Info("cache",
                       "schema=pinned-prefix-v1 event=demand_warm tokens=" +
                           std::to_string(request.cached) +
                           " ram_payload_bytes=" +
                           std::to_string(ResidentBytes(*checkpoint)));
        } catch (const std::exception&) {
          // A lazy pin remains durable and usable when the bounded RAM pool
          // cannot hold it; never evict another pin or rewrite the prefix.
          Logger::Info(
              "cache",
              "schema=pinned-prefix-v1 event=warm_refused reason=ram_capacity");
        }
        request.pin_capture = false;
        const std::lock_guard lock(impl_->mutex);
        pin.warming = false;
      }
    }
    request.restore_time =
        request.restored
            ? std::chrono::duration<double, std::milli>(Clock::now() - started)
                  .count()
            : 0;
    request.metrics.reuse = live               ? "live"
                            : request.disk_hit ? "disk"
                            : request.cached   ? "ram"
                                               : "none";
    request.metrics.selection_reason = static_cast<std::uint32_t>(found.reason);
    request.metrics.selected = request.cached;
    request.metrics.restored_bytes = request.restored;
    for (const auto& candidate : found.candidates) {
      auto& deepest = candidate.live         ? request.metrics.deepest_live
                      : candidate.checkpoint ? request.metrics.deepest_ram
                                             : request.metrics.deepest_disk;
      deepest = std::max(deepest, static_cast<std::size_t>(candidate.boundary));
    }
    std::size_t nearest = 0;
    for (const auto& candidate : found.candidates)
      nearest = std::max(nearest, static_cast<std::size_t>(candidate.boundary));
    request.lookup_result = {request.cached      ? ""
                             : !reuse            ? "disabled"
                             : common || nearest ? "no_compatible_boundary"
                                                 : "no_checkpoint",
                             request.cached ? 0 : common,
                             request.cached ? request.cached : nearest};
    Logger::Info("cache",
                 "schema=component-cache-v1 event=acquire reuse=" +
                     std::string(live               ? "live"
                                 : request.disk_hit ? "disk"
                                 : request.cached   ? "ram"
                                                    : "none") +
                     " selected_tokens=" + std::to_string(request.cached) +
                     " restored_bytes=" + std::to_string(request.restored) +
                     " acquisition_id=" + std::to_string(acquisition_id) +
                     " admission_index_version=" +
                     std::to_string(admission_version) + " selected_entry_id=" +
                     std::to_string(request.metrics.selected_entry_id) +
                     " restore_attempts=" +
                     std::to_string(request.metrics.restore_attempts) +
                     " restore_failures=" +
                     std::to_string(request.metrics.restore_failures) +
                     " restore_failure_reason=" +
                     request.metrics.restore_failure_reason);
    return lease;
  } catch (...) {
    if (lease)
      lease->Invalidate();
    else {
      const std::lock_guard lock(impl_->mutex);
      entry->busy = false;
    }
    throw;
  }
}

std::optional<TextPendingPublication> ComponentTextCache::PendingPublication(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    std::uint64_t checkpoint_id) const {
  if (!impl_->queue)
    return std::nullopt;
  const auto input = Inputs(prompt.size(), context);
  const std::lock_guard lock(impl_->mutex.native);
  const auto found =
      impl_->index.Lookup({impl_->identity, prompt, input, 0, true});
  const std::size_t best = found.selected ? found.selected->boundary : 0;
  std::optional<TextPendingPublication> selected;
  for (const auto& capture : impl_->pending_durable) {
    if (!capture)
      continue;
    const auto id = capture->manifest.checkpoint.value;
    if (checkpoint_id && id != checkpoint_id)
      continue;
    const auto& tokens = capture->manifest.tokens;
    if (tokens.size() > prompt.size() ||
        capture->manifest.input != At(input, tokens.size()) ||
        !std::equal(tokens.begin(), tokens.end(), prompt.begin()))
      continue;
    TextPendingPublication candidate{id, best, tokens.size()};
    if (!checkpoint_id &&
        candidate.Gain() <= TextPendingPublication::kMinimumGainTokens)
      continue;
    if (!selected || candidate.pending_tokens > selected->pending_tokens ||
        (candidate.pending_tokens == selected->pending_tokens &&
         id < selected->checkpoint_id))
      selected = candidate;
  }
  return selected;
}

std::span<const std::uint8_t> ComponentTextCache::CompatibilityIdentity()
    const {
  return impl_->identity;
}

std::optional<ContinuationReuse> ComponentTextCache::Lookup(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool reuse, std::size_t stable) const {
  const auto input = Inputs(prompt.size(), context);
  const auto version = impl_->lookup_version.load(std::memory_order_acquire);
  const auto view = impl_->lookup_view.load(std::memory_order_acquire);
  const auto observation =
      impl_->next_observation.fetch_add(1, std::memory_order_relaxed);
  const auto observed = [&](ContinuationReuse result, std::uint64_t entry,
                            std::string_view state, std::uint64_t finished) {
    result.observation_id = observation;
    if (impl_->lookup_enabled)
      Logger::Info(
          "cache",
          "schema=cache-lookup-v1 event=lookup lookup_id=" +
              std::to_string(observation) +
              " entry_id=" + std::to_string(entry) +
              " index_version=" + std::to_string(result.index_version) +
              " version_start=" + std::to_string(version) + " version_finish=" +
              std::to_string(finished) + " state=" + std::string(state) +
              " matched_tokens=" + std::to_string(result.matched_tokens));
    return result;
  };
  const auto unknown = [&](std::uint64_t generation, std::string_view state) {
    ContinuationReuse result;
    result.index_version = generation;
    return observed(result, 0, state, generation);
  };
  if (impl_->resources->test_barrier)
    impl_->resources->test_barrier("lookup_observed");
  if (version & 1)
    return unknown(version, "mutation_in_flight");
  if (!view)
    return unknown(version, "unpublished");
  const auto candidates =
      view->index.Lookup({impl_->identity, prompt, input, stable, reuse});
  ContinuationReuse result;
  std::uint64_t selected_entry = 0;
  // Match Acquire's full-boundary readiness and ordered fallback rules. Busy
  // leases have no live index entry; their immutable checkpoints still count.
  if (!candidates.empty() && candidates.front().live &&
      (candidates.front().boundary < prompt.size() ||
       candidates.front().full_ready)) {
    result = {candidates.front().boundary, "live"};
    selected_entry = candidates.front().entry.value;
  } else {
    for (const auto& candidate : candidates) {
      if (candidate.live ||
          (candidate.boundary == prompt.size() && !candidate.full_ready))
        continue;
      result = {candidate.boundary, candidate.ram ? "ram" : "disk",
                candidate.transfer_bytes};
      selected_entry = candidate.entry.value;
      break;
    }
  }
  // Never retry behind a writer. A mutation that overlapped inspection makes
  // the entire observation uncertain, so under-report instead of returning
  // an entry that may already have been retired.
  const auto finished = impl_->lookup_version.load(std::memory_order_acquire);
  if (version != finished)
    return unknown(finished, "mutation_overlap");
  result.index_version = view->version;
  result.snapshot_age_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - view->published)
          .count();
  return observed(result, selected_entry, "published", finished);
}

std::size_t ComponentTextCache::CachedPrefixTokens(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool common) const {
  const auto input = Inputs(prompt.size(), context);
  const std::lock_guard lock(impl_->mutex.native);
  const cache::PrefixQuery query{impl_->identity, prompt, input};
  return common ? impl_->index.CommonPrefixTokens(query)
                : impl_->index.CachedPrefixTokens(query);
}
ComponentTextCache::Lease::Lease(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ComponentTextCache::Lease::~Lease() {
  Invalidate();
}
ContinuationState& ComponentTextCache::Lease::state() const {
  return *impl_->state;
}
bool ComponentTextCache::Lease::cache_hit() const noexcept {
  return impl_->cached != 0;
}
std::size_t ComponentTextCache::Lease::cached_tokens() const noexcept {
  return impl_->cached;
}
std::size_t ComponentTextCache::Lease::restored_snapshot_bytes()
    const noexcept {
  return impl_->restored;
}
double ComponentTextCache::Lease::restore_ms() const noexcept {
  return impl_->restore_time;
}
bool ComponentTextCache::Lease::restored_from_disk() const noexcept {
  return impl_->disk_hit;
}
ContinuationLookup ComponentTextCache::Lease::lookup() const noexcept {
  return impl_->lookup_result;
}
bool ComponentTextCache::Lease::HasSnapshotFor(
    std::span<const ContinuationToken> tokens) const {
  const std::lock_guard lock(impl_->cache->mutex.native);
  return impl_->cache->index.HasCheckpoint(
      {impl_->cache->identity, tokens,
       cache::InputIdentity(tokens.size(), At(impl_->input, tokens.size()))},
      tokens.size());
}
bool ComponentTextCache::Lease::TryReserveSnapshot(
    std::size_t bytes, std::size_t boundary, bool preserve,
    SnapshotPurpose purpose, std::span<const ContinuationToken> tokens) {
  if (!impl_->cache->ram_enabled &&
      (!impl_->cache->queue || purpose == SnapshotPurpose::kHistory))
    return false;
  if (!boundary || boundary > impl_->prompt.size() ||
      (bytes > impl_->cache->budget &&
       (!impl_->cache->queue || purpose == SnapshotPurpose::kHistory)))
    return false;
  if (tokens.empty())
    tokens = std::span<const cache::Token>(impl_->prompt).first(boundary);
  // A capture within the disk step has nowhere to go when the RAM cache is
  // off. Refuse it here so callers do not log it as a failed capture.
  if (!impl_->cache->ram_enabled && purpose != SnapshotPurpose::kBranchPoint &&
      impl_->cache->WithinDiskStep(tokens, At(impl_->input, tokens.size())))
    return false;
  impl_->capture_boundary = boundary;
  impl_->capture_tokens.assign(tokens.begin(), tokens.end());
  impl_->preserve_source = preserve;
  impl_->purpose = purpose;
  return true;
}
void ComponentTextCache::Lease::SkipSnapshot(SnapshotEventReason reason,
                                             std::size_t bytes,
                                             std::size_t boundary) noexcept {
  try {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=snapshot_skipped reason=" +
                     std::to_string(static_cast<unsigned>(reason)) +
                     " tokens=" + std::to_string(boundary) +
                     " bytes=" + std::to_string(bytes));
  } catch (...) {
  }
}
void ComponentTextCache::Lease::PreparePrefill(std::size_t tokens) {
  if (impl_->cache->resources->prepare_prefill)
    impl_->cache->resources->prepare_prefill(tokens);
}

std::shared_ptr<CheckpointTransferBlob> ComponentTextCache::Export(
    std::span<const ContinuationToken> prompt, const TextPromptContext* context,
    bool reuse, std::size_t stable, std::string configuration) const {
  auto quota = std::make_shared<TransferQuota>(impl_->transfer_gate);
  cache::PrefixCandidate candidate;
  bool ready = false;
  {
    // A metadata lease, not acquisition/admission: never wait for an owner
    // that may be in GPU work. R2 lookup does not use either of these gates.
    std::unique_lock lock(impl_->mutex.native, std::try_to_lock);
    if (!lock.owns_lock())
      throw CheckpointTransferError(503, "cache_transfer_busy");
    auto plan =
        impl_->Plan(prompt, Inputs(prompt.size(), context), stable, reuse);
    auto found = std::ranges::find_if(plan.candidates, [&](const auto& c) {
      if (c.live)
        return false;  // Live state must first be captured by its owner.
      const bool full_ready =
          impl_->runner->LookupFullCheckpointReady() ||
          (c.checkpoint &&
           std::ranges::any_of(impl_->readiness, [&](const auto& r) {
             return r.decode_ready && r.checkpoint.lock() == c.checkpoint;
           }));
      return c.boundary < prompt.size() || full_ready;
    });
    if (found == plan.candidates.end())
      throw CheckpointTransferError(404, "cache_transfer_not_found");
    candidate = std::move(*found);
    ready =
        impl_->runner->LookupFullCheckpointReady() ||
        (candidate.checkpoint &&
         std::ranges::any_of(impl_->readiness, [&](const auto& r) {
           return r.decode_ready && r.checkpoint.lock() == candidate.checkpoint;
         }));
  }
  CheckpointTransferHeader header;
  header.configuration = std::move(configuration);
  header.identity = impl_->resources->adapter->CompatibilityIdentity();
  header.stable_tokens = std::min<std::uint64_t>(stable, candidate.boundary);
  header.decode_ready = ready;
  if (context)
    for (const auto& p : context->cache_prefixes)
      if (p.token_count <= candidate.boundary)
        header.image_prefixes.push_back({p.token_count, p.identity});
  std::shared_ptr<CheckpointTransferBlob> blob;
  std::vector<std::uint8_t> staging(kCheckpointTransferPieceBytes);
  if (candidate.durable) {
    auto& store = candidate.durable->Store();
    auto snapshot = store.Open(candidate.durable->Manifest().checkpoint);
    if (!snapshot || !candidate.disk_pin ||
        candidate.disk_pin.Epoch() != candidate.durable->Epoch())
      throw CheckpointTransferError(409, "cache_transfer_source_changed");
    header.manifest = snapshot->Manifest();
    header.manifest.compatibility = [&] {
      crypto::Sha256Hasher hash;
      hash.Update(header.identity);
      return hash.Finish();
    }();
    auto payloads = CheckpointTransferPayloads(snapshot->Manifest());
    std::vector<bool> private_file;
    for (const auto& c : snapshot->Manifest().components) {
      for (std::size_t i = 0; i < c.chunks.size(); ++i)
        private_file.push_back(false);
      if (c.tail)
        private_file.push_back(true);
      if (c.private_state)
        private_file.push_back(true);
    }
    blob = CheckpointTransferBlob::Export(
        std::move(header), [&](auto i, const auto& consume) {
          (void)store.ReadPayload(*snapshot, private_file.at(i),
                                  *payloads.at(i), staging,
                                  [&](auto, auto bytes) { consume(bytes); });
        });
  } else {
    auto checkpoint = candidate.checkpoint;
    if (!checkpoint || !checkpoint->IsValid())
      throw CheckpointTransferError(409, "cache_transfer_source_changed");
    auto& m = header.manifest;
    m.checkpoint = checkpoint->Id();
    m.lineage = checkpoint->Lineage();
    crypto::Sha256Hasher hash;
    hash.Update(header.identity);
    m.compatibility = hash.Finish();
    m.input = checkpoint->Input();
    m.tokens.assign(checkpoint->Tokens().begin(), checkpoint->Tokens().end());
    m.purpose = checkpoint->Purpose();
    std::vector<const cache::Payload*> payloads;
    std::uint64_t serial = 0;
    const auto append = [&](const cache::Payload& payload, std::size_t bytes) {
      payloads.push_back(&payload);
      return cache::DiskPayload{impl_->File(0x200000000ULL, ++serial), bytes,
                                0};
    };
    for (const auto& c : checkpoint->Components()) {
      cache::DiskComponent d{.descriptor = c.descriptor,
                             .position = c.position};
      for (const auto& chunk : c.chunks)
        d.chunks.push_back(
            append(chunk.Storage(),
                   (chunk.End() - chunk.First()) * c.descriptor.row_bytes));
      if (c.tail)
        d.tail = append(*c.tail,
                        (c.position.valid_rows % c.descriptor.rows_per_chunk) *
                            c.descriptor.row_bytes);
      if (c.private_state)
        d.private_state = append(*c.private_state, c.descriptor.state_bytes);
      m.components.push_back(std::move(d));
    }
    auto lengths = CheckpointTransferPayloads(m);
    blob = CheckpointTransferBlob::Export(
        std::move(header), [&](auto i, const auto& consume) {
          const auto& payload = *payloads.at(i);
          for (std::uint64_t offset = 0; offset < lengths[i]->bytes;) {
            auto piece = std::span(staging).first(std::min<std::uint64_t>(
                staging.size(), lengths[i]->bytes - offset));
            if (auto rows = payload.BorrowedStorage()) {
              // Never hold a row pin over network or filesystem I/O.
              // Reassignment may wait for this one bounded copy, not the whole
              // export.
              auto pin = rows->Pin();
              auto source = pin.Bytes().subspan(offset, piece.size());
              if (pin.Location()) {
                if (!impl_->resources->copy_observed_rows)
                  throw CheckpointTransferError(
                      409, "cache_transfer_source_changed");
                auto stream = impl_->Stream();
                Check(impl_->resources->copy_observed_rows(
                    source, std::as_writable_bytes(piece), *stream));
              } else {
                std::memcpy(piece.data(), source.data(), piece.size());
              }
            } else {
              auto owner = payload.Owner();
              if (!owner)
                throw CheckpointTransferError(409,
                                              "cache_transfer_source_changed");
              std::memcpy(
                  piece.data(),
                  static_cast<const std::uint8_t*>(owner.get()) + offset,
                  piece.size());
            }
            consume(piece);
            offset += piece.size();
          }
        });
  }
  blob->quota = std::move(quota);
  return blob;
}
std::shared_ptr<CheckpointTransferBlob> ComponentTextCache::Import(
    const CheckpointTransferReader& reader, std::uint64_t length,
    std::string_view configuration) {
  // Import admits a RAM checkpoint, not a durable disk publication. Off is a
  // policy even for zero-byte payloads; do not spool an unusable upload first.
  if (!impl_->ram_enabled)
    throw CheckpointTransferError(507, "cache_transfer_budget");
  auto quota = std::make_shared<TransferQuota>(impl_->transfer_gate);
  auto blob = CheckpointTransferBlob::Import(
      reader, length, impl_->resources->adapter->CompatibilityIdentity(),
      configuration);
  const auto& header = blob->Header();
  const auto& m = header.manifest;
  const auto descriptors = impl_->resources->adapter->Components();
  if (descriptors.size() != m.components.size())
    throw CheckpointTransferError(409, "cache_transfer_layout_mismatch");
  std::vector<cache::ComponentPosition> positions;
  for (std::size_t i = 0; i < descriptors.size(); ++i) {
    const auto& a = descriptors[i];
    const auto& b = m.components[i].descriptor;
    if (a.id != b.id || a.kind != b.kind ||
        a.layout_version != b.layout_version || a.row_bytes != b.row_bytes ||
        a.rows_per_chunk != b.rows_per_chunk || a.state_bytes != b.state_bytes)
      throw CheckpointTransferError(409, "cache_transfer_layout_mismatch");
    positions.push_back(m.components[i].position);
  }
  if (m.tokens.size() > impl_->runner->Descriptor().max_context)
    throw CheckpointTransferError(409, "cache_transfer_context_mismatch");
  if (blob->PayloadBytes() > impl_->budget)
    throw CheckpointTransferError(507, "cache_transfer_budget");
  cache::RetentionRequest retain{
      .tokens = m.tokens,
      .compatibility = impl_->identity,
      .input = m.input,
      .stable = header.stable_tokens,
      .new_payload_bytes = static_cast<std::size_t>(blob->PayloadBytes())};
  // Serializes admission/allocation/publication, never a slot restore/prefill.
  // The provider may grow/register bounded host backing. There is no execution
  // lease or live frontier replacement; serving can keep decoding.
  const std::lock_guard lock(impl_->mutex);
  // A failed index reservation must release the preceding attempt before
  // retention retries. Only the policy, not this observer, retains payloads.
  std::weak_ptr<const cache::Checkpoint> checkpoint;
  impl_->readiness.reserve(impl_->readiness.size() + 1);
  std::exception_ptr transfer_failure;
  const bool admitted = impl_->retention.Admit(retain, [&] {
    try {
      auto history = cache::ExecutionHistory::Cold(impl_->ledger, descriptors,
                                                   impl_->identity);
      std::vector<ComponentCaptureBuffer> buffers;
      cache::ResourceReservation buffer_metadata;
      std::size_t next = 0;
      auto captured = history.Capture(
          {m.tokens, cache::InputIdentity(m.tokens.size(), m.input), positions,
           m.purpose, 0},
          [&](const auto&) { return buffers.at(next++).finish(); },
          [&](std::span<const cache::PayloadRequest> requests) {
            buffer_metadata = impl_->ledger.Reserve(
                cache::ResourceCategory::kMetadata,
                requests.size() *
                    (sizeof(ComponentCaptureBuffer) + sizeof(std::size_t)));
            if (impl_->resources->can_capture &&
                !impl_->resources->can_capture(requests))
              throw cache::ResourceExhausted();
            buffers.resize(requests.size());
            std::vector<std::size_t> order(requests.size());
            std::iota(order.begin(), order.end(), 0);
            std::ranges::stable_sort(order, [&](auto a, auto b) {
              return requests[a].bytes > requests[b].bytes;
            });
            for (auto i : order)
              buffers[i] = impl_->resources->allocate(requests[i]);
            for (std::size_t i = 0; i < buffers.size(); ++i) {
              auto bytes = std::as_writable_bytes(
                  buffers[i].bytes.first(requests[i].bytes));
              blob->Load(i, {reinterpret_cast<std::uint8_t*>(bytes.data()),
                             bytes.size()});
            }
          });
      if (!impl_->index.IsResidentCoherent(*captured))
        throw CheckpointTransferError(409, "cache_transfer_layout_mismatch");
      checkpoint = captured;
      return captured;
    } catch (const CheckpointTransferError&) {
      transfer_failure = std::current_exception();
      throw;
    } catch (const std::invalid_argument&) {
      // Retention turns generic capture exceptions into refusal. Preserve the
      // packed model's semantic cause instead of reporting a false budget507;
      // the factory has not returned a publishable checkpoint at this point.
      if (PackedKvIngress(impl_->runner->Descriptor(), descriptors))
        transfer_failure = std::current_exception();
      throw;
    }
  });
  if (!admitted) {
    if (transfer_failure)
      std::rethrow_exception(transfer_failure);
    throw CheckpointTransferError(507, "cache_transfer_budget");
  }
  std::erase_if(impl_->readiness,
                [](const auto& r) { return r.checkpoint.expired(); });
  impl_->readiness.push_back({checkpoint, header.decode_ready});
  return blob;
}
std::size_t ComponentTextCache::Lease::CaptureBytes() const {
  return impl_->entry->history->NewPayloadBytes(
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution()));
}
std::size_t ComponentTextCache::Lease::CaptureBytes(
    std::size_t boundary) const {
  return impl_->entry->history->NewPayloadBytes(
      impl_->cache->resources->adapter->PrefillCheckpointPositions(
          impl_->slot.Execution(), boundary));
}
std::unique_ptr<TextRunnerSnapshot> ComponentTextCache::Lease::Capture() {
  auto& request = *impl_;
  auto& cache = *request.cache;
  auto positions =
      cache.resources->adapter->Positions(request.slot.Execution());
  const auto bytes = request.entry->history->NewPayloadBytes(positions);
  const auto retain = request.Retain(request.capture_tokens, bytes);
  const auto refused = [&] {
    Logger::Info("cache",
                 "schema=component-cache-v1 event=capture_refused "
                 "reason=byte_capacity tokens=" +
                     std::to_string(request.capture_tokens.size()));
  };
  const auto failed = [](const std::exception& error) {
    Logger::Info(
        "cache",
        std::string("schema=component-cache-v1 event=capture_failed reason=") +
            error.what());
  };
  if (!cache.ram_enabled || bytes > cache.budget) {
    try {
      return request.CaptureDurable();
    } catch (const cache::ResourceExhausted&) {
      refused();
      throw;
    } catch (const std::exception& error) {
      failed(error);
      throw;
    }
  }
  const auto reclaim = [&] {
    const std::lock_guard lock(cache.mutex);
    return cache.retention.Reclaim(retain);
  };
  std::shared_ptr<const cache::Checkpoint> checkpoint;
  std::size_t captured_capacity = 0;
  std::size_t borrowed_capacity = 0, copied_payload = 0;
  while (true) {
    try {
      captured_capacity = 0;
      borrowed_capacity = copied_payload = 0;
      cache::ResourceReservation buffer_metadata;
      std::vector<ComponentCaptureBuffer> buffers;
      std::vector<std::span<const std::byte>> borrowed;
      std::size_t next_buffer = 0;
      checkpoint = request.entry->history->Capture(
          {request.capture_tokens,
           request.input.Through(request.capture_tokens.size()), positions,
           request.purpose == SnapshotPurpose::kHistory
               ? cache::CheckpointPurpose::kGrid
           : request.purpose == SnapshotPurpose::kBranchPoint
               ? cache::CheckpointPurpose::kLearned
               : cache::CheckpointPurpose::kPrompt,
           0},
          [&](const cache::PayloadRequest&) {
            const auto i = next_buffer++;
            auto& buffer = buffers.at(i);
            auto storage = borrowed[i].empty()
                               ? buffer.finish()
                               : buffer.borrow(request.slot, borrowed[i]);
            captured_capacity += storage.Bytes();
            if (!borrowed[i].empty())
              borrowed_capacity += storage.Bytes();
            return storage;
          },
          [&](std::span<const cache::PayloadRequest> payloads) {
            buffer_metadata = cache.ledger.Reserve(
                cache::ResourceCategory::kMetadata,
                payloads.size() *
                    (sizeof(ComponentCaptureBuffer) +
                     sizeof(std::span<const std::byte>) + sizeof(std::size_t)));
            buffers.reserve(payloads.size());
            borrowed.reserve(payloads.size());
            // Small row/tail allocations must not split the remaining runs
            // that larger private states require. Keep equal-size requests
            // in their original order; transfers still follow component order.
            std::vector<std::size_t> allocation_order(payloads.size());
            std::iota(allocation_order.begin(), allocation_order.end(), 0);
            std::ranges::sort(allocation_order, [&](auto a, auto b) {
              return payloads[a].bytes != payloads[b].bytes
                         ? payloads[a].bytes > payloads[b].bytes
                         : a < b;
            });
            while (true) {
              try {
                if (cache.resources->can_capture &&
                    !cache.resources->can_capture(payloads))
                  throw cache::ResourceExhausted();
                buffers.resize(payloads.size());
                borrowed.resize(payloads.size());
                for (const auto i : allocation_order) {
                  const auto& payload = payloads[i];
                  buffers[i] = cache.resources->allocate(payload);
                  borrowed[i] =
                      !request.pin_capture && buffers[i].borrow &&
                              payload.category ==
                                  cache::ResourceCategory::kBackingAssigned
                          ? cache.resources->adapter->BorrowRowsView(
                                request.slot.Execution(), payload.component,
                                payload.first, payload.end)
                          : std::span<const std::byte>{};
                }
                break;
              } catch (const cache::ResourceExhausted&) {
                // No transfer has started. Release partial allocations before
                // eviction, but keep the plan's shared chunks alive. Rebuilding
                // the whole checkpoint can otherwise lose those predecessors
                // and allocate their rows again after their last owner leaves.
                borrowed.clear();
                buffers.clear();
                if (!reclaim())
                  throw;
              }
            }
            TransferBatch transfers(cache.resources->transfer_batch_size);
            for (std::size_t i = 0; i < payloads.size();) {
              if (!borrowed[i].empty()) {
                ++i;
                continue;
              }
              const auto& payload = payloads[i];
              auto end = payload.end;
              auto bytes = payload.bytes;
              auto next = i + 1;
              if (payload.category != cache::ResourceCategory::kPrivateState)
                while (buffers[i].allocation && next < payloads.size() &&
                       borrowed[next].empty() &&
                       buffers[next].allocation == buffers[i].allocation &&
                       payloads[next].component == payload.component &&
                       payloads[next].category !=
                           cache::ResourceCategory::kPrivateState &&
                       payloads[next].first == end &&
                       buffers[next].bytes.data() ==
                           buffers[i].bytes.data() + bytes) {
                  end = payloads[next].end;
                  bytes += payloads[next].bytes;
                  ++next;
                }
              const std::span<std::byte> buffer(buffers[i].bytes.data(), bytes);
              transfers.Add(
                  [&] { return cache.resources->stream(); },
                  [&](cache::Stream& stream) {
                    return payload.category ==
                                   cache::ResourceCategory::kPrivateState
                               ? cache.resources->adapter->CapturePrivate(
                                     request.slot.Execution(),
                                     payload.component, buffer, stream)
                               : cache.resources->adapter->CopyRowsOut(
                                     request.slot.Execution(),
                                     payload.component, payload.first, end,
                                     buffer, stream);
                  });
              copied_payload += bytes;
              i = next;
            }
            transfers.Finish();
          });
      break;
    } catch (const cache::ResourceExhausted&) {
      if (!reclaim()) {
        if (!request.pin_capture && cache.queue &&
            request.purpose != SnapshotPurpose::kHistory) {
          try {
            if (auto durable = request.CaptureDurable())
              return durable;
          } catch (const cache::ResourceExhausted&) {
            // Staging for disk did not fit either. Refuse below.
          } catch (const std::exception& error) {
            failed(error);
            throw;
          }
        }
        // RAM refused the copy; the disk fallback, when there is one, already
        // holds a checkpoint within its step of this boundary.
        refused();
        throw;
      }
    } catch (const std::exception& error) {
      failed(error);
      throw;
    }
  }
  ++request.metrics.captures;
  request.metrics.captured_checkpoint_ids.push_back(checkpoint->Id().value);
  request.metrics.captured_bytes += captured_capacity;
  request.metrics.borrowed_capacity_bytes += borrowed_capacity;
  request.metrics.copied_payload_bytes += copied_payload;
  {
    const auto ready = cache.runner->HasDecodeFrontier(*request.state);
    const std::lock_guard lock(cache.mutex);
    std::erase_if(cache.readiness,
                  [](const auto& r) { return r.checkpoint.expired(); });
    cache.readiness.push_back({checkpoint, ready});
  }
  auto snapshot = std::make_unique<ComponentSnapshot>();
  snapshot->checkpoint = std::move(checkpoint);
  snapshot->bytes = bytes;
  return snapshot;
}
std::size_t ComponentTextCache::Lease::PublishSnapshot(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot, bool preserve) {
  const auto* component =
      dynamic_cast<const ComponentSnapshot*>(snapshot.get());
  if (component && component->durable) {
    const auto started = Clock::now();
    impl_->metrics.disk_queued_bytes +=
        impl_->cache->Persist(component->durable);
    impl_->metrics.disk_enqueue_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
    return 0;
  }
  if (!component || !component->checkpoint)
    return 0;
  auto retain = impl_->Retain(tokens);
  retain.preserve_source = preserve;
  bool admitted;
  {
    const std::lock_guard lock(impl_->cache->mutex);
    admitted = impl_->cache->retention.Admit(
        retain, [&] { return component->checkpoint; });
  }
  if (!admitted)
    return 0;
  if (retain.purpose == cache::RetentionPurpose::kHistory) {
    if (impl_->edit_boundary_count == impl_->edit_boundaries.size())
      throw std::logic_error("too many edit boundaries in one turn");
    impl_->edit_boundaries[impl_->edit_boundary_count++] =
        component->checkpoint->Id();
  }
  try {
    const auto started = Clock::now();
    impl_->metrics.disk_queued_bytes +=
        impl_->cache->Persist(component->checkpoint);
    impl_->metrics.disk_enqueue_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - started)
            .count();
  } catch (const std::exception& error) {
    Logger::Info("cache", std::string("event=persistence_refused reason=") +
                              error.what());
  }
  return component->bytes;
}
std::size_t ComponentTextCache::Lease::Commit(
    std::vector<ContinuationToken> tokens,
    std::shared_ptr<const ContinuationSnapshot> snapshot,
    std::vector<ContinuationToken> live) {
  const auto bytes =
      snapshot ? PublishSnapshot(std::move(tokens), std::move(snapshot), true)
               : 0;
  if (live.empty()) {
    Invalidate();
    return bytes;
  }
  auto positions =
      impl_->cache->resources->adapter->Positions(impl_->slot.Execution());
  const auto location = impl_->slot.Location();
  const bool decode_ready =
      impl_->cache->runner->HasDecodeFrontier(*impl_->state);
  impl_->slot.Commit();
  {
    const std::lock_guard lock(impl_->cache->mutex);
    impl_->entry->decode_ready = decode_ready;
    impl_->entry->live_tokens = live;
    impl_->entry->live_input = At(impl_->input, impl_->prompt.size());
    impl_->entry->live = impl_->cache->index.Insert(cache::LiveFrontier{
        location, std::move(live), impl_->cache->identity,
        At(impl_->input, impl_->prompt.size()), std::move(positions)});
    impl_->entry->busy = false;
  }
  impl_->active = false;
  return bytes;
}
void ComponentTextCache::Lease::Invalidate() noexcept {
  if (!impl_ || !impl_->active)
    return;
  try {
    impl_->state->SetCancellationCheck({});
    impl_->slot = {};
    impl_->entry->history.reset();
    const std::lock_guard lock(impl_->cache->mutex);
    impl_->entry->branch.reset();
    impl_->entry->durable.reset();
    impl_->entry->busy = false;
  } catch (...) {
  }
  impl_->active = false;
}

std::vector<std::size_t> ComponentTextCache::Lease::PlannedCaptures(
    std::size_t stable) const {
  // Off with no disk cache keeps no checkpoint, so skip the planning.
  if (!impl_->cache->ram_enabled && !impl_->cache->queue)
    return {};
  const auto& adapter = *impl_->cache->resources->adapter;
  std::size_t private_bytes = 0;
  for (const auto& d : adapter.Components())
    if (d.kind == cache::ComponentKind::kPrivateState)
      private_bytes += d.state_bytes;
  const auto plan = cache::PlanCaptures(
      adapter, impl_->slot.Execution(),
      {impl_->prompt.size(),
       impl_->cached,
       stable,
       0,
       true,
       false,
       {},
       cache::GridLimit(impl_->cache->budget, private_bytes)});
  std::vector<std::size_t> boundaries;
  for (const auto& capture : plan)
    if (!capture.required &&
        !HasSnapshotFor(std::span<const cache::Token>(impl_->prompt)
                            .first(capture.boundary)))
      boundaries.push_back(capture.boundary);
  return boundaries;
}
std::vector<std::size_t> ServingCacheLease::PlannedCaptures(
    std::size_t stable) const {
  return component_->PlannedCaptures(stable);
}

ComponentCacheMetrics ComponentTextCache::Lease::Metrics() const {
  auto report = impl_->metrics;
  report.ledger = impl_->cache->ledger.Snapshot();
  if (impl_->cache->queue)
    report.persistence = impl_->cache->queue->Stats();
  for (const auto& entry : impl_->cache->entries) {
    const auto spill = entry->slot->Metrics();
    report.spill.idle_bytes += spill.idle_bytes;
    report.spill.foreground_bytes += spill.foreground_bytes;
    report.spill.idle_copy_ns += spill.idle_copy_ns;
    report.spill.foreground_copy_ns += spill.foreground_copy_ns;
    report.spill.stream_wait_ns += spill.stream_wait_ns;
    report.spill.residual_wait_ns += spill.residual_wait_ns;
    report.spill.reassignments += spill.reassignments;
    report.spill.idle_failures += spill.idle_failures;
  }
  return report;
}
std::optional<ComponentCacheMetrics> ServingCacheLease::Metrics() const {
  return component_ ? std::optional(component_->Metrics()) : std::nullopt;
}

#define FORWARD_RESULT(type, name, qualifiers, args, call)         \
  type ServingCacheLease::name args qualifiers {                   \
    return component_ ? component_->name call : legacy_.name call; \
  }
FORWARD_RESULT(ContinuationState&, state, const, (), ())
FORWARD_RESULT(bool, cache_hit, const noexcept, (), ())
FORWARD_RESULT(std::size_t, cached_tokens, const noexcept, (), ())
FORWARD_RESULT(std::size_t, restored_snapshot_bytes, const noexcept, (), ())
FORWARD_RESULT(double, restore_ms, const noexcept, (), ())
FORWARD_RESULT(bool, restored_from_disk, const noexcept, (), ())
FORWARD_RESULT(ContinuationLookup, lookup, const noexcept, (), ())
FORWARD_RESULT(bool, HasSnapshotFor, const,
               (std::span<const ContinuationToken> tokens), (tokens))
FORWARD_RESULT(bool, TryReserveSnapshot, ,
               (std::size_t bytes, std::size_t boundary, bool preserve,
                SnapshotPurpose purpose,
                std::span<const ContinuationToken> tokens),
               (bytes, boundary, preserve, purpose, tokens))
FORWARD_RESULT(void, SkipSnapshot, noexcept,
               (SnapshotEventReason reason, std::size_t bytes,
                std::size_t boundary),
               (reason, bytes, boundary))
FORWARD_RESULT(std::size_t, PublishSnapshot, ,
               (std::vector<ContinuationToken> tokens,
                std::shared_ptr<const ContinuationSnapshot> snapshot,
                bool preserve),
               (std::move(tokens), std::move(snapshot), preserve))
FORWARD_RESULT(std::size_t, Commit, ,
               (std::vector<ContinuationToken> tokens,
                std::shared_ptr<const ContinuationSnapshot> snapshot,
                std::vector<ContinuationToken> live),
               (std::move(tokens), std::move(snapshot), std::move(live)))
FORWARD_RESULT(void, Invalidate, noexcept, (), ())
#undef FORWARD_RESULT
void ServingCacheLease::AdoptRestoredPrefix(std::size_t tokens,
                                            std::size_t bytes, double ms) {
  if (component_)
    throw std::logic_error("component restore is owned by the component cache");
  legacy_.AdoptRestoredPrefix(tokens, bytes, ms);
}
void ServingCacheLease::PreparePrefill(std::size_t tokens) {
  if (component_)
    component_->PreparePrefill(tokens);
}

std::size_t ServingCacheLease::CaptureBytes() const {
  return component_->CaptureBytes();
}
std::size_t ServingCacheLease::CaptureBytes(std::size_t boundary) const {
  return component_->CaptureBytes(boundary);
}
std::unique_ptr<TextRunnerSnapshot> ServingCacheLease::Capture() {
  return component_->Capture();
}
}  // namespace gufo::server
