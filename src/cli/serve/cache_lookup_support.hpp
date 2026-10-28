#ifndef GUFO_SERVER_CACHE_LOOKUP_SUPPORT_HPP_
#define GUFO_SERVER_CACHE_LOOKUP_SUPPORT_HPP_

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <random>
#include <set>
#include <sstream>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/core/crypto/sha256.hpp"

namespace gufo::server {
// Router epochs are deliberately stronger than the on-disk ABI. Computed once
// at pool creation, not during lookup. Binary hashing also distinguishes dirty
// development builds and builds whose revision was not supplied by packaging.
// This is a local configuration identity, never a cross-host transfer promise.
inline std::string CacheLookupEpoch(
    const TextRunnerDescriptor& descriptor,
    std::span<const std::uint8_t> identity, std::size_t slots,
    const std::optional<TextRunnerDiskCacheOptions>& disk,
    const TextRunnerRamCacheOptions& ram, std::string_view context = {}) {
  crypto::Sha256Hasher hash;
  const auto add = [&](std::string_view value) {
    const auto size = static_cast<std::uint64_t>(value.size());
    for (unsigned shift = 0; shift < 64; shift += 8) {
      const auto byte = static_cast<std::uint8_t>(size >> shift);
      hash.Update({&byte, 1});
    }
    hash.Update(
        {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()});
  };
  add("gufo-local-cache-lookup-v1");
#ifdef GUFO_REVISION
  add(GUFO_REVISION);
#else
  add("revision-unrecorded");
#endif
  static const auto executable = crypto::Sha256FileHex("/proc/self/exe");
  add(executable);
  add(descriptor.model_id);
  add(descriptor.state_abi);
  add(context);
  // Some RAM-only runners have no artifact fingerprint. Give each loaded
  // lifetime a distinct epoch instead of equating different weights that share
  // a human model ID/ABI. Fingerprinted models retain deterministic epochs.
  if (!descriptor.persistence) {
    static const auto process_nonce = [] {
      std::random_device random;
      return std::to_string(random()) + ":" + std::to_string(random()) + ":" +
             std::to_string(random()) + ":" + std::to_string(random());
    }();
    static std::atomic<std::uint64_t> sequence{};
    add(process_nonce);
    add(std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
  }
  add(crypto::Sha256Hex(identity));
  if (descriptor.persistence)
    add(crypto::Sha256Hex(descriptor.persistence->compatibility_identity));
  std::ostringstream flags;
  flags << descriptor.max_context << ':' << slots << ':'
        << (descriptor.persistence ? descriptor.persistence->payload_version
                                   : 0)
        << ':' << ram.enabled << ':' << ram.capacity_bytes << ':'
        << ram.pinned_prefix.enabled() << ':' << ram.pinned_prefix.lazy << ':'
        << ram.pinned_prefix.capacity_bytes;
  for (const auto& file : ram.pinned_prefix.files)
    flags << ':' << file.string();
  flags << ':' << ram.pinned_prefix.directory.string();
  if (disk)
    flags << ':' << disk->directory.string() << ':' << disk->capacity_bytes
          << ':' << disk->staging_capacity_bytes << ':'
          << disk->shared_prefix_min_tokens << ':'
          << disk->shared_prefix_max_boundaries << ':'
          << disk->min_checkpoint_step_tokens;
  add(flags.str());
  std::set<std::string> environment;
  for (auto** p = environ; p && *p; ++p) {
    const std::string_view entry(*p);
    if (entry.starts_with("GUFO_") || entry.starts_with("HIP_") ||
        entry.starts_with("HSA_") || entry.starts_with("ROCR_") ||
        entry.starts_with("ROCBLAS_") || entry.starts_with("AMD_"))
      environment.emplace(entry);
  }
  for (const auto& entry : environment)
    add(entry);
  return hash.FinishHex();
}

// Cheap derivation from the startup pool epoch. Request contents are absent.
inline std::string CacheLookupConfiguredEpoch(
    std::string_view base, const ReasoningOptions& reasoning,
    std::size_t decode_active_prefill_tokens, std::string_view scheduler_flags,
    std::string_view build_info) {
  std::ostringstream options;
  const auto boolean = [&](const std::optional<bool>& value) {
    options << (value ? (*value ? 1 : 0) : -1) << ':';
  };
  options << base << ":prefill-chunk=4096:" << decode_active_prefill_tokens
          << ':';
  boolean(reasoning.enabled);
  boolean(reasoning.preserve_thinking);
  options << (reasoning.effort ? static_cast<int>(*reasoning.effort) : -1)
          << ':' << reasoning.budget_tokens.value_or(0) << ':'
          << reasoning.budget_tokens.has_value() << ':' << scheduler_flags
          << ':' << build_info;
  const auto text = options.str();
  return crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

// Bounded recent host-wall observations, excluding scheduler queueing. Prefill
// excludes checkpoint capture; restore uses the existing acquisition interval,
// which can include reconciliation and lazy-pin warming. Prefer an exact
// observed shape, then a measured EWMA (alpha=1/4). Restore uses bytes/ms
// independently for RAM/disk; providers without byte metadata fall back to
// measured token/ms. Prefill uses token/ms. These are estimates, not latency
// guarantees: depth, chunk geometry, concurrent load and I/O can change costs.
class CacheLookupTimings {
public:
  void Restore(std::string_view tier, std::size_t tokens, std::size_t bytes,
               double ms) {
    if (!tokens || !std::isfinite(ms) || ms <= 0 ||
        (tier != "ram" && tier != "disk"))
      return;
    const std::lock_guard lock(mutex_);
    // Only literals are retained; callers may supply a temporary string.
    tier = tier == "ram" ? "ram" : "disk";
    restores_[restore_next_++ % restores_.size()] = {tokens, bytes, ms, tier};
    auto& rate = tier == "ram" ? ram_rate_ : disk_rate_;
    Update(rate.tokens, tokens / ms);
    if (bytes)
      Update(rate.bytes, bytes / ms);
  }
  void Prefill(std::size_t tokens, double ms) {
    if (!tokens || !std::isfinite(ms) || ms <= 0)
      return;
    const std::lock_guard lock(mutex_);
    prefills_[prefill_next_++ % prefills_.size()] = {tokens, 0, ms, {}};
    Update(prefill_rate_, tokens / ms);
  }
  void Estimate(TextGenerationBackend::CacheLookupResult& result,
                std::size_t restore_bytes = 0) const {
    const auto remaining = result.total_tokens - result.matched_tokens;
    if (!remaining)
      result.prefill_estimate_ms = 0;
    if (!result.matched_tokens || result.tier == "live")
      result.restore_estimate_ms = 0;
    const std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock())
      return;
    for (std::size_t i = 0; i < std::min(restore_next_, restores_.size());
         ++i) {
      const auto& sample =
          restores_[(restore_next_ - 1 - i) % restores_.size()];
      if (sample.tokens == result.matched_tokens &&
          sample.tier == result.tier &&
          (!restore_bytes || sample.bytes == restore_bytes)) {
        result.restore_estimate_ms = sample.ms;
        break;
      }
    }
    if (!result.restore_estimate_ms && result.matched_tokens &&
        (result.tier == "ram" || result.tier == "disk")) {
      const auto& rate = result.tier == "ram" ? ram_rate_ : disk_rate_;
      if (restore_bytes && rate.bytes)
        result.restore_estimate_ms = restore_bytes / *rate.bytes;
      else if (rate.tokens)
        result.restore_estimate_ms = result.matched_tokens / *rate.tokens;
    }
    for (std::size_t i = 0; i < std::min(prefill_next_, prefills_.size());
         ++i) {
      const auto& sample =
          prefills_[(prefill_next_ - 1 - i) % prefills_.size()];
      if (sample.tokens == remaining) {
        result.prefill_estimate_ms = sample.ms;
        break;
      }
    }
    if (!result.prefill_estimate_ms && prefill_rate_)
      result.prefill_estimate_ms = remaining / *prefill_rate_;
  }

private:
  static void Update(std::optional<double>& rate, double sample) {
    if (std::isfinite(sample) && sample > 0)
      rate = rate ? 0.75 * *rate + 0.25 * sample : sample;
  }
  struct Rate {
    std::optional<double> bytes, tokens;
  };
  Rate ram_rate_, disk_rate_;
  std::optional<double> prefill_rate_;
  struct Sample {
    std::size_t tokens{}, bytes{};
    double ms{};
    std::string_view tier;
  };
  mutable std::mutex mutex_;
  std::array<Sample, 16> restores_{}, prefills_{};
  std::size_t restore_next_{}, prefill_next_{};
};
}  // namespace gufo::server
#endif
