#ifndef GUFO_SERVER_GENERATION_METRICS_HPP_
#define GUFO_SERVER_GENERATION_METRICS_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string_view>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/json.hpp"

namespace gufo::server {

// Process-wide `/metrics` values. Token counters advance per prefill chunk and
// per generated token, so scrapes see work while requests are still running.
namespace detail {
inline std::atomic<std::uint64_t>& DeviceLostTotal() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalGenTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
// Request totals below are added once per request when it finishes.
inline std::atomic<std::uint64_t>& TotalCachedPromptTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<double>& TotalPromptSeconds() {
  static std::atomic<double> seconds{0.0};
  return seconds;
}
inline std::atomic<double>& TotalGenSeconds() {
  static std::atomic<double> seconds{0.0};
  return seconds;
}
inline std::atomic<std::uint64_t>& MaxSequenceTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftRounds() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalDraftAcceptedTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalLookupTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<std::uint64_t>& TotalLookupAcceptedTokens() {
  static std::atomic<std::uint64_t> count{0};
  return count;
}
inline std::atomic<double>& LastPromptSpeed() {
  static std::atomic<double> val{0.0};
  return val;
}
inline std::atomic<double>& LastGenSpeed() {
  static std::atomic<double> val{0.0};
  return val;
}
// Schedulers add their own deltas, so a draining scheduler and its
// replacement can both contribute during a model reload.
inline std::atomic<std::int64_t>& RequestsProcessing() {
  static std::atomic<std::int64_t> count{0};
  return count;
}
inline std::atomic<std::int64_t>& RequestsDeferred() {
  static std::atomic<std::int64_t> count{0};
  return count;
}

// `usage.gufo.cache_miss_reason` values from both text caches; any other
// reason counts as `unknown`. A request that never reached a cache lookup, such
// as one cancelled while queued, reports no reason and counts as neither.
inline constexpr std::array<std::string_view, 6> kCacheMissReasonNames{
    "disabled",       "no_checkpoint", "no_compatible_boundary",
    "prefix_changed", "input_changed", "unknown"};
inline auto& CacheHitCounts() {
  static std::array<std::atomic<std::uint64_t>, kCacheHitTierNames.size()>
      counts{};
  return counts;
}
inline auto& CacheMissCounts() {
  static std::array<std::atomic<std::uint64_t>, kCacheMissReasonNames.size()>
      counts{};
  return counts;
}

/// Fixed-bucket speed histogram per context-depth band. Buckets hold
/// per-band counts; the exposition makes them cumulative. Recording is a few
/// relaxed atomic adds, so a scrape may see one request's sample partly added.
template<std::size_t Bounds>
struct SpeedHistogram {
  std::array<double, Bounds> bounds;
  std::array<std::array<std::atomic<std::uint64_t>, Bounds + 1>, 6> buckets{};
  std::array<std::atomic<std::uint64_t>, 6> counts{};
  std::array<std::atomic<double>, 6> sums{};
};
// Upper bounds of the context-depth bands, in prompt tokens; the last band is
// open-ended. Labels name each band.
inline constexpr std::array<std::size_t, 5> kContextDepthBounds{
    1024, 4096, 16384, 65536, 262144};
inline constexpr std::array<std::string_view, 6> kContextDepthNames{
    "0_1k", "1k_4k", "4k_16k", "16k_64k", "64k_256k", "256k_plus"};
inline auto& PrefillSpeedHistogram() {
  static SpeedHistogram<9> histogram{
      .bounds = {50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000}};
  return histogram;
}
inline auto& DecodeSpeedHistogram() {
  static SpeedHistogram<11> histogram{
      .bounds = {1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200}};
  return histogram;
}
// Shorter prefills mostly measure launch and scheduling latency.
inline constexpr std::size_t kMinPrefillSpeedTokens = 64;
}  // namespace detail

inline std::size_t ContextDepthBand(std::size_t prompt_tokens) {
  return static_cast<std::size_t>(
      std::ranges::lower_bound(detail::kContextDepthBounds, prompt_tokens) -
      detail::kContextDepthBounds.begin());
}

template<std::size_t Bounds>
void RecordSpeed(detail::SpeedHistogram<Bounds>& histogram, std::size_t band,
                 double tokens_per_second) {
  const auto bucket = static_cast<std::size_t>(
      std::ranges::lower_bound(histogram.bounds, tokens_per_second) -
      histogram.bounds.begin());
  histogram.buckets[band][bucket].fetch_add(1, std::memory_order_relaxed);
  histogram.counts[band].fetch_add(1, std::memory_order_relaxed);
  auto& sum = histogram.sums[band];
  double current = sum.load(std::memory_order_relaxed);
  while (!sum.compare_exchange_weak(current, current + tokens_per_second,
                                    std::memory_order_relaxed)) {
  }
}

/// The tier a finished request's cache hit is counted under. Backends without
/// component metrics are classified from their disk and restore flags.
inline CacheHitTier ResultCacheHitTier(
    const TextGenerationBackend::Result& result) {
  if (!result.cache_hit)
    return CacheHitTier::kNone;
  if (result.component_cache &&
      result.component_cache->hit_tier != CacheHitTier::kNone)
    return result.component_cache->hit_tier;
  return result.cache_disk_hit        ? CacheHitTier::kDisk
         : result.cache_restore_bytes ? CacheHitTier::kRam
                                      : CacheHitTier::kLive;
}

inline std::size_t CacheMissReasonIndex(std::string_view reason) {
  const auto found = std::ranges::find(detail::kCacheMissReasonNames, reason);
  return found == detail::kCacheMissReasonNames.end()
             ? detail::kCacheMissReasonNames.size() - 1
             : static_cast<std::size_t>(found -
                                        detail::kCacheMissReasonNames.begin());
}

inline double PrefillTokensPerSecond(
    const TextGenerationBackend::Result& result) {
  return result.prefill_ms > 0.0 ? static_cast<double>(result.prefill_tokens) *
                                       1000.0 / result.prefill_ms
                                 : 0.0;
}

/// Adds one finished request to the request totals and speed gauges. The
/// scheduler calls this when a request ends, whether or not its client reads
/// the result; failed requests are not added.
inline void RecordRequestMetrics(const TextGenerationBackend::Result& result) {
  const auto add_seconds = [](std::atomic<double>& total, double ms) {
    double current = total.load(std::memory_order_relaxed);
    while (!total.compare_exchange_weak(current, current + ms / 1000.0,
                                        std::memory_order_relaxed)) {
    }
  };
  detail::TotalCachedPromptTokens().fetch_add(
      std::min(result.cached_prompt_tokens, result.prompt_tokens),
      std::memory_order_relaxed);
  add_seconds(detail::TotalPromptSeconds(), result.prefill_ms);
  add_seconds(detail::TotalGenSeconds(), result.decode_ms);
  const std::uint64_t sequence_tokens =
      result.prompt_tokens + result.completion_tokens;
  std::uint64_t max_tokens =
      detail::MaxSequenceTokens().load(std::memory_order_relaxed);
  while (max_tokens < sequence_tokens &&
         !detail::MaxSequenceTokens().compare_exchange_weak(
             max_tokens, sequence_tokens, std::memory_order_relaxed)) {
  }
  detail::TotalDraftRounds().fetch_add(result.draft_rounds,
                                       std::memory_order_relaxed);
  detail::TotalDraftTokens().fetch_add(result.draft_tokens,
                                       std::memory_order_relaxed);
  detail::TotalDraftAcceptedTokens().fetch_add(result.draft_accepted_tokens,
                                               std::memory_order_relaxed);
  detail::TotalLookupTokens().fetch_add(result.lookup_tokens,
                                        std::memory_order_relaxed);
  detail::TotalLookupAcceptedTokens().fetch_add(result.lookup_accepted_tokens,
                                                std::memory_order_relaxed);
  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double tok_per_sec =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  if (prompt_per_second > 0.0) {
    detail::LastPromptSpeed().store(prompt_per_second,
                                    std::memory_order_relaxed);
  }
  if (tok_per_sec > 0.0) {
    detail::LastGenSpeed().store(tok_per_sec, std::memory_order_relaxed);
  }
  if (const auto tier = ResultCacheHitTier(result); tier != CacheHitTier::kNone)
    detail::CacheHitCounts()[static_cast<std::size_t>(tier)].fetch_add(
        1, std::memory_order_relaxed);
  else if (!result.cache_miss_reason.empty())
    detail::CacheMissCounts()[CacheMissReasonIndex(result.cache_miss_reason)]
        .fetch_add(1, std::memory_order_relaxed);
  const std::size_t band = ContextDepthBand(result.prompt_tokens);
  if (prompt_per_second > 0.0 &&
      result.prefill_tokens >= detail::kMinPrefillSpeedTokens)
    RecordSpeed(detail::PrefillSpeedHistogram(), band, prompt_per_second);
  if (tok_per_sec > 0.0)
    RecordSpeed(detail::DecodeSpeedHistogram(), band, tok_per_sec);
}

template<std::size_t Bounds>
void WriteSpeedHistogram(std::ostream& out, std::string_view name,
                         std::string_view help,
                         const detail::SpeedHistogram<Bounds>& histogram) {
  out << "# HELP " << name << ' ' << help << "\n# TYPE " << name
      << " histogram\n";
  for (std::size_t band = 0; band < detail::kContextDepthNames.size(); ++band) {
    const auto context = detail::kContextDepthNames[band];
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i <= Bounds; ++i) {
      cumulative += histogram.buckets[band][i].load(std::memory_order_relaxed);
      out << name << "_bucket{context=\"" << context << "\",le=\"";
      if (i < Bounds)
        out << histogram.bounds[i];
      else
        out << "+Inf";
      out << "\"} " << cumulative << '\n';
    }
    out << name << "_sum{context=\"" << context << "\"} "
        << histogram.sums[band].load(std::memory_order_relaxed) << '\n'
        << name << "_count{context=\"" << context << "\"} "
        << histogram.counts[band].load(std::memory_order_relaxed) << '\n';
  }
}

/// Cache outcome counters and speed histograms for `/metrics`, after the
/// llama.cpp-compatible block.
inline void WriteCacheAndSpeedMetrics(std::ostream& out) {
  out << "# HELP gufo_cache_hits_total Finished text requests that reused a "
         "cached prefix, by source tier\n"
      << "# TYPE gufo_cache_hits_total counter\n";
  for (std::size_t i = 1; i < kCacheHitTierNames.size(); ++i)
    out << "gufo_cache_hits_total{tier=\"" << kCacheHitTierNames[i] << "\"} "
        << detail::CacheHitCounts()[i].load(std::memory_order_relaxed) << '\n';
  out << "# HELP gufo_cache_misses_total Finished text requests that reused "
         "no cached prefix, by usage.gufo.cache_miss_reason\n"
      << "# TYPE gufo_cache_misses_total counter\n";
  for (std::size_t i = 0; i < detail::kCacheMissReasonNames.size(); ++i)
    out << "gufo_cache_misses_total{reason=\""
        << detail::kCacheMissReasonNames[i] << "\"} "
        << detail::CacheMissCounts()[i].load(std::memory_order_relaxed) << '\n';
  out << "# HELP gufo_cache_acquisitions_total Component cache acquisitions, "
         "including later cancelled ones, by reuse reason\n"
      << "# TYPE gufo_cache_acquisitions_total counter\n";
  for (std::size_t i = 1; i < kCacheReuseReasonNames.size(); ++i)
    out << "gufo_cache_acquisitions_total{reason=\""
        << kCacheReuseReasonNames[i] << "\"} "
        << CacheReuseReasonCounts()[i].load(std::memory_order_relaxed) << '\n';
  WriteSpeedHistogram(out, "gufo_prefill_tokens_per_second",
                      "Per-request prefill speed of at least 64 prefilled "
                      "tokens, by prompt context depth",
                      detail::PrefillSpeedHistogram());
  WriteSpeedHistogram(out, "gufo_decode_tokens_per_second",
                      "Per-request decode speed, by prompt context depth",
                      detail::DecodeSpeedHistogram());
}

inline std::string GenerationLogDetails(
    const TextGenerationBackend::Result& result) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1)
      << "prompt_tokens=" << result.prompt_tokens
      << " prefill_tokens=" << result.prefill_tokens
      << " generated_tokens=" << result.completion_tokens << " finish="
      << (result.cancelled ? "cancelled"
          : result.finish_reason == TextGenerationBackend::FinishReason::kLength
              ? "length"
          : result.finish_reason ==
                  TextGenerationBackend::FinishReason::kStopSequence
              ? "stop_sequence"
              : "stop")
      << " cache="
      << (result.cache_disk_hit ? "disk"
          : result.cache_hit    ? "memory"
                                : "miss")
      << " cached_tokens=" << result.cached_prompt_tokens
      << " cache_restore_ms=" << result.cache_restore_ms
      << " queue_depth=" << result.queue_depth_at_submit
      << " resident_at_admission=" << result.resident_requests_at_admission
      << " queue_ms=" << result.queue_ms
      << " shared_prefix_wait_ms=" << result.shared_prefix_wait_ms
      << " ttft_ms=" << result.ttft_ms
      << " prefill_tps=" << PrefillTokensPerSecond(result) << " decode_tps="
      << (result.decode_ms > 0
              ? 1000.0 * result.completion_tokens / result.decode_ms
              : 0.0)
      << " batch_width=" << result.physical_execution_width
      << " plan=" << result.execution_plan
      << " draft_accepted=" << result.draft_accepted_tokens
      << " draft_proposed=" << result.draft_tokens
      << " draft_rounds=" << result.draft_rounds;
  if (result.component_cache)
    out << " cache_schema=component-cache-v1 cache_reuse="
        << result.component_cache->reuse
        << " cache_captures=" << result.component_cache->captures
        << " cache_captured_capacity_bytes="
        << result.component_cache->captured_bytes
        << " cache_global_ledger_bytes="
        << result.component_cache->ledger.total_bytes;
  if (result.draft_tokens > 0)
    out << " acceptance_pct="
        << 100.0 * result.draft_accepted_tokens / result.draft_tokens;
  if (result.lookup_tokens > 0)
    out << " lookup_accepted=" << result.lookup_accepted_tokens
        << " lookup_proposed=" << result.lookup_tokens;
  if (result.ngram_rows > 0)
    out << " ngram_rows=" << result.ngram_rows
        << " ngram_dedup_rows=" << result.ngram_dedup_rows
        << " ngram_cache_hits=" << result.ngram_cache_hits
        << " ngram_disk_reads=" << result.ngram_disk_reads;
  if (result.cache_snapshot_bytes > 0)
    out << " cache_snapshot_bytes=" << result.cache_snapshot_bytes;
  if (result.cache_disk_queued_bytes > 0)
    out << " cache_disk_queued_bytes=" << result.cache_disk_queued_bytes;
  if (!result.cache_hit && !result.cache_miss_reason.empty())
    out << " cache_miss_reason=" << result.cache_miss_reason
        << " common_prefix_tokens=" << result.cache_common_prefix_tokens
        << " nearest_checkpoint_tokens=" << result.cache_checkpoint_tokens;
  return out.str();
}

/// Timed prefill counts work actually executed; usage counts the full prompt.
inline json::Value GenerationTimings(
    const TextGenerationBackend::Result& result) {
  json::Value timings = json::Value::object();
  timings["prompt_n"] = result.prefill_tokens;
  timings["prompt_ms"] = result.prefill_ms;
  timings["prompt_per_token_ms"] =
      result.prefill_tokens > 0
          ? result.prefill_ms / static_cast<double>(result.prefill_tokens)
          : 0.0;
  timings["prompt_per_second"] = PrefillTokensPerSecond(result);
  timings["predicted_n"] = result.completion_tokens;
  timings["predicted_ms"] = result.decode_ms;
  timings["predicted_per_token_ms"] =
      result.completion_tokens > 0
          ? result.decode_ms / static_cast<double>(result.completion_tokens)
          : 0.0;
  timings["predicted_per_second"] =
      result.decode_ms > 0.0 ? static_cast<double>(result.completion_tokens) *
                                   1000.0 / result.decode_ms
                             : 0.0;
  timings["cache_n"] = result.cached_prompt_tokens;
  timings["cache_restore_ms"] = result.cache_restore_ms;
  timings["cache_snapshot_ms"] = result.cache_snapshot_ms;
  timings["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  timings["draft_rounds"] = result.draft_rounds;
  timings["draft_n"] = result.draft_tokens;
  timings["draft_n_accepted"] = result.draft_accepted_tokens;
  if (result.component_cache)
    timings["component_cache"] = ComponentCacheJson(*result.component_cache);
  if (result.lookup_tokens > 0) {
    timings["lookup_n"] = result.lookup_tokens;
    timings["lookup_n_accepted"] = result.lookup_accepted_tokens;
  }
  return timings;
}

/// llama-server `prompt_progress` object for streamed `return_progress`.
inline json::Value PromptProgressJson(
    const TextGenerationBackend::PromptProgress& progress) {
  json::Value value = json::Value::object();
  value["total"] = progress.total;
  value["cache"] = progress.cache;
  value["processed"] = progress.processed;
  value["time_ms"] = progress.time_ms;
  return value;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_GENERATION_METRICS_HPP_
