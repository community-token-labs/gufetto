#ifndef GUFO_SERVER_DIAGNOSTIC_MEMORY_HPP_
#define GUFO_SERVER_DIAGNOSTIC_MEMORY_HPP_

#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/cli/serve/logging.hpp"
#include "src/core/startup_admission.hpp"

namespace gufo::server {

// E is immutable after startup sealing. OFF leaves arithmetic/messages
// unchanged.
inline void RequireDiagnosticMemoryFits(std::size_t bytes,
                                        std::size_t ceiling) {
  if (bytes > ceiling)
    throw std::invalid_argument(
        "--full-logits-api needs " + std::to_string(bytes) +
        " bytes; snapshot ceiling is " + std::to_string(ceiling) +
        " bytes; lower --context or --sessions, free memory, or "
        "restart without --full-logits-api");
}

// One startup-only text diagnostic activation. E is persistent + maximum phase,
// including sizeof(DiagnosticMemoryClaim). It never decreases, even after
// failed loading, detach, send, teardown or uncertain cleanup. This is NOT
// proof of resident memory or a replacement for native drain/unsafe-resource
// retention.
class DiagnosticMemoryClaim {
public:
  struct Stats {
    std::size_t bytes{};
    std::uint64_t unknown_outcomes{};
  };
  static DiagnosticMemoryClaim& Process() {
    static DiagnosticMemoryClaim claim;
    return claim;
  }
  // The caller supplies a fresh observation before entering the allocator.
  // Capacity is after OS/dispatch reserves, not reconstructed resident credit.
  bool Activate(std::size_t bytes, core::ExecutionCapacityObservation capacity,
                std::string* error) {
    const std::lock_guard lock(mutex_);
    const char* refusal = nullptr;
    if (stats_.bytes)
      refusal =
          "full-logits diagnostics can be activated only once per process; "
          "restart Gufo to load another diagnostic Model";
    else if (sealed_)
      refusal =
          "full-logits diagnostics must be activated before cache admission; "
          "restart Gufo with --full-logits-api";
    else if (bytes < sizeof(DiagnosticMemoryClaim))
      refusal = "full-logits workspace omits its static claim record";
    else if (bytes > capacity.Capacity())
      refusal =
          "full-logits diagnostic workspace exceeds available startup memory";
    if (refusal) {
      if (error)
        *error = refusal;
      Logger::Error("memory", refusal);
      return false;
    }
    Logger::Info(
        "memory",
        "event=full_logits_static_claim bytes=" + std::to_string(bytes) +
            " retention=process_lifetime; the model stays loaded "
            "until the server restarts; model switching/detach doesn't free "
            "it");
    stats_.bytes = bytes;
    return true;
  }
  // First cache admission freezes the startup choice BEFORE sampling or growth.
  // There is intentionally no unseal/release/replacement API.
  std::size_t SealStartup() {
    const std::lock_guard lock(mutex_);
    sealed_ = true;
    return stats_.bytes;
  }
  [[nodiscard]] Stats Snapshot() const {
    const std::lock_guard lock(mutex_);
    return stats_;
  }
  [[nodiscard]] std::size_t PendingBytes(std::size_t ordinary) const {
    return core::AddMemoryBytes(ordinary, Snapshot().bytes);
  }
  void ReportUnknown(std::string_view reason) noexcept {
    {
      const std::lock_guard lock(mutex_);
      sealed_ = true;
      if (stats_.unknown_outcomes != std::numeric_limits<std::uint64_t>::max())
        ++stats_.unknown_outcomes;
    }
    try {
      Logger::Error("memory", "event=full_logits_memory_unknown claim_bytes=" +
                                  std::to_string(Snapshot().bytes) +
                                  " reason=" + std::string(reason));
    } catch (...) {
      std::fputs("full-logits memory outcome unknown; static claim retained\n",
                 stderr);
    }
  }

private:
  mutable std::mutex mutex_;
  Stats stats_;
  bool sealed_{};
};

}  // namespace gufo::server
#endif
