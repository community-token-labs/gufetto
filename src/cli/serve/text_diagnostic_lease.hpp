#ifndef GUFO_SERVER_TEXT_DIAGNOSTIC_LEASE_HPP_
#define GUFO_SERVER_TEXT_DIAGNOSTIC_LEASE_HPP_

#include <cstddef>

#include "src/cache/slot.hpp"

namespace gufo::server {
class TextRunnerState;

// Internal cold/no-publish/no-retain policy encoded in a distinct lease type.
// No ordinary Request, Commit, Cancel, snapshot or persistence surface exists.
// The model owner must outlive this lease and retain quarantined native state.
class TextDiagnosticLease {
public:
  using DiscardResult = cache::DiagnosticDiscardResult;
  virtual ~TextDiagnosticLease() = default;
  TextDiagnosticLease(const TextDiagnosticLease&) = delete;
  TextDiagnosticLease& operator=(const TextDiagnosticLease&) = delete;
  [[nodiscard]] virtual TextRunnerState& state() const = 0;
  [[nodiscard]] virtual std::size_t cached_tokens() const noexcept = 0;
  virtual void PreparePrefill(std::size_t tokens) = 0;
  virtual DiscardResult Discard() = 0;
  virtual void Quarantine() noexcept = 0;

protected:
  TextDiagnosticLease() = default;
};
}  // namespace gufo::server
#endif
