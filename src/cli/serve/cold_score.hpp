#ifndef GUFO_CLI_SERVE_COLD_SCORE_HPP_
#define GUFO_CLI_SERVE_COLD_SCORE_HPP_

#include <memory>

#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/qwen38_flash_next/cold_score.hpp"

namespace gufo::server {
// Implemented only by the native Flash-Next binding; unsupported families fail
// closed without another capability list or a second Session implementation.
class ColdScoreState : public TextRunnerState {
public:
  virtual std::uint32_t ColdVocab() const noexcept = 0;
  virtual void ScoreCold(std::span<const std::int32_t>, std::uint32_t rows,
                         std::span<float>, std::span<std::uint32_t> pass_ends,
                         std::size_t& passes, const CancellationCheck&,
                         const std::function<void()>& progress) = 0;
  virtual bool DrainCold() noexcept = 0;
};

class ColdScoreResult {
public:
  struct Storage;
  ~ColdScoreResult();
  ColdScoreResult(ColdScoreResult&&) noexcept;
  ColdScoreResult& operator=(ColdScoreResult&&) noexcept;
  std::span<const float> logits() const;
  std::span<const std::uint32_t> pass_ends() const;
  std::uint32_t vocab() const;

private:
  explicit ColdScoreResult(std::unique_ptr<Storage>);
  std::unique_ptr<Storage> storage_;
  friend ColdScoreResult ScoreColdOnWorker(
      std::shared_ptr<TextRunnerPool>, std::span<const std::int32_t>,
      std::uint32_t, const TextRunnerState::CancellationCheck&,
      const std::function<void()>&);
};

// Synchronous, ONLY on an exclusive model-owner worker with ordinary work
// quiescent. Native lease settlement is finished before returning host data.
// HTTP/scheduler admission is a later consumer, not bypassed by this contract.
// Cancellation captures must own their data through uncertain native cleanup.
// Result ownership serializes diagnostics through send; unsafe work is retained
// permanently, including buffers/pool/lease. No allocator-sized response tree.
ColdScoreResult ScoreColdOnWorker(
    std::shared_ptr<TextRunnerPool>, std::span<const std::int32_t>,
    std::uint32_t rows, const TextRunnerState::CancellationCheck& = {},
    const std::function<void()>& progress = {});
std::size_t ColdScoreControlBytes() noexcept;
// Called ONLY by Model's ON approval before Executor::Create; concrete native
// binding and cold-request sizes are added by its implementation.
bool ApproveColdScoreMemory(
    const models::qwen38_flash_next::DiagnosticLogitCapacity&,
    std::size_t context, std::string* error);
}  // namespace gufo::server
#endif
