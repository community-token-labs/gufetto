#include "src/cli/serve/cold_score.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <stdexcept>

#include "src/cli/serve/diagnostic_memory.hpp"

namespace gufo::server {
namespace {
using namespace models::qwen38_flash_next;
std::atomic_bool active{false};
struct Gate {
  Gate() {
    if (active.exchange(true))
      throw std::runtime_error("cold score is busy or quarantined");
  }
  ~Gate() { active = false; }
};
}  // namespace
struct ColdScoreResult::Storage {
  Gate gate;
  std::shared_ptr<TextRunnerPool> pool;
  std::unique_ptr<TextRunnerPool::DiagnosticLease> lease;
  ColdScoreState* state{};
  std::unique_ptr<std::int32_t[]> input;
  std::unique_ptr<float[]> output;
  std::array<std::uint32_t, kColdScoreMaxPasses> ends{};
  std::size_t passes{}, values{};
  std::uint32_t vocabulary{};
  bool started{};
};
std::size_t ColdScoreControlBytes() noexcept {
  return sizeof(ColdScoreResult::Storage) + sizeof(ColdScoreResult) +
         sizeof(active) + sizeof(DiagnosticMemoryClaim);
}
ColdScoreResult ScoreColdOnWorker(
    std::shared_ptr<TextRunnerPool> pool, std::span<const std::int32_t> input,
    std::uint32_t rows, const TextRunnerState::CancellationCheck& cancelled,
    const std::function<void()>& progress) {
  if (!pool || input.empty() || input.size() > kColdScoreMaxTokens || !rows ||
      rows > kColdScoreMaxRows || rows > input.size() ||
      input.size() > pool->runner().Descriptor().max_context ||
      !DiagnosticMemoryClaim::Process().Snapshot().bytes)
    throw std::invalid_argument("cold score input or startup claim is invalid");
  auto work = std::make_unique<ColdScoreResult::Storage>();
  work->pool = std::move(pool);
  std::exception_ptr failure;
  try {
    if (!work->pool->usable_capacity())
      throw TextGenerationError(TextGenerationErrorCode::kCacheUnavailable,
                                "no usable cache slots; restart required");
    work->lease = work->pool->TryAcquireDiagnostic();
    if (!work->lease || work->lease->cached_tokens())
      throw std::runtime_error("cold score requires a healthy empty slot");
    work->state = dynamic_cast<ColdScoreState*>(&work->lease->state());
    if (!work->state)
      throw std::runtime_error("cold score is unsupported by this model");
    work->vocabulary = work->state->ColdVocab();
    if (!work->vocabulary || work->vocabulary > kColdScoreMaxVocab)
      throw std::invalid_argument("cold score vocabulary is unsupported");
    for (auto id : input)
      if (id < 0 || std::uint32_t(id) >= work->vocabulary)
        throw std::invalid_argument("cold score token is out of range");
    work->input = std::make_unique_for_overwrite<std::int32_t[]>(input.size());
    std::copy(input.begin(), input.end(), work->input.get());
    work->values = std::size_t(rows) * work->vocabulary;
    work->output = std::make_unique_for_overwrite<float[]>(work->values);
    if (cancelled && cancelled())
      throw std::runtime_error("cold score cancelled");
    work->started = true;
    work->state->ScoreCold({work->input.get(), input.size()}, rows,
                           {work->output.get(), work->values}, work->ends,
                           work->passes, cancelled, progress);
    if (!work->passes || work->passes > work->ends.size() ||
        work->ends[work->passes - 1] != input.size())
      throw std::runtime_error("cold score pass ledger is incomplete");
    for (std::size_t i = 0; i < work->passes; ++i)
      if (!work->ends[i] || (i && work->ends[i] <= work->ends[i - 1]))
        throw std::runtime_error("cold score pass ledger is invalid");
    for (std::size_t i = 0; i < work->values; ++i)
      if (!std::isfinite(work->output[i]))
        throw std::runtime_error("cold score logits are not finite");
  } catch (...) {
    failure = std::current_exception();
  }
  bool safe = true;
  try {
    // F2: drain ALL started runs, including non-cancel throws between passes.
    safe = !work->started || work->state->DrainCold();
    if (safe && work->lease)
      safe =
          work->lease->Discard() == cache::DiagnosticDiscardResult::kComplete;
  } catch (...) {
    safe = false;
  }
  if (!safe) {
    if (work->lease)
      work->lease->Quarantine();
    (void)work.release();  // Actual owners + gate retained, not just a counter.
    DiagnosticMemoryClaim::Process().ReportUnknown("cold score cleanup");
    throw std::runtime_error("cold score quarantined; restart required");
  }
  work->lease
      .reset();  // Settles/destroys ONLY on this owner worker, no queue lock.
  work->pool.reset();
  work->state = nullptr;
  if (failure)
    std::rethrow_exception(failure);
  return ColdScoreResult(std::move(work));
}
ColdScoreResult::ColdScoreResult(std::unique_ptr<Storage> storage)
    : storage_(std::move(storage)) {}
ColdScoreResult::~ColdScoreResult() = default;
ColdScoreResult::ColdScoreResult(ColdScoreResult&&) noexcept = default;
ColdScoreResult& ColdScoreResult::operator=(ColdScoreResult&&) noexcept =
    default;
std::span<const float> ColdScoreResult::logits() const {
  return {storage_->output.get(), storage_->values};
}
std::span<const std::uint32_t> ColdScoreResult::pass_ends() const {
  return {storage_->ends.data(), storage_->passes};
}
std::uint32_t ColdScoreResult::vocab() const {
  return storage_->vocabulary;
}
}  // namespace gufo::server
