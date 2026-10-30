#ifndef GUFO_SERVER_RERANK_REFERENCE_SCORER_HPP_
#define GUFO_SERVER_RERANK_REFERENCE_SCORER_HPP_

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/cli/serve/rerank_api.hpp"
#include "src/models/qwen3_reranker/prompt.hpp"
#include "src/models/qwen3_reranker/reference/cpu_scorer.hpp"

namespace gufo::server {

struct ReferenceRerankScorerOptions {
  std::string model_id{"qwen3-reranker-0.6b"};
  std::string instruction{models::qwen3_reranker::kDefaultInstruction};
  std::size_t max_length{models::qwen3_reranker::kDefaultMaxLength};
  /// Cut a pair that does not fit to its first tokens, as the model card
  /// does, instead of refusing it.
  bool truncate{false};
};

/// Qwen3-Reranker on the scalar float32 CPU reference. It is exact and slow:
/// for tests and small inputs, not for serving.
class ReferenceRerankScorer final : public RerankScorer {
public:
  /// Fails, leaving the scorer not ready with `initialization_error()`, when
  /// the weights or the tokenizer's frame are inconsistent.
  ReferenceRerankScorer(
      std::shared_ptr<const models::qwen3_reranker::reference::Weights> weights,
      models::qwen3_reranker::Encoder encode,
      ReferenceRerankScorerOptions options = {});

  [[nodiscard]] bool ready() const noexcept override { return ready_; }
  [[nodiscard]] std::string model_id() const override {
    return options_.model_id;
  }
  [[nodiscard]] std::string backend_name() const override {
    return "cpu-reference";
  }
  [[nodiscard]] std::size_t max_pair_tokens() const override {
    return options_.max_length;
  }
  [[nodiscard]] const std::string& initialization_error() const noexcept {
    return error_;
  }
  [[nodiscard]] const models::qwen3_reranker::PromptTokens& frame()
      const noexcept {
    return frame_;
  }

  RerankStatus Score(const std::string& query,
                     const std::vector<std::string>& documents,
                     const std::function<bool()>& is_cancelled,
                     RerankScores* scores, std::string* error) override;

private:
  std::shared_ptr<const models::qwen3_reranker::reference::Weights> weights_;
  models::qwen3_reranker::Encoder encode_;
  ReferenceRerankScorerOptions options_;
  models::qwen3_reranker::PromptTokens frame_;
  bool ready_{false};
  std::string error_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_RERANK_REFERENCE_SCORER_HPP_
