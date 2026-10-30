#include "src/cli/serve/rerank_reference_scorer.hpp"

#include <exception>
#include <utility>

namespace gufo::server {

namespace qwen3_reranker = models::qwen3_reranker;

ReferenceRerankScorer::ReferenceRerankScorer(
    std::shared_ptr<const qwen3_reranker::reference::Weights> weights,
    qwen3_reranker::Encoder encode, ReferenceRerankScorerOptions options)
    : weights_(std::move(weights)),
      encode_(std::move(encode)),
      options_(std::move(options)) {
  if (weights_ == nullptr || !encode_) {
    error_ = "the reference scorer needs weights and a tokenizer";
    return;
  }
  if (!qwen3_reranker::reference::ValidateWeights(*weights_, &error_) ||
      !qwen3_reranker::TokenizeFrame(encode_, &frame_, &error_))
    return;
  if (frame_.yes >= weights_->config.vocab_size ||
      frame_.no >= weights_->config.vocab_size) {
    error_ = "the answer tokens are outside the model's vocabulary";
    return;
  }
  if (frame_.prefix.size() + frame_.suffix.size() >= options_.max_length) {
    error_ = "the prompt frame does not fit max_length";
    return;
  }
  ready_ = true;
}

RerankStatus ReferenceRerankScorer::Score(
    const std::string& query, const std::vector<std::string>& documents,
    const std::function<bool()>& is_cancelled, RerankScores* scores,
    std::string* error) {
  if (!ready_) {
    *error = error_;
    return RerankStatus::kFailed;
  }
  std::vector<std::uint32_t> tokens;
  for (std::size_t i = 0; i < documents.size(); ++i) {
    if (is_cancelled && is_cancelled())
      return RerankStatus::kCancelled;
    // Request text never becomes a control token.
    const auto pair = encode_(
        qwen3_reranker::FormatPair(options_.instruction, query, documents[i]),
        false);
    if (!qwen3_reranker::FramePair(frame_, pair, options_.max_length,
                                   options_.truncate, &tokens)) {
      scores->failed_index = i;
      return RerankStatus::kInputTooLong;
    }
    try {
      scores->scores.push_back(qwen3_reranker::reference::ScoreTokens(
          *weights_, tokens, frame_.yes, frame_.no));
    } catch (const std::exception& failure) {
      *error = failure.what();
      return RerankStatus::kFailed;
    }
    scores->token_counts.push_back(tokens.size());
  }
  return RerankStatus::kOk;
}

}  // namespace gufo::server
