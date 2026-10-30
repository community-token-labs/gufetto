#ifndef GUFO_SERVER_RERANK_API_HPP_
#define GUFO_SERVER_RERANK_API_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/http_server.hpp"

namespace gufo::server {

inline constexpr std::string_view kRerankApiSchema = "gufo.rerank-api.v1";

/// Scores for one request, in document order.
struct RerankScores {
  /// Probability in [0, 1] that each document answers the query.
  std::vector<double> scores;
  /// Tokens the model read for each query/document pair, including its
  /// template and markers, after any truncation.
  std::vector<std::size_t> token_counts;
  /// Index of the offending document, set only with
  /// `RerankStatus::kInputTooLong`.
  std::size_t failed_index{0};
};

enum class RerankStatus {
  kOk,
  kCancelled,
  kInputTooLong,
  kFailed,
  /// The device failed a probe after an error. The scorer stays lost.
  kDeviceLost,
};

/// A loaded reranker model. The HTTP layer validates requests and shapes
/// responses; implementations only score query/document pairs. Tests use a
/// CPU fake.
///
/// Concurrency: `HttpServer` runs each connection on its own thread and calls
/// every member, `Score` included, from those threads at once, without a
/// lock. Implementations must be thread-safe. One that drives a single device
/// serializes internally, as `EmbeddingService` does: concurrent `Score` calls
/// queue in arrival order, and each polls its own `is_cancelled` while it
/// waits. The other members are called concurrently too and must not block
/// behind a running `Score`.
class RerankScorer {
public:
  virtual ~RerankScorer() = default;

  [[nodiscard]] virtual bool ready() const noexcept = 0;
  /// True once any request reported `RerankStatus::kDeviceLost`, and true
  /// from then on: the flag is sticky and never clears. It is published
  /// atomically (for example a `std::atomic<bool>` stored before `Score`
  /// returns `kDeviceLost`), so another thread that reads it after that return
  /// sees true. The server polls it around every request.
  [[nodiscard]] virtual bool device_lost() const noexcept { return false; }
  [[nodiscard]] virtual std::string model_id() const = 0;
  [[nodiscard]] virtual std::string backend_name() const = 0;
  /// Longest accepted query/document pair in tokens.
  [[nodiscard]] virtual std::size_t max_pair_tokens() const = 0;

  /// Scores every document against `query` as one request. Returns
  /// `kCancelled` when `is_cancelled` reports a disconnect before scoring
  /// finishes.
  [[nodiscard]] virtual RerankStatus Score(
      const std::string& query, const std::vector<std::string>& documents,
      const std::function<bool()>& is_cancelled, RerankScores* scores,
      std::string* error) = 0;
};

[[nodiscard]] bool IsRerankApiPath(std::string_view path) noexcept;

/// llama.cpp-compatible `POST /v1/rerank` in its Jina/Cohere form
/// (`documents`) and its Text Embeddings Inference form (`texts`). A null or
/// unready `scorer` answers 503.
[[nodiscard]] HttpResponse HandleRerankApiRequest(const HttpRequest& request,
                                                  RerankScorer* scorer);

}  // namespace gufo::server

#endif  // GUFO_SERVER_RERANK_API_HPP_
