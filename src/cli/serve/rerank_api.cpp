#include "src/cli/serve/rerank_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {
namespace {

constexpr std::string_view kRerankPath = "/v1/rerank";
// llama.cpp's aliases. All four share one handler there.
constexpr std::string_view kRerankingPath = "/v1/reranking";
constexpr std::string_view kNativeRerankPath = "/rerank";
constexpr std::string_view kNativeRerankingPath = "/reranking";
// Cohere's per-request document limit.
constexpr std::size_t kMaximumDocuments = 1000;

std::string_view ReasonPhrase(int status) {
  switch (status) {
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 499:
      return "Client Closed Request";
    case 503:
      return "Service Unavailable";
    default:
      return "Internal Server Error";
  }
}

HttpResponse Error(int status, std::string message, std::string code) {
  json::Value root = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = status < 500 ? "invalid_request_error" : "server_error";
  error["code"] = std::move(code);
  error["param"] = nullptr;
  root["error"] = std::move(error);
  return {
      .status = status,
      .reason = std::string(ReasonPhrase(status)),
      .body = root.dump(),
      .headers = {{"Content-Type", "application/json"}},
      .streaming_body = {},
      .log_details = {},
  };
}

HttpResponse BadRequest(std::string message, std::string code) {
  return Error(400, std::move(message), std::move(code));
}

bool IsKnownField(std::string_view name) {
  return name == "model" || name == "query" || name == "documents" ||
         name == "texts" || name == "top_n" || name == "return_documents" ||
         name == "return_text";
}

/// The member's value, or nullptr when it is absent or null.
const json::Value* Optional(const json::Value& root, const std::string& name) {
  const json::Value* value = root.find(name);
  return value == nullptr || value->is_null() ? nullptr : value;
}

/// Reads an optional boolean member into `out`. Returns false when it has
/// another type.
bool ReadFlag(const json::Value& root, const std::string& name, bool* out) {
  const json::Value* value = Optional(root, name);
  if (value == nullptr)
    return true;
  if (!value->is_bool())
    return false;
  *out = value->as_bool();
  return true;
}

HttpResponse Rerank(const HttpRequest& request, RerankScorer& scorer) {
  json::Value root;
  try {
    root = json::parse(request.body);
  } catch (const std::exception& error) {
    return BadRequest(error.what(), "parse_error");
  }
  if (!root.is_object()) {
    return BadRequest("request body must be a JSON object",
                      "invalid_request_body");
  }
  for (const auto& [name, unused] : root.members()) {
    (void)unused;
    if (!IsKnownField(name)) {
      return BadRequest("unsupported rerank parameter: " + name,
                        "unsupported_field");
    }
  }

  if (const json::Value* model = Optional(root, "model"); model != nullptr) {
    if (!model->is_string()) {
      return BadRequest("'model' must be a string", "invalid_parameter_type");
    }
    if (model->str() != scorer.model_id()) {
      return Error(404, "requested reranker model is not loaded",
                   "model_not_found");
    }
  }

  const json::Value* query = Optional(root, "query");
  if (query == nullptr) {
    return BadRequest("'query' is required", "missing_query");
  }
  if (!query->is_string()) {
    return BadRequest("'query' must be a string", "invalid_parameter_type");
  }
  if (query->str().empty()) {
    return BadRequest("'query' must not be empty", "empty_query");
  }

  // As in llama.cpp, the member that carries the documents selects the form:
  // `texts` is Text Embeddings Inference and `documents` is Jina/Cohere.
  const json::Value* documents_value = Optional(root, "documents");
  const json::Value* texts_value = Optional(root, "texts");
  if (documents_value != nullptr && texts_value != nullptr) {
    return BadRequest("send either 'documents' or 'texts', not both",
                      "conflicting_documents");
  }
  const bool tei = texts_value != nullptr;
  const json::Value* list = tei ? texts_value : documents_value;
  const char* list_name = tei ? "'texts'" : "'documents'";
  if (list == nullptr) {
    return BadRequest("'documents' or 'texts' is required",
                      "missing_documents");
  }
  if (!list->is_array()) {
    return BadRequest(std::string(list_name) + " must be an array",
                      "invalid_parameter_type");
  }
  if (list->items().empty()) {
    return BadRequest(std::string(list_name) + " must not be empty",
                      "empty_documents");
  }
  if (list->size() > kMaximumDocuments) {
    return BadRequest(std::string(list_name) + " accepts at most " +
                          std::to_string(kMaximumDocuments) + " items",
                      "too_many_documents");
  }
  std::vector<std::string> documents;
  documents.reserve(list->size());
  for (const auto& item : list->items()) {
    if (item.is_string()) {
      documents.push_back(item.str());
      continue;
    }
    // Jina and Cohere also take `{"text": "..."}`. Image documents, which
    // multimodal rerankers accept, are not supported.
    if (!tei && item.is_object()) {
      const json::Value* text = item.find("text");
      if (item.contains("image")) {
        return BadRequest("image documents are not supported",
                          "unsupported_input");
      }
      if (item.size() == 1 && text != nullptr && text->is_string()) {
        documents.push_back(text->str());
        continue;
      }
    }
    return BadRequest(tei ? "'texts' items must be strings"
                          : "'documents' items must be strings or "
                            "{\"text\": string} objects",
                      "invalid_parameter_type");
  }

  std::size_t top_n = documents.size();
  if (const json::Value* value = Optional(root, "top_n"); value != nullptr) {
    constexpr std::size_t kInvalid = std::numeric_limits<std::size_t>::max();
    const std::size_t parsed = value->as_size(kInvalid);
    if (parsed == kInvalid) {
      return BadRequest("'top_n' must be a non-negative integer",
                        "invalid_top_n");
    }
    top_n = std::min(parsed, documents.size());
  }

  // `return_text` is TEI's echo flag and `return_documents` is Jina's and
  // Cohere's. Each belongs to its own form.
  bool return_text = false;
  if (!ReadFlag(root, "return_text", &return_text) ||
      !ReadFlag(root, "return_documents", &return_text)) {
    return BadRequest("'return_text' and 'return_documents' must be booleans",
                      "invalid_parameter_type");
  }
  if (Optional(root, tei ? "return_documents" : "return_text") != nullptr) {
    return BadRequest(tei ? "'return_documents' requires 'documents'; use "
                            "'return_text' with 'texts'"
                          : "'return_text' requires 'texts'; use "
                            "'return_documents' with 'documents'",
                      "unsupported_field");
  }

  RerankScores scored;
  std::string error;
  switch (scorer.Score(query->str(), documents, request.is_cancelled, &scored,
                       &error)) {
    case RerankStatus::kOk:
      break;
    case RerankStatus::kCancelled:
      return Error(499, "rerank request cancelled", "cancelled");
    case RerankStatus::kInputTooLong:
      return BadRequest(
          "query and document at index " + std::to_string(scored.failed_index) +
              " exceed the " + std::to_string(scorer.max_pair_tokens()) +
              "-token context window",
          "context_length_exceeded");
    case RerankStatus::kFailed:
      return Error(500, error.empty() ? "rerank failed" : std::move(error),
                   "rerank_failed");
    case RerankStatus::kDeviceLost:
      return Error(503, "GPU device lost; restart the server", "device_lost");
  }
  if (request.is_cancelled && request.is_cancelled()) {
    return Error(499, "rerank request cancelled", "cancelled");
  }
  if (scored.scores.size() != documents.size() ||
      scored.token_counts.size() != documents.size()) {
    return Error(500, "reranker returned an inconsistent batch",
                 "rerank_failed");
  }
  for (const double score : scored.scores) {
    if (!std::isfinite(score) || score < 0.0 || score > 1.0) {
      return Error(500, "reranker returned a score outside [0, 1]",
                   "rerank_failed");
    }
  }
  std::size_t prompt_tokens = 0;
  for (const std::size_t count : scored.token_counts) {
    prompt_tokens += count;
  }

  // Highest score first. Ties keep request order, so equal scores rank
  // deterministically.
  std::vector<std::size_t> order(documents.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::stable_sort(order, [&](std::size_t lhs, std::size_t rhs) {
    return scored.scores[lhs] > scored.scores[rhs];
  });
  order.resize(top_n);

  json::Value results = json::Value::array();
  for (const std::size_t index : order) {
    json::Value result = json::Value::object();
    result["index"] = index;
    result[tei ? "score" : "relevance_score"] = scored.scores[index];
    if (return_text && tei) {
      result["text"] = documents[index];
    } else if (return_text) {
      json::Value document = json::Value::object();
      document["text"] = documents[index];
      result["document"] = std::move(document);
    }
    results.push_back(std::move(result));
  }
  // TEI's form is a bare array, without model or usage.
  std::string body;
  if (tei) {
    body = results.dump();
  } else {
    json::Value usage = json::Value::object();
    usage["prompt_tokens"] = prompt_tokens;
    usage["total_tokens"] = prompt_tokens;
    json::Value object = json::Value::object();
    object["model"] = scorer.model_id();
    object["object"] = "list";
    object["usage"] = std::move(usage);
    object["results"] = std::move(results);
    body = object.dump();
  }

  HttpResponse response;
  response.status = 200;
  response.reason = "OK";
  response.body = std::move(body);
  response.headers = {
      {"Content-Type", "application/json"},
      {"X-Gufo-Schema", std::string(kRerankApiSchema)},
      {"X-Gufo-Rerank-Backend", scorer.backend_name()},
  };
  response.log_details = "model=" + scorer.model_id() +
                         " documents=" + std::to_string(documents.size()) +
                         " top_n=" + std::to_string(top_n) +
                         " prompt_tokens=" + std::to_string(prompt_tokens) +
                         (tei ? " shape=tei" : " shape=jina");
  return response;
}

}  // namespace

bool IsRerankApiPath(std::string_view path) noexcept {
  return path == kRerankPath || path == kRerankingPath ||
         path == kNativeRerankPath || path == kNativeRerankingPath;
}

HttpResponse HandleRerankApiRequest(const HttpRequest& request,
                                    RerankScorer* scorer) {
  if (!IsRerankApiPath(request.path) || request.method != "POST") {
    return Error(405, "HTTP method is not supported for this rerank endpoint",
                 "method_not_allowed");
  }
  if (scorer == nullptr) {
    return Error(503, "no reranker model is loaded on this server",
                 "reranker_not_loaded");
  }
  if (!scorer->ready()) {
    return Error(503, "reranker model is not ready",
                 "reranker_service_unavailable");
  }
  return Rerank(request, *scorer);
}

}  // namespace gufo::server
