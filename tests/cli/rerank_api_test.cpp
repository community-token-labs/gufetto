// POST /v1/rerank contract against a CPU fake scorer: no model files, no
// device, and no socket. The test calls the HTTP handler directly.
#include "src/cli/serve/rerank_api.hpp"

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace {

using gufo::server::HandleRerankApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::IsRerankApiPath;
using gufo::server::RerankScorer;
using gufo::server::RerankScores;
using gufo::server::RerankStatus;
namespace json = gufo::json;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL rerank_api_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

/// Scores a document by the share of query bytes it contains, so ranking is
/// deterministic. A few document spellings trigger each scorer status.
class FakeScorer final : public RerankScorer {
public:
  [[nodiscard]] bool ready() const noexcept override { return ready_; }
  [[nodiscard]] bool device_lost() const noexcept override { return lost_; }
  [[nodiscard]] std::string model_id() const override { return "rerank-test"; }
  [[nodiscard]] std::string backend_name() const override { return "cpu-fake"; }
  [[nodiscard]] std::size_t max_pair_tokens() const override { return 64; }

  RerankStatus Score(const std::string& query,
                     const std::vector<std::string>& documents,
                     const std::function<bool()>& is_cancelled,
                     RerankScores* scores, std::string* error) override {
    ++calls;
    last_query = query;
    last_documents = documents;
    for (std::size_t i = 0; i < documents.size(); ++i) {
      const std::string& document = documents[i];
      if (is_cancelled && is_cancelled())
        return RerankStatus::kCancelled;
      if (document == "TOOLONG") {
        scores->failed_index = i;
        return RerankStatus::kInputTooLong;
      }
      if (document == "BROKEN") {
        *error = "device exploded";
        return RerankStatus::kFailed;
      }
      if (document == "LOST") {
        lost_ = true;
        return RerankStatus::kDeviceLost;
      }
      std::size_t hits = 0;
      for (const char byte : query)
        hits += document.find(byte) != std::string::npos ? 1U : 0U;
      double score =
          static_cast<double>(hits) / static_cast<double>(query.size());
      if (document == "NAN")
        score = std::numeric_limits<double>::quiet_NaN();
      if (document == "BIG")
        score = 1.5;
      scores->scores.push_back(score);
      scores->token_counts.push_back(query.size() + document.size() + 3U);
    }
    if (short_batch)
      scores->token_counts.pop_back();
    return RerankStatus::kOk;
  }

  bool ready_{true};
  bool lost_{false};
  bool short_batch{false};
  int calls{0};
  std::string last_query;
  std::vector<std::string> last_documents;
};

HttpResponse Post(RerankScorer* scorer, const std::string& body,
                  const std::string& path = "/v1/rerank") {
  HttpRequest request;
  request.method = "POST";
  request.path = path;
  request.body = body;
  return HandleRerankApiRequest(request, scorer);
}

std::string ErrorCode(const HttpResponse& response) {
  const json::Value body = json::parse(response.body);
  const json::Value* error = body.find("error");
  return error != nullptr ? error->member_str("code") : "";
}

void ExpectError(RerankScorer* scorer, const std::string& body, int status,
                 const std::string& code) {
  const HttpResponse response = Post(scorer, body);
  Check(response.status == status && ErrorCode(response) == code,
        "expected " + std::to_string(status) + " " + code + " for " + body +
            ", got " + std::to_string(response.status) + " " + response.body);
}

void TestPaths() {
  for (const char* path :
       {"/v1/rerank", "/v1/reranking", "/rerank", "/reranking"}) {
    Check(IsRerankApiPath(path), std::string("rerank path ") + path);
  }
  Check(!IsRerankApiPath("/v1/embeddings"), "embeddings is not a rerank path");
  HttpRequest request;
  request.method = "GET";
  request.path = "/v1/rerank";
  FakeScorer scorer;
  Check(HandleRerankApiRequest(request, &scorer).status == 405,
        "GET answers 405");
  Check(scorer.calls == 0, "GET does not score");
}

void TestNotLoaded() {
  const std::string body = R"({"query":"q","documents":["a"]})";
  ExpectError(nullptr, body, 503, "reranker_not_loaded");
  const HttpResponse response = Post(nullptr, body);
  Check(response.body.find("no reranker model is loaded") != std::string::npos,
        "the error names the missing reranker: " + response.body);
  FakeScorer scorer;
  scorer.ready_ = false;
  ExpectError(&scorer, body, 503, "reranker_service_unavailable");
  Check(scorer.calls == 0, "an unready scorer is not called");
}

void TestJinaShape() {
  FakeScorer scorer;
  const HttpResponse response =
      Post(&scorer,
           R"({"model":"rerank-test","query":"abc","documents":["xyz","abc",)"
           R"({"text":"a"},"cb"],"top_n":3,"return_documents":true})");
  Check(response.status == 200, "Jina request succeeds: " + response.body);
  Check(scorer.last_query == "abc", "query reaches the scorer");
  Check(scorer.last_documents ==
            std::vector<std::string>{"xyz", "abc", "a", "cb"},
        "documents reach the scorer in request order, objects unwrapped");
  const json::Value body = json::parse(response.body);
  Check(body.member_str("model") == "rerank-test", "model is the loaded id");
  Check(body.member_str("object") == "list", "object is list");
  const json::Value* usage = body.find("usage");
  // Pairs are 3+3+3, 3+3+3, 3+1+3 and 3+2+3 tokens.
  Check(usage != nullptr && usage->member_size("prompt_tokens") == 33 &&
            usage->member_size("total_tokens") == 33,
        "usage sums pair tokens: " + response.body);
  const json::Value* results = body.find("results");
  Check(results != nullptr && results->size() == 3, "top_n truncates");
  const std::size_t expected_order[] = {1, 3, 2};
  const double expected_scores[] = {1.0, 2.0 / 3.0, 1.0 / 3.0};
  for (std::size_t i = 0; i < 3; ++i) {
    const json::Value& result = results->items()[i];
    Check(result.member_size("index", 99) == expected_order[i],
          "results sort by descending score: " + response.body);
    Check(result.member_double("relevance_score", -1.0) == expected_scores[i],
          "relevance_score carries the score");
    Check(!result.contains("score"), "Jina results use relevance_score");
    const json::Value* document = result.find("document");
    Check(document != nullptr && document->member_str("text") ==
                                     scorer.last_documents[expected_order[i]],
          "return_documents echoes the text");
  }
  Check(response.body.find("\"model\":\"rerank-test\",\"object\":\"list\"") !=
            std::string::npos,
        "members keep llama.cpp's order");
  bool schema = false;
  bool backend = false;
  for (const auto& [name, value] : response.headers) {
    schema =
        schema || (name == "X-Gufo-Schema" && value == "gufo.rerank-api.v1");
    backend =
        backend || (name == "X-Gufo-Rerank-Backend" && value == "cpu-fake");
  }
  Check(schema && backend, "schema and backend headers");

  // Without top_n or return_documents: every document, no echoed text.
  const HttpResponse all =
      Post(&scorer, R"({"query":"abc","documents":["x","abc"]})", "/rerank");
  Check(all.status == 200, "alias succeeds");
  const json::Value all_body = json::parse(all.body);
  Check(all_body.find("results")->size() == 2, "default top_n is all");
  Check(all.body.find("\"document\"") == std::string::npos,
        "documents are not echoed by default");

  // Equal scores keep request order.
  const HttpResponse ties =
      Post(&scorer, R"({"query":"q","documents":["a","b","c"],"top_n":0})");
  Check(ties.status == 200 && json::parse(ties.body).find("results")->empty(),
        "top_n 0 returns no results");
  const HttpResponse ordered =
      Post(&scorer, R"({"query":"q","documents":["a","b","q","c"]})");
  const json::Value ranked = json::parse(ordered.body);
  std::vector<std::size_t> indices;
  for (const auto& result : ranked.find("results")->items())
    indices.push_back(result.member_size("index", 99));
  Check(indices == std::vector<std::size_t>{2, 0, 1, 3},
        "ties keep request order: " + ordered.body);
}

void TestTeiShape() {
  FakeScorer scorer;
  const HttpResponse response = Post(
      &scorer, R"({"query":"ab","texts":["b","ab","z"],"return_text":true})");
  Check(response.status == 200, "TEI request succeeds: " + response.body);
  const json::Value body = json::parse(response.body);
  Check(body.is_array() && body.size() == 3, "TEI returns a bare array");
  const json::Value& first = body.items()[0];
  Check(first.member_size("index", 99) == 1 &&
            first.member_double("score", -1.0) == 1.0 &&
            first.member_str("text") == "ab" &&
            !first.contains("relevance_score"),
        "TEI items carry index, score and text: " + response.body);
  const HttpResponse plain = Post(&scorer, R"({"query":"ab","texts":["b"]})");
  Check(plain.body.find("\"text\"") == std::string::npos,
        "TEI omits text by default");
}

void TestValidation() {
  FakeScorer scorer;
  ExpectError(&scorer, "{", 400, "parse_error");
  ExpectError(&scorer, "[]", 400, "invalid_request_body");
  ExpectError(&scorer, R"({"query":"q","documents":["a"],"rank_fields":[]})",
              400, "unsupported_field");
  ExpectError(&scorer, R"({"model":7,"query":"q","documents":["a"]})", 400,
              "invalid_parameter_type");
  ExpectError(&scorer, R"({"model":"other","query":"q","documents":["a"]})",
              404, "model_not_found");
  ExpectError(&scorer, R"({"documents":["a"]})", 400, "missing_query");
  ExpectError(&scorer, R"({"query":["q"],"documents":["a"]})", 400,
              "invalid_parameter_type");
  ExpectError(&scorer, R"({"query":"","documents":["a"]})", 400, "empty_query");
  ExpectError(&scorer, R"({"query":"q"})", 400, "missing_documents");
  ExpectError(&scorer, R"({"query":"q","documents":["a"],"texts":["a"]})", 400,
              "conflicting_documents");
  ExpectError(&scorer, R"({"query":"q","documents":"a"})", 400,
              "invalid_parameter_type");
  ExpectError(&scorer, R"({"query":"q","documents":[]})", 400,
              "empty_documents");
  ExpectError(&scorer, R"({"query":"q","documents":[1]})", 400,
              "invalid_parameter_type");
  ExpectError(&scorer, R"({"query":"q","texts":[{"text":"a"}]})", 400,
              "invalid_parameter_type");
  ExpectError(&scorer, R"({"query":"q","documents":[{"text":"a","x":1}]})", 400,
              "invalid_parameter_type");
  ExpectError(&scorer,
              R"({"query":"q","documents":[{"text":"a","image":"u"}]})", 400,
              "unsupported_input");
  std::string many = R"({"query":"q","documents":[)";
  for (int i = 0; i < 1001; ++i)
    many += i == 0 ? "\"a\"" : ",\"a\"";
  many += "]}";
  ExpectError(&scorer, many, 400, "too_many_documents");
  for (const char* top_n : {"-1", "1.5", "\"2\"", "true"}) {
    ExpectError(
        &scorer,
        std::string(R"({"query":"q","documents":["a"],"top_n":)") + top_n + "}",
        400, "invalid_top_n");
  }
  ExpectError(&scorer,
              R"({"query":"q","documents":["a"],"return_documents":"yes"})",
              400, "invalid_parameter_type");
  ExpectError(&scorer, R"({"query":"q","documents":["a"],"return_text":true})",
              400, "unsupported_field");
  ExpectError(&scorer, R"({"query":"q","texts":["a"],"return_documents":true})",
              400, "unsupported_field");
  Check(scorer.calls == 0, "invalid requests never reach the scorer");
  // Null members count as absent, as on /v1/embeddings.
  const HttpResponse nulls = Post(
      &scorer, R"({"model":null,"query":"q","documents":["q"],"top_n":null,)"
               R"("return_documents":null,"texts":null})");
  Check(nulls.status == 200, "null members are absent: " + nulls.body);
}

void TestScorerFailures() {
  FakeScorer scorer;
  ExpectError(&scorer, R"({"query":"q","documents":["a","TOOLONG"]})", 400,
              "context_length_exceeded");
  const HttpResponse too_long =
      Post(&scorer, R"({"query":"q","documents":["a","TOOLONG"]})");
  Check(too_long.body.find("index 1 exceed the 64-token") != std::string::npos,
        "the error names the pair and the limit: " + too_long.body);
  ExpectError(&scorer, R"({"query":"q","documents":["BROKEN"]})", 500,
              "rerank_failed");
  Check(Post(&scorer, R"({"query":"q","documents":["BROKEN"]})")
                .body.find("device exploded") != std::string::npos,
        "a failure keeps the scorer's message");
  ExpectError(&scorer, R"({"query":"q","documents":["NAN"]})", 500,
              "rerank_failed");
  ExpectError(&scorer, R"({"query":"q","documents":["BIG"]})", 500,
              "rerank_failed");
  scorer.short_batch = true;
  ExpectError(&scorer, R"({"query":"q","documents":["a"]})", 500,
              "rerank_failed");
  scorer.short_batch = false;
  ExpectError(&scorer, R"({"query":"q","documents":["LOST"]})", 503,
              "device_lost");
  Check(scorer.device_lost(), "the fake reports the lost device");

  FakeScorer cancelled;
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/rerank";
  request.body = R"({"query":"q","documents":["a"]})";
  request.is_cancelled = [] { return true; };
  const HttpResponse response = HandleRerankApiRequest(request, &cancelled);
  Check(response.status == 499 && ErrorCode(response) == "cancelled",
        "a disconnected client is cancelled");
}

}  // namespace

int main() {
  TestPaths();
  TestNotLoaded();
  TestJinaShape();
  TestTeiShape();
  TestValidation();
  TestScorerFailures();
  std::cout << "PASS rerank_api_test\n";
  return 0;
}
