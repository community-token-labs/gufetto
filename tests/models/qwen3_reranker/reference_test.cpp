// Qwen3-Reranker config, prompt framing and the scalar CPU reference scorer,
// checked against hand-computed tiny models. No checkpoint is needed; set
// GUFO_QWEN3_RERANKER_MODEL to also parse a real config.json.
//
// TODO(rerank topic 5): reference scores from `transformers` for the pinned
// Qwen/Qwen3-Reranker-0.6B checkpoint, produced on a nimo. None are
// committed yet, and none may be written by hand.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/cli/serve/rerank_api.hpp"
#include "src/cli/serve/rerank_reference_scorer.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen3_reranker/config.hpp"
#include "src/models/qwen3_reranker/prompt.hpp"
#include "src/models/qwen3_reranker/reference/cpu_scorer.hpp"

namespace {

namespace json = gufo::json;
namespace reranker = gufo::models::qwen3_reranker;
namespace reference = gufo::models::qwen3_reranker::reference;
using gufo::server::HandleRerankApiRequest;
using gufo::server::HttpRequest;
using gufo::server::HttpResponse;
using gufo::server::ReferenceRerankScorer;
using gufo::server::ReferenceRerankScorerOptions;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "FAIL qwen3_reranker_reference_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition)
    Fail(message);
}

void Near(double actual, double expected, const std::string& message) {
  if (!(std::fabs(actual - expected) <= 1e-6))
    Fail(message + ": got " + std::to_string(actual) + ", expected " +
         std::to_string(expected));
}

double Sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

// Qwen/Qwen3-Reranker-0.6B's config.json at e61197ed (sha256 d479c427...).
constexpr std::string_view kQwen3Config = R"({
  "architectures": ["Qwen3ForCausalLM"], "attention_bias": false,
  "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
  "head_dim": 128, "hidden_act": "silu", "hidden_size": 1024,
  "initializer_range": 0.02, "intermediate_size": 3072,
  "max_position_embeddings": 40960, "max_window_layers": 28,
  "model_type": "qwen3", "num_attention_heads": 16, "num_hidden_layers": 28,
  "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_scaling": null,
  "rope_theta": 1000000, "sliding_window": null, "tie_word_embeddings": true,
  "torch_dtype": "bfloat16", "transformers_version": "4.51.3",
  "use_cache": true, "use_sliding_window": false, "vocab_size": 151669})";

void TestConfig() {
  reranker::Config config;
  std::string error;
  Check(reranker::ParseConfig(json::parse(kQwen3Config), &config, &error),
        "Qwen3-Reranker-0.6B config parses: " + error);
  Check(config.hidden_size == 1024 && config.num_hidden_layers == 28 &&
            config.num_attention_heads == 16 &&
            config.num_key_value_heads == 8 && config.head_dim == 128 &&
            config.intermediate_size == 3072 && config.vocab_size == 151669 &&
            config.max_position_embeddings == 40960 &&
            config.tie_word_embeddings && config.rope_theta == 1000000.0 &&
            config.rms_norm_eps == 1.0e-6F,
        "Qwen3-Reranker-0.6B geometry");
  const auto rejects = [&](std::string_view from, std::string_view to) {
    std::string text(kQwen3Config);
    const auto at = text.find(from);
    Check(at != std::string::npos, "test edit applies: " + std::string(from));
    text.replace(at, from.size(), to);
    reranker::Config ignored;
    std::string reason;
    Check(!reranker::ParseConfig(json::parse(text), &ignored, &reason) &&
              !reason.empty(),
          "config rejects " + std::string(to));
  };
  rejects(R"("Qwen3ForCausalLM")", R"("Qwen3ForSequenceClassification")");
  rejects(R"("model_type": "qwen3")", R"("model_type": "qwen2")");
  rejects(R"("hidden_act": "silu")", R"("hidden_act": "gelu")");
  rejects(R"("rope_scaling": null)", R"("rope_scaling": {"type": "yarn"})");
  rejects(R"("use_sliding_window": false)", R"("use_sliding_window": true)");
  rejects(R"("attention_bias": false)", R"("attention_bias": true)");
  rejects(R"("num_key_value_heads": 8)", R"("num_key_value_heads": 3)");
  rejects(R"("head_dim": 128)", R"("head_dim": 127)");
  rejects(R"("hidden_size": 1024)", R"("hidden_size": 0)");
  rejects(R"("rms_norm_eps": 1e-06)", R"("rms_norm_eps": -1)");
  rejects(R"("tie_word_embeddings": true)", R"("tie_word_embeddings": 1)");
  Check(!reranker::LoadConfig("/nonexistent/qwen3-reranker", &config, &error),
        "a missing directory fails");

  if (const char* root = std::getenv("GUFO_QWEN3_RERANKER_MODEL")) {
    reranker::Config real;
    Check(reranker::LoadConfig(root, &real, &error),
          "the pinned reranker config parses: " + error);
    Check(real.num_hidden_layers == 28 && real.hidden_size == 1024 &&
              real.vocab_size == 151669 && real.tie_word_embeddings,
          "the pinned reranker geometry");
  }
}

void TestPrompt() {
  // Byte goldens from the model card, written out independently of the
  // shared control-token constants.
  Check(reranker::PromptPrefix() ==
            "<|im_start|>system\nJudge whether the Document meets the "
            "requirements based on the Query and the Instruct provided. Note "
            "that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n",
        "prefix golden");
  Check(reranker::PromptSuffix() ==
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "suffix golden");
  Check(reranker::FormatPair("Find it", "where?", "here.") ==
            "<Instruct>: Find it\n<Query>: where?\n<Document>: here.",
        "pair golden");
  Check(reranker::kDefaultInstruction ==
            "Given a web search query, retrieve relevant passages that answer "
            "the query",
        "default instruction golden");

  const reranker::PromptTokens frame{
      .prefix = {7, 8}, .suffix = {9}, .yes = 2, .no = 3};
  std::vector<std::uint32_t> tokens;
  Check(reranker::FramePair(frame, {1, 2, 3}, 6, false, &tokens) &&
            tokens == std::vector<std::uint32_t>{7, 8, 1, 2, 3, 9},
        "a pair that fits is framed whole");
  Check(!reranker::FramePair(frame, {1, 2, 3}, 5, false, &tokens),
        "a long pair is refused without truncation");
  Check(reranker::FramePair(frame, {1, 2, 3}, 5, true, &tokens) &&
            tokens == std::vector<std::uint32_t>{7, 8, 1, 2, 9},
        "truncation keeps the pair's first tokens and the whole frame");
  Check(!reranker::FramePair(frame, {1}, 3, true, &tokens),
        "a frame that fills max_length leaves no room");

  const reranker::Encoder multi = [](std::string_view text, bool) {
    return std::vector<std::uint32_t>(text.size(), 1);
  };
  reranker::PromptTokens ignored;
  std::string error;
  Check(!reranker::TokenizeFrame(multi, &ignored, &error) && !error.empty(),
        "multi-token answers are refused");
}

void TestMath() {
  std::vector<float> out(2);
  reference::RmsNorm(std::vector<float>{3, 4}, std::vector<float>{1, 2}, 0.0F,
                     out);
  Near(out[0], 3.0 / std::sqrt(12.5), "RMSNorm x0");
  Near(out[1], 8.0 / std::sqrt(12.5), "RMSNorm x1 with weight 2");

  std::vector<float> head{1, 2, 3, 4};
  reference::ApplyRope(head, 0, 100.0);
  Check(head == std::vector<float>{1, 2, 3, 4}, "position 0 is identity");
  reference::ApplyRope(head, 2, 100.0);
  // Pairs (0, 2) at frequency 1 and (1, 3) at 100^(-1/2) = 0.1.
  Near(head[0], 1 * std::cos(2.0) - 3 * std::sin(2.0), "RoPE x0");
  Near(head[1], 2 * std::cos(0.2) - 4 * std::sin(0.2), "RoPE x1");
  Near(head[2], 3 * std::cos(2.0) + 1 * std::sin(2.0), "RoPE x2");
  Near(head[3], 4 * std::cos(0.2) + 2 * std::sin(0.2), "RoPE x3");
}

/// Hidden 2, one head of width 2, one layer, four tokens: 0 = [1, 0],
/// 1 = [0, 1], yes (2) = [2, 0], no (3) = [0, 2]. Every norm weight is 1 and
/// every projection is zero until a case sets it.
reference::Weights TinyModel() {
  reference::Weights w;
  w.config = reranker::Config{
      .vocab_size = 4,
      .hidden_size = 2,
      .intermediate_size = 1,
      .num_hidden_layers = 1,
      .num_attention_heads = 1,
      .num_key_value_heads = 1,
      .head_dim = 2,
      .max_position_embeddings = 64,
      .rms_norm_eps = 1.0e-12F,
      .rope_theta = 10000.0,
      .tie_word_embeddings = true,
      .torch_dtype = "float32",
  };
  w.embed_tokens = {1, 0, 0, 1, 2, 0, 0, 2};
  w.norm = {1, 1};
  reference::LayerWeights layer;
  layer.input_layernorm = {1, 1};
  layer.post_attention_layernorm = {1, 1};
  layer.q_norm = {1, 1};
  layer.k_norm = {1, 1};
  layer.q_proj = layer.k_proj = layer.v_proj = layer.o_proj = {0, 0, 0, 0};
  layer.gate_proj = layer.up_proj = {0, 0};
  layer.down_proj = {0, 0};
  w.layers.push_back(layer);
  return w;
}

/// The score of a final hidden state [a, b]: normalized, then
/// sigmoid(yes - no) with yes = 2a / r and no = 2b / r.
double ExpectedScore(double a, double b) {
  const double r = std::sqrt((a * a + b * b) / 2.0);
  return Sigmoid(2.0 * (a - b) / r);
}

void TestForward() {
  const std::vector<float> identity{1, 0, 0, 1};
  std::string error;
  {
    // No attention or MLP output: the last embedding alone decides.
    const reference::Weights w = TinyModel();
    Check(reference::ValidateWeights(w, &error), "tiny model is valid");
    Near(reference::ScoreTokens(w, std::vector<std::uint32_t>{0}, 2, 3),
         Sigmoid(2.0 * std::sqrt(2.0)), "token 0 says yes");
    Near(reference::ScoreTokens(w, std::vector<std::uint32_t>{1, 0, 1}, 2, 3),
         Sigmoid(-2.0 * std::sqrt(2.0)), "token 1 last says no");
  }
  {
    // Zero queries and keys attend uniformly, so the last position adds the
    // mean of the normalized values: [0, sqrt2] and [sqrt2, 0].
    reference::Weights w = TinyModel();
    w.layers[0].v_proj = w.layers[0].o_proj = identity;
    const double half = std::sqrt(2.0) / 2.0;
    Near(reference::ScoreTokens(w, std::vector<std::uint32_t>{1, 0}, 2, 3),
         ExpectedScore(1.0 + half, half), "uniform attention averages values");
  }
  {
    // Identity projections: q, k and v are the normalized inputs, then RoPE
    // turns position 1 by one radian (head_dim 2 has frequency 1).
    reference::Weights w = TinyModel();
    w.layers[0].q_proj = w.layers[0].k_proj = identity;
    w.layers[0].v_proj = w.layers[0].o_proj = identity;
    const double s2 = std::sqrt(2.0);
    // k0 = [0, s2], q1 = k1 = s2 [cos 1, sin 1]; scores scale by 1/sqrt(2).
    const double score0 = s2 * std::sin(1.0);
    const double score1 = s2;
    const double w0 = std::exp(score0) / (std::exp(score0) + std::exp(score1));
    const double w1 = 1.0 - w0;
    // Values are not rotated: v0 = [0, s2], v1 = [s2, 0]; x1 = [1, 0].
    Near(reference::ScoreTokens(w, std::vector<std::uint32_t>{1, 0}, 2, 3),
         ExpectedScore(1.0 + w1 * s2, w0 * s2),
         "RoPE and the scaled dot product weight the values");
    // Causality: the first position never sees the second.
    const auto first =
        reference::LastHiddenState(w, std::vector<std::uint32_t>{1});
    Near(first[1], s2, "a single token attends to itself");
  }
  {
    // SwiGLU only: gate = up = x0 of the normed input, and down writes the
    // product into the second coordinate.
    reference::Weights w = TinyModel();
    w.layers[0].gate_proj = w.layers[0].up_proj = {1, 0};
    w.layers[0].down_proj = {0, 1};
    const double g = std::sqrt(2.0);
    const double m = g / (1.0 + std::exp(-g)) * g;
    Near(reference::ScoreTokens(w, std::vector<std::uint32_t>{0}, 2, 3),
         ExpectedScore(1.0, m), "SwiGLU adds silu(gate) * up");
  }
  {
    // GQA: two query heads share one key/value head, and an untied head
    // reads lm_head instead of the embeddings.
    reference::Weights w = TinyModel();
    w.config.num_attention_heads = 2;
    w.layers[0].q_proj = {0, 0, 0, 0, 0, 0, 0, 0};
    w.layers[0].o_proj = {0, 0, 0, 0, 0, 0, 0, 0};
    Check(reference::ValidateWeights(w, &error),
          "GQA sizes validate: " + error);
    w.config.tie_word_embeddings = false;
    Check(!reference::ValidateWeights(w, &error), "untied needs lm_head");
    w.lm_head = {0, 0, 0, 0, 0, 2, 2, 0};  // yes and no swap their rows.
    Check(reference::ValidateWeights(w, &error), "untied head validates");
    Near(reference::ScoreTokens(w, std::vector<std::uint32_t>{0}, 2, 3),
         Sigmoid(-2.0 * std::sqrt(2.0)), "the untied head decides");
    w.layers[0].k_proj.pop_back();
    Check(!reference::ValidateWeights(w, &error) &&
              error.find("k_proj") != std::string::npos,
          "a wrong tensor size is named");
  }
}

/// Token 2 spells "yes" and 3 spells "no". Other bytes become token 0 or 1 by
/// parity. It records whether each call parsed control tokens.
struct ToyTokenizer {
  std::vector<std::uint32_t> Encode(std::string_view text, bool parse_special) {
    calls.emplace_back(text, parse_special);
    if (text == "yes")
      return {2};
    if (text == "no")
      return {3};
    std::vector<std::uint32_t> ids;
    for (const char byte : text)
      ids.push_back(static_cast<unsigned char>(byte) % 2U);
    return ids;
  }
  std::vector<std::pair<std::string, bool>> calls;
};

HttpResponse Post(gufo::server::RerankScorer* scorer, const std::string& body,
                  std::function<bool()> cancelled = {}) {
  HttpRequest request;
  request.method = "POST";
  request.path = "/v1/rerank";
  request.body = body;
  request.is_cancelled = std::move(cancelled);
  return HandleRerankApiRequest(request, scorer);
}

void TestScorer() {
  reference::Weights tiny = TinyModel();
  const std::vector<float> identity{1, 0, 0, 1};
  tiny.layers[0].q_proj = tiny.layers[0].k_proj = identity;
  tiny.layers[0].v_proj = tiny.layers[0].o_proj = identity;
  tiny.layers[0].gate_proj = tiny.layers[0].up_proj = {1, 0};
  tiny.layers[0].down_proj = {0, 1};
  const auto weights = std::make_shared<const reference::Weights>(tiny);
  auto tokenizer = std::make_shared<ToyTokenizer>();
  const reranker::Encoder encode = [tokenizer](std::string_view text,
                                               bool special) {
    return tokenizer->Encode(text, special);
  };
  ReferenceRerankScorer scorer(weights, encode,
                               {.model_id = "tiny", .max_length = 400});
  Check(scorer.ready(), "scorer ready: " + scorer.initialization_error());
  Check(scorer.frame().yes == 2 && scorer.frame().no == 3, "answer tokens");

  const HttpResponse response = Post(
      &scorer, R"({"query":"q?","documents":["first doc","2nd"],"top_n":2})");
  Check(response.status == 200, "reference scores: " + response.body);
  const json::Value body = json::parse(response.body);
  std::size_t expected_tokens = 0;
  const std::vector<std::string> documents{"first doc", "2nd"};
  std::vector<double> expected(2);
  for (std::size_t i = 0; i < 2; ++i) {
    std::vector<std::uint32_t> tokens;
    Check(reranker::FramePair(
              scorer.frame(),
              tokenizer->Encode(
                  reranker::FormatPair(reranker::kDefaultInstruction, "q?",
                                       documents[i]),
                  false),
              400, false, &tokens),
          "direct framing");
    expected[i] = reference::ScoreTokens(*weights, tokens, 2, 3);
    expected_tokens += tokens.size();
  }
  for (const auto& result : body.find("results")->items()) {
    const std::size_t index = result.member_size("index", 9);
    Check(index < 2, "result index");
    Near(result.member_double("relevance_score", -1), expected[index],
         "HTTP score equals the direct reference score");
  }
  Check(body.find("usage")->member_size("prompt_tokens") == expected_tokens,
        "usage counts framed tokens");
  for (const auto& [text, special] : tokenizer->calls) {
    if (text.find("<Query>") != std::string::npos)
      Check(!special, "request text is never parsed for control tokens");
    if (text.find("<|im_start|>") != std::string::npos)
      Check(special, "the frame is parsed for control tokens");
  }

  // The toy frame is 242 tokens and the default pair text 110 more, so a
  // 400-token limit refuses a 100-byte document unless truncation is on.
  const std::string long_doc(100, 'x');
  const std::string request =
      R"({"query":"q","documents":["short",")" + long_doc + R"("]})";
  ReferenceRerankScorer strict(weights, encode, {.max_length = 400});
  Check(strict.ready(),
        "strict scorer ready: " + strict.initialization_error());
  const HttpResponse refused = Post(&strict, request);
  Check(refused.status == 400 &&
            refused.body.find("index 1 exceed the 400-token") !=
                std::string::npos,
        "a long pair is refused: " + refused.body);
  ReferenceRerankScorer cut(weights, encode,
                            {.max_length = 400, .truncate = true});
  const HttpResponse truncated = Post(&cut, request);
  Check(truncated.status == 200 &&
            truncated.body.find("\"prompt_tokens\":") != std::string::npos,
        "truncation scores the long pair: " + truncated.body);
  Check(
      json::parse(truncated.body).find("usage")->member_size("prompt_tokens") <=
          800,
      "each truncated pair fits max_length");

  Check(Post(&scorer, R"({"query":"q","documents":["a"]})",
             [] {
               return true;
             }).status == 499,
        "cancellation stops scoring");

  reference::Weights broken = tiny;
  broken.norm.pop_back();
  ReferenceRerankScorer unready(
      std::make_shared<const reference::Weights>(broken), encode);
  Check(!unready.ready() && !unready.initialization_error().empty(),
        "inconsistent weights leave the scorer unready");
  Check(Post(&unready, R"({"query":"q","documents":["a"]})").status == 503,
        "an unready reference answers 503");
}

}  // namespace

int main() {
  TestConfig();
  TestPrompt();
  TestMath();
  TestForward();
  TestScorer();
  std::cout << "PASS qwen3_reranker_reference_test\n";
  return 0;
}
