#include "src/models/qwen3_reranker/reference/cpu_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gufo::models::qwen3_reranker::reference {
namespace {

/// `output = W input` for a row-major `[rows][input.size()]` matrix, with
/// double accumulation.
void MatVec(const std::vector<float>& matrix, std::span<const float> input,
            std::span<float> output) {
  const std::size_t columns = input.size();
  for (std::size_t row = 0; row < output.size(); ++row) {
    double sum = 0.0;
    const float* weights = matrix.data() + row * columns;
    for (std::size_t column = 0; column < columns; ++column)
      sum += static_cast<double>(weights[column]) *
             static_cast<double>(input[column]);
    output[row] = static_cast<float>(sum);
  }
}

double Dot(std::span<const float> lhs, std::span<const float> rhs) {
  double sum = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i)
    sum += static_cast<double>(lhs[i]) * static_cast<double>(rhs[i]);
  return sum;
}

bool CheckSize(const std::vector<float>& tensor, std::size_t expected,
               const std::string& name, std::string* error) {
  if (tensor.size() == expected)
    return true;
  if (error != nullptr)
    *error = name + " has " + std::to_string(tensor.size()) +
             " values, expected " + std::to_string(expected);
  return false;
}

}  // namespace

bool ValidateWeights(const Weights& weights, std::string* error) {
  const Config& c = weights.config;
  const std::size_t hidden = c.hidden_size;
  const std::size_t q_width = std::size_t{c.num_attention_heads} * c.head_dim;
  const std::size_t kv_width = std::size_t{c.num_key_value_heads} * c.head_dim;
  const std::size_t inter = c.intermediate_size;
  const std::size_t vocab = c.vocab_size;
  if (weights.layers.size() != c.num_hidden_layers) {
    if (error != nullptr)
      *error = "layer count does not match config";
    return false;
  }
  if (!CheckSize(weights.embed_tokens, vocab * hidden, "embed_tokens", error) ||
      !CheckSize(weights.norm, hidden, "norm", error) ||
      !CheckSize(weights.lm_head, c.tie_word_embeddings ? 0 : vocab * hidden,
                 "lm_head", error))
    return false;
  for (std::size_t i = 0; i < weights.layers.size(); ++i) {
    const LayerWeights& l = weights.layers[i];
    const std::string at = "layers." + std::to_string(i) + ".";
    if (!CheckSize(l.input_layernorm, hidden, at + "input_layernorm", error) ||
        !CheckSize(l.q_proj, q_width * hidden, at + "q_proj", error) ||
        !CheckSize(l.k_proj, kv_width * hidden, at + "k_proj", error) ||
        !CheckSize(l.v_proj, kv_width * hidden, at + "v_proj", error) ||
        !CheckSize(l.q_norm, c.head_dim, at + "q_norm", error) ||
        !CheckSize(l.k_norm, c.head_dim, at + "k_norm", error) ||
        !CheckSize(l.o_proj, hidden * q_width, at + "o_proj", error) ||
        !CheckSize(l.post_attention_layernorm, hidden,
                   at + "post_attention_layernorm", error) ||
        !CheckSize(l.gate_proj, inter * hidden, at + "gate_proj", error) ||
        !CheckSize(l.up_proj, inter * hidden, at + "up_proj", error) ||
        !CheckSize(l.down_proj, hidden * inter, at + "down_proj", error))
      return false;
  }
  return true;
}

void RmsNorm(std::span<const float> input, std::span<const float> weight,
             float eps, std::span<float> output) {
  double sum = 0.0;
  for (const float value : input)
    sum += static_cast<double>(value) * static_cast<double>(value);
  const double scale = 1.0 / std::sqrt(sum / static_cast<double>(input.size()) +
                                       static_cast<double>(eps));
  for (std::size_t i = 0; i < input.size(); ++i)
    output[i] = static_cast<float>(static_cast<double>(input[i]) * scale *
                                   static_cast<double>(weight[i]));
}

void ApplyRope(std::span<float> head, std::size_t position, double theta) {
  const std::size_t half = head.size() / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const double frequency =
        std::pow(theta, -2.0 * static_cast<double>(i) /
                            static_cast<double>(head.size()));
    const double angle = static_cast<double>(position) * frequency;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double first = head[i];
    const double second = head[i + half];
    head[i] = static_cast<float>(first * cosine - second * sine);
    head[i + half] = static_cast<float>(second * cosine + first * sine);
  }
}

std::vector<float> LastHiddenState(const Weights& weights,
                                   std::span<const std::uint32_t> tokens) {
  const Config& c = weights.config;
  if (tokens.empty())
    throw std::invalid_argument("cannot score an empty sequence");
  const std::size_t n = tokens.size();
  const std::size_t hidden = c.hidden_size;
  const std::size_t head_dim = c.head_dim;
  const std::size_t heads = c.num_attention_heads;
  const std::size_t kv_heads = c.num_key_value_heads;
  const std::size_t group = heads / kv_heads;
  const std::size_t q_width = heads * head_dim;
  const std::size_t kv_width = kv_heads * head_dim;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  // Residual stream, one row per position.
  std::vector<float> x(n * hidden);
  for (std::size_t t = 0; t < n; ++t) {
    if (tokens[t] >= c.vocab_size)
      throw std::out_of_range("token ID is outside the vocabulary");
    std::copy_n(weights.embed_tokens.begin() +
                    static_cast<std::ptrdiff_t>(tokens[t] * hidden),
                hidden, x.begin() + static_cast<std::ptrdiff_t>(t * hidden));
  }
  std::vector<float> normed(hidden);
  std::vector<float> q(n * q_width);
  std::vector<float> k(n * kv_width);
  std::vector<float> v(n * kv_width);
  std::vector<float> attention(q_width);
  std::vector<float> projected(hidden);
  std::vector<float> gate(c.intermediate_size);
  std::vector<float> up(c.intermediate_size);
  std::vector<double> probabilities(n);

  for (const LayerWeights& layer : weights.layers) {
    for (std::size_t t = 0; t < n; ++t) {
      const std::span<const float> row(x.data() + t * hidden, hidden);
      RmsNorm(row, layer.input_layernorm, c.rms_norm_eps, normed);
      const std::span<float> q_row(q.data() + t * q_width, q_width);
      const std::span<float> k_row(k.data() + t * kv_width, kv_width);
      MatVec(layer.q_proj, normed, q_row);
      MatVec(layer.k_proj, normed, k_row);
      MatVec(layer.v_proj, normed, {v.data() + t * kv_width, kv_width});
      // Qwen3 normalizes each query and key head before RoPE.
      for (std::size_t h = 0; h < heads; ++h) {
        const auto head = q_row.subspan(h * head_dim, head_dim);
        RmsNorm(head, layer.q_norm, c.rms_norm_eps, head);
        ApplyRope(head, t, c.rope_theta);
      }
      for (std::size_t h = 0; h < kv_heads; ++h) {
        const auto head = k_row.subspan(h * head_dim, head_dim);
        RmsNorm(head, layer.k_norm, c.rms_norm_eps, head);
        ApplyRope(head, t, c.rope_theta);
      }
    }
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t kv = h / group;
        const std::span<const float> query(
            q.data() + t * q_width + h * head_dim, head_dim);
        // Causal: position t attends to positions 0..t.
        double peak = -INFINITY;
        for (std::size_t s = 0; s <= t; ++s) {
          probabilities[s] =
              Dot(query, {k.data() + s * kv_width + kv * head_dim, head_dim}) *
              scale;
          peak = std::max(peak, probabilities[s]);
        }
        double total = 0.0;
        for (std::size_t s = 0; s <= t; ++s) {
          probabilities[s] = std::exp(probabilities[s] - peak);
          total += probabilities[s];
        }
        for (std::size_t d = 0; d < head_dim; ++d) {
          double sum = 0.0;
          for (std::size_t s = 0; s <= t; ++s)
            sum += probabilities[s] *
                   static_cast<double>(v[s * kv_width + kv * head_dim + d]);
          attention[h * head_dim + d] = static_cast<float>(sum / total);
        }
      }
      const std::span<float> row(x.data() + t * hidden, hidden);
      MatVec(layer.o_proj, attention, projected);
      for (std::size_t i = 0; i < hidden; ++i)
        row[i] += projected[i];
      RmsNorm(row, layer.post_attention_layernorm, c.rms_norm_eps, normed);
      MatVec(layer.gate_proj, normed, gate);
      MatVec(layer.up_proj, normed, up);
      for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        gate[i] = static_cast<float>(g / (1.0 + std::exp(-g)) *
                                     static_cast<double>(up[i]));
      }
      MatVec(layer.down_proj, gate, projected);
      for (std::size_t i = 0; i < hidden; ++i)
        row[i] += projected[i];
    }
  }
  std::vector<float> last(hidden);
  RmsNorm({x.data() + (n - 1) * hidden, hidden}, weights.norm, c.rms_norm_eps,
          last);
  return last;
}

double ScoreTokens(const Weights& weights,
                   std::span<const std::uint32_t> tokens, std::uint32_t yes,
                   std::uint32_t no) {
  const Config& c = weights.config;
  if (yes >= c.vocab_size || no >= c.vocab_size)
    throw std::out_of_range("answer token is outside the vocabulary");
  const std::vector<float> last = LastHiddenState(weights, tokens);
  const std::vector<float>& head =
      c.tie_word_embeddings ? weights.embed_tokens : weights.lm_head;
  const std::size_t hidden = c.hidden_size;
  const double yes_logit = Dot(last, {head.data() + yes * hidden, hidden});
  const double no_logit = Dot(last, {head.data() + no * hidden, hidden});
  // softmax over the two logits, in the form that cannot overflow.
  return 1.0 / (1.0 + std::exp(no_logit - yes_logit));
}

}  // namespace gufo::models::qwen3_reranker::reference
