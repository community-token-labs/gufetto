#include "src/models/qwen3_reranker/config.hpp"

#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace gufo::models::qwen3_reranker {
namespace {

/// Reads a required positive integer member.
bool ReadCount(const json::Value& root, const char* name, std::uint32_t* out,
               std::string* error) {
  const json::Value* value = root.find(name);
  constexpr std::size_t kInvalid = 0;
  const std::size_t parsed =
      value != nullptr ? value->as_size(kInvalid) : kInvalid;
  if (parsed == kInvalid ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    *error =
        std::string("config.json '") + name + "' must be a positive integer";
    return false;
  }
  *out = static_cast<std::uint32_t>(parsed);
  return true;
}

bool ReadPositive(const json::Value& root, const char* name, double* out,
                  std::string* error) {
  const json::Value* value = root.find(name);
  if (value == nullptr || !value->is_number() || !(value->as_double() > 0.0) ||
      !std::isfinite(value->as_double())) {
    *error = std::string("config.json '") + name + "' must be positive";
    return false;
  }
  *out = value->as_double();
  return true;
}

/// An optional member that must be absent, null, or `expected`.
bool RequireFlag(const json::Value& root, const char* name, bool expected,
                 std::string* error) {
  const json::Value* value = root.find(name);
  if (value == nullptr || value->is_null())
    return true;
  if (value->is_bool() && value->as_bool() == expected)
    return true;
  *error = std::string("unsupported config.json '") + name + "'";
  return false;
}

}  // namespace

bool ParseConfig(const json::Value& root, Config* config, std::string* error) {
  std::string scratch;
  if (error == nullptr)
    error = &scratch;
  if (!root.is_object()) {
    *error = "config.json must be an object";
    return false;
  }
  const json::Value* architectures = root.find("architectures");
  if (architectures == nullptr || !architectures->is_array() ||
      architectures->size() != 1 ||
      architectures->items()[0].str() != "Qwen3ForCausalLM" ||
      root.member_str("model_type") != "qwen3") {
    *error = "config.json must describe a qwen3 Qwen3ForCausalLM checkpoint";
    return false;
  }
  if (root.member_str("hidden_act") != "silu") {
    *error = "config.json 'hidden_act' must be silu";
    return false;
  }
  if (const json::Value* scaling = root.find("rope_scaling");
      scaling != nullptr && !scaling->is_null()) {
    *error = "config.json 'rope_scaling' is not supported";
    return false;
  }
  if (!RequireFlag(root, "attention_bias", false, error) ||
      !RequireFlag(root, "use_sliding_window", false, error))
    return false;

  Config parsed;
  double eps = 0.0;
  if (!ReadCount(root, "vocab_size", &parsed.vocab_size, error) ||
      !ReadCount(root, "hidden_size", &parsed.hidden_size, error) ||
      !ReadCount(root, "intermediate_size", &parsed.intermediate_size, error) ||
      !ReadCount(root, "num_hidden_layers", &parsed.num_hidden_layers, error) ||
      !ReadCount(root, "num_attention_heads", &parsed.num_attention_heads,
                 error) ||
      !ReadCount(root, "num_key_value_heads", &parsed.num_key_value_heads,
                 error) ||
      !ReadCount(root, "head_dim", &parsed.head_dim, error) ||
      !ReadCount(root, "max_position_embeddings",
                 &parsed.max_position_embeddings, error) ||
      !ReadPositive(root, "rms_norm_eps", &eps, error) ||
      !ReadPositive(root, "rope_theta", &parsed.rope_theta, error))
    return false;
  parsed.rms_norm_eps = static_cast<float>(eps);
  if (parsed.num_attention_heads % parsed.num_key_value_heads != 0) {
    *error =
        "config.json attention heads must be a multiple of key/value heads";
    return false;
  }
  if (parsed.head_dim % 2 != 0) {
    *error = "config.json 'head_dim' must be even for RoPE";
    return false;
  }
  const json::Value* tied = root.find("tie_word_embeddings");
  if (tied == nullptr || !tied->is_bool()) {
    *error = "config.json 'tie_word_embeddings' must be a boolean";
    return false;
  }
  parsed.tie_word_embeddings = tied->as_bool();
  parsed.torch_dtype = root.member_str("torch_dtype", parsed.torch_dtype);
  *config = parsed;
  return true;
}

bool LoadConfig(const std::filesystem::path& model_dir, Config* config,
                std::string* error) {
  const auto path = model_dir / "config.json";
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    if (error != nullptr)
      *error = "cannot read " + path.string();
    return false;
  }
  std::ostringstream text;
  text << input.rdbuf();
  json::Value root;
  try {
    root = json::parse(text.str());
  } catch (const std::exception& parse_error) {
    if (error != nullptr)
      *error = path.string() + ": " + parse_error.what();
    return false;
  }
  return ParseConfig(root, config, error);
}

}  // namespace gufo::models::qwen3_reranker
