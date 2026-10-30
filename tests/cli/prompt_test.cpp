#include "src/cli/prompt/prompt.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/kv_cache_options.hpp"
#include "src/core/gguf_reader.hpp"
#include "tests/support/temporary_directory.hpp"

void TestDefaultOptions() {
  const std::array<const char*, 2> args = {"Hello", "world"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->prompt_text == "Hello world");
  assert(opt->max_tokens == 128);
  assert(opt->sampling.temperature == 0.0F);
  assert(!opt->sampling_supplied.temperature && !opt->sampling_supplied.top_k);
  assert(opt->use_chat_template);
  assert(opt->system_prompt.empty());
  assert(opt->reasoning_mode == "auto");
  assert(opt->reasoning_effort == "auto");
  assert(opt->preserve_thinking == "auto");
  assert(!opt->verbose);
  assert(opt->draft_tokens == 7);
  assert(opt->min_draft_tokens == 1);
}

void TestKvCacheType() {
  using gufo::models::qwen38_flash_next::KvCacheType;
  const auto defaults = gufo::cli::ParsePromptOptions(std::array{"Hello"});
  assert(defaults && defaults->kv_cache_type == KvCacheType::kF16);
  for (const auto value : {"f16", "q8_0"}) {
    const auto parsed = gufo::cli::ParsePromptOptions(
        std::array{"--kv-cache-type", value, "Hello"});
    assert(parsed && parsed->kv_cache_type == (std::string_view(value) == "f16"
                                                   ? KvCacheType::kF16
                                                   : KvCacheType::kQ8_0));
  }
  for (const auto type :
       {KvCacheType::kQ4_0Rotated, KvCacheType::kQ8_0Q4_0Rotated}) {
    const auto* value = gufo::models::qwen38_flash_next::KvCacheTypeName(type);
    const auto parsed = gufo::cli::ParsePromptOptions(
        std::array{"--kv-cache-type", value, "Hello"});
    assert(parsed && parsed->kv_cache_type == type);
    assert(!gufo::cli::ParsePromptOptions(
        std::array{"--kv-cache-type", value, "--cpu"}));
    assert(gufo::cli::ValidateKvCacheTypeForArchitecture(type, "qwen4exp",
                                                         nullptr));
    assert(
        !gufo::cli::ValidateKvCacheTypeForArchitecture(type, "qwen3", nullptr));
  }
  for (const auto type : {KvCacheType::kQ4_0, KvCacheType::kQ8_0Q4_0}) {
    const auto* value = gufo::models::qwen38_flash_next::KvCacheTypeName(type);
    std::string error;
    assert(!gufo::cli::ParsePromptOptions(
        std::array{"--kv-cache-type", value, "Hello"}, &error));
    assert(error ==
           "plain q4 KV modes failed the quality screen; use "
           "q4_0-rotated or q8_0/q4_0-rotated");
    error.clear();
    assert(!gufo::cli::ValidateKvCacheTypeForArchitecture(type, "qwen4exp",
                                                          &error));
    assert(error.find("failed the quality screen") != std::string::npos);
  }
  for (const auto value : {"", "q8", "Q8_0", "bf16", "q4", "q4_0/q8_0"}) {
    std::string error;
    assert(!gufo::cli::ParsePromptOptions(std::array{"--kv-cache-type", value},
                                          &error));
    assert(error ==
           "--kv-cache-type must be f16, q8_0, "
           "q8_0-rotated, q4_0-rotated or q8_0/q4_0-rotated");
  }
  assert(!gufo::cli::ParsePromptOptions(
      std::array{"--kv-cache-type", "q8_0", "--cpu"}));
  const auto rotated = gufo::cli::ParsePromptOptions(
      std::array{"--kv-cache-type", "q8_0-rotated", "Hello"});
  assert(rotated && rotated->kv_cache_type == KvCacheType::kQ8_0Rotated);
  std::string rotated_error;
  assert(!gufo::cli::ParsePromptOptions(
      std::array{"--kv-cache-type", "q8_0-rotated", "--cpu"}, &rotated_error));
  assert(rotated_error ==
         "--kv-cache-type q8_0-rotated requires the ROCm backend");
  assert(gufo::cli::ValidateKvCacheTypeForArchitecture(
      KvCacheType::kQ8_0Rotated, "qwen4exp", nullptr));
  assert(!gufo::cli::ValidateKvCacheTypeForArchitecture(
      KvCacheType::kQ8_0Rotated, "qwen3next", nullptr));
  assert(!gufo::cli::ValidateKvCacheTypeForArchitecture(
      static_cast<KvCacheType>(99), "qwen4exp", nullptr));
  std::string error;
  assert(gufo::cli::ValidateKvCacheTypeForArchitecture(KvCacheType::kQ8_0,
                                                       "qwen4exp", &error));
  for (const auto architecture :
       {"qwen2", "qwen3", "qwen3next", "deepseek4", ""}) {
    assert(!gufo::cli::ValidateKvCacheTypeForArchitecture(
        KvCacheType::kQ8_0, architecture, &error));
    assert(error.find("requires Qwen3.8-Flash-Next") != std::string::npos);
    assert(gufo::cli::ValidateKvCacheTypeForArchitecture(KvCacheType::kF16,
                                                         architecture, &error));
  }
}

void TestExplicitFlags() {
  const std::array<const char*, 10> args = {
      "--model", "model.gguf", "-n", "256",  "--temperature",
      "0.7",     "--raw",      "-v", "Test", "prompt"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->model_path == "model.gguf");
  assert(opt->max_tokens == 256);
  assert(opt->sampling.temperature > 0.69F &&
         opt->sampling.temperature < 0.71F);
  assert(opt->sampling_supplied.temperature && !opt->sampling_supplied.top_p);
  assert(!opt->use_chat_template);
  assert(opt->verbose);
  assert(opt->prompt_text == "Test prompt");
  const char* neutral[] = {"--temperature=0", "--top-k=0",
                           "--presence-penalty=0", "-s=0", "Hello"};
  const auto zero = gufo::cli::ParsePromptOptions(neutral);
  assert(zero);
  const auto sampling = gufo::sampling::ResolveTextSampling(
      gufo::sampling::TextModelPreset::kQwen38, false, zero->sampling,
      zero->sampling_supplied);
  assert(sampling.temperature == 0 && sampling.top_k == 0 &&
         sampling.presence_penalty == 0 && sampling.seed == 0);
  assert(sampling.top_p == 0.8F);
}

void TestFlashMtpFlags() {
  const char* args[] = {"--speculative",  "mtp", "--mtp-model",   "mtp.gguf",
                        "--draft-tokens", "3",   "--temperature", "0.7",
                        "--seed",         "1",   "Prompt"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->speculative_backend == "mtp");
  assert(opt->draft_tokens == 3);
  assert(opt->sampling.temperature == 0.7F && opt->sampling.seed == 1);
}

void TestImageFlags() {
  const char* args[] = {"--image", "a.png",           "--image",
                        "b.jpg",   "--add-vision-id", "Compare"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt && opt->add_vision_id);
  assert((opt->image_paths == std::vector<std::string>{"a.png", "b.jpg"}));
  assert(!gufo::cli::ParsePromptOptions(std::array{"Compare"})->add_vision_id);
  std::vector<const char*> many;
  for (unsigned i = 0; i < 17; ++i)
    many.insert(many.end(), {"--image", "a.png"});
  many.push_back("Compare");
  const auto parsed = gufo::cli::ParsePromptOptions(many);
  assert(parsed && parsed->image_paths.size() == 17);
  assert(!gufo::cli::ParsePromptOptions(std::array{"--image", ""}));
}

void TestInvalidFlags() {
  std::string err;
  const std::array<const char*, 1> args1 = {"--model"};
  assert(!gufo::cli::ParsePromptOptions(args1, &err).has_value());
  assert(!err.empty());

  const std::array<const char*, 2> args2 = {"-n", "not_a_number"};
  assert(!gufo::cli::ParsePromptOptions(args2, &err).has_value());

  const std::array<const char*, 2> args3 = {"-t", "invalid_float"};
  assert(!gufo::cli::ParsePromptOptions(args3, &err).has_value());

  const std::array<const char*, 4> args5 = {"--draft-tokens", "3",
                                            "--min-draft-tokens", "4"};
  assert(!gufo::cli::ParsePromptOptions(args5, &err).has_value());

  const std::array<const char*, 2> args6 = {"--top-p", "-0.01"};
  assert(!gufo::cli::ParsePromptOptions(args6, &err).has_value());
  const auto zero_top_p =
      gufo::cli::ParsePromptOptions(std::array{"--top-p", "0"}, &err);
  assert(zero_top_p && zero_top_p->sampling.top_p == 0.0F);

  const std::array<const char*, 6> unsupported_floor = {
      "--speculative",      "dflash2", "--dflash-model", "draft.gguf",
      "--min-draft-tokens", "2"};
  assert(!gufo::cli::ParsePromptOptions(unsupported_floor, &err).has_value());
  assert(err.find("min-draft-tokens") != std::string::npos);
  for (const char* policy : {"fixed", "adaptive", "unknown"}) {
    const std::array<const char*, 6> args = {"--speculative",  "dflash2",
                                             "--dflash-model", "draft.gguf",
                                             "--draft-policy", policy};
    const auto parsed = gufo::cli::ParsePromptOptions(args, &err);
    assert(parsed.has_value() == (std::string_view(policy) != "unknown"));
    if (parsed)
      assert(parsed->draft_policy == policy);
  }
  const std::array<const char*, 2> policy_without_backend = {"--draft-policy",
                                                             "adaptive"};
  assert(!gufo::cli::ParsePromptOptions(policy_without_backend, &err));

  const std::array<const char*, 2> args8 = {"--chat-template", "qwen"};
  assert(!gufo::cli::ParsePromptOptions(args8, &err).has_value());

  const std::array<const char*, 2> args9 = {"--reasoning-budget", "1024"};
  assert(!gufo::cli::ParsePromptOptions(args9, &err).has_value());

  for (const auto* backend : {"dflash2", "mtp"}) {
    const std::array<const char*, 2> missing_path = {"--speculative", backend};
    assert(!gufo::cli::ParsePromptOptions(missing_path, &err).has_value());
    assert(err.find("requires --") != std::string::npos);
  }
  const std::array<const char*, 5> cpu_spec = {
      "--cpu", "--speculative", "dflash2", "--dflash-model", "draft.gguf"};
  assert(!gufo::cli::ParsePromptOptions(cpu_spec, &err).has_value());
  assert(err.find("ROCm") != std::string::npos);
  const std::array<const char*, 3> cpu_ar = {"--cpu", "--speculative", "off"};
  assert(gufo::cli::ParsePromptOptions(cpu_ar, &err).has_value());
}

void TestSamplingAndReasoningFlags() {
  const std::array<const char*, 29> args = {"--model",
                                            "model.gguf",
                                            "--top-p",
                                            "0.9",
                                            "--top-k",
                                            "50",
                                            "--min-p",
                                            "0.05",
                                            "--min-keep",
                                            "3",
                                            "-s",
                                            "42",
                                            "--repeat-penalty",
                                            "1.15",
                                            "--repeat-last-n",
                                            "128",
                                            "--frequency-penalty",
                                            "0.25",
                                            "--presence-penalty",
                                            "0.5",
                                            "--think",
                                            "on",
                                            "--reasoning-effort",
                                            "high",
                                            "--preserve-thinking",
                                            "off",
                                            "--no-display-prompt",
                                            "-p",
                                            "Explicit prompt text"};
  const auto opt = gufo::cli::ParsePromptOptions(args);
  assert(opt.has_value());
  assert(opt->model_path == "model.gguf");
  assert(opt->sampling.top_p > 0.89F && opt->sampling.top_p < 0.91F);
  assert(opt->sampling.top_k == 50);
  assert(opt->sampling.min_p > 0.04F && opt->sampling.min_p < 0.06F);
  assert(opt->sampling.min_keep == 3);
  assert(opt->sampling.seed == 42);
  assert(opt->sampling.repeat_penalty > 1.14F &&
         opt->sampling.repeat_penalty < 1.16F);
  assert(opt->sampling.repeat_last_n == 128);
  assert(opt->sampling.frequency_penalty > 0.24F &&
         opt->sampling.frequency_penalty < 0.26F);
  assert(opt->sampling.presence_penalty > 0.49F &&
         opt->sampling.presence_penalty < 0.51F);
  const auto resolved = gufo::sampling::ResolveTextSampling(
      gufo::sampling::TextModelPreset::kQwen38, false, opt->sampling,
      opt->sampling_supplied);
  assert(resolved.temperature == 0.7F);
  assert(resolved.top_p == opt->sampling.top_p &&
         resolved.top_k == opt->sampling.top_k &&
         resolved.min_p == opt->sampling.min_p &&
         resolved.min_keep == opt->sampling.min_keep &&
         resolved.seed == opt->sampling.seed &&
         resolved.repeat_penalty == opt->sampling.repeat_penalty &&
         resolved.repeat_last_n == opt->sampling.repeat_last_n &&
         resolved.frequency_penalty == opt->sampling.frequency_penalty &&
         resolved.presence_penalty == opt->sampling.presence_penalty);
  assert(opt->reasoning_mode == "on");
  assert(opt->reasoning_effort == "high");
  assert(opt->preserve_thinking == "off");
  assert(!opt->display_prompt);
  assert(opt->prompt_text == "Explicit prompt text");
}

// Minimal GGUF v3 writer: string/u32 metadata and one BF16 embedding table.
class ArchitectureGguf {
public:
  void String(std::string_view key, std::string_view value) {
    Key(key, 8);
    Text(value);
  }
  void U32(std::string_view key, std::uint32_t value) {
    Key(key, 4);
    Pod(value);
  }
  std::string Build(std::uint64_t embedding_rows) const {
    std::string out = "GGUF";
    const auto pod = [&](auto value) {
      out.append(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    pod(std::uint32_t{3});
    pod(std::uint64_t{1});
    pod(count_);
    out += metadata_;
    const std::string_view name = "token_embd.weight";
    pod(static_cast<std::uint64_t>(name.size()));
    out += name;
    pod(std::uint32_t{2});
    pod(std::uint64_t{64});
    pod(embedding_rows);
    pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kBF16));
    pod(std::uint64_t{0});
    out.resize((out.size() + 31) / 32 * 32, '\0');
    out.append(64 * embedding_rows * 2, '\0');
    return out;
  }

private:
  template<typename T>
  void Pod(T value) {
    metadata_.append(reinterpret_cast<const char*>(&value), sizeof(value));
  }
  void Text(std::string_view value) {
    Pod(static_cast<std::uint64_t>(value.size()));
    metadata_ += value;
  }
  void Key(std::string_view key, std::uint32_t type) {
    Text(key);
    Pod(type);
    ++count_;
  }
  std::string metadata_;
  std::uint64_t count_{0};
};

struct PromptRun {
  int status;
  std::string error;
};

// Quantized KV is a GPU-only option, so those runs omit --cpu; the KV gate
// rejects them before any device or weight access.
PromptRun RunPromptOn(const std::filesystem::path& path,
                      const char* kv_cache_type = nullptr) {
  const auto model = path.string();
  std::vector<const char*> args = {"--model", model.c_str(), "-n", "1"};
  if (kv_cache_type)
    args.insert(args.end(), {"--kv-cache-type", kv_cache_type});
  else
    args.push_back("--cpu");
  args.push_back("hi");
  std::ostringstream error;
  std::ostringstream output;
  auto* const old_error = std::cerr.rdbuf(error.rdbuf());
  auto* const old_output = std::cout.rdbuf(output.rdbuf());
  const int status = gufo::cli::RunPrompt(args);
  std::cerr.rdbuf(old_error);
  std::cout.rdbuf(old_output);
  return {status, error.str()};
}

// Dispatch on general.architecture through the real prompt loader. Hybrid
// Qwen hyperparameters are read under the declared architecture's prefix, so
// a missing or renamed architecture must not borrow another family's keys.
void TestArchitectureDispatch() {
  TemporaryDirectory directory("gufo-prompt-arch");
  const auto write = [&](std::optional<std::string_view> architecture,
                         std::string_view prefix) {
    ArchitectureGguf gguf;
    if (architecture)
      gguf.String("general.architecture", *architecture);
    const std::string p(prefix);
    for (const auto& [key, value] :
         std::initializer_list<std::pair<std::string_view, std::uint32_t>>{
             {"block_count", 4},
             {"embedding_length", 64},
             {"feed_forward_length", 64},
             {"attention.head_count", 1},
             {"attention.head_count_kv", 1},
             {"attention.key_length", 64},
             {"full_attention_interval", 4},
             {"rope.dimension_count", 64},
             {"ssm.conv_kernel", 4},
             {"ssm.state_size", 128},
             {"ssm.group_count", 1},
             {"ssm.time_step_rank", 1},
             {"ssm.inner_size", 128}})
      gguf.U32(p + "." + std::string(key), value);
    const auto path =
        directory.path() /
        (std::string(architecture.value_or("absent")) + "-" + p + ".gguf");
    std::ofstream(path, std::ios::binary) << gguf.Build(2);
    return path;
  };
  constexpr std::string_view kQwenLoad = "Error creating Qwen generator: ";

  // The supported hybrid Qwen family passes the architecture and structure
  // checks and reaches tensor validation.
  const auto qwen = RunPromptOn(write("qwen35", "qwen35"));
  assert(qwen.status == 1);
  assert(qwen.error.starts_with(
      std::string(kQwenLoad) +
      "Missing required Qwen tensor: output_norm.weight\n"));

  // Missing, empty, unknown and renamed architectures are refused by name
  // before weights, never by adopting qwen35.* metadata or built-in defaults.
  for (const auto& [architecture, refusal] : std::initializer_list<
           std::pair<std::optional<std::string_view>, std::string_view>>{
           {std::nullopt, "GGUF has no general.architecture"},
           {"", "GGUF has no general.architecture"},
           {"unknown", "unsupported GGUF architecture 'unknown'"},
           {"llama", "unsupported GGUF architecture 'llama'"}}) {
    const auto run = RunPromptOn(write(architecture, "qwen35"));
    assert(run.status == 1);
    assert(run.error.starts_with(std::string(kQwenLoad) + std::string(refusal) +
                                 "\n"));
  }
  // Qwen3.6 MoE has its own loader; its refusal names the architecture.
  const auto moe = RunPromptOn(write("qwen35moe", "qwen35moe"));
  assert(moe.status == 1 && moe.error.starts_with(kQwenLoad) &&
         moe.error.find("qwen35moe") != std::string::npos);

  // Quantized KV is admitted only for the Flash-Next architecture.
  for (const auto architecture : {"qwen35", "unknown", "deepseek4"}) {
    const auto run = RunPromptOn(write(architecture, architecture), "q8_0");
    assert(run.status == 1);
    assert(run.error ==
           "Error: quantized --kv-cache-type requires Qwen3.8-Flash-Next "
           "(GGUF architecture qwen4exp)\n");
  }

  // DeepSeek and Flash-Next route to their own loaders, never to Qwen.
  // A failed load also reports its load time on the error stream.
  const auto deepseek = RunPromptOn(write("deepseek4", "deepseek4"));
  assert(deepseek.status == 1);
#if defined(ENGINE_ENABLE_HIP)
  assert(deepseek.error.starts_with(
      "DeepSeek V4 Flash is supported only by the ROCm backend\n"));
#else
  assert(deepseek.error.starts_with(
      "DeepSeek V4 Flash requires ENGINE_ENABLE_HIP=ON\n"));
#endif
  const auto flash_path = write("qwen4exp", "qwen4exp");
  const auto flash = RunPromptOn(flash_path);
  assert(flash.status == 1);
#if defined(ENGINE_ENABLE_HIP)
  assert(flash.error ==
         "Flash-Next requires ROCm and supports MTP with "
         "--min-draft-tokens 1\n");
#else
  // CPU builds have no Flash-Next runtime: it passes the quantized-KV gate
  // and the architecture check, then fails on this fixture's tensors.
  for (const auto& run : {flash, RunPromptOn(flash_path, "q8_0")})
    assert(run.status == 1 &&
           run.error.starts_with(
               std::string(kQwenLoad) +
               "Missing required Qwen tensor: output_norm.weight\n"));
#endif
}

int main() {
  TestDefaultOptions();
  TestKvCacheType();
  TestExplicitFlags();
  TestSamplingAndReasoningFlags();
  TestFlashMtpFlags();
  TestImageFlags();
  TestInvalidFlags();
  TestArchitectureDispatch();
  std::cout << "All prompt CLI tests passed.\n";
  return 0;
}
