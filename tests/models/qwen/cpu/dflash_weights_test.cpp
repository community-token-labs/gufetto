#include "src/models/qwen/dflash_weights.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "tests/support/gguf_builder.hpp"

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << "\n";
    std::exit(1);
  }
}

using Overrides = std::map<std::string, std::uint32_t>;

// Loads DFlash-2 metadata without tensors, so success through the config
// checks ends at the first required tensor.
std::string LoadDFlashMetadata(
    const Overrides& overrides,
    const std::vector<std::int32_t>& target_layers = {2, 17, 33}) {
  std::map<std::string, std::uint32_t> metadata = {
      {"dflash.embedding_length", 5120},
      {"dflash.feed_forward_length", 17408},
      {"dflash.attention.head_count", 32},
      {"dflash.attention.head_count_kv", 8},
      {"dflash.attention.key_length", 128},
      {"dflash.rope.dimension_count", 128},
      {"dflash.context_length", 32768},
      {"dflash.block_count", 5},
      {"dflash.block_size", 16},
      {"dflash.conv_kernel_size", 4},
      {"dflash.conv_group_size", 64},
      {"dflash.selector_rank", 64},
      {"dflash.selector_top_k", 8},
      {"dflash.attention.sliding_window", 2048},
      {"tokenizer.ggml.mask_token_id", 248077},
  };
  for (const auto& [key, value] : overrides) {
    metadata[key] = value;
  }
  GgufBuilder builder;
  builder.AddMetadataString("general.architecture", "dflash");
  for (const auto& [key, value] : metadata) {
    builder.AddMetadataUint32(key, value);
  }
  builder.AddMetadataFloat32("dflash.rope.freq_base", 10000000.0F);
  builder.AddMetadataBool("dflash.attention.causal", false);
  builder.AddMetadataInt32Array("dflash.target_layers", target_layers);
  const auto binary = builder.Build(64);
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(binary.data(), binary.size(), &error);
  Expect(reader != nullptr, "synthetic DFlash reader opens: " + error);
  Expect(!gufo::speculative::QwenDFlashWeights::LoadFromGguf(*reader, &error)
              .has_value(),
         "tensor-less DFlash metadata never loads");
  return error;
}

void ExpectRefused(const Overrides& overrides, std::string_view field,
                   std::string_view case_name,
                   const std::vector<std::int32_t>& target_layers = {2, 17,
                                                                     33}) {
  const std::string error = LoadDFlashMetadata(overrides, target_layers);
  Expect(error.find(field) != std::string::npos,
         std::string(case_name) + " is refused naming " + std::string(field) +
             ": " + error);
}

void TestValidMetadataReachesTensors() {
  const std::string error = LoadDFlashMetadata({});
  Expect(error.find("selector_predecessor.weight") != std::string::npos,
         "valid metadata passes every config check: " + error);
}

// Each divisor is refused before its division or modulo runs.
void TestZeroDivisors() {
  ExpectRefused({{"dflash.attention.head_count_kv", 0}}, "transformer topology",
                "zero head_count_kv (head modulo)");
  ExpectRefused({{"dflash.conv_group_size", 0}}, "diffusion topology",
                "zero conv_group_size (group divisor)");
  ExpectRefused({{"dflash.selector_rank", 0}}, "diffusion topology",
                "zero selector_rank (vocabulary divisor)");
}

// Each expected tensor size that wraps 32 or 64 bits is refused by name. The
// KV width cannot overflow alone: head_count_kv divides head_count.
void TestOverflowingWidths() {
  ExpectRefused({{"dflash.attention.head_count", 65537},
                 {"dflash.attention.head_count_kv", 1},
                 {"dflash.attention.key_length", 65536}},
                "head_count * key_length", "wrapped attention width");
  ExpectRefused({{"dflash.embedding_length", 1U << 31}}, "fc.weight",
                "fc.weight element count", {2, 17, 33, 40, 50});
  ExpectRefused({{"dflash.embedding_length", 0xFFFFFFFFU},
                 {"dflash.conv_group_size", 0xFFFFFFFFU},
                 {"dflash.conv_kernel_size", 0xFFFFFFFFU}},
                "conv_base", "conv_base element count", {2});
  ExpectRefused({{"dflash.embedding_length", 1U << 31},
                 {"dflash.conv_group_size", 1},
                 {"dflash.conv_kernel_size", 2}},
                "conv_proj.weight", "conv_proj.weight element count", {2});
}

}  // namespace

int main() {
  TestValidMetadataReachesTensors();
  TestZeroDivisors();
  TestOverflowingWidths();
  std::cout << "DFlash weights config tests passed\n";
  return 0;
}
