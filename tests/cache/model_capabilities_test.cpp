#include <array>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <string_view>

#include "tests/cache/model_capability_records.hpp"

namespace {
using gufo::cache::testing::Record;
constexpr auto& kRecords = gufo::cache::testing::kModelCapabilityRecords;

const Record* Find(std::string_view family) {
  for (const auto& record : kRecords)
    if (record.family == family)
      return &record;
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path models = argv[1];
  assert(std::filesystem::is_directory(models));

  std::set<std::string> families;
  for (const auto& entry : std::filesystem::directory_iterator(models))
    if (entry.is_directory())
      families.insert(entry.path().filename().string());

  bool complete = true;
  for (const auto& family : families) {
    if (Find(family) == nullptr) {
      std::cerr << "src/models/" << family
                << " has no cache capability record\n";
      complete = false;
    }
  }
  std::set<std::string_view> recorded;
  for (const auto& record : kRecords) {
    if (!recorded.insert(record.family).second ||
        !families.contains(std::string(record.family))) {
      std::cerr << record.family
                << " is duplicated or has no model directory\n";
      complete = false;
    }
    // Persistent encoding describes continuation state; it cannot exist alone.
    assert(record.capabilities.continuation ||
           !record.capabilities.persistent_encoding);
  }
  assert(complete);

  for (const auto family : {"embedding_gemma2", "qwen3_asr", "qwen3_reranker",
                            "qwen3_tts", "qwen_image_21", "minimax_h3"}) {
    assert(!Find(family)->capabilities.continuation);
    assert(!Find(family)->capabilities.persistent_encoding);
  }
  for (const auto family : {"qwen", "qwen38_flash_next", "deepseek_v4_flash"}) {
    assert(Find(family)->capabilities.continuation);
    assert(Find(family)->capabilities.persistent_encoding);
  }
}
