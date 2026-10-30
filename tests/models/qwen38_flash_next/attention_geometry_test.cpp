#include "src/models/qwen38_flash_next/attention_geometry.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/models/qwen38_flash_next/config.hpp"

namespace {

using namespace gufo::models::qwen38_flash_next;

constexpr char kExpectedAttention[] =
    "unsupported Flash-Next attention geometry: requires head_dim=256, "
    "positive compression ratio and Q/KV head grouping divisible by 4";
constexpr char kExpectedIndexer[] =
    "unsupported Flash-Next indexer geometry: requires 4 heads, dimension=128 "
    "and positive compression ratio";

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct ModelProxy {
  Config value;
  const Config& config() const { return value; }
};

void AssignError(std::string* error, const char* message) {
  if (error != nullptr)
    *error = message;
}

int create_fallthrough = 0;
int attention_fallthrough = 0;
int indexer_fallthrough = 0;

#include "attention_geometry_guards.inc"

ModelProxy Preview() {
  ModelProxy model;
  model.value.num_heads = 24;
  model.value.num_kv_heads = 2;
  model.value.head_dim = 256;
  model.value.compress_ratio = 4;
  model.value.indexer_heads = 4;
  model.value.indexer_head_dim = 128;
  return model;
}

template<typename Call>
void Throws(Call call, const char* expected) {
  try {
    call();
  } catch (const std::invalid_argument& error) {
    Require(std::string(error.what()) == expected, "wrong dispatch error");
    return;
  }
  throw std::runtime_error("unsupported geometry silently dispatched");
}

void CheckAttentionErrors() {
  for (const auto shape :
       std::array<std::array<unsigned, 4>, 8>{{{24, 2, 128, 4},
                                               {24, 4, 256, 4},
                                               {8, 0, 256, 4},
                                               {0, 2, 256, 4},
                                               {25, 2, 256, 4},
                                               {4, 4, 256, 4},
                                               {24, 2, 256, 0},
                                               {UINT32_MAX, 1, 256, 4}}}) {
    auto model = Preview();
    model.value.num_heads = shape[0];
    model.value.num_kv_heads = shape[1];
    model.value.head_dim = shape[2];
    model.value.compress_ratio = shape[3];
    std::string error;
    Require(CreateGuard(model, &error) == nullptr,
            "invalid create reached resource setup");
    Require(error == kExpectedAttention, "wrong create attention error");
    Require(CreateGuard(model, nullptr) == nullptr, "null error path failed");
    Throws([&] { AttentionGuard(shape[0], shape[1], shape[2], shape[3]); },
           kExpectedAttention);
  }
  Require(create_fallthrough == 0 && attention_fallthrough == 0,
          "bad attention reached downstream work");
}

void CheckIndexerErrors() {
  for (const auto shape : std::array<std::array<unsigned, 3>, 5>{
           {{8, 128, 4}, {4, 256, 4}, {0, 128, 4}, {4, 0, 4}, {4, 128, 0}}}) {
    Throws([&] { IndexerGuard(shape[0], shape[1], shape[2]); },
           kExpectedIndexer);
    // Zero ratio is already refused by the earlier attention guard.
    if (shape[2] == 0)
      continue;
    auto model = Preview();
    model.value.indexer_heads = shape[0];
    model.value.indexer_head_dim = shape[1];
    std::string error;
    Require(CreateGuard(model, &error) == nullptr,
            "invalid indexer reached resource setup");
    Require(error == kExpectedIndexer, "wrong create indexer error");
  }
  Require(create_fallthrough == 0 && indexer_fallthrough == 0,
          "bad indexer reached downstream work");
}

void CheckSupportedFallthrough() {
  // Kernel-contract shapes, not permission to load new model profiles.
  for (const auto shape :
       std::array<std::array<unsigned, 4>, 4>{{{24, 2, 256, 4},
                                               {16, 4, 256, 4},
                                               {8, 2, 256, 1},
                                               {24, 2, 256, 8}}}) {
    auto model = Preview();
    model.value.num_heads = shape[0];
    model.value.num_kv_heads = shape[1];
    model.value.head_dim = shape[2];
    model.value.compress_ratio = shape[3];
    std::string error;
    Require(CreateGuard(model, &error) != nullptr && error.empty(),
            "supported create rejected");
    AttentionGuard(shape[0], shape[1], shape[2], shape[3]);
    IndexerGuard(4, 128, shape[3]);
  }
  Require(create_fallthrough == 4 && attention_fallthrough == 4 &&
              indexer_fallthrough == 4,
          "supported dispatch changed");
}

}  // namespace

int main() {
  CheckAttentionErrors();
  CheckIndexerErrors();
  CheckSupportedFallthrough();
  std::cout << "Flash-Next create/dispatch geometry guards passed.\n";
}
