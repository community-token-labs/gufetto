#ifndef GUFO_TEST_MODEL_CAPABILITY_RECORDS_HPP_
#define GUFO_TEST_MODEL_CAPABILITY_RECORDS_HPP_

#include <array>
#include <string_view>

#include "src/models/deepseek_v4_flash/cache_capabilities.hpp"
#include "src/models/embedding_gemma2/cache_capabilities.hpp"
#include "src/models/minimax_h3/cache_capabilities.hpp"
#include "src/models/qwen/cache_capabilities.hpp"
#include "src/models/qwen36_moe/cache_capabilities.hpp"
#include "src/models/qwen38_flash_next/cache_capabilities.hpp"
#include "src/models/qwen3_asr/cache_capabilities.hpp"
#include "src/models/qwen3_tts/cache_capabilities.hpp"
#include "src/models/qwen_image_21/cache_capabilities.hpp"

namespace gufo::cache::testing {
struct Record {
  std::string_view family;
  gufo::cache::Capabilities capabilities;
};

// One record per src/models/ directory, keyed by the directory name.
inline constexpr std::array kModelCapabilityRecords{
    Record{"deepseek_v4_flash",
           gufo::models::deepseek_v4_flash::kCacheCapabilities},
    Record{"embedding_gemma2",
           gufo::models::embedding_gemma2::kCacheCapabilities},
    Record{"minimax_h3", gufo::minimax_h3::kCacheCapabilities},
    Record{"qwen", gufo::models::qwen::kCacheCapabilities},
    Record{"qwen36_moe", gufo::models::qwen36_moe::kCacheCapabilities},
    Record{"qwen38_flash_next",
           gufo::models::qwen38_flash_next::kCacheCapabilities},
    Record{"qwen3_asr", gufo::models::qwen3_asr::kCacheCapabilities},
    Record{"qwen3_tts", gufo::models::qwen3_tts::kCacheCapabilities},
    Record{"qwen_image_21", gufo::models::qwen_image_21::kCacheCapabilities},
};

}  // namespace gufo::cache::testing
#endif  // GUFO_TEST_MODEL_CAPABILITY_RECORDS_HPP_
