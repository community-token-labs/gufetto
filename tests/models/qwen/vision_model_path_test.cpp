#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include "src/cli/serve/inference_backend.hpp"
#include "src/models/qwen/vision/input_budget.hpp"
#include "src/models/qwen/vision/model_path.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "tests/support/temporary_directory.hpp"

// A CPU-only signature check: both arguments must be independently present.
static_assert(requires(gufo::server::InferenceBackend& backend,
                       std::string& error) {
  backend.load("missing.gguf", &error, 262144, 1, {}, {}, {}, {}, {}, {}, false,
               gufo::models::qwen38_flash_next::KvCacheType::kF16, false,
               16384U, 65536U);
});

int main() {
  using gufo::models::qwen::vision::ResolveModelPath;
  const gufo::models::qwen38_flash_next::ModelOptions options{
      .max_context = 262144,
      .enable_vision = false,
      .vision_max_tokens_per_slot = 16384,
      .vision_pool_tokens = 65536,
  };
  assert(!options.enable_vision && options.vision_max_tokens_per_slot == 16384);
  assert(options.vision_pool_tokens == 65536);
  assert(gufo::models::qwen::vision::OutputReservationBytes(65536, 2560) ==
         640ULL * 1024 * 1024);
  assert(gufo::models::qwen::vision::InputPositionBytes(262144) == 3145792);
  assert(gufo::models::qwen::vision::InputReservationBytes(
             options.max_context, 2560, options.vision_max_tokens_per_slot) ==
         170917952);
  const TemporaryDirectory directory("gufo-vision-model-path");
  const auto root = directory.path();
  const auto quant = root / "quant";
  std::filesystem::create_directory(quant);
  const auto model = quant / "model.gguf";
  const auto parent_sidecar = root / "mmproj-BF16.gguf";
  const auto local_sidecar = quant / "mmproj-BF16.gguf";
  assert(ResolveModelPath(model, {}).empty());
  assert(ResolveModelPath(model, {}, options.enable_vision).empty());
  std::ofstream(parent_sidecar) << "not a real projector";
  assert(ResolveModelPath(model, {}) == parent_sidecar);
  assert(ResolveModelPath(model, {}, false).empty());
  std::ofstream(local_sidecar) << "not a real projector";
  assert(ResolveModelPath(model, {}) == local_sidecar);
  assert(ResolveModelPath(model, {}, false).empty());
  assert(ResolveModelPath(model, parent_sidecar) == parent_sidecar);
  // Explicit paths retain normal load-time validation; disabled discovery
  // must not inspect/open either deliberately invalid projector above.
  const auto missing = root / "missing.gguf";
  assert(ResolveModelPath(model, missing) == missing);
  for (const auto& sidecar : {parent_sidecar, missing}) {
    bool rejected = false;
    try {
      (void)ResolveModelPath(model, sidecar, false);
    } catch (const std::invalid_argument& error) {
      rejected = std::string_view(error.what()) ==
                 "--no-vision cannot be combined with --mmproj";
    }
    assert(rejected);
  }
  // Discovery is limited to the model directory and its immediate parent.
  const auto deep = quant / "nested" / "deeper" / "model.gguf";
  assert(ResolveModelPath(deep, {}).empty());
}
