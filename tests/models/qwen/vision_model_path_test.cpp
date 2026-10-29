#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include "src/models/qwen/vision/model_path.hpp"
#include "tests/support/temporary_directory.hpp"

int main() {
  using gufo::models::qwen::vision::ResolveModelPath;
  const TemporaryDirectory directory("gufo-vision-model-path");
  const auto root = directory.path();
  const auto quant = root / "quant";
  std::filesystem::create_directory(quant);
  const auto model = quant / "model.gguf";
  const auto parent_sidecar = root / "mmproj-BF16.gguf";
  const auto local_sidecar = quant / "mmproj-BF16.gguf";
  assert(ResolveModelPath(model, {}).empty());
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
