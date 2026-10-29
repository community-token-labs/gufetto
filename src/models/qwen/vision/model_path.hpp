#ifndef GUFO_MODELS_QWEN_VISION_MODEL_PATH_HPP_
#define GUFO_MODELS_QWEN_VISION_MODEL_PATH_HPP_

#include <filesystem>
#include <stdexcept>

namespace gufo::models::qwen::vision {

/// Resolve without opening a projector. Disabled vision never discovers one.
[[nodiscard]] inline std::filesystem::path ResolveModelPath(
    const std::filesystem::path& target, const std::filesystem::path& sidecar,
    bool enabled = true) {
  if (!enabled) {
    if (!sidecar.empty())
      throw std::invalid_argument(
          "--no-vision cannot be combined with --mmproj");
    return {};
  }
  if (!sidecar.empty())
    return sidecar;
  auto directory = target.parent_path();
  for (unsigned level = 0; level < 2; ++level) {
    const auto path = directory / "mmproj-BF16.gguf";
    if (std::filesystem::is_regular_file(path))
      return path;
    directory = directory.parent_path();
  }
  return {};
}

}  // namespace gufo::models::qwen::vision
#endif
