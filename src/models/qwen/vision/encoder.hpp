#ifndef GUFO_MODELS_QWEN_VISION_ENCODER_HPP_
#define GUFO_MODELS_QWEN_VISION_ENCODER_HPP_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/qwen/vision/output_pool.hpp"
#include "src/models/qwen/vision/policy.hpp"

namespace gufo::core {
class ReclaimableMemory;
}

namespace gufo::models::qwen::vision {

/// Shared Qwen3.8 ViT operator graph. Each model supplies and validates its
/// own projector/output width; language-model state never lives here.
class Encoder {
public:
  class Embedding {
  public:
    ~Embedding();
    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;
    [[nodiscard]] const float* data() const noexcept { return data_; }
    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }

  private:
    friend class Encoder;
    Embedding() = default;
    float* data_{nullptr};
    std::shared_ptr<core::ReclaimableMemory> memory_budget_;
    std::size_t bytes_{0};
    std::uint32_t rows_{0};
    std::uint32_t width_{0};
  };

  using Observer =
      std::function<void(std::string_view, std::span<const float>)>;
  using CancellationCheck = std::function<bool()>;
  Encoder(const std::filesystem::path& path, std::uint32_t output_width,
          std::optional<std::uint32_t> pool_rows = std::nullopt);
  /// An explicit sidecar is required to exist. Otherwise discover the canonical
  /// sidecar beside the target or its quantization subdirectory.
  [[nodiscard]] static std::shared_ptr<Encoder> Open(
      const std::filesystem::path& target, const std::filesystem::path& sidecar,
      std::uint32_t output_width, bool enabled = true,
      std::optional<std::uint32_t> pool_rows = std::nullopt);
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;
  [[nodiscard]] const std::string& identity() const;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Bind only while quiescent, before any image usage. Null preserves the
  /// native allocation path; serving admits image work before entering it.
  void SetMemoryBudget(std::shared_ptr<core::ReclaimableMemory> budget);
  [[nodiscard]] const std::shared_ptr<core::ReclaimableMemory>& MemoryBudget()
      const noexcept {
    return memory_budget_;
  }
  [[nodiscard]] const std::shared_ptr<OutputPool>& Outputs() const noexcept {
    return outputs_;
  }
  [[nodiscard]] std::size_t OutputBudgetBytes() const;
  [[nodiscard]] const Policy& GetPolicy() const noexcept { return policy_; }
  /// Upload the weights now instead of in the first Encode. Repeated calls
  /// keep the existing upload.
  void Upload();
  /// Pooled encoders serialize callers; legacy Qwen relies on its scheduler.
  /// Pixel grid is already resized.
  /// Uploads the weights if Upload has not; scratch is bounded by image
  /// dimensions.
  [[nodiscard]] std::shared_ptr<const Embedding> Encode(
      const core::Image& image, const Observer& observer = {},
      const CancellationCheck& is_cancelled = {});

private:
  struct Impl;
  Policy policy_;
  std::shared_ptr<core::ReclaimableMemory> memory_budget_;
  std::shared_ptr<OutputPool> outputs_;
  std::uint32_t output_width_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::qwen::vision
#endif
