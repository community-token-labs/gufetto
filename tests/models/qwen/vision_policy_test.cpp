#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "src/models/qwen/vision/encoder.hpp"
#include "src/models/qwen/vision/policy.hpp"

int main() {
  using namespace gufo::models::qwen::vision;
  const auto qwen = ResolvePolicy(5120);
  assert(qwen.enabled && !qwen.pool_rows && !qwen.ProtectedExecution());
  assert(!qwen.PromptLimit() && !qwen.CountLimit(32));
  assert(qwen.PositionClaimBytes(262144) == 0);
  assert(qwen.SharedClaimBytes(262144, 5120) == 0);
  // Preserve even the old underclaims and raw-HIP admission semantics.
  assert(qwen.StateCapacityBytes(100, 1, 1000) == 100);
  assert(!qwen.StateCapacityBytes(std::nullopt, 100, 0));

  const auto flash = ResolvePolicy(2560);
  assert(flash.enabled && flash.pool_rows == 65536 &&
         flash.ProtectedExecution());
  assert(flash.PromptLimit() == 0 && flash.CountLimit(32) == 32);
  assert(flash.PositionClaimBytes(262144) == 3145792);
  assert(flash.SharedClaimBytes(262144, 2560) == 4142465024ULL);
  assert(flash.StateCapacityBytes(100, 60, 20) == 60);
  assert(flash.StateCapacityBytes(10, 60, 20) == 0);

  for (const auto width : {2560U, 5120U}) {
    for (const auto rows : {1U, 128U, 65536U, UINT32_MAX}) {
      const auto explicit_pool = ResolvePolicy(width, rows);
      assert(explicit_pool.pool_rows == rows &&
             explicit_pool.ProtectedExecution());
      assert(explicit_pool.PromptLimit() == 0);
      assert(explicit_pool.CountLimit(32) == 32);
    }
    for (const bool enabled : {false, true}) {
      bool rejected = false;
      try {
        (void)ResolvePolicy(width, 0, 0, enabled);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      assert(rejected);  // Disabled vision never turns zero into "default".
      const auto disabled = ResolvePolicy(width, std::nullopt, 0, enabled);
      assert(disabled.enabled == enabled);
    }
    bool rejected = false;
    try {
      (void)ResolvePolicy(width, 128).StateCapacityBytes(std::nullopt, 100, 0);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    assert(rejected);
  }
  const auto cap = ResolvePolicy(5120, std::nullopt, 64);
  assert(!cap.pool_rows && !cap.ProtectedExecution());
  assert(cap.PromptLimit() == 64 && cap.CountLimit(32) == 32);
  assert(cap.CountLimit(128) == 64);
  assert(cap.PositionClaimBytes(128) == 0 &&
         cap.SharedClaimBytes(128, 5120) == 0);
  const auto pool_cap = ResolvePolicy(5120, 128, 64);
  assert(pool_cap.PromptLimit() == 64 && pool_cap.CountLimit(128) == 64);
  assert(pool_cap.SharedClaimBytes(128, 5120) ==
         EncoderWorkspaceBytes(128) + 128ULL * 5120 * sizeof(float));
  // Preserve source compatibility for old and explicit numeric direct callers.
  static_assert(requires(const std::filesystem::path& path) {
    Encoder(path, 5120);
    Encoder(path, 5120, 128);
    Encoder::Open(path, path, 5120);
    Encoder::Open(path, path, 5120, false);
    Encoder::Open(path, path, 5120, true, 128);
  });
}
