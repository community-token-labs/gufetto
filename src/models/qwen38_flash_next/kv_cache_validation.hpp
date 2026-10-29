#pragma once

#include <cstddef>
#include <span>

namespace gufo::models::qwen38_flash_next::detail {
// Internal: ValidateKvRow has already checked the complete row geometry.
void ValidateQ8Rows(std::span<const std::byte> input);
void ValidateQ4Rows(std::span<const std::byte> input);
}  // namespace gufo::models::qwen38_flash_next::detail
