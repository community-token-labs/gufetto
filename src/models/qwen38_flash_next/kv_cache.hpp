#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace gufo::models::qwen38_flash_next {

enum class KvCacheType : std::uint32_t {
  kF16 = 0,
  kQ8_0 = 1,
  kQ8_0Rotated = 2,
  kQ4_0 = 3,
  kQ8_0Q4_0 = 4,
  kQ4_0Rotated = 5,
  kQ8_0Q4_0Rotated = 6
};

// Snapshot and persistence arithmetic versions. Version 18 predates packed KV;
// F16/plain and rotated representations have distinct compatibility identities.
inline constexpr std::uint32_t kKvCacheNumericsEpoch = 19;
inline constexpr std::uint32_t kKvQ8BlockElements = 32;

// Equal row widths do not make rotated and unrotated state interchangeable.
inline constexpr std::uint32_t kKvRotatedNumericsEpoch = 20;
static_assert(kKvRotatedNumericsEpoch > kKvCacheNumericsEpoch);
inline constexpr std::uint32_t kKvQ4NumericsEpoch = 24;
[[nodiscard]] constexpr bool IsKnownKvCacheType(KvCacheType type) noexcept {
  switch (type) {
    case KvCacheType::kF16:
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ8_0Rotated:
    case KvCacheType::kQ4_0:
    case KvCacheType::kQ8_0Q4_0:
    case KvCacheType::kQ4_0Rotated:
    case KvCacheType::kQ8_0Q4_0Rotated:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool KvUsesQ8Codec(KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
    case KvCacheType::kQ4_0:
    case KvCacheType::kQ4_0Rotated:
      return false;
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ8_0Rotated:
      return true;
    case KvCacheType::kQ8_0Q4_0:
    case KvCacheType::kQ8_0Q4_0Rotated:
      break;  // Mixed policies require an explicit K/V side.
  }
  throw std::invalid_argument("unsupported Flash-Next KV row codec");
}

[[nodiscard]] constexpr bool KvIsRotated(KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ4_0:
    case KvCacheType::kQ8_0Q4_0:
      return false;
    case KvCacheType::kQ8_0Rotated:
    case KvCacheType::kQ4_0Rotated:
    case KvCacheType::kQ8_0Q4_0Rotated:
      return true;
  }
  throw std::invalid_argument("unsupported Flash-Next KV cache type");
}

[[nodiscard]] constexpr std::uint32_t ExpectedKvPayloadVersion(
    KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
    case KvCacheType::kQ8_0:
      return kKvCacheNumericsEpoch;
    case KvCacheType::kQ8_0Rotated:
      return kKvRotatedNumericsEpoch;
    case KvCacheType::kQ4_0:
    case KvCacheType::kQ8_0Q4_0:
    case KvCacheType::kQ4_0Rotated:
    case KvCacheType::kQ8_0Q4_0Rotated:
      return kKvQ4NumericsEpoch;
  }
  throw std::invalid_argument("unsupported Flash-Next KV cache type");
}

// Resolve a session policy to a storage-only row codec. Rotation is owned by
// the producer, never by the codec or restore path.
[[nodiscard]] constexpr KvCacheType KvKeyType(KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
      return KvCacheType::kF16;
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ8_0Rotated:
    case KvCacheType::kQ8_0Q4_0:
    case KvCacheType::kQ8_0Q4_0Rotated:
      return KvCacheType::kQ8_0;
    case KvCacheType::kQ4_0:
    case KvCacheType::kQ4_0Rotated:
      return KvCacheType::kQ4_0;
  }
  throw std::invalid_argument("unsupported Flash-Next KV cache type");
}
[[nodiscard]] constexpr KvCacheType KvValueType(KvCacheType type) {
  if (type == KvCacheType::kQ8_0Q4_0 || type == KvCacheType::kQ8_0Q4_0Rotated)
    return KvCacheType::kQ4_0;
  return KvKeyType(type);
}
[[nodiscard]] constexpr bool KvIsPacked(KvCacheType type) {
  return KvKeyType(type) != KvCacheType::kF16;
}
[[nodiscard]] constexpr std::uint32_t KvComponentLayoutVersion(
    KvCacheType type) {
  if (!IsKnownKvCacheType(type))
    throw std::invalid_argument("unsupported Flash-Next KV cache type");
  return static_cast<std::uint32_t>(type) + 1;
}

// Standard q4_0: signed F16 scale, low nibbles for coordinates 0..15,
// high nibbles for 16..31. Nibbles represent signed codes offset by eight.
struct KvBlockQ4_0 {
  std::uint16_t scale;
  std::uint8_t values[16];
};
static_assert(sizeof(KvBlockQ4_0) == 18);
static_assert(offsetof(KvBlockQ4_0, values) == 2);
static_assert(std::is_trivially_copyable_v<KvBlockQ4_0>);

// Standard q8_0: IEEE F16 scale followed by 32 signed quantized values.
// There is no inter-block padding. Serialized scales (also F16 rows) are
// little-endian; byte-buffer APIs do not require aligned storage.
struct KvBlockQ8_0 {
  std::uint16_t scale;
  std::int8_t values[32];
};
static_assert(sizeof(KvBlockQ8_0) == 34);
static_assert(offsetof(KvBlockQ8_0, scale) == 0);
static_assert(offsetof(KvBlockQ8_0, values) == 2);
static_assert(std::is_standard_layout_v<KvBlockQ8_0>);
static_assert(std::is_trivially_copyable_v<KvBlockQ8_0>);

// Empty rows are valid. Unknown types, partial q8 blocks and size_t overflow
// are invalid. KvRowBytes throws invalid_argument for invalid geometry.
[[nodiscard]] constexpr bool IsValidKvRow(std::uint32_t elements,
                                          KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
      return elements <= std::numeric_limits<std::size_t>::max() / 2;
    case KvCacheType::kQ8_0:
    case KvCacheType::kQ8_0Rotated:
      return elements % kKvQ8BlockElements == 0 &&
             elements / kKvQ8BlockElements <=
                 std::numeric_limits<std::size_t>::max() / sizeof(KvBlockQ8_0);
    case KvCacheType::kQ4_0:
    case KvCacheType::kQ4_0Rotated:
      return elements % 32 == 0 &&
             elements / 32 <=
                 std::numeric_limits<std::size_t>::max() / sizeof(KvBlockQ4_0);
    case KvCacheType::kQ8_0Q4_0:
    case KvCacheType::kQ8_0Q4_0Rotated:
      return false;  // No single row stride for a mixed policy.
  }
  return false;
}

[[nodiscard]] constexpr std::size_t KvRowBytes(std::uint32_t elements,
                                               KvCacheType type) {
  if (!IsValidKvRow(elements, type))
    throw std::invalid_argument("invalid Flash-Next KV row geometry");
  return type == KvCacheType::kF16
             ? std::size_t{elements} * 2
             : std::size_t{elements / kKvQ8BlockElements} *
                   (KvUsesQ8Codec(type) ? sizeof(KvBlockQ8_0)
                                        : sizeof(KvBlockQ4_0));
}

[[nodiscard]] constexpr const char* KvCacheTypeName(KvCacheType type) {
  switch (type) {
    case KvCacheType::kF16:
      return "f16";
    case KvCacheType::kQ8_0:
      return "q8_0";
    case KvCacheType::kQ8_0Rotated:
      return "q8_0-rotated";
    case KvCacheType::kQ4_0:
      return "q4_0";
    case KvCacheType::kQ8_0Q4_0:
      return "q8_0/q4_0";
    case KvCacheType::kQ4_0Rotated:
      return "q4_0-rotated";
    case KvCacheType::kQ8_0Q4_0Rotated:
      return "q8_0/q4_0-rotated";
  }
  return "unknown";
}

// Portable IEEE binary32/binary16 conversions, with round-to-nearest-even
// independent of the host rounding mode. Preserve signed zero and infinity;
// NaNs become signed quiet NaNs. Finite overflow rounds to infinity.
[[nodiscard]] std::uint16_t KvFloatToHalfBits(float value) noexcept;
[[nodiscard]] float KvHalfBitsToFloat(std::uint16_t bits) noexcept;

// Canonical ggml q8_0 rounding primitive: roundf ties away from zero,
// clamp to [-127,127], rejecting nonfinite scaled values.
[[nodiscard]] std::int8_t RoundKvQ8Code(float scaled);

// CPU oracle for canonical live-row insertion, NOT snapshot compression.
// K input is F32 post-RoPE; V input is F32. Each q8 block independently uses
// d = max(abs(x))/127 in F32, q = clamp(roundf(x/d), -127, 127), where roundf
// ties go away from zero. Only the STORED scale is rounded to F16 RNE.
// All-zero blocks have scale=0/q=0. If F32 d itself underflows to zero, the
// block also becomes zero. If only the stored F16 scale underflows, preserve
// q computed from F32 d (no rescaling). CPU decode preserves zero signs.
// Decode rounds q * stored_scale to F16 before promoting to float. GPU scalar
// attention/WMMA operands require IEEE value equality, with +0/-0 equivalent;
// nonzero reconstructed values and packed bytes are not relaxed.
//
// q4 uses signed first-absmax / -8, F32 reciprocal, product then +8.5,
// truncation/clamp[0,15]. Signed scales (including -0) are valid for q4.
// Mixed session policies must be resolved with KvKeyType/KvValueType first.
//
// Exact buffer lengths are required. All APIs throw invalid_argument for
// invalid geometry, nonfinite input/payload, negative scales (including -0),
// -128 quants, or scale/reconstruction overflow to F16 infinity. Encode and
// Decode reject overlapping buffers. Validation finishes before any output
// writes, so a rejected call leaves the entire destination unchanged. Empty
// buffers are accepted only for an empty valid row. No HIP dependency.
// With rotation enabled these storage-only APIs accept kQ8_0Rotated as
// ALREADY-ROTATED coordinates. They never rotate, inverse-rotate or migrate
// bytes; the caller must choose the K/Q or V/output transform separately.
void EncodeKvRow(std::span<const float> input, KvCacheType type,
                 std::span<std::byte> output);
void DecodeKvRow(std::span<const std::byte> input, KvCacheType type,
                 std::span<float> output);
void ValidateKvRow(std::span<const std::byte> input, std::uint32_t elements,
                   KvCacheType type);

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_HPP_
