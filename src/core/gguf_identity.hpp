#ifndef GUFO_CORE_GGUF_IDENTITY_HPP_
#define GUFO_CORE_GGUF_IDENTITY_HPP_

#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace gufo::core {

/// Scheme tag for full-content GGUF identities.
inline constexpr std::string_view kGgufIdentityScheme = "gguf-sha256-v1";

/// Full-content identity of every shard, including metadata and tensor padding.
/// Memory images are always hashed. File memos require matching stat fields,
/// an optional kernel immutable digest, and freshly read header/index plus
/// bounded payload samples. Copies have identical identities; a memo miss
/// scans the full artifact. Old stat-only memos are not reused.
/// Throws on detected changes during lookup, NOT on every possible rewrite:
/// without fs-verity, a same-stat rewrite confined to unsampled bytes can
/// escape detection. Models must remain stable during lookup and use.
/// See docs/GGUF_IDENTITY.md for the trust boundary and sampling contract.
[[nodiscard]] std::string GgufIdentityHex(const GgufReader& reader);

}  // namespace gufo::core

#endif  // GUFO_CORE_GGUF_IDENTITY_HPP_
