#ifndef GUFO_CORE_PLATFORM_CACHE_IDENTITY_HPP_
#define GUFO_CORE_PLATFORM_CACHE_IDENTITY_HPP_

#include <array>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <link.h>
#include <unistd.h>
#endif

#include "src/core/crypto/sha256.hpp"
#include "src/core/platform/os_info.hpp"

namespace gufo::platform {

// Environment enumeration belongs in the platform leaf, not serving code.
inline std::vector<std::string> CacheIdentityEnvironment() {
  std::vector<std::string> result;
#if defined(_WIN32)
  // UCRT may define _environ as (*__p__environ()), not an identifier that
  // can be prefixed with a global scope qualifier.
  auto** environment = _environ;
#elif defined(__linux__)
  auto** environment = ::environ;
#else
  // Other hosts have no supported transfer lane. Their advisory epochs are
  // process-local, independently of environment enumeration.
  return result;
#endif
#if defined(_WIN32) || defined(__linux__)
  for (auto** p = environment; p && *p; ++p)
    result.emplace_back(*p);
#endif
  return result;
}

// Lookup is advisory and local, unlike transfer and registered pins. Keep
// Linux's executable binding; elsewhere a process nonce conservatively avoids
// equating different binaries without requiring Linux /proc or strict pins.
inline const std::string& CacheLookupExecutableIdentity() {
  static const auto identity = [] {
#if defined(__linux__)
    return crypto::Sha256FileHex("/proc/self/exe");
#else
    std::array<std::uint8_t, 32> nonce{};
    if (!FillRandomBytes(nonce))
      throw std::runtime_error("cannot identify local cache lookup process");
    return crypto::Sha256Hex(nonce);
#endif
  }();
  return identity;
}

struct CheckpointTransferRuntimeIdentity {
  std::string executable_sha256;
  std::set<std::string> object_sha256;
  std::vector<std::string> environment;
};

inline constexpr bool CheckpointTransferRuntimeSupported() {
#if defined(__linux__) && defined(__x86_64__)
  return true;
#else
  return false;
#endif
}

// Unlike advisory lookup, transfer never substitutes a revision or process
// nonce for exact executable/loaded-object contents. Do not reuse or change
// StrictRuntimeFingerprint: registered pins retain their separate contract.
inline CheckpointTransferRuntimeIdentity
ReadCheckpointTransferRuntimeIdentity() {
#if defined(__linux__) && defined(__x86_64__)
  CheckpointTransferRuntimeIdentity result;
  result.executable_sha256 = crypto::Sha256FileHex("/proc/self/exe");
  std::set<std::string> objects;
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t, void* data) {
        if (info->dlpi_name && info->dlpi_name[0] == '/')
          static_cast<std::set<std::string>*>(data)->insert(info->dlpi_name);
        return 0;
      },
      &objects);
  for (const auto& object : objects)
    result.object_sha256.insert(crypto::Sha256FileHex(object));
  result.environment = CacheIdentityEnvironment();
  return result;
#else
  throw std::runtime_error(
      "checkpoint transfer is unsupported on this host: Linux x86-64 "
      "runtime fingerprinting is required");
#endif
}

}  // namespace gufo::platform
#endif  // GUFO_CORE_PLATFORM_CACHE_IDENTITY_HPP_
