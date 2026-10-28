#ifndef GUFO_CORE_PLATFORM_RUNTIME_FINGERPRINT_HPP_
#define GUFO_CORE_PLATFORM_RUNTIME_FINGERPRINT_HPP_

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>

#if defined(__linux__)
#include <link.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <set>
#include <string>

#include "src/core/crypto/sha256.hpp"
#endif

namespace gufo::platform {

/// Strict executable/shared-object/kernel/environment identity for registered
/// pins. Fail closed on hosts without the Linux fingerprint contract; this is
/// deliberately stronger than the ordinary disk-cache compatibility identity.
inline std::array<std::uint8_t, 32> StrictRuntimeFingerprint(
    std::string_view domain) {
#if defined(__linux__)
  std::set<std::string> objects;
  dl_iterate_phdr(
      [](dl_phdr_info* info, std::size_t, void* data) {
        if (info->dlpi_name && info->dlpi_name[0] == '/')
          static_cast<std::set<std::string>*>(data)->insert(info->dlpi_name);
        return 0;
      },
      &objects);
  crypto::Sha256Hasher hash;
  const auto add = [&](std::string_view text) {
    hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(text.data()),
                          text.size()));
    const std::uint8_t zero{};
    hash.Update({&zero, 1});
  };
  add(domain);
  add(crypto::Sha256FileHex("/proc/self/exe"));
  for (const auto& object : objects) {
    add(object);
    add(crypto::Sha256FileHex(object));
  }
  utsname system{};
  if (uname(&system) != 0)
    throw std::runtime_error("cannot identify pinned-prefix runtime");
  add(system.release);
  add(system.machine);
  std::set<std::string> environment;
  for (auto** p = environ; p && *p; ++p) {
    const std::string_view entry(*p);
    if ((entry.starts_with("GUFO_") && !entry.starts_with("GUFO_PERF_TRACE")) ||
        entry.starts_with("HIP_") || entry.starts_with("HSA_") ||
        entry.starts_with("ROCR_") || entry.starts_with("ROCBLAS_") ||
        entry.starts_with("AMD_"))
      environment.emplace(entry);
  }
  for (const auto& value : environment)
    add(value);
  return hash.Finish();
#else
  (void)domain;
  throw std::runtime_error(
      "registered pins are unsupported on this host: Linux runtime "
      "fingerprinting is required");
#endif
}

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_RUNTIME_FINGERPRINT_HPP_
