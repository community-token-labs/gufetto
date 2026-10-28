#ifndef GUFO_CORE_PLATFORM_TEMPORARY_FILE_HPP_
#define GUFO_CORE_PLATFORM_TEMPORARY_FILE_HPP_

#include <cerrno>
#include <filesystem>
#include <string>
#include <system_error>

#include "src/core/platform/mapped_file.hpp"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>

#include <array>

#include "src/core/platform/os_info.hpp"
#else
#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#endif

namespace gufo::platform {

/// Creates a private, non-inheritable read/write spool in the OS temporary
/// directory. POSIX unlinks it before returning; Windows deletes it on last
/// close and denies other opens. The caller owns the returned handle and must
/// CloseFile() it. No persistent pathname is exposed.
inline PlatformFile OpenTemporaryFile(std::error_code& ec) {
  ec.clear();
  const auto directory = std::filesystem::temp_directory_path(ec);
  if (ec)
    return kInvalidFile;
#if defined(_WIN32)
  for (int attempt = 0; attempt < 8; ++attempt) {
    std::array<std::uint8_t, 16> random{};
    if (!FillRandomBytes(random)) {
      ec = std::make_error_code(std::errc::io_error);
      return kInvalidFile;
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string name = "gufo-transfer-";
    for (auto byte : random) {
      name.push_back(hex[byte >> 4]);
      name.push_back(hex[byte & 15]);
    }
    const auto path = directory / name;
    int fd = -1;
    const auto error = _wsopen_s(
        &fd, path.c_str(),
        _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY | _O_NOINHERIT | _O_TEMPORARY,
        _SH_DENYRW, _S_IREAD | _S_IWRITE);
    if (error == EEXIST)
      continue;
    if (error != 0) {
      ec = {error, std::generic_category()};
      return kInvalidFile;
    }
    // Hand ownership to the existing native-handle API without leaving a CRT
    // descriptor allocated. The duplicate retains delete-on-last-close.
    const auto file =
        DuplicateFile(reinterpret_cast<PlatformFile>(_get_osfhandle(fd)));
    (void)_close(fd);
    if (!FileValid(file))
      ec = std::make_error_code(std::errc::io_error);
    return file;
  }
  ec = std::make_error_code(std::errc::file_exists);
  return kInvalidFile;
#else
  auto path = (directory / "gufo-transfer-XXXXXX").string();
  const int fd = ::mkostemp(path.data(), O_CLOEXEC);
  if (fd < 0) {
    ec = {errno, std::generic_category()};
    return kInvalidFile;
  }
  if (::unlink(path.c_str()) != 0) {
    ec = {errno, std::generic_category()};
    (void)::close(fd);
    return kInvalidFile;
  }
  return fd;
#endif
}

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_TEMPORARY_FILE_HPP_
