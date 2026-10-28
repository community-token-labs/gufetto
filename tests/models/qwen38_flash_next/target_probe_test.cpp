#include "src/models/qwen38_flash_next/target_probe.hpp"

#if defined(__linux__)
#include <sys/stat.h>
#endif

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

#include "src/core/platform/env.hpp"

namespace probe = gufo::models::qwen38_flash_next::detail;
namespace fs = std::filesystem;
namespace {
void Require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
template<typename F>
void Refused(F&& action) {
  bool refused = false;
  try {
    action();
  } catch (const std::runtime_error&) {
    refused = true;
  }
  Require(refused, "unsafe or inconsistent probe operation was accepted");
}
std::vector<std::uint8_t> Read(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}
struct Temporary {
  fs::path path;
  ~Temporary() {
    gufo::platform::UnsetEnv("GUFO_FLASH_NEXT_TARGET_PROBE");
    fs::remove_all(path);
  }
};
}  // namespace

int main() {
  try {
    Require(gufo::platform::UnsetEnv("GUFO_FLASH_NEXT_TARGET_PROBE"),
            "cannot clear probe environment");
    const std::array<float, 3> logits{1.25F, -0.0F, -4.5F};
    probe::ProbeTargetLogits(2415, logits);  // Fully disabled: no directory.
    if (std::string_view(gufo::platform::HostOsName()) != "Linux") {
      Require(gufo::platform::SetEnv("GUFO_FLASH_NEXT_TARGET_PROBE",
                                     "unsupported-private-probe"),
              "cannot set probe environment");
      Require(probe::TargetProbeRange(2415, 0).second == 0,
              "empty probe range became active");
      try {
        probe::ProbeTargetLogits(2415, logits);
        throw std::logic_error("unsupported probe was accepted");
      } catch (const std::runtime_error& error) {
        Require(std::string_view(error.what()).find("unsupported") !=
                    std::string_view::npos,
                "unsupported probe error is unclear");
      }
      Require(gufo::platform::UnsetEnv("GUFO_FLASH_NEXT_TARGET_PROBE"),
              "cannot clear probe environment");
      std::cout << "target_probe_test passed (optional probe unsupported)\n";
      return 0;
    }
    const auto directory =
        fs::temp_directory_path() /
        ("qfn-target-probe-" + std::to_string(gufo::platform::ProcessId()));
    Require(fs::create_directory(directory),
            "temporary probe directory failed");
    Temporary temporary{directory};
    Require(gufo::platform::RestrictToOwner(directory),
            "cannot restrict probe directory");
    Require(gufo::platform::SetEnv("GUFO_FLASH_NEXT_TARGET_PROBE",
                                   directory.string()),
            "cannot set probe environment");
    const auto arm = [&](const char* text) {
      std::ofstream(temporary.path / "arm") << text;
      fs::permissions(temporary.path / "arm",
                      fs::perms::owner_read | fs::perms::owner_write);
    };
    Require(probe::TargetProbeRange(2415, 8).second == 0,
            "unarmed probe became active");
    arm("2415:2447\n");
    Require(probe::TargetProbeRange(2410, 8) == std::pair{2415U, 2418U},
            "probe intersection changed");
    Require(probe::TargetProbeRange(2447, 8).second == 0,
            "end-exclusive bound changed");
    probe::ProbeTargetLogits(2414, logits);
    Require(!fs::exists(temporary.path / "2414.logits"),
            "out-of-window data was written");
    probe::ProbeTargetLogits(2415, logits);
    probe::ProbeTargetLogits(2415, logits);  // Preview repeats must match.
    const auto raw = Read(temporary.path / "2415.logits");
    const std::vector<std::uint8_t> expected{
        0x00, 0x00, 0xa0, 0x3f, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x90, 0xc0};
    Require(raw == expected, "raw float bits or byte order changed");
    probe::ProbeTargetToken(2415, 0x12345678);
    Require(Read(temporary.path / "2415.token") ==
                std::vector<std::uint8_t>{0x78, 0x56, 0x34, 0x12},
            "token ID encoding changed");
    probe::ProbeTargetToken(2416, 0, true);
    Require(fs::file_size(temporary.path / "2416.stop") == 0,
            "terminal marker changed");
    Refused([&] { probe::ProbeTargetToken(2416, 1); });
    const std::array<float, 3> different{1.25F, 0.0F, -4.5F};
    Refused([&] { probe::ProbeTargetLogits(2415, different); });
    Require(Read(temporary.path / "2415.logits") == raw,
            "inconsistent repeat overwrote evidence");
    const auto mode = fs::status(temporary.path / "2415.logits").permissions();
    Require((mode & fs::perms::group_all) == fs::perms::none &&
                (mode & fs::perms::others_all) == fs::perms::none,
            "probe output is not private");
    fs::create_symlink("2415.logits", temporary.path / "2417.logits");
    Refused([&] { probe::ProbeTargetLogits(2417, logits); });
    for (const auto* invalid : {"2415:2448", "1:1", "-1:2", "1:2junk",
                                "1:2\nextra", "4294967296:4294967297"}) {
      arm(invalid);
      Refused([&] { (void)probe::TargetProbeRange(2415, 1); });
    }
    arm("2415:2447");
    fs::permissions(temporary.path / "arm", fs::perms::others_read,
                    fs::perm_options::add);
    Refused([&] { (void)probe::TargetProbeRange(2415, 1); });
    fs::remove(temporary.path / "arm");
    Require(fs::create_directory(temporary.path / "arm"),
            "nonregular arm fixture failed");
    Refused([&] { (void)probe::TargetProbeRange(2415, 1); });
    fs::remove(temporary.path / "arm");
#if defined(__linux__)
    // Retain the Linux nonblocking-open regression: a FIFO must be refused
    // rather than waiting for a writer before the regular-file check.
    Require(::mkfifo((temporary.path / "arm").c_str(), 0600) == 0,
            "FIFO arm fixture failed");
    Refused([&] { (void)probe::TargetProbeRange(2415, 1); });
    fs::remove(temporary.path / "arm");
#endif
    probe::ProbeTargetLogits(2418, logits);
    Require(!fs::exists(temporary.path / "2418.logits"),
            "disarmed probe wrote data");
    std::cout << "target_probe_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
