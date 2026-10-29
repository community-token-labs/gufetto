#include "src/core/gguf_identity.hpp"

#include <fcntl.h>
#include <linux/fsverity.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"
#include "src/core/platform/env.hpp"
#include "src/core/platform/fs.hpp"

namespace {
struct stat frozen_stat{};
bool freeze_stat = false;
std::atomic<std::size_t> model_read_bytes{0};
std::atomic<bool> mutate_on_read{false};
int mutation_fd = -1;
off_t mutation_offset = 0;
int fake_verity_byte = -1;
bool IsModel(int fd) {
  struct stat st{};
  return ::fstatat(fd, "", &st, AT_EMPTY_PATH) == 0 &&
         st.st_dev == frozen_stat.st_dev && st.st_ino == frozen_stat.st_ino;
}
}  // namespace

extern "C" int __real_fstat(int, struct stat*);
extern "C" int __wrap_fstat(int fd, struct stat* out) {
  const int result = __real_fstat(fd, out);
  if (result == 0 && freeze_stat && out->st_dev == frozen_stat.st_dev &&
      out->st_ino == frozen_stat.st_ino)
    *out = frozen_stat;
  return result;
}
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* buffer, size_t count,
                                off_t offset) {
  const auto result = __real_pread(fd, buffer, count, offset);
  if (result > 0 && IsModel(fd)) {
    model_read_bytes.fetch_add(static_cast<std::size_t>(result));
    if (offset == mutation_offset && mutate_on_read.exchange(false)) {
      const auto byte = static_cast<unsigned char*>(buffer)[0] ^ 0xffU;
      const unsigned char replacement = static_cast<unsigned char>(byte);
      if (::pwrite(mutation_fd, &replacement, 1, mutation_offset) != 1)
        std::abort();
    }
  }
  return result;
}
extern "C" int __real_ioctl(int, unsigned long, ...);
extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
  va_list args;
  va_start(args, request);
  void* argument = va_arg(args, void*);
  va_end(args);
  if (request == FS_IOC_MEASURE_VERITY && fake_verity_byte >= 0 &&
      IsModel(fd)) {
    auto* digest = static_cast<fsverity_digest*>(argument);
    digest->digest_algorithm = FS_VERITY_HASH_ALG_SHA256;
    digest->digest_size = 32;
    std::memset(digest->digest, fake_verity_byte, 32);
    return 0;
  }
  return __real_ioctl(fd, request, argument);
}

namespace {

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

constexpr std::size_t kWindow = 4096;

/// Minimal GGUF v3 image: one string metadata entry and two tensors whose
/// payloads span `first_bytes` and `second_bytes`.
std::vector<std::uint8_t> BuildImage(std::string_view name_value,
                                     std::string_view first_tensor_name,
                                     std::size_t first_bytes,
                                     std::size_t second_bytes) {
  std::vector<std::uint8_t> buffer;
  auto append = [&buffer](const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    buffer.insert(buffer.end(), bytes, bytes + size);
  };
  auto pod = [&append](auto value) { append(&value, sizeof(value)); };
  auto string = [&](std::string_view value) {
    pod(static_cast<std::uint64_t>(value.size()));
    append(value.data(), value.size());
  };

  append("GGUF", 4);
  pod(static_cast<std::uint32_t>(3));
  pod(static_cast<std::uint64_t>(2));  // tensor count
  pod(static_cast<std::uint64_t>(1));  // metadata count
  string("general.name");
  pod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
  string(name_value);

  string(first_tensor_name);
  pod(static_cast<std::uint32_t>(1));
  pod(static_cast<std::uint64_t>(first_bytes / 2));
  pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kF16));
  pod(static_cast<std::uint64_t>(0));
  string("second");
  pod(static_cast<std::uint32_t>(1));
  pod(static_cast<std::uint64_t>(second_bytes / 2));
  pod(static_cast<std::uint32_t>(gufo::core::GgmlType::kF16));
  pod(static_cast<std::uint64_t>(first_bytes));

  while (buffer.size() % 32 != 0) {
    buffer.push_back(0);
  }
  const std::size_t payload_base = buffer.size();
  buffer.resize(payload_base + first_bytes + second_bytes);
  for (std::size_t index = 0; index < first_bytes + second_bytes; ++index) {
    buffer[payload_base + index] = static_cast<std::uint8_t>(index * 7U + 3U);
  }
  return buffer;
}

std::size_t PayloadBase(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(reader != nullptr, "open image: " + error);
  return static_cast<std::size_t>(
      static_cast<const std::uint8_t*>(reader->GetTensors()[0].data) -
      image.data());
}

std::string Identity(const std::vector<std::uint8_t>& image) {
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(image.data(), image.size(), &error);
  Expect(reader != nullptr, "open image: " + error);
  return gufo::core::GgufIdentityHex(*reader);
}

void TestDeterministic() {
  const auto image = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  const std::string first = Identity(image);
  Expect(first.size() == 64, "identity is 64 hex characters");
  Expect(first == Identity(image), "identity is deterministic");
  Expect(first ==
             Identity(BuildImage("model", "first", 64 * kWindow, 2 * kWindow)),
         "identity depends on bytes, not on the buffer instance");
}

void TestHeaderChangesIdentity() {
  const auto base = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  Expect(Identity(base) != Identity(BuildImage("model-v2", "first",
                                               64 * kWindow, 2 * kWindow)),
         "metadata change alters identity");
  Expect(Identity(base) != Identity(BuildImage("model", "renamed", 64 * kWindow,
                                               2 * kWindow)),
         "tensor name change alters identity");
  Expect(Identity(base) !=
             Identity(BuildImage("model", "first", 64 * kWindow, 4 * kWindow)),
         "tensor extent change alters identity");
}

void TestEveryPayloadByte() {
  const std::size_t first_bytes = 64 * kWindow;
  auto image = BuildImage("model", "first", first_bytes, 2 * kWindow);
  const std::string base = Identity(image);
  const std::size_t payload = PayloadBase(image);

  auto flipped = [&](std::size_t offset) {
    auto copy = image;
    copy[payload + offset] ^= 0xFFU;
    return Identity(copy);
  };

  Expect(flipped(0) != base, "start bytes are hashed");
  Expect(flipped(first_bytes - 1) != base, "end bytes are hashed");
  const std::size_t middle = ((first_bytes - kWindow) / 2) & ~(kWindow - 1);
  Expect(flipped(middle + 17) != base, "middle bytes are hashed");
  Expect(flipped(kWindow + 100) != base,
         "bytes outside the old sample windows alter identity");

  // The short second tensor is hashed whole.
  Expect(flipped(first_bytes + kWindow + 5) != base,
         "small tensors are hashed in full");
}

void TestFileDigestCache() {
  auto temp = (std::filesystem::temp_directory_path() / "gufo-identity-XXXXXX")
                  .string();
  Expect(mkdtemp(temp.data()) != nullptr, "create cache test directory");
  const std::filesystem::path root(temp);
  const char* old_cache = std::getenv("XDG_CACHE_HOME");
  const std::string saved_cache = old_cache ? old_cache : "";
  const bool had_cache = old_cache != nullptr;
  gufo::platform::SetEnv("XDG_CACHE_HOME", temp);
  auto bytes = BuildImage("model", "first", 64 * kWindow, 2 * kWindow);
  const auto path = root / "model.gguf";
  auto write = [&] {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    Expect(out.good(), "write GGUF fixture");
  };
  write();
  auto read = [&] {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(path.string(), &error);
    Expect(reader != nullptr, "open file: " + error);
    return gufo::core::GgufIdentityHex(*reader);
  };
  Expect(stat(path.c_str(), &frozen_stat) == 0, "capture model stat");
  freeze_stat = true;
  const auto first = read();
  Expect(first == Identity(bytes), "file and memory full digests agree");
  Expect(read() == first, "cached digest agrees with initial scan");
  const auto cache = root / "gufo" / "gguf-sha256-memo-v2";
  Expect(std::distance(std::filesystem::directory_iterator(cache),
                       std::filesystem::directory_iterator{}) == 1,
         "one cached full digest exists");
  struct stat before{};
  Expect(stat(path.c_str(), &before) == 0, "stat original file");
  bytes[PayloadBase(bytes) + kWindow + 100] ^= 0xff;
  write();
  const timespec times[] = {before.st_atim, before.st_mtim};
  Expect(utimensat(AT_FDCWD, path.c_str(), times, 0) == 0,
         "restore original mtime");
  Expect(read() != first,
         "same-size edit with restored mtime invalidates cache");
  Expect(read() == Identity(bytes),
         "edited file still matches full memory digest");
  for (const auto& entry : std::filesystem::directory_iterator(cache)) {
    std::ofstream(entry.path()) << "broken";
  }
  Expect(read() == Identity(bytes), "malformed cached digest is recomputed");
  freeze_stat = false;
  if (had_cache)
    gufo::platform::SetEnv("XDG_CACHE_HOME", saved_cache);
  else
    gufo::platform::UnsetEnv("XDG_CACHE_HOME");
  std::filesystem::remove_all(root);
}

void TestSampledMemo() {
  auto temp =
      (std::filesystem::temp_directory_path() / "gufo-sampled-XXXXXX").string();
  Expect(mkdtemp(temp.data()) != nullptr, "create sampling test directory");
  const std::filesystem::path root(temp);
  const char* old_cache = std::getenv("XDG_CACHE_HOME");
  const std::string saved_cache = old_cache ? old_cache : "";
  const bool had_cache = old_cache != nullptr;
  gufo::platform::SetEnv("XDG_CACHE_HOME", temp);
  // A header larger than one sample block and a payload with unsampled gaps.
  auto bytes = BuildImage(std::string(2 * 1024 * 1024, 'x'), "first",
                          80 * 1024 * 1024, 8194);
  const auto payload = PayloadBase(bytes);
  const auto path = root / "model.gguf";
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    Expect(out.good(), "write sampled fixture");
  }
  mutation_fd = ::open(path.c_str(), O_RDWR);
  Expect(mutation_fd >= 0, "open sampled fixture for edits");
  Expect(stat(path.c_str(), &frozen_stat) == 0, "freeze large fixture stat");
  freeze_stat = true;
  const auto read = [&] {
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(path, &error);
    Expect(reader != nullptr, "open sampled file: " + error);
    Expect(reader->GetMappedRegions()[0].header_size == payload,
           "complete header extent is exposed");
    model_read_bytes = 0;
    return gufo::core::GgufIdentityHex(*reader);
  };
  const auto flip = [&](std::size_t offset) {
    bytes[offset] ^= 0xff;
    Expect(::pwrite(mutation_fd, &bytes[offset], 1, offset) == 1,
           "edit one byte");
  };
  const auto cache = root / "gufo" / "gguf-sha256-memo-v2";
  // A plausible old stat-only digest MUST NOT authorize a new memo hit.
  const auto old = root / "gufo" / "gguf-sha256-v1";
  std::filesystem::create_directories(old);
  Expect(::chmod(old.c_str(), 0700) == 0,
         "old memo directory has trusted permissions");
  const auto stamp =
      std::to_string(frozen_stat.st_dev) + ":" +
      std::to_string(frozen_stat.st_ino) + ":" +
      std::to_string(frozen_stat.st_size) + ":" +
      std::to_string(frozen_stat.st_mtim.tv_sec * 1'000'000'000LL +
                     frozen_stat.st_mtim.tv_nsec) +
      ":" +
      std::to_string(frozen_stat.st_ctim.tv_sec * 1'000'000'000LL +
                     frozen_stat.st_ctim.tv_nsec);
  gufo::crypto::Sha256Hasher key;
  key.Update(
      {reinterpret_cast<const std::uint8_t*>(stamp.data()), stamp.size()});
  std::ofstream(old / key.FinishHex()) << std::string(64, '0') << '\n';
  auto first = read();
  Expect(first == Identity(bytes) && model_read_bytes >= bytes.size(),
         "old stat-only memo ignored; initial scan hashes full contents");
  const std::size_t probe_bytes = payload + 16 * 1024 * 1024;
  Expect(read() == first, "large memo hit agrees");
  Expect(
      model_read_bytes == 2 * probe_bytes,
      "warm hit reads exactly header plus bounded windows twice, no full scan");

  const std::string_view tensor_name = "first";
  const auto tensor_name_offset = static_cast<std::size_t>(
      std::search(bytes.begin(), bytes.begin() + payload, tensor_name.begin(),
                  tensor_name.end()) -
      bytes.begin());
  Expect(tensor_name_offset < payload, "find tensor index name independently");
  const auto interior =
      payload + (bytes.size() - payload - 64 * 1024) * 128 / 255 + 17;
  for (const auto offset : {std::size_t{1024 * 1024}, tensor_name_offset,
                            payload, interior, bytes.size() - 1}) {
    flip(offset);
    const auto changed = read();
    Expect(changed != first && changed == Identity(bytes),
           "frozen-stat metadata, index, first/interior/tail edits force full "
           "rehash");
    Expect(model_read_bytes >= bytes.size(),
           "fingerprint mismatch scans entire file");
    first = changed;
  }
  // The exact layout leaves a gap after the first 64KiB block. This is an
  // intentional limitation check, not a claim that sampling is content-proof.
  flip(payload + 64 * 1024 + 1);
  Expect(
      read() == first && Identity(bytes) != first,
      "same-stat unsampled edit remains undetectable without a kernel token");
  fake_verity_byte = 1;
  first = read();
  Expect(first == Identity(bytes), "new immutable token forces full scan");
  flip(payload + 64 * 1024 + 2);
  fake_verity_byte = 2;
  Expect(read() == Identity(bytes) && model_read_bytes >= bytes.size(),
         "changed kernel token invalidates even an unsampled edit");
  fake_verity_byte = -1;
  std::filesystem::remove_all(cache);
  mutation_offset = static_cast<off_t>(payload);
  mutate_on_read = true;
  bool rejected = false;
  try {
    (void)read();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected && !mutate_on_read,
         "frozen-stat in-flight sampled mutation is rejected");
  Expect(std::filesystem::is_empty(cache), "failed scan publishes no memo");
  bytes[payload] ^= 0xff;
  first = read();
  mutate_on_read = true;
  rejected = false;
  try {
    (void)read();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected && !mutate_on_read,
         "memo hit also rechecks content before returning");
  ::close(mutation_fd);
  mutation_fd = -1;
  freeze_stat = false;
  if (had_cache)
    gufo::platform::SetEnv("XDG_CACHE_HOME", saved_cache);
  else
    gufo::platform::UnsetEnv("XDG_CACHE_HOME");
  std::filesystem::remove_all(root);
}

void TestPipelinedFileHash() {
  const auto bytes = BuildImage("pipeline", "first", 40 * 1024 * 1024, 8194);
  const auto path = std::filesystem::temp_directory_path() /
                    ("gufo-identity-pipeline-" + std::to_string(getpid()));
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(path.string(), &error);
  Expect(reader != nullptr, "pipelined fixture opens");
  Expect(gufo::core::GgufIdentityHex(*reader) == Identity(bytes),
         "parallel disk reads hash all chunks and the unaligned tail in order");
  std::filesystem::remove(path);
}

void TestSha256() {
  const std::string text = "abc";
  gufo::crypto::Sha256Hasher hash;
  for (unsigned char byte : text)
    hash.Update({&byte, 1});
  Expect(hash.FinishHex() ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "incremental SHA-256 matches independent standard vector");
  Expect(gufo::crypto::Sha256Hex({}) ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "empty SHA-256 matches standard vector");
}

}  // namespace

int main() {
  TestSha256();
  TestFileDigestCache();
  TestSampledMemo();
  TestPipelinedFileHash();
  TestDeterministic();
  TestHeaderChangesIdentity();
  TestEveryPayloadByte();
  std::cout << "gguf_identity_test passed\n";
  return 0;
}
