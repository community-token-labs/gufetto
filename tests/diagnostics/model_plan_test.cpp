#include "src/core/diagnostics/model_plan.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "tests/diagnostics/recovery_bounds.hpp"

namespace {
const std::string header =
    "group\tstatus\trepo\trevision\tfile\tbytes\tsha256\tlicense\tgated\trole\t"
    "why\tfit\tsource";
const std::string budgets =
    "context\thost_available_bytes\tdevice_available_bytes\tos_reserve_bytes\t"
    "weights_bytes\tkv_bytes\tvision_bytes\tdraft_bytes\tworkspace_bytes\t"
    "cache_bytes\tpersistence_staging_bytes";
const std::string sha =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

std::string Row(std::string file = "model.gguf", std::string bytes = "3",
                std::string revision = std::string(40, 'a'),
                std::string digest = sha) {
  return "test\tcandidate\towner/model\t" + revision + "\t" + file + "\t" +
         bytes + "\t" + digest +
         "\tMIT\tfalse\ttarget\ttest\tprose claims "
         "FIT\thttps://example.invalid";
}

std::string Budget(std::string kv = "1") {
  return "\t32\t4294967300\t4294967300\t4294967296\t3\t" + kv +
         "\t0\t0\t0\t0\t0";
}

struct Fixture {
  std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("gufo-model-plan-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  Fixture() { std::filesystem::create_directory(root); }
  ~Fixture() { std::filesystem::remove_all(root); }
  gufo::diagnostics::ModelPlanReport Run(const std::string& manifest) const {
    std::ofstream(root / "plan.tsv", std::ios::binary) << manifest;
    return gufo::diagnostics::PlanModels(root / "plan.tsv");
  }
};

void Reject(Fixture& fixture, const std::string& text) {
  bool rejected = false;
  try {
    (void)fixture.Run(text);
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
}

// Independent GGUF-v3 fixture, not generated from production config/layout.
struct Gguf {
  std::vector<std::uint8_t> metadata, tensors, payload;
  std::uint64_t metadata_count = 0, tensor_count = 0;
  template<class T>
  static void Pod(std::vector<std::uint8_t>& out, T value) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
  }
  static void String(std::vector<std::uint8_t>& out, const std::string& value) {
    Pod(out, std::uint64_t(value.size()));
    out.insert(out.end(), value.begin(), value.end());
  }
  void Text(const std::string& key, const std::string& value) {
    String(metadata, key);
    Pod(metadata, std::uint32_t{8});
    String(metadata, value);
    ++metadata_count;
  }
  void Number(const std::string& key, std::uint32_t value) {
    String(metadata, key);
    Pod(metadata, std::uint32_t{4});
    Pod(metadata, value);
    ++metadata_count;
  }
  void Number64(const std::string& key, std::uint64_t value) {
    String(metadata, key);
    Pod(metadata, std::uint32_t{10});
    Pod(metadata, value);
    ++metadata_count;
  }
  void Tensor(const std::string& name, std::uint64_t elements) {
    payload.resize((payload.size() + 31) / 32 * 32);
    String(tensors, name);
    Pod(tensors, std::uint32_t{1});
    Pod(tensors, elements);
    Pod(tensors, std::uint32_t{0});  // F32
    Pod(tensors, std::uint64_t(payload.size()));
    payload.resize(payload.size() + elements * 4);
    ++tensor_count;
  }
  void Save(const std::filesystem::path& path) {
    std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
    Pod(bytes, std::uint32_t{3});
    Pod(bytes, tensor_count);
    Pod(bytes, metadata_count);
    bytes.insert(bytes.end(), metadata.begin(), metadata.end());
    bytes.insert(bytes.end(), tensors.begin(), tensors.end());
    bytes.resize((bytes.size() + 31) / 32 * 32);
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
};

void LocalLowerBound(Fixture& fixture) {
  Gguf gguf;
  gguf.Text("general.architecture", "qwen35");
  for (const auto* key : {"embedding_length", "feed_forward_length",
                          "attention.key_length", "rope.dimension_count"})
    gguf.Number(std::string("qwen35.") + key, 2);
  for (const auto* key : {"attention.head_count", "attention.head_count_kv",
                          "ssm.group_count", "ssm.time_step_rank"})
    gguf.Number(std::string("qwen35.") + key, 1);
  gguf.Number("qwen35.block_count", 4);
  gguf.Number("qwen35.context_length", 32);
  gguf.Number("qwen35.ssm.inner_size", 128);
  gguf.Tensor("token_embd.weight", 4);
  gguf.Tensor("output_norm.weight", 2);
  for (int layer = 0; layer < 4; ++layer) {
    const auto prefix = "blk." + std::to_string(layer) + ".";
    for (const auto* name : {"attn_norm.weight", "ffn_norm.weight"})
      gguf.Tensor(prefix + name, 2);
    for (const auto* name :
         {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
      gguf.Tensor(prefix + name, 4);
    if (layer == 3) {
      gguf.Tensor(prefix + "attn_q.weight", 8);
      for (const auto* name :
           {"attn_k.weight", "attn_v.weight", "attn_output.weight"})
        gguf.Tensor(prefix + name, 4);
      gguf.Tensor(prefix + "attn_q_norm.weight", 2);
      gguf.Tensor(prefix + "attn_k_norm.weight", 2);
    } else {
      gguf.Tensor(prefix + "attn_qkv.weight", 768);
      gguf.Tensor(prefix + "attn_gate.weight", 256);
      gguf.Tensor(prefix + "ssm_a", 1);
      gguf.Tensor(prefix + "ssm_conv1d.weight", 1536);
      gguf.Tensor(prefix + "ssm_dt.bias", 1);
      gguf.Tensor(prefix + "ssm_alpha.weight", 2);
      gguf.Tensor(prefix + "ssm_beta.weight", 2);
      gguf.Tensor(prefix + "ssm_norm.weight", 128);
      gguf.Tensor(prefix + "ssm_out.weight", 256);
    }
  }
  const auto path = fixture.root / "tiny.gguf";
  gguf.Save(path);
  const auto row =
      Row("tiny.gguf", std::to_string(std::filesystem::file_size(path)),
          std::string(40, 'a'), gufo::crypto::Sha256FileHex(path));
  const auto h =
      header + "\tcontext\thost_available_bytes\tdevice_available_bytes\n";
  auto report = fixture.Run(h + row + "\t32\t8589934592\t8589934592\n");
  assert(!report.has_failures);
  assert(report.text.find(": UNPROVEN: lower_bound_bytes=") !=
         std::string::npos);
  assert(report.text.find("kv_bytes=256") != std::string::npos);
  assert(report.text.find("derived weights=mapped-file-regions") !=
         std::string::npos);
  report = fixture.Run(h + row + "\t32\t4294967297\t8589934592\n");
  assert(!report.has_failures);  // Device-first weights do not consume host
                                 // capacity.
  assert(report.text.find(": UNPROVEN:") != std::string::npos);
  // Device needs KV, not host-registerable weights: 256 bytes is sufficient
  // for the lower bound even when the complete file is larger than 256 bytes.
  report = fixture.Run(h + row + "\t32\t8589934592\t4294967552\n");
  assert(!report.has_failures);
  report = fixture.Run(h + row + "\t32\t8589934592\t4294967551\n");
  assert(report.has_failures);  // One byte below the device KV boundary.
  report = fixture.Run(h + row + "\t32\t8589934592\t4294967295\n");
  assert(report.has_failures);  // Below the dispatch reserve itself.
  report = fixture.Run(h + row + "\t33\t8589934592\t8589934592\n");
  assert(report.text.find(": NOT-EVALUATED:") != std::string::npos);
}
void NegativeCases(Fixture& fixture) {
  const auto complete_header = header + "\t" + budgets + "\n";
  constexpr std::uint64_t gib = 1ULL << 30;
  const auto carveout =
      "\t32\t" + std::to_string(28 * gib) + "\t" + std::to_string(96 * gib) +
      "\t" + std::to_string(4 * gib) + "\t" + std::to_string(60 * gib) + "\t" +
      std::to_string(gib) + "\t0\t0\t0\t0\t0\n";
  auto report = fixture.Run(complete_header + Row("absent.gguf") + carveout);
  assert(!report.has_failures);
  assert(report.text.find(": UNPROVEN:") != std::string::npos);
  for (const auto& device : {"4294967296", "4294967295"}) {
    report = fixture.Run(complete_header + Row("absent.gguf") +
                         "\t32\t4294967296\t" + device +
                         "\t4294967296\t3\t0\t0\t0\t0\t0\t0\n");
    assert(report.has_failures == (std::string(device) == "4294967295"));
  }
  auto watch = Row();
  watch.replace(watch.find("candidate"), 9, "watch-only-no-download");
  report = fixture.Run(header + "\n" + watch + "\n");
  assert(report.text.find(": NOT-EVALUATED:") != std::string::npos);
  Reject(fixture, header + "\n" + Row("model.gguf", "3", "bad") + "\n");
  Reject(fixture,
         header + "\n" + Row("model.gguf", "3", std::string(40, 'A')) + "\n");
  for (const auto& digest : {std::string(63, 'a'), std::string(64, 'A')})
    Reject(fixture, header + "\n" +
                        Row("model.gguf", "3", std::string(40, 'a'), digest) +
                        "\n");
  Reject(fixture, header + "\thost_available\n" + Row() + "\t32\n");
  for (const auto* name :
       {"context", "device_available_bytes", "cache_bytes"}) {
    try {
      (void)fixture.Run(header + "\t" + name + "\n" + Row() + "\t32K\n");
      assert(false);
    } catch (const std::invalid_argument& error) {
      assert(
          std::string(error.what()).find("row 2 column " + std::string(name)) !=
          std::string::npos);
    }
  }
  Reject(fixture, header + "\tos_reserve_bytes\n" + Row() + "\t1\n");
  std::filesystem::create_directory(fixture.root / "directory.gguf");
  report = fixture.Run(header + "\n" + Row("directory.gguf") + "\n");
  assert(report.has_failures &&
         report.text.find("not a regular file") != std::string::npos);
  std::error_code error;
  std::filesystem::create_symlink(fixture.root / "model.gguf",
                                  fixture.root / "link.gguf", error);
  if (!error) {
    report = fixture.Run(header + "\n" + Row("link.gguf") + "\n");
    assert(report.has_failures &&
           report.text.find("not a regular file") != std::string::npos);
  }
  const auto unreadable = fixture.root / "unreadable.gguf";
  std::ofstream(unreadable) << "abc";
  std::filesystem::permissions(unreadable, std::filesystem::perms::none, error);
  if (!error && !std::ifstream(unreadable).is_open()) {
    report = fixture.Run(header + "\n" + Row("unreadable.gguf") + "\n");
    assert(report.has_failures &&
           report.text.find("local=ERROR:") != std::string::npos);
  }
  std::filesystem::permissions(unreadable, std::filesystem::perms::owner_all,
                               error);
  Gguf unsupported;
  unsupported.Text("general.architecture", "llama");
  const auto path = fixture.root / "unsupported.gguf";
  unsupported.Save(path);
  const auto artifact_row = [&](const std::filesystem::path& p) {
    return Row(p.filename().string(),
               std::to_string(std::filesystem::file_size(p)),
               std::string(40, 'a'), gufo::crypto::Sha256FileHex(p));
  };
  report = fixture.Run(header + "\n" + artifact_row(path) + "\n");
  assert(report.text.find("NOT-EVALUATED: unsupported metadata-only family") !=
         std::string::npos);
  for (int shard = 0; shard < 2; ++shard) {
    Gguf split;
    split.Text("general.architecture", "qwen35");
    split.Number("split.count", 2);
    split.Number("split.no", shard);
    split.Number64("split.tensors.count", 0);
    split.Save(fixture.root / (shard == 0 ? "split-00001-of-00002.gguf"
                                          : "split-00002-of-00002.gguf"));
  }
  const auto first = fixture.root / "split-00001-of-00002.gguf";
  report = fixture.Run(header + "\n" + artifact_row(first) + "\n");
  assert(report.text.find("NOT-EVALUATED: split GGUF") != std::string::npos);
  assert(report.text.find("sibling shards are not evaluated") !=
         std::string::npos);
  std::filesystem::rename(fixture.root / "split-00002-of-00002.gguf",
                          fixture.root / "other.gguf");
  report = fixture.Run(header + "\n" + artifact_row(first) + "\n");
  assert(report.text.find(": NOT-EVALUATED:") != std::string::npos);
}
}  // namespace

int main() {
  const RecoveryBounds recovery({{1000, 100, 4}, {1004, 102, 4}});
  assert(recovery.Contains({1000, 102, 4}));
  assert(recovery.Contains({1100, 90, 3}));
  assert(!recovery.Contains({999, 100, 4}));
  assert(!recovery.Contains({1000, 103, 4}));
  assert(!recovery.Contains({1000, 100, 5}));
  const RecoveryBounds pages({{1000, 100, 4}, {1004, 102, 4}}, 16);
  assert(pages.Contains({984, 118, 4}));  // Inclusive preregistered slack.
  assert(!pages.Contains({983, 118, 4}));
  assert(!pages.Contains({984, 119, 4}));
  assert(!pages.Contains({1000, 100, 5}));  // No thread-count allowance.
  assert(pages.OutsideCount({{983, 100, 4}, {1000, 100, 4}, {1000, 119, 4}}) ==
         2);  // Retain/count all results, not just the first band miss.
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  const RecoveryBounds edges({{0, maximum, 4}}, 16);
  assert(edges.Contains({maximum, maximum, 4}));
  const RecoveryBounds extreme({{maximum, 0, 4}}, 16);
  assert(!extreme.Contains({0, maximum, 4}));
  bool empty_rejected = false;
  try {
    const RecoveryBounds empty({});
  } catch (const std::invalid_argument&) {
    empty_rejected = true;
  }
  assert(empty_rejected);
  const RecoveryTrend growth({{100, 10, 4}, {99, 11, 4}, {98, 11, 4}});
  assert(growth.rss_monotonic_growth && growth.heap_monotonic_loss);
  const RecoveryTrend jitter({{100, 10, 4}, {99, 12, 4}, {101, 11, 4}});
  assert(!jitter.rss_monotonic_growth && !jitter.heap_monotonic_loss);
  assert(!RecoveryTrend({{100, 10, 4}, {100, 10, 4}}).rss_monotonic_growth);
  static_assert(kRecoveryInconclusiveExit != 0 &&
                kRecoveryInconclusiveExit != 1 &&
                kRecoveryInconclusiveExit != 77);
  Fixture fixture;
  const auto unbudgeted = header + "\n" + Row() + "\n";
  auto report = fixture.Run(unbudgeted);
  assert(!report.has_failures);
  assert(report.text.find("insufficient admission evidence") !=
         std::string::npos);
  assert(report.text.find("local=MISSING") != std::string::npos);
  assert(report.text.ends_with(unbudgeted));
  assert(report.text.find("No FIT state") != std::string::npos);
  const auto complete_header = header + "\t" + budgets + "\n";
  report = fixture.Run(complete_header + Row() + Budget() + "\n");
  assert(!report.has_failures);
  assert(report.text.find(": UNPROVEN: lower_bound_bytes=4") !=
         std::string::npos);
  report = fixture.Run(complete_header + Row() + Budget("5") + "\n");
  assert(report.has_failures);
  assert(report.text.find("lower bound alone exceeds reserved capacity") !=
         std::string::npos);
  report = fixture.Run(complete_header + Row() + Budget() + "\n" +
                       Row("alternative.gguf") + Budget() + "\n");
  assert(!report.has_failures);
  report =
      fixture.Run(header + "\n" + Row("unknown", "unknown", "unknown") + "\n");
  assert(report.text.find("NOT-EVALUATED") != std::string::npos);
  std::ofstream(fixture.root / "model.gguf", std::ios::binary) << "abc";
  report = fixture.Run(complete_header + Row() + Budget() + "\n");
  assert(report.text.find("NOT-EVALUATED") != std::string::npos);
  assert(report.text.find("local=size=OK sha256=OK") != std::string::npos);
  std::ofstream(fixture.root / "model.gguf", std::ios::binary) << "corrupt";
  report = fixture.Run(unbudgeted);
  assert(report.has_failures);
  assert(report.text.find("size=MISMATCH sha256=MISMATCH") !=
         std::string::npos);
  Reject(fixture, "");
  Reject(fixture, header + "\n");
  Reject(fixture, header + "\tfile\n");
  Reject(fixture, header + "\nshort\n");
  for (const auto* path : {"../outside", "C:\\outside", "/outside"})
    Reject(fixture, header + "\n" + Row(path) + "\n");
  for (const auto* bytes : {"-1", "18446744073709551616"})
    Reject(fixture, header + "\n" + Row("model.gguf", bytes) + "\n");
  Reject(fixture, complete_header + Row("missing.gguf") +
                      Budget("18446744073709551615") + "\n");
  LocalLowerBound(fixture);
  NegativeCases(fixture);
  report = fixture.Run(header + "\r\n" + Row() + "\r\n");
  assert(report.has_failures);
  std::cout << "model_plan_test passed\n";
}
