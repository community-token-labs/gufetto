#include "src/core/diagnostics/model_plan.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"

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
  assert(report.has_failures);
  assert(report.text.find(": DOES-NOT-FIT:") != std::string::npos);
  // Device needs KV, not host-registerable weights: 256 bytes is sufficient
  // for the lower bound even when the complete file is larger than 256 bytes.
  report = fixture.Run(h + row + "\t32\t8589934592\t4294967552\n");
  assert(!report.has_failures);
  report = fixture.Run(h + row + "\t33\t8589934592\t8589934592\n");
  assert(report.text.find(": NOT-EVALUATED:") != std::string::npos);
}
}  // namespace

int main() {
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
  report = fixture.Run(complete_header + Row() + Budget("2") + "\n");
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
  report = fixture.Run(header + "\r\n" + Row() + "\r\n");
  assert(report.has_failures);
  std::cout << "model_plan_test passed\n";
}
