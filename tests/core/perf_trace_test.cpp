#include "src/core/perf_trace.hpp"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include "src/core/json.hpp"
#include "src/core/platform/env.hpp"

namespace {
std::filesystem::path trace_path;
void RemoveTrace() {
  std::error_code ec;
  std::filesystem::remove(trace_path, ec);
}
void TestCachePersistPiece() {
  using namespace gufo;
  perf::Record record{};
  record.kind = "cache_persist_piece";
  assert(json::parse(perf::Serialize(record, 1)).find("cache_persist_piece") ==
         nullptr);
  record.cache_persist_piece = perf::CachePersistPiece{
      9007199254740993ULL, {}, 9007199254740995ULL, 1048576, true};
  auto& file = record.cache_persist_piece->file;
  for (std::size_t i = 0; i < file.size(); ++i)
    file[i] = static_cast<std::uint8_t>(240 + i);
  auto* previous = std::exchange(perf::current, &record);
  for (const auto* label :
       {"cache_persist_source", "hip_transfer_owned_d2h_enqueue",
        "hip_transfer_owned_event_record", "cache_persist_completion_wait",
        "hip_transfer_owned_event_wait", "hip_transfer_owned_wait_drain",
        "cache_persist_stream_teardown",
        "hip_transfer_owned_stream_destructor_drain",
        "hip_transfer_owned_lease_destructor_drain"}) {
    perf::HostPhase phase(label);
    phase.Finish();
    phase.Finish();  // Explicit finish plus destruction records only once.
  }
  perf::current = previous;
  const auto serialized = perf::Serialize(record, perf::Now());
  const auto value = json::parse(serialized);
  const auto* piece = value.find("cache_persist_piece");
  assert(piece && piece->members().size() == 5);
  assert(serialized.find("\"checkpoint\":9007199254740993") !=
         std::string::npos);
  assert(serialized.find("\"offset\":9007199254740995") != std::string::npos);
  assert(piece->member_size("bytes") == 1048576);
  assert(piece->find("private_file")->as_bool());
  const auto& bytes = piece->find("file")->items();
  assert(bytes.size() == 16);
  for (std::size_t i = 0; i < bytes.size(); ++i)
    assert(bytes[i].is_number() && bytes[i].as_size() == 240 + i);
  const auto& phases = value.find("phases")->items();
  assert(phases.size() == 9);
  for (const auto& phase : phases) {
    assert(!phase.member_str("name").empty());
    assert(phase.member_size("end_ns") >= phase.member_size("start_ns"));
  }
  // No source contents or identifiers other than the numeric internal IDs.
  for (const auto* key : {"path", "pointer", "source", "payload", "tokens",
                          "prompt", "prompt_text", "input"}) {
    assert(piece->find(key) == nullptr);
    assert(value.find(key) == nullptr);
  }
}
void TestMtpCounters() {
  using namespace gufo;
  perf::Record record{};
  record.kind = "decode_batch";
  record.sessions.push_back({3, 42, 4096});
  const auto empty = perf::Serialize(record, 1);
  assert(empty.find("mtp_heads") == std::string::npos);
  assert(empty.find("mtp_cycles") == std::string::npos);
  assert(empty.find("draft_tables") == std::string::npos);
  auto* previous = std::exchange(perf::current, &record);
  perf::TraceMtpHead(1, 1, 1, 98304, 248320, 1024);
  perf::TraceMtpHead(1, 0, 0, 98304, 248320, 1024);
  perf::TraceMtpHead(4, 4, 4, 98304, 248320, 1024);
  perf::TraceMtpHead(4, 3, 3, 98304, 248320, 1024);
  perf::TraceMtpHead(4, 0, 0, 98304, 248320, 1024);
  perf::TraceMtpHead(4, 3, 0, 98304, 248320, 1024);
  // Two neural proposals, then three lookup copies. Anchor is never a draft.
  perf::TraceMtpCycle(4096, 5, 1, 3, 2);
  perf::TraceMtpCycle(4098, 5, 4, 3, 5);
  perf::TraceMtpCycle(4103, 3, 3, 3, 4);  // lookup-only
  perf::TraceMtpCycle(4107, 0, 0, 0, 1);  // AR anchor
  perf::TraceMtpCycle(4108, 2, 2, 0, 3);  // neural-only
  record.draft_tables.push_back({4096, 17, 1024});
  perf::current = previous;
  const auto serialized = perf::Serialize(record, 2);
  const auto value = json::parse(serialized);
  const auto& heads = value.find("mtp_heads")->items();
  assert(heads.size() == 6);
  constexpr const char* modes[] = {"scalar_subset", "scalar_full",
                                   "all_subset",    "mixed_full_gather",
                                   "full",          "full"};
  for (std::size_t i = 0; i < heads.size(); ++i)
    assert(heads[i].member_str("mode") == modes[i]);
  assert(heads[2].member_size("logical_subset_rows") == 4);
  const auto& subset = heads[2].find("projections")->items();
  assert(subset.size() == 5);
  assert(subset[0].member_str("kind") == "prefix");
  assert(subset[0].member_size("vocabulary_rows") == 98304);
  assert(subset[0].member_size("activation_rows") == 4);
  assert(subset[0].member_size("logit_values") == 393216);
  for (std::size_t i = 1; i < subset.size(); ++i) {
    assert(subset[i].member_str("kind") == "table");
    assert(subset[i].member_size("vocabulary_rows") == 1024);
    assert(subset[i].member_size("activation_rows") == 1);
  }
  for (const auto i : {3, 4, 5}) {
    const auto& ops = heads[i].find("projections")->items();
    assert(ops.size() == 1);
    assert(ops[0].member_str("kind") == "full");
    assert(ops[0].member_size("vocabulary_rows") == 248320);
    assert(ops[0].member_size("activation_rows") == 4);
    assert(ops[0].member_size("logit_values") == 993280);
  }
  assert(heads[3].member_str("fallback") == "mixed");
  assert(heads[5].member_str("fallback") == "scratch");
  assert(heads[5].member_size("logical_subset_rows") == 3);
  assert(heads[5].member_size("subset_rows") == 0);
  const auto& cycles = value.find("mtp_cycles")->items();
  assert(cycles[0].member_size("neural_proposed") == 2);
  assert(cycles[0].member_size("neural_accepted") == 1);
  assert(cycles[0].member_size("lookup_accepted") == 0);
  assert(cycles[0].member_size("committed") == 2);
  assert(cycles[1].member_size("neural_accepted") == 2);
  assert(cycles[1].member_size("lookup_accepted") == 2);
  assert(cycles[2].member_size("neural_proposed") == 0);
  assert(cycles[2].member_size("lookup_accepted") == 3);
  assert(cycles[3].member_size("committed") == 1);
  assert(cycles[3].member_size("neural_proposed") == 0);
  assert(cycles[4].member_size("neural_accepted") == 2);
  assert(cycles[4].member_size("lookup_proposed") == 0);
  assert(value.find("draft_tables")->items()[0].member_size("used_rows") == 17);
  assert(value.find("sessions")->items()[0].member_size("request_id") == 42);
  assert(serialized.find("\"tokens\"") == std::string::npos);
  assert(serialized.find("prompt_text") == std::string::npos);
}
}  // namespace

int main(int argc, char**) {
  using namespace gufo;
  if (argc > 1) {
    assert(platform::UnsetEnv("GUFO_PERF_TRACE"));
    assert(platform::UnsetEnv("GUFO_PERF_TRACE_MOE"));
    perf::Work work("decode_single");
    perf::Forward forward(false);
    perf::HostPhase phase("target_forward");
    assert(!perf::Enabled());
    assert(work.get() == nullptr);
    assert(perf::current == nullptr);
    assert(perf::forward_counter == 0);
    assert(!perf::sample_moe);
    perf::TraceMtpHead(4, 4, 4, 98304, 248320, 1024);
    perf::TraceMtpCycle(4096, 5, 4, 3, 5);
    assert(perf::current == nullptr);
    {
      perf::Work piece("cache_persist_piece");
      if (auto* record = piece.get())
        record->cache_persist_piece =
            perf::CachePersistPiece{1, {}, 0, 16, false};
      perf::HostPhase enqueue("hip_transfer_owned_d2h_enqueue");
      perf::HostPhase wait("hip_transfer_owned_event_wait");
      perf::HostPhase drain("hip_transfer_owned_wait_drain");
      assert(piece.get() == nullptr && perf::current == nullptr);
    }
    assert(perf::current == nullptr);
    return 0;
  }
  const auto path =
      std::filesystem::temp_directory_path() /
      ("gufo-perf-trace-" + std::to_string(platform::ProcessId()) + ".jsonl");
  std::filesystem::remove(path);
  trace_path = path;
  // Register before constructing Sink(): its stream must close before the
  // cleanup runs, because Windows cannot unlink an open ofstream.
  assert(std::atexit(RemoveTrace) == 0);
  assert(platform::SetEnv("GUFO_PERF_TRACE", path.string()));
  assert(platform::SetEnv("GUFO_PERF_TRACE_MOE", "2"));
  assert(perf::Enabled());
  TestMtpCounters();
  TestCachePersistPiece();
  {
    perf::Work work("prefill");
    auto* record = work.get();
    assert(record != nullptr);
    record->sessions.push_back({3, 42, 4096});
    record->tokens_requested = record->tokens_processed = 128;
    {
      perf::Forward first(true);
      first.Rows(4096, 128);
      assert(!perf::sample_moe);
    }
    {
      perf::Forward second(true);
      second.Rows(4224, 128);
      assert(perf::sample_moe);
      perf::layer_id = 7;
      perf::RouteWait(25);
      perf::RouteWait(75);
      perf::Layer layer{perf::forward_id, 7, 128, true};
      layer.experts[0] = 1;
      layer.experts[7] = std::uint64_t{1} << 63;
      record->moe.push_back(layer);
    }
    assert(perf::forward_id == 0);
    assert(!perf::sample_moe);
    assert(record->route_waits[0].count == 2);
    assert(record->route_waits[0].ns == 100);
    perf::HostPhase phase("prefill_trunk_forward");
    {
      perf::Work nested("capture");
      assert(perf::current != record);
    }
    assert(perf::current == record);
    record->start_ns = 9007199254740993ULL;
    const auto serialized = perf::Serialize(*record, 9007199254740994ULL);
    assert(serialized.find("9007199254740993") != std::string::npos);
    assert(serialized.find("9007199254740994") != std::string::npos);
    record->start_ns = perf::Now();
  }
  assert(perf::current == nullptr);
  // Concurrent append calls must produce whole JSONL objects.
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t)
    threads.emplace_back([] {
      for (int i = 0; i < 16; ++i) {
        perf::Work work("decode_single");
        work.get()->sessions.push_back({0, 99, 8192});
        perf::Decode(0, 3, 2);
      }
    });
  for (auto& thread : threads)
    thread.join();

  std::ifstream input(path);
  std::string line;
  std::size_t lines = 0;
  bool found_mask = false;
  while (std::getline(input, line)) {
    const auto value = json::parse(line);
    assert(value.member_size("schema") == 1);
    assert(value.member_size("pid") ==
           static_cast<std::size_t>(platform::ProcessId()));
    if (line.find("\"kind\":\"prefill\"") != std::string::npos) {
      const auto bits = value.find("moe")->items()[0].member_str("experts");
      assert(bits.size() == 128);
      assert(bits.front() == '8' && bits.back() == '1');
      found_mask = true;
    }
    ++lines;
  }
  assert(lines == 66);
  assert(found_mask);
  input.close();
}
