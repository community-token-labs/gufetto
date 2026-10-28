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
