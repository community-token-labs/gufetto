// Replays a routing trace through the expert-stream policy (no I/O, no HIP).
// Driven by tools/bench/expert_trace.py. Input on stdin, one record per line:
//   config <lru|scored> <cap> <chunk> <stage> <max_read> <class>[,<class>...]
//   plan <optional 0|1> <routes>   followed by <routes> lines of
//     <step> <d|p> <layer> <expert> <shard> <off0> <len0> <off1> <len1> <off2> <len2>
//   reset                          (zero counters; keep the cache; echoed)
// Every plan is completed at once. Output, one line per plan:
//   plan <layer> <first step> <hits> <misses> <demand bytes> <optional bytes>
//        <reads> <deferred> <staged>
// and a final `stats` line with allocation gauges.
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/expert_stream/policy.hpp"

namespace es = gufo::core::expert_stream;

int main() {
  std::ios::sync_with_stdio(false);
  std::optional<es::ExpertPool> pool;
  std::string word;
  while (std::cin >> word) {
    if (word == "config") {
      std::string policy, classes;
      es::PoolConfig config;
      std::cin >> policy >> config.cap_bytes >> config.chunk_bytes >>
          config.stage_bytes >> config.max_read_bytes >> classes;
      config.policy =
          policy == "lru" ? es::Policy::kLru : es::Policy::kScanResistant;
      std::istringstream list(classes);
      for (std::string item; std::getline(list, item, ',');) {
        config.class_bytes.push_back(std::stoull(item));
      }
      pool.emplace(std::move(config));
      if (!pool->Valid()) {
        std::cerr << "invalid config\n";
        return 2;
      }
    } else if (word == "reset" && pool) {
      pool->ResetCounters();
      std::cout << "reset\n";
    } else if (word == "plan" && pool) {
      int optional = 0;
      std::size_t count = 0;
      std::cin >> optional >> count;
      std::vector<es::ExpertRoute> routes(count);
      for (es::ExpertRoute& route : routes) {
        char phase = 'd';
        std::uint32_t shard = 0;
        std::cin >> route.step >> phase >> route.layer >> route.expert >> shard;
        route.phase = phase == 'p' ? es::Phase::kPrefill : es::Phase::kDecode;
        route.encoded_bytes = 0;
        for (es::SourceRange& range : route.source) {
          range.shard = shard;
          std::cin >> range.offset >> range.bytes;
          route.encoded_bytes += range.bytes;
        }
      }
      const es::Stats before = pool->stats();
      es::Plan plan;
      if (!std::cin ||
          pool->Prepare(routes, optional != 0, &plan) != es::PlanError::kNone) {
        std::cerr << "bad plan\n";
        return 2;
      }
      std::size_t staged = 0;
      for (const es::Lease& lease : plan.leases) {
        staged += lease.kind == es::LeaseKind::kStage ? 1 : 0;
      }
      if (pool->Complete(plan.id) != es::PlanError::kNone) {
        return 2;
      }
      const es::Stats& after = pool->stats();
      std::cout << "plan " << plan.layer << ' ' << routes.front().step << ' '
                << after.route_hits - before.route_hits << ' '
                << after.route_misses - before.route_misses << ' '
                << after.demand_read_bytes - before.demand_read_bytes << ' '
                << after.optional_read_bytes - before.optional_read_bytes
                << ' ' << plan.reads.size() << ' ' << plan.deferred.size()
                << ' ' << staged << '\n';
    } else {
      std::cerr << "unexpected record: " << word << '\n';
      return 2;
    }
  }
  if (!pool) {
    return 2;
  }
  const es::Stats& s = pool->stats();
  std::cout << "stats " << s.allocated_bytes << ' ' << s.peak_allocated_bytes
            << ' ' << s.resident_encoded_bytes << ' '
            << pool->FragmentationBytes() << ' ' << s.evictions << '\n';
  return 0;
}
