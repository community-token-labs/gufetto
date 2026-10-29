#ifndef GUFO_CORE_PERF_TRACE_HPP_
#define GUFO_CORE_PERF_TRACE_HPP_

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <locale>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "src/core/platform/os_info.hpp"

namespace gufo::perf {

inline std::uint64_t Now() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Configuration is read once. An invalid destination disables all hooks.
class Writer {
public:
  Writer() {
    const char* path = std::getenv("GUFO_PERF_TRACE");
    if (path == nullptr || *path == '\0')
      return;
    run_id = Now();
    process_id = static_cast<std::uint64_t>(platform::ProcessId());
    stream_.open(path, std::ios::app);
    enabled = stream_.is_open();
    if (!enabled) {
      std::fputs("GUFO_PERF_TRACE: cannot open trace destination\n", stderr);
      return;
    }
    if (const char* sample = std::getenv("GUFO_PERF_TRACE_MOE")) {
      const std::string_view value(sample);
      std::uint64_t parsed = 0;
      const auto result =
          std::from_chars(value.data(), value.data() + value.size(), parsed);
      if (result.ec == std::errc{} && result.ptr == value.data() + value.size())
        moe_every = parsed;
    }
  }

  void Append(const std::string& line) noexcept {
    try {
      const std::lock_guard<std::mutex> lock(mutex_);
      stream_ << line << '\n';
      // One buffered append per work unit; make completed units visible to
      // live readers and preserve them if a later device call fails.
      stream_.flush();
    } catch (...) {
    }
  }

  bool enabled{false};
  std::uint64_t moe_every{0}, run_id{0}, process_id{0};

private:
  std::mutex mutex_;
  std::ofstream stream_;
};

inline Writer& Sink() {
  static Writer writer;
  return writer;
}
inline bool Enabled() {
  return Sink().enabled;
}

struct Session {
  std::uint64_t slot, request_id, context_depth;
  std::uint64_t rows{0}, accepted_tokens{0};
};
struct Phase {
  const char* name;
  std::uint64_t start_ns, end_ns;
};
struct Layer {
  std::uint64_t forward, layer, rows;
  bool prefill;
  std::array<std::uint64_t, 8> experts{};
};
struct Wait {
  std::uint64_t forward, layer, count{0}, ns{0};
};
struct Gpu {
  std::uint64_t forward, host_start_ns, host_end_ns, stream_elapsed_ns;
};
struct Shape {
  std::uint64_t forward;
  bool prefill;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> sessions;
};
// Vocabulary rows are weight-matrix rows per projection, not output elements.
struct Projection {
  const char* kind;
  std::uint64_t vocabulary_rows, activation_rows;
};
struct MtpHead {
  const char* mode;
  const char* fallback;
  std::uint64_t cohort, logical_subset_rows, subset_rows;
  std::vector<Projection> projections;
};
struct MtpCycle {
  std::uint64_t context_depth, neural_proposed, neural_accepted;
  std::uint64_t lookup_proposed, lookup_accepted, committed;
};
struct DraftTable {
  std::uint64_t context_depth, used_rows, capacity;
};
// Internal identifiers only: file is the 16 numeric bytes of a disk file ID,
// never a path, pointer, prompt or payload. One record covers one staging
// piece.
struct CachePersistPiece {
  std::uint64_t checkpoint;
  std::array<std::uint8_t, 16> file;
  std::uint64_t offset, bytes;
  bool private_file;
};
struct Record {
  const char* kind;
  std::uint64_t start_ns{Now()}, thread_id{0};
  std::size_t prefilling_waiting{0}, decoders_due{0}, capturing{0}, waiting{0};
  std::uint64_t tokens_requested{0}, tokens_budget{0}, tokens_processed{0};
  bool through{false}, checkpoint{false};
  std::vector<Session> sessions;
  std::vector<Phase> phases;
  std::vector<Layer> moe;
  std::vector<Wait> route_waits;
  std::vector<Gpu> gpu;
  std::vector<Shape> forwards;
  std::vector<MtpHead> mtp_heads;
  std::vector<MtpCycle> mtp_cycles;
  std::vector<DraftTable> draft_tables;
  std::optional<CachePersistPiece> cache_persist_piece;
};
inline thread_local Record* current = nullptr;
inline thread_local std::uint64_t forward_counter = 0;
inline thread_local std::uint64_t forward_id = 0;
inline thread_local std::uint64_t layer_id = 0;
inline thread_local bool sample_moe = false;
inline thread_local bool prompt_rows = false;

// All labels come from fixed call sites; no model inputs, tokens or text enter
// the collector. Integers are serialized directly (no JSON double conversion).
inline std::string Serialize(const Record& r, std::uint64_t end) {
  std::ostringstream o;
  o.imbue(std::locale::classic());
  o << "{\"schema\":1,\"run_id\":" << Sink().run_id
    << ",\"pid\":" << Sink().process_id << ",\"thread_id\":" << r.thread_id
    << ",\"kind\":\"" << r.kind << "\",\"start_ns\":" << r.start_ns
    << ",\"end_ns\":" << end
    << ",\"queue\":{\"prefilling_waiting\":" << r.prefilling_waiting
    << ",\"decoders_due\":" << r.decoders_due
    << ",\"capturing\":" << r.capturing << ",\"waiting\":" << r.waiting
    << "},\"tokens_budget\":" << r.tokens_budget
    << ",\"tokens_requested\":" << r.tokens_requested
    << ",\"tokens_processed\":" << r.tokens_processed
    << ",\"through\":" << (r.through ? "true" : "false")
    << ",\"checkpoint\":" << (r.checkpoint ? "true" : "false")
    << ",\"sessions\":[";
  const char* sep = "";
  for (const auto& s : r.sessions) {
    o << sep << "{\"slot\":" << s.slot << ",\"request_id\":" << s.request_id
      << ",\"context_depth\":" << s.context_depth << ",\"rows\":" << s.rows
      << ",\"accepted_tokens\":" << s.accepted_tokens << '}';
    sep = ",";
  }
  o << "],\"forwards\":[";
  sep = "";
  for (const auto& f : r.forwards) {
    o << sep << "{\"forward\":" << f.forward
      << ",\"prefill\":" << (f.prefill ? "true" : "false") << ",\"sessions\":[";
    const char* row_sep = "";
    for (const auto& [depth, rows] : f.sessions) {
      o << row_sep << "{\"context_depth\":" << depth << ",\"rows\":" << rows
        << '}';
      row_sep = ",";
    }
    o << "]}";
    sep = ",";
  }
  o << "],\"phases\":[";
  sep = "";
  for (const auto& p : r.phases) {
    o << sep << "{\"name\":\"" << p.name << "\",\"start_ns\":" << p.start_ns
      << ",\"end_ns\":" << p.end_ns << '}';
    sep = ",";
  }
  o << "],\"gpu\":[";
  sep = "";
  for (const auto& g : r.gpu) {
    o << sep << "{\"forward\":" << g.forward
      << ",\"host_start_ns\":" << g.host_start_ns
      << ",\"host_end_ns\":" << g.host_end_ns
      << ",\"stream_elapsed_ns\":" << g.stream_elapsed_ns << '}';
    sep = ",";
  }
  o << "],\"route_waits\":[";
  sep = "";
  for (const auto& w : r.route_waits) {
    o << sep << "{\"forward\":" << w.forward << ",\"layer\":" << w.layer
      << ",\"count\":" << w.count << ",\"ns\":" << w.ns << '}';
    sep = ",";
  }
  o << "],\"moe\":[";
  sep = "";
  constexpr char hex[] = "0123456789abcdef";
  for (const auto& l : r.moe) {
    o << sep << "{\"forward\":" << l.forward << ",\"layer\":" << l.layer
      << ",\"rows\":" << l.rows << ",\"group\":\""
      << (l.prefill ? "prompt" : "decoder") << "\",\"experts\":\"";
    for (auto word = l.experts.rbegin(); word != l.experts.rend(); ++word)
      for (int shift = 60; shift >= 0; shift -= 4)
        o << hex[(*word >> shift) & 15];
    o << "\"}";
    sep = ",";
  }
  o << "]";
  if (!r.mtp_heads.empty()) {
    o << ",\"mtp_heads\":[";
    sep = "";
    for (const auto& h : r.mtp_heads) {
      o << sep << "{\"mode\":\"" << h.mode << "\",\"fallback\":\"" << h.fallback
        << "\",\"cohort\":" << h.cohort
        << ",\"logical_subset_rows\":" << h.logical_subset_rows
        << ",\"subset_rows\":" << h.subset_rows << ",\"projections\":[";
      const char* projection_sep = "";
      for (const auto& p : h.projections) {
        o << projection_sep << "{\"kind\":\"" << p.kind
          << "\",\"vocabulary_rows\":" << p.vocabulary_rows
          << ",\"activation_rows\":" << p.activation_rows
          << ",\"logit_values\":" << p.vocabulary_rows * p.activation_rows
          << '}';
        projection_sep = ",";
      }
      o << "]}";
      sep = ",";
    }
    o << ']';
  }
  if (!r.mtp_cycles.empty()) {
    o << ",\"mtp_cycles\":[";
    sep = "";
    for (const auto& c : r.mtp_cycles) {
      o << sep << "{\"context_depth\":" << c.context_depth
        << ",\"neural_proposed\":" << c.neural_proposed
        << ",\"neural_accepted\":" << c.neural_accepted
        << ",\"lookup_proposed\":" << c.lookup_proposed
        << ",\"lookup_accepted\":" << c.lookup_accepted
        << ",\"committed\":" << c.committed << '}';
      sep = ",";
    }
    o << ']';
  }
  if (!r.draft_tables.empty()) {
    o << ",\"draft_tables\":[";
    sep = "";
    for (const auto& t : r.draft_tables) {
      o << sep << "{\"context_depth\":" << t.context_depth
        << ",\"used_rows\":" << t.used_rows << ",\"capacity\":" << t.capacity
        << '}';
      sep = ",";
    }
    o << ']';
  }
  if (r.cache_persist_piece) {
    const auto& p = *r.cache_persist_piece;
    o << ",\"cache_persist_piece\":{\"checkpoint\":" << p.checkpoint
      << ",\"file\":[";
    sep = "";
    for (const auto byte : p.file) {
      o << sep << static_cast<unsigned>(byte);
      sep = ",";
    }
    o << "],\"offset\":" << p.offset << ",\"bytes\":" << p.bytes
      << ",\"private_file\":" << (p.private_file ? "true" : "false") << '}';
  }
  o << '}';
  return o.str();
}

class Work {
public:
  explicit Work(const char* kind) {
    if (!Enabled())
      return;
    record_ = std::make_unique<Record>();
    record_->kind = kind;
    static std::atomic<std::uint64_t> next_thread{0};
    static thread_local const auto thread =
        next_thread.fetch_add(1, std::memory_order_relaxed);
    record_->thread_id = thread;
    previous_ = std::exchange(current, record_.get());
  }
  ~Work() {
    if (!record_)
      return;
    const auto end = Now();
    current = previous_;
    try {
      Sink().Append(Serialize(*record_, end));
    } catch (...) {
      // Tracing must not turn a completed work unit into a serving failure.
    }
  }
  Record* get() const { return record_.get(); }

private:
  std::unique_ptr<Record> record_;
  Record* previous_{nullptr};
};

inline void Decode(std::uint64_t slot, std::uint64_t rows,
                   std::uint64_t accepted) {
  if (current == nullptr)
    return;
  for (auto& s : current->sessions)
    if (s.slot == slot) {
      s.rows += rows;
      s.accepted_tokens += accepted;
      return;
    }
}

// Called outside graph bodies: capture is not execution, replay skips bodies.
// Each entry describes one completed head's projection operations, not GPU
// hardware counters or bytes fetched from memory/cache.
inline void TraceMtpHead(std::uint64_t cohort, std::uint64_t logical_subset,
                         std::uint64_t subset, std::uint64_t prefix,
                         std::uint64_t full, std::uint64_t table) {
  if (current == nullptr)
    return;
  const bool all = subset == cohort;
  const char* mode = cohort == 1   ? (all ? "scalar_subset" : "scalar_full")
                     : all         ? "all_subset"
                     : subset != 0 ? "mixed_full_gather"
                                   : "full";
  MtpHead h{mode,
            logical_subset != 0 && subset == 0 ? "scratch"
            : subset != 0 && !all              ? "mixed"
            : subset == 0                      ? "logical_full"
                                               : "none",
            cohort,
            logical_subset,
            subset,
            {}};
  if (all) {
    h.projections.push_back({"prefix", prefix, cohort});
    for (std::uint64_t i = 0; i < cohort; ++i)
      h.projections.push_back({"table", table, 1});
  } else {
    h.projections.push_back({"full", full, cohort});
  }
  current->mtp_heads.push_back(std::move(h));
}

// proposed/accepted exclude the anchor; committed includes it. Lookup copies
// trail neural proposals, so the kept prefix classifies acceptance exactly.
inline void TraceMtpCycle(std::uint64_t depth, std::uint64_t proposed,
                          std::uint64_t accepted, std::uint64_t lookup,
                          std::uint64_t committed) {
  if (current == nullptr)
    return;
  const auto neural = proposed - lookup;
  const auto neural_accepted = accepted < neural ? accepted : neural;
  current->mtp_cycles.push_back({depth, neural, neural_accepted, lookup,
                                 accepted - neural_accepted, committed});
}

class HostPhase {
public:
  explicit HostPhase(const char* name) : name_(name) {
    if (current != nullptr)
      start_ = Now();
  }
  void Finish() noexcept {
    if (start_ != 0) {
      const auto start = std::exchange(start_, 0);
      try {
        current->phases.push_back({name_, start, Now()});
      } catch (...) {
      }
    }
  }
  ~HostPhase() { Finish(); }

private:
  const char* name_;
  std::uint64_t start_{0};
};

// Only target forwards count toward sampling; predictor layers are excluded.
class Forward {
public:
  explicit Forward(bool prefill)
      : old_id_(forward_id), old_sample_(sample_moe), old_prompt_(prompt_rows) {
    if (current == nullptr)
      return;
    forward_id = ++forward_counter;
    prompt_rows = prefill;
    const auto every = Sink().moe_every;
    sample_moe = every != 0 && forward_id % every == 0;
    current->forwards.push_back({forward_id, prefill, {}});
    active_ = true;
  }
  void Rows(std::uint64_t depth, std::uint64_t rows) {
    if (active_)
      current->forwards.back().sessions.emplace_back(depth, rows);
  }
  ~Forward() { Finish(); }
  void Finish() {
    if (active_) {
      active_ = false;
      forward_id = old_id_;
      sample_moe = old_sample_;
      prompt_rows = old_prompt_;
    }
  }

private:
  std::uint64_t old_id_;
  bool old_sample_, old_prompt_, active_{false};
};

inline void RouteWait(std::uint64_t ns) {
  if (current == nullptr)
    return;
  auto& waits = current->route_waits;
  if (waits.empty() || waits.back().forward != forward_id ||
      waits.back().layer != layer_id)
    waits.push_back({forward_id, layer_id});
  ++waits.back().count;
  waits.back().ns += ns;
}

}  // namespace gufo::perf

#endif  // GUFO_CORE_PERF_TRACE_HPP_
