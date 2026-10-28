#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PD_PROFILE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PD_PROFILE_HPP_

// No API, static initialization, runtime predicate, or includes when disabled.
// CMake owns GUFO_ENABLE_PD_PROFILING; undefined and 0 both mean OFF.
#if defined(GUFO_ENABLE_PD_PROFILING) && GUFO_ENABLE_PD_PROFILING
#define GUFO_PD_PROFILE_AVAILABLE 1

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <locale>
#include <optional>
#include <ostream>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/core/perf_trace.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen38_flash_next::pd {

// These are execution provenance, not a route selected from a filtered count.
// Backend MTP batch remains that route even if model filtering leaves one row.
enum class LegacyRoute { kGeneric, kScalarMtp, kBackendMtpBatch };
struct Association {
  std::optional<std::uint64_t> iteration{}, cohort{}, plan{};
  std::optional<LegacyRoute> route{};
  std::optional<std::uint64_t> scheduler_frozen{}, backend_filtered{},
      model_filtered{};
  std::optional<std::uint64_t> starting_state{};
  bool operator==(const Association&) const = default;
};
inline thread_local Association current_association;
// The scheduler/model owner may supply exact plan association. Absent
// association stays null: adjacent forwards, session pointers, or expert sets
// cannot fill it.
class AssociationScope {
public:
  explicit AssociationScope(Association association)
      : previous_(std::exchange(current_association, std::move(association))) {}
  ~AssociationScope() { current_association = previous_; }
  AssociationScope(const AssociationScope&) = delete;
  AssociationScope& operator=(const AssociationScope&) = delete;

private:
  Association previous_;
};

enum class TensorRole { kGate, kUp, kDown };
enum class ForwardRole { kPrompt, kDecoder };
enum class Execution { kUnsupported, kVector, kRoutedF16 };
struct Identity {
  std::uint64_t run{0}, pid{0}, work_start{0}, thread{0}, forward{0};
};
struct Row {
  std::uint32_t forward_row;
  // Process-local session identity only, NOT a request/lineage or state proof.
  std::uint64_t session, context;
};
struct Tensor {
  std::uintptr_t model{0}, allocation{0};
  std::uint32_t layer{0};
  TensorRole tensor{TensorRole::kGate};
  core::GgmlType packing{core::GgmlType::kF32};
  std::uint32_t rows{0}, cols{0}, experts{0};
};
struct Routing {
  Execution execution{Execution::kUnsupported};
  std::uint32_t used{0};
  std::span<const std::int32_t> ids{}, bounds{}, tokens{}, slots{}, tiles{};
  std::uint32_t tile_rows{0}, output_tile_rows{0};
};
struct Assignment {
  std::uint32_t row, slot, expert;
  std::optional<std::uint32_t> compact_row;
};
struct Tile {
  std::uint32_t expert, input_tile, input_tile_rows;
  std::uint32_t weight_row_begin, weight_row_end;
  // Half-open packed payload byte range, relative to tensor allocation.
  std::uint64_t byte_begin, byte_end;
  std::vector<std::uint32_t> assignments;
  // Lane-level loads/cache hits/DRAM bytes are NOT observable from dispatch.
  std::optional<std::uint64_t> load_multiplicity{};
};
struct Map {
  Identity identity;
  Association association;
  Tensor tensor;
  ForwardRole role{ForwardRole::kDecoder};
  Execution execution{Execution::kUnsupported};
  std::vector<Row> rows;
  std::vector<Assignment> assignments;
  std::vector<Tile> tiles;
  std::uint64_t payload_bytes{0}, expected_assignments{0},
      mapped_assignments{0};
  std::optional<std::uint64_t> requested_unique_bytes, requested_packed_blocks;
  bool addresses_complete{false}, tiles_complete{false};
  const char* missing{"not_collected"};
};

inline bool Product(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
  if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b)
    return false;
  out = a * b;
  return true;
}
// DeviceTensor expert weights retain GGUF row-major [expert][M][K] storage;
// device_model.cpp::Uploader::Copy does not repack them. Only payload bytes are
// covered, not its 4096-byte safety tail or unrelated scratch/activations.
inline std::uint64_t WeightBlockBytes(core::GgmlType packing) {
  switch (packing) {
    case core::GgmlType::kQ4_K:
      return sizeof(quant::block_q4_K);
    case core::GgmlType::kQ5_K:
      return sizeof(quant::block_q5_K);
    case core::GgmlType::kQ8_0:
      return sizeof(quant::block_q8_0);
    case core::GgmlType::kQ5_1:
      return 24;  // GGUF: two fp16 scalars, 4 high-bit bytes, 16 code bytes.
    default:
      return 0;
  }
}
inline std::uint64_t RowBytes(const Tensor& t) {
  const auto elements = quant::QuantizedBlockElements(t.packing);
  return t.cols != 0 && elements != 0 && t.cols % elements == 0
             ? t.cols / elements * WeightBlockBytes(t.packing)
             : 0;
}
inline Map Missing(Map m, const char* reason) {
  m.assignments.clear();
  m.tiles.clear();
  m.mapped_assignments = 0;
  m.requested_unique_bytes.reset();
  m.requested_packed_blocks.reset();
  m.addresses_complete = m.tiles_complete = false;
  m.missing = reason;
  return m;
}

// CPU-only decoder of OBSERVED routing. In the F16 case tokens/slots must be
// downloaded from RoutedScatterKernel: atomic order must never be reconstructed
// by sorting IDs. Every live assignment, padding sentinel, and launched tile is
// checked. An invalid/partial observation cannot become a positive overlap.
inline Map BuildMap(const Tensor& tensor, ForwardRole role,
                    std::span<const Row> rows, const Routing& r,
                    const Association& association = current_association) {
  Map m;
  m.tensor = tensor;
  m.role = role;
  m.execution = r.execution;
  m.association = association;
  m.rows.assign(rows.begin(), rows.end());
  if (!Product(rows.size(), r.used, m.expected_assignments) ||
      m.expected_assignments > std::numeric_limits<std::uint32_t>::max())
    return Missing(std::move(m), "assignment_overflow");
  const auto row_bytes = RowBytes(tensor);
  std::uint64_t expert_bytes = 0;
  if (tensor.model == 0 || tensor.allocation == 0 || tensor.rows == 0 ||
      tensor.experts == 0 || row_bytes == 0 ||
      !Product(row_bytes, tensor.rows, expert_bytes) ||
      !Product(expert_bytes, tensor.experts, m.payload_bytes) ||
      m.payload_bytes >
          std::numeric_limits<std::uintptr_t>::max() - tensor.allocation)
    return Missing(std::move(m), "unsupported_or_unbounded_packing");
  if (r.execution == Execution::kUnsupported)
    return Missing(std::move(m), "unobserved_dispatch_layout");
  if (rows.empty() || r.used == 0 || r.used > tensor.experts ||
      r.ids.size() != m.expected_assignments)
    return Missing(std::move(m), "missing_or_invalid_ids");
  std::set<std::uint32_t> forward_rows;
  std::set<std::pair<std::uint64_t, std::uint64_t>> session_rows;
  for (const auto& row : rows)
    if (!forward_rows.insert(row.forward_row).second || row.session == 0 ||
        !session_rows.emplace(row.session, row.context).second)
      return Missing(std::move(m), "aliased_context_rows");
  std::vector<std::uint32_t> counts(tensor.experts, 0);
  for (std::uint32_t i = 0; i < r.ids.size(); ++i) {
    const auto e = r.ids[i];
    if (e < 0 || static_cast<std::uint32_t>(e) >= tensor.experts)
      return Missing(std::move(m), "invalid_expert_id");
    ++counts[e];
    m.assignments.push_back(
        {i / r.used, i % r.used, static_cast<std::uint32_t>(e), std::nullopt});
  }
  const auto append = [&](std::uint32_t expert, std::uint32_t tile,
                          std::uint32_t height, std::uint32_t first,
                          std::uint32_t end,
                          const std::vector<std::uint32_t>& members) {
    m.tiles.push_back({expert, tile, height, first, end,
                       expert * expert_bytes + first * row_bytes,
                       expert * expert_bytes + end * row_bytes, members});
  };
  if (r.execution == Execution::kVector) {
    // Exact logical requested packed-address set for each expert and all its
    // row/slot users. mmq's private grouped-vector tiles are not exposed here;
    // neither their physical multiplicity nor physical tile coverage is
    // claimed.
    for (std::uint32_t e = 0; e < tensor.experts; ++e) {
      std::vector<std::uint32_t> members;
      for (std::uint32_t i = 0; i < r.ids.size(); ++i)
        if (r.ids[i] == static_cast<std::int32_t>(e))
          members.push_back(i);
      if (!members.empty())
        append(e, 0, 0, 0, tensor.rows, members);
    }
  } else {
    if ((r.tile_rows != 16 && r.tile_rows != 48 && r.tile_rows != 64 &&
         r.tile_rows != 128) ||
        (r.output_tile_rows != 64 && r.output_tile_rows != 128) ||
        r.bounds.size() != std::size_t{tensor.experts} + 1 ||
        r.bounds.front() != 0 || r.tokens.size() != r.slots.size())
      return Missing(std::move(m), "missing_compaction");
    std::uint64_t padded = 0;
    for (std::uint32_t e = 0; e < tensor.experts; ++e) {
      padded += (std::uint64_t{counts[e]} + 15) / 16 * 16;
      if (r.bounds[e + 1] < 0 ||
          static_cast<std::uint64_t>(r.bounds[e + 1]) != padded ||
          padded > r.tokens.size())
        return Missing(std::move(m), "invalid_padded_bounds");
    }
    if (padded != r.tokens.size())
      return Missing(std::move(m), "partial_compaction");
    std::vector<bool> seen(r.ids.size(), false);
    for (std::uint32_t e = 0; e < tensor.experts; ++e) {
      for (auto at = r.bounds[e]; at < r.bounds[e + 1]; ++at) {
        const auto token = r.tokens[at], slot = r.slots[at];
        if (static_cast<std::uint32_t>(at - r.bounds[e]) >= counts[e]) {
          if (token != -1 || slot != -1)
            return Missing(std::move(m), "nonempty_padding");
          continue;
        }
        if (slot < 0 || static_cast<std::size_t>(slot) >= r.ids.size() ||
            token < 0 ||
            static_cast<std::uint32_t>(token) !=
                static_cast<std::uint32_t>(slot) / r.used ||
            r.ids[slot] != static_cast<std::int32_t>(e) || seen[slot])
          return Missing(std::move(m), "aliased_or_invalid_compaction");
        seen[slot] = true;
        m.assignments[slot].compact_row = static_cast<std::uint32_t>(at);
      }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
      return Missing(std::move(m), "uncovered_assignment");
    std::set<std::pair<std::uint32_t, std::uint32_t>> observed_tiles;
    std::uint64_t expected_tiles = 0;
    for (const auto count : counts)
      expected_tiles += (std::uint64_t{count} + r.tile_rows - 1) / r.tile_rows;
    for (const auto encoded : r.tiles) {
      const auto word = static_cast<std::uint32_t>(encoded);
      const auto e = word & 0xffffU, tile = word >> 16;
      if (e >= tensor.experts || counts[e] == 0 ||
          std::uint64_t{tile} * r.tile_rows >= counts[e] ||
          !observed_tiles.emplace(e, tile).second)
        return Missing(std::move(m), "invalid_or_duplicate_tile");
      std::vector<std::uint32_t> members;
      const auto begin = static_cast<std::uint64_t>(r.bounds[e]) +
                         std::uint64_t{tile} * r.tile_rows;
      const auto end = std::min<std::uint64_t>(
          begin + r.tile_rows,
          static_cast<std::uint64_t>(r.bounds[e]) + counts[e]);
      for (auto at = begin; at < end; ++at)
        members.push_back(static_cast<std::uint32_t>(r.slots[at]));
      for (std::uint64_t first = 0; first < tensor.rows;
           first += r.output_tile_rows)
        append(e, tile, r.tile_rows, static_cast<std::uint32_t>(first),
               static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   tensor.rows, first + r.output_tile_rows)),
               members);
    }
    if (observed_tiles.size() != expected_tiles)
      return Missing(std::move(m), "uncovered_tile");
    m.tiles_complete = true;
  }
  std::uint64_t unique_bytes = 0;
  for (const auto count : counts)
    if (count != 0)
      unique_bytes += expert_bytes;
  m.requested_unique_bytes = unique_bytes;
  m.requested_packed_blocks = unique_bytes / WeightBlockBytes(tensor.packing);
  m.mapped_assignments = m.expected_assignments;
  m.addresses_complete = true;
  m.missing = nullptr;
  return m;
}

// Canonical address intersection, NOT saved traffic or a removal certificate.
// Explicit same-plan/state association is mandatory, even for complete maps.
inline std::optional<std::uint64_t> RequestedOverlap(const Map& a,
                                                     const Map& b) {
  if (!a.addresses_complete || !b.addresses_complete ||
      !a.association.iteration || !a.association.cohort ||
      !a.association.plan || !a.association.route ||
      !a.association.starting_state || a.association != b.association ||
      a.role == b.role || a.identity.run != b.identity.run ||
      a.identity.pid != b.identity.pid || a.tensor.model != b.tensor.model)
    return std::nullopt;
  const auto& x = a.tensor;
  const auto& y = b.tensor;
  if (x.allocation != y.allocation) {
    if (x.allocation < y.allocation + b.payload_bytes &&
        y.allocation < x.allocation + a.payload_bytes)
      return std::nullopt;  // Conflicting ownership/alias descriptors.
    return 0;
  }
  if (x.layer != y.layer || x.tensor != y.tensor || x.packing != y.packing ||
      x.rows != y.rows || x.cols != y.cols || x.experts != y.experts)
    return std::nullopt;
  const auto ranges = [](const Map& m) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> raw, merged;
    for (const auto& tile : m.tiles)
      raw.emplace_back(tile.byte_begin, tile.byte_end);
    std::sort(raw.begin(), raw.end());
    for (const auto& range : raw) {
      if (!merged.empty() && range.first <= merged.back().second)
        merged.back().second = std::max(merged.back().second, range.second);
      else
        merged.push_back(range);
    }
    return merged;
  };
  const auto ar = ranges(a), br = ranges(b);
  std::uint64_t intersection = 0;
  for (std::size_t i = 0, j = 0; i < ar.size() && j < br.size();) {
    const auto begin = std::max(ar[i].first, br[j].first);
    const auto end = std::min(ar[i].second, br[j].second);
    if (begin < end)
      intersection += end - begin;
    if (ar[i].second < br[j].second)
      ++i;
    else
      ++j;
  }
  return intersection;
}

inline std::optional<std::uint64_t> RequestedBlockOverlap(const Map& a,
                                                          const Map& b) {
  const auto bytes = RequestedOverlap(a, b);
  const auto block = WeightBlockBytes(a.tensor.packing);
  return bytes && block != 0 ? std::optional{*bytes / block} : std::nullopt;
}

template<typename T>
inline void Optional(std::ostream& o, const std::optional<T>& value) {
  if (value)
    o << *value;
  else
    o << "null";
}
inline void SerializeIdentity(std::ostream& o, const Identity& i,
                              const Association& a) {
  o << "\"schema\":\"pd-profile-v1\",\"run_id\":" << i.run
    << ",\"pid\":" << i.pid << ",\"work_start_ns\":" << i.work_start
    << ",\"thread_id\":" << i.thread << ",\"forward\":" << i.forward
    << ",\"iteration\":";
  Optional(o, a.iteration);
  o << ",\"cohort\":";
  Optional(o, a.cohort);
  o << ",\"plan\":";
  Optional(o, a.plan);
  o << ",\"legacy_route\":";
  if (!a.route)
    o << "null";
  else
    o << '"'
      << (*a.route == LegacyRoute::kGeneric     ? "generic_direct_sampler"
          : *a.route == LegacyRoute::kScalarMtp ? "scalar_mtp"
                                                : "backend_mtp_batch")
      << '"';
  o << ",\"scheduler_frozen\":";
  Optional(o, a.scheduler_frozen);
  o << ",\"backend_filtered\":";
  Optional(o, a.backend_filtered);
  o << ",\"model_filtered\":";
  Optional(o, a.model_filtered);
  o << ",\"starting_state\":";
  Optional(o, a.starting_state);
  o << ",\"request_lineage\":null,\"state_proof\":null"
       ",\"diagnostic\":true,\"performance_repeat\":false";
}
inline std::string Serialize(const Map& m) {
  std::ostringstream o;
  o.imbue(std::locale::classic());
  o << '{';
  SerializeIdentity(o, m.identity, m.association);
  o << ",\"kind\":\"packed_requests\",\"role\":\""
    << (m.role == ForwardRole::kPrompt ? "prompt" : "decoder")
    << "\",\"layer\":" << m.tensor.layer << ",\"tensor\":\""
    << (m.tensor.tensor == TensorRole::kGate ? "gate"
        : m.tensor.tensor == TensorRole::kUp ? "up"
                                             : "down")
    << "\",\"packing\":\"" << core::ToString(m.tensor.packing)
    << "\",\"model_instance\":" << m.tensor.model
    << ",\"allocation\":" << m.tensor.allocation
    << ",\"weight_rows\":" << m.tensor.rows
    << ",\"weight_cols\":" << m.tensor.cols
    << ",\"experts\":" << m.tensor.experts
    << ",\"payload_bytes\":" << m.payload_bytes << ",\"execution\":\""
    << (m.execution == Execution::kRoutedF16 ? "routed_f16"
        : m.execution == Execution::kVector  ? "q8_vector_logical_addresses"
                                             : "unobserved")
    << "\",\"coverage\":{\"expected_assignments\":" << m.expected_assignments
    << ",\"mapped_assignments\":" << m.mapped_assignments
    << ",\"addresses_complete\":" << (m.addresses_complete ? "true" : "false")
    << ",\"physical_tiles_complete\":" << (m.tiles_complete ? "true" : "false")
    << ",\"missing\":";
  if (m.missing)
    o << '"' << m.missing << '"';
  else
    o << "null";
  o << "},\"requested_unique_bytes\":";
  Optional(o, m.requested_unique_bytes);
  o << ",\"requested_packed_blocks\":";
  Optional(o, m.requested_packed_blocks);
  o << ",\"physical_tile_request_count\":";
  if (m.tiles_complete)
    o << m.tiles.size();
  else
    o << "null";
  o << ",\"row_semantics\":\"shared_projection_rows_not_accepted_outputs\""
       ",\"rows\":[";
  const char* sep = "";
  for (const auto& row : m.rows) {
    o << sep << "{\"forward_row\":" << row.forward_row
      << ",\"session_local\":" << row.session
      << ",\"context_position\":" << row.context << '}';
    sep = ",";
  }
  o << "],\"assignments\":[";
  sep = "";
  for (const auto& a : m.assignments) {
    o << sep << "{\"row\":" << a.row << ",\"slot\":" << a.slot
      << ",\"expert\":" << a.expert << ",\"compact_row\":";
    Optional(o, a.compact_row);
    o << '}';
    sep = ",";
  }
  o << "],\"tiles\":[";
  sep = "";
  for (const auto& t : m.tiles) {
    o << sep << "{\"expert\":" << t.expert << ",\"input_tile\":" << t.input_tile
      << ",\"input_tile_rows\":" << t.input_tile_rows
      << ",\"weight_row_begin\":" << t.weight_row_begin
      << ",\"weight_row_end\":" << t.weight_row_end
      << ",\"byte_begin\":" << t.byte_begin << ",\"byte_end\":" << t.byte_end
      << ",\"logical_assignment_uses\":" << t.assignments.size()
      << ",\"dispatch_request_multiplicity\":";
    if (m.tiles_complete)
      o << 1;  // One packed matrix tile request per recorded F16 dispatch tile.
    else
      o << "null";
    o << ",\"load_multiplicity\":null,\"assignments\":[";
    const char* inner = "";
    for (const auto member : t.assignments) {
      o << inner << member;
      inner = ",";
    }
    o << "]}";
    sep = ",";
  }
  o << "],\"address_semantics\":\"requested_packed_payload_union_not_dram\""
       ",\"critical_path\":null,\"gpu_busy_ns\":null,\"U_ns\":null,"
       "\"L_ns\":null,\"H_upper_ns\":null,\"saved_dram_bytes\":null,"
       "\"feasibility\":\"inconclusive\"}";
  return o.str();
}

inline Identity CurrentIdentity() {
  return {perf::Sink().run_id, perf::Sink().process_id,
          perf::current ? perf::current->start_ns : 0,
          perf::current ? perf::current->thread_id : 0, perf::forward_id};
}
inline void Publish(Map map) noexcept {
  try {
    map.identity = CurrentIdentity();
    perf::Sink().Append(Serialize(map));
  } catch (...) {
    // Lost diagnostics are missing evidence, never a serving failure or GO.
  }
}

// Host clock spans are overlapping timeline evidence, NOT disjoint costs.
// Enqueue time does not bound transfer completion or GPU work; event waits can
// overlap other kernels. A named-mechanism critical-path/removal certificate
// cannot be manufactured by adding these spans or subtracting route waits.
class HostSpan {
public:
  HostSpan(const char* name, const char* placement, bool observe = true,
           bool per_layer = false)
      : name_(name), placement_(placement) {
    if (observe && perf::current != nullptr && perf::forward_id != 0) {
      identity_ = CurrentIdentity();
      association_ = current_association;
      if (per_layer)
        layer_ = perf::layer_id;
      prompt_ = perf::prompt_rows;
      start_ = perf::Now();
    }
  }
  ~HostSpan() { Finish(); }
  void Finish() noexcept {
    if (start_ == 0)
      return;
    const auto end = perf::Now();
    const auto start = std::exchange(start_, 0);
    try {
      std::ostringstream o;
      o.imbue(std::locale::classic());
      o << '{';
      SerializeIdentity(o, identity_, association_);
      o << ",\"kind\":\"host_span\",\"name\":\"" << name_
        << "\",\"placement\":\"" << placement_ << "\",\"layer\":";
      Optional(o, layer_);
      o << ",\"role\":\"" << (prompt_ ? "prompt" : "decoder")
        << "\",\"start_ns\":" << start << ",\"end_ns\":" << end
        << ",\"clock\":\"host_steady_ns\",\"critical_path\":null,"
           "\"gpu_busy_ns\":null,\"L_ns\":null,\"H_upper_ns\":null}";
      perf::Sink().Append(o.str());
    } catch (...) {
    }
  }
  HostSpan(const HostSpan&) = delete;
  HostSpan& operator=(const HostSpan&) = delete;

private:
  const char* name_;
  const char* placement_;
  Identity identity_{};
  Association association_{};
  std::optional<std::uint64_t> layer_{};
  std::uint64_t start_{0};
  bool prompt_{false};
};

template<typename DeviceTensor>
inline Tensor Describe(const DeviceTensor& t, std::uintptr_t model,
                       std::uint32_t layer, TensorRole role) {
  return {model,  reinterpret_cast<std::uintptr_t>(t.data),
          layer,  role,
          t.type, t.rows,
          t.cols, t.experts};
}
struct DeviceRouting {
  Execution execution{Execution::kUnsupported};
  std::uint32_t used{0};
  const std::int32_t *ids{nullptr}, *bounds{nullptr}, *tokens{nullptr},
      *slots{nullptr};
  std::size_t compact_capacity{0};
  std::span<const std::int32_t> gate_tiles{}, down_tiles{};
  std::uint32_t gate_tile_rows{0}, down_tile_rows{0}, gate_output_rows{128};
};

// download MUST join its stream even on failure, before these host vectors die.
// Kept independent of HIP and the trace sink for CPU failure/coverage fixtures.
template<typename Download>
inline std::array<Map, 3> CollectLayer(const std::array<Tensor, 3>& tensors,
                                       ForwardRole role,
                                       std::span<const Row> rows,
                                       const DeviceRouting& d,
                                       Download&& download) {
  std::vector<std::int32_t> ids(rows.size() * d.used), bounds, tokens, slots;
  const char* missing = nullptr;
  if (d.execution == Execution::kUnsupported)
    missing = "unobserved_dispatch_layout";
  else if (!d.ids || !download(d.ids, std::span(ids)))
    missing = "ids_download_failed";
  if (!missing && d.execution == Execution::kRoutedF16) {
    bounds.resize(std::size_t{tensors[0].experts} + 1);
    if (!d.bounds || !d.tokens || !d.slots ||
        !download(d.bounds, std::span(bounds)) || bounds.back() < 0 ||
        static_cast<std::size_t>(bounds.back()) > d.compact_capacity) {
      missing = "bounds_download_failed_or_invalid";
    } else {
      tokens.resize(bounds.back());
      slots.resize(bounds.back());
      if (!download(d.tokens, std::span(tokens)) ||
          !download(d.slots, std::span(slots)))
        missing = "compaction_download_failed";
    }
  }
  std::array<Map, 3> maps;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    Routing r{d.execution,
              d.used,
              ids,
              bounds,
              tokens,
              slots,
              i == 2 ? d.down_tiles : d.gate_tiles,
              i == 2 ? d.down_tile_rows : d.gate_tile_rows,
              i == 2 ? 128 : d.gate_output_rows};
    auto map = BuildMap(tensors[i], role, rows, r);
    maps[i] = missing ? Missing(std::move(map), missing) : std::move(map);
  }
  return maps;
}

// Caller guarantees eager target execution: same existing sample_moe guard that
// disables target graph capture, never called for predictor/capture replays.
// No device allocation, kernel, arithmetic, scratch mutation, or route changes.
template<typename Download>
inline void RecordLayer(const std::array<Tensor, 3>& tensors, ForwardRole role,
                        std::span<const Row> rows, const DeviceRouting& d,
                        Download&& download) noexcept {
  if (!perf::sample_moe || perf::current == nullptr || perf::forward_id == 0)
    return;
  try {
    HostSpan observation("packed_map_observation", "perturbing_d2h_and_host",
                         true, true);
    for (auto& map :
         CollectLayer(tensors, role, rows, d, std::forward<Download>(download)))
      Publish(std::move(map));
  } catch (...) {
    // No fabricated empty-success map if host memory/serialization fails.
  }
}

}  // namespace gufo::models::qwen38_flash_next::pd
#endif  // GUFO_ENABLE_PD_PROFILING
#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_PD_PROFILE_HPP_
