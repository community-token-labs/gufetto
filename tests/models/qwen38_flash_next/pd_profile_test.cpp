#include "src/models/qwen38_flash_next/pd_profile.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(GUFO_ENABLE_PD_PROFILING) && GUFO_ENABLE_PD_PROFILING
namespace pd = gufo::models::qwen38_flash_next::pd;
using gufo::core::GgmlType;
namespace {
const pd::Association paired{11, 22, 33, pd::LegacyRoute::kBackendMtpBatch,
                             4, 3, 2, 44};
const std::array<pd::Row, 3> rows{{{0, 91, 100}, {1, 91, 101}, {2, 92, 900}}};
const std::array<std::int32_t, 6> ids{1, 0, 1, 2, 0, 1};
// Independently specified atomic scatter order, NOT the order of the IDs.
const std::array<std::int32_t, 4> bounds{0, 16, 32, 48};
const std::array<std::int32_t, 3> tiles{0, 1, 2};
struct Fixture {
  pd::Tensor tensor{70, 0x100000, 7, pd::TensorRole::kGate,
                    GgmlType::kQ4_K, 130, 256, 3};
  std::vector<std::int32_t> tokens = std::vector<std::int32_t>(48, -1);
  std::vector<std::int32_t> slots = std::vector<std::int32_t>(48, -1);
  Fixture() {
    tokens[0] = 2;
    slots[0] = 4;
    tokens[1] = 0;
    slots[1] = 1;
    tokens[16] = 2;
    slots[16] = 5;
    tokens[17] = 0;
    slots[17] = 0;
    tokens[18] = 1;
    slots[18] = 2;
    tokens[32] = 1;
    slots[32] = 3;
  }
  pd::Routing routing() const {
    return {pd::Execution::kRoutedF16, 2, ids, bounds, tokens, slots, tiles,
            16, 128};
  }
  pd::Map map() const {
    return pd::BuildMap(tensor, pd::ForwardRole::kPrompt, rows, routing(),
                        paired);
  }
};

void CheckPackedMap() {
  Fixture f;
  const auto map = f.map();
  assert(map.addresses_complete && map.tiles_complete);
  assert(map.expected_assignments == 6 && map.mapped_assignments == 6);
  assert(map.tiles.size() == 6);
  assert(map.requested_unique_bytes == 56160);
  assert(map.requested_packed_blocks == 390);
  // Q4_K: 144 bytes per K256 row. Expert 1 begins at 130*144=18720.
  const auto& t = map.tiles[2];
  assert(t.expert == 1 && t.input_tile == 0);
  assert(t.weight_row_begin == 0 && t.weight_row_end == 128);
  assert(t.byte_begin == 18720 && t.byte_end == 37152);
  assert((t.assignments == std::vector<std::uint32_t>{5, 0, 2}));
  assert(map.tiles[3].byte_begin == 37152);
  assert(map.tiles[3].byte_end == 37440);  // Ragged last 2 weight rows.
  assert(map.assignments[0].compact_row == 17);
  assert(map.assignments[5].compact_row == 16);
  assert(map.rows[2].session == 92 && map.rows[2].context == 900);
  assert(map.rows[1].context == 101 && map.tensor.layer == 7);
  assert(!t.load_multiplicity);  // Not measured lane loads or DRAM traffic.
  const auto json = pd::Serialize(map);
  assert(json.find("\"packing\":\"Q4_K\"") != std::string::npos);
  assert(json.find("\"L_ns\":null,\"H_upper_ns\":null") != std::string::npos);
  assert(json.find("\"critical_path\":null") != std::string::npos);
  assert(json.find("\"model_filtered\":2") != std::string::npos);
}

void CheckVectorAndOverlap() {
  Fixture f;
  auto prompt = f.map();
  const std::array<pd::Row, 1> decoder{{{0, 99, 2000}}};
  const std::array<std::int32_t, 2> decoder_ids{1, 2};
  const pd::Routing vector{pd::Execution::kVector, 2, decoder_ids};
  auto decode = pd::BuildMap(f.tensor, pd::ForwardRole::kDecoder, decoder,
                              vector, paired);
  assert(decode.addresses_complete && !decode.tiles_complete);
  assert(decode.tiles.size() == 2 && decode.mapped_assignments == 2);
  assert(decode.tiles[0].byte_begin == 18720);
  assert(decode.tiles[0].byte_end == 37440);
  // Two experts, no summing repeated token uses or output/input tiles.
  assert(pd::RequestedOverlap(prompt, decode) == 37440);
  assert(pd::RequestedBlockOverlap(prompt, decode) == 260);
  decode.association.plan.reset();
  assert(!pd::RequestedOverlap(prompt, decode));
  decode.association = paired;
  decode.association.starting_state.reset();
  assert(!pd::RequestedOverlap(prompt, decode));
  decode.association = paired;
  decode.association.iteration = 12;
  assert(!pd::RequestedOverlap(prompt, decode));
  decode.association = paired;
  decode.tensor.tensor = pd::TensorRole::kUp;  // Same address, bad alias.
  assert(!pd::RequestedOverlap(prompt, decode));
  decode.tensor.tensor = pd::TensorRole::kGate;
  decode.tensor.layer = 8;
  assert(!pd::RequestedOverlap(prompt, decode));
  decode.tensor.layer = 7;
  decode.tensor.allocation = 0x200000;
  assert(pd::RequestedOverlap(prompt, decode) == 0);
  decode.tensor.allocation = 0x100001;  // Partially aliased allocation.
  assert(!pd::RequestedOverlap(prompt, decode));
  decode.tensor.allocation = 0x100000;
  decode.tensor.packing = GgmlType::kQ5_K;
  assert(!pd::RequestedOverlap(prompt, decode));
}

void CheckRejections() {
  Fixture f;
  f.slots[17] = 5;  // Duplicate alias in a compaction bucket.
  f.tokens[17] = 2;
  assert(!f.map().addresses_complete);
  f = Fixture{};
  f.tokens[17] = 1;  // Slot zero belongs to token zero, not token one.
  assert(!f.map().addresses_complete);
  f = Fixture{};
  f.tokens[3] = 0;  // Nonempty padding must not silently claim coverage.
  assert(!f.map().addresses_complete);
  f = Fixture{};
  auto route = f.routing();
  const std::array<std::int32_t, 3> duplicate_tiles{0, 1, 1};
  route.tiles = duplicate_tiles;
  assert(!pd::BuildMap(f.tensor, pd::ForwardRole::kPrompt, rows, route, paired)
              .addresses_complete);
  route = f.routing();
  route.tiles = std::span(tiles).first(2);
  assert(!pd::BuildMap(f.tensor, pd::ForwardRole::kPrompt, rows, route, paired)
              .addresses_complete);
  f.tensor.cols = 255;
  assert(!f.map().addresses_complete);
  f.tensor.cols = 256;
  f.tensor.allocation = std::numeric_limits<std::uintptr_t>::max() - 10;
  assert(!f.map().addresses_complete);
  f = Fixture{};
  f.tensor.packing = GgmlType::kQ6_K;  // Unsupported telemetry stays missing.
  assert(!f.map().addresses_complete);
  route = f.routing();
  route.ids = {};
  assert(!pd::BuildMap(f.tensor, pd::ForwardRole::kPrompt, rows, route, paired)
              .addresses_complete);
  pd::Map absent;
  assert(!pd::RequestedOverlap(absent, absent));
  assert(!absent.requested_unique_bytes && !absent.requested_packed_blocks);
}

void CheckTileHeightsAndFormats() {
  // 129 live assignments: both 64- and 128-token maps, padded bound 144.
  std::vector<pd::Row> many_rows;
  std::vector<std::int32_t> many_ids(129, 0), token(144, -1), slot(144, -1);
  for (int i = 0; i < 129; ++i) {
    many_rows.push_back({static_cast<std::uint32_t>(i), 9,
                         static_cast<std::uint64_t>(800 + i)});
    token[i] = slot[i] = 128 - i;
  }
  const std::array<std::int32_t, 2> b{0, 144};
  const std::array<std::int32_t, 3> t64{0, 1 << 16, 2 << 16};
  const std::array<std::int32_t, 2> t128{0, 1 << 16};
  pd::Tensor tensor{70, 0x100000, 1, pd::TensorRole::kDown,
                     GgmlType::kQ5_1, 3, 64, 1};
  pd::Routing r{pd::Execution::kRoutedF16, 1, many_ids, b, token, slot,
                 t64, 64, 128};
  auto m = pd::BuildMap(tensor, pd::ForwardRole::kPrompt, many_rows, r, paired);
  assert(m.addresses_complete && m.tiles.size() == 3);
  assert(m.tiles[0].byte_end == 144);  // 3 rows * 2 K32 blocks * 24B.
  assert(m.tiles[2].assignments == std::vector<std::uint32_t>{0});
  assert(m.assignments[128].compact_row == 0);
  tensor.tensor = pd::TensorRole::kGate;
  tensor.packing = GgmlType::kQ5_K;
  tensor.cols = 256;
  r.tiles = t128;
  r.tile_rows = 128;
  r.output_tile_rows = 64;
  m = pd::BuildMap(tensor, pd::ForwardRole::kPrompt, many_rows, r, paired);
  assert(m.tiles.size() == 2 && m.tiles[0].byte_end == 528);
  tensor.packing = GgmlType::kQ8_0;
  tensor.cols = 64;
  r.tiles = t64;
  r.tile_rows = 64;
  m = pd::BuildMap(tensor, pd::ForwardRole::kPrompt, many_rows, r, paired);
  assert(m.tiles[0].byte_end == 204);  // 3 * 2 * 34B.
  const std::array<std::int32_t, 3> t48{0, 1 << 16, 2 << 16};
  r.tiles = t48;
  r.tile_rows = 48;
  m = pd::BuildMap(tensor, pd::ForwardRole::kPrompt, many_rows, r, paired);
  assert(m.addresses_complete && m.tiles.size() == 3);
  assert(m.tiles[2].assignments.size() == 33);
  assert(m.tiles[2].assignments.front() == 32);
}

void CheckDownloadCoverage() {
  Fixture f;
  std::array tensors{f.tensor, f.tensor, f.tensor};
  tensors[1].tensor = pd::TensorRole::kUp;
  tensors[2].tensor = pd::TensorRole::kDown;
  pd::DeviceRouting d{pd::Execution::kRoutedF16, 2, ids.data(), bounds.data(),
                       f.tokens.data(), f.slots.data(), 48, tiles, tiles,
                       16, 16, 128};
  std::size_t copies = 0;
  const auto download = [&](const std::int32_t* source,
                             std::span<std::int32_t> destination) {
    ++copies;
    std::copy_n(source, destination.size(), destination.begin());
    return true;
  };
  auto maps = pd::CollectLayer(tensors, pd::ForwardRole::kPrompt, rows, d,
                                download);
  assert(copies == 4 && maps[0].addresses_complete);
  assert(maps[0].tiles[2].assignments ==
         (std::vector<std::uint32_t>{5, 0, 2}));
  assert(maps[1].tensor.tensor == pd::TensorRole::kUp);
  copies = 0;
  d.compact_capacity = 47;  // Downloaded bound must not exceed owned capacity.
  maps = pd::CollectLayer(tensors, pd::ForwardRole::kPrompt, rows, d,
                           download);
  assert(copies == 2 && !maps[0].addresses_complete);
  assert(std::string(maps[0].missing) == "bounds_download_failed_or_invalid");
  d.compact_capacity = 48;
  maps = pd::CollectLayer(tensors, pd::ForwardRole::kPrompt, rows, d,
                           [](const auto*, auto) { return false; });
  assert(!maps[0].addresses_complete && maps[0].tiles.empty());
  copies = 0;
  d.execution = pd::Execution::kUnsupported;
  maps = pd::CollectLayer(tensors, pd::ForwardRole::kPrompt, rows, d,
                           download);
  assert(copies == 0 && !maps[0].addresses_complete);
}

void CheckSamplingGuard() {
  Fixture f;
  const std::array tensors{f.tensor, f.tensor, f.tensor};
  const pd::DeviceRouting d{pd::Execution::kVector, 2, ids.data()};
  bool downloaded = false;
  const auto download = [&](const auto*, auto) {
    downloaded = true;
    return false;
  };
  gufo::perf::Record record;
  gufo::perf::current = &record;
  gufo::perf::forward_id = 10;
  gufo::perf::sample_moe = false;
  pd::RecordLayer(tensors, pd::ForwardRole::kDecoder, rows, d, download);
  assert(!downloaded);  // Light traces must not introduce diagnostic readback.
  gufo::perf::current = nullptr;
  gufo::perf::sample_moe = true;
  pd::RecordLayer(tensors, pd::ForwardRole::kDecoder, rows, d, download);
  assert(!downloaded);
  gufo::perf::forward_id = 0;
  gufo::perf::sample_moe = false;
}

void CheckAssociationScope() {
  assert(!pd::current_association.iteration);
  {
    pd::AssociationScope scope(paired);
    assert(pd::current_association == paired);
    {
      pd::AssociationScope nested({});
      assert(!pd::current_association.iteration);
    }
    assert(pd::current_association == paired);
  }
  assert(!pd::current_association.iteration);
  // E1 representation fixture: never reselect a generic route from model N=1.
  auto singleton = paired;
  singleton.model_filtered = 1;
  {
    pd::AssociationScope scope(singleton);
    Fixture f;
    auto map = pd::BuildMap(f.tensor, pd::ForwardRole::kDecoder, rows,
                             f.routing());
    assert(map.association.route == pd::LegacyRoute::kBackendMtpBatch);
    assert(pd::Serialize(map).find("\"model_filtered\":1") != std::string::npos);
    assert(pd::Serialize(map).find("\"legacy_route\":\"backend_mtp_batch\"") !=
           std::string::npos);
    map.association.route = pd::LegacyRoute::kGeneric;
    assert(pd::Serialize(map).find("generic_direct_sampler") != std::string::npos);
  }
}
}  // namespace
#endif

int main() {
#if defined(GUFO_ENABLE_PD_PROFILING) && GUFO_ENABLE_PD_PROFILING
  CheckPackedMap();
  CheckVectorAndOverlap();
  CheckRejections();
  CheckTileHeightsAndFormats();
  CheckAssociationScope();
  CheckDownloadCoverage();
  CheckSamplingGuard();
  std::cout << "pd_profile_test passed (CPU mappings only)\n";
#else
#ifdef GUFO_PD_PROFILE_AVAILABLE
#error "profiling API leaked into a compile-off translation unit"
#endif
  std::cout << "pd_profile_test passed (compile-off)\n";
#endif
  return 0;
}
