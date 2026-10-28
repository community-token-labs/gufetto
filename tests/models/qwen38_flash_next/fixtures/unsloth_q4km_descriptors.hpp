#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_UNSLOTH_Q4KM_DESCRIPTORS_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_UNSLOTH_Q4KM_DESCRIPTORS_HPP_

#include <array>
#include <cstdint>
#include <string_view>

// Independent captured header, not generated from Gufo's binder.
// unsloth/Qwen3.8-Flash-Next-GGUF@766911a6b7369840a91dbcd95f9f997acaab6cd6
// MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf; 2026-10-26 capture.
// Header SHA-256:
// 7480b1456b08c060b1eff53ebf8eb5c7ac575404246eda1c05304c6777a91123 Source:
// strix-halo-knowledge/provenance/research/
// mtp-unsloth-family-20261026-sources/unsloth-q4-header.json.
// Only descriptors retained: no tokenizer or weight payload.
namespace unsloth_q4km_fixture {
struct Descriptor {
  std::string_view name;
  std::array<std::uint64_t, 3> ne;
  unsigned rank;
  unsigned type;
  std::uint64_t offset;
};
inline constexpr std::array<Descriptor, 32> kDescriptors{{
    {"blk.48.attn_k.weight", {2560, 512, 1}, 2, 12, 0},
    {"blk.48.attn_k_norm.weight", {256, 1, 1}, 1, 0, 737280},
    {"blk.48.attn_output.weight", {6144, 2560, 1}, 2, 12, 738304},
    {"blk.48.attn_q.weight", {2560, 12288, 1}, 2, 12, 9585664},
    {"blk.48.attn_q_norm.weight", {256, 1, 1}, 1, 0, 27280384},
    {"blk.48.attn_v.weight", {2560, 512, 1}, 2, 14, 27281408},
    {"blk.48.ffn_down_exps.weight", {640, 2560, 512}, 3, 8, 28356608},
    {"blk.48.ffn_down_shexp.weight", {640, 2560, 1}, 2, 8, 919646208},
    {"blk.48.ffn_gate_exps.weight", {2560, 640, 512}, 3, 12, 921387008},
    {"blk.48.ffn_gate_inp.weight", {2560, 512, 1}, 2, 0, 1393246208},
    {"blk.48.ffn_gate_inp_shexp.weight", {2560, 1, 1}, 1, 0, 1398489088},
    {"blk.48.ffn_gate_shexp.weight", {2560, 640, 1}, 2, 12, 1398499328},
    {"blk.48.ffn_up_exps.weight", {2560, 640, 512}, 3, 12, 1399420928},
    {"blk.48.ffn_up_shexp.weight", {2560, 640, 1}, 2, 12, 1871280128},
    {"blk.48.hc_attn_down.weight", {10240, 320, 1}, 2, 12, 1872201728},
    {"blk.48.hc_attn_inject.weight", {10240, 4, 1}, 2, 12, 1874044928},
    {"blk.48.hc_attn_norm.weight", {10240, 1, 1}, 1, 0, 1874067968},
    {"blk.48.hc_attn_up.weight", {320, 10240, 1}, 2, 6, 1874108928},
    {"blk.48.hc_ffn_down.weight", {10240, 320, 1}, 2, 14, 1876361728},
    {"blk.48.hc_ffn_inject.weight", {10240, 4, 1}, 2, 12, 1879049728},
    {"blk.48.hc_ffn_norm.weight", {10240, 1, 1}, 1, 0, 1879072768},
    {"blk.48.hc_ffn_up.weight", {320, 10240, 1}, 2, 6, 1879113728},
    {"blk.48.indexer.k_norm.weight", {128, 1, 1}, 1, 0, 1881366528},
    {"blk.48.indexer.k_proj.weight", {2560, 128, 1}, 2, 30, 1881367040},
    {"blk.48.indexer.q_norm.weight", {128, 1, 1}, 1, 0, 1882022400},
    {"blk.48.indexer.q_proj.weight", {2560, 512, 1}, 2, 30, 1882022912},
    {"blk.48.nextn.eh_proj.weight", {5120, 2560, 1}, 2, 12, 1884644352},
    {"blk.48.nextn.enorm.weight", {2560, 1, 1}, 1, 0, 1892017152},
    {"blk.48.nextn.hc_head_down.weight", {10240, 320, 1}, 2, 12, 1892027392},
    {"blk.48.nextn.hc_head_norm.weight", {10240, 1, 1}, 1, 0, 1893870592},
    {"blk.48.nextn.hc_head_up.weight", {320, 10240, 1}, 2, 6, 1893911552},
    {"blk.48.nextn.hnorm.weight", {10240, 1, 1}, 1, 0, 1896164352},
}};
inline constexpr std::uint64_t kHeaderBytes = 10946608;
inline constexpr std::uint64_t kFileBytes = 1907151936;
}  // namespace unsloth_q4km_fixture
#endif
