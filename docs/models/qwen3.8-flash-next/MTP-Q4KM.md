# Experimental shared Q4_K_M MTP control

**EXPERIMENTAL landing candidate; defaults unchanged.** Coordinator V5 on
`fe100d11` passed native build/synthetic operators and all four Q8 model gates
(audit, cache, sampling, batch). Q4 failed recursive carry; the route-specific
CPU-oracle defect described below is fixed in source, but its corrected model
result and M0 useful throughput remain pending. This is not release qualification
or production-container ELF equivalence. The owner controls landing on main.
No production default, target weight, shared vocabulary head, sampler or draft
length policy changes; existing numerical and final-AR gates remain unchanged.

## First control identity

Owner-supplied artifact identity (not a local payload verification):

- Repository: `unsloth/Qwen3.8-Flash-Next-GGUF`
- Revision: `766911a6b7369840a91dbcd95f9f997acaab6cd6`
- File: `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`
- Size: **1,907,151,936 bytes**
- SHA-256: `f521868a9e143718bef513772f6e04d9642551e362cf2439636d2abdbd149dfc`
- Modern **32-tensor `blk.48`** schema, including block-local
  `nextn.hc_head_*`; no private embedding or vocabulary head.
- The existing target and its shared **Q8_0 vocabulary head remain unchanged**.

The coordinator reports completed SHA-256 verification on Nimo-1 at
`/home/connorb/gac/mtp-unsloth-766911a6/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`
(mode `0444`). This is a coordinator receipt, not local payload verification.
The pinned URL is
`https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`. The editing agent downloaded/read no model bytes and
performed no host, SSH, lease, HIP build or GPU/model execution; coordinator
qualification is distinct and summarized above. Do not substitute
an older top-level `output_hc_*` sidecar, a self-contained model, or an EXL3 head.
The artifact identity is a qualification pin, not a loader hash allowlist.

## Loader and execution contract

Only these MTP roles gain formats; their previous accepted formats remain:

| Role in `blk.48` | Added format |
| --- | --- |
| `attn_q`, `attn_k`, `attn_output`, `nextn.eh_proj` | Q4_K |
| `ffn_gate_shexp`, `ffn_up_shexp` | Q4_K |
| `attn_v` | Q6_K |
| `hc_attn_down`, `nextn.hc_head_down` | Q4_K |
| `hc_ffn_down` | Q6_K |
| All three HC `*_up` matrices | Q5_0 |
| `hc_attn_inject`, `hc_ffn_inject` | Q4_K |

Q5_0 is 32 values / 22 bytes, `d * (five_bit_code - 16)`. HC up has
320 columns, so Q4_K's 256-value blocks are **not** admissible there.
Q4_K/Q6_K use their existing GGML numerical definitions and 144/210-byte
256-value geometry. Routed gate/up Q4_K and down Q8_0 already work; no expert
kernel is changed. Indexer weights, trunk dense/HC roles, embeddings and output
allowlists are unchanged. The production rejection of unsupported Q6_K
embedding/head/routed-expert roles remains.

Weights retain raw GGUF storage. EH's 5120-column row splits into two
2560-column matrices on Q4_K block boundaries. Mixed Q4_K/Q6_K Q/K/V stay
separate; the old Q8 stack/slice path is unchanged.

Packed dense rows dispatch before unquantized hipBLAS, at every batch width.
Only F32/F16/BF16 may use unchecked whole-vector loads. Packed projections
materialize decoded values, perform four ordered FMA products from zero per
lane step, add that step to the lane accumulator, then use the existing
descending XOR wave32 reduction. Scalar, eight-row groups and remainder rows
must retain this arithmetic. **Source spelling is not GPU qualification**:
fast-math, group/remainder and process-reload invariance remain device gates.

Packed predictor forwards scope HC mix/combine to the existing F32 path with
per-row combine arithmetic; an RAII guard restores executor policy on every
exit. This override activates only when predictor HC actually contains a new
packed format in any attention/FFN/readout down, up or inject projection, including
independently mixed roles. The exact Q4 control has Q4_K attention down, Q6_K FFN
down and Q5_0 up. All-Q8 MTP leaves the policy false and retains its pre-existing wide
F16/Q8 route, arithmetic, packed QKV slicing and fast paths; no unavoidable Q8
behavior change is intended. The trunk's wide route is not globally disabled.
Batched MTP already uses F32 normalization and separate quantized inject
projections. Same-binary A/B must verify the unchanged Q8 control, not assume it.

Cache-only forwards with separate K/V project only K and V, then retain the
existing K normalization/RoPE, KV storage, indexer-key storage and pooling.
They return before query, attention, output and FFN. Uniform cache-only cohorts
use each session's own row offset, position and cancellation check. Mixed
cache-only/proposal cohorts retain the full-body fallback. No attention kernel
is added. Static packed K+V traffic is 1,812,480 bytes versus 19,507,200 bytes
for Q+K+V (90.7% avoided QKV weight bytes), **not a latency measurement**.

## CPU checks

CPU-only builds explicitly use `ENGINE_ENABLE_HIP=OFF`:

- `qwen38_flash_next_mtp_q4km_test` / CTest `qwen38_flash_next.mtp_q4km`:
  sparse synthetic 32-tensor GGUF admission, independently captured exact
  descriptor/rank/type/extent cross-check, shared-Q8 regression, protected
  trunk-role and unsupported-MTP-role rejection, truncation, bad column
  alignment, signed Q5_0/high-bit/endpoint goldens, independent Q4_K/Q6_K
  goldens and stacked expert-row offsets.
- `qwen38_flash_next_config_test` / CTest `qwen38_flash_next.config`.
- `ggml_dequant_test` / CTest `ggml_dequant_test`.

All three passed in CPU RelWithDebInfo and Debug ASan+UBSan builds.
Synthetic GGUF payloads alias a sparse zero-filled range; tests never use model
weights. These checks do not compile the modified HIP dispatch or executor.
Changed C++ files pass clang-format 19.1.7; the repository-wide formatter run
with that available version fails on existing unrelated files. The pinned CI
formatter still needs the coordinator's normal toolchain check.

## V6 offline numerical diagnosis (candidate, not model-qualified)

Base: `feature/strix-mtp-q4km` at
`fe100d11bcafdc49929ca2bf9b027c3ea7813489`. The V6 edits remain uncommitted
for parent independent review. No HIP compilation, device/model execution,
model-byte access, downloads, host changes, lease, sealing or publication was
performed by this implementation task.

**Historical component evidence, not a V6 pass:** local harvest
`~/.cache/gufo-mtp-m0/lease-3e7c3ff48168`, `MANIFEST.txt` SHA-256
`8fc4ffa76a3295b37229998c4b223955473af6e17b1139c4f36f61b20334b1b6`.
All 60 listed payload hashes and sizes were checked locally. Its
`work/worker-result.json` records Q8 audit/cache-only/sampling-only/batch-only
return codes 0, but Q4 audit return code 1 and outer status `INCOMPLETE`.
`work/Q4-audit.stdout` records first-step relative RMS **0.0070986**, max abs
**0.0227612** at recursive carry, with preceding attention **0.0000019**.
The original gate verdict remains FAIL. Q8's four passes are an internal win
on this fork's native ROCr build, not production-ELF equivalence.

**Confirmed source defect:** `ReferenceModel::MatVec` previously reconstructed
Q8_1 activations only for Q8_0 weights. `Moe` passed Q4_K routed gate/up
through the same function, incorrectly retaining F32 activations. The fix
marks only those two routed projections `reference::Projection::kRoutedVector`;
`Storage::kDecode` now reconstructs Q8_1 for routed Q4_K too. `kFloat32`,
Q8 arithmetic, shared-expert/dense Q4, EH and HC projections stay unchanged.
Both CPU and device use the **same sidecar weights**, not a Q8 oracle baseline;
CPU decoding still reads original GGUF rows through `cpu::DequantizeRow`, not
GPU packed fragments.

Relevant source, under `src/models/qwen38_flash_next/kernels/rocm/`:

- `executor.cpp::GatedExperts` selects the gated vector route for small rows,
  then paired vector or MMQ fallbacks; `MoeExperts/WmmaExperts` can instead
  select F16 WMMA for wide/prefill rows. `batch.cpp::MoeBatch` explicitly keeps
  Q4 gate/up and Q8 down on vector arithmetic across request boundaries.
- `mmq/qfn_mmq.hip.cpp::moe_vector_projection` quantizes F32 inputs with
  `quantize_row_q8_1_hip`, or consumes the equivalent producer's Q8_1 rows.
  Per 32 values: `d=max(abs(x))/127`, `q=round(x/d)`, stored scale `D=fp16(d)`.
- `mmq/vecdotq.hpp::vec_dot_q4_K_q8_1_impl_vmmq` uses `D*sum(q)` for the
  min correction, **not** the separately stored `fp16(d*sum(q))`. Thus original
  Q4 dequantization dotted with reconstructed `D*q` has the vector formula,
  apart from floating reduction/weight-materialization rounding. Grouped and
  wave64 variants retain this formula (`mmvq.hip.cpp`, `moe_wave64.hip.cpp`).
  Gate and up reductions are separate; SwiGLU precedes Q8 down, and router
  fractions multiply down outputs in the MoE epilogue, not the gate/up inputs.
- The distinct MMQ dot uses the separately rounded stored sum; F16 WMMA uses
  half activations. This **single-token/vector** CPU reference does not claim
  either wide-route protocol. Adding a stored-sum correction to the scalar
  audit would be wrong. No GPU route was changed.
- `executor.cpp::Dense` sends packed Q4/Q5/Q6 to `SmallGemm` with F32 inputs.
  `MtpForward` scopes the packed-HC F32 mixer/combine override with RAII,
  including Q5_0 up. Neither is a reason to quantize all Q4 projections.

**Measured synthetic regression:** existing `qwen38_flash_next.mtp_q4km`
now compares two experts, two rows, two Q4 blocks and four activation patterns
against independent integer-dot formulas, including high scale/min bits.
The old-protocol negative control (same fixture, Q8-only activation predicate)
fails; restored candidate passes. Aggregate relative RMS: old **0.000313117**,
corrected **0**, incorrectly substituting the MMQ stored sum **7.68826e-06**.
Zero here reflects this exact binary-fraction fixture, not bit-exact GPU/model
parity. Dense Q4/Q5/Q6 retain F32 results; Q8 dense/routed results are
bit-identical to the legacy CPU activation protocol; zero Q4 input is covered.
The MTP Q4KM, config and GGML dequant tests all pass in existing CPU-only
RelWithDebInfo and Debug ASan+UBSan builds (GCC 14.2.0, CMake 4.4.4).
Local transcripts: `/tmp/gufo-v6-old-protocol-negative.txt`,
`/tmp/gufo-v6-cpu-tests.txt`, `/tmp/gufo-v6-sanitizer-tests.txt`.

`tests/models/qwen38_flash_next/mtp_audit.cpp` now prints all seven stage
statistics, including FFN HC-mix input and FFN output, before checking the
original five gates. FFN metrics are diagnostic only. The printed protocol
labels identify same-sidecar, same-boundary scope: GPU fusion, attention and
hidden feed subsequent CPU stages; FFN output includes the CPU HC mix from
GPU attention. This is **not** an end-to-end CPU rollout. Nonfinite values
still fail the existing gated comparisons. The audit instrumentation has not
been HIP-compiled or device-tested here.

**Pending:** this establishes a real oracle activation-protocol bug, not that
it accounts for the entire observed 0.7% model carry error. After independent
review and parent source freeze/reseal, V6 must rerun the same Q4 audit with
unchanged **5e-5** carry/head limits, inspect the new FFN metrics, and repeat
Q8/Q4 cache, sampling and batch gates. If residual error remains, localize HC
inject/router/gate/up/down/scatter before proposing arithmetic changes. No
Q4-specific tolerance is proposed; AR identity and strict isolation/cache gates
are unchanged. The parent owns packet metadata and wiki evidence integration.

## Coordinator qualification packet requested

Use **one candidate binary/toolchain** with both the existing shared-Q8 sidecar
and the exact new shared-Q4 artifact; retain immutable binary/model/harness
hashes, build commands and all failed observations. Also retain an unchanged
base binary for affected timing/regression controls. No automatic deployment.

1. Build and run the added operator coverage on Nimo-1:
   - target `qwen38_flash_next_projection_ops_test`, CTest
     `qwen38_flash_next.projection_ops`;
   - target `qwen38_flash_next_hc_mix_ops_test`, CTest
     `qwen38_flash_next.hc_mix_ops`.
   Both contain independent CPU dequantization/arithmetic checks, finite
   unsaturated inputs and row widths 1–9,16,17,32,33. Shapes cover HC
   320/10240, 2560-column EH halves and awkward output tails. Grouped rows must
   match scalar rows exactly. HC uses Q4_K down, Q5_0 up and Q4_K inject;
   Q6_K HC down is also covered by the projection target. Neither target has
   been HIP-compiled or run here.
2. Build target `qwen38_flash_next_gpu_probe`; run `--model TARGET
   --mtp-model SIDECAR --mtp-audit --batch 256 --context 4096` on each
   sidecar using the same binary. The existing `mtp_audit.cpp` checks the
   independent CPU predictor stages, recursive carry, image positions,
   chunk widths 1/8/9/32/33 and full/cache-only candidate replay. Run it in
   fresh processes, not only cached sessions. Supply exact stdout/stderr,
   loaded tensor format receipts and earliest failing stage captures.
3. Extend/retain the persistent-state qualification at byte level: K/V,
   indexer ring and pooled keys after scalar, chunked and uniform batched
   cache-only work, ragged per-session positions, sparse threshold crossing,
   mixed-cohort fallback, cancellation and snapshot restore. Existing
   `qwen38_flash_next_session_test --cache-only` and the `mtp_replay.hpp`
   sampling harness are follow-up gates, not CPU passes. Full-vs-cache-only
   candidate equality alone does not prove every persistent byte matches.
4. Within each sidecar require exact fixed-history scalar/chunk/batch/cache
   replay and seeded p/q rejection/residual/RNG/stop/restore correctness.
   **Draft logits and proposed tokens may differ across Q4/Q8. Final greedy
   committed output must match unchanged-target AR under the declared arithmetic
   contract.** Sampled execution retains within-arm rejection/residual/RNG/cache/
   restore/isolation and distribution gates; cross-arm identical sampled RNG
   bytes are not assumed. Compare draft Q4 versus Q8 on matched histories using
   finite/KL/NLL/top-k metrics, acceptance, committed tokens and completed useful
   work. Preserve target-logit correctness and the unchanged shared Q8 head.
5. Only after correctness: matched C1/C4 pp2048/tg128 controls at d0 and d32K,
   ordinary and high-acceptance text, with per-request prefill/decode/catch-up
   latency, draft/accepted/committed counts, peak memory and complete useful
   turn latency. Existing `--cost-audit` is diagnostic; do not recalibrate or
   change runtime policy in this implementation. A lower byte count or higher
   acceptance is not throughput qualification.

Owner priority is **0: loader/control first, then 2: draft head, then 1:
custom mix**. Stage 4 is **dropped entirely**; no training or hardware-feasibility
work is in scope. Head/custom-quant changes and runtime-policy changes are not
implemented here and require separate approval and evidence. An optional
double-quant control is coordinator-only and must be nearly free; it is not a
replacement for this first artifact. See [quality methodology](QUALITY.md) and
[testing](../../TESTING.md).
