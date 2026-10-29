# Independent CPU vision encoder reference

`vision_encoder_reference` is an offline teacher, not a production backend or a
kernel qualification. It imports no Gufo encoder, GPU code or candidate hidden
states. `vision_encoder_capture.py` independently decodes frozen pixels, pins
inputs and publishes selected intermediate rows plus merger embeddings privately.
It does not generate teacher-token histories or downstream language-model logits.

## Model and numerical contract

The fixed architecture is the Qwen4-Exp/Qwen3-VL vision specification at Transformers
`50e35cc0ca18ec19b80653690de553e6decf74f5`, model snapshot
`de4b8e4d43b917e7706784d8bb445c9af86a3540`: 27 blocks, hidden1152, 16 heads of72,
MLP4304, duplicated temporal patches of2×16×16, 48×48 learned positions,
2×2 spatial merger and embedding width2560. Model/source hashes are provenance,
not a claim that this is the upstream executable.

The reader requires the exact334-tensor GGUFv3 inventory, shapes and F32/BF16
types. Weights load on demand; BF16 values convert exactly to F32. Patch temporal
slices reconstruct C,T,Y,X columns without pre-summing their weights. Preprocessing
accepts only RGB8, noninterlaced, filter-0 PNGs with dimensions divisible by32,
area≤16,777,216 and no resizing. The pinned Torchvision backend fuses normalization:
`(F32(pixel)-127.5f)/127.5f`. Other PNG formats are refused, not approximated.
Position interpolation is bilinear, align-corners; axial split-half RoPE uses
theta10000. Blocks use pre-LayerNorm, epsilon1e-6 and tanh-GELU; merger normalization
precedes four-row packing and erf-GELU.

The approved accumulator declaration is `f32_qk_f64_softmax_pv`, with this exact
additional `arithmetic_plan`:

```json
{"projections":"f32","qk":"f32","softmax":"f64","pv":"f64","output":"f32","query_block":128,"full_keys":true}
```

Projection/QK reductions are ordered F32 without FMA or reassociation. Scaled QK
is F32; stable softmax and PV use F64, then store F32. One head at a time and at
most128 query rows see **every key**; keys are not sampled or separately normalized.
There is no BF16 activation rounding. OpenMP uses at most four threads. Without
OpenMP the same loops execute serially. Native x86 builds use the local ISA;
record actual compiler/options and binary hash rather than claiming portability
of executable bytes. This policy is deliberately not bit-identical to the
all-F32 scalar operator oracle or production BF16 boundaries.

## Build, fixture proof and capture

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --target vision_encoder_reference vision_ref_math_test --parallel 4
ctest --test-dir build/cpu-test -R '^vision_(ref_math|encoder_reference|encoder_capture|compare)_test$' --output-on-failure
```

Math fixtures use independent analytical/long-double expectations. The transport
suite compares small attention cases to the independent Python scalar oracle and
runs all27 blocks plus merger on a sparse synthetic GGUF with analytically known
outputs. The sparse full-tower fixture has zero transformer updates and a
bias-only merger; it tests transport/composition but not nontrivial trained
end-to-end numerics. These tests do not use real models or qualify a real encoder.

The separately invoked `nice -n 19 python3 -B tests/functional/vision_operator_4k.py`
checks the scalar oracle with one selected query, one head of72 dimensions, and
all65536 keys against an analytical uniform-attention result with a nonzero final
key contribution. It does not exercise the native full encoder or downstream
logits, and remains outside hosted CI.

Use the immutable production projector,907542944 bytes, SHA256
`2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355`.
`--development` accepts only the separately pinned local HF export and marks all
outputs `NON_QUALIFYING`; it cannot stand in for the matched production artifact.
The approved active full-reference order is `code-dense-1920`, `shapes-1024`,
`skyline-2048`. The `board-4096` full run is parked by owner workload priority,
not infeasibility; operator coverage does not qualify4K pixels-to-merger or logits.
The implementation's area bound is not authorization to run other cases.

Freeze the source revision and build before preparing packets. For each case:

```sh
python3 -B tools/quality/vision_encoder_capture.py prepare \
  --binary build/cpu-test/vision_encoder_reference --model /private/mmproj-BF16.gguf \
  --images /registration/images.json --plan /registration/tensors.json \
  --case code-dense-1920 --image /private/code-dense-1920.png \
  --revision FULL_COMMITTED_40_HEX_REVISION --toolchain 'actual compiler and flags' \
  --out-dir /private/new-case-packet
# Record the printed packet SHA, header and derived case-plan hashes externally
# BEFORE executing. Do not change source, build or inputs between prepare/run.
python3 -B tools/quality/vision_encoder_capture.py run \
  --packet /private/new-case-packet --packet-sha256 PRINTED_SHA256
```

`prepare` verifies PNG/RGB identities against the frozen manifest, preserves the
parent plan's row indices/descriptors, writes a derived one-case plan, and pins
all input/source/binary hashes. The source digest is canonical sorted JSON of
source basenames to SHA256, compact separators plus trailing newline. `run` checks
the externally frozen packet hash and identities before/after execution. Files
are private; each packet permits one attempt only, including failed attempts.
The native writer requires filesystem hard-link support for no-overwrite
publication. No reference output is reused silently.

Run one case at a time. The driver launches nice19/four threads and samples every
0.5s, stopping if MemAvailable<6GiB, disk free<80GB(decimal), or combined parent+
child RSS>12GiB. Peak RSS is **sampled**, not a continuous high-water guarantee.
Resource checks use Linux `/proc`; the C++ tool itself is OS-neutral. No memory
reservation is made against other users. On interruption/error, child termination
escalates to kill after5s; partial outputs/logs and failure metadata remain. Only a
complete finite-row/size/hash validation publishes `completion.json`.

Retain `raw/*.f32le`, `capture.jsonl`, logs and packet privately once captured;
commit hashes/provenance/resource receipts, not tensor binaries. A completion says
`referencecaptured`, `kernelqualified:false`—not that a candidate passed any gate.
Use the exact derived case-plan and header when freezing the three-arm comparator
contract, preserving all previously approved numerical thresholds. Intermediate
row sampling cannot assert full-tensor maxima.

## Deliberate remaining limits

No current/candidate GPU captures, task/speed qualification, resizing, video,
other architectures or arbitrary model packages. Downstream teacher histories
and matched decoder capture belong to the separately allocated native phase.
The downstream wrapper currently accepts the original `vision_accumulator_dtype:
f32` declaration; registering this mixed-policy reference there requires an
explicit metadata amendment, not relabeling it as all-F32. A real4K image workload
is the trigger to revisit the parked full-encoder run with fresh authority.
