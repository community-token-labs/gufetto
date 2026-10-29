# Offline vision attention numerics harness

This harness evaluates **supplied data**, not GPU execution or reference honesty.
It imports no encoder/kernel implementation and does not change production paths.
All results remain `NOT_QUALIFIED`; an offline PASS cannot approve a kernel,
performance, task correctness, or the capture producer. Thresholds are explicit
contracts, with a visible `PROPOSED` or `APPROVED` registration status. No default
numerical thresholds are supplied by the comparators.

## Images and independent references

```sh
python3 tools/quality/vision_images.py --out-dir NEW_PRIVATE_DIRECTORY
python3 tools/quality/vision_reference.py operator-input.json --output NEW_REFERENCE.json
```

The generator produces three invented geometric images (1024², 2048², 4096²)
and four synthetic editor/terminal/dashboard screenshots (1280×768, 1920×1088,
1536×1024, and a dense 1920×1088 code editor). The dense editor has 40 pseudocode
lines and 35 test/output lines, 10×14-pixel glyphs, and a frozen code-reading
prompt. Its height is aligned to 32 pixels, eight pixels above 1080. Adding it
preserves the original six PNG/RGB identities and prompts; the generator source
hash and seven-case manifest hash change. Freeze a matching amended tensor plan
and per-case downstream stratum before captures; never pool it with the equally
sized terminal case. No desktop capture, personal content, external assets or fonts.
PNG and uncompressed RGB SHA-256, original generator-source provenance, exact
sizes and frozen prompts are in `manifest.json`. Directory0700/files0600; existing
paths are refused. Retain assets privately; commit the manifest, not image binaries.
PNG hashes depend on the compression toolchain, RGB hashes do not. Freeze generated
files and their manifest rather than silently regenerating with a different zlib.
This small synthetic screen is not representative coverage of natural photos,
real software screenshots, videos or OCR languages.

`vision_reference.attention_f32(q,k,v,scale=None,max_work=2000000)` is an independently
written, bounded CPU **operator-only** formula. Full noncausal attention, stable
softmax, explicitly rounded binary32 products and additions, no FMA, no BF16
score/probability casts. Inputs are matrices, so one head or selected query rows
may be evaluated against **all keys**, never a sampled/renormalized key subset.
Its CLI uses bounded JSON (8MiB); the module docstring defines each rounding point
and work-budget accounting. Analytic and Decimal tests do not use the production
path. It is a small-fixture oracle, not an efficient 65K-key/full-encoder runner.

For **encoder scope**, a mandatory independent implementation must run from the
same pixels and matched model artifacts through the entire tower and merger, with
F32 accumulators. It must not consume candidate/current hidden states. A pinned
upstream teacher such as the official Qwen vision implementation is an appropriate
source; record source/config/weight hashes, actual arithmetic/casts and any bounded
attention adaptation. The scalar operator reference is NOT a substitute. Capture
adapters and real-model reference execution are deliberately separate from this
CPU harness. Independence metadata is a declaration requiring source audit.

Current Gufo vision uses16×72 heads, global per-image noncausal attention and
27 layers. Q/K/V are BF16 after bias/RoPE. Scores accumulate F32, but softmax rounds
scores to BF16, scales by1/sqrt72 and rounds again, then stores probabilities BF16.
PV accumulates F32 and casts its output BF16 before projection. The F32 oracle
intentionally differs at these boundaries; bit-exact agreement is not assumed.

## Three-arm tensor stream

```sh
python3 tools/quality/vision_compare.py --plan tensors.json --contract gates.json \
  --reference ref/capture.jsonl --current current/capture.jsonl \
  --candidate candidate/capture.jsonl --out-dir NEW_DIRECTORY --max-rows 200000
```

A plan is finite JSON (≤64MiB), with explicit row selection **frozen before capture**:

```json
{
  "schema": "gufo-vision-tensors-v1",
  "scope": "encoder",
  "image_set_sha256": "<SHA256 of exact image manifest bytes>",
  "tensors": [{
    "id": "case/layer0", "case_id": "case", "stage": "layer0",
    "layer": 0, "head": -1, "stratum": "1024x1024/layer0",
    "width": 1152, "row_indices": [0, 31, 32, 4095],
    "input_sha256": "<RGB hash for encoder, QKV bundle hash for operator>"
  }]
}
```

Scope is `encoder` or `attention_operator`. Each tensor is a separate, unpooled gate
unit; case/stage/layer/head/stratum remain in summaries. `head:-1` means a whole
hidden/embedding row, not a head. For attention use one descriptor per head,
width72 and selected query indices. Row indices must be unique and increasing;
maximum total row count is an explicit refusal budget, not a sampling request.
Maximum width16384. Full embeddings can list every merged row; intermediates may
be a visibly declared sample. Sampled maxima do not assert a whole-tensor maximum.

Each of the three capture JSONL files starts with:

```json
{
  "schema": "gufo-vision-capture-v1", "plan_sha256": "<plan hash>",
  "role": "reference", "dtype": "f32le", "rows": 4,
  "producer": {
    "implementation": "independent-teacher", "revision": "<pin>",
    "arithmetic": "<actual casts/reductions/rounding>", "toolchain": "<pin>",
    "model_sha256": "<matched artifact identity>", "source_sha256": "<source hash>",
    "accumulator_dtype": "f32", "input_origin": "pixels", "independent": true,
    "independence": "<audited independent source/forward, no candidate hidden states>"
  }
}
```

Current/candidate roles have their own producer blocks. All arms require identical
model identity; independent source/implementation differs from both tested paths.
Operator reference origin is `frozen_qkv`, encoder reference origin is `pixels`.
Extra provenance (preprocess/RoPE/layout/precision/device/library/build identities)
is retained and pinned with the entire header, not ignored in comparison.

Then one row per plan tensor in plan order, then its indices in order:

```json
{"tensor_id":"case/layer0","row":31,"file":"chunk.bin","offset":4608,"sha256":"<full row hash>"}
```

Raw IEEE F32 little-endian only, width×4 bytes per row, aligned byte offsets,
confined relative regular files, explicit SHA-256. Convert stored BF16 values
exactly to F32 when exporting; label the pre/post-cast boundary. There is no
implicit transpose or head flattening. Missing/extra/reordered/corrupt/truncated/
nonfinite rows, mismatched headers and unregistered gate coverage are INVALID.

A contract `gufo-vision-gates-v1` has `plan_sha256`, `status`, nonempty `approval`
(or pending-approval explanation), positive `relative_floor`, `headers` mapping
reference/current/candidate to their **complete exact headers**, and `tensors`
mapping every tensor ID to these four entries:

```json
{
  "candidate_reference": {"atol":0.0,"rtol":0.0,"relative_l2":0.0},
  "current_reference": {"atol":0.0,"rtol":0.0,"relative_l2":0.0},
  "candidate_current": {"atol":0.0,"rtol":0.0,"relative_l2":0.0},
  "noninferiority": {"factor":1.0,"slack":0.0}
}
```

Zeros above are a format example, **not proposed model thresholds**. Per pair,
with left arm R and right arm C, every selected row must satisfy both:

- `max(abs(C−R)) <= atol + rtol*max(abs(R))`;
- `norm2(C−R)/max(norm2(R),relative_floor) <= relative_l2`.

Noninferiority also requires each row's candidate/reference max error ≤
`factor * current/reference max error + slack`. Both reference comparisons and
the direct candidate/current comparison must pass. A passing average cannot
rescue one failed row/tensor. Diagnostics include max absolute error, pointwise
`max(abs(delta)/max(abs(R),floor))`, relative L2, R RMS, raw-byte equality,
per-row gate decisions, worst absolute-error row per pair, full headers and hashes.
Pointwise relative error near zero is diagnostic only, not misreported as stable.
Signed-zero differences appear in byte equality but not numerical error.

Memory is O(row width + tensor count), plus the explicit bounded plan/contract;
raw data streams one row per arm. Outputs are exclusive, partial failures retained,
no successful report is published after invalid input. CLI0=offline PASS,
1=valid FAIL,2=INVALID. `PROPOSED` PASS is still only a hypothetical gate evaluation.

Existing `Encoder::Observer` exposes F32 arrays for `patch`, `layer0`…`layer26`,
and `embedding`; producers can export selected intermediate rows and complete
final embeddings. It synchronizes/copies: **capture outside timed runs**. It does
not expose QKV/attention; that diagnostic bench hook belongs to the kernel owner.

## Downstream teacher-token logits

```sh
python3 tools/quality/vision_downstream.py --image-set manifest.json \
  --history history.json --reference ref/logits.jsonl \
  --current current/logits.jsonl --candidate candidate/logits.jsonl \
  --current-contract current-gates.json --candidate-contract candidate-gates.json \
  --top-k 20 --max-positions 4096 --out-dir NEW_DIRECTORY
```

This calls [the full-vocabulary scorer](README.md) twice using the **same** reference
and exact history. Both comparison results and all per-row data remain; a failed
arm/stratum cannot be pooled away. The wrapper requires all frozen image cases,
after-image and text phases, and `history.provenance.image_set_sha256` matching the
exact image manifest. Freeze actual tokenizer/template/expanded histories separately
before capture; this tool does not tokenize or pretend synthetic token IDs are real.

Image sequences use `phase:"after-image"` and conditioning with `kind:"image"`,
`case_id`, `image_sha256`, `prompt_sha256` (UTF-8 exact prompt), and
`expanded_spans:[[start,end],...]` (half-open, ordered, nonoverlapping token spans).
Selected prediction positions must be at/after the final span end. Include actual
preprocessing/encoder artifact identities as additional conditioning fields.
Text controls use `phase:"text"` and exactly `conditioning:{"kind":"text"}`.
Use distinct case×resolution×phase strata; no overall pooling. Every arm consumes
the same teacher tokens, not its own generated continuation.

**Only invariant image/input identities belong in shared conditioning.** Output
embedding bytes are intentionally allowed to differ: put arm-specific embedding
hashes in pinned producer metadata, not the shared history conditioning hash.
Freeze exact header changes in the two #697 contracts. The reference producer must declare `vision_accumulator_dtype:"f32"`,
`vision_input_origin:"pixels"`, `vision_independent:true`, a nonempty
`vision_independence` explanation and `vision_source_sha256` distinct from both
tested producers' `vision_source_sha256`. All three `model_sha256` identities must
match; this wrapper isolates vision arithmetic, not a different model package.
Register these fields (with actual values)
in every producer header and the exact #697 change maps. Missing/contradictory
reference declarations are rejected before row scoring. A stratum cannot pool different image
case IDs or text prompt IDs; one good case cannot dilute a failing case.
Reference vision must be independent; a shared, unchanged text decoder can consume independently computed
vision embeddings in separately initialized sessions if the registered comparison
isolates only the vision change. Label that mixed pipeline explicitly—do not call
it an independently reimplemented whole language model. The wrapper verifies
identity declarations/spans, not that pixels were actually processed or that
rendering/tokenization was correct. See the model's numerical registration for
thresholds and separate task/speed admission.
