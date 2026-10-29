# Offline matched-history quality

For synthetic images, independent F32 attention fixtures and three-arm
encoder/downstream comparisons, see [the vision harness](VISION.md).

`matched_kl.py` scores **complete vocabulary logits** against one frozen set of
teacher-token histories. It reports per-token log probabilities, NLL and rank,
KL(reference || candidate), top-1 agreement, top-k overlap, and per-stratum tails.
It does not load a model, generate text, call an endpoint, or choose quality gates.
A pinned contract explicitly registers allowed metadata changes and per-stratum gates.
Python 3.10+ standard library only; no NumPy, PyTorch or network dependency.

```sh
python3 tools/quality/matched_kl.py \
  --history artifacts/quality/history.json \
  --reference artifacts/quality/reference/capture.jsonl \
  --candidate artifacts/quality/candidate/capture.jsonl \
  --contract artifacts/quality/contract.json \
  --top-k 20 --max-positions 4096 --out-dir artifacts/quality/comparison
```

The output directory must not exist. Exit 0 means a complete measurement with
logit-gate status `PASS` or `NOT_EVALUATED` (empty gates); exit 1 means a registered
logit gate `FAIL`. **Neither exit 0 nor logit PASS is model qualification**:
`qualification` stays `NOT_QUALIFIED`, with scope `offline_logit_gates_only`.
Exit 2 means invalid/incomplete inputs; no success report is produced. An error after capture processing begins retains
`failure.json` and any partial positions for diagnosis; do not resume into that
directory. Reusing a previous success directory is refused.

## Independent reference, not self-consistency

Generate the token history once, then freeze it and its hash **before** scoring.
For scored absolute position `p`, both producers must emit the unsampled logits
**after consuming exactly `tokens[:p]`, before consuming `tokens[p]`**. Position
zero cannot be scored without a preceding context; include any required BOS and
chat-template tokens explicitly. All subsequent conditioning uses these fixed
teacher IDs, never either candidate's free-running choices. A producer must
record the actual route/partition separately when it matters to the experiment.

The reference must come from an **independent implementation**, such as a pinned
HF/PyTorch or llama.cpp reference appropriate to the model/precision, not Gufo
scoring its own output. Export its complete logits with the same tokenizer ID
mapping; numerical normalization is performed here, without importing a Gufo
sampler, engine or served scorer. The reference producer must describe its
independence and exact revision/artifacts/arithmetic/toolchain. A different
precision flag or executable name on the same implementation is not independence.
The tool rejects identical implementation identifiers, but a metadata assertion
cannot prove code independence: **audit the producer and retain its source and
capture commands**. This intake supplies the offline importer/analyzer, not a
model-specific capture adapter or newly qualified real-model reference. The CPU
math oracle independently uses 100-digit Decimal exponentials and probability
ratios; it is a math test, not model-quality evidence.

For precision matching, score each candidate policy against the same independent
reference/history set. Compare strata under **preregistered** gates before using
speed measurements. Changing the reference precision, tokenizer, template,
history, scored windows or model artifacts changes the experiment. Keep weights,
activation, accumulator, KV, prefill/decode and rounding policies in `arithmetic`
and extended provenance; do not label two runs merely “F16” if these differ.
Transport/P/D exactness still requires its independent state-byte, token and
cache-work controls. Small KL or identical logit hashes cannot waive those gates.

## Format v1

### Frozen history JSON

```json
{
  "schema": "gufo-matched-history-v1",
  "vocab_size": 4,
  "tokenizer_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "provenance": {"corpus": "illustration-only", "template": "explicit-token-ids"},
  "sequences": [
    {"id": "example", "prompt_id": "prompt-1", "phase": "continuation",
     "stratum": "code/short/cold/gpu/continuation", "conditioning": {"kind": "text"},
     "tokens": [0, 1, 2, 3],
     "positions": [1, 3]}
  ]
}
```

The hashes and tokens above are illustrative, not a model benchmark corpus.
Use the SHA-256 of the exact tokenizer assets/ID mapping, not a tokenizer name.
`provenance` is a nonempty object retained in the report; record corpus revision,
source prompt-set hash, template/tokenizer settings and teacher-generation seed
there. Sequence IDs are unique; `prompt_id` groups windows from the same prompt.
`phase` is explicit (for example `prefill-boundary` or `continuation`); a stratum
cannot pool different phases. Strata are nonempty explicit strings preserving
**domain × length × history mode × execution mode × phase**, for example
`code/4096/cold/gpu/prefill-boundary` or `reasoning/128K/restored/npu/continuation`.
For matched arms use the same comparison-stratum label; record each arm's actual
execution policy in its producer metadata, with changes registered below. Scored positions are nonempty,
strictly increasing integers in `[1, len(tokens))`; all tokens are valid integer
vocabulary IDs. Sparse **positions** are supported; sparse vocabulary rows are not.
For differently labeled windows, use separate sequence IDs with the same frozen
tokens and disjoint selected positions. There is no implicit prompt truncation,
window selection or sampling.

Every sequence has an explicit nonempty `conditioning` object: for text use
`{"kind":"text"}`. For image prompts retain image content hashes, actual expanded
image spans, preprocessing/crop settings and identity, mmproj/encoder artifacts,
and any injected embedding identity. Hash the whole object as UTF-8 JSON with
sorted keys, compact separators, ASCII escaping (`ensure_ascii=True`), no NaN/Inf.
Each capture row must carry this recomputed `conditioning_sha256`, in addition
to its token-prefix hash. Thus equal token IDs alone cannot substitute text for
image conditioning. This object is a frozen producer-owned identity, **not proof
that image preprocessing or the declared spans were executed correctly**. Producers
must retain/validate the actual image and processed inputs. Score the downstream
prediction positions after expanded image spans, and use separate `after-image`
and `text` stratum labels; do not treat image-span placeholders as generated target
IDs. Compare flash and current image paths separately to the same independent
F32-accumulation reference; no native vision/reference claim is made by these CPU tests.

The exact history file bytes, not a short digest or reconstructed subset, are
SHA-256-bound in both capture headers. Limit: 64 MiB of history JSON, vocabulary
at most 1,000,000, and an explicit positive position budget (`--max-positions`,
default 4096). Exceeding the budget rejects the whole input, never drops rows.
A larger explicit budget increases scalar-summary memory and runtime.

### Streamed capture index and compact chunks

Each capture is JSONL: one header, followed by exactly one record per selected
position, in history sequence order then position order. No blank lines,
duplicate JSON keys, extra/missing/reordered positions, or nonfinite constants.
Each index line is limited to 64 KiB. Extra metadata keys are retained in the
header; v1 required fields retain their meanings.

Header (replace illustrative hashes with real hashes):

```json
{"schema":"gufo-full-logits-v1","history_sha256":"<64 lowercase hex>","tokenizer_sha256":"<same as history>","vocab_size":248320,"positions":33,"dtype":"f32le","producer":{"implementation":"independent-reference-engine","revision":"<pinned source revision>","model_sha256":"<artifact or ordered artifact-manifest SHA256>","arithmetic":"<weights/activation/accumulator/KV/rounding and execution policy>","toolchain":"<compiler/runtime/library versions>","independence":"<why this reference does not share the candidate implementation>"}}
```

`independence` is required on the reference producer; other producer fields are
required on both. Full producer objects are retained, so additional hardware,
partition, precision, corpus, program, build and capture-configuration hashes can
be added without modifying the analyzer. Capture headers must agree with the
history's tokenizer, vocabulary, history SHA-256 and selected-position count.
Model hashes may differ intentionally (for example quantized artifacts), but
that difference must be part of the registered experiment.

Position record:

```json
{"sequence_id":"example","phase":"continuation","position":135,"token":760,"prefix_sha256":"<64 lowercase hex>","conditioning_sha256":"<64 lowercase hex>","file":"chunk-000.bin","offset":0,"sha256":"<64 lowercase hex>"}
```

- `prefix_sha256`: hash of `tokens[:position]` packed as **uint32 little-endian**,
  without a header. The scorer recomputes this independently of both producers.
- `token`: exactly `tokens[position]`, not a sampled or imputed token. `phase`
  and `conditioning_sha256` must match the frozen history. `(sequence_id, position)` is the stable row ID;
  row byte length is explicitly determined by header vocabulary × dtype size.
- `file`: relative regular-file path within the capture index's directory
  (symlink/path escapes refused); `offset`: nonnegative aligned **byte offset**.
- `sha256`: hash of this position's exact raw row bytes, verified before scoring.
- `dtype`: one capture-wide `f32le` (IEEE binary32) or `f16le` (IEEE binary16).
  Each row has exactly `vocab_size` values in token-ID order. Chunks have no
  embedded header and may contain multiple contiguous rows; offsets select rows.
  The analyzer reads exactly one complete row per position, not an entire chunk.
  Producers must reject partial rows. Every value must be finite, including tails.

Using raw little-endian chunks avoids `.npy`/pickle loading and third-party
libraries; exporters can write an array's little-endian bytes directly. This is
not a safetensors/NPY file parser. Do not silently downcast a reference to save
space: f16 capture quantization is part of the numerical experiment and must be
recorded. Prefer f32 for established quality gates.

At vocabulary **248,320**, one f32 position is **993,280 bytes** (970 KiB), f16
**496,640 bytes** (485 KiB). A pair of f32 captures for 4096 positions is
8,136,949,760 raw bytes (~7.58 GiB), plus small indices. The scorer holds one
reference/candidate row pair plus O(vocabulary) binary64 work arrays; it never
loads the full position-by-vocabulary matrix. It retains O(selected positions)
scalar metrics/identities for exact empirical quantiles. Per-row top-k lists are
written out and discarded, not accumulated for the entire corpus. Keep full
captures in ignored `artifacts/`, not Git; this tool does not delete inputs.

## Pinned comparison contract

`--contract` is mandatory, versioned `gufo-matched-quality-contract-v1`. Freeze
it **before capture/measurement**, and retain it with the artifacts. It contains:

- `history_sha256`: the exact history file's hash, and `top_k`: the explicit
  diagnostic K (must match the invocation).
- `reference_header`: the **complete reference capture header object**, pinned
  exactly, including extra provenance. Both arms use the same metadata key
  structure; use explicit values rather than silently omitted keys.
- `expected_changes`: every reference-to-candidate header leaf difference, keyed
  by JSON Pointer (`~0` encodes `~`, `~1` encodes `/`), each with exact `reference`
  and `candidate` values. For example, `/producer/implementation` and
  `/producer/arithmetic`. Lists are whole leaves. Missing, unexpected, stale or
  no-op exemptions fail; no “ignore metadata” option. A dtype change must also
  be registered. Unchanged fields are pinned by `reference_header`.
- `strata`: **all and only** the frozen strata, each with expected `positions`,
  unique `prompts` count and `gates`. Every row and prompt must be accounted for.

One illustrative stratum's contract, using NPU's **prospective** gates (not a
universal safety claim or tool defaults):

```json
{
  "positions": 264,
  "prompts": 8,
  "gates": {
    "kl_mean": {"max": 0.002},
    "kl_p95": {"max": 0.01},
    "kl_max": {"max": 0.05},
    "top1_agreement": {"min": 0.99},
    "perplexity_ratio": {"max": 1.005}
  }
}
```

Specify appropriate counts and separate prefill/continuation strata for the
actual frozen windows. Gates allow inclusive `min`/`max` bounds on the summary
metrics listed below. A missing/null/nonfinite gate value fails; empty gates
produce `NOT_EVALUATED`, never PASS. Any failing stratum makes the overall logit
status FAIL; otherwise any ungated stratum makes it NOT_EVALUATED. The report
retains each threshold, actual value and decision plus the contract SHA-256.
The tool cannot prove the contract was registered in advance; keep the pre-run
commit/receipt as the chronology evidence. Unknown transport/state/task gates
cannot be represented as implicit successes here.

## Output and interpretation

`positions.jsonl` retains sequence, stratum, absolute position, token and prefix
hash; both row hashes; per-token logprob/NLL/rank; top-k IDs; top-1 agreement;
intersection-size/k overlap; full-distribution KL; raw max absolute logit delta;
reference top-1 margin (null for a singleton vocabulary); and a same-format
logit-hash equality flag. This equality flag is not a state snapshot or transport
proof. The largest-KL row/prompt is identified overall and for every stratum.

`report.json` retains input identities, complete producer provenance, contract and
scorer-source hashes, bytes per position, the supplied budget, and overall plus
**every stratum**:

- position and unique prompt counts; KL mean, p50, p95, p99 and maximum;
- top-1 agreement and top-k overlap;
- mean reference/candidate teacher-token NLL and their signed delta;
- perplexity ratio `exp(mean(candidate NLL − reference NLL))`.

Positions have equal weight within each reported group. Quantiles use linear
interpolation at `(n−1)q`; no mean of stratum means or averaging away a failing
stratum. NLL uses natural logarithms (nats); one-based ranks tie-break by lower
token ID. Overflowing perplexity ratio is JSON `null` with an explicit overflow
flag, never silently dropped. All other metrics must be finite. Per-row data
allows consumers to impose stricter per-request/per-position gates.

KL sums over **all vocabulary entries**, with stable float64 log-softmax and
`math.fsum`, no top-k truncation, missing-mass imputation, probability floor, or
renormalized top-k distribution. “Exact KL” here means full-distribution rather
than sparse-estimated KL, not symbolic exact arithmetic. Binary64 exponential
underflow is possible; only roundoff-scale negative KL is clamped (the bound is
documented in `scoring_metrics.py`), not a quality tolerance. No NaN/Inf row is
ignored or averaged out. Raw logit delta changes with a constant offset even
when probabilities do not.

No default scientific thresholds are baked in: NPU/q4/P/D consumers must register
their own per-stratum acceptance, any additional per-row requirements, and retain
the reference and candidate artifacts. Accuracy metrics do not qualify free-running task/tool behavior,
cache correctness, performance or NPU hardware safety.

## Tests, source and limits

```sh
python3 -B tests/tools/scoring_metrics_test.py
python3 -B tests/tools/scoring_contract_test.py
python3 -B tests/tools/matched_kl_test.py
# Also registered with CTest and the hosted check-pr target.
```

The independent Decimal oracle covers full-tail KL, shift invariance, ties,
underflow/extreme logits and percentile/NLL semantics. File/CLI tests exercise
f32/f16 chunks, actual 248,320-column rows, bounded reads, corrupt/nonfinite
rows, mismatched token/image conditioning, after-image versus text strata, invalid
budgets, missing/extra rows and output
isolation. These are CPU tooling checks, not a native/model quality pass.

Adapted from **jabata**, upstream PR #697, `feat/prompt-logprobs-scoring`,
`c1ab796010ba2a72278e692c6d5e17d363235373`, under the repository's MIT license.
Retained ideas: freeze teacher IDs, score identical conditioning, exact teacher
NLL/rank and top-k agreement, grouped reports. Not imported: served prompt-scoring
API/scheduler, Flash-Next-only unbatched execution, built-in Qwen chat framing,
network capture, or top-20 imputation. The offline format is model-family-agnostic;
actual full-logit export and independent-reference support remain producer-owned.
This is not a claim that every model already has such a capture adapter.

Consumer follow-ups: NPU's unchanged-reference repeat and matched transport-byte
comparisons remain separate protocols, not independent-quality comparisons
(disguising the same implementation under two names is invalid). This tool
requires a genuinely independent reference forward path; the existing Gufo scalar
reference probe may supply bounded independently implemented fixtures after its
producer is audited, but is not a practical 128K exporter. Target-logprob-only
inputs are rejected, not partially qualified. Real fixed corpora, long-context
capture adapters, selected-operator/program/ABI identities and runtime proof of
actual completed NPU work remain consumer-owned; preserve those fields in the
pinned producer metadata. No hardware or real-model claim is made here.
