# Gufo-only benchmark cards

Use a card when no reference engine supports a model. It records Gufo evidence,
not a comparative speed claim. Template: copy [benchmark-card.metadata.json](benchmark-card.metadata.json)
to an ignored artifact directory and fill every known field from the actual run.

```sh
python3 tools/bench/benchmark-card.py artifacts/serving.json \
  --metadata artifacts/card.metadata.json > artifacts/card.md
python3 tests/tools/benchmark_card_test.py
```

## Inputs and source contracts

`gufo bench` currently has **no `--json` option**. Its native output is a
Markdown-style `model / size / [params] / backend / test / t/s` table
(`src/cli/bench/bench.cpp`). `t/s` contains mean ± standard deviation across
repetitions. Do not invent a direct-bench JSON schema or infer milliseconds
from those rounded aggregate rates. `gufo-kernel-bench --json` is a different
microbenchmark and is not supported by this card generator.

The generator accepts:

- Native `gufo bench` text, or the Gufo pp/tg columns in the rendered model
  `BENCHMARKS.md` tables: retained as **aggregate-only**, never request timings.
- Existing `artifactType: model-bench-table`, `target: gufo` JSON from
  `tools/bench/model-bench.py` (`tools/gufo/model_bench/artifacts.py`). Single-user
  rows retain rates and counts, not raw phase durations; they too are aggregate-only.
- Existing `artifactType: servingBenchmark`,
  `benchmarkSchema: gufo.serving-benchmark.v1`, `workload.endpointProfile: gufo`
  JSON from `tools/serving/gufo-serving-bench.py`, also used by model-bench
  concurrency tables. Each `results.<group>.samples[]` entry supplies
  `repetition`, `request_index`, `queue_ms`, `prefill_ms`, `decode_ms`,
  `server_ttft_ms`, and `wall_ms` (`tools/gufo/serving_bench.py`). Every sample
  gets its own row; no averaging across requests. Null stage metrics remain UNKNOWN.

Retain raw inputs, settings, counts and the input SHA256 alongside the card.
The small JSON test fixture is synthetic, not measured performance evidence.
Review input and metadata for secrets/private paths before publishing; the
renderer is not a log scrubber. No reference columns or gains are imported.

## Metadata template contract

- `model`: actual loaded model and quantization identity, not just its alias.
- `artifacts`: one `{ "name": "target", "sha256": "..." }` per target shard,
  draft, projector and tokenizer/template artifact used. Compute SHA256 from
  the actual files; do not use a completion hash as a model hash.
- `stack.runtime`: explicitly identify **host HIP/ROCr** versus **Nix ROCm**.
  Record actual versions, including ROCm 10.1 only if that is what was used.
  A host `rocminfo` version is not proof of the libraries loaded by a Nix binary.
- `stack.hip`, `stack.rocr`: exact loaded/runtime provenance and versions;
  `compiler`, `commit`, `build_flags`, `hardware`: actual compiler, source SHA
  plus dirty state, full CMake/compiler settings, hardware/OS fingerprint.
  Resolve the measured binary's libraries/build records; never guess from a
  machine name, the current checkout, or a historical benchmark paragraph.
- `settings`: exact commands, model context/output limits, input/suite hash,
  prompt/output counts, sampling/seed, speculation/draft counts, concurrency,
  cache history (cold/warm and persistence), warm-up/repetitions, timed scope,
  power/temperature observations and harness revision. Preserve per-request
  actual counts in the input, not just requested token counts.
- `quality.status`: `PASS`, `FAIL`, `INCONCLUSIVE`, or `NOT_RUN`;
  `quality.evidence`: retained numerical quality/parity, selected functional
  suites, real tool results and timing qualification receipts with revisions.
  No missing-model skip is a pass. No reference engine means the unavailable
  comparison stays explicit; it does not automatically qualify quality.

The renderer never infers stack or artifact identity from rates. Missing
metadata or per-request phase evidence makes card status **INCONCLUSIVE**;
quality FAIL remains FAIL. Quality is separately shown verbatim, not silently
upgraded. A generated card is evidence packaging, not an automated quality gate.

## Card layout

The generated Markdown is the card template: model, card/quality statuses,
quality evidence, source digest and missing evidence; artifact SHA256 table;
exact stack; settings; per-request queue/prefill/decode/TTFT/wall milliseconds;
retained counters; and an explicitly aggregate-only appendix when applicable.
Keep timings mandatory per request/phase at the existing 5% and 10 ms policy
when comparing candidate/main. Preserve noisy runs as INCONCLUSIVE; do not
widen margins or average away individual regressions.
