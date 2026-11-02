# Per-layer n-gram table precision: IQ4_NL or BF16

The PLE table is `per_layer_token_embd.weight`. Its GGUF descriptor selects
IQ4_NL or BF16; **there is no PLE precision command-line flag**. The binder and
on-disk row reader already support both. Changing PLE precision does not change
K/V cache precision, the other model tensors, or the F32 gathered-row consumers.

**Decision, 2026-10-27: IQ4_NL remains the recommended default.** BF16 is a
supported, internally tested opt-in artifact, not an overall quality upgrade.
The measured quality difference is a statistical tie; speed has no meaningful
cost or gain. The retained BF16 append artifact adds about 95 GiB of storage
without a demonstrated overall benefit. No C++ or global-default change is
needed. These are internal results, not public release qualification.

## Measured comparison and decision

Immutable internal report (private `gufo-agent-concurrency` record `docs/ple-final-decision-20261027.md` at `d9549bf6`, not published)
records source `337eaed6`, native ELF `f21c0020`, the same plain `q8_0` KV
policy, AR/MTP-off, S4/context 200000, matched body tensors and auxiliaries.
Quality scored 144 matched rows across 18 histories, at actual prefixes
1984–32831 (not an actual 200K quality test):

- KL(BF16 || IQ4), nats: mean **0.00351607**, p99 **0.06196798**,
  max **0.31206733**. P99 uses linear interpolation.
- Top-1 agreement **97.9167%** (141/144), not task accuracy.
- Mean NLL: BF16 **0.193845877**, IQ4 **0.196186947**.
- Delta NLL (BF16 minus IQ4): **−0.00234107** nats; paired whole-history
  bootstrap 95% interval **[−0.01024697, +0.00516589]**. No proven benefit
  or formal noninferiority result. The tool-JSON hint is favorable but unproven;
  code/prose category means slightly favor IQ4.

Lean speed used one four-slot-capable process per arm; S1 means one active
slot in that pool. Inputs were 32K synthetic agent-style histories with up to
128 decode tokens, first use plus three warm repetitions per cell. Values
below are the coordinator's rounded warm-median summary; S4 native rates are
per-request, not aggregate four-slot throughput. Ratios were supplied from the
summary, not recomputed from the separately rounded displayed rates.

| Metric | S1 BF16 | S1 IQ4 | BF16/IQ4 | S4 BF16 | S4 IQ4 | BF16/IQ4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native decode, tokens/s | 29.25 | 28.92 | 1.012 | 16.26 | 16.15 | 1.007 |
| Native prefill, tokens/s | 1705 | 1709 | 0.997 | 1633 | 1624 | 1.006 |
| TTFT | — | — | 1.003 | — | — | 0.994 |
| Cohort useful output throughput | — | — | 1.000 | — | — | 1.007 |

Startup was 21.7 s for BF16 versus 22.0 s for IQ4. Absolute TTFT/useful-output
rates, raw speed variance and the cache-drop outcome were not transferred with
this summary. First-use capture is not independently OS-cold qualification:
sequential cells/arms inherit cache state. These limits do not change the
owner's closed internal decision; do not call the point estimates statistical
equivalence, extrapolate to 200K prefixes, or transfer them to a newer binary.

### Why more table precision is not automatically better

Fidelity to original parameter values and predictive loss are different
objectives. BF16 preserves table values more closely, but NLL is the negative
log probability of the observed target token. A parameter perturbation can
raise or lower that probability; numerical precision does not guarantee
monotonic improvement. Only PLE changed, not the rest of the quantized body
or q8 KV. Interactions or error cancellation are plausible, not established
causes in this experiment.

The data offers a narrower explanation: most distributions barely move
(median KL about 0.00000524; 141/144 identical top-1), category effects have
both signs, and the overall interval crosses zero. Even the largest-KL prose
row keeps the same top-1 while favoring IQ4 target NLL (0.04040 versus BF16
0.29981). A weak tool-JSON mean hint is not demonstrated task success.
With measured utility tied, retaining the smaller artifact is reasonable;
this does not prove IQ4 universally superior or quantization always harmless.

The Q8-PLE reader is **permanently parked**, not another planned sweep or a
nightly migration candidate. This does not retire the independent q8 KV mode.

## Artifact packaging contract

[Reference BF16 artifact profile](artifacts/ple-bf16-profile.json) records the
four filenames, byte counts and full SHA256 identities. It is **metadata**, not
a runtime configuration file: Gufo does not consume this JSON as `--config`.

Keep all four splits together in a separate directory. Point `--model` at the
first split; the loader discovers the rest. The first split is metadata-only
and must not be omitted. Filenames alone do not distinguish the two variants:
the changed second split and its descriptor/hash do.

The reference artifact preserves all1223 non-PLE tensors, tokenizer metadata,
and unrelated descriptor bytes. Its BF16 table has GGUF dimensions
`[160, 320001536]`,102400491520bytes. The four-split logical size is
213735147168bytes (about199.1GiB). This is **not a RAM requirement**: PLE rows
are read on demand with a bounded row cache. The continuation-cache budget
`--cache-ram-bytes` is not a request to resident-load the whole PLE table.
Whole-model loading/inference
memory must still be measured under the requested sessions/context/KV policy.

The storage-constrained packaging route requires four filesystem reflink
clones, then appends the original BF16 table and patches only its dtype/offset
(12descriptor bytes) in split2. It leaves the old IQ4 table extent unreferenced;
it does not repack or requantize the other tensors. A destination must support
FICLONE; **ordinary copying is not an automatic fallback**. Preserve original
inputs, use a fresh output directory, retain the manifest/COMPLETE and full
hash verification, and maintain the required free-space reserve. Moving the
artifact to another filesystem requires a new storage/capacity assessment;
logical bytes are not physical allocation, and sharing is not portable.

Do not rename a publisher IQ4 artifact BF16 or treat an unrelated re-export as
the reference. The recipe and source weights must establish dtype, byte layout,
source identity, unchanged tensors, and output hashes before use. No model
payloads or conversion/download side effects are bundled with this page.

## Supported opt-in launch path

Use the documented [source build](../../../README.md#build-from-source) or an
existing verified main build. Coordinate model I/O and service lifecycle with
the host owner. The example is a private test endpoint, not a production
service replacement. Supply your normal private test credential without shell
tracing; do not put it in a checked-in file.

```sh
GUFO=/path/to/verified-main/bin/gufo
BF16_MODEL_DIR=/path/to/verified-bf16-ple-artifact
MMPROJ=/path/to/verified/mmproj-BF16.gguf
DISABLED_MTP="$BF16_MODEL_DIR/.mtp-disabled-do-not-create.gguf"
: "${GUFO_API_KEY:?use the normal private test credential mechanism}"
test ! -e "$DISABLED_MTP" && test ! -L "$DISABLED_MTP" || exit 1

"$GUFO" serve llm \
  --model "$BF16_MODEL_DIR/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
  --mmproj "$MMPROJ" --mtp-model "$DISABLED_MTP" \
  --host 127.0.0.1 --port 18080 --api-key "$GUFO_API_KEY" \
  --sessions 4 --context 200000 --kv-cache-type q8_0 \
  --prefill-chunk 4096 --speculative off --think off \
  --cache-ram-bytes 1073741824 --max-tokens 512
```

This explicitly selects **plain q8_0 KV**, independent of BF16 PLE; no YaRN,
rotated KV or MTP experiment is enabled. For an S1-configured server change
`--sessions 4` to `--sessions 1`; one active request on a four-slot server is
not the same allocation. The placeholder MTP path must remain nonexistent to
prevent accidental sidecar discovery in this comparison configuration.

If the build enables `GUFO_ENABLE_PREFILL_PREREQUISITES`, additionally pass
`--experimental-prefill-policy legacy --experimental-prefill-budget 0` to
match the explicit experimental-build controls. Ordinary builds do not expose
these two conditional options: omit them there. Do not mistake unknown-option
errors for BF16 incompatibility or silently enable adaptive prefill.

The measured comparison's native executable/source identity must be recorded
separately from a newer main build. Source support or a successful load is not
proof that a new binary has inherited older quality/performance measurements.

## Verification, decision and rollback

Before activation, verify the profile's full split/MMProj hashes and the
producer manifest/COMPLETE. Never mix split2 from one variant with neighboring
splits from another package by accident. Keep both variant directories intact.

Under the coordinated stopped-service test lease, verify `/ready`, all four
idle `/slots`, and the accepted context/KV policy. Then run a short uncached
completion and inspect actual token counts, errors, available memory and swap
growth. Readiness alone does not establish that the first prefill/decode fits.
If main differs from the measured build, retain its source/ELF/config identity
and this smoke result before calling the main configuration operational.

The decision is closed and workload-scoped: keep IQ4_NL recommended; BF16 is
supported and tested but not demonstrably better overall. Its retained
append/reflink artifact adds about 95.37 GiB (about 102.4 GB decimal); this
includes retention of unreferenced old IQ4 extents, not just the compact
BF16-minus-IQ4 table-size difference. Leave the existing BF16 artifact in place
unless its owner separately authorizes cleanup. Do not infer a universal
quality ranking from bit width or relabel this internal comparison as public
quality/performance or two-host portability qualification.

Rollback is only the model-directory choice: restore the original IQ4_NL
first-split path with the same applicable runtime options. Its original hashes
remain in [model identities](artifacts/model-identities.json). Do not delete the
BF16 artifact, overwrite the baseline, change a global precision default, or
restart production automatically. The owner's coordinated deployment procedure
continues to own service changes.
