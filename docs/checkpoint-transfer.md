# Portable continuation checkpoints (experimental, opt-in)

Roadmap C step 2; based on `gac/cache-lookup-r2` (`355cddef`). This is a
**separate claim class** from advisory lookup. CPU tests do not establish GPU
restoration, cross-host numerical parity, production readiness or transfer speed.
No model arithmetic or Flash-Next payload ABI changes: payload version remains
**20**. Linux x86-64 / gfx1151 / Qwen3.8 Flash-Next only.

## Enable and call

Add `--cache-transfer-api` to `gufo serve llm`. Both routes are absent (404)
without it. `--cache-lookup-api` remains independently opt-in. Use the existing
API-key policy and a trusted, allowlisted LAN, or an authenticated encrypted
tunnel. Do not expose these routes to the Internet or untrusted tenants.

Provision an **encrypted, disk-backed `TMPDIR`** with at least 8 GiB free per
server before enabling transfers. The default system temporary directory may be
tmpfs: a 2.7 GiB spool there consumes RAM and is unsuitable on a tight host.
Spools are created 0600, unlinked immediately, close-on-exec, and closed on every
exit/disconnect. No client paths or filenames are accepted.

```text
POST /v1/cache/export
Content-Type: application/json
Authorization: Bearer <existing server API key>

<the same complete object as POST /v1/chat/completions>
```

Rendering, constraint-authored framing, tools, image preparation, tokenizer,
thinking/preserve-thinking and stable-prefix selection are engine-owned, shared
with lookup/generation. Sampling controls are validated but no sampler,
generation, prefill or new snapshot capture is submitted. `stream:true` does
not select SSE. `cache_prompt:false` makes export a miss. V1 accepts a full chat
body, not a lookup handle; R2 lookup does not issue reservation handles.

Success is `200 application/vnd.gufo.checkpoint`, `Content-Length`,
`Cache-Control: no-store`, with a binary streamed body. It exports the deepest
**immutable, exact reusable checkpoint**, using the native prefix-index rules,
including input identity and stable-fallback checks. It can be RAM or disk.
Live frontiers are not checkpoints: they are skipped until their serving owner
has captured them. Thus export depth can be smaller than lookup's live depth.
A full-prompt boundary must attest decode readiness. No arbitrary GDN rewind.

```text
POST /v1/cache/import
Content-Type: application/vnd.gufo.checkpoint
Content-Length: <exact complete blob size>
Authorization: Bearer <existing server API key>

<blob bytes>
```

`Content-Length` is required; chunked uploads are rejected. Clients should
materialize a private file and upload it, not pipe an unknown-length response
into chunked HTTP. `Expect: 100-continue` is supported. Authentication, route and
content-type checks precede the first upload read/100 response. Export request
JSON still uses the ordinary body limit; only import has the larger limit.

```json
{"admitted_tokens":98304,"payload_bytes":2880000000,"epoch":"<sha256>","tier":"ram"}
```

Illustrative numbers, not measurements. Admission means an immutable, evictable
RAM checkpoint, **not** a live session, durable save, retention lease, routing
reservation or proof that a later restore will succeed. Send the unchanged full
chat request afterwards. The ordinary engine still verifies token/image matches,
stable dependencies and all model restore/Validate conditions before execution.
No suffix-only protocol, imported RNG rewind or automatic request retry.

### Errors

| Status | Stable code / meaning |
| --- | --- |
| 400 | `cache_transfer_invalid`: malformed/truncated/corrupt envelope or payload; invalid prompt has `cache_transfer_invalid_prompt` |
| 401 | Existing server authentication failure |
| 404 | Route disabled, or `cache_transfer_not_found` for no immutable reusable boundary |
| 409 | `cache_transfer_epoch_mismatch`, `cache_transfer_layout_mismatch`, `cache_transfer_context_mismatch`, `cache_transfer_source_changed` |
| 413 | `cache_transfer_too_large` (codec) or existing HTTP `payload_too_large` |
| 415 | `cache_transfer_content_type` |
| 501 | `cache_transfer_unsupported` for another backend/request shape |
| 503 | Existing loading/device-lost errors, or `cache_transfer_busy` for the transfer quota/contended export owner |
| 507 | `cache_transfer_budget` or `cache_transfer_spool_io` |

A failure while streaming export closes the response short of Content-Length;
it never appends JSON to the blob. A recipient must reject incomplete data.
Normal source filesystem/device failures may also return the existing 5xx error.

## Compatibility and exact identities

The transfer epoch is SHA-256 of three u64-length-prefixed fields:
`gufo-checkpoint-transfer-v1`, raw adapter compatibility identity, configuration.
The displayed epoch is the hex encoding of that digest. It is **not** the R2
lookup epoch (which intentionally includes local cache paths/budgets).

The header carries both the raw identity and inspectable configuration:

- Full-content GGUF hashes (every shard, metadata, padding and tensors), target
  quant/tokenizer/template identity and enabled MTP sidecar hash. Enabling the
  flag forces artifact fingerprinting even with disk/pins OFF. Typed embedding
  callers must supply valid fingerprints. Native identity also binds context,
  position/RoPE/YaRN policy, MTP/draft limits/concurrency, prefill capacity and
  component ABI/layout. There is no human model-name-only compatibility.
- Payload version 20, exact executable SHA-256 (including kernel code), loaded
  runtime-object content hashes, fixed Linux/x86-64/little-endian/gfx1151 lane,
  image-encoder identity, relevant GUFO/HIP/HSA/ROCR/ROCBLAS/AMD environment
  hashes, prefill chunk/decode-active policy and server reasoning defaults,
  including preserve-thinking. Environment values and host paths are not sent.
- Exact token vector, count and SHA-256 of uint32 little-endian token IDs;
  cumulative native image/input prefix identities and selected-boundary input
  identity. These are image identities, not image pixels. The later request
  supplies matching image attachments. Request-specific reasoning overrides
  affect the rendered token vector; there is no mode switch that bypasses exact
  token/input matching. Defaults must match between hosts.
- Component descriptors, exact target/private and draft row frontiers, stable
  fallback attestation, decode-readiness bit, byte lengths and per-object hashes.

No local cache directory, RAM budget, pin-registration name or process-local
pointer is a portable key. Incoming checkpoint/chunk/lineage IDs are **not**
installed: a fresh local `ExecutionHistory` allocates fresh IDs and a fresh
lineage. No sharing is inferred from token equality or sender IDs. Re-exported
payloads are byte-identical; local envelope IDs may differ.

Compatibility is deliberately stricter than native disk ABI. Different builds
are rejected even if their payload version is still 20. This closes the U1
failure mode: arithmetic changes cannot inherit stale remote state merely
because a payload-version bump was missed. V1 does not promise compatibility
across compilers, runtime lanes, architectures or future protocol versions.

## Wire format, protocol 1

All integers are **unsigned u64 little-endian**, except native manifest fields
(which retain native v2's documented explicit encoding) and token IDs (u32).
There are no native C++ struct images in the envelope. `field` is u64 length
followed by that many bytes. SHA-256 values are 32 raw bytes.

```text
8 bytes magic: GUFOCK01
u64 total header bytes, including magic/size and trailing header SHA-256
u64 established stable tokens; u64 decode-ready (0/1); u64 total payload bytes
32 bytes compatibility epoch; 32 bytes token-prefix SHA-256
field adapter identity; field configuration
u64 image/input prefix count
  repeated: u64 token boundary; field cumulative input identity
field native EncodeManifest(v2) bytes (including its CRC-64)
u64 payload object count
  repeated: 32 bytes object SHA-256
32 bytes SHA-256 of every preceding header byte
body: objects in manifest component order, each component's chunks, tail, private state
```

The manifest contains exact token count/vector and each object's size and
CRC-64/ECMA-182 as well as component geometry. Header is capped at 16 MiB;
**entire blob is capped at 8 GiB** with overflow-checked sums. Header sizes,
counts, stable/image boundaries, identity/epoch, token hash and geometry are
checked before payload receipt. Every complete object's SHA-256 and CRC are
verified before any RAM allocation/admission/eviction. Trailing or missing bytes
within the declared upload are rejected. Header hash provides integrity, not
sender authentication: a trusted sender can construct new valid blobs.

## Ownership, publication and side effects

One transfer quota per loaded pool covers import or export, including slow
export responses; excess calls fail immediately. Existing max-connections and
120-second socket timeouts apply. Import reads/export writes also check a
two-minute request deadline between pieces (a blocking socket call may consume
one additional socket-timeout interval). There are no background transfer jobs.

Export briefly try-locks native owner metadata to select/pin an immutable
candidate, without the pool admission mutex, an execution lease or LRU touch.
It releases that lock before reading payloads. Checkpoint and disk publication
pins keep the source coherent across eviction. Borrowed RAM rows acquire a
source pin for **one bounded copy**, using a transfer stream and waiting for its
completion when the bytes are still in a device slot. Pins are released before
spool/network I/O. Export may therefore wait for preservation/device transfers;
slot reassignment may wait for that one source copy. It never reads an unguarded
live pointer or captures/clobbers a live session. Disk export uses the existing
verified `ReadPayload` API and publication pin.

Export spools and hashes once before sending the header, then streams the spool
in 1 MiB pieces. Import verifies into a separate bounded spool before touching
retention. It checks the registered local inventory/context, then uses existing
host backing allocations and `RetentionPolicy::Admit` for RAM bytes, record
limits, protected pins and eligible eviction. Partial allocations unwind; failed
index reservation releases payloads before a retry. Backing may grow/register
host pages through the existing provider and can be refused under memory
pressure. No slot acquisition, GPU restore, prefill or live frontier mutation.
Valid imports can evict ordinary RAM records; invalid/corrupt/epoch-mismatched
imports cannot. Capacity refusal can follow eligible eviction, as in ordinary
capture admission. Pins and existing backing charges are never bypassed.

The final RAM copy rechecks each object's SHA against the actual spool bytes
before publication, so post-receipt spool corruption cannot silently enter RAM.

RAM copy/admission/publication serialize under the component owner. This can
delay new admission or checkpoint publication, but existing slots can continue
executing. R2 owner invalidation/publication runs around that mutation; lookup
still reads only published metadata, never these locks, spool files, source pins
or device work. An overlapping import produces a conservative lookup miss; a
completed publication becomes visible immediately. Import is not hidden work on
the scheduler/generation queue.

## Sensitivity and limits

**Blobs contain token IDs, logits, recurrent/KV state and image identities. Token
IDs can reconstruct prompt/code/secrets. Treat them as private conversation
content**, not harmless cache metadata. No prompt/token/hash/path/body is added
to request logs or `--trace`; transfer routes are excluded from body tracing.
Existing transport logs contain route/status/counts only. Caller-managed saved
blob files need their own 0600/encryption/retention policy. Checksums are not
access control, encryption, tenant isolation or protection from a malicious
trusted worker.

V1 intentionally has no resumability, chunk negotiation/deduplication, durable
import, transfer reservation handles, shared tenant namespace or router changes.
Spooling adds local read/write/hash costs and SSD traffic; **1–3 seconds is a
hypothesis, not a throughput claim**. Measure complete source read+spool,
network, destination verification+spool, RAM admission and subsequent restore,
not just bytes/link-rate. Avoid repeated multi-GiB writes on QLC SSDs without a
budget. A future direct/chunked preparation protocol needs a separate contract.

## Verification and later GPU gate

On the Agent PC, the HIP-OFF build and 10 focused CTests pass; the envelope and
component suites also pass ThreadSanitizer (`halt_on_error=1`). All 40 existing
lookup-harness unit tests pass. Changed C++ files pass Clang 22.1.8 formatting;
the full shared check still flags inherited, untouched
`src/core/json_constraint.cpp:2003`. No HIP build or GPU run is claimed.

CPU targets: `checkpoint_transfer_test`, `component_text_cache_test`,
`http_server_test`, plus existing runner/disk/cache/chat/scheduler regressions.
They exercise byte-exact wire and payload round trips, every truncation of a
small fixture, header/object corruption, extra bytes, incompatible model/ABI/
build configuration, bounded size refusal, capacity refusal, RAM/disk sources,
image-identity misses, borrowed sources, quota release, 9 MiB HTTP streaming,
auth-before-100, and serving/lookup during import receive and owner admission.
No weights or GPU are used. CPU/HIP-OFF compilation is not HIP compilation.

Before deployment, reserve **both hosts through their `~/gac/QUEUE.md` and
operation locks**. Package the exact private revision once on an unloaded
builder, install identical immutable runtime images on both hosts, then validate:

1. A exports; fresh B imports with no prefill. Check exact admitted depth and
   actual `cached_tokens/cache_n` on the unchanged next chat. Repeat B→A.
2. At 2K, 32K, 100K and an unaligned learned boundary (including 2415), compare
   greedy imported-warm to cold bypass on B and the same cold reference on A.
   Compare **all output token IDs, stop/counts, top-k IDs/order and raw target
   logits at the restored boundary and teacher-forced suffix rows**, not text
   alone. Run AR and real-MTP, tool framing, preserve-thinking on/off and images.
   HTTP logprobs are not implemented: extend the native/component integration
   driver or use a private, bounded test-only target-logit probe in **identical
   server builds on both hosts**. That GPU transfer driver/probe remains to be
   prepared for the queued run; the current CPU driver is not it. A different
   helper executable must not bypass the executable epoch to import a blob.
   Observe logits from the restored executing slot, not just the copied logit
   payload; never pretend HTTP text or payload byte equality proves numerics.
3. Reject cross-build (even ABI20), ABI18/19, changed target/MTP/vision, context,
   numerics environment/default policy; reject truncated, bit-flipped, oversized
   and under-budget imports. No admitted record/live-session change for invalid
   blobs. Advance unrelated busy slots throughout import; compare their outputs
   with a no-transfer control. Check pin survival, eviction and R2 latency.
4. Measure 100K blob bytes, SHA/CRC/spool/copy/restore phases and end-to-end
   latency, peak RSS/MemAvailable and SSD bytes for ordinary LAN and USB4 IP
   `10.44.0.1/10.44.0.2`, both directions, three repetitions. Keep transfer and
   inference-interference timings separate. Compare against measured cold
   re-prefill, not the historical ~65–70 s estimate.

Require exact same-build token/logit parity (including top-k), zero unsafe
admissions and preserved unrelated sessions. If cold cross-host controls already
differ, stop and localize; do not loosen tolerances to qualify transfer. Preserve
failed/partial runs. No GPU run, queue modification or production change is
part of this code-only implementation.
