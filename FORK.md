# Our Gufo fork

This is our forward-moving Gufo product line for AMD Strix Halo (`gfx1151`),
not a topic stack regenerated on each upstream release. Upstream supplies the
engine and model/API foundation; the following work is ours. The initial
candidate is the checked `integration-r5` composition (`bdb68d19`). Upstream base:
**`81ff13c3d394eb919d91074331ddd8d21701db45`** (v0.26.0 + #849).
Newer upstream is not included. The YaRN/MTP fix is folded into this foundation
(upstream #854, `0f34adcb`), not maintained as a separate ongoing topic.
q8 KV is fork-only: upstream #855 was closed without merging.

## Included work (historical topic names)

- `q8-kv`: experimental plain and rotated block32 q8 Flash-Next KV storage.
- `q8-speed`: exact q8 prefill/decode/verification optimizations.
- `kv-q4`: opt-in rotated q4 and rotated K=q8/V=q4 Flash-Next KV storage.
- `fork-docs`: q8 usage, evidence and numerical limitations.
- `perf-tracing`: opt-in Flash-Next serving diagnostics.
- `pinned-prefixes`: persistent, registered multi-prefix caching.
- `cache-lookup`: read-only advisory prefix-cache lookup API.
- `checkpoint-transfer`: portable checkpoint export/import API.
- `target-logit-probe`: bounded private numerical qualification probe.
- `cache-platform`: portable cache identity, diagnostics and contract checks.
- `shared-q4-mtp`: experimental shared-Q4 MTP loading and trace counters.
- `pd-prerequisites`: guarded prefill policies and P/D measurement prerequisites.
- `query-local-prefill`: query-local sparse attention in prefill.
- `q8-u2-compat`: composition/fixtures reconciling q8 with query-local prefill.
- `snapshot-image-identity`: invalidate stale image identity after failed restore.
- `cache-packed-transfer`: validate plain/rotated packed KV before publication.
- `npu-hybrid-wip`: default-OFF experimental XDNA2 projection offload.
- `fork-feature-docs`: fork feature, build and qualification documentation.

Parked `fork-maintenance` and the independent `agent-cache-r1`,
`q8-checkpoint-pages-r1`, `q8-rotation-route-r1` work are **not included**.
Their preserved branches remain available for separate admission.

## Defaults and limits

Normal release builds enable HIP; tests and development tools are OFF.
**F16 KV remains the runtime default**, not q8. Both q8 implementations are
compiled into HIP builds but require `--kv-cache-type q8_0` or `q8_0-rotated`;
plain q8's eligible transient prefill view is automatic. q8 is lossy and has
not met the original strict quality gates. Shared-Q4 MTP requires an explicit
compatible model and speculative selection. Our standard development profile
explicitly enables **shared-Q8 MTP at C1–C4** (`--speculative mtp --mtp-model …`);
the generic CLI defaults to speculation off, not implicit Q8 draft loading.
RAM caching defaults to AUTO (`--cache-ram-bytes 0`); `off` disables retention.
Disk caching, pins, `--cache-lookup-api`, `--cache-transfer-api` and `--trace`
require opt-in. `GUFO_ENABLE_NPU`, `GUFO_ENABLE_PREFILL_PREREQUISITES` and
`GUFO_ENABLE_PD_PROFILING` all default OFF; NPU remains unqualified WIP.
See [README](README.md), [q8 reproduction](docs/q8-kv-cache/reproduce.md) and
[testing](docs/TESTING.md): CPU passes and a short smoke are not model-quality,
performance or public-release qualification.

## Rotated q4 capacity options

**F16 remains the default.** For opt-in capacity, prefer
`--kv-cache-type q8_0/q4_0-rotated` (K=q8, V=q4); use
`--kv-cache-type q4_0-rotated` for maximum capacity. Both apply the existing
Hadamard basis and are lossy, Flash-Next-only modes. Plain `q4_0` and
`q8_0/q4_0` failed the numerical screen and are rejected by the CLI, not aliases.
The shared low-level q4 codec/geometry remains tested for the rotated paths.
Snapshot identities distinguish the policies and refuse mismatched restores.

Qualification: **screen-level pass (144-row screen); larger original gates not
run**. Rotated K8/V4 mean KL was 0.001949 with 99.31% top-1 agreement versus F16;
rotated q4 was 0.003314 /97.92%. Rotated K8/V4 is near the q8/BF16-K control
error level in this screen (plain q8 KL 0.001371; BF16-K 0.001796), not a claim
of irreducible noise or permission to subtract control error. Category/long
strata with only two histories remain underpowered. The original full corpus,
independent real-long, retrieval, greedy and coding-agent-loop gates were not run.

**Open speed warning:** free-running MTP can be materially slower with q4 KV;
its changed output histories and draft acceptance do not isolate loader cost.
The standard depth8192 benchmark measured rotated q4 about42.3 tokens/s and
rotated K8/V4 about42.2, versus rotated q8 about47.8 and F16 about96.2. Separate
fixed-ID whole-model AR measurements met the 0.97×q8 floor for all q4 policies,
with roughly1–2.4% slowdown. The HTTP S4 matrix also has decode/prefill/snapshot
timing flags; neither result proves general speed parity. No speed claim is made.

**More cache per GiB, slower restores until the fast path lands.** At8192 cached
tokens, S1 AR RAM restores measured303–304ms for rotated q4,184–186ms for rotated
K8/V4, and67–68ms for rotated q8. The extra235–237ms /116–118ms repeats across
all three warm/branch runs and is not an MTP-acceptance effect. Packed-row CPU
validation checks every q4 coordinate's half reconstruction, versus one overflow
check per q8 block. A validation-only CPU replay confirms approximately linear
scaling from8192 to16384 rows per component; it is not a full-restore GPU profile.

**Estimate, not a long-restore measurement:** extrapolating that per-token cost,
a100K-token restore would take about3.7s /2.3s /0.8s for rotated q4 /rotated K8/V4
/rotated q8; at150K, about5.6s /3.4s /1.2s. That is roughly4.3s /2.1s extra at150K,
which matters when returning to long conversations. Actual larger-context times
can differ. The validation fast path is a separate follow-up, not part of this
landing; the registered timing failures remain failures.

Measured S1 memory at100000-token capacity, actual target frontier99899:

| Storage | Target tokens/GiB ordinary live payload | Session allocation (decimal GB) |
|---|---:|---:|
| F16 |40330|3.08|
| q8 |75915|1.83|
| rotated q4 |143396|1.16|
| rotated K8/V4 |99274|1.49|

Plain/rotated variants have equal persistent geometry. Session allocation
excludes shared scratch; ordinary borrowed spans are canonical payload, not an
extra physical allocation. These are not retained-checkpoint RAM measurements.
The memory run auto-loaded vision; its85.16GB model residency and separate1GiB
backing allocation are not in the session column. Raw per-format results and
limitations are retained in the private `gufo-agent-concurrency` repository,
`docs/kv-q4-r1.md` and `bench/kv-q4-r1/`.

## Known limits

MTP output can differ slightly from plain decoding; cause not yet identified. SSD restore with images is exact (same-MTP live vs disk matches).

## Build and process

With the dependencies described by README installed:

```sh
nix build && ./result/bin/gufo diagnose
# Or native CMake:
cmake --preset release
cmake --build --preset release --parallel 4
# CPU contracts:
cmake --preset cpu-test
cmake --build --preset pr --parallel 4
```

Forgejo is primary. After the lead's one-time establishment, `main` moves
forward only, never rebased/regenerated. New `topic/*` branches start from our
`main`; deliberate `intake/*` branches admit selected upstream changes (default:
skip) through the same tests/review. Preserve retired refs with immutable
`archive/*` tags before any cleanup. Nothing is published publicly without
owner approval. The candidate itself does not move `main` or retire any ref.
