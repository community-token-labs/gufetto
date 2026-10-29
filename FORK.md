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
Vision-enabled startup charges one shared encoder workspace in addition to
per-slot input storage. After allocating slots, cache budgeting protects their
still-unallocated execution storage and deferred shared scratch; resident
weights and eager state are not charged twice. AUTO halves the remaining
protected headroom, so correcting formerly unprotected promises can reduce the
cache budget. Explicit and disk-only backing retain execution/OS reserves too.
The frozen growth guard is conservative after lazy buffers materialize. CPU
image preprocessing, other processes and runtime overhead remain separate
pressure sources; this is not an unconditional OOM guarantee.
RAM caching defaults to AUTO (`--cache-ram-bytes 0`); `off` disables retention.
Disk caching, pins, `--cache-lookup-api`, `--cache-transfer-api` and `--trace`
require opt-in. `GUFO_ENABLE_NPU`, `GUFO_ENABLE_PREFILL_PREREQUISITES` and
`GUFO_ENABLE_PD_PROFILING` all default OFF; NPU remains unqualified WIP.
See [README](README.md), [q8 reproduction](docs/q8-kv-cache/reproduce.md) and
[testing](docs/TESTING.md): CPU passes and a short smoke are not model-quality,
performance or public-release qualification.

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
