# Experimental Strix Halo hybrid projections

**Default OFF; experimental nightly candidate, not main/stable qualification.**
The measured **shared-expert-only** profile completed pp8K/16K and eight fresh
Session cycles in one Model process on nimo-2, lease `b732f8607982`, source
`8d518692`. Same-binary runtime OFF/ON: **1882.90 → 1843.28 tok/s** at8K
(−2.10%), **1834.00 → 1832.43 tok/s** at16K (−0.086%): no speedup established.
The short memory screen has flat post-warmup endpoints within each shape, one
24 KiB RSS step, and constant GTT peak/VRAM/fd envelope; not long-run stability.
Full-logit quality, two-host portability and concurrent decode remain unrun.

The separate **whole/two-bank profile is broken**: routed counts became zero
at layer34/bank1. First sampled NaNs appear after GDN/output projection;
preceding NPU/GPU projection samples are finite with0.87–1.05% relative L2
error. That localizes an interval, not a proved race or root-cause fix.
All stages had no selected kernel faults. Production was verified restored;
the freshly captured prior controls were **high/auto**, not auto/auto.
Earlier exact99/d4ed completed555Process/3511reaps then failed at layer44.

This landing generation rebases only the NPU feature onto frozen nightly C2
`9df6ea7f`; it preserves C2's q8 policy, guards and HIP include fix unchanged.
The earlier C1-based22e06e74 build packet is retired unconsumed: it lacked that
include fix and must not be used as native qualification.
Rebased native build/CPU qualification must be recorded separately from the
8d hardware results. Statistics synchronization, failed stream-release
ownership, QKV-only index-key policy and backing accounting are fixed in
source. Trace-ON O_DSYNC costs were93.9–94.0% of measured worker time;
1178.18 tok/s traced admission is diagnostic, not clean cold performance. Soft owner target: stable/PR-ready about October 27 19:00Z, only
after real two-host gates and separate Opus/GPT-6.1-sol reviews. A 5–10% prefill
gain remains a goal, not a prerequisite for useful experimental progress.

## Enable explicitly

Build with `-DGUFO_ENABLE_NPU=ON` and XRT 2.21 headers/libraries, selected with
`GUFO_XRT_ROOT` or `GUFO_XRT_INCLUDE_DIR`/`GUFO_XRT_COREUTIL_LIBRARY`.
OFF builds do not look up or link XRT. Runtime requires both
`GUFO_STRIX_NPU=1` and `GUFO_NPU_DIR` pointing at the donor's fixed
`final_4096x2560x1024_128x64x128.xclbin` and matching
`insts_4096x2560x1024_128x64x128.txt`. A directory alone does not enable it.

The next explicit **shared-expert-only ablation** uses `GUFO_NPU_SLICES=0`,
`GUFO_NPU_GATE_SLICES=0`, `GUFO_NPU_WHOLE=0`, `GUFO_TWO_BANK=0`, and
`GUFO_NPU_SHEXP=1`. It is a distinct declared configuration, not a silent retry
or fallback after selected work fails. Its correctness/performance are pending.
Use this full shared-only configuration, not merely `GUFO_STRIX_NPU=1`:

```sh
GUFO_STRIX_NPU=1 GUFO_NPU_DIR=/path/to/programs \
GUFO_NPU_SLICES=0 GUFO_NPU_GATE_SLICES=0 GUFO_NPU_WHOLE=0 \
GUFO_TWO_BANK=0 GUFO_NPU_SHEXP=1 GUFO_NPU_ISOLATE=1 GUFO_NPU_KEEPALIVE=0 \
gufo bench -m /path/to/model.gguf -p 8192,16384 -n 1 -r 3 --speculative off
```

The ordinary opt-in defaults can also select SSM/attention slices; those are
**not** qualified by the shared-only screen. Full interleaving remains
separately diagnostic until its failure is resolved. Persisted cache identity
does not yet separate NPU arithmetic; do not mix ON/OFF durable state. MTP,
checkpointed/cached serving, q8-KV attention and true decode contention are not
covered by these pp numbers. Never describe successful completion as full
numerical-quality acceptance.
`GUFO_NPU_PROJECTION_REFERENCE=1` with the existing bounded dump controls adds
an independent first-16-row Q8/F16 GPU projection for localization only, never
for consumption/fallback. Its narrow arithmetic is not a wide-parity oracle.
`GUFO_NPU_BENCH_MARKERS=1` logs prefill/session-release boundaries outside the
timer; disabled by default. Raw activation/logit dumps must remain private.

Experiments on the shared Nimos require coordinator host time, qualified
hostlease-r4 with production stopped, a stock-top FCLK hold and independent
restoration guard. The library does not acquire a lease or change clocks.
Never enable this flag in production merely because a build succeeded.

## Scope and ownership

- Supported Q8_0 dense matrices with K2560 use fixed 4096-row NPU groups:
  shared-expert gate/up, SSM z slices and attention query/gate slices.
  Unsupported dtype/geometry, tails and unsupported attention modes retain
  the original GPU path. Existing routed MoE, MTP, KV and checkpoint logic
  are preserved; decode weights are not requantized.
- The NPU computes block-float products, with GPU packing and consumption.
  This is hybrid inference, not NPU-only prefill or bit-identical arithmetic.
  F32 residuals remain the original path; donor F16 residual defaults are
  deliberately not imported.
- Private XRT BOs with CPU copies are the default transport. The donor's
  imported-HIP mapping failed EINVAL on our hosts; private BOs completed
  61 Process calls/364 successful reaps without observed kernel faults.
  That is a real hardware-execution win, not proof of numerical correctness
  or complete model prefill. The donor then rejected routed-MoE dispatch.
- `GUFO_TWO_BANK=1` with `GUFO_NPU_WHOLE=1` admits supported text passes of
  at least 8192 rows, processed as fixed 4096+4096 pairs. The next candidate
  submits A(next layer) before consuming B(current), and vice versa, with
  ticket-specific waits and separate persistent residual/mixer partitions.
  Full SSM/QKV outputs decode to the original F32 GPU consumers. GDN/KV
  state writes remain causal A-before-B for each layer. Runtime overlap
  and gains still require measurement; do not claim donor published speed.
  Pair admission excludes MTP-enabled sessions, images, checkpoints and
  unsupported verifier/logit states; those retain the original slice/GPU
  path. Single-buffer PLE staging is drained between prefixes; next-chunk
  PLE prefetch is conservatively suppressed during a pair.
- Ordinary 4096-token scheduling remains unchanged. A server scheduling cap
  of 4096 cannot enter the optional 8192-row pair; slice/shared-expert work
  can still run on eligible 4096-row chunks. No global-chunk shrink or
  enlargement is required.
- Model-owned weights and one executor scratch set allocated in the runtime
  are reused across sessions. Interleaving partitions existing wide scratch,
  reuses the NPU A/C sets and adds one pinned control record, not per-session
  buffers. Whole weights share slice IDs instead of duplicating their BOs. GPU and NPU backing compete for the same UMA capacity; actual
  peak/high-water stability still needs measurement.
- Submit/Wait order work on the HIP stream; synchronization and Healthy()
  are checked before accepting a batch. Failed selected NPU work is not
  silently retried on GPU. Uncertain runtime resources are quarantined;
  discard the failed request and restart the process before further NPU use.
  A model reload cannot clear process-wide quarantine.
  Stream callbacks are drained before executor scratch/stream destruction.

## First qualification

The first OFF/ON build and focused CPU slot is queued. The next snapshot
also needs the full assertions-enabled HIP-OFF CPU no-regression comparison
required for nightly admission, retaining raw known failures and skips.
Prepare a separate guarded native smoke and same-build flag-OFF/ON pp8192/16384
comparison. Diagnostic synchronous tracing is separate from trace-OFF timing;
native wait statistics and actual completed commands establish offload, not
successful device discovery. Record crashes, kernel messages, missing evidence,
partial successes, memory costs and restoration. Quality and concurrent decode
retention remain separate follow-ups; no public publication or main/stable
promotion is authorized by this candidate.
