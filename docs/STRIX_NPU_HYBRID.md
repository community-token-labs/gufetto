# Experimental Strix Halo hybrid projections

**Default OFF; experimental, not correct/faster or release-qualified yet.**
Frozen `99ab429f` built OFF/ON on nimo-2 in lease `d4ed1f0d2152` (focused CPU
5/5 OFF, 6/6 ON). Our runtime completed 555 Process calls and 3511 successful
reaps, with no selected kernel fault or boot change. Model prefill then failed
at layer 44, bank 1: routed gate/up counts were zero. No completed hybrid timed
row or speed gain was obtained. Production was verified restored on attempt 1;
the freshly captured prior controls were **high/auto**, not auto/auto.

This follow-up, based on nightly `5722416e`, fixes statistics synchronization,
failed stream-release ownership and NPU backing-byte reporting. It adds bounded
numerical and trace-cost diagnostics; it does **not** claim the route-count cause
is fixed. Trace-ON O_DSYNC times are diagnostic, not ordinary worker/device
performance. Soft owner target: stable/PR-ready about October 27 19:00Z, only
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
Full interleaving remains separately diagnostic until its failure is resolved.
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
