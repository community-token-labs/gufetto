# Experimental Strix Halo hybrid projections

Implementation candidate on `feature/strix-npu-hybrid`, based on nightly
`f2ae3424`. **Default OFF; not deployment or quality qualification.** Native
compilation and model measurements are pending. The owner target is a working
hybrid by October 27 17:00Z; 5–10% prefill gain is a goal, not an acceptance gate.

## Enable explicitly

Build with `-DGUFO_ENABLE_NPU=ON` and XRT 2.21 headers/libraries, selected with
`GUFO_XRT_ROOT` or `GUFO_XRT_INCLUDE_DIR`/`GUFO_XRT_COREUTIL_LIBRARY`.
OFF builds do not look up or link XRT. Runtime requires both
`GUFO_STRIX_NPU=1` and `GUFO_NPU_DIR` pointing at the donor's fixed
`final_4096x2560x1024_128x64x128.xclbin` and matching
`insts_4096x2560x1024_128x64x128.txt`. A directory alone does not enable it.

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
  at least 8192 rows. The current candidate stages complete SSM/QKV
  projections in two 4096-row banks and decodes to the original F32 GPU
  consumers. **Donor-style layer-interleaved dual-residual prefetch is not
  implemented yet.** Do not claim its published speedup for this version.
- Ordinary 4096-token scheduling remains unchanged. A server scheduling cap
  of 4096 cannot enter the optional 8192-row pair; slice/shared-expert work
  can still run on eligible 4096-row chunks. No global-chunk shrink or
  enlargement is required.
- Model-owned weights and one executor scratch set allocated in the runtime
  are reused across sessions. Whole weights share slice IDs instead of duplicating
  their BOs. GPU and NPU backing compete for the same UMA capacity; actual
  peak/high-water stability still needs measurement.
- Submit/Wait order work on the HIP stream; synchronization and Healthy()
  are checked before accepting a batch. Failed selected NPU work is not
  silently retried on GPU. Uncertain runtime resources are quarantined;
  discard the failed request and restart the process before further NPU use.
  A model reload cannot clear process-wide quarantine.
  Stream callbacks are drained before executor scratch/stream destruction.

## First qualification

Request a coordinator build slot for OFF/ON builds and focused CPU checks,
then a separate guarded native smoke and same-build flag-OFF/ON pp8192/16384
comparison. Diagnostic synchronous tracing is separate from trace-OFF timing;
native wait statistics and actual completed commands establish offload, not
successful device discovery. Record crashes, kernel messages, missing evidence,
partial successes, memory costs and restoration. Quality and concurrent decode
retention remain separate follow-ups; no public publication or main/stable
promotion is authorized by this candidate.
