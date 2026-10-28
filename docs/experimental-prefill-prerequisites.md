# Experimental prompt/decode measurement prerequisites

Unqualified, default OFF. This adds no mixed target/kernel and makes no speedup,
numerical-quality or native zero-cost claim. It supplies a cheaper adaptive
comparator and a **serial** same-cut study for later coordinated measurements.

Build explicitly with `-DGUFO_ENABLE_PREFILL_PREREQUISITES=ON`. Otherwise the
new CLI/options, policy state and runner API are preprocessor-excluded. Even in
that build, the runtime default is `legacy`, preserving the existing scheduling
policy. The text server accepts:

- `--experimental-prefill-policy legacy|adaptive|serial-shadow` (default legacy).
- `--experimental-prefill-budget N` (default 0/OFF). Enabled modes require
  64..4224 and at most eight sessions; legacy requires zero. Validation precedes
  model loading. Existing `--prefill-chunk` remains the unchanged legacy bound.

`adaptive` proposes N prompt tokens while eligible ready/non-capturing decoders
wait, otherwise 4096. The existing pool/model owns checkpoint clipping, legal
final tails, capture/restore and publication. A finite ready cohort drains
before another prompt cut; newcomer/capture resumes do not indefinitely reopen
it. A large quiet pass is not preempted by an arrival. Component cache stays ON
and is specifically included in the CPU fixture; no parallel fake engine or
replacement continuation cache is introduced.

`serial-shadow` reserves a conservative verifier-row upper bound for the frozen
cohort and prepares a legal positive single-pass prompt cut at K minus that bound.
It executes prompt and legacy decoders separately. The pool uses its real
preservation/admission hook before decoder mutation and freezes the actual cut;
changed range/frontier rejects before execution. A pending capture is deferred,
not synchronously waited on while runnable decoders can progress. Unsupported
planning is counted and falls back visibly, not relabelled as a valid shadow.

**All production component shadow cycles remain mixed-ineligible/unsupported.**
The Qwen3.8 Flash Next runner always uses the component cache; its existing
component capture/restore remains literal serial behavior. No component-row
commit or mixed component contract is implemented. Usage reports explicit
`serial_shadow_mixed_eligible=false`, step/unsupported/decline counters and upper
row totals. These costs cannot certify production mixed feasibility. Decoder
sampling and policy observers remain the literal legacy operation on this
unsupported path. Generic remaining-one/MTP-off routes still directly sample
the pool sampler; MTP backend batches retain their route after model filtering.

## Historical Phase 2 result: retain relief, reject fixed smaller grants

The registered 18-run replay sweep used immutable candidate023 (`1b781e89`),
one due-decoder cohort, not this unmeasured prerequisite build. Original gates
required both candidate repeats to clear the descriptive control floor AND the
median to reach the control mean. Both fixed smaller grants are
`REJECT_THROUGHPUT` for every S; no post-data tolerance change or cross-S pooling.

| Separate replay group | 2048 median rate vs control | 1024 median rate vs control |
|---|---:|---:|
| S2 / nimo-1 | -1.6871% | -6.2984% |
| S3 / nimo-1 | -2.1269% | -7.4094% |
| S4 / nimo-2 | -1.8976% | -7.6564% |

The partial decode benefit is real **client-payload proxy** evidence, not native
GPU/token timing: worst-candidate versus best-control peer-overlap gap-p95 relief
was 43.2% / 41.9% for 2048 in S2 / S3, and 64.6% / 64.4% for 1024. Integrated
excess-stall relief after 0.5 s was only 6.8% / 8.5% and 19.2% / 30.3%
respectively. Pauses became more frequent and native whole-request mean selection-
publication ITL worsened. Native token p50/p95 and per-chunk GPU timing were not
recorded. Full-32K responsiveness, Pareto, numerical quality and adoption remain
unqualified; no S4 decode-relief percentile is inferred from the other groups.
Natural EOS shortfalls and the S4 1024/r1 +29-prefill/-29-cached difference are
retained, not normalized. Scalar evidence and methods are in the private
`JCFrags/gufo-agent-concurrency` repository under
`docs/pd-interference-live-r1/{s2,s3,s4}-block-r1/` and
`docs/pd-interference-live-r1/decode-side-method-r1.md`.

## Diagnostics

`-DGUFO_ENABLE_PD_PROFILING=ON` separately compiles perturbing diagnostics, OFF by
default. It uses the existing `GUFO_PERF_TRACE` sink and `GUFO_PERF_TRACE_MOE`
sampling guard. Do not use diagnostic runs for ordinary performance gates.

Trace JSONL mixes core schema 1 with explicit string schema `pd-profile-v1`;
consumers must distinguish them. Routed-F16 gate/up/down maps describe actual
compaction rows, tensor/packing/layer/expert identity, packed-address ranges,
dispatch tiles and coverage. Vector paths expose logical requested address
sets; private physical grouping/load multiplicity is explicitly missing. Expert
sets or address overlap are NOT eliminated loads/DRAM savings.

Scheduler cohort/plan/iteration and actual backend/model-filter counts are
attached where available. A backend MTP batch reduced to a model singleton is
not retroactively a generic scalar call. Numerical starting-state proof remains
null; a session pointer is not a lineage/checkpoint certificate. Host spans can
overlap each other/GPU work. Critical-path removability, GPU busy time, U/L/H and
saved DRAM remain null/inconclusive rather than inferred from span sums/overlap.
Lost/incomplete diagnostics are missing evidence, never a zero-cost/removal GO.

The compile-time OFF path contains no new diagnostic includes, state, runtime
checks or calls. Actual native object/ISA comparison with identical toolchain/
flags is still required to prove that claim; a CPU header probe or source-only
comparison cannot qualify the F16 paths. No GPU execution is part of this change's
CPU prerequisite acceptance.

## Checks and limits

Focused CPU targets: `prefill_prerequisites_test`, `pd_profile_test`,
`component_text_cache_test`, `text_model_runner_test`,
`text_generation_scheduler_test`. Enable the relevant build options and
assertions. Strict request-contract/offline E2E and coordinated source/build
receipts live in the private project evidence, not this product tree.

Design acceptance: private project `gufo-agent-concurrency` commit `50089e0`,
`docs/pd-interference-phase3-design-review-r2.md` (design/spec only, with the
explicit accepted E1 correction). Completed outgoing-code Opus review, full CPU
compile/link/format checks and actual native OFF proof are still pending at this
WIP revision. Native quality, performance, profiler/runtime qualification,
immutable execution packet and explicit host/time release remain separate.
Owner integration policy retains correct, safe partial experiments on fork
main, default OFF. Landing requires a green immutable prerequisite preflight
(three links, 5/5 focused CPU tests, formatting and six strict OFF-object matches)
and the one focused outgoing-code Opus review. The coordinator performs that
landing under the maintainer lease; this branch worker does not push fork main.
Integration is not release, deployment, activation, native quality, GPU execution
or a mixed-step feasibility certificate. Keep experimental and inconclusive
labels until their separate gates are actually met.
