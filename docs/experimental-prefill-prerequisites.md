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

`serial-shadow` budgets a conservative verifier-row upper bound for the frozen
cohort and prepares a legal positive single-pass prompt cut at K minus that bound.
It executes prompt and legacy decoders separately. The pool prepares
preservation and freezes the actual cut; changed range/frontier rejects before
execution.
`PreparePrefill` / `GrowAhead` is only a non-blocking memory-growth hint, NOT a
failable memory reservation or admission guarantee. Serial SC executes the
prompt before its decoders; future mixed work requires a real reservation/admission
contract before decoder mutation and cannot treat this hint as that proof.
A pending capture is deferred, not synchronously waited on while runnable
decoders can progress. Unsupported planning is counted and falls back visibly, not
relabelled as a valid shadow.

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
attached where available. AD creates a fresh association when its actual ready
roster freezes after the preceding prompt cut; it does not reuse that cut's
pre-cut count or claim SC-style same-cut pairing. SC retains its prepared-cut
association through the frozen drain. A backend MTP batch reduced to a model
singleton is not retroactively a generic scalar call. Numerical starting-state
proof remains null; a session pointer is not a lineage/checkpoint certificate. Host spans can
overlap each other/GPU work. Critical-path removability, GPU busy time, U/L/H and
saved DRAM remain null/inconclusive rather than inferred from span sums/overlap.
Lost/incomplete diagnostics are missing evidence, never a zero-cost/removal GO.

The compile-time OFF path excludes the new diagnostic includes, state, checks
and calls. Actual matched native object comparison, not a header/source probe,
qualifies that boundary only for its exact source/toolchain/flags. The earlier
8c9b7b97-versus-ba7c7119 preflight established six identical OFF objects; that
receipt does NOT transfer across the q8-v2 rebase or these follow-up edits.
No GPU execution is part of CPU prerequisite acceptance.

## Checks and limits

Focused CPU targets: `prefill_prerequisites_test`, `pd_profile_test`,
`component_text_cache_test`, `text_model_runner_test`,
`text_generation_scheduler_test`. Enable the relevant build options and
assertions. Strict request-contract/offline E2E and coordinated source/build
receipts live in the private project evidence, not this product tree.

Design acceptance: private project `gufo-agent-concurrency` commit `50089e0`,
`docs/pd-interference-phase3-design-review-r2.md` (design/spec only, with the
explicit accepted E1 correction). Qualification is recorded by immutable
revision, not inferred from the presence of these flags:

- 8c9b7b97 versus ba7c7119 passed owner r3: three links, 5/5 focused CPU tests,
  formatting and six strict OFF-object matches.
- ee1defc1 versus 8542bb22 passed the same matched gates in owner r4; this is
  the q8-v2-based baseline, not qualification of the follow-up fix.
- Full Opus CODE review of 8c9b7b97 returned BLOCK for missing drain trace scopes;
  default-OFF safety, scheduling behavior and E1 sampler/routes were upheld.
- This follow-up adds the trace repair/regression and bounded corrections.
  Landing requires the NEW commit's matched preflight against 8542bb22 plus the
  authorized narrow fix-diff Opus PASS. Consult the private evidence ledger for
  the accepted commit and verdict; the historical BLOCK is not rewritten.

Native quality, performance, profiler/runtime qualification, immutable execution
packet and explicit host/time release remain separate. Owner integration policy
retains correct, safe partial experiments on fork main, default OFF. The
coordinator performs landing under the maintainer lease; this branch worker does not push fork main.
Integration is not release, deployment, activation, native quality, GPU execution
or a mixed-step feasibility certificate. Keep experimental and inconclusive
labels until their separate gates are actually met.
