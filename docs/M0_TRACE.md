# M0: diagnostic trace counters

Extends existing `GUFO_PERF_TRACE=<jsonl-path>` (unset/empty remains OFF).
No new flags, policy, scheduling, synchronization, GPU events or model changes.
Schema 1 gains optional arrays; records with no observations omit those arrays.
Only fixed labels and integer metadata are collected, never tokens or prompt text.

## Physical head operations: `mtp_heads`

`TraceMtpHead(cohort, logical_subset, subset, prefix, full, table)` records
the selected projection operations. Call sites supply dimensions from the
actual tensors and effective subset count, not `small_head_cycles`.

- `mode`: `scalar_subset`, `scalar_full`, `all_subset`,
  `mixed_full_gather`, or `full`.
- `cohort`: activation rows in the head call (including retained cancelled
  row layout in shared batch projections).
- `logical_subset_rows`: backend subset-eligible rows before scratch fallback.
  This is after `UseDraftSubset`, not merely requested logical policy.
- `subset_rows`: effective subset layout after scratch capacity checking.
- `fallback`: `scratch` when that check resets a nonzero subset count;
  `mixed` when full projection serves mixed subset/full rows;
  `logical_full` when no subset is selected; otherwise `none`.
  `logical_full` does not diagnose why the model/request declined a subset.
- `projections`: one entry per vocabulary projection operation, with
  `kind` (`prefix`, `table`, `full`), `vocabulary_rows`,
  `activation_rows`, and their product `logit_values`.

All-subset C4: one prefix operation of 98,304 weight rows × 4 activations,
then four table operations of 1,024 × 1. Mixed C4: one full operation of
248,320 × 4, even if three rows logically use subsets; gathers do not project
weights. Scratch fallback likewise projects full × cohort. Scalar subset
uses prefix × 1 plus table × 1. Dimensions are not hardcoded in the hooks.

These are source-derived projection counters, not measured DRAM bytes,
cache misses, hardware wave/tile work, or counts of all auxiliary kernels.
Each entry counts matrix rows once per projection operation, not once per
activation; sum `logit_values` separately. Head cohorts are at most eight,
so the head's DenseBatch does not take its >8-row chunking path.
Scalar recording is outside Run/MtpHead graph bodies: eager execution,
first capture+launch, and cached replay each count once. KV-only/headless
catch-up is excluded. Batch recording follows successful vocabulary
projections, before selection; scalar recording follows successful head/run.
Failed/partial calls are not a complete physical-work ledger and must not
be interpreted as normal successful benchmark cycles.

## Table occupancy: `draft_tables`

After successful SyncDraftVocab, snapshots contain `context_depth`,
`used_rows = draft_specials + rare().size()`, and `capacity = 1024`.
The snapshot is occupancy, not an operation count; do not sum it as traffic.
Even a table with 17 occupied rows projects all 1,024 weight rows in a
subset operation; unused slots are masked afterwards. A mixed full/gather
operation projects zero table weight rows regardless of occupancy.
No device ID readback is added. Snapshots exist only when vocab synchronization
runs; absence does not mean zero occupancy.

## Committed cycles: `mtp_cycles`

`TraceMtpCycle(depth, proposed, accepted, lookup, committed)` runs after
successful publication in FinishDecode. Proposed and accepted exclude the
anchor; committed includes it. Lookup copies trail neural proposals:

- neural proposed = chain size − 1 − lookup proposed;
- neural accepted = min(kept size − 1, neural proposed);
- lookup accepted = kept size − 1 − neural accepted;
- committed = kept size (anchor plus accepted proposals).

AR emits zero proposals/acceptances and committed=1; an empty/stopped-before-
anchor chain emits nothing. Lookup-only chains have zero neural proposals.
These count actual final chains, not attempted head calls or rejected retry
preparations. Rejections/target stop can leave proposals unaccepted; that is
not a fabricated neural failure histogram. Deferred sampled correction is
the next cycle's anchor, not an accepted draft in this cycle.
Existing scheduler rows/accepted_tokens also include the anchor; existing
HTTP aggregate draft counts include lookup. Neither is redefined here.

## Attribution and timing limits

All arrays belong to the enclosing existing work record, whose `sessions`
contains actual scheduler slot/request IDs. This provides cohort attribution,
not per-head-row request attribution. Head subsets can narrow between rounds;
model/backend sessions do not expose scheduler IDs. Cycle/table context_depth
is real engine metadata, not a unique join key (peers can have equal depths).
Do not infer per-slot head membership, table identity, or cycle identity by
array order. Exact per-row attribution is unavailable under the no-API-change
scope; work-unit aggregate NN/lookup and physical projection totals are exact
on successful paths.

Existing phases already bracket scalar `target_forward`, batched
`target_forward_batch`, `draft_catchup(_batch)`, `mtp_proposal_round`,
and `finish_rollback(_batch)`. They are coarse host wall intervals, can nest
or include waits/transfers/capture, and are neither disjoint nor GPU-only.
No missing scalar verification bracket was found. Do not sum nested intervals
into a GPU phase partition. No decode-rounds-per-prefill=8 assumption is made.

## CPU checks and remaining gate

Existing targets: `perf_trace_test`, `perf_trace_disabled_test`.
Standalone source gates: `python3 tests/core/mtp_trace_source_test.py`.
CPU serialization tests cover modes, scratch fallback, operation vs output
rows, occupancy, anchor/neural/lookup classification, omission of empty fields,
disabled guards, and existing JSONL concurrency. Source gates constrain hook
placement; they do not execute HIP or prove model/runtime correctness.

No GPU/HIP build or model qualification was performed here. Parent independent
review must precede commit and the coordinator's one combined Stage0/M0 build
and matched Q8/Q4 ON/OFF C1–C4 agent-prompt packet. Trace-enabled timing overhead
must remain visible; compare equivalent tracing settings.
