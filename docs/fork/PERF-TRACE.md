# Opt-in serving performance trace

This is diagnostic instrumentation, not a scheduling or numerical change. It
measures the serialized prefill/decode boundary for Qwen3.8 Flash-Next serving.
No prompt, generated text, token IDs, logits, client names or cache keys are
recorded. Expert IDs are represented only by a routing-set bitset.

## Enable

```sh
GUFO_PERF_TRACE="$HOME/.cache/gufo-perf.jsonl" gufo serve <normal serving options>
# Separate, short routing study; do not use these timings as a latency baseline:
GUFO_PERF_TRACE="$HOME/.cache/gufo-routes.jsonl" GUFO_PERF_TRACE_MOE=1 \
  gufo serve <normal serving options>
```

- `GUFO_PERF_TRACE`: append destination, read once per process. Unset/empty is
  disabled. An open failure prints a fixed warning and disables instrumentation.
  Parent directories must already exist. Prefer a local filesystem.
- `GUFO_PERF_TRACE_MOE=N`: positive integer, sample every Nth **target forward**
  on the scheduler thread (N, 2N, ...). Unset, zero, invalid or overflowing values
  disable routing-set sampling, not phase/RouteHints timings. This variable alone
  enables nothing. MTP predictor forwards do not advance the sample counter.
  With sparse sampling a periodic workload can alias the sample period; use
  N=1 for a short expert-overlap study so adjacent prompt/decode units both sample.

Disabled hooks take only cheap conditionals: no clocks, record allocations,
writer locking, event creation/recording, readbacks or added synchronization.
The enabled writer buffers a complete object and appends/flushes it under one
process-local mutex per work unit. Serialization/flush occurs **after** `end_ns`;
its host cost therefore appears as a gap before the next unit. There is no writer
thread or unbounded background queue. Use one trace destination per process;
the mutex does not coordinate separate processes. The file grows until stopped.
Do not rotate it by renaming while serving (the open descriptor is retained).

## Schema 1

One object per completed/returned scheduler work unit, one object per line.
Integers are emitted directly as decimal, without conversion to floating point.
All host times are `steady_clock` nanoseconds since its implementation-defined
monotonic epoch, not Unix time. Do not combine these with client wall clocks.
A device hang or process termination may leave an unfinished unit with no record;
completed units have been flushed. Do not infer successful inference merely from
an `end_ns`: the scope also closes on handled failures/cancellation.

| Field | Meaning |
| --- | --- |
| `schema` | Integer `1`. |
| `run_id`, `pid`, `thread_id` | Process-start trace monotonic timestamp, OS PID and process-local scheduler thread number. Partition all analyses by this tuple; counters/slots can repeat across runs or workers. |
| `start_ns`, `end_ns` | Inclusive host scope boundaries, before serialization. Sorted units and their gaps reconstruct the scheduler timeline. |
| `kind` | `prefill`, `decode_batch`, `decode_single`, `capture`, `admit`, or `idle-wait`. Decode kind is the scheduler-selected physical cohort, not a guarantee all selected lanes survive preparation. |
| `queue.prefilling_waiting` | Runnable prefill deque size immediately before selection (includes the selected prefiller). |
| `queue.decoders_due` | Number with `decode_due` set before selection; not all resident decoders necessarily have that flag set. |
| `queue.capturing`, `queue.waiting` | Capture-parked and cold shared-prefix/publication-waiting counts. These are not incoming HTTP queue lengths. Admission samples these after popping the arrival. |
| `sessions[]` | Selected/reserved `{slot, request_id, context_depth, rows, accepted_tokens}`. Numeric slot IDs can be reused; use request ID too. Admission depth is zero before restoring state. Capture depth is the retained prompt frontier. Decode depth is the logical committed context at dispatch, excluding a published-but-not-evaluated preview/pending advance. |
| `sessions[].rows` | On successful decode, anchor + proposed verifier rows (including prompt-lookup proposals), summed on serial fallback. Zero means no successful decode result; for non-decode units it is zero. |
| `sessions[].accepted_tokens` | Engine/pool committed selections returned by the decode, including anchor, before transport/stop-sequence publication. It is not necessarily the number of newly streamed tokens (first-token preview can have been streamed already). |
| `tokens_budget` | Scheduler prefill budget, before checkpoint/native capacity clipping. |
| `tokens_requested` | Prefill pool budget after clipping at known frontiers, bounded by remaining prompt; backend/native capacity can still clip further. |
| `tokens_processed` | Actual successful prefill consumption; zero on early exit/failure. |
| `through` | Pool chose the in-pass `PrefillThrough` route. |
| `checkpoint` | Engine requested a native in-pass checkpoint within `Feed`. Not a claim that an async capture has finished, and not a census of ordinary boundary snapshots. |
| `forwards[]` | Actual target-forward attempts: `{forward, prefill, sessions:[{context_depth,rows}]}`. Depths/row widths come directly from native session state and input spans. Ordered packed lanes may be a subset of the scheduler cohort after cancellation/fallback. These native shapes are authoritative for timing aggregation; they intentionally contain no native pointers or token IDs. |
| `phases[]` | `{name,start_ns,end_ns}` host intervals. Repeated names are separate calls. Some intervals nest: do not sum all phases into one total. |
| `gpu[]` | `{forward,host_start_ns,host_end_ns,stream_elapsed_ns}` for completed target forwards. Two timing-enabled HIP events, allocated once per executor, surround compute-stream target work outside graph capture. The end event is queued before the existing stream synchronization; querying elapsed time adds no extra synchronization. Missing entries mean unsuccessful/unavailable measurements, not zero GPU work. |
| `route_waits[]` | `{forward,layer,count,ns}` summed host wait at the existing `RouteHints` expert-count event synchronization. Present without MoE sampling. Layers are zero-based trunk layers. This measures the **unhidden host wait**, not the copy's device duration or the whole RouteHints tile-map construction. Predictor waits are excluded. |
| `moe[]` | Sampled `{forward,layer,rows,group,experts}`; group is `prompt` or `decoder`. `experts` is exactly 128 lowercase hex digits, a 512-bit integer: bit e means expert e was touched. Expert 0 is the low bit/rightmost digit; expert 511 is the high bit/leftmost digit. Decoder set is the union of all packed decoder rows, including speculative rows ultimately rejected. Predictor MoE is excluded. |

Fields irrelevant to a kind remain zero/false/empty. `forward` is a per-worker
monotonic target counter, scoped by the run/thread tuple. A pool step can contain
multiple native forwards, so join layer/wait/GPU entries using that counter.
For a shortened final prefill layer, `moe[].rows` describes only the rows actually
routed by the existing final-layer optimization, not imaginary skipped rows.

### Host phase names and owners

- `draft_catchup`, `draft_catchup_batch`: `Session::DraftCatchUp{,Batch}`. Scalar
  catch-up can also compute the first proposal head when requested by the caller.
- `mtp_proposal_round`: scalar `PrepareDecode` proposal loop and
  `DecodeBatchImpl` predictor rounds. A final no-op batch-loop check can have a
  short interval. Setup outside these scopes remains visible in work-unit wall.
- `target_forward_batch`: `Executor::ForwardBatch`, including native preparation,
  target trunk/head and final stream join.
- `target_forward`, `prefill_trunk_forward`: `Executor::Forward`, around the
  target submission/graph execution and existing final join. Earlier input/
  checkpoint/PLE preparation remains in scheduler-unit wall time.
- `finish_rollback`: `Session::FinishDecode`; nested in
  `finish_rollback_batch`, which also includes per-session verifier-logit selection.
- `prefill_mtp_kv_catchup`: post-trunk MTP KV-only work in `Executor::Forward`.

`capture` units measure **poll/completion handling**, not the lifetime or GPU
busy time of asynchronous snapshot streams. `admit` measures acquisition/cache
restore and admission bookkeeping, after removal from the incoming queue.
`idle-wait` includes timed probe handling when the scheduler has no runnable
prefill/decoder. Other bookkeeping is visible as inter-unit host gaps.

## Timing and perturbation limits

A HIP event pair measures **stream elapsed**, including host submission bubbles,
PLE waits, graph warmup/capture overhead and sampled routing synchronization.
It is not a sum of kernel execution times. MTP and independent snapshot streams
are not GPU-timed. Consequently **true GPU-busy versus GPU-idle fractions cannot
be recovered from these records**. The analyzer reports those as unavailable,
and separately reports measured target-stream elapsed / scheduler wall span and
host inter-unit gaps. Do not label either one GPU utilization/idle. Use a kernel
profiler if that distinction is the deciding evidence.

In ordinary prefill matrix routes, expert sets reuse the counts already downloaded
and synchronized by RouteHints. Sampled vector routes and `ForwardBatch` copy
only the router top-k IDs into a single reusable pinned buffer, then join the
compute stream before reading it. A sampled scalar forward runs eagerly, even if
its shape normally replays a graph; unsampled calls retain the original graph
path. No math kernels, quantization, routing, RNG, rollback or scheduling policy
are changed. Nevertheless sampling adds substantial latency and eager/graph
behavior still needs same-build numerical/replay validation on the GPU. Measure
normal phase timings with MoE sampling off and expert sets in a separate run.
Failed optional HIP-event creation/readback omits the measurement; it does not
fabricate an all-zero routing set. Failed pinned allocation omits vector samples.

## Analysis and qualification

The schema is usable by any JSON-lines reader. The separate evidence project
maintains `bench/perf_trace_report.py`; it is not a Gufo runtime dependency.
Partition by run/thread/forward, distinguish selected scheduler cohorts from
successful native forwards, and keep incomplete/failed work visible. Expert
set overlap is observational, not predicted weight reuse or concurrency.

CPU tests cover disabled collection, scope restoration, integer/bitset
serialization and concurrent whole-line append. The integrated GPU event path,
sampled graph/eager numerical replay and enabled/disabled overhead need fresh
GPU qualification. Instrumented timings must not be presented as uninstrumented
production measurements. See [the qualification ledger](../STRIX-QUALIFICATION.md).
