# Component-cache reuse diagnostics

Each acquired component-cache request reports `component_cache.reuse_reason`
in response usage/timings and the `schema=component-cache-v1 event=acquire` log.
`reuse` still names the source actually used (`live`, `ram`, `disk`, `none`).
A partial hit can have a reason explaining a shortfall even when `reuse` is not
`none`. These fields do not change selection or relax checkpoint compatibility.

| Reason | Observation |
| --- | --- |
| `live`, `ram`, `disk` | Selected source covers all identity-qualified indexed agreement |
| `disabled` | Request disabled reuse |
| `unknown_compatibility` | No registered index for this compatibility identity |
| `no_indexed_prefix` | No identity-qualified resident agreement; not necessarily the first request |
| `no_compatible_boundary` | Indexed agreement exists but no usable exact checkpoint boundary |
| `input_identity` | Supplemental input identity rejected an indexed boundary |
| `missing_component` | A matching boundary has unavailable components |
| `stable_prefix_boundary` | No eligible immutable fallback attests the required stable prefix |
| `boundary_gap` | Successful source is shallower than indexed token agreement |
| `restore_failed` | At least one attempted candidate failed; fallback may still succeed |
| `frontier_not_ready` | A full-prompt candidate lacked a decode frontier; fallback may succeed |
| `cancelled` | Cancellation stopped candidate iteration before successful reuse |

`not_acquired` is the default for a metrics object that has not recorded an
acquisition; it is not an acquisition outcome. Disabled/cancelled outcomes take
precedence, then restore failure, unready frontier, gap, and ordinary source.
Existing numerical `selection_reason` describes the initial index selection,
not necessarily the source that ultimately restored. `selected_entry_id` is
that planned entry; `used_entry_id` is the successful entry, or zero.

`indexed_common_tokens` is captured before retiring the chosen slot's old live
frontier, including on hits. `indexed_gap_tokens` is its nonnegative difference
from selected tokens. It includes available token agreement, not proof that a
hybrid recurrent checkpoint exists at every token. Evicted state is absent,
and complete supplemental-input identity gates agreement. Neither field is an
all-history reusable-token oracle or permission to truncate recurrent state.

Reliability reports must show **client-history reusable tokens R** and
**resident indexed agreement I** separately. The **99.9% target uses R**: the
longest token-exact prefix previously processed for a compatible client history,
including matching histories across chats. Eviction or a never-captured recurrent
boundary does not reduce R; it can cause a miss. I is `indexed_common_tokens` at
lookup and is only a resident-state diagnostic. If either value is unavailable,
report unknown (or a labelled bound), not zero or a substituted denominator.
The HTTP diagnostics alone cannot establish R after historical state is lost.

Same-boundary restore, split-invariance, numerical quality and per-request timing
are separate gates. Token-prefix agreement and other engines' cached/fresh
bit-equality disclaimers do not relax any of them.

`unready_frontiers`, `restore_attempts`, `restore_failures`, and the existing
static `restore_failure_reason` separate readiness from transfer/validation
failures. Exception text and prompt/token contents are not logged. The restore
failure observability idea also appears in Moonki Kang's (namote/gufo)
`6324aa9fc432289e544a9eaedfcd866eeb8787c0`; this implementation builds on Gufo's
existing phase-coded diagnostics rather than copying exception text.

`GET /props` exposes `cache_reuse_acquisitions`, process-lifetime counters by
reason. They include acquisitions later cancelled or failed during generation,
but not requests cancelled before acquisition, exceptions before acquisition
finalization, or metadata-only lookups. Counters survive cache recreation in
the same process; restart resets them. They are not completion counts, per-model
counts, or per-request deltas. Concurrent reads need not be one atomic snapshot.

Diagnosing historical eviction, admission refusal, rendering changes, or
compaction requires joining these observations with retained lifecycle events
and exact input histories. `no_indexed_prefix` must not be relabelled cold simply
because the server no longer retains evidence of an earlier prompt.
