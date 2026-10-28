# Query-local sparse prefill

Flash-Next may split a prompt at cache frontiers, including learned boundaries.
The selected query's sparse attention must not change merely because another
query shares its prefill batch or future rows occupy the same tile.

The Strix fix gives each query's 12 heads one padded 16-row WMMA tile, uses
ascending selected-key tiles, and keeps K/V causal extent query-local. This
removes companion-dependent key compaction and future-value padding effects
before F16 probability rounding. It is a model-kernel correction, not a cache
planner workaround, minimum-tail clamp, precision reduction or changed input.
No option is needed: it is the default sparse prefill path.

This arithmetic change originally advanced snapshot payload 18 to 20, rejecting
both upstream 18 and the earlier local payload-19 compatibility fix. The
current q8-capable product uses payload **21** with explicit storage dtype;
old checkpoints must not be relabeled. Registered pins and transfer epochs also
bind actual executable/runtime identity.

The adjacent `attention_ops_test.cpp` includes failing-before frozen-input
companion, future-value and split regressions; snapshot tests check rejection.
The historical source study demonstrates exact equality across its tested
split geometries, not universal scalar-decode/prefill equality or superiority
to an FP32 teacher. Source-label quality proxies were mixed and some SDK timing
phases remained inconclusive. Fresh GPU qualification is required after rebasing
or rebuilding. See the immutable fix report in
[the historical qualification ledger](http://git.home.arpa/JCFrags/gufo/src/commit/7b360b41136a4134855659d001051e6f4c8fdc82/docs/STRIX-QUALIFICATION.md).
