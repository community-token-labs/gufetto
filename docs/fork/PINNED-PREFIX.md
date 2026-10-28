# Registered pinned prefixes (experimental, default off)

## Operator contract

```sh
gufo serve llm --model /models/target.gguf \
  --pin-prefix-file /private/agent-system-tools.rendered.txt \
  --pin-prefix-file /private/reviewer-system-tools.rendered.txt \
  --pin-prefix-dir /private/gufo-pins \
  --pin-prefix-bytes 1073741824
```

The file is **already rendered text**, including template framing, not a JSON
chat request and not an unrendered system message. It must contain 1..16 MiB,
tokenize nonempty, and leave execution context for the changing tail.
Use exactly the serving model's template and tool ordering. A file ending at
the rendered system/tools boundary is the intended registration.

Repeat `--pin-prefix-file` for **1..64 explicit registrations** in one owned
directory. `--pin-prefix-bytes` is the **total managed disk byte bound**, not a
per-file allowance. A single file retains the original behavior and on-disk
layout. Duplicate exact token registrations are rejected, even through different
paths or equivalent rendered text. All files are validated/tokenized before
any old registration is retired. Order and file names do not affect identity.

No unauthenticated request can register arbitrary private prompts. Automatic
learning remains ordinary evictable cache behavior: **automatic pin promotion,
a live administrative unpin API and image-backed registration are not
implemented**. This is explicit registration, not another learning heuristic.

The exact rendered token vector is the authority. Equal text or a string
boundary does not prove matching tokens: merges and content-special-token
escaping can differ. In particular, literal control-token spellings inside
client text are protected by the chat renderer, whereas this rendered-text
file is tokenized as framing. Such a mismatch is a safe miss, not permission
to restore a merely similar prefix. Prefer a clean system/tools fixture and
verify the first request's actual selected token depth.

### RAM cache off versus automatic sizing

`--cache-ram-bytes 0` keeps automatic sizing and normal pin creation and RAM
warmth. `--cache-ram-bytes off` keeps no RAM checkpoint, including registered
pins, and never demand-warms a lazy disk pin. Existing compatible registrations
can still be read with `--pin-prefix-lazy`; live-session reuse also remains.
Without ordinary `--cache-disk`, these restores need no cache backing.
Ordinary disk captures still use upstream's bounded 512 MiB backing when off.

Mandatory RAM-warm startup with off is rejected explicitly. Creating or
replacing an incompatible pin also requires RAM caching to be enabled, even
in lazy mode: startup checks the complete requested set before retiring any
old registration. Enable AUTO or an explicit RAM budget to create the pins,
then restart with off and lazy mode to use them without RAM retention.

## Identity and compatibility

The registration key is the complete exact token vector, empty supplemental
input identity (text-only registration), and component compatibility identity.
Lookup compares all tokens and the request's input identity at the checkpoint
boundary, including its image-prefix identity rules.

With pins enabled, component compatibility additionally binds:

- Full target GGUF artifact hashes (all shards), tokenizer and compiled
  template identity, snapshot payload version and component/private layout.
- Enabled MTP sidecar hash and settings; context/position/YaRN policy;
  model prefill capacity and decode concurrency, inherited from the adapter.
- SHA-256 of the **actual executable** and loaded absolute shared-library
  objects, kernel release/architecture, relevant GUFO/HIP/HSA/ROCR/ROCBLAS/AMD
  environment values. Trace variables are excluded because they do not change
  numerics. The registration prefill policy is versioned (4096-token grants).

The hash input is not logged. A changed build/runtime/configuration cannot
restore an old pin just because its snapshot ABI happens to be unchanged.
This deliberately also strengthens ordinary component-disk identity when pins
are enabled. Pins-off identity/selection remains the preexisting path.

Preserve-thinking and tool policy affect request rendering. Exact rendered
tokens remain mandatory; a flag that only rewrites history *after* an identical
system prefix does not change the state of that prefix. File/model/context
changes explicitly replace the affected registration. At startup the supplied
file set is authoritative: omitted registrations and incompatible identities
are retired, not kept as hidden pins. Pins remain default OFF; supplying no
pin options neither opens nor changes the registry.

## Lifecycle, persistence and endurance

`registered → exact captured boundary → synchronous durable publication →
materialized RAM checkpoint → ready`.

A new registration temporarily uses a normal execution slot before readiness:
prefill stops exactly at its registered boundary, then the existing component
adapter captures all rows/private state, including GDN, convolution, logits,
indexer/PLE and MTP frontiers. It does not trim a later recurrent state.
A compatible restart reads/checksums/restores those bytes through the existing
streaming adapter and materializes immutable host backing; it does **no prefix
prefill and no new pin publication**. The temporary slot is then released,
not permanently occupied by the pin.

The dedicated component `DiskStore` already provides owned-directory locking,
bounded transfers, payload CRC-64, length/geometry checks, fsync ordering and
atomic manifest rename. Pin files are never given to `DiskCatalog` or its LRU.
The existing `v2/manifests`, `v2/chunks`, `v2/private` and `v2/tmp` layout is
unchanged: **one manifest per registration in one `DiskStore`** is the registry.
There is no new registry file, per-pin subdirectory, lock or independent budget.
A legacy single-manifest directory opens without format migration; when its
strong identity matches, adding registrations preserves it without rewriting.
A changed executable/configuration still requires identity replacement.
Removing all but one configured registration returns to the single-manifest
layout. An older single-pin binary
cannot open a multi-manifest directory as a multi-pin registry (and a changed
executable still rejects compatibility).

The configured byte bound counts the sum of managed payload, manifest,
temporary and orphan files across every registration. Publication uses the
store's aggregate admission; a later registration cannot spend the entire
budget again or evict an earlier pin. A compatible set above a reduced budget
fails startup, eagerly or lazily, rather than being evicted. New registrations
publish serially, not as an all-or-nothing batch: an interrupted/refused startup
may leave a valid subset. A retry preserves that subset and creates only missing
identities. Changing the configured set/identity explicitly retires unmatched
registrations. Interrupted unpublished files may be reclaimed when any new
registration needs publication; compatible-only restarts do not rewrite or
silently remove over-budget files. Corrupt/truncated manifests or
payloads **fail required warm startup**: repair/remove that owned directory
explicitly. They are not silently re-prefilled and rewritten on every restart.

Each pin is written once per accepted identity, not on a request, cache hit,
normal RAM eviction or compatible restart. The existing ordinary disk tier is
independent: enabling `--cache-disk` still allows its existing descendant
checkpoint writes. Cross-directory dependencies are not spliced. For minimal
SSD wear, use pins with ordinary disk caching off; this change does not
claim to solve ordinary conversation-checkpoint write amplification or qualify
220-TBW endurance.

## RAM admission, lazy mode and lookup

Every pin's materialized payload, allocation rounding and metadata, including
the registry vector, use the **same resource ledger/backing pool** as ordinary
checkpoints; no per-registration unbounded host pool is allocated. RAM roots
are indexed directly, outside `RetentionPolicy`, so neither byte nor 128-record
pressure can remove them. Eager readiness requires **all roots together** to fit.
Other captures may be refused when the permanently charged roots leave too
little backing. Configured RAM is not usable payload: Flash-Next's backing
plan is approximately `(cap − 4 MiB) × 0.9` before page metadata/fragmentation.

Each enabled disk store requires its own fixed 1-MiB streaming staging buffer;
the entire pin registry uses 1 MiB regardless of count, pins plus ordinary disk
use 2 MiB. Both are charged against host headroom before backing commits.
Startup logs report
`schema=pinned-prefix-v1 event=ready` per registration, token depth,
`ram_payload_bytes` and aggregate managed disk bytes. Materialized registrations
also report restored/new status and startup duration. Existing global
component ledger/category metrics include every pin.

By default, failure to hydrate any required pin prevents readiness. With
`--pin-prefix-lazy`, compatible existing pins are indexed without loading their
payloads. The first matching request for **each** registration restores from
SSD and independently attempts permanent RAM warming. If the shared bounded
backing pool cannot admit it (including because other roots are permanent),
that request still uses its validated disk restore; subsequent requests may
restore from disk. No failed RAM admission rewrites or evicts any durable pin
or another already-warmed root.
**Initial creation still needs transient RAM capture**; lazy mode allows a
previously created pin to be served with a smaller RAM budget, not disk-only
initial capture. This limitation is deliberate and explicit.

Pins enter the existing exact prefix index and stable-fallback gate. The
**longest exact compatible token prefix** wins, independent of registration
order, including nested and disjoint registrations. A deeper compatible
live/RAM/ordinary-disk checkpoint still wins; equal-depth live beats RAM, which
beats disk. Otherwise the pin restores into the selected session
and only the unmatched tail prefills. `cache_prompt: false` bypasses pin reuse
as well as the ordinary tiers. Corruption discovered during lazy demand restore
invalidates the destination and affected pin index entries, then falls back
safely to a shorter compatible root or cold prefill. Unrelated pins remain
indexed; partially restored state never executes.

## Verification and qualification

`component_text_cache_test` adds deterministic recurrent-state fixtures for:

- New creation, eager restart without any prefix prefill, lazy demand warming.
- Greedy cold equality and seeded replay, pins with ordinary disk off/on.
- 140 unrelated captures/record/backing pressure without pin eviction.
- Input/image identity rejection, cache bypass, numerics-identity and token
  replacement, reduced disk budget refusing instead of evicting, ledger charge.
- Payload corruption and truncation refusing required warm readiness.
- File size/mtime inventory unchanged through compatible restarts and requests.
- Nested/disjoint longest-prefix selection, greedy and seeded cold equality,
  order/path-independent multi-pin restarts, single→multi→single layout.
- Per-registration lazy warming/refusal, aggregate RAM residency and disk limits
  (ordinary disk off/on, eager/lazy, temporary/orphan bytes), without LRU eviction.
- Duplicate/invalid/count-bound registration rejection before retirement,
  one-token replacement, numerics rejection of the entire set, and lazy
  corruption fallback preserving unrelated registrations.

Run the deterministic CPU fixture without a model or device:

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --target component_text_cache_test --parallel 2
ctest --preset cpu-test -R '^component_text_cache_test$' --output-on-failure
```

Registered pins currently require Linux's strict executable/runtime fingerprint
lane; unsupported hosts are designed to refuse this optional feature rather
than accept weaker identity. Native Windows builds/execution remain unverified.

`tests/functional/pinned_prefix.py` probes an **already running** server with
the same rendered file, checks cold greedy equality, seeded RAM replay, reused
depth/prefill accounting, and writes restart-replay evidence. After an
independently managed compatible restart, rerun with `--restore FIRST.json`.
It performs no service actions. Repeat both file/depth options in matching
order to probe multiple registrations, including nested prefixes. Multi-pin
reports use `gufo-pinned-prefix-probe-v2`; single-pin reports retain v1 and old
single-pin restart reports remain readable. Restart identity matching is
order-independent. Example (exact depths come from startup logs):

```sh
python3 tests/functional/pinned_prefix.py --prefix-file prefix.txt \
  --prefix-tokens 10000 --prefix-file reviewer.txt \
  --prefix-tokens 12000 --output first.json
# Authorized owner restarts the same binary, model and pin directory.
python3 tests/functional/pinned_prefix.py --prefix-file prefix.txt \
  --prefix-tokens 10000 --prefix-file reviewer.txt \
  --prefix-tokens 12000 --restore first.json --output restarted.json
```

Use `--exact-prefix` with fresh ordinary cache/slots on both runs to require
selection of precisely each supplied registration depth (not a deeper ordinary
checkpoint). The functional probe alone does not measure startup prefill or
prove persistence without file inventories and independently managed restarts.
`python3 tests/functional/test_pinned_prefix.py` checks only the probe's offline
multi-report/accounting contracts; it is not model qualification.

## Qualification status

The v0.23 split-fixed multi-registration candidate has bounded historical
qualification for four real registrations on one host. That result does not
qualify this rebased build, every template, lazy mode or arbitrary workloads.
A changed build/runtime requires recapture and requalification. The original
single-pin production failure and later candidate-024 failure remain recorded.
See [the qualification ledger](../STRIX-QUALIFICATION.md) for immutable evidence,
new-build gaps and the inherited file-verification risk. Registration files and
model/cache backing files must not be rewritten in place while serving.
