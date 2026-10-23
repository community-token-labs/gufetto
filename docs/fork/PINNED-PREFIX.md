# Registered pinned prefixes (experimental, default off)

## Operator contract

```sh
gufo serve llm --model /models/target.gguf \
  --pin-prefix-file /private/agent-system-tools.rendered.txt \
  --pin-prefix-dir /private/gufo-pins \
  --pin-prefix-bytes 1073741824
```

The file is **already rendered text**, including template framing, not a JSON
chat request and not an unrendered system message. It must contain 1..16 MiB,
tokenize nonempty, and leave execution context for the changing tail.
Use exactly the serving model's template and tool ordering. A file ending at
the rendered system/tools boundary is the intended registration.

One explicit operator registration per directory is the hard count bound.
No unauthenticated request can register arbitrary private prompts. Automatic
learning remains ordinary evictable cache behavior: **automatic pin promotion,
multiple registrations, administrative unpin and image-backed registration are
not implemented** in this first slice. This is the explicit-registration
alternative, not a second prefix learning heuristic.

The exact rendered token vector is the authority. Equal text or a string
boundary does not prove matching tokens: merges and content-special-token
escaping can differ. In particular, literal control-token spellings inside
client text are protected by the chat renderer, whereas this rendered-text
file is tokenized as framing. Such a mismatch is a safe miss, not permission
to restore a merely similar prefix. Prefer a clean system/tools fixture and
verify the first request's actual selected token depth.

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
changes explicitly replace this directory's one registration, not another pin.

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
There is at most one active manifest. The configured byte bound counts managed
payload/manifest/temporary/orphan files. A compatible pin above a reduced
budget fails startup rather than being evicted; changing identity explicitly
retires the old registration. Interrupted unpublished files may be reclaimed
before the first successful publication. Corrupt/truncated manifests or
payloads **fail required warm startup**: repair/remove that owned directory
explicitly. They are not silently re-prefilled and rewritten on every restart.

The pin is written once per accepted identity, not on a request, cache hit,
normal RAM eviction or compatible restart. The existing ordinary disk tier is
independent: enabling `--cache-disk` still allows its existing descendant
checkpoint writes. Cross-directory dependencies are not spliced. For minimal
P310 QLC wear, use pins with ordinary disk caching off; this change does not
claim to solve ordinary conversation-checkpoint write amplification or qualify
220-TBW endurance.

## RAM admission, lazy mode and lookup

The pin's materialized payload, allocation rounding and metadata all use the
**same resource ledger/backing pool** as ordinary checkpoints; no second
unbounded host buffer is allocated. The RAM root is indexed directly, outside
`RetentionPolicy`, so neither byte nor 128-record pressure can remove it.
Other captures may be refused when the permanently charged root leaves too
little backing. Configured RAM is not usable payload: Flash-Next's backing
plan is approximately `(cap − 4 MiB) × 0.9` before page metadata/fragmentation.

Each enabled disk store requires its own fixed 1-MiB streaming staging buffer;
pins alone use 1 MiB, pins plus ordinary disk use 2 MiB. Both are charged
against host headroom before backing commits. Startup logs report
`schema=pinned-prefix-v1 event=ready`, token depth, `ram_payload_bytes`,
managed disk bytes, restored/new status and startup duration. Existing global
component ledger/category metrics include the pin.

By default, failure to hydrate the required pin prevents readiness. With
`--pin-prefix-lazy`, a compatible existing pin is indexed without loading its
payload. The first matching request restores from SSD and attempts permanent
RAM warming. If the bounded backing pool cannot admit it, that request still
uses its validated disk restore; subsequent requests may restore from disk.
No failed RAM admission rewrites or evicts the durable pin.
**Initial creation still needs transient RAM capture**; lazy mode allows a
previously created pin to be served with a smaller RAM budget, not disk-only
initial capture. This limitation is deliberate and explicit.

Pins enter the existing exact prefix index and stable-fallback gate. A deeper
compatible live/RAM/ordinary-disk checkpoint still wins; equal-depth live beats
RAM, which beats disk. Otherwise the pin restores into the selected session
and only the unmatched tail prefills. `cache_prompt: false` bypasses pin reuse
as well as the ordinary tiers. Corruption discovered during lazy demand restore
invalidates the destination and pin index entry, then falls back safely;
it never executes partially restored state.

## Verification and qualification

`component_text_cache_test` adds deterministic recurrent-state fixtures for:

- New creation, eager restart without any prefix prefill, lazy demand warming.
- Greedy cold equality and seeded replay, pins with ordinary disk off/on.
- 140 unrelated captures/record/backing pressure without pin eviction.
- Input/image identity rejection, cache bypass, numerics-identity and token
  replacement, reduced disk budget refusing instead of evicting, ledger charge.
- Payload corruption and truncation refusing required warm readiness.
- File size/mtime inventory unchanged through compatible restarts and requests.

Run on the authorized test host, with no model loaded while compiling:

```sh
GAC_HOST=nimo-2 tools/nimo-build.sh gpu-test component_text_cache_test
# Under that host's operation ownership/preflight:
build/gpu-test/component_text_cache_test
```

`tests/functional/pinned_prefix.py` probes an **already running** server with
the same rendered file, checks cold greedy equality, seeded RAM replay, reused
depth/prefill accounting, and writes restart-replay evidence. After an
independently managed compatible restart, rerun with `--restore FIRST.json`.
It performs no service actions. Example (exact depth comes from startup logs):

```sh
python3 tests/functional/pinned_prefix.py --prefix-file prefix.txt \
  --prefix-tokens 10000 --output first.json
# Authorized owner restarts the same binary, model and pin directory.
python3 tests/functional/pinned_prefix.py --prefix-file prefix.txt \
  --prefix-tokens 10000 --restore first.json --output restarted.json
```

A deterministic CPU fixture is not qualification of Flash-Next floating-point
split invariance or sampled MTP behavior. Sparse-attention grouping/pass shapes
can affect cold comparisons. Real greedy/cached/seeded replay, exact selected
depth, actual draft counts and raw timings must be retained before claiming
the model-level exactness/performance gate passes. Hardware runs and
new-chat A/B results live in the concurrency project's `docs/pinned-prefix.md`.
