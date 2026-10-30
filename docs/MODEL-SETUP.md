# Read-only model setup planning

```sh
gufo diagnose --plan /path/to/models.tsv
```

This CPU-only command reads a pinned TSV, checks existing local artifacts, and
prints the input verbatim with its SHA-256. It never downloads, tunes the host,
allocates GPU memory, or writes a report/artifact. Redirect stdout yourself if
needed. `--plan` cannot be combined with `--output`, `--json`, or another
diagnostic mode. No network requests are made; source URLs are provenance, not
instructions to fetch anything.

## Manifest

Use this header (literal tabs between columns):

```text
group	status	repo	revision	file	bytes	sha256	license	gated	role	why	fit	source
```

Each row is independent, including alternative quantizations and sidecar rows.
The command does not group models by name or infer that alternatives should be
loaded together. `file` is relative to the manifest directory, not the process
working directory. Absolute paths, `..` components, backslashes, and final-file
symlinks are rejected. Keep separate manifests/directories when different
repositories use the same filename (for example, `mmproj-BF16.gguf`).

`revision` must be a 40-character lowercase hexadecimal immutable revision;
`sha256` must be 64 lowercase hexadecimal characters. `bytes` is an unsigned
integer. Watch-only rows or literal `unknown` pins are `NOT-EVALUATED`;
other malformed revision/checksum pins are input errors (exit 2). All other existing
regular files are checked for both exact size and SHA-256, even when the size
is wrong. Missing files are reported, not acquired. The prose `fit` column is
never admission evidence. Malformed headers, column counts, paths, or numbers
are errors. Unknown column names are rejected rather than silently ignoring a
misspelled budget. Optional numeric errors identify the row and column.

Append optional numeric columns to describe the scenario:

| Column | Meaning |
|---|---|
| `context` | Requested token context, positive integer |
| `host_available_bytes` | Declared available UMA RAM before loading, not total installed RAM |
| `device_available_bytes` | Declared available HIP heap before loading |
| `os_reserve_bytes` | Optional reserve, at least the server's host reserve; defaults to that reserve |
| `weights_bytes` | Declared resident target weights |
| `kv_bytes` | Declared KV at the stated context |
| `vision_bytes` | Declared projector/vision weights and execution obligations |
| `draft_bytes` | Declared MTP/draft weights and state |
| `workspace_bytes` | Declared execution scratch/arena obligations |
| `cache_bytes` | Declared retained RAM cache obligations |
| `persistence_staging_bytes` | Declared persistence staging RAM, not disk capacity |

For absent artifacts, all seven component-byte columns must be supplied, with
explicit zeroes for disabled components. They describe one complete scenario,
not just that row's file. They are labelled **declared**, never verified. Missing
or `unknown` values leave the result `UNPROVEN`. Capacity inputs always remain
declared: this command does not substitute measurements from the editing host.
Disk capacity, permissions, licensing consent, and runtime model support are
not certified by this report.

## Three states, no positive fit claim

- **DOES-NOT-FIT**: the counted lower bound (or declared components) already
  exceeds the declared capacity after reserves.
- **UNPROVEN**: the bound is within capacity, or admission evidence is missing.
  This is **not** a fit/quality/startup guarantee. The report names uncounted
  components. Use `gufo serve`'s measured startup admission and its bounded
  `verified to fit: --context N` recommendation on the target GPU host.
- **NOT-EVALUATED**: watch/unpinned rows or local artifacts without a supported
  metadata-only lower bound. This does not mean the model cannot run.

For a local `qwen35` GGUF, the existing Qwen weight loader validates the model.
The lower bound counts the mapped weight-region bytes copied by the Qwen GPU
loader and the **F16** KV rows at `context` from its existing CPU continuation
layout. It omits recurrent/private state, separate mmproj/MTP files, executor,
arena/workspace, retained cache, and persistence staging. The report names these
omissions. Other local families cannot fall back to declared values to hide an
unsupported derivation. Split GGUFs are `NOT-EVALUATED`: the reader may open
sibling shards, but only the manifest row's own file is size/hash verified; no
unverified sibling contributes to a capacity bound. A present corrupt file
cannot be overridden by a budget.

Weights prefer the device heap, with registered-host fallback
(`src/models/qwen/hip/model_loader.cpp:206–215`); KV needs device allocation
(`src/models/qwen/hip/arena.cpp:792–797`). A large BIOS VRAM carve-out must not be
mistaken for host RAM. With available host bytes H and device bytes D after the
HIP reserve, the placement lower bound fails only if KV > D, or if
weights + KV > D **and** weights > H. Declared lane-unknown extras only enter an
aggregate H + D necessary-capacity check. This optimistic combined capacity is
not proof of extra capacity on shared UMA; it can only leave results UNPROVEN.

`core::kHostExecutionReserveBytes` remains the single source of serve's 4-GiB
post-loading host execution/cache reserve (re-exported as
`server::kHostSnapshotHeadroomBytes`; used by `ObserveHostSnapshotHeadroom`).
The weight loader does **not** enforce that host reserve before registration,
so the negative weight-load proof uses raw host availability, not H minus that
reserve. The report prints the post-load reserve but does not claim it was
satisfied. Native execution/cache admission can still refuse a loading scenario
left UNPROVEN here. Device capacity deducts `hip::kDispatchReserve`, as the loader
and `ObserveExecutionCapacity` do. Overflow-safe arithmetic reuses
`FitsMemoryBudget`. This is only a necessary-capacity check, not a substitute for
the server's measured model/session admission.

Exit status: `0` means a report was generated without a negative-capacity or
artifact-verification failure, **not that a model fits**; `1` means such a failure
was found; `2` means malformed input or an incompatible command invocation.
