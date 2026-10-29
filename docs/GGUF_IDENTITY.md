# GGUF identity and persistent digest memos

`GgufIdentityHex` retains the `gguf-sha256-v1` identity scheme: SHA-256 over the
ordered shard sizes and each shard's full-content SHA-256. Metadata, tensor
index, payload and padding all contribute. Identical copies have identical
identities. Memory images are always hashed in full.

File digests are memoized privately in
`${XDG_CACHE_HOME:-$HOME/.cache}/gufo/gguf-sha256-memo-v2` (the platform user-cache
location on Windows). Old `gguf-sha256-v1` **memo directories** are ignored;
the identity scheme itself has not changed. The first lookup after this upgrade
therefore scans each file completely, even if an old memo exists.

## What is checked on every file lookup

The memo key combines:

- Device, inode, size, nanosecond mtime and ctime from the already-open file.
- A Linux fs-verity measured digest, including its algorithm, **if already
  enabled on that inode**. Gufo never enables it. Unsupported, disabled or
  failed measurements yield no token; Windows currently has no such token.
- A fresh SHA-256 fingerprint of the complete parsed GGUF header and tensor
  index (including alignment padding), plus deterministic payload windows.
  The payload is sampled as 256 evenly spaced 64 KiB windows, including its
  first and last blocks: at most **16 MiB selected payload bytes per shard**.
  Payloads no larger than 16 MiB are read completely. Header bytes are additional,
  not capped at 16 MiB. File reads use the open handle, not a reopened pathname.

Every shard is probed before looking up any digest and again after all digests
are resolved, including memo hits. Stat fields are also checked around each
probe. Thus a warm lookup reads the header and selected payload **twice**
(up to 32 MiB of payload per shard). A missing/malformed memo, changed stamp,
changed kernel token or changed fingerprint causes a full rehash. An observed
change during the lookup, or an incomplete fingerprint read, throws; no new
memo is published from a failed lookup. Completed memos are published by atomic
rename with the existing owner-only directory/file checks.

## Guarantees and limits

**Sampling is not proof that a mutable file is unchanged.** Without a trustworthy
kernel immutable token, a same-stat rewrite confined to unsampled bytes can
reuse a stale full digest. Deterministic windows are not an adversarial integrity
check. Small files have no unsampled payload, but concurrent writers still
prevent an atomic snapshot: writes/reverts between observations can escape,
and a full scan under a concurrent writer can observe mixed contents. Changes
after the final probe or during later inference are outside this API's checks.
Keep model artifacts stable from reader opening through their entire use.

Enabled fs-verity makes the measured inode's content immutable and verifies
reads through its Merkle tree. Its measured digest is a **change detector**, not
the GGUF identity: even this path initially computes the ordinary full SHA-256.
A read-only bind mount or ordinary permissions alone are not equivalent to
fs-verity: the inode may still be writable through another path.

The Linux UAPI headers checked for this implementation expose no
`STATX_CHANGE_COOKIE`/i_version field, including on the inspected ext4 and btrfs
hosts. An ext4 `iversion` mount option alone does not provide a usable statx
counter. `FS_IOC_GETVERSION` is inode generation, not a per-write counter, and
btrfs transaction generation is not used as one either. No counter is invented
from those values. Until a supported per-write ABI is available, the mutable
file path uses the sampled fallback.

Memos are trusted user-local performance data, not authenticated manifests.
Another process running as the same user can forge a syntactically valid memo;
the Windows owner check also currently relies on inherited ACLs. This mechanism
is not a boundary against that attacker or against storage corruption that
escapes the checks. For deliberate mutable-model replacement, stop users of the
old model, replace the artifact, and remove its digest memos (removing the v2
memo directory forces full scans) before reopening. Prefer immutable model
storage where content-proof reuse is required. Existing restored inference
caches are only as trustworthy as their model identity and their own checks.
If a model was previously rewritten in place under the stat-only memo policy,
invalidate its old inference caches too: this change cannot retroactively
validate cache entries created with an incorrect model identity.

## Regression coverage

`gguf_identity_test` freezes all fstat fields for the model inode without
sleeping or depending on filesystem timestamp resolution. It retains full-hash
and restored-mtime assertions, verifies bounded warm reads, complete large
header/first/tail detection, old memo rejection, malformed memo recomputation,
and rejected in-flight mutations on both misses and hits. A kernel-token test
double checks invalidation independently of samples; it is not an fs-verity
filesystem integration test. An explicit unsampled frozen-stat test documents
the remaining blind spot rather than claiming it is fixed.
