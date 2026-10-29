# Persistent disk cache

The serving default is **off**. Use one canonical option:

- `--cache-disk auto`: use `$XDG_CACHE_HOME/gufo/continuations`, falling back
  to the platform user cache base (on Linux, `$HOME/.cache`).
- `--cache-disk DIR`: select an explicit directory. Its existing default quota
  is 8 GiB; `--cache-disk-bytes N` still overrides it.
- `--cache-disk off`: disable the ordinary disk tier. Explicit registered pins
  are a separate store, not disabled by this option.

Auto mode without a size override chooses `min(64 GiB, available_bytes / 10)`.
It recalculates on restart, so a smaller quota can evict older records. The quota
counts managed manifests, payloads, temporaries and orphans; it is not a promise
that every queued checkpoint becomes durable.

Both enabled modes keep a free-space reserve of
`max(8 GiB, filesystem_capacity / 50)`. Override it with
`--cache-disk-free-bytes N`; zero selects the automatic reserve. Explicitly
choosing a smaller reserve reduces protection against other writers. Startup
logs report the effective path, quota, reserve and enable/fallback reason.

Only writable local durable filesystems are admitted: ext-family, Btrfs, XFS
and F2FS on Linux; fixed NTFS/ReFS volumes on Windows. Unknown, network, memory
and read-only filesystems are not automatic SSD storage. Auto mode downgrades
to no ordinary disk cache when planning cannot admit the path; explicit paths
report an error. Windows checks directory mutation rights as well as volume
flags. Existing cache-leaf symlinks remain refused; intermediate symlinks are
supported. Later ownership/open failures still report errors rather than
silently selecting a different directory.

The guard checks a complete new publication and bounded write pieces, leaving
64 KiB allocation/metadata slack in addition to the reserve. Pressure triggers
disposable-entry eviction at startup or write attempts; active read pins and
explicit registered pins are not sacrificed. Partial component writes aborted
for space are reclaimed before retry. Reads remain available when optional
writes cannot proceed. An unavailable/read-only filesystem does not invite
eviction. There is no idle filesystem watcher.

This is a conservative admission guard, **not an atomic filesystem reservation**.
Concurrent external writers, quotas, filesystem metadata allocation and device
errors can still consume space or reject writes between checks. Failed writes
are not acknowledged as durable; incompatible/corrupt records miss safely.
It is not a substitute for host free-space monitoring or SSD endurance planning.

The policy changes neither ordinary/pin staging sizes nor the disk-only RAM
backing requirement. A disk quota that fits does not imply that model sessions,
transfer staging or backing fit in RAM.
