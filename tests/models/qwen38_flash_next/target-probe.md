# Private same-server target-logit capture

Code-only qualification support; no GPU result is claimed. This probe is
Flash-Next-specific and is **not an HTTP option or supported public API**.
It does not alter payload ABI20, transfer compatibility, kernels, graph policy,
sampling, rollback, or cache ownership. Do not use the older
`GUFO_FLASH_NEXT_DUMP_DIR` activation dump for this procedure: that dump disables
graphs and does not capture these target selections.

## What is observed

- AR: `QwenFlashNextTextRunner::SelectNext` observes the live session's
  `Logits()` at the actual sampler selection, including first-token preview.
- MTP: `Session::PrepareDecode` observes the same live distribution for each
  anchor. `Session::FinishDecode` downloads bounded **fresh target** LM-head
  rows using `Executor::ReadVerificationRows`, with the selected session's
  batch offset, before verification. Only rows belonging to kept inputs are
  written. Neural draft logits and rejected/discarded suffix rows are excluded.
- A restored first frontier lives in `Session::logits_` and is precisely what
  the executing sampler consumes; it is not a re-export of an immutable cache
  record. Fresh suffix rows test the restored recurrent/KV state through real
  execution. The probe does not recompute the imported first frontier and does
  not claim that observing that one saved row alone proves restoration.

Position **N** denotes selection of the token after **N consumed tokens**.
Thus N = full rendered prompt length is the first decode, not necessarily the
checkpoint's shorter admitted/stable depth. The unchanged request still runs
its ordinary suffix prefill when those depths differ.

Files, named by decimal next-token position:

- `N.logits`: complete raw target vocabulary, little-endian IEEE float32, in
  ascending token-ID order; no sampler penalties/masks/postprocessing.
- `N.token`: one little-endian u32 selected/emitted token ID.
- `N.stop`: empty marker for a terminal selection; EOS is not an emitted token.

MTP may additionally write the final kept frontier's `.logits` before a token
has been selected there. Compare token-associated rows and stop rows for a
fixed generated suffix; do not mistake that extra frontier for another output.
A repeated observation (preview, anchor, retained frontier) must be byte-exact;
a different or truncated existing file fails the request rather than replacing
its evidence. Any request error/cancellation makes the capture incomplete and
unqualified. Preserve partial runs.

The supported qualification is **greedy, sequential requests**, separately in
AR and real-MTP modes, with identical model/settings/request on both hosts.
Equal selected IDs establish the same teacher-forced input history for every
compared suffix row; stop at the first divergence. This does not add an API for
forcing an arbitrary caller-supplied decode suffix. Concurrent captures with
colliding absolute positions are not supported. Do not use instrumented runs
for latency qualification: enabled capture adds transfers, synchronization and
private disk I/O. The unset path does no probe I/O or device transfers.

## Exact operational procedure (operator runs later)

1. Build the final source **once** after the parent's CPU checks, then install
   the **identical immutable executable and runtime objects** on both hosts.
   No helper executable imports transfers and no epoch check is bypassed.
2. Before either server starts, provision the same absolute private path, on
   encrypted disk, owned by the server UID and mode 0700. Export exactly the
   same environment on both hosts, including:

   ```sh
   export GUFO_FLASH_NEXT_TARGET_PROBE=/absolute/private/qfn-target-probe
   umask 077
   install -d -m 700 "$GUFO_FLASH_NEXT_TARGET_PROBE"
   ```

   Start each server with the parent's existing qualified options and transfer
   flag. Keep all GUFO/HIP/HSA/etc environment values identical. The new variable
   is included in the existing compatibility environment hash; **do not change
   its value or unset it between export and import**. Different archive paths
   are allowed outside the server environment. No `arm` file means no capture.
3. For cold, bypass cache using the existing ordinary request control. For
   local-warm, prime the exact intended cache history **while disarmed**. For
   imported-warm, import while disarmed into the intended fresh destination,
   verify admission, and send the unchanged full request. This probe does not
   itself admit/restore/check cache work; retain the ordinary cached-token and
   stop/count evidence separately.
4. Immediately before the one measured correctness request, ensure prior probe
   outputs have been archived and removed from this fixed directory. Let `N`
   be its known full rendered prompt token count (not blob admitted depth), and
   choose `K` in 1..32. Arm atomically:

   ```sh
   printf '%s:%s\n' "$N" "$((N + K))" > "$GUFO_FLASH_NEXT_TARGET_PROBE/arm.new"
   chmod 600 "$GUFO_FLASH_NEXT_TARGET_PROBE/arm.new"
   mv "$GUFO_FLASH_NEXT_TARGET_PROBE/arm.new" "$GUFO_FLASH_NEXT_TARGET_PROBE/arm"
   ```

   The arm file is strictly `first:end`, optionally one trailing newline, an
   end-exclusive absolute position window of at most 32 rows. Bounds live in
   this local file, **not an environment change**. Do not edit/replace it during
   a request. Directory and arm/output files reject non-owner/group/other
   access, symlinks, and nonregular files. No paths/content are logged on errors.
5. Issue the ordinary identical greedy chat request, bounded to the desired
   suffix. After it completes, remove `arm`, then archive the private files
   outside the fixed probe directory; clear outputs before the next case.
   Never leave a previous capture in place when arming another run. Repeat
   cold/local-warm/imported-warm and cross-host controls with the same bounds.
6. Compare the complete emitted `.token` set, terminal `.stop` position and raw
   `.logits` bytes for all selected positions (plus terminal if present), not
   decoded text alone. Require the expected contiguous token positions and
   expected counts from the HTTP response; missing files are a failure, not an
   equal empty capture. A fixed suffix of K emitted tokens needs K `.token`
   and K associated `.logits` files in `[N,N+K)` unless the model stops earlier.
   Full raw equality implies equal top-k IDs/order under the same explicit tie
   rule (descending finite logit, ascending token ID for ties); offline top-k
   extraction must use that rule. Compare only matched modes/histories; keep
   extra MTP frontier rows identifiable. Do not relax tolerances on mismatch.

Files reveal token IDs and model distributions and may reconstruct private
content. Keep 0600/encryption/retention controls, do not attach them to public
logs/issues, and do not add a server request-body trace. Captures are diagnostic
files, not durable transactional storage; any write/request failure invalidates
the run.

## Focused CPU check (after parent releases build activity)

`qwen38_flash_next_target_probe_test` / CTest
`qwen38_flash_next.target_probe` checks the disabled/disarmed path, bounded arm
parsing/intersection, exact float/token bytes, stop markers, private modes,
symlink refusal and inconsistent duplicate preservation. It is header-only
instrumentation coverage, not HIP compilation or GPU restoration evidence.
The existing GPU qualification still must exercise AR, real-MTP, images, tool
framing, preserve-thinking variants, learned/unaligned boundaries and both
transfer directions with the final identical executable.
