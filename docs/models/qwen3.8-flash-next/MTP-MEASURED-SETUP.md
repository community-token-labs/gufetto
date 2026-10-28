# Measured internal MTP setup: Q8 ON through C4

**Internal recommendation, 2026-10-27:** use `--speculative mtp` with the
shared-Q8_0 sidecar for the tested C1–C4 profile. Q8 won every measured
concurrency on both hosts; neither a slot cutoff nor a new flag is justified.
The two-host addendum below preserves the distinct product revisions and limits. Shared-Q4_K_M
remains a supported, experimental lower-weight-memory option, not the measured
throughput winner. This recommendation changes no compiled CLI default.

## Original nimo-1 measurement

Coordinator-only nimo-1 runs used the native binary built from Gufo
`8f487a48611a33ad6fbe17276c2c86e5d1c312b6`, target UD-Q4_K_XL,
shared target Q8 embedding/output head, four sessions, context **65536**,
prefill chunk4096, RAM cache **4294967296 bytes**, thinking off, temperature0,
F16 KV and YaRN off. **No pinned prefixes; deployment pin behaviour not measured.**
This is not the production200000-context/11329267712-byte/pinned-prefix profile,
not production-container ELF equivalence or, by itself, stable/public qualification.
Current channel approval is recorded separately in [STRIX-STATUS](../../STRIX-STATUS.md).

The ordinary Gufo `tools/serving/gufo-serving-bench.py` performed the measurement;
no custom map/registry monitoring was used. The client workspace revision95089ca9
is not the product revision. Run metadata records the copied binary SHA.

- Rates: lease `a28538f7845a`, worker success, production VERIFIED,689s.
  Canonical nonce prompts, C1/2/3/4, max256 outputs, one warm-up, three repetitions.
  Each cell contains3/6/9/12 measured requests respectively; all returned256 tokens,
  with zero reported prompt-cache hits. Warm-up is excluded.
- Identity: lease `b6c6fae0c92e`, worker success, production VERIFIED,204s.
  Official fixed corpus `speculative-corpus.json`, three quick cases, C1/C4,
  max256, one warm-up, one repetition, `--no-cache-prompt`. OFF first, then Q8
  with `--reference-report OFF-identity.json`.

The raw reports omit generated text and retain completion hashes. Aggregate
output t/s is total completion tokens divided by summed measured round spans,
including prefill/queueing; it is not mean request decode speed or whole lease
throughput. Request completion rate is not a semantic-correct agent-turn score.

## Canonical nonce-prompt rates

| Concurrency | OFF output t/s | Q8 output t/s | Q8/OFF | Q4 output t/s | Q4/OFF |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 28.20 | **39.80** | **1.41×** | 36.24 | 1.28× |
| C2 | 43.04 | **61.54** | **1.43×** | 55.60 | 1.29× |
| C3 | 54.29 | **69.14** | **1.27×** | 66.19 | 1.22× |
| C4 | 63.37 | **76.51** | **1.21×** | 73.63 | 1.16× |

Completed requests per measured minute, C1→C4:
OFF6.61/10.09/12.72/14.85; Q8**9.33/14.42/16.20/17.93**;
Q4 8.49/13.03/15.51/17.26. With256 outputs/request this is a restatement of
output throughput, not independent evidence of useful agent work.

| C | Median request decode t/s OFF | Q8 | Q4 | Q8 acceptance | Q4 acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 29.52 | 42.70 | 38.25 | 55.24% | 55.69% |
| 2 | 23.51 | 35.19 | 30.87 | 57.36% | 60.52% |
| 3 | 20.03 | 26.02 | 25.92 | 62.45% | 60.27% |
| 4 | 17.72 | 22.07 | 21.37 | 62.64% | 61.98% |

Decode rates use server decode time, not client wall time or measured GPU time.
Acceptance is total accepted drafts divided by total drafted tokens, not an
average of request ratios; these reports do not split neural and lookup drafts.
Q4's higher C2 acceptance did not make it faster. Sidecar weight files are
2,786,568,256B for Q8 versus1,907,151,936B for Q4 (about31.6% smaller); this is not
a measured serving RSS, DRAM-traffic or peak-memory reduction.

## TTFT tradeoff: show median and p95

Client-observed TTFT milliseconds, **median / p95**:

| C | OFF | Q8 | Q4 |
| --- | ---: | ---: | ---: |
| 1 | 401.09 / 401.16 | 408.03 / 408.20 | 441.45 / 442.37 |
| 2 | 603.69 / 805.89 | **818.14 / 818.19** | 885.08 / 886.39 |
| 3 | 804.33 / 1238.98 | 818.23 / 1228.49 | 885.75 / 1330.45 |
| 4 | 1020.97 / 1680.31 | 1028.68 / 1728.97 | 1127.02 / 1888.67 |

At C2, Q8 adds **214.46ms (+35.52%) to the median**, while p95 rises
**12.30ms (+1.53%)**. Server median TTFT likewise rises603.06→817.39ms.
The distributions differ; this is not a uniform per-request214ms penalty.
There are only3/6/9/12 requests per cell: these are descriptive sample quantiles,
not robust tail guarantees or confidence intervals.

## Fixed-suite greedy identity and confirmation

The canonical default injects a different nonce into every prompt. Its rates
are useful internal evidence, but its hashes cannot prove cross-arm identity.
The separate official fixed-suite run supplies that missing comparison:

| C | OFF output t/s | Q8 output t/s | Q8/OFF | Q8 acceptance | Q8 hashes versus OFF |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 29.18 | 58.00 | 1.99× | 71.92% | **3/3 exact**, all3 cases |
| 4 | 64.24 | 91.06 | 1.42× | 69.00% | **4/4 exact** |

Raw-sample analysis confirms both OFF-same-C and stable OFF-C1 equality, including
OFF itself at C4. These seven Q8 requests are byte-identical to OFF for the fixed
suite. This does not qualify every prompt, sampling setting or context length.
The corpus and acceptance differ from the nonce run; do not pool their rates.
Q4 did not receive this identity follow-up. Its cache/sampling/batch gates passed
in earlier native runs; the original recursive-carry oracle FAIL remains a labelled
internal record-only result, not a public numerical PASS.

## Recommended invocation and limits

Keep the measured profile explicit; adapt model paths to the installation:

```sh
gufo serve llm --model "$TARGET_GGUF" --mtp-model "$SHARED_Q8_MTP_GGUF" \
  --speculative mtp --sessions 4 --context 65536 --prefill-chunk 4096 \
  --cache-ram-bytes 4294967296 --kv-cache-type f16 --think off
```

No pin flags or `--enable-yarn`. Q8 MTP stays ON for every tested C≤4. There is
no measured reason to add a slot-aware cutoff, shorter-draft default or new flag.
For memory constraints, select the shared-Q4_K_M sidecar explicitly and retain its
experimental qualification limits. Do not combine Q4 with rotated-q8 KV until
R50's fix is qualified; YaRN batched catch-up qualification is also separate.

Unsloth's [pinned MTP notes](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/766911a6b7369840a91dbcd95f9f997acaab6cd6/MTP/README.md)
report a C8 net slowdown of **0.81–0.87×**. Their published setup is not matched
to this host/workload/implementation. Our C4 gain of1.21× neither contradicts
that result nor predicts C8. No universal concurrency cutoff is established.

## Two-host addendum — 2026-10-27

Coordinator called and completed nimo-2 lease **a8e32788276a**: worker success,
production VERIFIED,809s. It reused the same r2 and identity scripts with only
the installation path changed and the same measurement settings. N2 used the
C2 production binary SHA256
`2cdbea45de3e34b7300c35144bbd38305b77f5d84a68bd0a1c9f19a377762c82`;
**2cdbea45 is a binary hash, not a Git revision**. Its non-test product code is
identical to C2t/main/stable `242b4fcf477109e1ee10a281581e11badc90238c`, as
recorded by the coordinator and STRIX-STATUS. N1 used the earlier8f build.
Thus this repeats benefit/identity across two hosts, not a same-build paired
experiment isolating hardware effects.

N2 canonical output t/s; parentheses are **N2 versus N1 percent deltas**:

| C | OFF N2 (delta) | Q8 N2 (delta) | Q4 N2 (delta) | Q8/OFF N2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 28.46 (+0.91%) | **39.61 (-0.48%)** | 37.66 (+3.92%) | **1.392×** |
| 2 | 43.41 (+0.86%) | **58.83 (-4.42%)** | 55.01 (-1.06%) | **1.355×** |
| 3 | 54.97 (+1.25%) | **69.30 (+0.24%)** | 65.95 (-0.37%) | **1.261×** |
| 4 | 64.56 (+1.89%) | **77.99 (+1.93%)** | 74.47 (+1.14%) | **1.208×** |

Q4/OFF on N2 is1.323×/1.267×/1.200×/1.154×. **Q8 wins every C on both
hosts; MTP remains positive at C4. Recommendation unchanged: Q8 ON through C4,
Q4 lower-weight-memory option, no slot policy or new flags.**

### Is the C2 Q8 spread noise?

N2 is **2.72t/s (-4.42%)** below N1's61.54t/s. The three individual round
rates are N1 **63.23,64.26,57.58** and N2 **56.78,58.17,61.75**t/s. Their ranges
overlap; sample round-rate CV is5.84% and4.35%, respectively. The aggregate
headline is still ratio-of-sums, not the arithmetic mean used for that CV.
The gap is consistent with observed round variation, **not established as
noise or equivalence**. Three unpaired rounds, different nonce prompts, hosts,
builds and sequential arm order cannot identify a host penalty or regression.
It does not overturn Q8's within-host win; no further run is requested.

C2 client TTFT, milliseconds **median / p95**:

| Host | OFF | Q8 | Q8 median change | Q8 p95 change |
| --- | ---: | ---: | ---: | ---: |
| nimo-1 | 603.69 / 805.89 | 818.14 / 818.19 | +214.46ms (+35.52%) | +12.30ms (+1.53%) |
| nimo-2 | 600.42 / 801.63 | 816.57 / 818.05 | +216.14ms (+36.00%) | +16.42ms (+2.05%) |

The same median/tail tradeoff repeats; small-sample quantiles are not tail
SLO guarantees. Fixed-suite N2 output rates are29.26→58.22t/s at C1 (**1.990×**)
and64.48→91.90 at C4 (**1.425×**). Q8 matches OFF **3/3 C1 and4/4 C4** against
both same-C and OFF-C1 references. All compared OFF/Q8 fixed-suite hashes also
match across hosts for those seven requests. Different corpus: do not pool these
rates with the canonical matrix. No Q4 fixed-suite identity claim is added.

### Channel and qualification scope after the repeat

MTP support and the earlier recommendation docs already belong to approved
main/stable C2t. This addendum records bounded two-host performance and identity;
it neither independently promotes a prefix nor retroactively turns these
opportunistic measurements into preregistered public qualification. Preserve the
coordinator's adopted C2t acceptance and its limitations in STRIX-STATUS. There
is no claim for production200K/pins, rotated KV, YaRN, longer outputs or C>4 from
these runs. Stable is a public/upstream-candidate line, **not publication permission**.
This docs-only feature is based on origin/nightly; the coordinator lands it.
MTP is done and idle: no further model, harness or repeat is queued.
