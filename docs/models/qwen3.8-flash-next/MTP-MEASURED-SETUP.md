# Measured internal MTP setup: Q8 ON through C4

**Internal recommendation, 2026-10-27:** use `--speculative mtp` with the
shared-Q8_0 sidecar for the tested C1–C4 profile. Q8 won every measured
concurrency; neither a slot cutoff nor a new flag is justified. Shared-Q4_K_M
remains a supported, experimental lower-weight-memory option, not the measured
throughput winner. This recommendation changes no compiled CLI default.

## What was measured

Coordinator-only nimo-1 runs used the native binary built from Gufo
`8f487a48611a33ad6fbe17276c2c86e5d1c312b6`, target UD-Q4_K_XL,
shared target Q8 embedding/output head, four sessions, context **65536**,
prefill chunk4096, RAM cache **4294967296 bytes**, thinking off, temperature0,
F16 KV and YaRN off. **No pinned prefixes; deployment pin behaviour not measured.**
This is not the production200000-context/11329267712-byte/pinned-prefix profile,
not production-container ELF equivalence and not stable/public qualification.

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

## Promotion boundary and deferred stable packet

This is an internal recommendation from one host, sequential arm order and a
small, unpaired run—not preregistered quality/performance qualification.
Main already contains the experimental MTP implementation. The coordinator owns
approval of any nightly-prefix promotion after native Nimo build and short smoke;
this docs-only commit does not push nightly, main or stable or activate a service.

For a later stable packet, **do not queue work now**: q8 C admission goes first
on nimo-2. Once corrected C exists and is coordinator-approved, preregister the
same r2 rate matrix plus fixed-suite OFF→Q8 identity scripts, versions, settings,
quality/performance thresholds and repetition/order controls on nimo-1 and
nimo-2. Native portability and unchanged output/cache/sampling behavior must be
checked on that exact corrected prefix. Keep production200K/pin behaviour,
rotated KV, YaRN, longer outputs and C>4 out of the claim unless separately
measured. Stable requires coordinator approval plus preregistered quality/perf
and two-host portability; only stable is a public/upstream-candidate line, not
permission to publish.
