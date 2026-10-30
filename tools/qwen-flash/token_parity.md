# Emitted-token-ID parity diagnostics

This tool compares **actual emitted token IDs**, not tokens reconstructed from
completion text. Exact equality across long greedy generations is not a sound
quality criterion: a near-tie can amplify into different valid continuations.
The tool never claims that either continuation is coherent.

## Capability audit (pinned sources, no host/model runs)

llama.cpp reference commit:
`6d5ac3c8d5aae1831edf17c2d5cda11abe025bb3` in the preserved ggml-org mirror.
At that revision:

- `tools/server/server-common.cpp:991–1025` accepts a numeric `prompt` array on
  native `/completion`; a numeric-first array does **not** add BOS or tokenize
  text. Supply the complete intended prompt, including desired control IDs.
- `tools/server/README.md:589,599,609,619–674` documents `n_probs`,
  `return_tokens`, `post_sampling_probs` and response `tokens`,
  `completion_probabilities`, `generation_settings`.
- `tools/server/server-task.cpp:256–291` serializes each candidate's **ID** and
  natural-log `logprob` inside `top_logprobs` when post-sampling probabilities
  are off. The adapter requests pre-sampling top-k, greedy temperature0,
  top_k1 and neutral penalties. Both engines stop naturally at EOS within the
  registered `n_predict` budget; **ignore_eos is deliberately omitted**.

**Current Gufo cannot perform the equivalent live request.** At the readiness
base, `src/cli/serve/http_server.cpp:1381–1384` requires a nonempty **string** on
`/completion`; `/v1/completions` also parses text prompts. Neither response
returns emitted token IDs; logprobs/top_logprobs are rejected at
`http_server.cpp:690–695` and `openai_chat.cpp:1010–1023`. Streaming text does
not recover token identity. The common native adapter therefore reports **BLOCKED** on current Gufo's
HTTP400 or absent `tokens`, without a text fallback. The approved token-ID
`/completion` implementation can plug into the same transport unchanged;
absent per-token distributions still mean UNKNOWN. OpenAI-compatible text
endpoints alone are insufficient.

Gufo's opt-in generation trace has real `prompt_token_ids`,
`generated_token_ids`, sampling and max_tokens
(`text_generation_scheduler.cpp:426–454,477–496`). Those can supply **recorded**
evidence, paired with the actual response text and launch provenance. Capture
an authorized Gufo workflow independently, then send its captured prompt IDs
to llama.cpp. Never retokenize the response or submit a second textual prompt
as a substitute for ID ingress. Traces may contain private prompts; keep
qualification evidence private and outside product git.

The existing functional helpers were inspected: `pinned_prefix.call` imposes
text prompts and timing/cache response assumptions; `service_http.Evidence`
is httpx-based for non-chat/SSE services. Neither is an ID adapter. This small
stdlib native-llama adapter deliberately avoids changing those shared helpers.
CPU tests live with the functional harnesses and use only synthetic recordings
and a loopback fake server. The tool never launches/stops a model server.

## CLI / input contract

Pre-register a margin **before seeing outputs**, e.g. 0.02 natural-log units.
This is an operator-selected diagnostic bound, not a Gufo quality tolerance;
retain the original registration/command and never tune it to make a run pass.

```sh
# Two independent, actual-ID capture envelopes (fully offline):
python3 tools/qwen-flash/token_parity.py \
  --prompts prompts.json --gufo-recorded gufo-recorded.json \
  --llama-recorded llama-recorded.json \
  --tokens 64 --horizon 32 --margin 0.02 > token-parity.json

# Existing Gufo ID capture + running pinned native llama server:
python3 tools/qwen-flash/token_parity.py \
  --prompts prompts.json --gufo-recorded gufo-recorded.json \
  --llama-url http://127.0.0.1:8080 --llama-config llama-config.json \
  --tokens 64 --horizon 32 --margin 0.02 --top-k 5 > token-parity.json

# Once Gufo's separate numeric-prompt API lands (same native transport):
python3 tools/qwen-flash/token_parity.py \
  --prompts prompts.json --gufo-url http://127.0.0.1:8081 \
  --gufo-config gufo-config.json --llama-url http://127.0.0.1:8080 \
  --llama-config llama-config.json --tokens 64 --horizon 32 --margin 0.02 \
  > token-parity.json
```

`--llama-url` is the server base (optional `/v1` suffix), **not** the completion
route. Each live request sends the exact same numeric array registered for both sides,
never visits a tokenizer/chat endpoint, does not retry, and disables prompt
cache. Numeric arrays are essential: no inferred BOS, templates or text pass.
A live Gufo URL uses exactly the same numeric-prompt protocol; rejection or
missing IDs is BLOCKED until the separate token-ID product API lands. Supply
`--gufo-url URL --gufo-config gufo-config.json` instead of a recording.

`prompts.json`:

```json
[{"id":"prompt-1","prompt_token_ids":[248045,872,198]}]
```

Each recording envelope has `config` and `requests`. Example schema (the
repeated digest characters below are placeholders, **not real provenance**):

```json
{
  "config": {
    "revision":"FULL_ENGINE_COMMIT",
    "flags":["server","EXACT","ARGV"],
    "stack":"host HIP/ROCr OR Nix ROCm 10.1; host/compiler/dependencies",
    "model_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "tokenizer_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "mode":"ar"
  },
  "requests":[{
    "id":"prompt-1",
    "prompt_token_ids":[248045,872,198],
    "generated_token_ids":[123,456],
    "text":"Actual returned continuation, not reconstructed IDs",
    "max_tokens":2,
    "stop_type":"limit",
    "sampling":{"temperature":0,"top_k":1,"top_p":1,"min_p":0,"repeat_penalty":1,"frequency_penalty":0,"presence_penalty":0,"seed":0},
    "top_logprobs":[
      [{"id":123,"logprob":-0.5},{"id":789,"logprob":-0.51}],
      [{"id":456,"logprob":-0.3},{"id":123,"logprob":-0.9}]
    ]
  }]
}
```

For this two-token example use `--tokens 2 --horizon 2`. `top_logprobs` is
optional; omit it for today's Gufo recordings. Entries must be position-aligned
actual observations, **not fabricated**. Preserve raw responses/traces and
capture provenance alongside the normalized envelope. A live llama config
file contains just the `config` object, with the exact pinned llama revision
above. The tool requires matching model/tokenizer SHA256 values, declared AR
mode and registered greedy settings/counts, and records both engine configs,
all emitted IDs, continuation text, request settings and live raw responses.
Config/flags are operator declarations, **not runtime attestation**; separately
verify loaded mode, binary/model hashes, disabled drafts and actual flags.

### Natural EOS and stop normalization

The pinned native llama server serializes `stop_type` as eos/limit/word
(`server-task.cpp:241–247,332–346`). It appends the EOG token to generated IDs
before detecting EOS (`server-context.cpp:2143–2144,2255`). On an EOS response
only, the adapter strips llama's trailing EOG ID from comparison, retaining it
as `terminal_eog_id` and preserving the original raw IDs. Gufo's compatible
`tokens` **excludes** terminal EOG; never strip a Gufo ordinary output token.

Normalize `stop_type` or booleans `stopped_eos`, `stopped_limit` /
`stopped_length`, and `stopped_word` to eos/limit/word; everything else is other.
`stopping_word` is retained. Missing/unrecognized stop reason means UNKNOWN
for that request, never a match/pass; contradictory recognized reasons are
invalid input. Cancellation has no EOS implication.

An EOS-vs-token decision is a divergence at the shorter normalized length;
if the emitted IDs match but stop reasons differ, the divergence is at len.
The near-tie rule applies at that boundary, but without an **actually captured**
EOG/other winner ID and top-k on both sides it stays UNKNOWN. An optional
`terminal_eog_id` in a Gufo recording must come from independent actual capture,
not a guessed EOS or retokenized text. At a limit boundary there may be no next
token observation; do not fabricate one. Natural EOS before the token budget
is valid, not a truncated fixed-count failure.

## Diagnostics and acceptance boundary

For **each prompt** output includes:

1. First differing **zero-based emitted-token position**, or null; a short
   shared history and the surrounding ID sequences from both engines.
2. Both engines' ID+logprob top-k at that position when available; otherwise
   explicit null/UNKNOWN. First divergence is the only free-running mismatch
   where the conditioning history is guaranteed identical.
3. The top1–top2 gap, plus the gap to the **other engine's winner** in each
   engine. Near-tie YES requires both winners to be captured within the
   registered margin of each engine's maximum and each emitted token to be a
   documented greedy maximum. A small unrelated top1/top2 gap is not enough.
   Truncated top-k that omits the other winner gives UNKNOWN unless a large
   observed top1/top2 gap already proves NO. Missing Gufo top-k is UNKNOWN,
   never evidence of a near-tie or a pass.
4. Same-index agreement over up to the **registered short horizon** (default32
   within a64-token budget), with denominator `min(horizon, common emitted
   length)` explicitly reported. Zero denominator gives null, not100%.
   EOS differences are separately reported divergences, not hidden by a short
   denominator. Limit stops before the requested count or captures beyond the
   budget are incomplete/UNKNOWN. Post-divergence same-index agreement is
   descriptive only: histories have diverged.
5. Full continuation text on both sides for **human coherence review**.

**Acceptance rule:** a qualification PASS requires divergent decisions to be
at registered near-ties **and** coherent continuations reviewed by a human.
This tool emits **no automatic quality PASS**: a near-tie at the first split
explains that split only, not every later decision under different histories.
Further suspect splits require aligned-prefix probes/numerical evidence.
Keep human findings with the immutable report; no automatic coherence claim.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Matching bounded numerical sequence and recognized stop reason; **human review still required**, not a quality PASS |
| 1 | First divergence demonstrably not a near-tie in at least one engine |
| 2 | Invalid/mismatched provenance/input, request failure, or blocked live capability |
| 3 | UNKNOWN/degraded/incomplete stop/capture, or first split near-tie pending human review |

All results retain the margin, units, horizon and top-k request. Do not collapse
2/3 into success in an acceptance recipe. In particular today's Gufo cannot
prove the both-engine near-tie condition from server responses alone.

**Upgrade path:** once Gufo has token-ID ingress, emitted IDs and per-position
raw top-k/full logits, add a validated live adapter. The stronger acceptance
pack is full-logit KL + top-1 margins on **identical teacher-forced prefixes**,
with independently registered thresholds; raw text equality is not a substitute.
This tool is a diagnostic component, not a replacement for model numerical
quality, long-context/cache correctness or matched-stack speed benchmarks.
