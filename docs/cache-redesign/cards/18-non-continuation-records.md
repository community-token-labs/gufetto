# 18 · Capability records for non-continuation models

**Milestone:** Adapters · **Depends on:** 02 · **Size:** S ·
**Affects:** nothing user-visible · **Status:** done

## Goal

Every model family states its continuation capability explicitly, so "all
models are covered" is checkable rather than implied.

## Scope

- EmbeddingGemma 2, Qwen3 ASR, Qwen3 TTS, Qwen Image 2.1 and MiniMax H3 declare
  `continuation = false`.
- Check whether ASR or TTS has internal state that a later request could
  reuse (streaming sessions, for example). If so, record it as a separate
  future capability, not a token-prefix continuation.
- A test enumerates every registered model family and fails if one has no
  capability record.

## Not in this PR

Any reuse for these models. Diffusion trajectories are not forced into the
token-prefix interface.

## Test first

The enumeration test, failing until each family has its record.

## Step baseline

None: declarations only.

## Done when

- [x] Every family has a record and the test passes.

Each `src/models/<family>/cache_capabilities.hpp` holds the record, and
`cache_model_capabilities_test` matches all eight records against the model
directories. Qwen and Flash-Next adapters return their records directly;
DeepSeek's legacy runner consumes its record for prefix reuse and persistent
encoding (which still requires a loaded-model fingerprint). No runtime
capability values change.

Embedding requests encode complete inputs. Neither ASR nor TTS reuses state
across requests: live ASR sessions transcribe each committed buffer independently,
and streamed TTS segments are separate requests. No separate capability is
recorded.

## RFC

[Component descriptors and model modules](../RFC.md#component-descriptors-and-model-modules)

## Review notes

