# Pi → Gufo

Sources: official [model guide](https://pi.dev/models) and installed Pi 1.4.0
`docs/models.md` / `docs/models-configuration.md`. Read [the common contract](README.md).
Add a custom provider to a private `models.json` in your Pi agent directory.
For actual loaded `C=32768`, output `O=4096`:

```json
{
  "providers": {
    "gufo": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "${GUFO_API_KEY}",
      "authHeader": true,
      "models": [{
        "id": "gufo-local",
        "name": "Gufo loaded model",
        "reasoning": false,
        "input": ["text"],
        "contextWindow": 32768,
        "maxTokens": 4096,
        "compat": {
          "supportsUsageInStreaming": true,
          "supportsStrictMode": false
        }
      }]
    }
  }
}
```

Pi's `openai-completions` API name here means **Chat Completions**, not the
raw `/v1/completions` endpoint. `apiKey` uses Pi 1.4 environment interpolation;
`authHeader:true` requests bearer auth explicitly. Do not put a key literal in
git. Reopen `/model` to reload, then select `gufo/gufo-local`.

## Reasoning/cache and alternate transports

Start with `--thinking off` and Gufo `--think off`. For a qualified thinking
model, set `reasoning:true`, use verified `thinkingFormat:openai` /
`supportsReasoningEffort:true` compatibility and supported effort mappings.
Don't enable a reasoning capability just to make a UI option visible.
Keep the compatibility settings minimal: false strict mode is a conservative
recipe setting, not a claim that Gufo lacks schema constraints.

Do **not** add `promptCache` lifetime metadata: Gufo has no guaranteed short/
long TTL. Pi only warms caches with declared lifetimes; ordinary Gufo prefix
reuse still applies. OpenAI-compatible `samplingParams` can carry
`cache_prompt`/`cache_persist` for a separately qualified workflow; neither
means no-retention privacy.

Use a separate model/profile with `api:openai-responses` for Responses.
It must replay full history and `store:false`, without previous-response
state or hosted tools. Anthropic mode is not the default recipe: its SDK may
send `x-api-key`; authenticated Gufo requires bearer, explicitly configured
and checked on the wire. No guaranteed signature/Anthropic TTL semantics.

## Qualification slot

Follow [the exact Pi replay procedure](QUALIFICATION.md), retaining
`session.jsonl`, tool results, wire and server log. The maintained functional
harness is pinned to Pi **0.87.0**, not the installed 1.4.0 model-guide version:
keep those versions distinct. Qualify an interactive 1.4.0 profile separately;
do not bypass the harness version assertion or call it equivalent evidence.
