# OpenCode → Gufo

Source: official [provider guide](https://opencode.ai/docs/providers.md),
especially the llama.cpp/LM Studio custom-provider examples. Use the
`@ai-sdk/openai-compatible` package for Chat Completions, not a cloud provider
whose default may be Responses.

Read [the common contract](README.md). With loaded `C=32768`, `O=4096`,
put this in an isolated OpenCode config (no literal secrets):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "gufo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Gufo local",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "{env:GUFO_API_KEY}"
      },
      "models": {
        "gufo-local": {
          "name": "Gufo loaded model",
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  }
}
```

Use `/models` and select `gufo/gufo-local`. The API key goes as bearer
authentication; Anthropic `x-api-key` alone would not authenticate to Gufo.
If this client version uses `/connect` for custom credentials, store the key
there privately instead; do not check its auth store into git.

## Reasoning/cache and limitations

Keep the server `--think off` for the first qualification; don't enable a
cloud-model reasoning variant by name. Explicit Chat overrides, if supported
by the pinned client version, must use `reasoning_effort` with a supported
Gufo value. No guaranteed cache TTL: use normal prefix reuse, not provider
cache-price/retention promises. Set image capabilities only for a loaded,
qualified projector. Context/output metadata must match actual capacity and
outbound limits, not the official example's model or its output size.

## Qualification slot

In the prepared disposable repository, launch `opencode`, select the custom
model, and submit the exact [read/edit/test prompt](QUALIFICATION.md).
Retain session export, tool approvals, actual file read/edit/test results and
request SSE/usage. Verify tool-call index 0, assembled JSON and call-ID replay;
then run the paired Pi edit/bugfix replay. A saved provider config or successful
`/models` is only setup evidence.
