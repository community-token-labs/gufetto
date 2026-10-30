# Client recipes

These are source-checked recipes, not a blanket client/model support promise.
CPU route fixtures exercise the protocol with a fake backend. Real tool execution
and model correctness must pass the [qualification procedure](QUALIFICATION.md)
in the affected model's allocated slot before describing that combination as tested.

- [Claude Code](claude-code.md): Anthropic Messages, experimental non-Claude routing
- [OpenCode](opencode.md): OpenAI Chat Completions
- [Cline](cline.md): OpenAI Compatible
- [Aider](aider.md): OpenAI-compatible chat/edit workflow
- [Open WebUI](open-webui.md): Chat Completions and optional native tools
- [Pi](pi.md): Chat Completions; stateless Responses as a separate profile

## Common server contract

Use a production binary and the model's documented startup command; do not
replace its draft/projector/context settings with another model's. For a small
initial text-only session, a **requested** context of 32768 and output budget
4096 are examples, not universal model limits:

```sh
/path/to/gufo serve --host 127.0.0.1 --port 8080 --api-key "$GUFO_API_KEY" \
  llm --model "$MODEL" --served-model-name gufo-local --context 32768 --think off
curl -fsS -H "Authorization: Bearer $GUFO_API_KEY" http://127.0.0.1:8080/v1/models
```

Choose `C` from the actual loaded context reported by `/v1/models`, constrained
by the model's supported context and memory fit; choose output `O < C`, leaving
room for system prompt, tools and full history. The examples below use `C=32768`,
`O=4096` **only after loading that capacity**. Client metadata does not expand
server capacity; inspect actual outbound limits. Never copy a cloud model's 1M
context label or use a Claude `[1m]` alias for a local model. No Gufo alias
changes model architecture/capabilities. See [context and limits](../SERVER.md).

- OpenAI base: `http://127.0.0.1:8080/v1` (client appends `/chat/completions`
  or `/responses`). Anthropic SDK base: `http://127.0.0.1:8080` (appends `/v1/messages`).
- Gufo recognizes **one shared bearer key**, not per-user keys. Anthropic SDKs
  commonly send `x-api-key`; **that alone fails** against authenticated Gufo.
  Use documented bearer/auth-token support. Do not fix this by disabling auth
  on a LAN bind. Native `x-api-key` support is a separate auth change.
- Start thinking off (`--think off`). Chat `reasoning_effort: none` disables it;
  enable only a model-supported mode during explicit qualification. Responses
  uses `reasoning.effort`; Messages uses `thinking` / `output_config.effort`.
  Do not send contradictory template/reasoning controls. Output budget includes thinking.
- Prefix caching is automatic/best effort, not a promised TTL. `cache_prompt`
  controls reuse; `cache_persist` controls request disk capture, subject to
  server configuration. Neither is a secure no-retention switch. Anthropic
  `cache_control` markers do not buy Anthropic TTL/cache-write semantics.
- Responses is **stateless**: send full prior input/output/tool history,
  `store:false`, no `previous_response_id` or background/server conversations.
  Tools execute in the client, never in Gufo. Hosted web search is not supplied.
- Text-only unless the exact loaded projector/model is qualified for images.
  Disable unsupported active sampler, logprob, FIM and hosted-tool features.

## CPU contracts and known limitations

`http_server_test` replays recorded fake-backend SSE for Chat, Responses and
Messages in 1-byte, 7-byte and whole-frame fragments, then exercises each real
HTTP route and replays its tool result with the returned call ID. It checks
Chat tool indexes/argument assembly/finish/usage/DONE; Responses sequence and
argument/completion events; Messages block ordering, JSON deltas and terminal
usage. Fixtures live in `tests/fixtures/client-recipes/`; their provenance is
not a capture from a real client or a model run.

Known auth limitation has a **passing negative regression** in `TestAuthorization`:
`x-api-key` without bearer returns 401 before generation. Proposed future
acceptance test: same Messages/count_tokens requests with only a matching
`x-api-key` succeed; wrong/mixed/duplicate keys fail consistently. That proposal
is deliberately NOT enabled here and server behavior is unchanged.

Claude Code's non-Claude routing is not supported by Anthropic; model-specific
thinking signatures, hosted tools and larger context assumptions remain
unqualified. Aider edits may use text rather than native tool calls. GUI
connection verification and a plausible final answer are not real tool-loop
qualification.

Official pages were read on 2026-10-30. Pin actual client versions in each
qualification receipt; these moving docs are not evidence for every release.
