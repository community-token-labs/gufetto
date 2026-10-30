# Claude Code → Gufo

**Experimental, not Anthropic-supported for non-Claude models.** The official
[gateway guide](https://code.claude.com/docs/en/llm-gateway) explicitly excludes
support for routing Claude Code to non-Claude models. This is a bounded
compatibility recipe, not a substitute Claude service.

Read the [common limits/auth contract](README.md) first.
The official [gateway connection guide](https://code.claude.com/docs/en/llm-gateway-connect)
distinguishes bearer `ANTHROPIC_AUTH_TOKEN` from `ANTHROPIC_API_KEY` (`x-api-key`).
Gufo accepts either for its single key; set only one, since a request sending
both headers must send the key in both. This recipe uses the bearer variable:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN="$GUFO_API_KEY"
export ANTHROPIC_MODEL=gufo-local
export ANTHROPIC_DEFAULT_HAIKU_MODEL=gufo-local
export ANTHROPIC_DEFAULT_SONNET_MODEL=gufo-local
export ANTHROPIC_DEFAULT_OPUS_MODEL=gufo-local
export CLAUDE_CODE_SUBAGENT_MODEL=gufo-local
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
claude --model gufo-local
```

These mappings target the actual served name, not spoofed Claude model IDs.
The official [environment variable reference](https://code.claude.com/docs/en/env-vars)
confirms all four default/subagent variable names and the output-cap knob
(source-verified 2026-10-30; high confidence in names, **not live-qualified**).
Haiku also handles background functionality such as titles/command checks;
without these mappings Gufo rejects unserved model IDs with `model_not_found`.
The output cap applies to most requests, not a guaranteed cap on every auxiliary
request: check actual outbound budgets and all background requests.

Current [subagent documentation](https://code.claude.com/docs/en/sub-agents#run-every-subagent-on-one-model)
says an explicit model in a subagent definition/spawn can override the default.
For versions 2.1.257+, setting `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1` together with
the mapped subagent variable forces that served model for such requests. This
recipe's bounded qualification task prohibits subagents; if you later qualify
them, inspect their actual model IDs rather than assuming default mapping wins.
Older builds may differ; do not guess legacy variable support.

Use an isolated client profile/environment; don't overwrite a personal
subscription configuration. Remove conflicting API-key/provider routing
variables from that launch environment; confirm the credential/model with
`/status`. Never put real keys in a project settings file.
The SDK appends `/v1/messages`: don't add a second `/v1` to this base URL.

## Limits, reasoning and caching

- Gufo's actual loaded context is authoritative; the common example is 32768
  total tokens and 4096 output, not a Claude model's cloud context.
- The [model configuration guide](https://code.claude.com/docs/en/model-config)
  describes Claude aliases, not arbitrary Gufo model capacity. Do not use
  `sonnet[1m]`, `opus[1m]`, or spoof a Claude ID to obtain larger limits.
  This recipe does **not** claim a verified generic context/output override in
  every Claude Code version. Check outgoing `max_tokens` and token-count requests;
  if they exceed the loaded budget or the client assumes a larger window,
  record BLOCKED/INCONCLUSIVE rather than calling the recipe accepted.
- Start the server thinking off; inspect actual `thinking` and
  `output_config.effort` fields. Empty/local thinking signatures are not
  Anthropic signature authentication. Avoid adaptive-thinking assumptions
  until that exact model/client version passes replay.
- Leave prefix caching automatic; Anthropic cache markers are not Gufo TTL
  reservations. Do not advertise cache creation billing or guaranteed retention.
  Hosted web-search/document tools are not implemented by this local endpoint.

## Qualification slot

Follow [the shared procedure](QUALIFICATION.md): open the prepared disposable
repository with this CLI, approve only the bounded read/edit/test task, retain
the session and actual Bash/tool results. Verify `tool_use` IDs survive
`tool_result` replay and the terminal `message_stop`. Then replay the same
edit/test workflow with the documented Pi harness. Any startup validation,
unsupported model capability or oversized request is a retained failure/blocker,
not permission to silently substitute a different model or disable auth.
