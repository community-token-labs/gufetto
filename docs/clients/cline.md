# Cline → Gufo

Source: official [OpenAI Compatible guide](https://docs.cline.bot/provider-config/openai-compatible).
Read [the common limits/auth contract](README.md).

In Cline's settings (gear), use:

| Field | Value |
| --- | --- |
| API Provider | OpenAI Compatible |
| Base URL | `http://127.0.0.1:8080/v1` |
| API Key | Gufo's single bearer key, stored privately |
| Model ID | `gufo-local` (or an actual `/v1/models` ID) |
| Context Window | Actual loaded capacity; 32768 only for the common example |
| Max Output Tokens | 4096 only for the common example; leave history/tool headroom |
| Image Support | Off unless the exact loaded model/projector is qualified |
| Computer Use | Off for this text-only recipe |
| Azure Identity Authentication | Off (not Gufo bearer authentication) |

Gufo also accepts the Anthropic SDK's `x-api-key`, but this recipe qualifies
only the OpenAI-compatible provider. The official guide's Verify action confirms connectivity, not a tool loop.

## Reasoning/cache and limitations

Start with server `--think off`. If the installed Cline build exposes extra
request fields, use Chat `reasoning_effort:none`, not a conflicting thinking
template or a Claude-specific reasoning configuration. This recipe does not
invent an advanced-field UI where a version doesn't have one. Prefix reuse is
automatic; no guaranteed TTL or Anthropic cache-control semantics. Disable
unsupported active samplers/logprobs and remote hosted tools. Verify whether
the chosen Cline mode emits native tools or its text/XML tool protocol; record
that mode, don't describe XML execution as native function-call qualification.

## Qualification slot

Open the prepared disposable repository in the editor, select this provider
and **Act** mode, submit the exact [read/edit/test task](QUALIFICATION.md), and
approve only its bounded local operations. Retain Cline's exported task,
actual tool results, final diff and independent test result. Capture request
history/SSE and inspect call/result IDs where native tools are used. Run the
same workflow with Pi afterwards; a model that only describes edits fails.
