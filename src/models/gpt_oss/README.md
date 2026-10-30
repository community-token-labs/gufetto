# gpt-oss

Model-private code for OpenAI gpt-oss (`general.architecture = gpt-oss`),
pinned to `ggml-org/gpt-oss-120b-GGUF@7200b74b`. The runtime is not wired yet;
this directory holds the Harmony chat format. MXFP4 weights decode through the
shared GGML path in `src/core/quant/ggml_dequant.hpp`.

## Harmony control tokens

[`control_tokens.hpp`](control_tokens.hpp) owns the Harmony framing spellings
and their pinned vocabulary ids, following the Qwen catalogue's rules
(`src/models/qwen/AGENTS.md`):

- Use the named constants in production rendering, parsing and stop handling.
- Confirm new tokens against the pinned GGUF metadata.
- Tests compare against independent literals and rendered-byte goldens, never
  against values derived from the constants they validate.
- `<|endoftext|>` shares its spelling with Qwen's `kEndOfText` but belongs to
  this vocabulary; do not mix the families.

| Constant | Id | Meaning |
|---|---|---|
| `kStartOfText` | 199998 | BOS. |
| `kEndOfText` | 199999 | Padding. |
| `kReturn` | 200002 | EOS: ends a final answer. |
| `kConstrain` | 200003 | Precedes a content type in a message header. |
| `kChannel` | 200005 | Precedes the channel: `analysis`, `commentary` or `final`. |
| `kStart` | 200006 | Opens a message, followed by the role header. |
| `kEnd` | 200007 | Closes a message without ending generation. |
| `kMessage` | 200008 | Separates the header from the body. |
| `kCall` | 200012 | Ends a tool call; generation stops. |
| `kEndOfPrompt` | 200018 | Not rendered by the chat template. |

## Chat template

[`harmony.hpp`](harmony.hpp) compiles the GGUF's `tokenizer.chat_template`
(SHA-256 `a4c9919c…`, identical to llama.cpp's
`models/templates/openai-gpt-oss-120b.jinja`). Known differences:

- An empty `thinking` is treated as absent.
- Tool-call arguments must be JSON; they are re-serialised with Jinja's
  `tojson` spacing. Numbers keep Gufo's JSON spelling.
- `oneOf` parameter schemas are rejected until the renderer covers them.
- Built-in `browser`/`python` tools are not rendered.

`tests/models/gpt_oss/make_harmony_goldens.py` regenerates the rendered-byte
goldens with Jinja2 from the pinned template.
