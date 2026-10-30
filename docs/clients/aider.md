# Aider → Gufo

Sources: official [OpenAI-compatible setup](https://aider.chat/docs/llms/openai-compat.html)
and [advanced model metadata/settings](https://aider.chat/docs/config/adv-model-settings.html).
Read [the common contract](README.md).

```sh
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY="$GUFO_API_KEY"
aider --model openai/gufo-local
```

Use the `openai/` prefix exactly; the model ID after it must be served by Gufo.
This sends bearer auth. Use a disposable repo, not personal project/auth
configuration.

For loaded `C=32768`, `O=4096`, example private
`.aider.model.metadata.json` (input allowance reserves output space):

```json
{
  "openai/gufo-local": {
    "max_tokens": 4096,
    "max_input_tokens": 28672,
    "max_output_tokens": 4096,
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "litellm_provider": "openai",
    "mode": "chat"
  }
}
```

Zero price means no metered API charge, not zero hardware cost. Aider's official
guide says metadata does **not enforce** limits: Gufo still bounds the request.
An example `.aider.model.settings.yml` uses the documented extra-params hook:

```yaml
- name: openai/gufo-local
  extra_params:
    max_tokens: 4096
```

Inspect actual outgoing fields from the installed LiteLLM version. Start the
server with `--think off`; if request-level reasoning controls are added,
qualify their translation instead of guessing how LiteLLM forwards them.
Gufo cache flags are optional extensions, not Anthropic cache TTL guarantees.

## Qualification slot

Start Aider in the [prepared fixture repo](QUALIFICATION.md), add
`src/calc.py` and `test_calc.py`, then send the exact read/edit/test prompt.
Use the client's test command or manually approve its proposed test run;
retain its edit transcript, actual patch and executed test output. Aider may
perform text-based edits rather than native function calls; **do not claim
native tool-loop coverage from this run**. Replay the edit/bugfix task with Pi
for the native streamed tool-result contract. Retain both results separately.
