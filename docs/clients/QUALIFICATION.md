# Client qualification procedure (model-slot work, not CPU fixture acceptance)

No GPU or real-client run was performed to author these recipes. Run this only
inside the affected model's allocated qualification slot, with the coordinator's
selected modes/settings. Keep numerical quality tests and the standard speed
benchmark; client success does not replace them. Never treat a missing-model
skip or a mocked SSE fixture as model quality acceptance.

## 1. Pin and prepare

Record candidate/main SHAs, binary SHA256, exact compiler/loaded HIP/ROCr or Nix
stack, all model/draft/projector/tokenizer hashes, client + Node/Python versions,
harness revision, full server flags and loaded mode. Use the candidate for
correctness; main is for matched affected timing controls, not duplicate full
correctness. Save every result under a new ignored `artifacts/` directory.
Use [benchmark-card metadata](../BENCHMARK-CARDS.md) for the resulting card.

Start the production server using that model's documented command, loopback
only, informational log retained, actual context `C` and output `O < C`.
The recipe examples use C=32768/O=4096 only if that capacity was actually loaded.
Check `/v1/models` using the single bearer key, including context and modalities.
Disable all unqualified hosted tools, extra providers and automatic cloud
fallback. Run a bearer-auth smoke request before giving the client tools.
Do not log auth headers. Never put client credentials in argv, project config
or git. Gufo's current server `--api-key` option does expose its key in server
argv: see the [known server credential caveat](README.md); there is no env/file
alternative at this source revision. Plain HTTP on a LAN exposes bearer keys;
use a TLS reverse proxy for LAN access, not an unauthenticated bind.

## 2. Exact bounded task for each client's native run

Create a fresh disposable Git repository **for each client/run** containing:

`src/calc.py`:

```python
def add(a, b):
    return a - b
```

`test_calc.py`:

```python
import unittest
from src.calc import add

class AddTest(unittest.TestCase):
    def test_values(self):
        for a, b in [(2, 3), (-2, 3), (0, 0), (5, -8)]:
            self.assertEqual(add(a, b), a + b)

if __name__ == "__main__":
    unittest.main()
```

Task, identical for Claude Code, OpenCode, Cline, Aider, Open WebUI and interactive Pi:

```text
Read src/calc.py and test_calc.py using your file tools. Fix add without changing
or weakening tests. Run python3 -m unittest discover -s . -p test_calc.py using
your command tool. Read the modified file back. Report the actual test result.
Do not install packages, use network, spawn agents or modify other files.
```

Launch/select the provider exactly as on that client's recipe page. Approve only
fixture-scoped operations. For Aider add both files to its editable context; for
Cline use Act mode; for WebUI enable only a reviewed fixture-scoped executor
and Native calling. If that executor is absent, mark WebUI tool execution
BLOCKED, not PASS. Do not paste fabricated tool results as a substitute.

Retain the client task/session export, raw request bodies/SSE without credentials,
actual read/edit/command outputs and exit status, final diff, and server log.
Execute the unittest command independently, verify it passes and that the tests
were not edited. Inspect actual tools, not just the model's final sentence.
Check at least one second model request containing the real prior call IDs and
results, terminal events, usage and expected cache/prefill work. Save errors,
truncations, unsupported parameters and compaction/capacity mismatches verbatim.
For Claude Code, confirm every auxiliary/background request uses a served
model and the server log contains no `model_not_found`; verify actual output
limits as well as the main conversation's model. Record protocol mode (native
calls versus text/XML edit flow); these aren't
interchangeable coverage. Start a fresh session and then a retained-history
follow-up; use the same files and request ordering for matched timing controls.

Existing `claude_code_agent.py` and `opencode_agent.py` can provide additional
version-pinned automation, but their historical defaults are not automatically
this recipe: inspect their limits, auth and exact tasks first. In particular,
OpenCode's existing harness advertises 262144 context/16384 output, so it is
not a valid smaller-context recipe without a separately reviewed configuration.
Do not silently reuse those limits for an arbitrary model.

## 3. Mandatory real Pi replay of the affected coding loop

After **each client's affected native workflow**, run the matching Pi edit/bugfix
workflow against the same candidate/model/mode; give each client and API its own
fresh output directory. This is paired evidence, not proof that Pi emulates the
other client's payloads. Pi's maintained harness pins version **0.87.0**; install
or select it through the slot's approved setup, not by removing its version gate.
The 1.4.0 interactive recipe is separately versioned.

```sh
export GUFO_API_KEY  # already supplied privately; no secret on argv
python3 tests/functional/pi_agent.py \
  --base-url http://127.0.0.1:8080 --model gufo-local \
  --pi /path/to/pi-0.87.0 --server-log "$SERVER_LOG" \
  --context-window "$C" --max-output-tokens "$O" \
  --api openai-completions --thinking off \
  --case edit --case bugfix --passes 1 --conversation \
  --output "artifacts/client-qualification/$CLIENT/pi-chat-candidate-1"
```

The harness now accepts explicit loaded-model budgets; historical defaults
262144/8192 are retained for existing callers. **Always pass the actual C/O
for these recipes.** `GUFO_API_KEY` is forwarded as bearer upstream only, not
recorded in arguments or headers on disk. Client config retains a dummy local
proxy key, not the server credential. Images are not exercised by these cases.

For a client using Responses (including the alternate Pi profile), repeat only
the affected replay with `--api openai-responses` and a fresh output directory.
For Anthropic/Claude Code, retain the native Messages transcript as well as the
Pi Chat replay; Pi replay alone does not qualify Messages. Additional targeted
`--case tools`, `--case literal-protocol` or `--case wide-tool` is appropriate
when those contracts are affected; do not run the default full task sweep.

Inspect `report.json`, retained `session.jsonl`, each task's `result.json`, tool
errors/actual results, and `wire/request-*.json`, `.sse`, `.timing.json`. Verify
actual loaded draft mode and nonzero drafts where expected, completion framing,
output budget, generated files and independently executed tests. Retain the
actual server log and cache history; a request that does unexpected prefill
fails the work contract even when its answer is correct.

## 4. Correctness, timing and outcome

Select the affected model/modes explicitly in `tests/functional/run.py` (see
[functional test guide](../../tests/functional/README.md)); include `long-context`
and `cache` for continuation changes. Use all four profiles only for shared
behavior changes. This documentation/fixture change does not itself authorize
model sweeps, video generation, GPU access or a qualification slot.

Timings remain mandatory **per request and phase** at 5% and 10 ms, never
averaged across requests. Run once, investigate flags, then alternate main and
candidate only for affected histories; keep unchanged-main controls when needed
and retain every result. Compare only identical toolchain, harness, artifacts,
settings, model mode and cache history. Fix confirmed regressions before
publishing. Noisy evidence stays INCONCLUSIVE/unqualified; don't widen margins.

Record each combination as PASS / FAIL / INCONCLUSIVE / BLOCKED with its exact
reason and receipt links. A client-native pass, Pi replay pass, numerical quality
pass and speed evidence are separate columns in the qualification receipt.
Do not label all six clients "tested" based only on the CPU route contracts.
