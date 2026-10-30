# Gemma 4 tests

Host-only contracts for the `gemma4` target and its `gemma4-assistant` MTP
drafter. The hosted PR suite runs every test below except those marked
`external-model`, which skip (exit 77) without the library files.

| Test | What it checks |
|---|---|
| `gemma4.config` | Config and weight binding on synthetic GGUFs. |
| `gemma4.artifact_config` | Config and header-only weight binding of the library target and drafter (`external-model`). |
| `gemma4.tokenizer_contract` | Tokenizer semantics on a synthetic vocabulary. |
| `gemma4.tokenizer_goldens` | Token ids of the library vocabulary (`external-model`). |
| `gemma4.template` | Chat template against Jinja2 goldens of the artifact's template. |
| `gemma4.tool_syntax` | Tool-call parsing and the Gemma 4 tool grammar. |
| `gemma4.reference` | The scalar CPU reference (the device runtime's judge) on a tiny synthetic model. |

## Library files

The pinned artifact is `unsloth/gemma-4-26B-A4B-it-GGUF` at revision
`c099eb48e663fd284577b04978a94ffccb261841`. On the development host:

- Files: `/mnt/wd20-backup/models/unsloth__gemma-4-26B-A4B-it-GGUF@c099eb48e663fd284577b04978a94ffccb261841/`
- Manifest with sha256 per file: `/mnt/wd20-backup/models/MANIFESTS/Gemma4-26B-A4B.json`
  (the same records as `MANIFEST.json` in the files directory)

```sh
M=/mnt/wd20-backup/models/unsloth__gemma-4-26B-A4B-it-GGUF@c099eb48e663fd284577b04978a94ffccb261841
GUFO_GEMMA4_MODEL=$M/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
GUFO_GEMMA4_MTP_MODEL=$M/mtp-gemma-4-26B-A4B-it.gguf \
  ctest --preset cpu-test -R '^gemma4\.'
```

The tests read GGUF headers only, never tensor data.

## Fixtures and their generators

Every generator reads only the GGUF metadata section and records the model
file's sha256 from the manifest. None of them are needed to run the tests.

- `fixtures/tokenizer_goldens.json`: `tools/gemma4/tokenizer_goldens.py`, a
  Python port of llama.cpp 6d5ac3c8's gemma4 tokenizer, cross-checked against
  the llama.cpp b11069 ids recorded by upstream PR #426.
- `fixtures/template_goldens.json`: `tools/gemma4/template_goldens.py`, which
  renders `fixtures/template_cases.json` with Jinja2 (Hugging Face settings)
  from the artifact's own template. The template text is not committed.
- `fixtures/reference_tiny.json`: `tools/gemma4/reference_oracle.py`, an
  independent float64 numpy port of llama.cpp 6d5ac3c8's gemma4 graph run on
  the synthetic model that `tiny_model.hpp` specifies (no library file).

```sh
python3 -I tools/gemma4/tokenizer_goldens.py --model $M/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --manifest /mnt/wd20-backup/models/MANIFESTS/Gemma4-26B-A4B.json
PYTHON_WITH_JINJA2 -I tools/gemma4/template_goldens.py --model $M/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --manifest /mnt/wd20-backup/models/MANIFESTS/Gemma4-26B-A4B.json
```

## Known limitations

- Numbers are doubles (`gufo::json`). The tool-call parser rejects a plain
  integer literal beyond ±2^53 and any numeric literal outside the double
  range instead of rounding them or keeping them as text. The guard covers
  plain integer literals only: `9007199254740993.0` or `9007199254740993e0`
  rounds like any other double. A template argument such as `5.0` renders
  as `5`.
- The tool-call parser rejects a key repeated within one object
  (`call:f{a:1,a:2}`); it once kept the last value. This strictness is a
  Gufo choice, not a verified match for another implementation.
- The tool grammar does not adopt upstream #426's fix for array schemas
  without `items` (`6de3b729cb794f0a6060f4730698d9402038b09f`): in best effort
  such an array still opens the whole call's arguments.
- `ParseToolCall` does not check the function name against the declared
  tools. Callers must validate it.
