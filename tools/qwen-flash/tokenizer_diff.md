# Qwen release tokenizer metadata diff

From the repository root, offline (default):

```sh
python3 tools/qwen-flash/tokenizer_diff.py /path/to/new-hf-metadata > tokenizer-diff.json
```

The local directory must contain `tokenizer.json`, `tokenizer_config.json`, and
at least one chat template (the config's `chat_template` or `chat_template.jinja`).
Keep both template sources when the repository supplies both.

Explicit public HF metadata access, **never weights**:

```sh
python3 tools/qwen-flash/tokenizer_diff.py 'OWNER/REPO@REVISION' --allow-network > tokenizer-diff.json
```

The revision is resolved to an immutable HF commit first and recorded in output.
Only the three named metadata files are fetched (32 MiB per file limit, timeout).
No login, token, Transformers, remote code, model instantiation or disk cache.
Read-only network is optional and disabled unless `--allow-network` is supplied.

Exit codes: **0** semantic fingerprints/control IDs match; **1** changes require
review; **2** incomplete/invalid input or fetch failure. Do not treat exit 1 as a
crash or automatically enable support after exit 0.

## Independent pinned reference

`qwen38_tokenizer_reference.json` records Qwen/Qwen3.8-Flash-Next at
[`de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540).
It contains original-file SHA256s, canonical JSON section SHA256s, chat-template
SHA256s and independently captured control spellings/IDs. It is **not generated
from Gufo's C++ or Python control constants**. No model vocabulary, merges, or
original template source is vendored. The upstream model card declares `other`
license; consult the upstream license before redistributing upstream content.

Canonical fingerprints use Python JSON with sorted keys, UTF-8, no whitespace,
and `ensure_ascii=False`. Tokenizer `model` is split into its immediate fields
(vocab, merges, etc.); other tokenizer/config fields each get a fingerprint.
Thus vocabulary/merge changes are detected without a giant lexical diff, while
JSON whitespace/key order alone do not count. Templates preserve their bytes
(after UTF-8 decoding; config template objects/lists are canonicalized), so
whitespace changes in a template count. Reports show changed section/template
keys and before/after hashes, not a full source patch. Original-file hashes are
provenance, not semantic equality criteria.

Controls come from added tokens in either JSON, plus control-looking vocabulary
entries (including think/tool markers). Conflicting IDs across these sources are
an input error. `control_changes` lists additions, removals and ID changes,
annotated with membership in `src/models/qwen/control_tokens.hpp`. The header
owns **spellings, not IDs**: the pinned reference owns expected IDs. Existing
reference controls outside that shared framing catalogue are not automatically
errors, but changes to them are flagged. Shared catalogue entries absent from
the reference are reported separately, not falsely asserted to have known IDs.

`--reference PATH` accepts the same schema-1 snapshot format for controlled
comparisons; retain its independent provenance. `--control-tokens PATH` selects
the header to inspect. Tests use tiny independent synthetic JSON documents.

```sh
python3 tests/models/qwen38_flash_next/tokenizer_diff_test.py
ctest --preset cpu-test -R '^qwen38_flash_next.tokenizer_diff$'
```

## Day-0 review boundary

Inspect every changed hash against the pinned and candidate metadata. Check
rendered chat bytes/tool framing/thinking behavior with independent goldens;
this fingerprint tool does **not** execute Jinja, tokenize prompts, establish
GGUF converter fidelity, or qualify inference. Config/shape support, real
weights, numerical quality, continuation/cache, and matched-stack measurements
remain separate gates.
