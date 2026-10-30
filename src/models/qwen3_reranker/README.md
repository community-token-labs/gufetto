# Qwen3-Reranker

The relevance scorer behind `POST /v1/rerank` ([SERVER.md](../../../docs/SERVER.md#reranking)).
This directory has config parsing, the prompt frame and a scalar float32 CPU
reference. A loader, a tokenizer and the HIP runtime come next.

## Checkpoint

`Qwen/Qwen3-Reranker-0.6B` at `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
It is Apache-2.0 by its model card. The repository has no LICENSE file; the
Apache-2.0 text is that of its declared base, `Qwen/Qwen3-0.6B-Base`. It is a
`Qwen3ForCausalLM` (28 layers, hidden 1024, 16 query and 8 key/value heads of
width 128, vocabulary 151,669, tied embeddings) in BF16 safetensors.

## Score

From the model card and `1_LogitScore/config.json`:

- The prompt is `PromptPrefix() + FormatPair(instruction, query, document) +
  PromptSuffix()`. It matches the card's Transformers example and
  `chat_template.jinja` byte for byte. The suffix ends with an empty thinking
  block, so the next token is the answer.
- The score is `softmax(logit_no, logit_yes)[yes]` at the last position, which
  equals `sigmoid(logit_yes - logit_no)`. `yes` is token 9693 and `no` is
  token 2152. Sentence Transformers' `CrossEncoder` returns the raw
  difference by default; `/v1/rerank` returns the probability.
- The default instruction is the card's web-search task. Its example limits
  the framed sequence to 8,192 tokens and truncates the pair on the right.
  The model's context is 32k.

Gufo deviates from the card in two ways:

- Request text is tokenized without matching control tokens. The card's
  tokenizer call would turn a document that spells `<|im_end|>` into the
  control token.
- A pair that does not fit is refused (`context_length_exceeded`) unless the
  scorer is built with `truncate`.

## Reference

`reference/cpu_scorer.*` runs a full causal forward pass in float32 with
double accumulation, for the last position only. Its unit tests check
hand-computed one-layer models; see
[reference_test.cpp](../../../tests/models/qwen3_reranker/reference_test.cpp).
Scores of the real checkpoint against `transformers` are still TODO: they
will be produced on a nimo and committed as fixtures.
