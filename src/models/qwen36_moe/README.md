# Qwen3.6-35B-A3B (`qwen35moe`)

Loader package for Qwen3.5/3.6 MoE GGUF artifacts. Status: **metadata, loader
map and binder only; no runtime.** The dense Qwen loader rejects `qwen35moe`
with an explicit message.

The trunk is the dense Qwen3.5 layout (Gated DeltaNet on three of every four
layers, gated GQA with per-head `q|gate` on the fourth) with the same tensor
names. Every dense FFN is replaced by:

| Tensor | Shape (GGUF ne) | Role |
|---|---|---|
| `ffn_gate_inp` | [hidden, experts] F32 | router: softmax, top-k, renormalised |
| `ffn_{gate,up}_exps` | [hidden, expert_ff, experts] | routed SwiGLU input |
| `ffn_down_exps` | [expert_ff, hidden, experts] | routed SwiGLU output |
| `ffn_gate_inp_shexp` | [hidden] F32 | scalar sigmoid gate of the shared expert |
| `ffn_{gate,up,down}_shexp` | dense | always-on shared expert |

`Config::TrunkConfig()` maps the artifact to `core::ModelConfig`, so the
shared attention, DeltaNet and state-sizing code can be reused unchanged.
`Config::IsQualifiedGeometry()` marks the one shape (40 layers, hidden 2048,
256 experts, top-8, expert width 512) a future runtime will be qualified for.

The pinned UD-Q4_K_XL table in `tests/models/qwen36_moe/fixtures/` was written
by `tests/models/qwen36_moe/extract_gguf_table.py`, which reads only the GGUF
header. Regenerate it with that script when the pinned artifact changes.
