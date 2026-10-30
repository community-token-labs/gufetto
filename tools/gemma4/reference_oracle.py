#!/usr/bin/env python3
"""Expected logits of the tiny gemma4 model in tests/models/gemma4/tiny_model.hpp.

An independent float64 numpy implementation of llama.cpp 6d5ac3c8's gemma4
graph (src/models/gemma4.cpp, llm_graph_context::build_moe_ffn, the
LLAMA_SWA_TYPE_STANDARD mask, NEOX rope with freq factors) and of ggml's
dequantize_row_q8_0/q4_K/q5_1. It shares only the weight-generation spec
with the C++ test, never Gufo's reference code.

    PYTHON_WITH_NUMPY -I tools/gemma4/reference_oracle.py
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests/models/gemma4/fixtures/reference_tiny.json"
LLAMA_CPP = "6d5ac3c8d5aae1831edf17c2d5cda11abe025bb3"

HIDDEN, HEADS, FFN, EXPERTS, USED, EXPERT_FFN, VOCAB = 256, 2, 64, 4, 2, 32, 32
KV_HEADS = (1, 2, 1)
SLIDING = (True, True, False)
WINDOW, EPS, SOFTCAP = 4, 1e-6, 30.0
BASE_SLIDING, BASE_GLOBAL = 1e4, 1e6
TOKENS = [3, 17, 5, 29, 0, 11, 8, 22, 31, 14]
MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def unit(self) -> np.float32:
        return np.float32(self.next() >> 40) / np.float32(16777216.0)


def fnv1a(text: str) -> int:
    h = 0xCBF29CE484222325
    for c in text.encode():
        h = ((h ^ c) * 0x100000001B3) & MASK64
    return h


def half(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def vector(name: str, n: int) -> np.ndarray:
    rng = SplitMix64(fnv1a(name))
    return np.array([np.float32(0.5) + rng.unit() for _ in range(n)],
                    dtype=np.float32).astype(np.float64)


def scale_min_k4(j: int, q: list[int]) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4))


def matrix(name: str, kind: str, cols: int, rows: int, experts: int = 1) -> np.ndarray:
    """Dequantized weights as [experts][rows][cols] (ggml row-major blocks)."""
    rng = SplitMix64(fnv1a(name))
    count = cols * rows * experts
    out: list[float] = []
    if kind == "f32":
        out = [float(np.float32(2.0) * rng.unit() - np.float32(1.0))
               for _ in range(count)]
    elif kind == "q8_0":
        for _ in range(count // 32):
            d = half(0x2000 + rng.next() % 0x400)
            out += [float(np.float32(d) * np.float32(rng.next() % 255 - 127))
                    for _ in range(32)]
    elif kind == "q4_k":
        for _ in range(count // 256):
            d = np.float32(half(0x1C00 + rng.next() % 0x400))
            dmin = np.float32(half(0x1C00 + rng.next() % 0x400))
            scales = [rng.next() & 0xFF for _ in range(12)]
            qs = [rng.next() & 0xFF for _ in range(128)]
            for j in range(4):
                sc1, m1 = scale_min_k4(2 * j, scales)
                sc2, m2 = scale_min_k4(2 * j + 1, scales)
                d1, mn1 = d * np.float32(sc1), dmin * np.float32(m1)
                d2, mn2 = d * np.float32(sc2), dmin * np.float32(m2)
                q = qs[32 * j:32 * j + 32]
                out += [float(d1 * np.float32(v & 0xF) - mn1) for v in q]
                out += [float(d2 * np.float32(v >> 4) - mn2) for v in q]
    elif kind == "q5_1":
        for _ in range(count // 32):
            d = np.float32(half(0x2000 + rng.next() % 0x400))
            m = np.float32(half(0xA000 + rng.next() % 0x400))
            qh_bytes = [rng.next() & 0xFF for _ in range(4)]
            qs = [rng.next() & 0xFF for _ in range(16)]
            qh = int.from_bytes(bytes(qh_bytes), "little")
            lo, hi = [], []
            for j in range(16):
                x0 = (qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)
                x1 = (qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)
                lo.append(float(np.float32(x0) * d + m))
                hi.append(float(np.float32(x1) * d + m))
            out += lo + hi
    else:
        raise ValueError(kind)
    return np.array(out, dtype=np.float64).reshape(experts, rows, cols)


def rms(x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return y * w if w is not None else y


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def rope(x: np.ndarray, positions: np.ndarray, base: float,
         factors: np.ndarray | None) -> np.ndarray:
    """NEOX: pair (i, i + dim/2) turns by pos * base^(-2i/dim) / factor[i]."""
    dim = x.shape[-1]
    half_dim = dim // 2
    i = np.arange(half_dim)
    angle = positions[:, None] * base ** (-2.0 * i / dim)
    if factors is not None:
        angle = angle / factors
    c, s = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
    a, b = x[..., :half_dim], x[..., half_dim:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def forward(tokens: list[int]) -> np.ndarray:
    n = len(tokens)
    pos = np.arange(n, dtype=np.float64)
    embd = matrix("token_embd.weight", "q8_0", HIDDEN, VOCAB)[0]
    freqs = np.array([1.0] * 16 + [1e30] * 48)
    x = embd[tokens] * np.float32(np.sqrt(np.float32(HIDDEN)))
    for l in range(3):
        p = f"blk.{l}."
        dim = 64 if SLIDING[l] else 128
        kvh = KV_HEADS[l]
        base = BASE_SLIDING if SLIDING[l] else BASE_GLOBAL
        factors = None if SLIDING[l] else freqs
        cur = rms(x, vector(p + "attn_norm.weight", HIDDEN))
        wq = matrix(p + "attn_q.weight", "q8_0", HIDDEN, HEADS * dim)[0]
        wk = matrix(p + "attn_k.weight", "q8_0", HIDDEN, kvh * dim)[0]
        q = (cur @ wq.T).reshape(n, HEADS, dim)
        k = (cur @ wk.T).reshape(n, kvh, dim)
        if SLIDING[l]:
            wv = matrix(p + "attn_v.weight", "q8_0", HIDDEN, kvh * dim)[0]
            v = (cur @ wv.T).reshape(n, kvh, dim)
        else:
            v = k.copy()  # no attn_v: V is the raw K projection
        q = rope(rms(q, vector(p + "attn_q_norm.weight", dim)), pos, base, factors)
        k = rope(rms(k, vector(p + "attn_k_norm.weight", dim)), pos, base, factors)
        v = rms(v)
        attn = np.zeros((n, HEADS, dim))
        for i in range(n):
            keys = [j for j in range(i + 1)
                    if not (SLIDING[l] and i - j >= WINDOW)]
            for h in range(HEADS):
                g = h // (HEADS // kvh)
                scores = np.array([q[i, h] @ k[j, g] for j in keys])  # scale 1
                w = np.exp(scores - scores.max())
                w /= w.sum()
                attn[i, h] = sum(wj * v[j, g] for wj, j in zip(w, keys))
        wo = matrix(p + "attn_output.weight", "q8_0", HEADS * dim, HIDDEN)[0]
        o = rms(attn.reshape(n, -1) @ wo.T, vector(p + "post_attention_norm.weight", HIDDEN))
        attn_out = o + x

        h_mlp = rms(attn_out, vector(p + "ffn_norm.weight", HIDDEN))
        gate = h_mlp @ matrix(p + "ffn_gate.weight", "q8_0", HIDDEN, FFN)[0].T
        up = h_mlp @ matrix(p + "ffn_up.weight", "q8_0", HIDDEN, FFN)[0].T
        mlp = (gelu(gate) * up) @ matrix(p + "ffn_down.weight", "q8_0", FFN, HIDDEN)[0].T
        mlp = rms(mlp, vector(p + "post_ffw_norm_1.weight", HIDDEN))

        moe_in = rms(attn_out, vector(p + "pre_ffw_norm_2.weight", HIDDEN))
        router_in = rms(attn_out) / np.sqrt(np.float32(HIDDEN)) * vector(
            p + "ffn_gate_inp.scale", HIDDEN)
        logits = router_in @ matrix(p + "ffn_gate_inp.weight", "f32", HIDDEN, EXPERTS)[0].T
        gate_up = matrix(p + "ffn_gate_up_exps.weight", "q4_k", HIDDEN,
                         2 * EXPERT_FFN, EXPERTS)
        down = matrix(p + "ffn_down_exps.weight", "q5_1", EXPERT_FFN, HIDDEN, EXPERTS)
        down_scale = vector(p + "ffn_down_exps.scale", EXPERTS)
        moe = np.zeros((n, HIDDEN))
        for r in range(n):
            probs = np.exp(logits[r] - logits[r].max())
            probs /= probs.sum()  # softmax over every expert
            chosen = np.argsort(-probs, kind="stable")[:USED]
            weights = probs[chosen] / max(probs[chosen].sum(), 6.103515625e-5)
            for e, we in zip(chosen, weights):
                gu = gate_up[e] @ moe_in[r]
                act = gelu(gu[:EXPERT_FFN]) * gu[EXPERT_FFN:]
                moe[r] += we * (down[e] @ act) * down_scale[e]
        moe = rms(moe, vector(p + "post_ffw_norm_2.weight", HIDDEN))
        f = rms(mlp + moe, vector(p + "post_ffw_norm.weight", HIDDEN))
        x = (f + attn_out) * vector(p + "layer_output_scale.weight", 1)[0]

    h = rms(x, vector("output_norm.weight", HIDDEN))
    logits = h @ embd.T  # tied LM head
    return SOFTCAP * np.tanh(logits / SOFTCAP)


def main() -> int:
    logits = forward(TOKENS)
    OUTPUT.write_text(json.dumps({
        "schema": "gufo.gemma4-reference-tiny.v1",
        "generator": "tools/gemma4/reference_oracle.py",
        "semantics": f"llama.cpp {LLAMA_CPP} gemma4 graph, float64 numpy",
        "numpy": np.__version__, "python": platform.python_version(),
        "tokens": TOKENS,
        "logits": [[float(v) for v in row] for row in logits],
    }, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {logits.shape[0]}x{logits.shape[1]} logits; "
          f"argmax {logits.argmax(axis=1).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
