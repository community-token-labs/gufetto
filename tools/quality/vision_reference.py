#!/usr/bin/env python3
"""Independent CPU scalar F32 full noncausal attention, NOT an encoder teacher.

For R(x) = struct binary32 round-to-nearest/ties-to-even, first R every input
and the scale (default R(1 / sqrt(dim)), with the default computed in binary64).
In increasing dimension/key order, with each accumulator initialized to +0:
  dot[i,j] = fold_d R(acc + R(q[i,d] * k[j,d]))
  s[i,j] = R(dot[i,j] * scale)
  e[i,j] = R(exp(R(s[i,j] - max_j(s[i,j]))))
  z[i] = fold_j R(acc + e[i,j]); p[i,j] = R(e[i,j] / z[i])
  out[i,c] = fold_j R(acc + R(p[i,j] * v[j,c]))
There is no mask, fused multiply-add, BF16 conversion, or BF16 probability
rounding. Python binary64 evaluates each elementary operation before explicit
binary32 rounding; exp uses the host math/libm implementation, not a promise of
correctly rounded transcendental arithmetic on every platform. Underflow is
allowed; nonfinite inputs and overflowing intermediates are rejected.

This mandatory F32 arithmetic oracle intentionally differs from the current
production BF16 score, scaled-score and probability boundaries. Frozen Q/K/V
(including production Q/K/V after RoPE) may be shared standalone operator inputs
for identity/current/candidate comparisons. That does NOT make this a full
encoder reference: a full-encoder teacher must independently run from pixels,
not consume candidate hidden states. This tool must not claim qualification.
No Gufo, production, NumPy, Torch, GPU, or model imports are used.

CLI: python3 -B tools/quality/vision_reference.py input.json [--output new.json]
Input is {"q": [[...]], "k": [[...]], "v": [[...]], "scale": optional_number}.
Use '-' for stdin; omitted --output writes JSON to stdout. Input is limited to
8 MiB; output files are exclusively created, never overwritten.
"""

import argparse
from collections.abc import Sequence
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform
import struct
import sys


DEFAULT_MAX_WORK = 2_000_000
MAX_INPUT_BYTES = 8 * 1024 * 1024
ARITHMETIC = (
    "scalar binary32 via struct pack/unpack; inputs and scale rounded to F32; "
    "default scale=F32(1/sqrt(dim)) computed in binary64; left-to-right F32 "
    "product and add (no FMA); F32 score scaling, max subtraction, math.exp "
    "result, denominator add, probability division, PV product and add; "
    "no BF16 rounding; full noncausal attention"
)
LIMITATION = (
    "Operator-only reference, not full-encoder reference or qualification. "
    "Frozen production Q/K/V may be shared operator inputs. A full-encoder "
    "teacher must independently run from pixels, not candidate hidden states. "
    "F32 deliberately differs from production BF16 score/scaled-score/probability boundaries."
)


def _f32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("value or intermediate overflows finite F32") from exc
    if not math.isfinite(result):
        raise ValueError("value or intermediate must be finite F32")
    return result


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("inputs and scale must be finite real numbers, not booleans")
    try:
        return _f32(float(value))
    except OverflowError as exc:
        raise ValueError("input cannot be represented as finite F32") from exc


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _shape(matrix, name):
    if not _sequence(matrix) or not matrix:
        raise ValueError(f"{name} must be a nonempty rectangular matrix")
    if not _sequence(matrix[0]) or not matrix[0]:
        raise ValueError(f"{name} rows must be nonempty sequences")
    return len(matrix), len(matrix[0])


def attention_f32(q, k, v, scale=None, max_work=DEFAULT_MAX_WORK):
    """Return [queries, value_dim] rows using the formula above.

    ValueError rejects invalid shapes, nonfinite/unrepresentable arithmetic, or
    a non-positive/non-integer budget (bool and integral floats are not ints).
    The budget counts input elements plus output elements plus
    queries * keys * (2*dim + 2*value_dim + 5) rounded scalar operations.
    Shapes and budget are checked before copying/converting numeric values or
    doing attention work. No truncation, mutation, imports of model code, or
    dependence on head count, tile size, or production reduction order.
    """
    if isinstance(max_work, bool) or not isinstance(max_work, int) or max_work <= 0:
        raise ValueError("max_work must be a finite positive integer")
    queries, dim = _shape(q, "q")
    keys, key_dim = _shape(k, "k")
    value_keys, value_dim = _shape(v, "v")
    if dim != key_dim or keys != value_keys:
        raise ValueError("q/k dimensions and k/v key counts must match")
    work = ((queries + keys) * dim + keys * value_dim + queries * value_dim
            + queries * keys * (2 * dim + 2 * value_dim + 5))
    if work > max_work:
        raise ValueError(f"attention work {work} exceeds max_work {max_work}")
    for name, matrix, width in (("q", q, dim), ("k", k, dim), ("v", v, value_dim)):
        if any(not _sequence(row) or len(row) != width for row in matrix):
            raise ValueError(f"{name} must be rectangular with nonempty rows")
    factor = _number(1.0 / math.sqrt(dim) if scale is None else scale)
    q, k, v = ([[_number(x) for x in row] for row in matrix] for matrix in (q, k, v))
    output = []
    for query in q:
        scores = []
        for key in k:
            dot = 0.0
            for a, b in zip(query, key):
                dot = _f32(dot + _f32(a * b))
            scores.append(_f32(dot * factor))
        maximum = max(scores)
        weights = [_f32(math.exp(_f32(score - maximum))) for score in scores]
        total = 0.0
        for weight in weights:
            total = _f32(total + weight)
        probabilities = [_f32(weight / total) for weight in weights]
        row = []
        for column in range(value_dim):
            accumulator = 0.0
            for probability, value in zip(probabilities, v):
                accumulator = _f32(accumulator + _f32(probability * value[column]))
            row.append(accumulator)
        output.append(row)
    return output


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="bounded JSON input file, or '-' for stdin")
    parser.add_argument("--output", type=Path, help="new JSON file (default: stdout)")
    parser.add_argument("--max-work", type=int, default=DEFAULT_MAX_WORK)
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        else:
            with open(args.input, "rb") as stream:
                raw = stream.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise ValueError(f"JSON input exceeds {MAX_INPUT_BYTES} bytes")
        payload = json.loads(raw, parse_constant=_reject_constant,
                             object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or not {"q", "k", "v"} <= payload.keys():
            raise ValueError("JSON input must be an object containing q, k and v")
        if payload.keys() - {"q", "k", "v", "scale"}:
            raise ValueError("JSON input has unknown fields")
        if "scale" in payload and payload["scale"] is None:
            raise ValueError("JSON scale must be a finite number when supplied")
        result = attention_f32(payload["q"], payload["k"], payload["v"],
                               payload.get("scale"), args.max_work)
        document = {
            "output": result,
            "provenance": {
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256(raw).hexdigest(),
                "arithmetic": ARITHMETIC,
                "independence": "stdlib-only scalar formula; no Gufo or production imports",
                "operator_only": True,
                "qualification": False,
                "limitation": LIMITATION,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "max_work": args.max_work,
            },
        }
        rendered = json.dumps(document, allow_nan=False, indent=2) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
    except (ValueError, OSError, RecursionError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
