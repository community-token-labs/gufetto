#!/usr/bin/env python3
"""Bounded 4K-image key-count scalar fixture; NOT a full-encoder test.

One selected query, one head, dim72, all65536 keys. No model or image data.
Kept outside hosted CI: invoke explicitly at nice19, once for the receipt.
"""
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality"))
from vision_reference import attention_f32


def main():
    started = time.monotonic()
    keys, dim = 65536, 72
    scale = [float(1 << (d % 4)) for d in range(dim)]
    q = [[0.0] * dim]
    k = [[float((d % 5) - 2) for d in range(dim)]] * keys
    # q=0 gives exact uniform probability2^-16. Adjacent +/- terms cancel
    # exactly in F32. Only the final key leaves half a positive scale term.
    # This checks that the last key participates, not merely a zero result.
    positive, negative = scale, [-x for x in scale]
    v = [positive if i % 2 == 0 else negative for i in range(keys)]
    v[-1] = [-x / 2 for x in scale]
    result = attention_f32(q, k, v, max_work=30_000_000)
    expected = [[math.ldexp(x, -17) for x in scale]]
    if result != expected:
        raise AssertionError("65536-key scalar fixture disagrees with analytic uniform mean")
    raw = struct.pack("<72f", *result[0])
    print(json.dumps(dict(status="PASS", scope="scalar_attention_operator_only",
        queries=1, keys=keys, head_dim=dim, heads=1, full_keys=True,
        expected="scale[d] * 2^-17; scale[d]=2^(d mod4)",
        output_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        oracle_sha256=hashlib.sha256((ROOT / "tools/quality/vision_reference.py").read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic() - started,
        linux_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        full_encoder_qualified=False,
        downstream_qualified=False, kernel_qualified=False), sort_keys=True))


if __name__ == "__main__":
    main()
