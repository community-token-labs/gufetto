#!/usr/bin/env python3
"""Replay a MoE routing trace through the expert-stream policy and project
streamed decode speed. A projection, not a speed claim.

Routes come from an int16/int32 `[layers, tokens, top_k]` .npy/.npz (for
example the DS4.1 oracle routes) or a deterministic synthetic Zipf trace;
NumPy is not required. The policy itself runs in the C++ replay driver
(`expert_stream_replay`), so the numbers come from the product code.

Schedules:
  gate  tokens [0, prefill) as one prefill batch, the rest as single-token
        decode steps (the A1 pre-registered workload);
  scan  decode, a prefill scan, decode again (LRU vs scan-resistant check).

Projection per decode token (no read/compute overlap credit):
  t = t_resident + sum over layers of (miss_bytes / bandwidth + latency if the
  layer missed), reported as tokens / total time.
"""
import argparse
import ast
import json
import math
import random
import re
import subprocess
import sys
import zipfile
from pathlib import Path

MiB, GiB = 1 << 20, 1 << 30
EXPERTS = 384


def load_routes(path):
    """Minimal .npy/.npz reader: little-endian int16/int32, C order."""
    data = Path(path).read_bytes()
    if data[:2] == b"PK":
        with zipfile.ZipFile(path) as z:
            data = z.read(z.namelist()[0])
    if data[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy array")
    major = data[6]
    hlen_size = 2 if major == 1 else 4
    hlen = int.from_bytes(data[8:8 + hlen_size], "little")
    header = ast.literal_eval(data[8 + hlen_size:8 + hlen_size + hlen].decode())
    if header["fortran_order"] or header["descr"] not in ("<i2", "<i4"):
        raise ValueError(f"unsupported array {header}")
    width = int(header["descr"][2])
    layers, tokens, top_k = header["shape"]
    body = memoryview(data)[8 + hlen_size + hlen:]
    values = body.cast("h" if width == 2 else "i")
    return [[list(values[(l * tokens + t) * top_k:(l * tokens + t + 1) * top_k])
             for t in range(tokens)] for l in range(layers)]


def synthetic_routes(layers, tokens, top_k, seed):
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** 0.9 for i in range(EXPERTS)]
    perms = [rng.sample(range(EXPERTS), EXPERTS) for _ in range(layers)]
    out = []
    for l in range(layers):
        rows = []
        for _ in range(tokens):
            picked = set()
            while len(picked) < top_k:
                picked.add(perms[l][rng.choices(range(EXPERTS), weights)[0]])
            rows.append(sorted(picked))
        out.append(rows)
    return out


def layer_sizes(layers, mean_bytes, mixed):
    """Encoded expert bytes per layer: three equal 256-byte-aligned projections."""
    if mixed:  # eight classes spanning the donor's 7.38-9.76 MiB, same mean
        spread = [7.38 + i * (9.76 - 7.38) / 7 for i in range(8)]
        scale = mean_bytes / (sum(spread) / 8 * MiB)
        raw = [spread[l % 8] * MiB * scale for l in range(layers)]
    else:
        raw = [mean_bytes] * layers
    return [3 * max(256, round(r / 3 / 256) * 256) for r in raw]


def read_ranking(path, layers):
    """Donor expert_rank.inc: comma-separated layer * 384 + expert values."""
    text = re.sub(r"/\*.*?\*/", "", Path(path).read_text(), flags=re.S)
    ranked = [int(v) for v in re.findall(r"\d+", text)]
    return [(v // EXPERTS, v % EXPERTS) for v in ranked if v // EXPERTS < layers]


class Trace:
    def __init__(self, routes, sizes):
        self.routes, self.sizes = routes, sizes
        self.base, at = [], 0
        for size in sizes:
            self.base.append(at)
            at += size * EXPERTS

    def line(self, step, phase, layer, expert):
        proj = self.sizes[layer] // 3
        offs = [self.base[layer] + p * proj * EXPERTS + expert * proj for p in range(3)]
        return f"{step} {phase} {layer} {expert} 0 " + " ".join(f"{o} {proj}" for o in offs)


def build_input(trace, args, cap, policy, segments, seed_rank):
    sizes = trace.sizes
    classes = sorted(set(sizes))
    lines = [f"config {policy} {cap} {128 * MiB} {EXPERTS * max(sizes)} {32 * MiB} "
             + ",".join(map(str, classes))]
    if seed_rank:
        budget, used = cap // 16 * 15, 0
        for layer, expert in seed_rank:
            used += sizes[layer]
            if used > budget:
                break
            lines += ["plan 1 1", trace.line(0, "d", layer, expert)]
        lines.append("reset")
    for kind, first, last in segments:
        if kind == "p":
            for layer, rows in enumerate(trace.routes):
                plan = [trace.line(t, "p", layer, e)
                        for t in range(first, last) for e in rows[t] if e >= 0]
                if plan:
                    lines += [f"plan 0 {len(plan)}"] + plan
        else:
            for t in range(first, last):
                for layer, rows in enumerate(trace.routes):
                    plan = [trace.line(t, "d", layer, e) for e in rows[t] if e >= 0]
                    if plan:
                        lines += [f"plan 0 {len(plan)}"] + plan
    return "\n".join(lines) + "\n"


def replay(binary, text):
    proc = subprocess.run([binary], input=text, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"replay failed: {proc.stderr.strip()}")
    plans, stats = [], None
    for line in proc.stdout.splitlines():
        f = line.split()
        if f[0] == "plan":
            plans.append(dict(zip(("layer", "step", "hits", "misses", "demand", "optional",
                                   "reads", "deferred", "staged"), map(int, f[1:]))))
        elif f[0] == "stats":
            stats = dict(zip(("allocated", "peak_allocated", "resident_payload",
                              "fragmentation", "evictions"), map(int, f[1:])))
    return plans, stats


def summarize(plans, segments, layers, bucket):
    out = []
    for kind, first, last in segments:
        sel = [p for p in plans if first <= p["step"] < last]
        hits, misses = sum(p["hits"] for p in sel), sum(p["misses"] for p in sel)
        out.append({"phase": "prefill" if kind == "p" else "decode", "tokens": [first, last],
                    "route_hits": hits, "route_misses": misses,
                    "hit_rate": hits / (hits + misses) if hits + misses else None,
                    "demand_bytes": sum(p["demand"] for p in sel),
                    "read_tasks": sum(p["reads"] for p in sel),
                    "staged": sum(p["staged"] for p in sel),
                    "deferred": sum(p["deferred"] for p in sel)})
    steps = max(p["step"] for p in plans) + 1 if plans else 0
    heat = [[0] * ((steps + bucket - 1) // bucket) for _ in range(layers)]
    for p in plans:
        heat[p["layer"]][p["step"] // bucket] += p["misses"]
    return out, heat


def project(plans, first, last, t_res, bandwidth, latency):
    per_token = {}
    for p in plans:
        if first <= p["step"] < last:
            t = per_token.setdefault(p["step"], t_res)
            per_token[p["step"]] = t + p["demand"] / bandwidth + (latency if p["misses"] else 0)
    total = sum(per_token.values())
    return {"tokens": len(per_token), "mean_ms": 1e3 * total / len(per_token),
            "tok_per_s": len(per_token) / total}


def ascii_heat(heat, max_value):
    shades = " .:-=+*#%@"
    return ["L%02d |" % l + "".join(
        shades[min(9, math.ceil(9 * v / max_value))] if v else " " for v in row) + "|"
        for l, row in enumerate(heat)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True, help="expert_stream_replay binary")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--routes", help="[layers, tokens, top_k] .npy/.npz")
    src.add_argument("--synthetic", type=int, metavar="TOKENS")
    ap.add_argument("--schedule", choices=("gate", "scan"), default="gate")
    ap.add_argument("--prefill", type=int, default=2048, help="gate: prefill tokens")
    ap.add_argument("--scan", default="500,2048", help="scan: decode tokens,scan tokens")
    ap.add_argument("--policy", choices=("lru", "scored"), action="append")
    ap.add_argument("--cap-gib", type=float, action="append")
    ap.add_argument("--seed-rank", help="ranking (donor expert_rank.inc) to seed 15/16 of cap")
    ap.add_argument("--expert-mib", type=float, default=8.20)
    ap.add_argument("--mixed-sizes", action="store_true")
    ap.add_argument("--t-res-ms", type=float, default=43.0)
    ap.add_argument("--read-model", help="fio summary.json (gating when qualified)")
    ap.add_argument("--bracket", action="append", metavar="GIBPS,MS",
                    help="non-gating bandwidth/latency bracket")
    ap.add_argument("--bucket", type=int, default=64, help="heatmap tokens per column")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    routes = (load_routes(args.routes) if args.routes
              else synthetic_routes(40, args.synthetic, 6, 1))
    layers, tokens = len(routes), len(routes[0])
    trace = Trace(routes, layer_sizes(layers, args.expert_mib * MiB, args.mixed_sizes))
    if args.schedule == "gate":
        segments = [("p", 0, args.prefill), ("d", args.prefill, tokens)]
    else:
        d, s = map(int, args.scan.split(","))
        segments = [("d", 0, d), ("p", d, d + s), ("d", d + s, tokens)]
    rank = read_ranking(args.seed_rank, layers) if args.seed_rank else None

    models = []
    if args.read_model:
        summary = json.loads(Path(args.read_model).read_text())
        if summary.get("read_model"):
            m = summary["read_model"]
            models.append({"name": f"measured {m['source']}", "gating": True,
                           "bandwidth": m["bandwidth_bytes_per_s"], "latency": m["latency_p50_s"]})
    for b in args.bracket or []:
        gibps, ms = map(float, b.split(","))
        models.append({"name": f"bracket {gibps} GiB/s {ms} ms", "gating": False,
                       "bandwidth": gibps * GiB, "latency": ms / 1e3})

    report = {"trace": args.routes or f"synthetic:{args.synthetic}", "layers": layers,
              "tokens": tokens, "schedule": segments, "expert_bytes": trace.sizes,
              "zero_hit_bytes_per_token": sum(trace.sizes) * len(routes[0][0]),
              "seeded": bool(rank), "t_res_ms": args.t_res_ms, "runs": []}
    for policy in args.policy or ["scored"]:
        for cap_gib in args.cap_gib or [64.0]:
            text = build_input(trace, args, int(cap_gib * GiB), policy, segments, rank)
            plans, stats = replay(args.replay, text)
            segs, heat = summarize(plans, segments, layers, args.bucket)
            run = {"policy": policy, "cap_gib": cap_gib, "segments": segs,
                   "allocation": stats, "heatmap_misses": heat, "projection": []}
            _, first, last = segments[-1]
            for m in models:
                run["projection"].append({"read_model": m["name"], "gating": m["gating"],
                    **project(plans, first, last, args.t_res_ms / 1e3,
                              m["bandwidth"], m["latency"])})
            report["runs"].append(run)
            print(f"== {policy} cap {cap_gib:g} GiB seeded={bool(rank)}")
            for s in segs:
                print(f"  {s['phase']:7} {s['tokens']}: hit {100 * (s['hit_rate'] or 0):.2f}% "
                      f"demand {s['demand_bytes'] / GiB:.2f} GiB reads {s['read_tasks']} "
                      f"staged {s['staged']} deferred {s['deferred']}")
            print(f"  alloc {stats['allocated'] / GiB:.2f} GiB peak "
                  f"{stats['peak_allocated'] / GiB:.2f} payload "
                  f"{stats['resident_payload'] / GiB:.2f} frag "
                  f"{stats['fragmentation'] / MiB:.1f} MiB evictions {stats['evictions']}")
            for p in run["projection"]:
                print(f"  {p['read_model']:32} {p['tok_per_s']:6.2f} tok/s "
                      f"({p['mean_ms']:.1f} ms){'  [gating]' if p['gating'] else ''}")
            peak = max(max(r) for r in heat) or 1
            print("\n".join("  " + r for r in ascii_heat(heat, peak)))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
