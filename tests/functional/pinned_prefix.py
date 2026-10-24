#!/usr/bin/env python3
"""Probe an already-running pin-enabled server, then rerun after restart.

Uses raw /completion prompts, so FILEs must be the same rendered text supplied
at server startup. Repeat --prefix-file and --prefix-tokens in matching order.
This runner never starts/stops a server or deletes caches. Greedy output must
equal cold prefill; seeded output must replay in RAM and across --restore.
A sampled cold split is not used as an exactness oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def call(url, payload):
    start = time.monotonic()
    request = Request(url.rstrip("/").removesuffix("/v1") + "/completion",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        result = json.load(response)
        server_timing = response.headers.get("Server-Timing", "")
    timing = result["timings"]
    ttft_ms = next((float(item.split("dur=", 1)[1]) for item in server_timing.split(",")
                    if item.strip().startswith("ttft;dur=")), None)
    result["probe"] = {"wall_s": time.monotonic() - start,
                       "ttft_ms": ttft_ms,
                       "prefill_tokens": timing["prompt_n"],
                       "cached_tokens": result["tokens_cached"],
                       "disk": timing.get("component_cache", {}).get("reuse") == "disk"}
    assert result["tokens_evaluated"] == (result["tokens_cached"] +
                                          timing["prompt_n"]), result
    return result


def answer(result):
    return result["content"], result["tokens_predicted"]


def identity(registration):
    return registration["prefix_sha256"], registration["prefix_tokens"]


def probe_registration(args, prefix, report, previous):
    def probe(label, tail, temperature, cold=False):
        payload = {"prompt": prefix + tail, "n_predict": 32,
                   "temperature": temperature, "seed": 31337,
                   "top_k": 20, "top_p": 0.95, "min_p": 0,
                   "cache_prompt": not cold}
        result = call(args.url, payload)
        report["cases"][label] = result
        if cold:
            assert result["tokens_cached"] == 0, result
        else:
            depth = report["prefix_tokens"]
            if args.exact_prefix and label.endswith("_warm"):
                assert result["tokens_cached"] == depth, (
                    "expected exact longest registration (use a fresh ordinary cache)",
                    depth, result)
            assert result["tokens_cached"] >= depth, result
            assert result["probe"]["prefill_tokens"] <= (result["tokens_evaluated"] -
                                                          depth), result
        return result

    for name, temperature in (("greedy", 0), ("sampled", 0.7)):
        tail = ("<|im_start|>user\n" + name +
                ": Name three prime numbers, with no explanation."
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        warm = probe(name + "_warm", tail, temperature)
        if previous:
            assert answer(warm) == answer(previous["cases"][name + "_warm"]), (
                "restart replay differs", name, warm, previous["cases"])
        retry = probe(name + "_ram_replay", tail, temperature)
        assert answer(warm) == answer(retry), ("RAM replay differs", name, warm, retry)
        if name == "greedy":
            cold = probe(name + "_cold", tail, temperature, cold=True)
            assert answer(warm) == answer(cold), ("cold greedy differs", warm, cold)
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix-file", required=True, type=Path, action="append")
    parser.add_argument("--prefix-tokens", required=True, type=int, action="append",
                        help="exact token depth from each pin startup log")
    parser.add_argument("--exact-prefix", action="store_true",
                        help="require initial probes to select exactly each supplied depth; "
                             "needs fresh ordinary cache/slots, including after restart")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--restore", type=Path,
                        help="report written before an independently managed restart")
    parser.add_argument("--churn", type=int, default=0,
                        help="optional unrelated bounded prompts between pin probes")
    args = parser.parse_args()
    if (len(args.prefix_file) != len(args.prefix_tokens) or
            not 1 <= len(args.prefix_file) <= 64 or
            any(depth <= 0 for depth in args.prefix_tokens) or
            not 0 <= args.churn <= 140):
        parser.error("1..64 matching file/depth pairs, positive depths and churn in 0..140 required")
    if args.output.exists():
        parser.error("output already exists")
    prefixes, registrations = [], []
    for path, depth in zip(args.prefix_file, args.prefix_tokens):
        raw = path.read_bytes()
        prefixes.append(raw.decode("utf-8"))
        registrations.append({"prefix_sha256": hashlib.sha256(raw).hexdigest(),
                              "prefix_tokens": depth, "cases": {}, "status": "running"})
    keys = [identity(registration) for registration in registrations]
    if len(set(keys)) != len(keys):
        parser.error("duplicate prefix file/depth identity")
    previous = json.loads(args.restore.read_text()) if args.restore else None
    previous_by_key = {}
    if previous:
        if previous.get("status") != "passed":
            parser.error("restore report did not pass")
        old = previous.get("registrations", [previous])
        previous_by_key = {identity(registration): registration for registration in old}
        if len(previous_by_key) != len(old) or set(previous_by_key) != set(keys):
            parser.error("restore report belongs to different registered prefixes")
    # Preserve the original single-pin report shape and restore compatibility.
    if len(registrations) == 1:
        report = registrations[0]
        report.update(schema="gufo-pinned-prefix-probe-v1", restore=bool(previous))
    else:
        report = {"schema": "gufo-pinned-prefix-probe-v2", "restore": bool(previous),
                  "registrations": registrations, "status": "running"}
    try:
        for index in range(args.churn):
            # Distinct from registrations at token zero; explicitly bounded.
            call(args.url, {"prompt": f"Unrelated fixture {index}: " +
                                     (f"record_{index} has a field. " * 80) +
                                     "Reply with OK.",
                            "n_predict": 1, "temperature": 0,
                            "cache_prompt": True})
        for prefix, registration in zip(prefixes, registrations):
            probe_registration(args, prefix, registration,
                               previous_by_key.get(identity(registration)))
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
    summaries = [{"prefix_tokens": item["prefix_tokens"],
                  "cases": {key: value["probe"] for key, value in item["cases"].items()}}
                 for item in registrations]
    summary = {"status": report["status"]}
    if len(summaries) == 1:
        summary["cases"] = summaries[0]["cases"]
    else:
        summary["registrations"] = summaries
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
