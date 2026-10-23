#!/usr/bin/env python3
"""Probe an already-running pin-enabled server, then rerun after restart.

Uses raw /completion prompts, so FILE must be the same rendered text supplied
at server startup. This runner never starts/stops a server or deletes caches.
Greedy output must equal cold prefill; seeded output must replay in RAM and
across --restore. A sampled cold split is not used as an exactness oracle.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix-file", required=True, type=Path)
    parser.add_argument("--prefix-tokens", required=True, type=int,
                        help="exact token depth from the pin startup log")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--restore", type=Path,
                        help="report written before an independently managed restart")
    parser.add_argument("--churn", type=int, default=0,
                        help="optional unrelated bounded prompts between pin probes")
    args = parser.parse_args()
    if args.prefix_tokens <= 0 or not 0 <= args.churn <= 140:
        parser.error("positive prefix tokens and churn in 0..140 required")
    if args.output.exists():
        parser.error("output already exists")
    prefix_bytes = args.prefix_file.read_bytes()
    prefix = prefix_bytes.decode("utf-8")
    identity = hashlib.sha256(prefix_bytes).hexdigest()
    previous = json.loads(args.restore.read_text()) if args.restore else None
    if previous and (previous["prefix_sha256"] != identity or
                     previous["prefix_tokens"] != args.prefix_tokens):
        parser.error("restore report belongs to a different registered prefix")
    report = {"schema": "gufo-pinned-prefix-probe-v1", "prefix_sha256": identity,
              "prefix_tokens": args.prefix_tokens, "restore": bool(previous),
              "cases": {}, "status": "running"}

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
            assert result["tokens_cached"] >= args.prefix_tokens, result
            assert result["probe"]["prefill_tokens"] <= (result["tokens_evaluated"] -
                                                          args.prefix_tokens), result
        return result

    try:
        for index in range(args.churn):
            # Distinct from the registration at token zero; no special marker
            # can make this an accidental pin hit. Keep this explicitly bounded.
            call(args.url, {"prompt": f"Unrelated fixture {index}: " +
                                     (f"record_{index} has a field. " * 80) +
                                     "Reply with OK.",
                            "n_predict": 1, "temperature": 0,
                            "cache_prompt": True})
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
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
    print(json.dumps({"status": report["status"],
                      "cases": {key: value["probe"] for key, value in report["cases"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
