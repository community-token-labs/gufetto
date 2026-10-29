#!/usr/bin/env python3
"""Focused vision-budget requests against an already-running isolated server.

Reuses the existing image-history request helper and Pi Recorder. Native
vision_serving_test owns exact-logit, reader-fence and deterministic queue oracles.
This runner retains wire/timings; successful HTTP alone is not those oracles.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import threading
import urllib.error

from image_history import encoded_variant, request
from pi_agent import Recorder, save


def body(urls, text="Describe these screenshots briefly."):
    return {"model": "gufo", "messages": [{"role": "user", "content": [
        {"type": "text", "text": text},
        *({"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
          for url in urls)]}], "max_tokens": 16, "temperature": 0, "seed": 41,
        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}


def exercise(base, fixtures, policy, report, record, *, request_timeout=600,
             independent_control=False):
    def url(size, index):
        path = fixtures / f"screenshot-{size}x{size}-{index}.png"
        data = path.read_bytes()
        report["fixtures"][path.name] = hashlib.sha256(data).hexdigest()
        return "data:image/png;base64," + base64.b64encode(data).decode()

    small = [url(256, i) for i in range(3)]

    def send(label, payload, *, reject=False, count=False):
        endpoint = "/v1/chat/completions" + ("/input_tokens" if count else "")
        record.case = label
        try:
            result, elapsed = request(base, endpoint, payload, timeout=request_timeout)
        except urllib.error.HTTPError as error:
            raw = error.read().decode()
            report["checks"][label] = {"status": error.code, "error": raw}
            assert reject and error.code == 400, (label, error.code, raw)
            # Budget refusal must be explicit, not a generic allocation error.
            assert "image" in raw.lower() or "vision" in raw.lower(), raw
            return None
        assert not reject, (label, "never-fit request accepted", result)
        report["checks"][label] = {"response": result, "wall_ms": elapsed}
        if count:
            assert result["object"] == "response.input_tokens", result
            assert result["input_tokens"] > 0, result
            return result["input_tokens"]
        assert result["usage"]["completion_tokens"] > 0, result
        assert result["choices"][0]["finish_reason"] in ("stop", "length"), result
        return result

    if policy in ("cap64", "pool128"):
        limit = 1 if policy == "cap64" else 2
        payload = body(small[:limit])
        expected = send("fits-count", payload, count=True)
        accepted = send("fits", payload)
        assert accepted["usage"]["prompt_tokens"] == expected, accepted
        send("never-fit-count", body(small[:limit + 1]), reject=True, count=True)
        send("never-fit", body(small[:limit + 1]), reject=True)
        # A previously accepted image remains in the aggregate history.
        historical = body(small[:limit])
        historical["cache_prompt"] = True
        first = send("history-root", historical)
        historical["messages"] += [
            {"role": "assistant", "content": first["choices"][0]["message"]["content"]},
            {"role": "user", "content": body([small[limit]])["messages"][0]["content"]}]
        send("history-overflow", historical, reject=True)
        alternate = [encoded_variant(small[0], 991), *small[1:limit + 1]]
        send("alternate-encoding-overflow", body(alternate), reject=True)
        send("recovery", payload)
    elif policy == "late":
        text = " word" * 32768
        plain = body([], text)
        assert send("late-text-count", plain, count=True) >= 32768
        payload = body([small[0]], text + " Describe the screenshot in one sentence.")
        expected = send("late-image-count", payload, count=True)
        cold = send("late-image-cold", payload)
        repeat = send("late-image-uncached-repeat", payload)
        assert cold["usage"]["prompt_tokens"] == expected, cold
        assert repeat["choices"] == cold["choices"], (cold, repeat)
        assert repeat["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, repeat
    else:
        large = [url(4096, i) for i in range(8)]
        small_count = send("small-shape-count", body([small[0]]), count=True)
        large_count = send("max-shape-count", body([large[0]]), count=True)
        assert large_count - small_count == 16384 - 64, (small_count, large_count)
        send("four-max-count", body(large[:4]), count=True)
        send("five-max-count", body(large[:5]), reject=not independent_control, count=True)
        if not independent_control:
            send("five-max", body(large[:5]), reject=True)
        else:
            report["control_exclusion"] = (
                "No legacy whole-pool bound: count accepts five; omit the newly-invalid "
                "generation rather than add unmatched encoder/cache history.")
        send("four-max", body(large[:4]))
        # Four disjoint histories, two max images each; request leases need
        # twice the fixed pool. Native oracle separately proves gate vs slots.
        record.case = "four-concurrent-two-max"
        def peer(index):
            return request(base, "/v1/chat/completions", body(large[index:index + 2]),
                           timeout=request_timeout)
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(peer, (0, 2, 4, 6)))
        for result, elapsed in responses:
            assert result["usage"]["completion_tokens"] > 0, result
        report["checks"]["four-concurrent-two-max"] = responses
        for label, pair in (("reencode-A", large[:2]), ("reencode-B", large[2:4]),
                            ("reencode-C", large[4:6]), ("reencode-A-replay", large[:2])):
            send(label, body(pair))
        first = report["checks"]["reencode-A"]["response"]
        replay = report["checks"]["reencode-A-replay"]["response"]
        assert first["choices"] == replay["choices"], (first, replay)
        assert replay["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, replay
        report["required_server_evidence"] = (
            "Inspect vision_pool waits, evictions, reencodes and peak rows. "
            "HTTP completion alone does not qualify gate/re-encode execution.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("cap64", "pool128", "late", "default"), required=True)
    parser.add_argument("--request-timeout", type=float, default=600,
                        help="Bounded client/upstream deadline; does not change timing margins")
    parser.add_argument("--independent-control", action="store_true",
                        help="Default-policy legacy control lacks the new whole-pool bound")
    args = parser.parse_args()
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be finite and positive")
    if args.independent_control and args.policy != "default":
        parser.error("--independent-control is only for the default policy")
    args.output.mkdir(parents=True, exist_ok=False)
    wire = args.output / "wire"
    wire.mkdir()
    recorder = Recorder(args.base_url, wire, request_timeout=args.request_timeout)
    thread = threading.Thread(target=recorder.serve_forever)
    thread.start()
    report = {"policy": args.policy, "fixtures": {}, "checks": {}, "status": "running",
              "request_timeout": args.request_timeout,
              "independent_control": args.independent_control,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        exercise(f"http://127.0.0.1:{recorder.server_port}", args.fixtures,
                 args.policy, report, recorder, request_timeout=args.request_timeout,
                 independent_control=args.independent_control)
        report["status"] = "http_checks_passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        recorder.shutdown()
        thread.join()
        recorder.server_close()
        report["requests"] = recorder.requests
        save(args.output / "report.json", report)


if __name__ == "__main__":
    main()
