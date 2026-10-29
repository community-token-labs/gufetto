#!/usr/bin/env python3
"""Bounded physical image-reclamation history; uses existing request/Recorder.

Run pressure and reference against separate fresh processes of the same binary.
Compare only after both processes have stopped and flushed their server logs.
This is a correctness/physical-reclamation gate, not a matched speed claim.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
import threading

from image_history import request
from pi_agent import Recorder, save
from vision_admission import body

FILL_REQUESTS = 96
FILL_WORDS = 4096


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def fill_body(index):
    assert 0 <= index < FILL_REQUESTS
    payload = body([], f"History {index:03d}:" + " word" * FILL_WORDS)
    payload.update(cache_prompt=True, max_tokens=1)
    return payload


def image_body(fixture):
    data = fixture.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "PNG fixture required"
    assert struct.unpack(">II", data[16:24]) == (256, 256), "64-row fixture required"
    return body(["data:image/png;base64," + base64.b64encode(data).decode()],
                "Describe the screenshot in one sentence."), hashlib.sha256(data).hexdigest()


def exercise(base, fixture, phase, recorder, report):
    payload, fixture_hash = image_body(fixture)
    report.update(fixture_sha256=fixture_hash, image_body_sha256=digest(payload),
                  fill_body_sha256=[], checks={})

    def send(label, value):
        recorder.case = label
        response, elapsed = request(base, "/v1/chat/completions", value, timeout=600)
        assert response["usage"]["completion_tokens"] > 0, (label, response)
        assert response["choices"][0]["finish_reason"] in ("length", "stop"), response
        report["checks"][label] = {"response": response, "wall_ms": elapsed}
        return response

    if phase == "pressure":
        for index in range(FILL_REQUESTS):
            fill = fill_body(index)
            report["fill_body_sha256"].append(digest(fill))
            response = send(f"fill-{index:03d}", fill)
            assert response["usage"]["prompt_tokens"] >= FILL_WORDS, response
    response = send("image", payload)
    assert response["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, response
    report["status"] = "requests_complete"  # NOT the physical-reclamation verdict


def fields(line):
    return dict(re.findall(r"([a-zA-Z_]+)=([^ ]+)", line))


def compare(pressure, reference, log):
    assert pressure["status"] == reference["status"] == "requests_complete"
    assert pressure["phase"] == "pressure" and reference["phase"] == "reference"
    assert pressure["source_revision"] == reference["source_revision"]
    assert pressure["binary_sha256"] == reference["binary_sha256"]
    for key in ("fixture_sha256", "image_body_sha256", "harness_sha256"):
        assert pressure[key] == reference[key], key
    assert pressure["fill_body_sha256"] == [digest(fill_body(i)) for i in range(FILL_REQUESTS)]
    assert reference["fill_body_sha256"] == []
    a, b = (report["checks"]["image"] for report in (pressure, reference))
    assert a["response"]["choices"] == b["response"]["choices"], "image output mismatch"
    assert a["response"]["usage"]["prompt_tokens"] == b["response"]["usage"]["prompt_tokens"]
    for row in (a, b):
        assert row["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    image_requests = [row for row in pressure["requests"] if row["case"] == "image"]
    assert len(image_requests) == 1 and image_requests[0]["request_id"]
    request_id = image_requests[0]["request_id"]
    lines = log.splitlines()
    begin = next(i for i, line in enumerate(lines)
                 if f"request={request_id} event=received " in line)
    admitted = next(i for i in range(begin, len(lines))
                    if "event=image_claim_admitted " in lines[i])
    completed = next(i for i in range(admitted, len(lines))
                     if "event=image_claim_completed " in lines[i])
    state = fields(lines[completed])
    assert int(state["outstanding_bytes"]) == 0
    assert int(state["resident_bytes"]) >= 64 * 2560 * 4, "retained outputs unaccounted"
    window = lines[begin:admitted]
    assert any("event=retention action=2 reason=1 " in line for line in window), (
        "bounded fill did not exercise logical checkpoint eviction")
    before = [fields(line) for line in window if "event=image_backing_reclaim_before " in line]
    after = [fields(line) for line in window if "event=image_backing_reclaim_after " in line]
    assert before and len(before) == len(after), "no paired physical-reclaim evidence"
    released = 0
    rss_decreased = False
    for first, last in zip(before, after):
        delta = int(last["released_bytes"])
        assert delta > 0
        assert int(first["committed_bytes"]) - int(last["committed_bytes"]) == delta
        rss_decreased |= int(last["rss_mib"]) < int(first["rss_mib"])
        released += delta
    assert rss_decreased, "committed bytes fell but physical RSS did not"
    assert any("event=vision_pool " in line and int(fields(line)["encodes"]) > 0
               for line in lines[admitted:]), "actual Encode not observed"
    completion = next(fields(line) for line in lines[completed:]
                      if f"request={request_id} event=completed " in line)
    assert int(completion["draft_rounds"]) > 0 and int(completion["draft_proposed"]) > 0, (
        "image request did not exercise actual MTP drafts")
    return {"status": "passed", "released_bytes": released,
            "reclaim_before": before, "reclaim_after": after,
            "image_claim_completed": state, "image_server_completion": completion,
            "pressure_image_wall_ms": a["wall_ms"],
            "cold_image_wall_ms": b["wall_ms"],
            "timing_scope": "Different cache histories; observations, not a speed qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pressure", "reference", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--pressure-report", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--server-log", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"phase": args.phase, "status": "running",
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.phase == "compare":
        if not all((args.pressure_report, args.reference_report, args.server_log)):
            parser.error("compare requires both reports and --server-log")
        try:
            report.update(compare(json.loads(args.pressure_report.read_text()),
                                  json.loads(args.reference_report.read_text()),
                                  args.server_log.read_text()))
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            save(args.output / "report.json", report)
        return
    if not all((args.base_url, args.fixture, args.binary, args.source_revision)):
        parser.error("request phases require base URL, fixture, binary and source revision")
    report.update(source_revision=args.source_revision,
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    wire = args.output / "wire"
    wire.mkdir()
    recorder = Recorder(args.base_url, wire, request_timeout=600)
    thread = threading.Thread(target=recorder.serve_forever)
    thread.start()
    try:
        exercise(f"http://127.0.0.1:{recorder.server_port}", args.fixture,
                 args.phase, recorder, report)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        recorder.shutdown()
        thread.join()
        recorder.server_close()
        report["requests"] = recorder.requests
        save(args.output / "report.json", report)


if __name__ == "__main__":
    main()
