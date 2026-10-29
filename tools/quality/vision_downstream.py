#!/usr/bin/env python3
"""Run #697 twice against one independent vision teacher and one pinned history."""

import argparse
import hashlib
from pathlib import Path
import sys

from matched_kl import Capture, compare, digest, integer, load_history, require, text, write_json
from vision_compare import load


def run(image_set, history, reference, current, candidate, current_contract,
        candidate_contract, out_dir, top_k=20, max_positions=4096):
    images, image_hash = load(image_set)
    require(images.get("schema") == "gufo-vision-images-v1", "invalid image set")
    cases = images.get("cases")
    require(isinstance(cases, list) and cases, "empty image set")
    by_id = {case["id"]: case for case in cases}
    require(len(by_id) == len(cases), "duplicate image case")
    data, history_hash, expected = load_history(Path(history), max_positions)
    require(data["provenance"].get("image_set_sha256") == image_hash,
            "history is not bound to the frozen image set")
    seen, phases, stratum_owners = set(), set(), {}
    for sequence in data["sequences"]:
        phase, conditioning = sequence["phase"], sequence["conditioning"]
        require(phase in ("after-image", "text"), "explicit after-image/text phase required")
        phases.add(phase)
        if phase == "text":
            require(conditioning == {"kind": "text"}, "text control has image conditioning")
            owner = (phase, sequence["prompt_id"])
            require(stratum_owners.setdefault(sequence["stratum"], owner) == owner,
                    "stratum pools different cases/prompts")
            continue
        require(conditioning.get("kind") == "image", "image conditioning required")
        case_id = conditioning.get("case_id")
        require(case_id in by_id, "unknown image case")
        case = by_id[case_id]
        owner = (phase, case_id)
        require(stratum_owners.setdefault(sequence["stratum"], owner) == owner,
                "stratum pools different cases/prompts")
        require(conditioning.get("image_sha256") == digest(case["sha256"], "image hash"),
                "image content identity mismatch")
        prompt_hash = hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest()
        require(conditioning.get("prompt_sha256") == prompt_hash, "frozen prompt mismatch")
        spans = conditioning.get("expanded_spans")
        require(isinstance(spans, list) and spans, "explicit expanded image spans required")
        end = 0
        for span in spans:
            require(isinstance(span, list) and len(span) == 2, "invalid image span")
            start, stop = span
            integer(start, "span start")
            integer(stop, "span stop", 1)
            require(end <= start < stop <= len(sequence["tokens"]), "overlapping/out-of-range image span")
            end = stop
        require(all(position >= end for position in sequence["positions"]),
                "scored position is inside/before an image span")
        seen.add(case_id)
    require(seen == set(by_id), "image case coverage mismatch")
    require(phases == {"after-image", "text"}, "both after-image and text controls required")
    producers = {}
    for arm, path in (("reference", reference), ("current", current), ("candidate", candidate)):
        capture = Capture(Path(path), data, history_hash, len(expected))
        try:
            producers[arm] = capture.header["producer"]
            digest(producers[arm].get("vision_source_sha256"), "vision source hash")
        finally:
            capture.close()
    teacher = producers["reference"]
    require(teacher.get("vision_accumulator_dtype") == "f32" and
            teacher.get("vision_input_origin") == "pixels" and
            teacher.get("vision_independent") is True,
            "mandatory independent F32 vision teacher from pixels")
    text(teacher.get("vision_independence"), "vision independence explanation")
    require(all(teacher["model_sha256"] == producers[arm]["model_sha256"]
                for arm in ("current", "candidate")), "unmatched downstream model artifacts")
    require(all(teacher["vision_source_sha256"] != producers[arm]["vision_source_sha256"]
                for arm in ("current", "candidate")), "vision reference shares tested source")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    try:
        reports = {}
        for arm, capture, contract in (("current", current, current_contract),
                                       ("candidate", candidate, candidate_contract)):
            reports[arm] = compare(Path(history), Path(reference), Path(capture),
                                   out_dir / arm, Path(contract), top_k, max_positions)
        require(reports["current"]["reference"] == reports["candidate"]["reference"],
                "reference changed between comparisons")
        require(reports["current"]["history_sha256"] == history_hash ==
                reports["candidate"]["history_sha256"], "history changed between comparisons")
        statuses = {report["status"] for report in reports.values()}
        result = {"schema": "gufo-vision-downstream-v1",
                  "status": "FAIL" if "FAIL" in statuses else
                            "NOT_EVALUATED" if "NOT_EVALUATED" in statuses else "PASS",
                  "qualification": "NOT_QUALIFIED", "image_set_sha256": image_hash,
                  "history_sha256": history_hash,
                  "comparisons": {arm: {"status": report["status"], "strata": report["strata"],
                                        "gates": report["gates"], "report": arm + "/report.json"}
                                  for arm, report in reports.items()}}
        write_json(out_dir / "report.json", result)
        return result
    except Exception as error:
        write_json(out_dir / "failure.json", {"status": "INVALID", "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image-set", "history", "reference", "current", "candidate",
                 "current-contract", "candidate-contract", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-positions", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        result = run(args.image_set, args.history, args.reference, args.current, args.candidate,
                     args.current_contract, args.candidate_contract, args.out_dir,
                     args.top_k, args.max_positions)
    except (ValueError, OSError, OverflowError, TypeError, KeyError) as error:
        print(f"vision_downstream: INVALID: {error}", file=sys.stderr)
        return 2
    print(f"{result['status']} (offline logits only; NOT_QUALIFIED)")
    return int(result["status"] == "FAIL")


if __name__ == "__main__":
    sys.exit(main())
