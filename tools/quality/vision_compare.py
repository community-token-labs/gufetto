#!/usr/bin/env python3
"""Stream three vision tensor captures against a frozen, per-row gate plan.

This evaluates supplied captures, not the truth of their provenance declarations.
See VISION.md. No model, HIP, or online-softmax implementation is imported.
"""

import argparse
import array
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys

from matched_kl import (MAX_HISTORY_BYTES, MAX_INDEX_LINE_BYTES, digest, integer,
                        parse_json, require, text, write_json)

PAIRS = {"candidate_reference": ("reference", "candidate"),
         "current_reference": ("reference", "current"),
         "candidate_current": ("current", "candidate")}
ARMS = ("reference", "current", "candidate")


def load(path):
    with Path(path).open("rb") as source:
        raw = source.read(MAX_HISTORY_BYTES + 1)
    require(len(raw) <= MAX_HISTORY_BYTES, "JSON exceeds 64 MiB budget")
    return parse_json(raw), hashlib.sha256(raw).hexdigest()


def finite(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f"{name}: expected finite number >= {minimum}")
    return value


def row_metrics(reference, candidate, relative_floor):
    require(len(reference) == len(candidate) and len(reference) > 0, "row shape mismatch")
    require(all(math.isfinite(x) for x in reference) and
            all(math.isfinite(x) for x in candidate), "nonfinite tensor value")
    finite(relative_floor, "relative_floor")
    require(relative_floor > 0, "relative_floor must be positive")
    delta = [abs(float(a) - float(b)) for a, b in zip(reference, candidate)]
    ref_norm = math.sqrt(math.fsum(float(x) * x for x in reference))
    error_norm = math.sqrt(math.fsum(x * x for x in delta))
    return {"max_abs": max(delta),
            "max_relative": max(d / max(abs(r), relative_floor)
                                for r, d in zip(reference, delta)),
            "relative_l2": error_norm / max(ref_norm, relative_floor),
            "reference_max_abs": max(abs(x) for x in reference),
            "reference_rms": ref_norm / math.sqrt(len(reference))}


def validate(plan, contract, plan_hash, max_rows):
    integer(max_rows, "max_rows", 1)
    require(plan.get("schema") == "gufo-vision-tensors-v1", "unsupported tensor plan")
    require(contract.get("schema") == "gufo-vision-gates-v1", "unsupported gate contract")
    require(contract.get("plan_sha256") == plan_hash, "contract plan hash mismatch")
    require(contract.get("status") in ("PROPOSED", "APPROVED"), "contract status required")
    text(contract.get("approval"), "approval (or pending approval)")
    require(plan.get("scope") in ("encoder", "attention_operator"), "unsupported scope")
    digest(plan.get("image_set_sha256"), "image set hash")
    floor = finite(contract.get("relative_floor"), "relative_floor")
    require(floor > 0, "relative_floor must be positive")
    tensors = plan.get("tensors")
    require(isinstance(tensors, list) and tensors, "empty tensor plan")
    ids, total = set(), 0
    for tensor in tensors:
        require(isinstance(tensor, dict), "invalid tensor descriptor")
        ident = text(tensor.get("id"), "tensor id")
        require(ident not in ids, "duplicate tensor id")
        ids.add(ident)
        for key in ("case_id", "stage", "stratum"):
            text(tensor.get(key), key)
        for key in ("layer", "head"):
            integer(tensor.get(key), key, -1)
        width = integer(tensor.get("width"), "width", 1)
        require(width <= 16384, "row width exceeds budget")
        digest(tensor.get("input_sha256"), "tensor input hash")
        indices = tensor.get("row_indices")
        require(isinstance(indices, list) and indices, "explicit selected row indices required")
        previous = -1
        for index in indices:
            integer(index, "row index")
            require(index > previous, "row indices must be strictly increasing")
            previous = index
        total += len(indices)
        require(total <= max_rows, "tensor row budget exceeded (no sampling)")
    headers = contract.get("headers")
    require(isinstance(headers, dict) and set(headers) == set(ARMS), "all three pinned headers required")
    for arm in ARMS:
        header = headers[arm]
        require(isinstance(header, dict), "invalid header")
        require(header.get("schema") == "gufo-vision-capture-v1" and
                header.get("plan_sha256") == plan_hash and header.get("role") == arm and
                header.get("dtype") == "f32le" and header.get("rows") == total,
                "header schema/plan/role/dtype/count mismatch")
        integer(header["rows"], "header rows", 1)
        producer = header.get("producer")
        require(isinstance(producer, dict), "producer required")
        for key in ("implementation", "revision", "arithmetic", "toolchain"):
            text(producer.get(key), key)
        for key in ("model_sha256", "source_sha256"):
            digest(producer.get(key), key)
    ref = headers["reference"]["producer"]
    precision = ref.get("accumulator_dtype")
    require(precision in ("f32", "f32_qk_f64_softmax_pv"),
            "mandatory F32 or approved F32/F64 reference")
    if precision == "f32_qk_f64_softmax_pv":
        expected = {"projections": "f32", "qk": "f32", "softmax": "f64",
                    "pv": "f64", "output": "f32", "query_block": 128, "full_keys": True}
        require(json.dumps(ref.get("arithmetic_plan"), sort_keys=True) ==
                json.dumps(expected, sort_keys=True), "incomplete/unapproved reference arithmetic plan")
    required_origin = "pixels" if plan["scope"] == "encoder" else "frozen_qkv"
    require(ref.get("input_origin") == required_origin, "reference has wrong input boundary")
    require(ref.get("independent") is True, "independent reference declaration required")
    text(ref.get("independence"), "reference independence explanation")
    for arm in ("current", "candidate"):
        producer = headers[arm]["producer"]
        require(producer["implementation"] != ref["implementation"] and
                producer["source_sha256"] != ref["source_sha256"],
                "reference must not share implementation/source under test")
        require(producer["model_sha256"] == ref["model_sha256"], "unmatched model artifacts")
    gates = contract.get("tensors")
    require(isinstance(gates, dict) and set(gates) == ids, "gate coverage mismatch")
    for ident, entry in gates.items():
        require(isinstance(entry, dict) and set(entry) == set(PAIRS) | {"noninferiority"},
                f"all pair gates and noninferiority required: {ident}")
        for pair in PAIRS:
            limits = entry[pair]
            require(isinstance(limits, dict) and set(limits) == {"atol", "rtol", "relative_l2"},
                    "expected atol/rtol/relative_l2 gates")
            for key, value in limits.items():
                finite(value, key)
        ni = entry["noninferiority"]
        require(isinstance(ni, dict) and set(ni) == {"factor", "slack"}, "invalid noninferiority gate")
        finite(ni["factor"], "noninferiority factor", 1)
        finite(ni["slack"], "noninferiority absolute slack")
    return total


class TensorCapture:
    def __init__(self, path, stack, expected_header):
        self.path = Path(path)
        self.source = stack.enter_context(self.path.open("rb"))
        self.hasher = hashlib.sha256()
        self.header = self.record()
        # JSON canonicalization is type-sensitive (e.g. true cannot stand for 1).
        require(json.dumps(self.header, sort_keys=True) == json.dumps(expected_header, sort_keys=True),
                "capture header does not match frozen contract")

    def record(self):
        raw = self.source.readline(MAX_INDEX_LINE_BYTES + 1)
        if not raw:
            return None
        require(len(raw) <= MAX_INDEX_LINE_BYTES, "index line exceeds budget")
        self.hasher.update(raw)
        return parse_json(raw)

    def row(self, tensor, index):
        entry = self.record()
        require(entry is not None, "missing tensor row")
        require(entry.get("tensor_id") == tensor["id"] and
                type(entry.get("row")) is int and entry["row"] == index,
                "tensor row identity/order mismatch")
        offset = integer(entry.get("offset"), "row offset")
        require(offset % 4 == 0, "unaligned f32le offset")
        relative = Path(text(entry.get("file"), "row file"))
        root = self.path.parent.resolve()
        path = (root / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(root) and path.is_file(),
                "row must be a regular file inside capture directory")
        row_hash = digest(entry.get("sha256"), "row hash")
        with path.open("rb") as source:
            source.seek(offset)
            raw = source.read(tensor["width"] * 4)
        require(len(raw) == tensor["width"] * 4, "truncated tensor row")
        require(hashlib.sha256(raw).hexdigest() == row_hash, "tensor row hash mismatch")
        values = array.array("f")
        require(values.itemsize == 4, "unsupported host float")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        require(all(math.isfinite(x) for x in values), "nonfinite tensor value")
        return values, row_hash

    def finish(self):
        require(self.record() is None, "extra tensor row")
        return self.hasher.hexdigest()


def compare(plan_path, contract_path, reference_path, current_path, candidate_path,
            out_dir, max_rows=200000):
    plan, plan_hash = load(plan_path)
    contract, contract_hash = load(contract_path)
    total = validate(plan, contract, plan_hash, max_rows)
    out_dir = Path(out_dir)
    with ExitStack() as stack:
        captures = {arm: TensorCapture(path, stack, contract["headers"][arm])
                    for arm, path in zip(ARMS, (reference_path, current_path, candidate_path))}
        out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        try:
            summaries = {}
            partial = out_dir / "rows.partial.jsonl"
            with partial.open("x", encoding="utf-8") as out:
                for tensor in plan["tensors"]:
                    summary = {"rows": 0, "failed_rows": 0, "pairs": {},
                               "case_id": tensor["case_id"], "stage": tensor["stage"],
                               "layer": tensor["layer"], "head": tensor["head"],
                               "stratum": tensor["stratum"]}
                    for index in tensor["row_indices"]:
                        values = {arm: captures[arm].row(tensor, index) for arm in ARMS}
                        pairs = {}
                        for pair, (left, right) in PAIRS.items():
                            metric = row_metrics(values[left][0], values[right][0], contract["relative_floor"])
                            limits = contract["tensors"][tensor["id"]][pair]
                            ceiling = limits["atol"] + limits["rtol"] * metric["reference_max_abs"]
                            metric.update(bytes_equal=values[left][1] == values[right][1],
                                          max_abs_limit=ceiling,
                                          passed=metric["max_abs"] <= ceiling and
                                          metric["relative_l2"] <= limits["relative_l2"])
                            pairs[pair] = metric
                            aggregate = summary["pairs"].setdefault(pair, {
                                "max_abs": 0, "max_relative": 0, "relative_l2": 0,
                                "byte_equal_rows": 0, "worst_abs_row": index})
                            if metric["max_abs"] > aggregate["max_abs"]:
                                aggregate["worst_abs_row"] = index
                            for field in ("max_abs", "max_relative", "relative_l2"):
                                aggregate[field] = max(aggregate[field], metric[field])
                            aggregate["byte_equal_rows"] += metric["bytes_equal"]
                        ni = contract["tensors"][tensor["id"]]["noninferiority"]
                        ni_limit = ni["factor"] * pairs["current_reference"]["max_abs"] + ni["slack"]
                        noninferior = pairs["candidate_reference"]["max_abs"] <= ni_limit
                        passed = all(p["passed"] for p in pairs.values()) and noninferior
                        summary["rows"] += 1
                        summary["failed_rows"] += not passed
                        out.write(json.dumps({"tensor_id": tensor["id"], "row": index,
                                              "pairs": pairs, "hashes": {a: v[1] for a, v in values.items()},
                                              "noninferiority": {"passed": noninferior, "limit": ni_limit},
                                              "passed": passed}, allow_nan=False) + "\n")
                    summary["status"] = "FAIL" if summary["failed_rows"] else "PASS"
                    summaries[tensor["id"]] = summary
            index_hashes = {arm: capture.finish() for arm, capture in captures.items()}
            report = {"schema": "gufo-vision-comparison-v1", "scope": plan["scope"],
                      "status": "FAIL" if any(s["failed_rows"] for s in summaries.values()) else "PASS",
                      "qualification": "NOT_QUALIFIED", "registration_status": contract["status"],
                      "reference_independence": "declared_not_proven_by_metrics",
                      "rows": total, "max_rows": max_rows, "tensors": summaries,
                      "plan_sha256": plan_hash, "contract_sha256": contract_hash,
                      "image_set_sha256": plan["image_set_sha256"], "headers": contract["headers"],
                      "index_sha256": index_hashes,
                      "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            partial.rename(out_dir / "rows.jsonl")
            write_json(out_dir / "report.json", report)
            return report
        except Exception as error:
            write_json(out_dir / "failure.json", {"status": "INVALID", "error": str(error)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("plan", "contract", "reference", "current", "candidate", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=200000)
    args = parser.parse_args(argv)
    try:
        report = compare(args.plan, args.contract, args.reference, args.current,
                         args.candidate, args.out_dir, args.max_rows)
    except (ValueError, OSError, OverflowError, TypeError) as error:
        print(f"vision_compare: INVALID: {error}", file=sys.stderr)
        return 2
    print(f"{report['status']} ({report['registration_status']}; NOT_QUALIFIED)")
    return int(report["status"] == "FAIL")


if __name__ == "__main__":
    sys.exit(main())
