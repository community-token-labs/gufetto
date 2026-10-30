"""Gufo-only cards; never turn aggregate rates into request timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

STATUSES = {"PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN"}
PHASES = ("queue_ms", "prefill_ms", "decode_ms", "server_ttft_ms", "wall_ms")


def cell(value):
    if value is None or value == "":
        return "UNKNOWN"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def observations(document):
    """Read the existing serving-benchmark schema, not a new bench JSON dialect."""
    if document.get("artifactType") != "servingBenchmark" or document.get(
        "benchmarkSchema"
    ) != "gufo.serving-benchmark.v1":
        raise ValueError("expected gufo.serving-benchmark.v1 servingBenchmark")
    rows = []
    for group, result in document["results"].items():
        for sample in result["samples"]:
            for phase in PHASES:
                value = sample.get(phase)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                ):
                    raise ValueError(f"invalid {phase}")
            rows.append((group, sample))
    return rows


def render(raw: bytes, metadata: dict) -> str:
    quality = metadata.get("quality", {})
    status = quality.get("status", "INCONCLUSIVE")
    if status not in STATUSES:
        raise ValueError("quality status must be PASS, FAIL, INCONCLUSIVE or NOT_RUN")
    text = raw.decode("utf-8")
    document = None
    samples = []
    aggregate = []
    if text.lstrip().startswith("{"):
        document = json.loads(text)
        kind = document.get("artifactType")
        if kind == "servingBenchmark":
            if document.get("workload", {}).get("endpointProfile") != "gufo":
                raise ValueError("serving card requires endpointProfile=gufo")
            samples = observations(document)
        elif kind == "model-bench-table":
            if document.get("target") != "gufo":
                raise ValueError("only target=gufo model-bench tables are supported")
            aggregate = [(key, {k: v for k, v in row.items() if k != "command"})
                         for key, row in document["rows"].items()]
        else:
            raise ValueError("unsupported JSON: gufo bench has no --json at this revision")
    else:
        # Native gufo bench and rendered BENCHMARKS.md tables. Keep only Gufo
        # columns; reference comparisons and gains are deliberately excluded.
        headers = None
        table_lines = text.splitlines()
        for index, line in enumerate(table_lines):
            if not line.strip().startswith("|"):
                headers = None
                continue
            values = [v.strip() for v in line.strip().strip("|").split("|")]
            following = table_lines[index + 1].strip() if index + 1 < len(table_lines) else ""
            is_header = following.startswith("|") and all(
                not v.strip() or set(v.strip()) <= set("-: ")
                for v in following.strip("|").split("|")
            )
            if is_header:
                headers = values if (
                    "t/s" in values and "model" in values
                    or any(v.startswith("Gufo pp") or v.startswith("Gufo tg") for v in values)
                ) else None
            elif headers and len(values) == len(headers) and not all(
                not v or set(v) <= set("-: ") for v in values
            ):
                keep = [i for i, h in enumerate(headers)
                        if i == 0 or h in {"size", "params", "backend", "test", "t/s"}
                        or h.startswith("Gufo pp") or h.startswith("Gufo tg")]
                aggregate.append((str(len(aggregate)), {headers[i]: values[i] for i in keep}))
        if not aggregate:
            raise ValueError("no recognized Gufo benchmark table")
    if document and document.get("target") not in (None, "gufo"):
        raise ValueError("reference results cannot produce a Gufo-only card")
    stack = metadata.get("stack", {})
    artifacts = metadata.get("artifacts", [])
    for artifact in artifacts:
        digest = artifact.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("artifact SHA256 must be 64 lowercase hex characters")
    missing = []
    for key in ("runtime", "hip", "rocr", "compiler", "commit", "build_flags", "hardware"):
        if not stack.get(key):
            missing.append("stack." + key)
    if not artifacts:
        missing.append("artifact SHA256s")
    if not metadata.get("model"):
        missing.append("model")
    if not metadata.get("settings"):
        missing.append("settings")
    if not quality.get("evidence"):
        missing.append("quality evidence")
    if not samples or any(s.get(p) is None for _, s in samples for p in PHASES):
        missing.append("complete per-request phase timings")
    card_status = "FAIL" if status == "FAIL" else "INCONCLUSIVE" if missing or status != "PASS" else "PASS"
    lines = [f"# Gufo-only benchmark card: {cell(metadata.get('model'))}", "",
             f"- Card status: **{card_status}** (not a cross-engine comparison)",
             f"- Quality status: **{status}**", f"- Quality evidence: {cell(quality.get('evidence'))}",
             f"- Input SHA256: `{hashlib.sha256(raw).hexdigest()}`",
             f"- Missing evidence: {cell(missing) if missing else 'none'}", "", "## Artifacts", "",
             "| Artifact | SHA256 |", "| --- | --- |"]
    lines += [f"| {cell(a.get('name'))} | {a['sha256']} |" for a in artifacts]
    if not artifacts:
        lines.append("| UNKNOWN | UNKNOWN |")
    lines += ["", "## Exact stack (operator supplied; not inferred)", ""]
    lines += [f"- {key}: {cell(stack.get(key))}" for key in
              ("runtime", "hip", "rocr", "compiler", "commit", "build_flags", "hardware")]
    lines += ["", "## Settings", "", cell(metadata.get("settings")), "",
              "## Per-request phases (milliseconds; never averaged)", "",
              "| Group / repetition / request | Queue | Prefill | Decode | Server TTFT | Wall |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for group, sample in samples:
        identity = f"{group} / {sample['repetition']} / {sample['request_index']}"
        lines.append("| " + " | ".join([cell(identity)] + [cell(sample.get(p)) for p in PHASES]) + " |")
    if not samples:
        lines.append("| NOT RECORDED | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |")
    lines += ["", "## Retained settings / counters", ""]
    if document:
        for key in ("source", "workload", "model", "mode", "table", "warnings", "notes"):
            if key in document:
                lines.append(f"- {key}: {cell(document[key])}")
    for group, sample in samples:
        counters = {k: v for k, v in sample.items() if k not in PHASES}
        lines.append(f"- {cell(group)}: {cell(counters)}")
    if aggregate:
        lines += ["", "## Aggregate-only source (NOT per-request evidence)", "",
                  "Rates below cannot reconstruct request phase durations. Card remains INCONCLUSIVE.", ""]
        lines += [f"- {cell(key)}: {cell(value)}" for key, value in aggregate]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(render(args.input.read_bytes(), json.loads(args.metadata.read_text())), end="")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"benchmark-card: {error}\n")
    return 0
