#!/usr/bin/env python3
"""Frozen R3 causal classification. Offline; no HTTP/model/host operations.

Use --fixtures LOG for deterministic CPU signatures, or REPORT [REPORT ...]
--logs SERVER_LOG for sanitized harness cases. Existing raw harness pass/fail
is never rewritten. One process/pool per log/report set (IDs are process-local).
Retirement logs need not be adjacent to the pair. No prompt text is emitted.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def uint(value):
    return type(value) is int and 0 <= value <= 2**64 - 1


def fields(line):
    return {key: int(value) if value.isdecimal() else value
            for key, value in re.findall(r"([a-z_]+)=([A-Za-z0-9_.-]+)", line)}


class Evidence:
    def __init__(self, lines=()):
        self.lookups, self.requests, self.acquisitions, self.retirements = {}, {}, {}, []
        self.restore_failures = []
        self.has_logs = False
        for line in lines:
            self.has_logs = True
            row = fields(line)
            if row.get("schema") == "cache-lookup-v1":
                if row.get("event") == "lookup":
                    self.lookups[row["lookup_id"]] = row
                elif row.get("event") == "http_lookup":
                    self.requests[row["request"]] = row["lookup_id"]
                elif row.get("event") == "retire":
                    self.retirements.append(row)
                elif row.get("event") == "restore_failure":
                    self.restore_failures.append(row)
            elif row.get("schema") == "component-cache-v1" and row.get("event") == "acquire":
                self.acquisitions[row["acquisition_id"]] = row

    def lookup(self, case):
        lookup = case.get("lookup", {})
        identity = lookup.get("observation_id", self.requests.get(lookup.get("request_id")))
        return self.lookups.get(identity, {})

    def admission(self, case):
        metrics = case.get("actual", {}).get("component_cache", {})
        return self.acquisitions.get(metrics.get("acquisition_id"), metrics)


def classify(case, evidence=None):
    evidence = evidence or Evidence()
    lookup, actual = case.get("lookup", {}), case.get("actual", {})
    matched, cached, version = lookup.get("matched_tokens"), actual.get("cached_tokens"), lookup.get("index_version")
    if not all(uint(v) for v in (matched, cached, version)):
        return "invalid_pair"
    observed, admission = evidence.lookup(case), evidence.admission(case)
    # A mismatched join must never turn a raw failure into a pass.
    if observed and (observed.get("matched_tokens") != matched or observed.get("index_version") != version):
        return "invalid_evidence"
    entry = observed.get("entry_id", 0)
    retired = [r for r in evidence.retirements if entry and r.get("entry_id") == entry
               and r.get("reason") in ("erased", "live_unavailable")]
    # Check invalid publication BEFORE accepting a newer admission as an excuse.
    if any(uint(r.get("retirement_version")) and r["retirement_version"] <= version for r in retired):
        return "bug_stale_publication"
    failures = admission.get("restore_failures")
    if uint(failures) and failures:
        return "bug_restore_failure"
    if "selected_tokens" in admission and admission["selected_tokens"] != cached:
        return "invalid_evidence"
    if matched > cached:
        admitted = admission.get("admission_index_version")
        # A later retirement AFTER admission cannot explain this pair. Missing
        # failure counters also cannot exclude case (b), even with a new index.
        if not uint(failures):
            return "unresolved_overreport"
        if uint(admitted) and any(uint(r.get("retirement_version")) and
                                 version < r["retirement_version"] <= admitted for r in retired):
            return "advisory_retirement"
        if uint(admitted) and admitted > version:
            return "advisory_later_version"
        return "unresolved_overreport"
    if matched < cached:
        # R2's odd seqlock version explicitly attests an owner mutation.
        # Even versions need the logged overlap, not an age-zero heuristic.
        overlap = observed.get("state") == "mutation_overlap" and (
            uint(observed.get("version_start")) and uint(observed.get("version_finish")) and
            observed["version_start"] != observed["version_finish"])
        if version & 1 or observed.get("state") == "mutation_in_flight" or overlap:
            return "conservative_mutation_underreport"
        return "unexplained_underreport"
    return "exact"


def summarize(cases, evidence=None):
    evidence = evidence or Evidence()
    results = []
    for case in cases:
        observed, admission = evidence.lookup(case), evidence.admission(case)
        label = case.get("case", "")
        label = label if isinstance(label, str) and re.fullmatch(r"[a-z0-9_]{1,80}", label) else "unlabelled"
        results.append({
            "case": label,
            "matched_tokens": case.get("lookup", {}).get("matched_tokens"),
            "cached_tokens": case.get("actual", {}).get("cached_tokens"),
            "index_version": case.get("lookup", {}).get("index_version"),
            "admission_index_version": admission.get("admission_index_version"),
            "entry_id": observed.get("entry_id"),
            "retirement_versions": [r["retirement_version"] for r in evidence.retirements
                                    if observed.get("entry_id") and
                                    r.get("entry_id") == observed["entry_id"] and
                                    r.get("reason") in ("erased", "live_unavailable")],
            "lookup_state": observed.get("state"),
            "restore_attempts": admission.get("restore_attempts"),
            "restore_failures": admission.get("restore_failures"),
            "restore_failure_reason": admission.get("restore_failure_reason"),
            "diagnostics_complete": bool(observed) and all(uint(admission.get(k)) for k in
                                      ("admission_index_version", "restore_attempts", "restore_failures")),
            "classification": classify(case, evidence)})
    counts = Counter(r["classification"] for r in results)
    n = len(results)
    over = sum(r["matched_tokens"] > r["cached_tokens"] for r in results
               if uint(r["matched_tokens"]) and uint(r["cached_tokens"]))
    under = sum(r["matched_tokens"] < r["cached_tokens"] for r in results
                if uint(r["matched_tokens"]) and uint(r["cached_tokens"]))
    return {"pairs": n, "overreports": over, "underreports": under,
            "missing_diagnostic_pairs": sum(not r["diagnostics_complete"] for r in results),
            "restore_failure_events": len(evidence.restore_failures) if evidence.has_logs else None,
            "overreport_rate": over / n if n else None,
            "underreport_rate": under / n if n else None,
            "classifications": dict(counts),
            "classification_rates": {key: value / n for key, value in counts.items()} if n else {},
            "cases": results,
            "hard_nonpass": not n or bool(evidence.restore_failures) or any(
                key not in ("exact", "advisory_retirement", "advisory_later_version",
                            "conservative_mutation_underreport") for key in counts)}


def fixture_results(lines):
    results, section = [], []
    for line in lines:
        row = fields(line)
        if row.get("schema") == "cache-lookup-fixture-v1":
            if row.get("event") == "begin":
                section = []
            else:
                evidence = Evidence(section)
                observed = evidence.lookups[row["lookup_id"]]
                acquired = evidence.acquisitions[row["acquisition_id"]]
                case = {"case": row["case"],
                        "lookup": dict(observed, observation_id=row["lookup_id"]),
                        "actual": {"cached_tokens": row["actual_tokens"],
                                   "component_cache": acquired}}
                result = summarize([case], evidence)
                results.append(result)
        section.append(line)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-diagnostics", action="store_true",
                        help="R3 rerun gate: missing lookup/admission joins are nonpass")
    args = parser.parse_args()
    paths = args.reports + [p for p in (args.logs, args.fixtures) if p]
    if args.fixtures:
        results = fixture_results(args.fixtures.read_text().splitlines())
        expected = {"control": "exact", "retirement": "advisory_retirement",
                    "restore_failure": "bug_restore_failure",
                    "in_flight_mutation": "conservative_mutation_underreport",
                    "completed_mutation_overlap": "conservative_mutation_underreport"}
        actual = {r["cases"][0]["case"]: r["cases"][0]["classification"] for r in results}
        report = {"schema": "cache-lookup-r3-classification-v1", "fixture_signatures": results,
                  "signatures_verified": len(results) == len(expected) and actual == expected}
        passed = report["signatures_verified"]
    else:
        evidence = Evidence(args.logs.read_text().splitlines() if args.logs else ())
        cases = [case for path in args.reports for case in json.loads(path.read_text()).get("cases", [])
                 if case.get("phase") == "load_diagnostic"]
        report = dict(summarize(cases, evidence), schema="cache-lookup-r3-classification-v1")
        if args.require_diagnostics and (not args.logs or report["missing_diagnostic_pairs"]):
            report["hard_nonpass"] = True
        passed = not report["hard_nonpass"]
    report["input_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
