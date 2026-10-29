#!/usr/bin/env python3
"""CPU-only attribution of native publication gaps to peer prefill wall time.

Usage: python3 tools/prof/pd_attribution.py trace.jsonl

All times and durations remain integer nanoseconds. Inter-burst gaps use only
explicit native burst IDs; work boundaries are reported separately and never
substitute for missing burst evidence. Exposure
is correlation, not a causal GPU attribution. U/L/H require independent proof
and are deliberately null. This tool does not compare performance arms.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


STALL_NS = 500_000_000
MIN_EXPOSED = 20
IDENTITY = ("request_id", "slot", "slot_generation", "logical_agent")


def integer(value):
    return type(value) is int and value >= 0


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def intersection(start, end, intervals):
    return sum(max(0, min(end, right) - max(start, left))
               for left, right in intervals)


def quantiles(values):
    values = sorted(values)
    return {name: values[(len(values) * percentile + 99) // 100 - 1]
            if values else None
            for name, percentile in (("p50_ns", 50), ("p95_ns", 95),
                                     ("p99_ns", 99), ("max_ns", 100))}


def weighted_p95(pairs):
    total = sum(weight for _, weight in pairs)
    if not total:
        return None
    accumulated = 0
    for value, weight in sorted(pairs):
        accumulated += weight
        if accumulated * 100 >= total * 95:
            return value
    return None


def summarize(gaps, exposed):
    samples = [gap for gap in gaps if not exposed or gap["exposure_ns"] > 0]
    weight = "exposure_ns" if exposed else "duration_ns"
    stall = "exposed_stall_ns" if exposed else "stall_ns"
    return {
        "completed_samples": len(samples),
        **quantiles([gap["duration_ns"] for gap in samples]),
        "time_weighted_p95_ns": weighted_p95(
            [(gap["duration_ns"], gap[weight]) for gap in samples]),
        "total_weight_ns": sum(gap[weight] for gap in samples) if samples else None,
        "stall_area_ns": sum(gap[stall] for gap in samples) if samples else None,
    }


def read_jsonl(path):
    records, issues = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, RecursionError):
                issues.append(f"line {line_number}: invalid/truncated JSON")
                continue
            if not isinstance(record, dict):
                issues.append(f"line {line_number}: record is not an object")
                continue
            records.append(record)
    return records, issues


def identity(item):
    return (isinstance(item, dict)
            and all(integer(item.get(field)) for field in IDENTITY[:3])
            and (integer(item.get("logical_agent"))
                 or isinstance(item.get("logical_agent"), str)))


def candidate_join(prefill, works, incomplete):
    """Observed roster census, not a removable-work or state-equivalence proof.

    Order by host time within one scheduler thread, never by global ID arithmetic.
    The execution roster may span several physical batches/MTP rounds: only the
    first recorded target forward can supply the optimistic candidate cohort.
    Unknown forward members invalidate that cohort instead of borrowing IDs from
    adjacent work, matching depths, or later forwards.
    """
    ready = prefill["ready"]
    result = {"next_decode_work_id": None, "execution_request_ids": None,
              "matched_execution_request_ids": None,
              "unmatched_ready_request_ids": [r["request_id"] for r in ready],
              "actual_target_forwards": None, "first_target_forward": None,
              "candidate_request_ids": None,
              "outside_first_target_request_ids": None, "declines": []}
    if incomplete:
        result["declines"].append("trace integrity incomplete")
        return result
    later = [w for w in works if w["kind"] in ("decode_single", "decode_batch")
             and all(w[k] == prefill[k] for k in ("run_id", "pid", "thread_id"))
             and w["start_ns"] >= prefill["end_ns"]]
    if not later:
        result["declines"].append("no next decode work on the same scheduler thread")
        return result
    later.sort(key=lambda w: w["start_ns"])
    decode = later[0]
    if len(later) > 1 and (later[1]["start_ns"] == decode["start_ns"]
                           or later[1]["start_ns"] < decode["end_ns"]):
        result["declines"].append("ambiguous next decode work ordering")
        return result
    if any(w["kind"] == "prefill"
           and w["attribution"]["work_id"] != prefill["work_id"]
           and all(w[k] == prefill[k] for k in ("run_id", "pid", "thread_id"))
           and w["end_ns"] > prefill["end_ns"]
           and w["start_ns"] < decode["start_ns"] for w in works):
        result["declines"].append("another prefill intervened before next decoder target")
        return result
    result["next_decode_work_id"] = decode["attribution"]["work_id"]
    execution = {e["request_id"]: e for e in decode["attribution"]["execution"]}
    result["execution_request_ids"] = sorted(execution)
    matched = {r["request_id"] for r in ready
               if r["request_id"] in execution and r["slot_generation"] > 0
               and all(r[k] == execution[r["request_id"]][k] for k in IDENTITY)}
    result["matched_execution_request_ids"] = sorted(matched)
    result["unmatched_ready_request_ids"] = sorted(
        {r["request_id"] for r in ready} - matched)
    if result["unmatched_ready_request_ids"]:
        result["declines"].append("ready members absent or identity-mismatched in next execution")
    forwards = decode.get("forwards")
    if (not isinstance(forwards, list) or any(
            not isinstance(f, dict) or not integer(f.get("forward"))
            or type(f.get("prefill")) is not bool
            or not isinstance(f.get("sessions"), list) for f in forwards)):
        result["declines"].append("missing/invalid actual target forwards")
        return result
    targets = [f for f in forwards if not f["prefill"]]
    # Retain only the fixed numerical schema, never arbitrary input payloads.
    result["actual_target_forwards"] = [
        {"forward": f["forward"], "sessions": [
            {k: row.get(k) if integer(row.get(k)) else None
             for k in ("request_id", "native_session", "context_depth", "rows")}
            if isinstance(row, dict) else None for row in f["sessions"]]}
        for f in targets]
    if not targets:
        result["declines"].append("no actual decoder target forward")
        return result
    first = result["actual_target_forwards"][0]
    result["first_target_forward"] = first["forward"]
    rows = first["sessions"]
    if (not rows or any(row is None or row["request_id"] not in execution
                       or row["request_id"] == 0 or row["rows"] in (None, 0)
                       or row["context_depth"] is None for row in rows)
            or len({row["request_id"] for row in rows if row}) != len(rows)):
        result["declines"].append("first target cohort has missing/ambiguous execution lineage")
        return result
    cohort = {row["request_id"] for row in rows}
    result["candidate_request_ids"] = sorted(matched & cohort)
    result["outside_first_target_request_ids"] = sorted(matched - cohort)
    if matched - cohort:
        result["declines"].append("matched execution members outside first target cohort")
    return result


def analyze(records, input_issues=()):
    """Return observed statistics; qualification never implies a speedup PASS.

    Any malformed/missing/dropped data invalidates qualification globally:
    an unidentifiable missing record could have contained any peer's work.
    Observed samples remain available, explicitly marked incomplete.
    """
    issues = list(input_issues)
    requests = defaultdict(lambda: {"events": [], "identities": set(),
                                    "ready": [], "issues": []})
    prefills = []
    work_ids = set()
    works = []
    sidecar_count = 0

    for index, record in enumerate(records, 1):
        label = f"record {index}"
        if isinstance(record, dict) and record.get("schema") == "pd-profile-v1":
            # A separate diagnostic schema, not native perf or evidence of U.
            sidecar_count += 1
            continue
        if (not isinstance(record, dict) or record.get("schema") != 1
                or any(not integer(record.get(field)) for field in
                       ("run_id", "pid", "thread_id", "start_ns", "end_ns"))
                or not isinstance(record.get("kind"), str)
                or record["end_ns"] < record["start_ns"]):
            issues.append(f"{label}: invalid perf record")
            continue
        run, pid = record["run_id"], record["pid"]
        attr = record.get("attribution")
        if (not isinstance(attr, dict) or attr.get("version") != 1
                or not integer(attr.get("work_id"))
                or not integer(attr.get("dropped"))):
            issues.append(f"{label}: missing/invalid attribution")
            continue
        work_id = attr["work_id"]
        if (run, pid, work_id) in work_ids:
            issues.append(f"{label}: duplicate work identity")
            continue
        work_ids.add((run, pid, work_id))
        works.append(record)
        if attr["dropped"]:
            issues.append(f"{label}: dropped {attr['dropped']} attribution entries")
        for field in ("ready", "events", "execution"):
            if not isinstance(attr.get(field), list):
                issues.append(f"{label}: missing/invalid {field} array")
                continue
            seen = set()
            for item in attr[field]:
                if not identity(item):
                    issues.append(f"{label}: missing/invalid {field} identity")
                    continue
                key = (run, pid, item["request_id"])
                request = requests[key]
                request["identities"].add(tuple(item[name] for name in IDENTITY[1:]))
                if field != "events":
                    if key in seen:
                        issues.append(f"{label}: duplicate {field} request identity")
                    seen.add(key)
                    if field == "ready" and record["kind"] == "prefill":
                        request["ready"].append(record["start_ns"])
                    continue
                if (item.get("event") not in ("token", "terminal", "admit", "submit")
                        or not integer(item.get("ordinal"))
                        or not integer(item.get("time_ns"))):
                    issues.append(f"{label}: invalid publication/lifecycle event")
                    continue
                # submit may precede the work recording its admission.
                if (item["time_ns"] > record["end_ns"]
                        or (item["event"] != "submit"
                            and item["time_ns"] < record["start_ns"])):
                    issues.append(f"{label}: event outside work interval")
                burst = item.get("burst_id")
                if (not integer(burst)
                        or (item["event"] == "token" and burst == 0)
                        or (item["event"] != "token" and burst != 0)):
                    issues.append(f"{label}: missing/invalid burst_id")
                    burst = None
                request["events"].append({
                    **{name: item[name] for name in (*IDENTITY, "event", "ordinal", "time_ns")},
                    "work_id": work_id, "burst_id": burst,
                })
        if record["kind"] == "prefill":
            sessions = record.get("sessions")
            if (not isinstance(sessions, list) or not sessions
                    or any(not isinstance(s, dict) or not integer(s.get("request_id"))
                           for s in sessions)):
                issues.append(f"{label}: missing/invalid prefiller sessions")
                continue
            if len({s["request_id"] for s in sessions}) != len(sessions):
                issues.append(f"{label}: duplicate prefiller identity")
            prefiller_ids = []
            execution = attr.get("execution", [])
            # Legacy scheduler IDs are NEVER the trace-ID namespace. A selected
            # request may cancel before any execution scope exists.
            for session in sessions:
                explicit = session.get("attribution_request_id")
                matches = {e["request_id"] for e in execution if identity(e)
                           and e.get("native_request_id") == session["request_id"]
                           and e["slot"] == session.get("slot")} if isinstance(execution, list) else set()
                if integer(explicit) and explicit > 0:
                    matches.add(explicit)
                if len(matches) != 1:
                    issues.append(f"{label}: missing/ambiguous native prefiller identity join")
                else:
                    prefiller_ids.extend(matches)
            if len(prefiller_ids) != len(sessions):
                continue
            phases = record.get("phases")
            if (any(not integer(record.get(name)) for name in
                    ("tokens_processed", "tokens_requested"))
                    or not isinstance(phases, list)
                    or any(not isinstance(p, dict) or not isinstance(p.get("name"), str)
                           for p in phases)):
                issues.append(f"{label}: missing/invalid prefill accounting")
            prefills.append({
                "run_id": run, "pid": pid, "work_id": work_id,
                "thread_id": record["thread_id"],
                "ready": [{k: e[k] for k in IDENTITY} for e in attr.get("ready", [])
                          if identity(e)] if isinstance(attr.get("ready"), list) else [],
                "start_ns": record["start_ns"], "end_ns": record["end_ns"],
                "request_ids": prefiller_ids,
                "tokens_requested": record.get("tokens_requested")
                if integer(record.get("tokens_requested")) else None,
                "tokens_processed": record.get("tokens_processed")
                if integer(record.get("tokens_processed")) else None,
                "phase_names": [p["name"] for p in phases
                                if isinstance(p, dict) and isinstance(p.get("name"), str)]
                if isinstance(phases, list) else [],
            })

    if not work_ids:
        issues.append("no valid attribution work records")
    results = []
    for (run, pid, request_id), request in sorted(requests.items()):
        reasons = request["issues"]
        events = sorted(request["events"], key=lambda event: event["time_ns"])
        tokens = sorted((e for e in events if e["event"] == "token"),
                        key=lambda event: (event["ordinal"], event["time_ns"]))
        terminals = [e for e in events if e["event"] == "terminal"]
        submits = [e for e in events if e["event"] == "submit"]
        admits = [e for e in events if e["event"] == "admit"]
        if len({(type(i[2]), i[2]) for i in request["identities"]}) != 1:
            reasons.append("conflicting logical-agent identity")
        # Admission may bind a previously unbound slot. Publications must not
        # silently change slot generation within the same request.
        if len({(e["slot"], e["slot_generation"]) for e in tokens}) > 1:
            reasons.append("conflicting publication slot identity")
        if len(submits) != 1:
            reasons.append("missing/duplicate submit event")
        if len(admits) > 1 or (tokens and len(admits) != 1):
            reasons.append("missing/duplicate admit event")
        if len(admits) == 1:
            admission = admits[0]["time_ns"]
            if ((submits and admission < submits[0]["time_ns"])
                    or (tokens and admission > tokens[0]["time_ns"])
                    or (terminals and admission > terminals[0]["time_ns"])):
                reasons.append("invalid admission lifecycle ordering")
        if len(terminals) != 1:
            reasons.append("missing/duplicate terminal event (possibly truncated trace)")
        if not tokens:
            reasons.append("no native token publications")
        if tokens and tokens[0]["ordinal"] not in (0, 1):
            reasons.append("missing initial token ordinals")
        if any(b["ordinal"] != a["ordinal"] + 1 for a, b in zip(tokens, tokens[1:])):
            reasons.append("duplicate/gapped token ordinals")
        if any(b["time_ns"] < a["time_ns"] for a, b in zip(tokens, tokens[1:])):
            reasons.append("token timestamps regress with ordinal")
        if submits and events and submits[0]["time_ns"] > events[0]["time_ns"]:
            reasons.append("events precede submit")
        if terminals and tokens and terminals[0]["time_ns"] < tokens[-1]["time_ns"]:
            reasons.append("token after terminal")
        # The native publisher uses the final published-token count at terminal.
        # Accept zero-based synthetic token ordinals as well as native one-based.
        if len(terminals) == 1 and terminals[0]["ordinal"] != (
                tokens[-1]["ordinal"] + (tokens[0]["ordinal"] == 0) if tokens else 0):
            reasons.append("terminal token count disagrees with publications")
        peer_intervals = union([
            (p["start_ns"], p["end_ns"]) for p in prefills
            if p["run_id"] == run and p["pid"] == pid
            and request_id not in p["request_ids"]])
        # IDs are opaque and noncontiguous, but a closed burst cannot resume.
        closed_bursts = set()
        last_burst = None
        for token in tokens:
            burst = token["burst_id"]
            if burst != last_burst:
                if burst is not None and burst in closed_bursts:
                    reasons.append("noncontiguous publications reuse burst_id")
                if last_burst is not None:
                    closed_bursts.add(last_burst)
                last_burst = burst
        gaps = []
        for previous, current in zip(tokens, tokens[1:]):
            # Never fabricate a completed gap across lost publications.
            if (current["ordinal"] != previous["ordinal"] + 1
                    or current["time_ns"] < previous["time_ns"]):
                continue
            start, end = previous["time_ns"], current["time_ns"]
            gaps.append({
                "previous_ordinal": previous["ordinal"], "ordinal": current["ordinal"],
                "start_ns": start, "end_ns": end, "duration_ns": end - start,
                "previous_work_id": previous["work_id"], "work_id": current["work_id"],
                "inter_work": previous["work_id"] != current["work_id"],
                "previous_burst_id": previous["burst_id"], "burst_id": current["burst_id"],
                "inter_burst": previous["burst_id"] != current["burst_id"]
                if previous["burst_id"] is not None and current["burst_id"] is not None
                else None,
                "exposure_ns": intersection(start, end, peer_intervals),
                "stall_ns": max(0, end - start - STALL_NS),
                "exposed_stall_ns": intersection(start + STALL_NS, end, peer_intervals),
            })
        groups = {"token_gaps": gaps,
                  "inter_burst_gaps": [g for g in gaps if g["inter_burst"] is True],
                  "intra_burst_gaps": [g for g in gaps if g["inter_burst"] is False],
                  "inter_work_gaps": [g for g in gaps if g["inter_work"]],
                  "intra_work_gaps": [g for g in gaps if not g["inter_work"]]}
        statistics = {name: {"all_time": summarize(group, False),
                             "exposed": summarize(group, True)}
                      for name, group in groups.items()}
        if statistics["token_gaps"]["exposed"]["completed_samples"] < MIN_EXPOSED:
            reasons.append("fewer than 20 completed exposed token gaps")
        if statistics["inter_burst_gaps"]["exposed"]["completed_samples"] < MIN_EXPOSED:
            reasons.append("fewer than 20 completed exposed inter-burst service gaps")
        terminal_silence = None
        if len(terminals) == 1:
            origin = (tokens[-1] if tokens else
                      admits[0] if admits else submits[0] if submits else None)
            if origin and origin["time_ns"] <= terminals[0]["time_ns"]:
                start, end = origin["time_ns"], terminals[0]["time_ns"]
                terminal_silence = {
                    "origin": origin["event"], "start_ns": start, "end_ns": end,
                    "all_time_lower_bound_ns": end - start,
                    "exposed_lower_bound_ns": intersection(start, end, peer_intervals),
                    "all_time_stall_lower_bound_ns": max(0, end - start - STALL_NS),
                    "exposed_stall_lower_bound_ns": intersection(
                        start + STALL_NS, end, peer_intervals),
                }
        if terminals and request["ready"] and terminals[0]["time_ns"] < max(
                (end for start, end in peer_intervals
                 if any(start <= ready < end for ready in request["ready"])), default=0):
            reasons.append("ready decoder terminated before peer prefill ended")
        results.append({
            "run_id": run, "pid": pid, "request_id": request_id,
            "identities": [dict(zip(IDENTITY[1:], item)) for item in
                           sorted(request["identities"], key=repr)],
            "token_count": len(tokens), "gaps": gaps, **statistics,
            "terminal_censored_silence": terminal_silence,
            "dispatch_to_first_token_ns": tokens[0]["time_ns"] - submits[0]["time_ns"]
            if tokens and len(submits) == 1 and tokens[0]["time_ns"] >= submits[0]["time_ns"]
            else None,
            "peer_prefill_union_ns": peer_intervals,
            "responsiveness": {"status": "INCONCLUSIVE" if issues or reasons else "SUFFICIENT_SAMPLES",
                               "reasons": (["trace integrity incomplete; see issues"] if issues else []) + reasons},
        })
    for prefill in prefills:
        prefill["candidate_join"] = candidate_join(prefill, works, bool(issues))
    return {
        "schema": 1, "time_unit": "ns", "issues": issues,
        "diagnostic_sidecar_records": sidecar_count,
        "status": "INCONCLUSIVE" if issues or not results or any(
            r["responsiveness"]["status"] == "INCONCLUSIVE" for r in results)
        else "SUFFICIENT_SAMPLES",
        "requests": results, "prefills": prefills,
        "U": None, "L": None, "H": None,
        "limitations": [
            "Observed host intervals are correlation, not GPU/removable-work proof.",
            "Inter-burst gaps require explicit burst_id; inter-work gaps are a separate metric.",
            "No manifest, matched-arm comparison, throughput or numerical quality qualification.",
            "Candidate joins are an optimistic identity census on the same scheduler thread, not U/L/H proof.",
            "Only the first actual target cohort is eligible; unknown members are never inferred from adjacency.",
            "pd-profile-v1 sidecars are counted but not interpreted as performance or U evidence.",
            "Unmarked whole-record loss cannot be detected without an external completeness manifest.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="native perf schema:1 JSONL trace")
    args = parser.parse_args()
    try:
        records, issues = read_jsonl(args.trace)
    except (OSError, UnicodeError):
        records, issues = [], ["trace cannot be read as UTF-8 JSONL"]
    print(json.dumps(analyze(records, issues), indent=2))


if __name__ == "__main__":
    main()
