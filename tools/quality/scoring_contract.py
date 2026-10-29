"""Strict preregistered metadata and per-stratum quality contracts.

No numerical defaults or tolerances are supplied. These decisions do not
qualify transport, state, task correctness, or performance. Mode support and
independent-reference checks belong to the consumer, not this module.
"""

import math
import re


SCHEMA = "gufo-matched-quality-contract-v1"
GATE_METRICS = frozenset({
    "kl_mean", "kl_p50", "kl_p95", "kl_p99", "kl_max",
    "top1_agreement", "topk_overlap", "reference_nll", "candidate_nll",
    "nll_delta", "perplexity_ratio",
})


def _finite_number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _json(value, location, active=None):
    """Reject non-JSON Python objects, nonfinite floats, and cycles."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) not in (dict, list):
        raise ValueError(f"{location}: expected a finite JSON value")
    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ValueError(f"{location}: cyclic JSON value")
    active.add(identity)
    try:
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{location}: JSON object keys must be strings")
                _json(item, f"{location}/{_escape(key)}", active)
        else:
            for index, item in enumerate(value):
                _json(item, f"{location}/{index}", active)
    finally:
        active.remove(identity)


def _object(value, location):
    if type(value) is not dict:
        raise ValueError(f"{location}: expected a JSON object")


def _keys(value, expected, location):
    _object(value, location)
    if set(value) != set(expected):
        raise ValueError(f"{location}: expected exactly keys {sorted(expected)}")


def _equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _escape(key):
    return key.replace("~", "~0").replace("/", "~1")


def _hash(value, location):
    if type(value) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{location}: expected a hex64 SHA-256")


def _positive_int(value, location):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{location}: expected a positive integer")


def _shape(contract):
    _keys(contract, {"schema", "history_sha256", "reference_header",
                     "expected_changes", "strata", "top_k"}, "contract")
    _json(contract, "contract")
    if contract["schema"] != SCHEMA:
        raise ValueError(f"contract.schema: expected {SCHEMA}")
    _hash(contract["history_sha256"], "contract.history_sha256")
    _positive_int(contract["top_k"], "contract.top_k")
    _object(contract["reference_header"], "contract.reference_header")
    _object(contract["expected_changes"], "contract.expected_changes")
    for path, change in contract["expected_changes"].items():
        if not path.startswith("/") or re.search(r"~(?:[^01]|$)", path):
            raise ValueError(f"expected_changes: invalid JSON pointer {path!r}")
        _keys(change, {"reference", "candidate"}, f"expected_changes[{path!r}]")
        if _equal(change["reference"], change["candidate"]):
            raise ValueError(f"expected_changes[{path!r}]: no-op change")
    strata = contract["strata"]
    _object(strata, "contract.strata")
    if not strata:
        raise ValueError("contract.strata: register at least one stratum")
    for name, spec in strata.items():
        if not name:
            raise ValueError("contract.strata: stratum names must be nonempty")
        _keys(spec, {"positions", "prompts", "gates"}, f"strata[{name!r}]")
        for key in ("positions", "prompts"):
            _positive_int(spec[key], f"strata[{name!r}].{key}")
        _object(spec["gates"], f"strata[{name!r}].gates")
        for metric, bounds in spec["gates"].items():
            if metric not in GATE_METRICS:
                raise ValueError(f"strata[{name!r}]: unsupported gate {metric!r}")
            _object(bounds, f"gate {metric}")
            if not bounds or set(bounds) - {"min", "max"}:
                raise ValueError(f"gate {metric}: expected min and/or max")
            if not all(_finite_number(value) for value in bounds.values()):
                raise ValueError(f"gate {metric}: thresholds must be finite numbers")
            if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
                raise ValueError(f"gate {metric}: min exceeds max")


def _differences(reference, candidate, path=""):
    if type(reference) is dict and type(candidate) is dict:
        if reference.keys() != candidate.keys():
            raise ValueError(f"header{path}: dictionary key sets differ")
        result = {}
        for key in reference:
            result.update(_differences(reference[key], candidate[key],
                                       f"{path}/{_escape(key)}"))
        return result
    if _equal(reference, candidate):
        return {}
    return {path: {"reference": reference, "candidate": candidate}}


def validate_contract(contract: dict, reference_header: dict,
                      candidate_header: dict, history_sha256: str) -> None:
    """Bind full reference/history identities and require the exact change map.

    Arrays are leaves; corresponding objects must have identical key sets.
    Scalar types are significant, including int versus float and bool versus int.
    Unknown contract fields (including ignore metadata) are rejected.
    """
    _shape(contract)
    _hash(history_sha256, "history_sha256")
    for name, header in (("reference_header", reference_header),
                         ("candidate_header", candidate_header)):
        _object(header, name)
        _json(header, name)
    if contract["history_sha256"] != history_sha256:
        raise ValueError("history_sha256 does not match the preregistered history")
    if not _equal(contract["reference_header"], reference_header):
        raise ValueError("reference_header does not match the full preregistered header")
    actual = _differences(reference_header, candidate_header)
    if not _equal(actual, contract["expected_changes"]):
        raise ValueError("header differences do not exactly match expected_changes")


def evaluate_strata(contract: dict, summaries: dict[str, dict],
                    counts: dict[str, dict]) -> dict:
    """Evaluate only explicit gates, with inclusive bounds and no aggregation.

    Count/stratum mismatches and malformed registrations raise ValueError.
    A gate result contains thresholds, the actual value, passed, and reason.
    Missing/nonfinite values are represented by None with an explicit failure
    reason, so reports remain standard JSON.
    Call validate_contract separately to bind the actual capture metadata.
    """
    _shape(contract)
    strata = contract["strata"]
    _keys(summaries, strata, "summaries")
    _keys(counts, strata, "counts")
    decisions = {}
    for name, spec in strata.items():
        _object(summaries[name], f"summaries[{name!r}]")
        _keys(counts[name], {"positions", "prompts"}, f"counts[{name!r}]")
        for key in ("positions", "prompts"):
            _positive_int(counts[name][key], f"counts[{name!r}].{key}")
            if counts[name][key] != spec[key]:
                raise ValueError(f"counts[{name!r}].{key}: preregistered count mismatch")
        gates = {}
        for metric, bounds in spec["gates"].items():
            value = summaries[name].get(metric)
            reason = None
            if metric not in summaries[name]:
                reason = "missing metric"
            elif value is None:
                reason = "null metric"
            elif not _finite_number(value):
                reason = "metric is not a finite number"
            elif "min" in bounds and value < bounds["min"]:
                reason = "below minimum"
            elif "max" in bounds and value > bounds["max"]:
                reason = "above maximum"
            gates[metric] = {"thresholds": dict(bounds),
                             "value": value if _finite_number(value) else None,
                             "passed": reason is None, "reason": reason}
        status = ("NOT_EVALUATED" if not gates else
                  "PASS" if all(gate["passed"] for gate in gates.values()) else "FAIL")
        decisions[name] = {"status": status, "positions": counts[name]["positions"],
                           "prompts": counts[name]["prompts"], "gates": gates}
    statuses = {decision["status"] for decision in decisions.values()}
    overall = ("FAIL" if "FAIL" in statuses else
               "NOT_EVALUATED" if "NOT_EVALUATED" in statuses else "PASS")
    return {"status": overall, "strata": decisions}
