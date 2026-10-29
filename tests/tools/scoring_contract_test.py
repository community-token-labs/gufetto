#!/usr/bin/env python3
"""Independent stdlib contract fixtures; imports no CLI or metric implementation.

Run: python3 -B tests/tools/scoring_contract_test.py
"""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scoring_contract", ROOT / "tools" / "quality" / "scoring_contract.py"
)
assert SPEC is not None and SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)
HISTORY = "a1" * 32
REFERENCE = {"model": {"sha256": "b2" * 32, "backend": "cpu"},
             "identity": [1, "fixture", None], "empty": {}}


def registration():
    return {
        "schema": "gufo-matched-quality-contract-v1",
        "history_sha256": HISTORY,
        "top_k": 20,
        "reference_header": deepcopy(REFERENCE),
        "expected_changes": {"/model/backend": {"reference": "cpu", "candidate": "npu"}},
        "strata": {
            "short": {"positions": 2, "prompts": 1,
                      "gates": {"kl_mean": {"min": 0, "max": 0.1}}},
            "long": {"positions": 10000, "prompts": 100,
                     "gates": {"top1_agreement": {"min": 0.9}}},
        },
    }


def candidate():
    result = deepcopy(REFERENCE)
    result["model"]["backend"] = "npu"
    return result


def observations():
    return ({"short": {"kl_mean": 0.05}, "long": {"top1_agreement": 1}},
            {"short": {"positions": 2, "prompts": 1},
             "long": {"positions": 10000, "prompts": 100}})


class MetadataContractTest(unittest.TestCase):
    def validate(self, contract=None, reference=None, changed=None, history=HISTORY):
        CONTRACT.validate_contract(
            registration() if contract is None else contract,
            REFERENCE if reference is None else reference,
            candidate() if changed is None else changed, history,
        )

    def test_exact_registration_and_identity_case(self):
        self.assertIsNone(self.validate())
        contract = registration()
        contract["expected_changes"] = {}
        self.validate(contract, changed=deepcopy(REFERENCE))

    def test_unexpected_and_missing_changes(self):
        changed = candidate()
        changed["model"]["sha256"] = "c3" * 32
        with self.assertRaises(ValueError):
            self.validate(changed=changed)
        contract = registration()
        contract["expected_changes"] = {}
        with self.assertRaises(ValueError):
            self.validate(contract)

    def test_stale_wrong_and_no_op_exemptions(self):
        for change in ({"reference": "gpu", "candidate": "npu"},
                       {"reference": "cpu", "candidate": "gpu"},
                       {"reference": "cpu", "candidate": "cpu"}):
            with self.subTest(change=change):
                contract = registration()
                contract["expected_changes"]["/model/backend"] = change
                with self.assertRaises(ValueError):
                    self.validate(contract)
        contract = registration()
        contract["expected_changes"]["/absent"] = {"reference": 0, "candidate": 1}
        with self.assertRaises(ValueError):
            self.validate(contract)
        contract = registration()
        contract["expected_changes"]["/identity"] = {
            "reference": deepcopy(REFERENCE["identity"]),
            "candidate": deepcopy(REFERENCE["identity"]),
        }
        with self.assertRaises(ValueError):
            self.validate(contract)

    def test_pointer_escaping_arrays_empty_dict_and_type_changes(self):
        reference = {"a/b": {"~x": 1}, "": [True], "empty": {}}
        changed = {"a/b": {"~x": 1.0}, "": [1], "empty": None}
        contract = registration()
        contract["reference_header"] = deepcopy(reference)
        contract["expected_changes"] = {
            "/a~1b/~0x": {"reference": 1, "candidate": 1.0},
            "/": {"reference": [True], "candidate": [1]},
            "/empty": {"reference": {}, "candidate": None},
        }
        self.validate(contract, reference, changed)
        for pointer in ("", "a/b", "/a~", "/a~2b", "/a~01b/~0x"):
            with self.subTest(pointer=pointer):
                invalid = deepcopy(contract)
                invalid["expected_changes"][pointer] = invalid["expected_changes"].pop("/a~1b/~0x")
                with self.assertRaises(ValueError):
                    self.validate(invalid, reference, changed)

    def test_dict_key_sets_cannot_be_exempted(self):
        for reference, changed, path in (
            ({"a": {}}, {"a": {"new": 1}}, "/a"),
            ({"a": {"old": 1}}, {"a": {}}, "/a"),
            ({"old": 1}, {"new": 1}, ""),
        ):
            contract = registration()
            contract["reference_header"] = reference
            contract["expected_changes"] = {
                path: {"reference": reference.get("a", reference),
                       "candidate": changed.get("a", changed)}
            }
            with self.assertRaises(ValueError):
                self.validate(contract, reference, changed)

    def test_full_reference_and_history_binding(self):
        for key, value in (("model", {"sha256": "c3" * 32, "backend": "cpu"}),
                           ("identity", [True, "fixture", None]),
                           ("empty", {"unexpected": 1})):
            contract = registration()
            contract["reference_header"][key] = value
            with self.assertRaises(ValueError):
                self.validate(contract)
        for value in ("", "a" * 63, "z" * 64, None, True, "a1" * 31 + "b2"):
            with self.subTest(history=value):
                with self.assertRaises(ValueError):
                    self.validate(history=value)
                contract = registration()
                contract["history_sha256"] = value
                with self.assertRaises(ValueError):
                    self.validate(contract)

    def test_bool_and_number_changes_are_strict(self):
        for old, new in ((True, 1), (False, 0), (1, 1.0)):
            contract = registration()
            contract["reference_header"] = {"x": old}
            contract["expected_changes"] = {"/x": {"reference": old, "candidate": new}}
            self.validate(contract, {"x": old}, {"x": new})
            contract["expected_changes"] = {}
            with self.assertRaises(ValueError):
                self.validate(contract, {"x": old}, {"x": new})

    def test_finite_json_objects_only(self):
        for invalid in (float("nan"), float("inf"), -float("inf"),
                        (1,), {1: "non-string key"}, {1, 2}, object()):
            for destination in ("reference", "candidate", "contract"):
                with self.subTest(invalid=repr(invalid), destination=destination):
                    contract, reference, changed = registration(), deepcopy(REFERENCE), candidate()
                    if destination == "contract":
                        contract["expected_changes"]["/model/backend"]["candidate"] = invalid
                    else:
                        header = reference if destination == "reference" else changed
                        header["identity"] = [invalid]
                    with self.assertRaises(ValueError):
                        self.validate(contract, reference, changed)
        for value in ([], None, True, "header"):
            with self.assertRaises(ValueError):
                CONTRACT.validate_contract(registration(), value, candidate(), HISTORY)
            with self.assertRaises(ValueError):
                CONTRACT.validate_contract(registration(), REFERENCE, value, HISTORY)
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises(ValueError):
            self.validate(changed=cyclic)

    def test_malformed_registration(self):
        for key in registration():
            contract = registration()
            del contract[key]
            with self.assertRaises(ValueError):
                self.validate(contract)
        for key, value in (("schema", "unknown"), ("ignore", ["/model/backend"]),
                           ("expected_changes", []), ("reference_header", []),
                           ("strata", {})):
            contract = registration()
            contract[key] = value
            with self.assertRaises(ValueError):
                self.validate(contract)
        contract = registration()
        contract["expected_changes"]["/model/backend"]["ignore"] = True
        with self.assertRaises(ValueError):
            self.validate(contract)


class StratumGateTest(unittest.TestCase):
    def evaluate(self, contract=None, summaries=None, counts=None):
        defaults, default_counts = observations()
        return CONTRACT.evaluate_strata(
            registration() if contract is None else contract,
            defaults if summaries is None else summaries,
            default_counts if counts is None else counts,
        )

    def test_inclusive_thresholds_and_retained_decisions(self):
        for value in (0, 0.1):
            summaries, _ = observations()
            summaries["short"]["kl_mean"] = value
            result = self.evaluate(summaries=summaries)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["strata"]["short"], {
                "status": "PASS", "positions": 2, "prompts": 1,
                "gates": {"kl_mean": {"thresholds": {"min": 0, "max": 0.1},
                                       "value": value, "passed": True, "reason": None}},
            })

    def test_one_failed_small_stratum_is_not_averaged_away(self):
        for value, reason in ((-1e-20, "below minimum"),
                              (math.nextafter(0.1, math.inf), "above maximum")):
            summaries, _ = observations()
            summaries["short"]["kl_mean"] = value
            result = self.evaluate(summaries=summaries)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["strata"]["long"]["status"], "PASS")
            gate = result["strata"]["short"]["gates"]["kl_mean"]
            self.assertFalse(gate["passed"])
            self.assertEqual(gate["reason"], reason)

    def test_stratum_sets_and_exact_counts(self):
        for target in ("summaries", "counts"):
            for operation in ("missing", "extra"):
                summaries, counts = observations()
                mapping = summaries if target == "summaries" else counts
                if operation == "missing":
                    del mapping["short"]
                else:
                    mapping["extra"] = deepcopy(mapping["short"])
                with self.assertRaises(ValueError):
                    self.evaluate(summaries=summaries, counts=counts)
        for key in ("positions", "prompts"):
            for value in (True, 1.0, 0, -1, None, 999):
                _, counts = observations()
                counts["short"][key] = value
                with self.assertRaises(ValueError):
                    self.evaluate(counts=counts)
            _, counts = observations()
            del counts["short"][key]
            with self.assertRaises(ValueError):
                self.evaluate(counts=counts)
        _, counts = observations()
        counts["short"]["extra"] = 1
        with self.assertRaises(ValueError):
            self.evaluate(counts=counts)

    def test_missing_null_nonfinite_and_nonnumeric_metrics_fail(self):
        contract = registration()
        contract["strata"]["short"]["gates"] = {"perplexity_ratio": {"max": 2}}
        for summary in ({}, {"perplexity_ratio": None}, {"perplexity_ratio": math.inf},
                        {"perplexity_ratio": -math.inf}, {"perplexity_ratio": math.nan},
                        {"perplexity_ratio": True}, {"perplexity_ratio": "1"}):
            summaries, _ = observations()
            summaries["short"] = summary
            result = self.evaluate(contract, summaries)
            self.assertEqual(result["status"], "FAIL")
            gate = result["strata"]["short"]["gates"]["perplexity_ratio"]
            self.assertFalse(gate["passed"])
            self.assertTrue(gate["reason"])
            self.assertEqual(gate["thresholds"], {"max": 2})

    def test_empty_gates_never_pass_and_failure_takes_precedence(self):
        contract = registration()
        contract["strata"]["short"]["gates"] = {}
        result = self.evaluate(contract)
        self.assertEqual(result["status"], "NOT_EVALUATED")
        self.assertEqual(result["strata"]["short"]["status"], "NOT_EVALUATED")
        self.assertEqual(result["strata"]["short"]["gates"], {})
        summaries, _ = observations()
        summaries["long"]["top1_agreement"] = 0
        self.assertEqual(self.evaluate(contract, summaries)["status"], "FAIL")
        contract["strata"]["long"]["gates"] = {}
        self.assertEqual(self.evaluate(contract)["status"], "NOT_EVALUATED")

    def test_all_allowed_metrics_and_no_implicit_gates(self):
        metrics = ("kl_mean", "kl_p50", "kl_p95", "kl_p99", "kl_max",
                   "top1_agreement", "topk_overlap", "reference_nll", "candidate_nll",
                   "nll_delta", "perplexity_ratio")
        contract = registration()
        contract["strata"]["short"]["gates"] = {m: {"min": -2, "max": 2} for m in metrics}
        summaries, _ = observations()
        summaries["short"] = {m: -2 for m in metrics}
        summaries["short"]["unregistered"] = math.nan
        self.assertEqual(self.evaluate(contract, summaries)["status"], "PASS")

    def test_invalid_gate_and_stratum_shapes(self):
        for gates in ([], {"latency": {"max": 1}}, {"kl_mean": {}},
                      {"kl_mean": {"min": 2, "max": 1}},
                      {"kl_mean": {"max": True}}, {"kl_mean": {"max": math.nan}},
                      {"kl_mean": {"min": math.inf}}, {"kl_mean": {"max": None}},
                      {"kl_mean": {"max": "1"}}, {"kl_mean": {"tolerance": 0.1}}):
            contract = registration()
            contract["strata"]["short"]["gates"] = gates
            with self.assertRaises(ValueError):
                self.evaluate(contract)
        for key in ("positions", "prompts"):
            for value in (True, 0, -1, 1.0, None):
                contract = registration()
                contract["strata"]["short"][key] = value
                with self.assertRaises(ValueError):
                    self.evaluate(contract)
        contract = registration()
        contract["strata"]["short"]["ignore"] = True
        with self.assertRaises(ValueError):
            self.evaluate(contract)

    def test_inputs_are_not_mutated(self):
        contract, summaries, counts = registration(), *observations()
        before = deepcopy((contract, summaries, counts))
        result = self.evaluate(contract, summaries, counts)
        result["strata"]["short"]["gates"]["kl_mean"]["thresholds"]["max"] = 9
        self.assertEqual((contract, summaries, counts), before)


if __name__ == "__main__":
    unittest.main()
