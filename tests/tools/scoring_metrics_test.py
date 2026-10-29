#!/usr/bin/env python3
"""Independent, stdlib-only numerical oracles for offline logit metrics.

Run directly: python3 -B tests/tools/scoring_metrics_test.py
The Decimal oracle constructs probabilities using direct exponentiation and
computes ln(p/q); it shares no normalization or reduction code with the scorer.
"""

from copy import deepcopy
from decimal import Decimal, localcontext
import importlib.util
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "scoring_metrics", ROOT / "tools" / "quality" / "scoring_metrics.py"
)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def decimal_oracle(reference, candidate, token):
    """100-digit full-distribution oracle, intentionally not log-softmax."""
    with localcontext() as context:
        context.prec = 100
        ref_weights = [Decimal.from_float(float(x)).exp() for x in reference]
        cand_weights = [Decimal.from_float(float(x)).exp() for x in candidate]
        ref_total = sum(ref_weights, Decimal(0))
        cand_total = sum(cand_weights, Decimal(0))
        ref_probs = [x / ref_total for x in ref_weights]
        cand_probs = [x / cand_total for x in cand_weights]
        kl = sum((p * (p / q).ln() for p, q in zip(ref_probs, cand_probs)), Decimal(0))
        ref_logprob = ref_probs[token].ln()
        cand_logprob = cand_probs[token].ln()
        return {
            "kl": float(kl),
            "reference_logprob": float(ref_logprob),
            "candidate_logprob": float(cand_logprob),
            "nll_delta": float(ref_logprob - cand_logprob),
        }


def summary_row(kl=0.0, reference_logprob=-1.0, candidate_logprob=-1.0,
                agreement=True, overlap=1.0):
    """Synthetic fixtures; never call the implementation to build expectations."""
    return {
        "kl": kl,
        "reference_logprob": reference_logprob,
        "candidate_logprob": candidate_logprob,
        "nll_delta": reference_logprob - candidate_logprob,
        "reference_rank": 1,
        "candidate_rank": 1 if agreement else 2,
        "reference_top1": 0,
        "candidate_top1": 0 if agreement else 1,
        "top1_agreement": agreement,
        "topk_overlap": overlap,
        "reference_topk": [0, 1],
        "candidate_topk": [0, 1] if agreement else [1, 2],
        "max_abs_logit_delta": 1.0,
    }


class CompareLogitsTest(unittest.TestCase):
    def assert_json(self, value):
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)

    def test_decimal_full_distribution_oracle(self):
        cases = [
            ([0.0, 1.0, -2.0], [-0.5, 2.0, 0.25]),
            ([3.0, 3.0, 3.0, 3.0], [1.0, 1.0, -1.0, 1.0]),
            ([1000.0, 0.0, -1000.0], [-1000.0, 1000.0, 0.0]),
            ([0.0, -700.0, -745.0, -1000.0], [-2.0, 0.0, -800.0, -900.0]),
            ([1e-8, 0.0, -1e-8], [0.0, -1e-8, 1e-8]),
            ([-900.0, -901.0, -902.0], [901.0, 900.0, 902.0]),
        ]
        for reference, candidate in cases:
            for token in range(len(reference)):
                with self.subTest(reference=reference, candidate=candidate, token=token):
                    result = METRICS.compare_logits(reference, candidate, token, 2)
                    expected = decimal_oracle(reference, candidate, token)
                    for key, value in expected.items():
                        self.assertAlmostEqual(result[key], value, delta=2e-12 * max(1, abs(value)))
                    self.assertGreaterEqual(result["kl"], 0.0)
                    self.assertEqual(result["max_abs_logit_delta"],
                                     max(abs(r - c) for r, c in zip(reference, candidate)))
                    self.assert_json(result)

    def test_exact_output_contract(self):
        result = METRICS.compare_logits([1.0, 0.0], [0.0, 1.0], 0, 1)
        self.assertEqual(set(result), {
            "kl", "reference_logprob", "candidate_logprob", "nll_delta",
            "reference_rank", "candidate_rank", "reference_top1", "candidate_top1",
            "reference_top1_margin",
            "top1_agreement", "topk_overlap", "reference_topk", "candidate_topk",
            "max_abs_logit_delta",
        })
        self.assertIs(result["top1_agreement"], False)
        self.assertEqual(result["reference_rank"], 1)
        self.assertEqual(result["reference_top1_margin"], 1.0)
        self.assertEqual(result["candidate_rank"], 2)
        self.assertAlmostEqual(result["nll_delta"], 1.0)
        self.assertEqual(result["topk_overlap"], 0.0)

    def test_identity_single_token_and_no_mutation(self):
        for logits in ([5.0], [2.0, 1.0, -3.0], [1e300, 1e300, 1e300]):
            saved = list(logits)
            result = METRICS.compare_logits(logits, logits, len(logits) - 1, len(logits))
            self.assertEqual(result["kl"], 0.0)
            self.assertEqual(result["nll_delta"], 0.0)
            self.assertEqual(result["max_abs_logit_delta"], 0.0)
            self.assertEqual(result["topk_overlap"], 1.0)
            self.assertIs(result["top1_agreement"], True)
            self.assertEqual(logits, saved)
            self.assert_json(result)
        singleton = METRICS.compare_logits([42.0], [-42.0], 0, 1)
        self.assertEqual(singleton["reference_logprob"], 0.0)
        self.assertEqual(singleton["candidate_logprob"], 0.0)
        self.assertEqual(singleton["reference_rank"], 1)
        self.assertEqual(singleton["candidate_topk"], [0])
        self.assertIsNone(singleton["reference_top1_margin"])

    def test_ties_rank_and_topk_boundary(self):
        reference = [2.0, 4.0, 4.0, 4.0, -1.0]
        candidate = [4.0, 4.0, -1.0, 4.0, 2.0]
        ref_order = [1, 2, 3, 0, 4]
        cand_order = [0, 1, 3, 4, 2]
        for token in range(5):
            for k in range(1, 6):
                result = METRICS.compare_logits(reference, candidate, token, k)
                self.assertEqual(result["reference_rank"], ref_order.index(token) + 1)
                self.assertEqual(result["candidate_rank"], cand_order.index(token) + 1)
                self.assertEqual(result["reference_topk"], ref_order[:k])
                self.assertEqual(result["candidate_topk"], cand_order[:k])
                self.assertEqual(result["reference_top1"], 1)
                self.assertEqual(result["candidate_top1"], 0)
                overlap = len(set(ref_order[:k]) & set(cand_order[:k])) / k
                self.assertEqual(result["topk_overlap"], overlap)

    def test_analytic_uniform_large_offsets(self):
        # Reconstructing max + log(sum(exp(shifted))) would lose ln(4).
        result = METRICS.compare_logits([1e300] * 4, [-1e300] * 4, 3, 2)
        self.assertEqual(result["reference_logprob"], -math.log(4))
        self.assertEqual(result["candidate_logprob"], -math.log(4))
        self.assertEqual(result["reference_topk"], [0, 1])
        self.assertEqual(result["reference_rank"], 4)
        self.assertEqual(result["kl"], 0.0)
        self.assertEqual(result["max_abs_logit_delta"], 2e300)

    def test_independent_shift_invariance(self):
        reference = [0.0, -0.25, 2.0, -4.0]
        candidate = [1.0, 0.0, -1.0, -3.0]
        expected = decimal_oracle(reference, candidate, 3)
        baseline = METRICS.compare_logits(reference, candidate, 3, 2)
        result = METRICS.compare_logits(
            [x + 2**50 for x in reference], [x - 2**49 for x in candidate], 3, 2
        )
        for key in baseline:
            if key != "max_abs_logit_delta":
                self.assertEqual(result[key], baseline[key])
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value, delta=1e-14)
        self.assertNotEqual(result["max_abs_logit_delta"], baseline["max_abs_logit_delta"])

    def test_asymmetry_and_tail_divergence_invisible_to_topk(self):
        # First two logits and top-k IDs match. Only the full tail changes.
        reference = [4.0, 3.0, 0.0, -1.0, -2.0]
        candidate = [4.0, 3.0, -20.0, -30.0, -40.0]
        forward = METRICS.compare_logits(reference, candidate, 4, 2)
        reverse = METRICS.compare_logits(candidate, reference, 4, 2)
        self.assertIs(forward["top1_agreement"], True)
        self.assertEqual(forward["topk_overlap"], 1.0)
        self.assertGreater(forward["kl"], 0.4)
        self.assertGreater(forward["kl"], reverse["kl"])
        for result, ref, cand in ((forward, reference, candidate), (reverse, candidate, reference)):
            expected = decimal_oracle(ref, cand, 4)
            self.assertAlmostEqual(result["kl"], expected["kl"], delta=1e-14)

    def test_invalid_vocabularies_and_indices(self):
        for reference, candidate in (([], []), ([0.0], []), ([], [0.0]), ([0.0], [0.0, 1.0])):
            with self.subTest(reference=reference, candidate=candidate):
                with self.assertRaises(ValueError):
                    METRICS.compare_logits(reference, candidate, 0, 1)
        for token in (-1, 2, True, False, 0.0, 1.0, "0", None):
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    METRICS.compare_logits([0.0, 1.0], [0.0, 1.0], token, 1)
        for k in (-1, 0, 3, True, False, 1.0, "1", None):
            with self.subTest(top_k=k):
                with self.assertRaises(ValueError):
                    METRICS.compare_logits([0.0, 1.0], [0.0, 1.0], 0, k)

    def test_all_logits_validated_even_outside_topk_and_teacher(self):
        for bad in (math.nan, math.inf, -math.inf, True, "1", None, 10**400):
            for side in range(2):
                for position in range(3):
                    inputs = [[2.0, 1.0, -1.0], [2.0, 1.0, -1.0]]
                    inputs[side][position] = bad
                    with self.subTest(bad=bad, side=side, position=position):
                        with self.assertRaises(ValueError):
                            METRICS.compare_logits(*inputs, token=0, top_k=1)

    def test_unrepresentable_finite_arithmetic_rejected(self):
        for reference, candidate in (
            ([1e308, -1e308], [0.0, 0.0]),  # centered logit overflow
            ([1e308], [-1e308]),  # raw logit delta overflow
        ):
            with self.assertRaises(ValueError):
                METRICS.compare_logits(reference, candidate, 0, 1)


class SummarizeTest(unittest.TestCase):
    def test_position_weighting_quantiles_and_nll(self):
        rows = [
            summary_row(0.0, -1.0, -2.0, True, 1.0),
            summary_row(1.0, -2.0, -4.0, False, 0.5),
            summary_row(2.0, -3.0, -3.0, False, 0.0),
            summary_row(9.0, -4.0, -7.0, True, 0.5),
        ]
        # Caller metadata (e.g. grouping size) must not reweight positions.
        rows[0]["group_size"] = 100
        saved = deepcopy(rows)
        result = METRICS.summarize(rows)
        expected = {
            "positions": 4, "kl_mean": 3.0, "kl_p50": 1.5,
            "kl_p95": 7.95, "kl_p99": 8.79, "kl_max": 9.0,
            "top1_agreement": 0.5, "topk_overlap": 0.5,
            "reference_nll": 2.5, "candidate_nll": 4.0, "nll_delta": 1.5,
            "perplexity_ratio": math.exp(1.5), "perplexity_ratio_overflow": False,
        }
        self.assertEqual(set(result), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value, places=13)
        self.assertIs(result["perplexity_ratio_overflow"], False)
        self.assertEqual(rows, saved)
        self.assertEqual(result, METRICS.summarize(list(reversed(rows))))
        json.dumps(result, allow_nan=False)

    def test_singleton_and_two_point_interpolation(self):
        single = METRICS.summarize([summary_row(3.0)])
        for key in ("kl_mean", "kl_p50", "kl_p95", "kl_p99", "kl_max"):
            self.assertEqual(single[key], 3.0)
        self.assertEqual(single["positions"], 1)
        pair = METRICS.summarize([summary_row(10.0), summary_row(0.0)])
        self.assertEqual(pair["kl_p50"], 5.0)
        self.assertEqual(pair["kl_p95"], 9.5)
        self.assertEqual(pair["kl_p99"], 9.9)

    def test_perplexity_overflow_is_explicit_standard_json(self):
        result = METRICS.summarize([summary_row(1.0, -1.0, -1001.0)])
        self.assertEqual(result["nll_delta"], 1000.0)
        self.assertIsNone(result["perplexity_ratio"])
        self.assertIs(result["perplexity_ratio_overflow"], True)
        self.assertIn('"perplexity_ratio": null', json.dumps(result, allow_nan=False))
        finite = METRICS.summarize([summary_row(1.0, -1.0, -701.0)])
        self.assertEqual(finite["perplexity_ratio"], math.exp(700.0))
        self.assertIs(finite["perplexity_ratio_overflow"], False)
        underflow = METRICS.summarize([summary_row(1.0, -1001.0, -1.0)])
        self.assertEqual(underflow["perplexity_ratio"], 0.0)
        self.assertIs(underflow["perplexity_ratio_overflow"], False)

    def test_large_finite_means_do_not_overflow_from_summing(self):
        rows = [summary_row(1e308, -1e308, -1e308) for _ in range(4)]
        result = METRICS.summarize(rows)
        self.assertEqual(result["kl_mean"], 1e308)
        self.assertEqual(result["reference_nll"], 1e308)
        self.assertEqual(result["candidate_nll"], 1e308)
        self.assertEqual(result["perplexity_ratio"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_subnormal_means_and_equal_quantiles(self):
        tiny = math.ulp(0.0)
        result = METRICS.summarize([summary_row(tiny, -tiny, -tiny) for _ in range(4)])
        for key in ("kl_mean", "kl_p50", "kl_p95", "kl_p99", "kl_max",
                    "reference_nll", "candidate_nll"):
            self.assertEqual(result[key], tiny)

    def test_empty_and_nonfinite_rows_rejected(self):
        with self.assertRaises(ValueError):
            METRICS.summarize([])
        for key in ("kl", "reference_logprob", "candidate_logprob", "nll_delta",
                    "topk_overlap", "max_abs_logit_delta"):
            for bad in (math.nan, math.inf, -math.inf, True):
                row = summary_row()
                row[key] = bad
                with self.subTest(key=key, bad=bad):
                    with self.assertRaises(ValueError):
                        METRICS.summarize([row])
        row = summary_row()
        row["top1_agreement"] = 1
        with self.assertRaises(ValueError):
            METRICS.summarize([row])


if __name__ == "__main__":
    unittest.main()
