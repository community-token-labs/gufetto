#!/usr/bin/env python3
"""Stdlib-only independent analytic/Decimal checks; no production fixtures.

Run: python3 -B tests/tools/vision_reference_test.py
Decimal expectations evaluate the real formula at 90 digits, with direct exp
and no shared F32 implementation. Exact cancellation cases separately pin the
specified F32 left-to-right rounding, rather than treating real arithmetic as
an exact floating-point oracle. Tolerances below are unit-test roundoff bounds,
not model qualification thresholds.
"""

from copy import deepcopy
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "quality" / "vision_reference.py"
SPEC = importlib.util.spec_from_file_location("vision_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
attention_f32 = REFERENCE.attention_f32


def decimal_attention(q, k, v, scale=None):
    """Independent high-precision real formula, deliberately no max subtraction."""
    with localcontext() as context:
        context.prec = 90
        factor = (Decimal(1) / Decimal(len(q[0])).sqrt() if scale is None
                  else Decimal.from_float(float(scale)))
        result = []
        for query in q:
            weights = []
            for key in k:
                dot = sum((Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                           for a, b in zip(query, key)), Decimal(0))
                weights.append((dot * factor).exp())
            denominator = sum(weights, Decimal(0))
            result.append([
                float(sum((weight * Decimal.from_float(float(row[column]))
                           for weight, row in zip(weights, v)), Decimal(0)) / denominator)
                for column in range(len(v[0]))
            ])
        return result


class AttentionTest(unittest.TestCase):
    def assert_rows_close(self, actual, expected, tolerance=5e-7):
        self.assertEqual(len(actual), len(expected))
        for row, wanted in zip(actual, expected):
            self.assertEqual(len(row), len(wanted))
            for value, target in zip(row, wanted):
                self.assertAlmostEqual(value, target, delta=tolerance * max(1, abs(target)))

    def test_uniform_average_and_full_noncausal(self):
        q = [[0, 0], [0, 0]]
        k = [[3, -2], [1, 4], [-1, 2], [8, 7]]
        v = [[1, 8], [3, 4], [5, 0], [7, -4]]
        self.assertEqual(attention_f32(q, k, v), [[4.0, 2.0], [4.0, 2.0]])
        # First query must see the last key; no triangular/causal masking.
        self.assertEqual(attention_f32([[1]], [[0], [0]], [[0], [2]]), [[1.0]])

    def test_single_key_and_no_mutation(self):
        q, k, v = [[1, -1], [-3, 2], [0, 0]], [[4, 2]], [[1.25, -2.5, 8]]
        saved = deepcopy((q, k, v))
        result = attention_f32(q, k, v)
        self.assertEqual(result, [[1.25, -2.5, 8.0]] * 3)
        self.assertEqual((q, k, v), saved)
        result[0][0] = 0
        self.assertEqual(result[1][0], 1.25)
        self.assertEqual(v[0][0], 1.25)

    def test_symmetry_and_permuted_keys(self):
        q, k, v = [[1], [-1], [0]], [[-1], [1]], [[-2], [2]]
        result = attention_f32(q, k, v, scale=0.5)
        self.assertEqual(result[0][0], -result[1][0])
        self.assertEqual(result[2], [0.0])
        self.assertEqual(result, attention_f32(q, k[::-1], v[::-1], scale=0.5))
        self.assert_rows_close(result, decimal_attention(q, k, v, 0.5))

    def test_nonuniform_decimal_small_cases(self):
        cases = [
            ([[1]], [[0], [1]], [[0], [1]], 1),
            ([[0.25, -0.5], [-0.75, 1]], [[1, 0], [0, 1], [-1, 1]],
             [[1, -2], [2, 0.5], [-0.25, 3]], None),
            ([[0.5, 0.25]], [[0.25, -0.5], [1, 0.5], [-1, 1]],
             [[3], [-1], [0.25]], -0.75),
            ([[2]], [[3], [-2], [0]], [[1], [4], [-2]], 0),
        ]
        for q, k, v, scale in cases:
            with self.subTest(q=q, scale=scale):
                self.assert_rows_close(attention_f32(q, k, v, scale),
                                       decimal_attention(q, k, v, scale))

    def test_unsaturated_probability_not_bf16(self):
        result = attention_f32([[1]], [[0], [1]], [[0], [1]], scale=1)[0][0]
        self.assertAlmostEqual(result, 0.7310585786300049, delta=6e-8)
        # Analytic exp(-1), then binary32 exp/denominator/division rounding.
        self.assertEqual(result, 0.7310585975646973)
        self.assertGreater(result, 0.7)
        self.assertLess(result, 0.8)
        self.assertNotEqual(result, 0.73046875)  # BF16 probability is not the oracle.

    def test_f32_dot_cancellation_order(self):
        # ((2**24 + 1) - 2**24) rounds to 0, not the real-arithmetic 1.
        q = [[1, 1, 1]]
        k = [[2**24, 1, -(2**24)], [0, 0, 0]]
        self.assertEqual(attention_f32(q, k, [[0], [2]], scale=1), [[1.0]])
        self.assertLess(decimal_attention(q, k, [[0], [2]], 1)[0][0], 0.6)
        # Moving the small term last preserves it; pins dimension traversal.
        reordered = [[2**24, -(2**24), 1], [0, 0, 0]]
        self.assert_rows_close(attention_f32(q, reordered, [[0], [2]], scale=1),
                               decimal_attention(q, reordered, [[0], [2]], 1))

    def test_f32_product_not_fused(self):
        # (1+2**-23)*(1-2**-23) rounds to 1 before adding -1.
        q = [[-1, 1 + 2**-23, 2**-46]]
        k = [[1, 1 - 2**-23, 1], [0, 0, 0]]
        # Scaling turns the tiny surviving term into a nonsaturated score of 1.
        self.assert_rows_close(attention_f32(q, k, [[0], [1]], scale=2**46),
                               [[0.2689414213699951]])
        self.assertEqual(decimal_attention(q, k, [[0], [1]], 2**46), [[0.5]])

    def test_f32_pv_cancellation_and_input_rounding(self):
        # Equal quarter weights produce [2**24, 1, -2**24, 1].
        q, k, v = [[0]], [[0]] * 4, [[2**26], [4], [-(2**26)], [4]]
        self.assertEqual(attention_f32(q, k, v), [[1.0]])
        self.assertEqual(decimal_attention(q, k, v), [[2.0]])
        self.assertEqual(attention_f32([[0]], [[0]], [[1 + 2**-24]]), [[1.0]])

    def test_stable_max_and_underflow(self):
        self.assertEqual(attention_f32([[1]], [[10000], [10000]], [[2], [4]], 1), [[3.0]])
        self.assertEqual(attention_f32([[1]], [[-10000], [-10000]], [[2], [4]], 1), [[3.0]])
        self.assertEqual(attention_f32([[1]], [[1000], [-1000]], [[2], [4]], 1), [[2.0]])

    def test_lengths_1_8_9_32_33_and_dim72(self):
        for queries, keys in ((1, 33), (8, 32), (9, 9), (32, 8), (33, 1)):
            with self.subTest(queries=queries, keys=keys):
                q = [[((i + d) % 7 - 3) / 16 for d in range(72)] for i in range(queries)]
                k = [[((j * 3 + d) % 9 - 4) / 16 for d in range(72)] for j in range(keys)]
                v = [[((j + c) % 11 - 5) / 8 for c in range(3)] for j in range(keys)]
                result = attention_f32(q, k, v)
                self.assert_rows_close(result, decimal_attention(q, k, v), tolerance=1e-6)
                self.assertEqual(result, attention_f32(q, k, v, scale=0.1178511301977579))

    def test_invalid_shapes_and_types(self):
        invalid = [None, [], [[]], [[1], []], [[1], [1, 2]], [1], "x", [[True]],
                   [["1"]], [[None]], [[1j]], [[float("nan")]], [[float("inf")]],
                   [[-float("inf")]], [[1e40]], [[10**400]]]
        for matrix in invalid:
            for position in range(3):
                with self.subTest(matrix=matrix, position=position):
                    args = [[[0]], [[0]], [[0]]]
                    args[position] = matrix
                    with self.assertRaises(ValueError):
                        attention_f32(*args)
        for args in (([[1, 2]], [[1]], [[1]]), ([[1]], [[1], [2]], [[1]])):
            with self.assertRaises(ValueError):
                attention_f32(*args)
        for scale in (True, "1", float("nan"), float("inf"), -float("inf"), 1e40, 1j):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                attention_f32([[0]], [[0]], [[0]], scale=scale)

    def test_intermediate_overflow_rejected(self):
        cases = [
            ([[1e30]], [[1e30]], [[1]], 1),  # product
            ([[2e38, 2e38]], [[1, 1]], [[1]], 1),  # dot sum
            ([[2e38]], [[1]], [[1]], 2),  # scaling
            ([[1]], [[2e38], [-2e38]], [[1], [1]], 1),  # max subtraction
        ]
        for q, k, v, scale in cases:
            with self.subTest(q=q), self.assertRaises(ValueError):
                attention_f32(q, k, v, scale)

    def test_budget_is_strict_and_precedes_numeric_work(self):
        for budget in (0, -1, True, False, 13.0, 13.5, float("inf"), float("nan"), "13", None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                attention_f32([[0]], [[0]], [[0]], max_work=budget)
        # Three input scalars, one output, nine scalar arithmetic operations.
        self.assertEqual(attention_f32([[0]], [[0]], [[7]], max_work=13), [[7.0]])
        with patch.object(REFERENCE, "_number", side_effect=AssertionError("numeric work")):
            with self.assertRaisesRegex(ValueError, "exceeds max_work"):
                attention_f32([[0]], [[0]], [[7]], max_work=12)
            with self.assertRaisesRegex(ValueError, "exceeds max_work"):
                attention_f32([[0] * 72] * 256, [[0] * 72] * 256, [[0]] * 256)


class CliTest(unittest.TestCase):
    def run_cli(self, raw, *args):
        return subprocess.run([sys.executable, "-B", "-S", str(SCRIPT), "-", *map(str, args)],
                              input=raw, text=True, capture_output=True, timeout=15)

    def test_stdout_provenance(self):
        raw = '{"q":[[0]],"k":[[0],[0]],"v":[[2],[4]],"scale":1}'
        run = self.run_cli(raw)
        self.assertEqual(run.returncode, 0, run.stderr)
        document = json.loads(run.stdout)
        self.assertEqual(document["output"], [[3.0]])
        provenance = document["provenance"]
        self.assertEqual(provenance["source_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
        self.assertEqual(provenance["input_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertIs(provenance["operator_only"], True)
        self.assertIs(provenance["qualification"], False)
        self.assertIn("no Gufo or production imports", provenance["independence"])
        self.assertIn("no BF16 rounding", provenance["arithmetic"])
        self.assertIn("pixels", provenance["limitation"])
        self.assertIn("not candidate hidden states", provenance["limitation"])

    def test_new_output_and_no_overwrite(self):
        raw = '{"q":[[0]],"k":[[0]],"v":[[2]]}'
        with tempfile.TemporaryDirectory() as directory:
            input_path, output_path = Path(directory) / "input.json", Path(directory) / "out.json"
            input_path.write_text(raw, encoding="utf-8")
            run = subprocess.run([sys.executable, "-B", "-S", str(SCRIPT), str(input_path),
                                  "--output", str(output_path)], capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout, "")
            self.assertEqual(json.loads(output_path.read_text())["output"], [[2.0]])
            before = output_path.read_bytes()
            rerun = self.run_cli(raw, "--output", output_path)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(output_path.read_bytes(), before)
            self.assertNotEqual(self.run_cli(raw, "--output", input_path).returncode, 0)
            self.assertEqual(input_path.read_text(), raw)

    def test_bad_json_and_bounded_input(self):
        valid = '{"q":[[0]],"k":[[0]],"v":[[2]]}'
        bad = ["{", "[]", "{}", valid[:-1] + ',"other":0}',
               valid[:-1] + ',"scale":null}', valid[:-1] + ',"q":[[1]]}',
               valid.replace("2", "NaN"), valid.replace("2", "Infinity"),
               valid.replace("2", "1e400"), "[" * 2000 + "]" * 2000,
               " " * (REFERENCE.MAX_INPUT_BYTES + 1)]
        for raw in bad:
            with self.subTest(raw=raw[:80]):
                run = self.run_cli(raw)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(run.stdout, "")
                self.assertNotIn("Traceback", run.stderr)
        for budget in ("12", "0", "1.5", "inf"):
            run = self.run_cli(valid, "--max-work", budget)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual(run.stdout, "")


if __name__ == "__main__":
    unittest.main()
