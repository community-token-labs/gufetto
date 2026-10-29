#!/usr/bin/env python3
"""CPU analytic metrics and fail-closed three-arm tensor transport fixtures."""

import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "quality"
sys.path.insert(0, str(TOOLS))
import vision_compare as tool


def save(path, value):
    path.write_text(json.dumps(value) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture:
    def __init__(self, root):
        self.root = root
        self.plan = {"schema": "gufo-vision-tensors-v1", "scope": "encoder",
                     "image_set_sha256": "a" * 64, "tensors": []}
        for ident, layer in (("image/layer0", 0), ("image/embedding", -1)):
            self.plan["tensors"].append({"id": ident, "case_id": "image", "stage": ident.split("/")[1],
                                          "stratum": ident, "layer": layer, "head": -1,
                                          "width": 3, "row_indices": [0, 8], "input_sha256": "b" * 64})
        self.plan_path = root / "plan.json"
        plan_hash = save(self.plan_path, self.plan)
        self.contract = {"schema": "gufo-vision-gates-v1", "plan_sha256": plan_hash,
                         "status": "PROPOSED", "approval": "pending lead decision",
                         "relative_floor": 1e-8, "headers": {}, "tensors": {}}
        self.indices = {}
        for arm in tool.ARMS:
            directory = root / arm
            directory.mkdir()
            producer = {"implementation": arm, "revision": "fixture", "arithmetic": "f32",
                        "toolchain": "python", "model_sha256": "c" * 64,
                        "source_sha256": ("d" if arm == "reference" else "e") * 64}
            if arm == "reference":
                producer.update(accumulator_dtype="f32", input_origin="pixels", independent=True,
                                independence="independent analytic fixture, not a model run")
            header = {"schema": "gufo-vision-capture-v1", "plan_sha256": plan_hash,
                      "role": arm, "dtype": "f32le", "rows": 4, "producer": producer}
            self.contract["headers"][arm] = header
            raw = struct.pack("<3f", 1.0, -2.0, 0.0)
            (directory / "rows.bin").write_bytes(raw * 4)
            records = [header]
            for tensor in self.plan["tensors"]:
                for index in tensor["row_indices"]:
                    records.append({"tensor_id": tensor["id"], "row": index, "file": "rows.bin",
                                    "offset": (len(records) - 1) * len(raw),
                                    "sha256": hashlib.sha256(raw).hexdigest()})
            self.indices[arm] = records
            self.save_index(arm)
        for tensor in self.plan["tensors"]:
            self.contract["tensors"][tensor["id"]] = {
                pair: {"atol": 0.125, "rtol": 0, "relative_l2": 0.25} for pair in tool.PAIRS}
            self.contract["tensors"][tensor["id"]]["noninferiority"] = {"factor": 1.05, "slack": 0.125}
        self.contract_path = root / "contract.json"
        self.save_contract()

    def save_index(self, arm):
        (self.root / arm / "capture.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in self.indices[arm]))

    def save_contract(self):
        save(self.contract_path, self.contract)

    def run(self, **kwargs):
        return tool.compare(self.plan_path, self.contract_path,
                            *(self.root / arm / "capture.jsonl" for arm in tool.ARMS),
                            self.root / "out", **kwargs)

    def change_row(self, arm, position, values):
        raw = struct.pack("<3f", *values)
        with (self.root / arm / "rows.bin").open("r+b") as out:
            out.seek((position - 1) * len(raw))
            out.write(raw)
        self.indices[arm][position]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.save_index(arm)


class VisionCompareTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_analytic_metrics(self):
        result = tool.row_metrics([3.0, 4.0], [6.0, 8.0], 1e-8)
        self.assertEqual(result["max_abs"], 4)
        self.assertEqual(result["max_relative"], 1)
        self.assertEqual(result["relative_l2"], 1)
        self.assertEqual(result["reference_rms"], 5 / math.sqrt(2))
        result = tool.row_metrics([0.0, 0.0], [0.0, 1e-9], 1e-8)
        self.assertAlmostEqual(result["max_relative"], 0.1)
        self.assertAlmostEqual(result["relative_l2"], 0.1)

    def test_equal_three_arms_and_exact_coverage(self):
        result = self.fixture.run()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["registration_status"], "PROPOSED")
        self.assertEqual(result["qualification"], "NOT_QUALIFIED")
        self.assertEqual(result["rows"], 4)
        rows = [json.loads(line) for line in (self.root / "out/rows.jsonl").read_text().splitlines()]
        self.assertEqual([r["row"] for r in rows], [0, 8, 0, 8])
        self.assertTrue(all(r["passed"] for r in rows))

    def test_one_bad_layer_cannot_be_pooled_away(self):
        self.fixture.change_row("candidate", 4, [1, -3, 0])
        result = self.fixture.run()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["tensors"]["image/layer0"]["status"], "PASS")
        bad = result["tensors"]["image/embedding"]
        self.assertEqual(bad["failed_rows"], 1)
        self.assertEqual(bad["pairs"]["candidate_reference"]["worst_abs_row"], 8)

    def test_noninferiority_blocks_even_with_loose_pair_gates(self):
        for gates in self.fixture.contract["tensors"].values():
            gates["noninferiority"]["slack"] = 0
        self.fixture.save_contract()
        self.fixture.change_row("candidate", 1, [1.0625, -2, 0])
        self.assertEqual(self.fixture.run()["status"], "FAIL")

    def test_current_failure_also_blocks(self):
        self.fixture.change_row("current", 1, [2, -2, 0])
        self.assertEqual(self.fixture.run()["status"], "FAIL")

    def test_nonfinite_with_valid_hash_never_succeeds(self):
        self.fixture.change_row("reference", 1, [float("nan"), -2, 0])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.fixture.run()
        self.assertFalse((self.root / "out/report.json").exists())
        self.assertTrue((self.root / "out/failure.json").exists())

    def test_missing_extra_wrong_order_hash_offset_path(self):
        edits = [lambda r: r.pop(), lambda r: r.append(r[-1]),
                 lambda r: r[1].update(row=9), lambda r: r[1].update(row=True),
                 lambda r: r[1].update(sha256="f" * 64), lambda r: r[1].update(offset=1),
                 lambda r: r[1].update(file="../plan.json"),
                 lambda r: r[1].update(file="/etc/passwd"),
                 lambda r: r[1].update(offset=1024)]
        for index, edit in enumerate(edits):
            with self.subTest(index=index):
                root = self.root / str(index)
                root.mkdir()
                fixture = Fixture(root)
                edit(fixture.indices["candidate"])
                fixture.save_index("candidate")
                with self.assertRaises(ValueError):
                    fixture.run()
                self.assertFalse((root / "out/report.json").exists())

    def test_reference_and_contract_fail_closed(self):
        edits = [lambda c: c["headers"].pop("reference"),
                 lambda c: c["headers"]["reference"]["producer"].update(independent=False),
                 lambda c: c["headers"]["reference"]["producer"].update(accumulator_dtype="bf16"),
                 lambda c: c["headers"]["reference"]["producer"].update(input_origin="candidate_hidden"),
                 lambda c: c["headers"]["reference"]["producer"].update(source_sha256="e" * 64),
                 lambda c: c["headers"]["candidate"]["producer"].update(model_sha256="f" * 64),
                 lambda c: c["tensors"].pop("image/layer0"),
                 lambda c: c["tensors"]["image/layer0"]["candidate_reference"].update(atol=-1),
                 lambda c: c.update(relative_floor=0), lambda c: c.update(plan_sha256="f" * 64)]
        for index, edit in enumerate(edits):
            with self.subTest(index=index):
                contract = copy.deepcopy(self.fixture.contract)
                edit(contract)
                with self.assertRaises(ValueError):
                    tool.validate(self.fixture.plan, contract,
                                  self.fixture.contract["plan_sha256"], 200000)

    def test_budget_and_output_exclusivity(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            self.fixture.run(max_rows=3)
        self.fixture.run()
        with self.assertRaises(FileExistsError):
            self.fixture.run()

    def test_json_overflow_and_header_metadata_mismatch(self):
        self.fixture.indices["candidate"][0]["extra"] = 2
        self.fixture.save_index("candidate")
        with self.assertRaisesRegex(ValueError, "header"):
            self.fixture.run()
        self.fixture.contract_path.write_text('{"extra":1e999}')
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.fixture.run()

    def test_cli(self):
        command = [sys.executable, str(TOOLS / "vision_compare.py"), "--plan", str(self.fixture.plan_path),
                   "--contract", str(self.fixture.contract_path), "--out-dir", str(self.root / "cli")]
        for arm in tool.ARMS:
            command += ["--" + arm, str(self.root / arm / "capture.jsonl")]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PROPOSED; NOT_QUALIFIED", result.stdout)


if __name__ == "__main__":
    unittest.main()
