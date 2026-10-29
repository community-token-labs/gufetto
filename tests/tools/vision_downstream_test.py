#!/usr/bin/env python3
"""Exercise three-arm vision downstream scoring through the real #697 reader."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/quality"))
from matched_kl_test import CaptureFixture
import vision_downstream as tool


class VisionDownstreamTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = CaptureFixture(self.root)
        self.images = self.root / "images.json"
        self.images.write_text(json.dumps({"schema": "gufo-vision-images-v1", "cases": [
            {"id": "synthetic", "sha256": "c" * 64, "prompt": "Name the colors."}]}))
        self.fixture.data["provenance"]["image_set_sha256"] = hashlib.sha256(self.images.read_bytes()).hexdigest()
        self.fixture.data["sequences"][0].update(
            phase="after-image", stratum="fixture/after-image",
            conditioning={"kind": "image", "case_id": "synthetic", "image_sha256": "c" * 64,
                          "prompt_sha256": hashlib.sha256(b"Name the colors.").hexdigest(),
                          "expanded_spans": [[0, 1]]})
        self.fixture.data["sequences"][1].update(phase="text", stratum="fixture/text")
        self.fixture.save_history()

    def captures(self, delta=0):
        self.ref, r = self.fixture.capture("reference")
        self.current, b = self.fixture.capture("current")
        self.candidate, c = self.fixture.capture("candidate", delta=delta)
        for path, records, is_reference in ((self.ref, r, True), (self.current, b, False),
                                            (self.candidate, c, False)):
            records[0]["producer"].update(
                vision_accumulator_dtype="f32", vision_input_origin="pixels",
                vision_independent=is_reference,
                vision_source_sha256=("d" if is_reference else "e") * 64,
                vision_independence="independent analytic fixture" if is_reference else "tested path")
            self.fixture.write_index(path, records)
        self.bc = self.fixture.contract(r[0], b[0], name="current-contract.json")
        self.cc = self.fixture.contract(r[0], c[0], name="candidate-contract.json")
        for path in (self.bc, self.cc):
            contract = json.loads(path.read_text())
            for stratum in contract["strata"].values():
                stratum["gates"] = {"kl_max": {"max": 0}, "top1_agreement": {"min": 1}}
            path.write_text(json.dumps(contract))

    def run_comparison(self):
        return tool.run(self.images, self.fixture.history, self.ref, self.current, self.candidate,
                        self.bc, self.cc, self.root / "out", top_k=2)

    def test_all_three_arms_and_after_image_text_strata(self):
        self.captures()
        report = self.run_comparison()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["qualification"], "NOT_QUALIFIED")
        for arm in ("current", "candidate"):
            self.assertEqual(set(report["comparisons"][arm]["strata"]),
                             {"fixture/after-image", "fixture/text"})
            self.assertTrue((self.root / "out" / arm / "positions.jsonl").is_file())

    def test_candidate_failure_not_rescued_by_current(self):
        self.captures(delta=4)
        report = self.run_comparison()
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["comparisons"]["current"]["status"], "PASS")
        self.assertEqual(report["comparisons"]["candidate"]["status"], "FAIL")

    def test_image_coverage_and_positions_rejected_before_capture(self):
        changes = [lambda s: s[0]["conditioning"].update(image_sha256="d" * 64),
                   lambda s: s[0]["conditioning"].update(prompt_sha256="d" * 64),
                   lambda s: s[0]["conditioning"].update(expanded_spans=[[0, 2]]),
                   lambda s: s[0]["conditioning"].update(expanded_spans=[[0, 1], [0, 1]]),
                   lambda s: s[0]["conditioning"].update(case_id="unregistered"),
                   lambda s: s[0].update(phase="text", conditioning={"kind": "text"}),
                   lambda s: s[1].update(phase="arbitrary")]
        original = json.dumps(self.fixture.data)
        for edit in changes:
            self.fixture.data = json.loads(original)
            edit(self.fixture.data["sequences"])
            self.fixture.save_history()
            with self.assertRaises(ValueError):
                tool.run(self.images, self.fixture.history, Path("missing"), Path("missing"),
                         Path("missing"), Path("missing"), Path("missing"), self.root / "out")
            self.assertFalse((self.root / "out").exists())

    def test_shared_image_stratum_rejected_before_scoring(self):
        # Both cases are covered, but a good case must not dilute a bad one's KL.
        images = json.loads(self.images.read_text())
        second = dict(images["cases"][0], id="second", width=2048, height=2048)
        images["cases"].append(second)
        self.images.write_text(json.dumps(images))
        self.fixture.data["provenance"]["image_set_sha256"] = hashlib.sha256(self.images.read_bytes()).hexdigest()
        sequence = json.loads(json.dumps(self.fixture.data["sequences"][0]))
        sequence.update(id="second-sequence", prompt_id="second-prompt")
        sequence["conditioning"]["case_id"] = "second"
        self.fixture.data["sequences"].append(sequence)
        self.fixture.save_history()
        self.captures()
        with self.assertRaisesRegex(ValueError, "stratum pools"):
            self.run_comparison()
        self.assertFalse((self.root / "out").exists())

    def test_invalid_vision_reference_provenance_is_rejected(self):
        self.captures()
        records = [json.loads(line) for line in self.ref.read_text().splitlines()]
        original = json.dumps(records)
        for update in ({"vision_accumulator_dtype": "bf16"},
                       {"vision_input_origin": "candidate_hidden"},
                       {"vision_independent": False},
                       {"vision_source_sha256": "e" * 64}):
            records = json.loads(original)
            records[0]["producer"].update(update)
            self.fixture.write_index(self.ref, records)
            with self.assertRaisesRegex(ValueError, "vision"):
                self.run_comparison()
            self.assertFalse((self.root / "out").exists())

    def test_model_identity_is_matched(self):
        self.captures()
        records = [json.loads(line) for line in self.candidate.read_text().splitlines()]
        records[0]["producer"]["model_sha256"] = "f" * 64
        self.fixture.write_index(self.candidate, records)
        with self.assertRaisesRegex(ValueError, "unmatched downstream model"):
            self.run_comparison()
        self.assertFalse((self.root / "out").exists())

    def test_history_image_set_pin(self):
        self.captures()
        self.images.write_text(self.images.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "frozen image set"):
            self.run_comparison()

    def test_bad_second_capture_retains_first_without_success_summary(self):
        self.captures()
        self.candidate.write_text(self.candidate.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "missing capture"):
            self.run_comparison()
        self.assertTrue((self.root / "out/current/report.json").exists())
        self.assertTrue((self.root / "out/failure.json").exists())
        self.assertFalse((self.root / "out/report.json").exists())


if __name__ == "__main__":
    unittest.main()
