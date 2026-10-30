#!/usr/bin/env python3
"""CPU-only card contracts; synthetic fixture, not performance evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gufo.benchmark_card import render


class CardTest(unittest.TestCase):
    def setUp(self):
        self.raw = (ROOT / "tests/fixtures/benchmark-card-serving.json").read_bytes()
        self.meta = {
            "model": "synthetic", "artifacts": [{"name": "target", "sha256": "a" * 64}],
            "stack": {key: "explicit fixture value" for key in
                      ("runtime", "hip", "rocr", "compiler", "commit", "build_flags", "hardware")},
            "settings": {"context": 4096}, "quality": {"status": "PASS", "evidence": "fixture"}
        }

    def test_individual_phases(self):
        card = render(self.raw, self.meta)
        self.assertIn("| 1 / 0 / 0 | 1 | 20 | 10 | 21 | 34 |", card)
        self.assertIn("| 1 / 1 / 0 | 2 | 0 | 14 | 3 | 18 |", card)
        self.assertIn("Card status: **PASS**", card)

    def test_unknown_stack_not_guessed(self):
        meta = copy.deepcopy(self.meta)
        meta["stack"] = {}
        card = render(self.raw, meta)
        self.assertIn("Card status: **INCONCLUSIVE**", card)
        self.assertIn("- hip: UNKNOWN", card)

    def test_statuses_and_digest(self):
        for status in ("FAIL", "INCONCLUSIVE", "NOT_RUN"):
            self.meta["quality"]["status"] = status
            self.assertIn("Quality status: **" + status + "**", render(self.raw, self.meta))
        self.meta["artifacts"][0]["sha256"] = "bad"
        with self.assertRaises(ValueError):
            render(self.raw, self.meta)

    def test_missing_metrics(self):
        data = json.loads(self.raw)
        data["results"]["1"]["samples"][0]["decode_ms"] = None
        self.assertIn("Card status: **INCONCLUSIVE**", render(json.dumps(data).encode(), self.meta))
        data["results"]["1"]["samples"][0]["decode_ms"] = -1
        with self.assertRaises(ValueError):
            render(json.dumps(data).encode(), self.meta)

    def test_reject_reference_and_unknown_json(self):
        for data in ({"artifactType": "invented-bench-json"}, {"artifactType": "model-bench-table", "target": "reference"}):
            with self.assertRaises(ValueError):
                render(json.dumps(data).encode(), self.meta)
        data = json.loads(self.raw)
        data["workload"]["endpointProfile"] = "openai"
        with self.assertRaises(ValueError):
            render(json.dumps(data).encode(), self.meta)

    def test_actual_retained_table_schema(self):
        path = ROOT / "docs/models/qwen3.8-27b/artifacts/single-ar-q4-gufo.json"
        card = render(path.read_bytes(), self.meta)
        self.assertIn("Aggregate-only source", card)
        self.assertIn("Card status: **INCONCLUSIVE**", card)
        self.assertIn("NOT RECORDED", card)

    def test_native_and_markdown_tables(self):
        for text in (
            "| model | size | backend | test | t/s |\n| --- | --- | --- | --- | --- |\n| fixture | 1 GiB | ROCm (HIP) | pp2048 | 12.34 ± 0.20 |",
            "| Depth | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) |\n| --- | --- | --- | --- | --- |\n| 0 | 42 | 999 | -10% | 12 |"
        ):
            card = render(text.encode(), self.meta)
            self.assertIn("Card status: **INCONCLUSIVE**", card)
            self.assertNotIn("999", card)
            self.assertNotIn("-10%", card)
        mixed = ("| Depth | Gufo pp (tok/s) |\n| --- | --- |\n| 0 | 42 |\n\n"
                 "| Reference only | Rate |\n| --- | --- |\n| other | 999 |\n")
        self.assertNotIn("999", render(mixed.encode(), self.meta))
        with self.assertRaises(ValueError):
            render(b"unrecognized", self.meta)


if __name__ == "__main__":
    unittest.main()
