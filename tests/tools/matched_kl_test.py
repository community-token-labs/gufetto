#!/usr/bin/env python3
"""CPU CLI/streaming contracts; captures are made without the scoring path."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "quality"))
import matched_kl as tool  # noqa: E402


class CaptureFixture:
    def __init__(self, root, vocab=4):
        self.root = root
        self.history = root / "history.json"
        self.data = {
            "schema": "gufo-matched-history-v1", "vocab_size": vocab,
            "tokenizer_sha256": "a" * 64,
            "provenance": {"teacher": "independent analytic fixture", "seed": 47},
            "sequences": [
                {"id": "code", "prompt_id": "code-prompt", "phase": "continuation",
                 "stratum": "code/short", "conditioning": {"kind": "text"}, "tokens": [0, 1, 2, 3],
                 "positions": [1, 3]},
                {"id": "prose", "prompt_id": "prose-prompt", "phase": "prefill-boundary",
                 "stratum": "prose/short", "conditioning": {"kind": "text"}, "tokens": [3, 2, 1],
                 "positions": [2]},
            ],
        }
        self.save_history()

    def save_history(self):
        self.history.write_text(json.dumps(self.data), encoding="utf-8")

    def capture(self, name, dtype="f32le", delta=0.0, vocab=None):
        directory = self.root / name
        directory.mkdir()
        vocab = vocab or self.data["vocab_size"]
        header = {
            "schema": "gufo-full-logits-v1", "vocab_size": self.data["vocab_size"],
            "tokenizer_sha256": "a" * 64, "dtype": dtype,
            "positions": sum(len(s["positions"]) for s in self.data["sequences"]),
            "history_sha256": hashlib.sha256(self.history.read_bytes()).hexdigest(),
            "producer": {"implementation": name, "revision": "fixture-v1",
                         "model_sha256": "b" * 64, "arithmetic": "analytic logits",
                         "toolchain": "stdlib struct", "independence": "analytic, no Gufo"},
        }
        rows = [header]
        offset = 0
        with (directory / "chunk.bin").open("wb") as out:
            for seq in self.data["sequences"]:
                for position in seq["positions"]:
                    # Independent synthetic reference: logits [0,1,2,3,...].
                    # Candidate perturbs token0 only, independently of scoring.
                    values = [float(i % 4) for i in range(vocab)]
                    values[0] += delta
                    raw = struct.pack("<" + ("f" if dtype == "f32le" else "e") * vocab,
                                      *values)
                    out.write(raw)
                    tokens = seq["tokens"]
                    prefix = struct.pack("<" + "I" * position, *tokens[:position])
                    rows.append({"sequence_id": seq["id"], "position": position,
                                 "token": tokens[position], "phase": seq["phase"],
                                 "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                                 "conditioning_sha256": hashlib.sha256(json.dumps(
                                     seq["conditioning"], sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest(),
                                 "file": "chunk.bin", "offset": offset,
                                 "sha256": hashlib.sha256(raw).hexdigest()})
                    offset += len(raw)
        path = directory / "capture.jsonl"
        self.write_index(path, rows)
        return path, rows

    def contract(self, reference, candidate, name="contract.json", top_k=2):
        changes = {}
        for key in reference["producer"]:
            if reference["producer"][key] != candidate["producer"][key]:
                changes["/producer/" + key] = {"reference": reference["producer"][key],
                                               "candidate": candidate["producer"][key]}
        if reference["dtype"] != candidate["dtype"]:
            changes["/dtype"] = {"reference": reference["dtype"], "candidate": candidate["dtype"]}
        data = {"schema": "gufo-matched-quality-contract-v1",
                "history_sha256": reference["history_sha256"], "reference_header": reference,
                "expected_changes": changes, "top_k": top_k, "strata": {}}
        for seq in self.data["sequences"]:
            data["strata"][seq["stratum"]] = {"positions": len(seq["positions"]), "prompts": 1,
                                              "gates": {"kl_max": {"max": 10.0}}}
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @staticmethod
    def write_index(path, records):
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


class MatchedQualityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = CaptureFixture(self.root)
        self.reference, self.ref_records = self.fixture.capture("independent-reference")
        self.candidate, self.cand_records = self.fixture.capture("candidate", delta=4.0)
        self.contract = self.fixture.contract(self.ref_records[0], self.cand_records[0])

    def run_compare(self, name="result", **kwargs):
        return tool.compare(self.fixture.history, self.reference, self.candidate,
                            self.root / name, self.contract, top_k=2, **kwargs)

    def test_per_position_strata_provenance_and_metrics(self):
        report = self.run_compare()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["qualification"], "NOT_QUALIFIED")
        self.assertEqual(report["overall"]["prompts"], 2)
        self.assertEqual(report["overall"]["positions"], 3)
        self.assertEqual(report["strata"]["code/short"]["positions"], 2)
        self.assertEqual(report["strata"]["prose/short"]["positions"], 1)
        self.assertEqual(report["overall"]["top1_agreement"], 0)
        self.assertEqual(report["overall"]["topk_overlap"], 0.5)
        self.assertGreater(report["overall"]["kl_mean"], 0)
        self.assertEqual(report["reference"]["bytes_per_position"], 16)
        self.assertEqual(report["logit_byte_equal_positions"], 0)
        self.assertEqual(report["candidate"]["producer"], self.cand_records[0]["producer"])
        self.assertEqual(report["reference"]["index_sha256"],
                         hashlib.sha256(self.reference.read_bytes()).hexdigest())
        rows = [json.loads(s) for s in (self.root / "result/positions.jsonl").read_text().splitlines()]
        self.assertEqual([(r["sequence_id"], r["position"], r["token"]) for r in rows],
                         [("code", 1, 1), ("code", 3, 3), ("prose", 2, 1)])
        # Analytic probabilities from integer logits, not the scorer implementation.
        expected_logp = 1.0 - math.log(1 + math.e + math.e**2 + math.e**3)
        self.assertAlmostEqual(rows[0]["reference_logprob"], expected_logp)
        self.assertEqual(rows[0]["reference_rank"], 3)
        self.assertEqual(rows[0]["candidate_rank"], 4)
        self.assertFalse((self.root / "result/positions.partial.jsonl").exists())
        json.dumps(report, allow_nan=False)

    def test_equal_and_f16_captures(self):
        for dtype in ("f32le", "f16le"):
            with self.subTest(dtype=dtype):
                self.candidate, records = self.fixture.capture("same-" + dtype, dtype=dtype)
                self.contract = self.fixture.contract(self.ref_records[0], records[0])
                report = self.run_compare("out-" + dtype)
                self.assertEqual(report["overall"]["kl_max"], 0)
                self.assertEqual(report["overall"]["perplexity_ratio"], 1)
                self.assertEqual(report["overall"]["top1_agreement"], 1)
                self.assertEqual(report["logit_byte_equal_positions"], 3 if dtype == "f32le" else 0)
                self.assertEqual(report["candidate"]["bytes_per_position"],
                                 16 if dtype == "f32le" else 8)

    def test_budget_refuses_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            self.run_compare(max_positions=2)
        self.assertFalse((self.root / "result").exists())
        self.assertEqual(self.run_compare(max_positions=3)["overall"]["positions"], 3)

    def test_invalid_histories(self):
        original = deepcopy(self.fixture.data)
        mutations = [
            lambda d: d.update(schema="unknown"),
            lambda d: d.update(vocab_size=True),
            lambda d: d.update(vocab_size=1_000_001),
            lambda d: d.update(tokenizer_sha256="wrong"),
            lambda d: d.update(provenance={}),
            lambda d: d.update(sequences=[]),
            lambda d: d["sequences"].append(deepcopy(d["sequences"][0])),
            lambda d: d["sequences"][0].update(tokens=[0, True, 2, 3]),
            lambda d: d["sequences"][0].update(tokens=[0, 4, 2, 3]),
            lambda d: d["sequences"][0].update(positions=[]),
            lambda d: d["sequences"][0].update(positions=[0]),
            lambda d: d["sequences"][0].update(positions=[4]),
            lambda d: d["sequences"][0].update(positions=[1, 1]),
            lambda d: d["sequences"][0].update(positions=[3, 1]),
            lambda d: d["sequences"][0].update(stratum=""),
            lambda d: d["sequences"][0].update(conditioning={}),
        ]
        for mutate in mutations:
            self.fixture.data = deepcopy(original)
            mutate(self.fixture.data)
            self.fixture.save_history()
            with self.subTest(mutate=mutate):
                with self.assertRaises(ValueError):
                    tool.load_history(self.fixture.history, 100)

    def test_header_identity_and_independent_reference_required(self):
        mutations = [
            lambda r: r[0].update(schema="top20-v1"),
            lambda r: r[0].update(history_sha256="c" * 64),
            lambda r: r[0].update(tokenizer_sha256="c" * 64),
            lambda r: r[0].update(vocab_size=3),
            lambda r: r[0].update(positions=2),
            lambda r: r[0].update(dtype="sparse-top20"),
            lambda r: r[0]["producer"].update(implementation="independent-reference"),
            lambda r: r[0]["producer"].update(arithmetic=""),
            lambda r: r[0]["producer"].update(model_sha256=""),
        ]
        for i, mutate in enumerate(mutations):
            records = deepcopy(self.cand_records)
            mutate(records)
            self.fixture.write_index(self.candidate, records)
            with self.subTest(mutation=i):
                with self.assertRaises(ValueError):
                    self.run_compare("bad-header-" + str(i))
        self.fixture.write_index(self.candidate, self.cand_records)
        records = deepcopy(self.ref_records)
        del records[0]["producer"]["independence"]
        self.fixture.write_index(self.reference, records)
        with self.assertRaisesRegex(ValueError, "independence"):
            self.run_compare()

    def test_row_binding_and_missing_extra_duplicate_fail_closed(self):
        mutations = [
            lambda r: r[1].update(sequence_id="other"),
            lambda r: r[1].update(position=2),
            lambda r: r[1].update(position=True),
            lambda r: r[1].update(token=0),
            lambda r: r[1].update(prefix_sha256="c" * 64),
            lambda r: r[1].update(conditioning_sha256="c" * 64),
            lambda r: r[1].update(phase="wrong"),
            lambda r: r[1].update(sha256="c" * 64),
            lambda r: r[1].update(offset=-1),
            lambda r: r[1].update(offset=1),
            lambda r: r[1].update(offset=True),
            lambda r: r[1].update(file="absent.bin"),
            lambda r: r[1].update(file="../history.json"),
            lambda r: r[1].update(file=str(self.candidate.parent / "chunk.bin")),
            lambda r: r.pop(),
            lambda r: r.append(deepcopy(r[-1])),
            lambda r: r.insert(2, deepcopy(r[1])),
        ]
        for i, mutate in enumerate(mutations):
            records = deepcopy(self.cand_records)
            mutate(records)
            self.fixture.write_index(self.candidate, records)
            name = "bad-row-" + str(i)
            with self.subTest(mutation=i):
                with self.assertRaises(ValueError):
                    self.run_compare(name)
                self.assertFalse((self.root / name / "report.json").exists())
                failure = json.loads((self.root / name / "failure.json").read_text())
                self.assertEqual(failure["status"], "INVALID")

    def test_nonfinite_tail_even_with_correct_hash_fails(self):
        for i, value in enumerate((math.nan, math.inf, -math.inf)):
            for capture, original in ((self.reference, self.ref_records),
                                      (self.candidate, self.cand_records)):
                rows = deepcopy(original)
                data = struct.pack("<4f", 0, 1, 2, value)
                with (capture.parent / "chunk.bin").open("r+b") as out:
                    out.write(data)
                rows[1]["sha256"] = hashlib.sha256(data).hexdigest()
                self.fixture.write_index(capture, rows)
                with self.subTest(value=value, capture=capture):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self.run_compare(f"nonfinite-{i}-{capture.parent.name}")
                # Restore before testing the other side.
                delta = 0 if capture == self.reference else 4
                data = struct.pack("<4f", delta, 1, 2, 3)
                with (capture.parent / "chunk.bin").open("r+b") as out:
                    out.write(data)
                self.fixture.write_index(capture, original)

    def test_corrupt_truncated_and_escaped_files(self):
        chunk = self.candidate.parent / "chunk.bin"
        original = chunk.read_bytes()
        for i, data in enumerate((b"x" + original[1:], original[:5])):
            chunk.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "SHA-256|truncated"):
                self.run_compare("bad-bytes-" + str(i))
        chunk.write_bytes(original)
        link = self.candidate.parent / "escape.bin"
        try:
            link.symlink_to(self.reference.parent / "chunk.bin")
        except (OSError, NotImplementedError):
            return  # symlink creation may require privileges on Windows
        rows = deepcopy(self.cand_records)
        rows[1]["file"] = link.name
        self.fixture.write_index(self.candidate, rows)
        with self.assertRaisesRegex(ValueError, "inside"):
            self.run_compare("symlink-escape")

    def test_json_strictness_and_bounded_index_lines(self):
        for raw in (b'{"schema":1,"schema":2}', b'{"x":NaN}', b'[]', b'\n',
                    b'{"extra":1e999}', b'{"nested":[{"x":-1e999}]}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    tool.parse_json(raw)
        self.candidate.write_bytes(b" " * (tool.MAX_INDEX_LINE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "line too large"):
            self.run_compare()

    def test_json_overflow_never_publishes_success(self):
        lines = self.candidate.read_text().splitlines()
        lines[1] = lines[1][:-1] + ', "extra": 1e999}'
        self.candidate.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "nonfinite JSON number"):
            self.run_compare("overflow-row")
        self.assertFalse((self.root / "overflow-row/report.json").exists())
        failure = json.loads((self.root / "overflow-row/failure.json").read_text())
        self.assertEqual(failure["status"], "INVALID")
        self.fixture.history.write_text(self.fixture.history.read_text().replace('"seed": 47', '"seed": 1e999'))
        with self.assertRaisesRegex(ValueError, "nonfinite JSON number"):
            self.run_compare("overflow-history")
        self.assertFalse((self.root / "overflow-history/report.json").exists())
        path = self.root / "never-published.json"
        with self.assertRaises(ValueError):
            tool.write_json(path, {"status": "PASS", "invalid": math.inf})
        self.assertFalse(path.exists())
        self.assertFalse(path.with_name(path.name + ".partial").exists())

    def test_output_is_exclusive_not_unsafe_resume(self):
        first = self.run_compare()
        with self.assertRaises(FileExistsError):
            self.run_compare()
        self.assertEqual(json.loads((self.root / "result/report.json").read_text()), first)

    def test_cli_success_and_invalid_exit_codes(self):
        command = [sys.executable, "-B", str(ROOT / "tools/quality/matched_kl.py"),
                   "--history", str(self.fixture.history), "--reference", str(self.reference),
                   "--candidate", str(self.candidate), "--contract", str(self.contract),
                   "--top-k", "2"]
        good = subprocess.run(command + ["--out-dir", str(self.root / "cli")],
                              text=True, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertIn("PASS", good.stdout)
        self.assertIn("NOT_QUALIFIED", good.stdout)
        contract = json.loads(self.contract.read_text())
        contract["strata"]["code/short"]["gates"]["kl_max"]["max"] = 0
        self.contract.write_text(json.dumps(contract))
        failed = subprocess.run(command + ["--out-dir", str(self.root / "cli-fail")],
                                text=True, capture_output=True)
        self.assertEqual(failed.returncode, 1, failed.stderr)
        self.assertIn("FAIL", failed.stdout)
        self.assertTrue((self.root / "cli-fail/report.json").is_file())
        bad = subprocess.run(command + ["--out-dir", str(self.root / "cli-bad"),
                                        "--max-positions", "2"], text=True, capture_output=True)
        self.assertEqual(bad.returncode, 2, bad.stderr)
        self.assertIn("budget", bad.stderr)
        self.assertFalse((self.root / "cli-bad/report.json").exists())

    def test_image_conditioning_and_after_image_strata(self):
        image_root = self.root / "image"
        image_root.mkdir()
        fixture = CaptureFixture(image_root)
        fixture.data["sequences"][0].update(
            stratum="vision/short/cold/comparison/after-image",
            conditioning={"kind": "image", "image_sha256": "c" * 64,
                          "span": [0, 1], "mmproj_sha256": "d" * 64,
                          "preprocess": "fixture-v1", "embedding_sha256": "e" * 64})
        fixture.save_history()
        reference, ref_records = fixture.capture("image-reference")
        candidate, cand_records = fixture.capture("image-candidate")
        contract = fixture.contract(ref_records[0], cand_records[0])
        report = tool.compare(fixture.history, reference, candidate, image_root / "good", contract, top_k=2)
        self.assertEqual(report["strata"]["vision/short/cold/comparison/after-image"]["positions"], 2)
        self.assertEqual(report["strata"]["prose/short"]["positions"], 1)
        self.assertEqual(report["status"], "PASS")
        # Same token prefix is insufficient: substituting text-only conditioning
        # for an image row must be rejected independently of the logit hashes.
        cand_records[1]["conditioning_sha256"] = cand_records[-1]["conditioning_sha256"]
        fixture.write_index(candidate, cand_records)
        with self.assertRaisesRegex(ValueError, "conditioning_sha256"):
            tool.compare(fixture.history, reference, candidate, image_root / "bad", contract, top_k=2)
        self.assertFalse((image_root / "bad/report.json").exists())

    def test_full_qwen_width_reads_one_row_at_a_time(self):
        root = self.root / "wide"
        root.mkdir()
        fixture = CaptureFixture(root, vocab=248320)
        fixture.data["sequences"] = [dict(fixture.data["sequences"][0], positions=[1])]
        fixture.save_history()
        reference, ref_records = fixture.capture("wide-independent")
        candidate, cand_records = fixture.capture("wide-candidate")
        contract = fixture.contract(ref_records[0], cand_records[0], top_k=20)
        original_open = Path.open
        read_sizes = []

        class BoundedReader:
            def __init__(self, source):
                self.source = source
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.source.close()
            def seek(self, offset):
                return self.source.seek(offset)
            def read(self, size=-1):
                self_test.assertEqual(size, 993280)
                read_sizes.append(size)
                return self.source.read(size)

        self_test = self
        def checked_open(path, *args, **kwargs):
            opened = original_open(path, *args, **kwargs)
            return BoundedReader(opened) if path.name == "chunk.bin" else opened

        with patch.object(Path, "open", checked_open):
            result = tool.compare(fixture.history, reference, candidate, root / "out", contract, top_k=20)
        self.assertEqual(read_sizes, [993280, 993280])
        self.assertEqual(result["overall"]["kl_max"], 0)
        self.assertEqual(result["logit_byte_equal_positions"], 1)


if __name__ == "__main__":
    unittest.main()
