"""Offline contract tests for the multi-registration functional probe, not model qualification."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pinned_prefix


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.short = self.root / "short.txt"
        self.long = self.root / "long.txt"
        self.short.write_text("A ")
        self.long.write_text("A B ")
        self.calls = []

    def call(self, url, payload):
        self.calls.append(payload)
        prompt = payload["prompt"]
        cached = max((len(text) for text in ("A ", "A B ")
                      if prompt.startswith(text)), default=0)
        if not payload["cache_prompt"]:
            cached = 0
        prefill = len(prompt) - cached
        return {"content": hashlib.sha256((prompt + str(payload["temperature"])).encode()).hexdigest(),
                "tokens_predicted": 32, "tokens_cached": cached,
                "tokens_evaluated": len(prompt), "timings": {"prompt_n": prefill},
                "probe": {"cached_tokens": cached, "prefill_tokens": prefill}}

    def run_probe(self, output, files=None, restore=None, extra=()):
        files = files or (self.short, self.long)
        args = ["pinned_prefix.py", "--output", str(output)]
        for path in files:
            args += ["--prefix-file", str(path), "--prefix-tokens", str(len(path.read_text()))]
        if restore:
            args += ["--restore", str(restore)]
        with patch("sys.argv", args + list(extra)), patch.object(
                pinned_prefix, "call", side_effect=self.call), contextlib.redirect_stdout(io.StringIO()):
            pinned_prefix.main()
        return json.loads(output.read_text())

    def test_multiple_longest_depths_and_reordered_restart(self):
        first = self.root / "first.json"
        report = self.run_probe(first, extra=("--exact-prefix",))
        self.assertEqual(report["schema"], "gufo-pinned-prefix-probe-v2")
        self.assertEqual(len(self.calls), 10)
        for registration in report["registrations"]:
            self.assertEqual(registration["status"], "passed")
            self.assertEqual(registration["cases"]["greedy_warm"]["tokens_cached"],
                             registration["prefix_tokens"])
            self.assertEqual(registration["cases"]["greedy_cold"]["tokens_cached"], 0)
        restart = self.run_probe(self.root / "restart.json", files=(self.long, self.short),
                                 restore=first, extra=("--exact-prefix",))
        self.assertTrue(restart["restore"])
        self.assertEqual(restart["status"], "passed")

    def test_legacy_single_report_restores(self):
        first = self.root / "single.json"
        report = self.run_probe(first, files=(self.short,))
        self.assertEqual(report["schema"], "gufo-pinned-prefix-probe-v1")
        self.assertNotIn("registrations", report)
        self.assertEqual(len(report["cases"]), 5)
        self.assertEqual(self.run_probe(self.root / "restarted.json", files=(self.short,),
                                        restore=first)["status"], "passed")

    def test_restore_identity_rejected_before_requests(self):
        first = self.root / "first.json"
        self.run_probe(first)
        self.calls.clear()
        self.long.write_text("different")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_probe(self.root / "rejected.json", restore=first)
        self.assertFalse(self.calls)
        self.assertFalse((self.root / "rejected.json").exists())

    def test_failed_assertion_retains_report(self):
        output = self.root / "failed.json"
        original_call = self.call

        def shorter(url, payload):
            result = original_call(url, payload)
            if payload["prompt"].startswith("A B ") and payload["cache_prompt"]:
                result["tokens_cached"] = 2
            return result

        self.call = shorter
        with self.assertRaises(AssertionError):
            self.run_probe(output, extra=("--exact-prefix",))
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("exact longest registration", report["error"])

    def test_mismatched_pair_counts_rejected(self):
        args = ["pinned_prefix.py", "--output", str(self.root / "unused.json"),
                "--prefix-file", str(self.short), "--prefix-tokens", "2",
                "--prefix-tokens", "4"]
        with patch("sys.argv", args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pinned_prefix.main()

    def test_http_accounting_rejects_inconsistent_counts(self):
        result = {"content": "ok", "tokens_predicted": 1,
                  "tokens_cached": 8, "tokens_evaluated": 10,
                  "timings": {"prompt_n": 2}}

        def response():
            value = io.BytesIO(json.dumps(result).encode())
            value.headers = {"Server-Timing": "ttft;dur=12.5"}
            return value

        with patch.object(pinned_prefix, "urlopen", return_value=response()):
            actual = pinned_prefix.call("http://fixture/v1", {"prompt": "text"})
        self.assertEqual(actual["probe"]["ttft_ms"], 12.5)
        result["tokens_evaluated"] = 11
        with patch.object(pinned_prefix, "urlopen", return_value=response()), self.assertRaises(AssertionError):
            pinned_prefix.call("http://fixture", {"prompt": "text"})


if __name__ == "__main__":
    unittest.main()
