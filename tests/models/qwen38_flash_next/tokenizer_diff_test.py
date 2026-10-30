#!/usr/bin/env python3
"""CPU-only synthetic metadata; expectations are independent of C++ constants."""

import contextlib
import copy
import hashlib
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "tokenizer_diff", ROOT / "tools/qwen-flash/tokenizer_diff.py")
DIFF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIFF)


def fixture():
    return {"tokenizer.json": json.dumps({
        "model": {"type": "BPE", "vocab": {"hello": 0, "<|im_start|>": 100},
                  "merges": []},
        "added_tokens": [{"content": "<|im_start|>", "id": 100, "special": True},
                         {"content": "<|im_end|>", "id": 101, "special": True}]
    }).encode(), "tokenizer_config.json": json.dumps({
        "eos_token": "<|im_end|>", "chat_template": "{{ messages }}",
        "added_tokens_decoder": {"100": {"content": "<|im_start|>"},
                                 "101": {"content": "<|im_end|>"}}
    }).encode(), "chat_template.jinja": b"{{ messages }}"}


class TokenizerDiffTest(unittest.TestCase):
    def setUp(self):
        self.raw = fixture()
        self.reference = DIFF.snapshot(self.raw, {"synthetic": "independent"})
        self.known = {"<|im_start|>", "<|im_end|>"}

    def report(self, raw=None):
        return DIFF.compare(self.reference,
                            DIFF.snapshot(raw or self.raw, {"synthetic": "candidate"}),
                            self.known)

    def change(self, name, edit):
        document = json.loads(self.raw[name])
        edit(document)
        self.raw[name] = json.dumps(document).encode()

    def test_identical(self):
        self.assertFalse(self.report()["review_required"])

    def test_json_formatting_does_not_change_semantics(self):
        self.raw["tokenizer.json"] = json.dumps(
            json.loads(self.raw["tokenizer.json"]), indent=4).encode()
        self.assertFalse(self.report()["review_required"])

    def test_id_change(self):
        self.change("tokenizer.json", lambda d: d["added_tokens"][1].update(id=202))
        self.change("tokenizer_config.json", lambda d: d.pop("added_tokens_decoder"))
        change = self.report()["control_changes"][0]
        self.assertEqual(change, {"key": "<|im_end|>", "before": 101,
                                  "after": 202, "in_cpp_catalogue": True})

    def test_control_added_outside_catalogue(self):
        self.change("tokenizer.json", lambda d: d["added_tokens"].append(
            {"content": "<|new_control|>", "id": 102, "special": False}))
        change = self.report()["control_changes"][0]
        self.assertIsNone(change["before"])
        self.assertEqual(change["after"], 102)
        self.assertFalse(change["in_cpp_catalogue"])

    def test_control_removed(self):
        self.change("tokenizer.json", lambda d: d["added_tokens"].pop())
        self.change("tokenizer_config.json", lambda d: d["added_tokens_decoder"].pop("101"))
        self.assertEqual(self.report()["missing_catalogue_tokens"], ["<|im_end|>"])
        self.assertIsNone(self.report()["control_changes"][0]["after"])

    def test_vocab_control_without_added_entry(self):
        self.change("tokenizer.json", lambda d: d["model"]["vocab"].update({"<think>": 103}))
        self.assertEqual(self.report()["control_changes"][0]["key"], "<think>")

    def test_vocab_and_merges_changes(self):
        self.change("tokenizer.json", lambda d: d["model"]["vocab"].update(world=2))
        self.change("tokenizer.json", lambda d: d["model"]["merges"].append(["h", "i"]))
        keys = [c["key"] for c in self.report()["section_changes"]]
        self.assertIn("tokenizer.json/model/vocab", keys)
        self.assertIn("tokenizer.json/model/merges", keys)

    def test_config_and_both_templates(self):
        self.change("tokenizer_config.json", lambda d: d.update(
            eos_token="<|im_start|>", chat_template=[{"name": "default", "template": "new"}]))
        self.raw["chat_template.jinja"] = b"new\n"
        self.assertEqual(len(self.report()["template_changes"]), 2)
        self.assertTrue(self.report()["review_required"])

    def test_template_fallback_and_missing(self):
        del self.raw["chat_template.jinja"]
        self.assertTrue(self.report()["review_required"])
        self.change("tokenizer_config.json", lambda d: d.pop("chat_template"))
        with self.assertRaisesRegex(ValueError, "no chat template"):
            self.report()

    def test_conflicting_control_id(self):
        self.change("tokenizer_config.json", lambda d: d["added_tokens_decoder"].update(
            {"999": {"content": "<|im_end|>"}}))
        with self.assertRaisesRegex(ValueError, "conflicting IDs"):
            self.report()

    def test_network_disabled(self):
        with patch.object(DIFF.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "requires --allow-network"):
                DIFF.load_source("owner/model@main")
            request.assert_not_called()

    def test_network_resolves_pin_and_fetches_only_metadata(self):
        revision = "a" * 40
        urls = []

        def fetch(url, optional=False, expected_sha=None):
            urls.append(url)
            if "/api/" in url:
                return json.dumps({"sha": revision}).encode()
            return self.raw[url.rsplit("/", 1)[1]]

        with patch.object(DIFF, "fetch", side_effect=fetch):
            result = DIFF.load_source("owner/model@main", True)
        self.assertEqual(result["source"]["revision"], revision)
        self.assertEqual(len(urls), 4)
        self.assertEqual([url.rsplit("/", 1)[1] for url in urls[1:]], list(DIFF.FILES))
        self.assertTrue(all(f"/resolve/{revision}/" in url for url in urls[1:]))

    def test_unexpected_exception_is_invalid_not_review(self):
        for error in (RecursionError("nested JSON"), IncompleteRead(b"partial")):
            with patch.object(DIFF, "load_source", side_effect=error):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(DIFF.main(["unused"]), 2)

    def test_strict_json_and_decoder_id(self):
        for data in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                DIFF.parse_json(data)
        for token_id in ("+1_01", " 101", "0101", "-1", "١٠١"):
            raw = fixture()
            config = json.loads(raw["tokenizer_config.json"])
            config["added_tokens_decoder"] = {token_id: {"content": "<|im_end|>"}}
            raw["tokenizer_config.json"] = json.dumps(config).encode()
            with self.assertRaisesRegex(ValueError, "canonical decimal"):
                DIFF.snapshot(raw, {})

    def test_named_control_flag_changes(self):
        self.change("tokenizer.json", lambda d: d["added_tokens"][0].update(special=False, lstrip=True))
        change = self.report()["control_changes"][0]
        self.assertEqual(change["key"], "<|im_start|>")
        self.assertEqual([item["key"] for item in change["flag_changes"]],
                         ["tokenizer.json/lstrip", "tokenizer.json/special"])

    def test_network_url_and_repo_restrictions(self):
        for url in ("http://huggingface.co/file", "https://example.org/file",
                    "https://huggingface.co.evil.invalid/file", "https://user@huggingface.co/file"):
            with self.assertRaises(ValueError):
                DIFF.validate_url(url)
        with patch.object(DIFF, "fetch") as fetch:
            with self.assertRaises(ValueError):
                DIFF.load_source("../..@main", True)
            fetch.assert_not_called()
        handler = DIFF.urllib.request.build_opener
        # Redirect destinations pass the same validator as initial requests.
        DIFF.validate_url("https://cdn-lfs.hf.co/file")

    def test_real_fetch_size_checksum_404_truncation_and_deadline(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/missing":
                    self.send_error(404)
                    return
                if self.path == "/slow":
                    time.sleep(.05)
                body = b"metadata"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body) + (1 if self.path == "/truncated" else 0)))
                self.send_header("ETag", '"' + hashlib.sha256(body).hexdigest() + '"')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                # Only this test bypasses HTTPS host validation to exercise real
                # urllib reads against loopback; production rejects this URL.
                with patch.object(DIFF, "validate_url"):
                    self.assertEqual(DIFF.fetch(base + "/ok"), b"metadata")
                    self.assertIsNone(DIFF.fetch(base + "/missing", optional=True))
                    with self.assertRaises(ValueError):
                        DIFF.fetch(base + "/truncated")
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        DIFF.fetch(base + "/ok", expected_sha="0" * 64)
                    with patch.object(DIFF, "MAX_BYTES", 2):
                        with self.assertRaisesRegex(ValueError, "exceeds"):
                            DIFF.fetch(base + "/ok")
                    with patch.object(DIFF, "DOWNLOAD_SECONDS", .01):
                        with self.assertRaises((TimeoutError, OSError)):
                            DIFF.fetch(base + "/slow")
            finally:
                server.shutdown()
                thread.join()

    def test_invalid_added_tokens(self):
        self.change("tokenizer.json", lambda d: d.update(added_tokens=[None]))
        with self.assertRaisesRegex(ValueError, "invalid added-token entry"):
            self.report()

    def test_bad_json_and_missing_file(self):
        bad = copy.deepcopy(self.raw)
        bad["tokenizer.json"] = b"{"
        with self.assertRaises(ValueError):
            self.report(bad)
        del bad["tokenizer.json"]
        with self.assertRaises(KeyError):
            self.report(bad)

    def test_real_catalogue_and_independent_pin(self):
        known = DIFF.catalogue(ROOT / "src/models/qwen/control_tokens.hpp")
        self.assertIn("<|im_start|>", known)
        reference = json.loads(DIFF.REFERENCE.read_text())
        self.assertEqual(reference["source"]["revision"],
                         "de4b8e4d43b917e7706784d8bb445c9af86a3540")
        self.assertEqual(reference["controls"]["<|im_start|>"], 248045)
        self.assertEqual(reference["controls"]["<|im_end|>"], 248046)

    def test_cli_local_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name, data in self.raw.items():
                (directory / name).write_bytes(data)
            reference = directory / "reference.json"
            reference.write_text(json.dumps(self.reference))
            args = [str(directory), "--reference", str(reference)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(DIFF.main(args), 0)
            (directory / "chat_template.jinja").write_text("changed")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(DIFF.main(args), 1)
            (directory / "tokenizer.json").write_text("invalid")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(DIFF.main(args), 2)


if __name__ == "__main__":
    unittest.main()
