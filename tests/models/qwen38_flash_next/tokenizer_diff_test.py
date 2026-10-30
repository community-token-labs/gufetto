#!/usr/bin/env python3
"""CPU-only synthetic metadata; expectations are independent of C++ constants."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
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

        def fetch(url, optional=False):
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
