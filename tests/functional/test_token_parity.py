#!/usr/bin/env python3
"""Synthetic ID recordings and loopback-only native llama protocol tests."""

import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("token_parity", ROOT / "tools/qwen-flash/token_parity.py")
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)
PROMPTS = [{"id": "tiny", "prompt_token_ids": [100, 101]}]


def config(engine):
    return {"revision": PARITY.LLAMA_REVISION if engine == "llama" else "a" * 40,
            "flags": ["server", "--speculative", "off"], "stack": "synthetic CPU fixture",
            "model_sha256": "b" * 64, "tokenizer_sha256": "c" * 64, "mode": "ar"}


def row(tokens=None):
    return {"id": "tiny", "prompt_token_ids": [100, 101],
            "generated_token_ids": [10, 11, 12, 13] if tokens is None else tokens,
            "text": "Independent continuation text, not retokenized.",
            "sampling": dict(PARITY.SAMPLING), "max_tokens": 4}


def tops(first=10, second=20, gap=.01):
    return [[{"id": first, "logprob": -.5}, {"id": second, "logprob": -.5 - gap}]]


class TokenParityTest(unittest.TestCase):
    def compare(self, a, b, margin=.02):
        return PARITY.compare_prompt(PROMPTS[0], a, b, 4, 4, margin)

    def test_exact_and_agreement(self):
        result = self.compare(row(), row())
        self.assertIsNone(result["first_divergence"])
        self.assertEqual(result["agreement_rate"], 1)
        self.assertEqual(result["human_coherence"], "NOT_REVIEWED")
        self.assertNotEqual(result["status"], "PASS")

    def test_first_divergence_and_context(self):
        result = self.compare(row(), row([10, 99, 98, 13]))
        self.assertEqual(result["first_divergence"], 1)
        self.assertEqual(result["agreement_rate"], .5)
        self.assertEqual(result["context"]["shared_prefix_tail"], [100, 101, 10])
        self.assertEqual(result["near_tie"], "UNKNOWN")

    def test_both_near_ties(self):
        a, b = row(), row([20, 21, 22, 23])
        a["top_logprobs"], b["top_logprobs"] = tops(), tops(20, 10)
        result = self.compare(a, b)
        self.assertEqual(result["near_tie"], "YES")
        self.assertEqual(result["status"], "NEAR_TIE_REVIEW_REQUIRED")
        self.assertEqual(result["first_divergence_top_k"]["gufo"][0]["id"], 10)

    def test_non_near_tie_one_engine_is_sufficient(self):
        a, b = row(), row([20, 11, 12, 13])
        a["top_logprobs"] = tops(gap=.8)
        self.assertEqual(self.compare(a, b)["status"], "NON_NEAR_TIE")

    def test_missing_cross_winner_is_unknown_with_small_margin(self):
        a, b = row(), row([20, 11, 12, 13])
        a["top_logprobs"] = tops(second=99)
        b["top_logprobs"] = tops(20, 10)
        self.assertEqual(self.compare(a, b)["near_tie"], "UNKNOWN")

    def test_wrong_greedy_winner_is_unknown(self):
        a, b = row(), row([20, 11, 12, 13])
        a["top_logprobs"] = tops(99, 10)
        b["top_logprobs"] = tops(20, 10)
        self.assertEqual(self.compare(a, b)["near_tie"], "UNKNOWN")

    def test_truncated_output_cannot_inflate_agreement(self):
        result = self.compare(row([10]), row([10]))
        self.assertEqual(result["agreement_rate"], .25)
        self.assertEqual(result["status"], "INCOMPLETE")
        result = self.compare(row(), row([10]))
        self.assertEqual(result["first_divergence"], 1)
        self.assertEqual(result["near_tie"], "UNKNOWN")

    def test_boolean_and_negative_ids_rejected(self):
        for invalid in ([True], [-1], ["10"], []):
            with self.assertRaises(ValueError):
                PARITY.ids(invalid, "fixture", nonempty=True)

    def test_invalid_distribution(self):
        for values in ([[{"id": 10, "logprob": float("nan")}]],
                       [[{"id": 10, "logprob": .5}]],
                       [[{"id": 10, "logprob": -.5}, {"id": 10, "logprob": -.6}]]):
            a = row()
            a["top_logprobs"] = values
            with self.assertRaises(ValueError):
                self.compare(a, row([20, 11, 12, 13]))

    def files(self, directory, right=None):
        paths = {}
        for name, value in (("prompts", PROMPTS),
                            ("gufo", {"config": config("gufo"), "requests": [row()]}),
                            ("llama", {"config": config("llama"), "requests": [right or row()]})):
            paths[name] = directory / (name + ".json")
            paths[name].write_text(json.dumps(value))
        return paths

    def invoke(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = PARITY.main(args)
        return code, json.loads(output.getvalue())

    def test_cli_recordings_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.files(Path(tmp))
            args = ["--prompts", str(p["prompts"]), "--gufo-recorded", str(p["gufo"]),
                    "--llama-recorded", str(p["llama"]), "--tokens", "4", "--horizon", "4", "--margin", ".02"]
            code, result = self.invoke(args)
            self.assertEqual(code, 0)
            self.assertFalse(result["automatic_quality_pass"])
            right = row([20, 11, 12, 13])
            self.files(Path(tmp), right)
            self.assertEqual(self.invoke(args)[0], 3)
            left = row()
            left["top_logprobs"] = tops(gap=1)
            p["gufo"].write_text(json.dumps({"config": config("gufo"), "requests": [left]}))
            self.assertEqual(self.invoke(args)[0], 1)
            right["prompt_token_ids"] = [999]
            self.files(Path(tmp), right)
            self.assertEqual(self.invoke(args)[0], 2)
            self.assertEqual(self.invoke(args[:-1] + ["nan"])[0], 2)

    def test_current_live_gufo_is_blocked_without_text_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.files(Path(tmp))
            c = Path(tmp) / "config.json"
            c.write_text(json.dumps(config("gufo")))
            error = PARITY.HTTPError("http://127.0.0.1:1/completion", 400, "Bad Request", {}, None)
            with patch.object(PARITY, "urlopen", side_effect=error) as network:
                code, result = self.invoke(["--prompts", str(p["prompts"]),
                    "--gufo-url", "http://127.0.0.1:1", "--gufo-config", str(c),
                    "--llama-recorded", str(p["llama"]), "--margin", ".02"])
            self.assertEqual(code, 2)
            self.assertIn("BLOCKED", result["error"])
            self.assertEqual(network.call_count, 1)
            sent = json.loads(network.call_args.args[0].data)
            self.assertEqual(sent["prompt"], [100, 101])

    def test_config_identity_pin_and_mode(self):
        for field, value in (("revision", "wrong-pin"), ("mode", "mtp"),
                             ("model_sha256", "not-a-digest"), ("flags", "not-argv")):
            c = config("llama")
            c[field] = value
            with self.assertRaises(ValueError):
                PARITY.validate_config(c, "llama")

    def test_native_llama_fake_server_receives_only_id_prompt(self):
        requests = []
        response = {"tokens": [10, 11, 12, 13], "content": "fixture", "truncated": False,
                    "completion_probabilities": [{"id": token, "top_logprobs": []}
                                                 for token in [10, 11, 12, 13]],
                    "generation_settings": {"temperature": 0}}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                _, outputs = PARITY.llama_live(f"http://127.0.0.1:{server.server_port}/v1",
                                               config("llama"), PROMPTS, 4, 5, 2)
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(outputs["tiny"]["generated_token_ids"], [10, 11, 12, 13])
        path, payload = requests[0]
        self.assertEqual(path, "/completion")
        self.assertEqual(payload["prompt"], [100, 101])
        self.assertTrue(payload["return_tokens"])
        self.assertFalse(payload["cache_prompt"])
        self.assertFalse(payload["post_sampling_probs"])
        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(payload["n_predict"], 4)
        self.assertEqual(payload["n_probs"], 5)


if __name__ == "__main__":
    unittest.main()
