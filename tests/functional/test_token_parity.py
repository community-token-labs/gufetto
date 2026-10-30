#!/usr/bin/env python3
"""Synthetic ID recordings and loopback-only native llama protocol tests."""

import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

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
            "sampling": dict(PARITY.SAMPLING), "max_tokens": 4, "stop_type": "limit"}


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
        b["top_logprobs"] = tops(20, 10)
        self.assertEqual(self.compare(a, b)["status"], "NON_NEAR_TIE")  # NO + YES stays NO.

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

    def test_short_capture_reports_actual_denominator_not_fake_completion(self):
        result = self.compare(row([10]), row([10]))
        self.assertEqual(result["agreement_rate"], 1)
        self.assertEqual(result["agreement_denominator"], 1)
        self.assertEqual(result["status"], "UNKNOWN")  # limit before requested budget is incomplete
        result = self.compare(row(), row([10]))
        self.assertEqual(result["first_divergence"], 1)
        self.assertEqual(result["near_tie"], "UNKNOWN")

    def test_both_native_eos_shapes_normalize_identically(self):
        a, b = row([10, 11]), row([10, 11, 99])
        del a["stop_type"]
        a["stopped_eos"] = True  # Gufo excludes terminal EOG.
        b["stop_type"] = "eos"  # Pinned llama includes it.
        result = self.compare(a, b)
        self.assertIsNone(result["first_divergence"])
        self.assertEqual(result["agreement_denominator"], 2)
        self.assertEqual(result["agreement_rate"], 1)
        self.assertEqual(result["stop_reasons"], {"gufo": "eos", "llama": "eos"})
        self.assertEqual(result["llama"]["terminal_eog_id"], 99)
        self.assertEqual(result["status"], "NUMERIC_MATCH_REVIEW_REQUIRED")

    def test_eos_versus_token_near_tie_and_same_length_stop_mismatch(self):
        a, b = row([10, 11]), row([10, 11, 20, 21])
        a.update(stop_type="eos", terminal_eog_id=99,
                 top_logprobs=[[], [], [{"id": 99, "logprob": -.5}, {"id": 20, "logprob": -.51}]])
        b["top_logprobs"] = [[], [], [{"id": 20, "logprob": -.5}, {"id": 99, "logprob": -.51}]]
        result = self.compare(a, b)
        self.assertEqual(result["first_divergence"], 2)
        self.assertEqual(result["near_tie"], "YES")
        self.assertTrue(result["stop_reason_mismatch"])
        a = row()
        a["stop_type"] = "eos"
        result = self.compare(a, row())
        self.assertEqual(result["first_divergence"], 4)
        self.assertEqual(result["near_tie"], "UNKNOWN")

    def test_unknown_stop_never_matches_and_word_flags(self):
        a = row()
        del a["stop_type"]
        self.assertEqual(self.compare(a, row())["status"], "UNKNOWN")
        a.update(stopped_word=True, stopping_word="STOP")
        self.assertEqual(PARITY.normalize_stop(a, "gufo")["stop_reason"], "word")
        a["stopped_limit"] = True
        with self.assertRaises(ValueError):
            PARITY.normalize_stop(a, "gufo")

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
            left["top_logprobs"] = tops(gap=1) + [[], [], []]
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
            with patch.object(PARITY, "open_direct", side_effect=error) as network:
                code, result = self.invoke(["--prompts", str(p["prompts"]),
                    "--gufo-url", "http://127.0.0.1:1", "--gufo-config", str(c),
                    "--llama-recorded", str(p["llama"]), "--tokens", "4", "--horizon", "4", "--margin", ".02"])
            self.assertEqual(code, 2)
            self.assertIn("BLOCKED", result["error"])
            self.assertEqual(network.call_count, 1)
            sent = json.loads(network.call_args.args[0].data)
            self.assertEqual(sent["prompt"], [100, 101])

    def test_live_missing_ids_and_unexpected_exception_are_invalid(self):
        with patch.object(PARITY, "open_direct", return_value=io.BytesIO(b'{"content":"no IDs"}')):
            with self.assertRaisesRegex(ValueError, "BLOCKED"):
                PARITY.gufo_live("http://127.0.0.1:1", config("gufo"), PROMPTS, 4, 5, 2)
        with tempfile.TemporaryDirectory() as tmp:
            p = self.files(Path(tmp))
            args = ["--prompts", str(p["prompts"]), "--gufo-recorded", str(p["gufo"]),
                    "--llama-recorded", str(p["llama"]), "--margin", ".02"]
            with patch.object(PARITY, "load_recordings", side_effect=RecursionError("deep JSON")):
                self.assertEqual(self.invoke(args)[0], 2)

    def test_recorded_distribution_cardinality_and_eos_extra_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            p = self.files(directory)
            for count in (1, 3, 5):
                recorded = row()
                recorded["top_logprobs"] = [[] for _ in range(count)]
                p["gufo"].write_text(json.dumps({"config": config("gufo"), "requests": [recorded]}))
                with self.assertRaisesRegex(ValueError, "positions"):
                    PARITY.load_recordings(p["gufo"], PROMPTS, 4, "gufo")
            recorded["stop_type"] = "eos"
            p["gufo"].write_text(json.dumps({"config": config("gufo"), "requests": [recorded]}))
            self.assertEqual(len(PARITY.load_recordings(p["gufo"], PROMPTS, 4, "gufo")[1]["tiny"]["top_logprobs"]), 5)
            p["llama"].write_text(json.dumps({"config": config("llama"), "requests": [recorded]}))
            with self.assertRaisesRegex(ValueError, "positions"):
                PARITY.load_recordings(p["llama"], PROMPTS, 4, "llama")
            recorded["top_logprobs"].pop()
            p["llama"].write_text(json.dumps({"config": config("llama"), "requests": [recorded]}))
            self.assertEqual(len(PARITY.load_recordings(p["llama"], PROMPTS, 4, "llama")[1]["tiny"]["top_logprobs"]), 4)

    def test_endpoint_schemes_and_connection_error(self):
        for url in ("file:///tmp/never-read", "ftp://localhost/never", "http://user:password@localhost/"):
            with self.assertRaisesRegex(ValueError, "direct HTTP"):
                PARITY.open_direct(PARITY.Request(url), 1)
        with tempfile.TemporaryDirectory() as tmp:
            p = self.files(Path(tmp))
            c = Path(tmp) / "config.json"
            c.write_text(json.dumps(config("gufo")))
            args = ["--prompts", str(p["prompts"]), "--gufo-url", "http://127.0.0.1:1",
                    "--gufo-config", str(c), "--llama-recorded", str(p["llama"]),
                    "--tokens", "4", "--horizon", "4", "--margin", ".02"]
            with patch.object(PARITY, "open_direct", side_effect=URLError("connection refused")):
                code, result = self.invoke(args)
            self.assertEqual(code, 2)
            self.assertEqual(set(result["configs"]), {"gufo", "llama"})

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
                    "stop_type": "limit",
                    "completion_probabilities": [{"id": token, "top_logprobs": []}
                                                 for token in [10, 11, 12, 13]],
                    "generation_settings": {"temperature": 0}}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, payload))
                if response.get("_redirect"):
                    self.send_response(302)
                    self.send_header("Location", "/never-follow")
                    self.end_headers()
                    return
                value = copy.deepcopy(response)
                if value.pop("_short_prompt", None) == payload["prompt"][0]:
                    value["completion_probabilities"] = value["completion_probabilities"][:1]
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                requests.append(("GET:" + self.path, {}))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/v1"
                with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "https_proxy": "http://127.0.0.1:1",
                                             "no_proxy": "", "NO_PROXY": ""}):
                    _, outputs = PARITY.llama_live(url, config("llama"), PROMPTS, 4, 5, 2)
                    _, gufo_outputs = PARITY.gufo_live(url, config("gufo"), PROMPTS, 4, 5, 2)
                # Pinned llama omits probability rows for incomplete UTF-8 pieces.
                response.update(content="emoji 🦉", _short_prompt=100)
                prompts = PROMPTS + [{"id": "other", "prompt_token_ids": [102, 103]}]
                _, batch = PARITY.llama_live(url, config("llama"), prompts, 4, 5, 2)
                self.assertTrue(batch["tiny"]["distributions_misaligned"])
                self.assertEqual(batch["tiny"]["top_logprobs"], [])
                self.assertEqual(batch["tiny"]["generated_token_ids"], [10, 11, 12, 13])
                self.assertFalse(batch["other"]["distributions_misaligned"])
                self.assertEqual(self.compare(row([20, 11, 12, 13]), batch["tiny"])["near_tie"], "UNKNOWN")
                self.assertEqual(self.compare(row(), batch["other"])["status"], "NUMERIC_MATCH_REVIEW_REQUIRED")
                response.pop("_short_prompt")
                response["completion_probabilities"][0]["id"] = 999
                _, shifted = PARITY.llama_live(url, config("llama"), PROMPTS, 4, 5, 2)
                self.assertTrue(shifted["tiny"]["distributions_misaligned"])
                response["completion_probabilities"] = []
                _, missing = PARITY.llama_live(url, config("llama"), PROMPTS, 4, 5, 2)
                self.assertTrue(missing["tiny"]["distributions_misaligned"])
                self.assertEqual(missing["tiny"]["generated_token_ids"], [10, 11, 12, 13])
                # Live EOS retains its EOG-position distribution after ID normalization.
                response.update(tokens=[10, 99], stop_type="eos", completion_probabilities=[
                    {"id": 10, "top_logprobs": []},
                    {"id": 99, "top_logprobs": [{"id": 99, "logprob": -.5}, {"id": 20, "logprob": -.51}]}])
                _, eos = PARITY.llama_live(url, config("llama"), PROMPTS, 4, 5, 2)
                gufo = row([10, 20, 12, 13])
                gufo["top_logprobs"] = [[], [{"id": 20, "logprob": -.5}, {"id": 99, "logprob": -.51}], [], []]
                self.assertEqual(self.compare(gufo, eos["tiny"])["near_tie"], "YES")
                response["_redirect"] = True
                with self.assertRaisesRegex(ValueError, "BLOCKED"):
                    PARITY.llama_live(url, config("llama"), PROMPTS, 4, 5, 2)
                self.assertFalse(any(path.startswith("GET:") for path, _ in requests))
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(outputs["tiny"]["generated_token_ids"], [10, 11, 12, 13])
        self.assertEqual(gufo_outputs["tiny"]["generated_token_ids"], [10, 11, 12, 13])
        self.assertEqual(requests[0], requests[1])  # No per-engine request dialect.
        path, payload = requests[0]
        self.assertEqual(path, "/completion")
        self.assertEqual(payload["prompt"], [100, 101])
        self.assertTrue(payload["return_tokens"])
        self.assertFalse(payload["cache_prompt"])
        self.assertFalse(payload["post_sampling_probs"])
        self.assertNotIn("ignore_eos", payload)
        self.assertEqual(payload["n_predict"], 4)
        self.assertEqual(payload["n_probs"], 5)


if __name__ == "__main__":
    unittest.main()
