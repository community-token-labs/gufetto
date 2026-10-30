#!/usr/bin/env python3
"""CPU-only tests for the setup skill and its server verify script."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import threading
from typing import Any, Callable, Iterator
import unittest
import urllib.request
from unittest import mock


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gufo", type=Path, required=True,
                        help="built gufo executable whose --help is checked")
    args, rest = parser.parse_known_args()
    return args, [sys.argv[0], *rest]


ARGS, UNITTEST_ARGS = parse_args()
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/setup/verify_server.py"
SKILL = ROOT / "docs/skills/gufo-setup/SKILL.md"
SPEC = importlib.util.spec_from_file_location("verify_server", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)

KEY = "test-key-not-a-secret"
TEXT_MODEL = {"id": "gufo-local", "object": "model", "owned_by": "gufo",
              "context_length": 32768,
              "architecture": {"input_modalities": ["text"]}}


def chat_body(content: Any = "ready", finish: Any = "stop",
              usage: Any = None) -> dict[str, Any]:
    return {"object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant",
                                     "content": content}}],
            "usage": usage if usage is not None else
            {"prompt_tokens": 12, "completion_tokens": 2}}


class FakeServer:
    """Records requests and answers like `gufo serve llm`.

    `models` and `chat` replace the 200 bodies; `raw` answers a path with a
    handler-level callback for transport faults.
    """

    def __init__(self, models: Any = None, key: str | None = None,
                 loading_polls: int = 0, chat: Any = None,
                 raw: dict[str, Callable] | None = None):
        self.models = {"object": "list", "data": [TEXT_MODEL]} \
            if models is None else models
        self.key = key
        self.loading_polls = loading_polls
        self.chat = chat_body() if chat is None else chat
        self.raw = raw or {}
        self.requests: list[dict[str, Any]] = []


def make_handler(state: FakeServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status: int, payload: Any):
            body = payload if isinstance(payload, bytes) else \
                json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def error(self, status: int, code: str, message: str):
            self.send(status, {"error": {"message": message, "code": code,
                                         "type": "invalid_request_error"}})

        def handle_any(self, method: str):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            state.requests.append({"method": method, "path": self.path,
                                   "auth": self.headers.get("Authorization"),
                                   "body": body})
            if self.path in state.raw:
                return state.raw[self.path](self)
            if state.loading_polls > 0:
                state.loading_polls -= 1
                return self.send(503, {"error": {
                    "message": "Loading model", "type": "unavailable_error",
                    "code": "loading"}})
            if state.key and self.headers.get("Authorization") != \
                    "Bearer " + state.key:
                return self.error(401, "invalid_api_key", "Unauthorized")
            if method == "GET" and self.path == "/v1/models":
                return self.send(200, state.models)
            if method == "POST" and self.path == "/v1/chat/completions":
                if body["model"] != "gufo-local":
                    return self.error(404, "model_not_found",
                                      "model not found")
                return self.send(200, state.chat)
            return self.error(404, "not_found", "unknown route")

        def do_GET(self):
            self.handle_any("GET")

        def do_POST(self):
            self.handle_any("POST")

    return Handler


@contextmanager
def serve(state: FakeServer, host: str = "127.0.0.1") -> Iterator[str]:
    server_class = ThreadingHTTPServer
    if ":" in host:
        class Server6(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        server_class = Server6
    server = server_class((host, 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shown = f"[{host}]" if ":" in host else host
    try:
        yield f"http://{shown}:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def ipv6_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


def run_main(argv: list[str], env: dict[str, str] | None = None
             ) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.dict(os.environ), mock.patch("sys.stdout", out), \
            mock.patch("sys.stderr", err):
        os.environ.pop("GUFO_API_KEY", None)
        os.environ.update(env or {})
        status = VERIFY.main(argv)
    return status, out.getvalue(), err.getvalue()


def run_script(argv: list[str], env: dict[str, str] | None = None
               ) -> subprocess.CompletedProcess:
    child = {k: v for k, v in os.environ.items() if k != "GUFO_API_KEY"}
    child.update(env or {})
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *argv],
                          capture_output=True, text=True, timeout=60,
                          env=child)


class VerifyServerTest(unittest.TestCase):
    def assert_fails(self, state: FakeServer, expected: str,
                     extra: list[str] | None = None):
        with serve(state) as url:
            status, out, err = run_main(["--base-url", url, "--json",
                                         *(extra or [])])
        self.assertEqual(status, 1, out + err)
        result = json.loads(out)
        self.assertFalse(result["ok"])
        self.assertIn(expected, result["error"])
        self.assertEqual(err, "")
        return result

    def test_passes_with_one_chat_request(self):
        state = FakeServer()
        with serve(state) as url:
            status, out, _ = run_main(["--base-url", url, "--json"])
        self.assertEqual(status, 0, out)
        result = json.loads(out)
        self.assertTrue(result["ok"])
        self.assertEqual(result["model"], "gufo-local")
        self.assertEqual(result["context_length"], 32768)
        self.assertEqual(result["content"], "ready")
        chats = [r for r in state.requests if r["method"] == "POST"]
        self.assertEqual(len(chats), 1)
        self.assertEqual(chats[0]["path"], "/v1/chat/completions")
        self.assertEqual(chats[0]["body"]["reasoning_effort"], "none")
        self.assertFalse(chats[0]["body"]["stream"])
        self.assertIsNone(chats[0]["auth"])

    def test_selects_text_model_among_services(self):
        tts = {"id": "qwen3-tts", "object": "model", "owned_by": "gufo"}
        state = FakeServer(models={"data": [tts, TEXT_MODEL]})
        with serve(state) as url:
            status, out, _ = run_main(["--base-url", url])
        self.assertEqual(status, 0, out)
        self.assertIn("PASS: gufo-local", out)
        self.assertEqual(state.requests[-1]["body"]["model"], "gufo-local")

    def test_bearer_key_from_environment_is_never_printed(self):
        state = FakeServer(key=KEY)
        with serve(state) as url:
            status, out, err = run_main(["--base-url", url],
                                        {"GUFO_API_KEY": KEY})
        self.assertEqual(status, 0, out)
        self.assertNotIn(KEY, out + err)
        self.assertIn("authenticated=true", out)
        self.assertTrue(all(r["auth"] == "Bearer " + KEY
                            for r in state.requests))

    def test_ipv6_loopback_gets_the_key(self):
        if not ipv6_available():
            self.skipTest("no IPv6 loopback on this host")
        state = FakeServer(key=KEY)
        with serve(state, "::1") as url:
            status, out, err = run_main(["--base-url", url],
                                        {"GUFO_API_KEY": KEY})
        self.assertEqual(status, 0, out + err)
        self.assertTrue(state.requests)

    def test_localhost_name_is_not_trusted_with_a_key(self):
        state = FakeServer(key=KEY)
        with serve(state) as url:
            named = url.replace("127.0.0.1", "localhost")
            status, out, err = run_main(["--base-url", named],
                                        {"GUFO_API_KEY": KEY})
            self.assertEqual(status, 1)
            self.assertIn("cleartext to localhost", out)
            self.assertEqual(state.requests, [])
            # Without a key, a name is fine: nothing secret is sent.
            status, out, _ = run_main(["--base-url", named])
            self.assertEqual(status, 1, out)  # 401 from the keyed server.
        self.assertNotIn(KEY, out + err)

    def test_only_numeric_loopback_is_trusted(self):
        for host in ("127.0.0.1", "127.1.2.3", "::1"):
            with self.subTest(host=host):
                self.assertTrue(VERIFY.is_numeric_loopback(host))
        for host in ("localhost", "LOCALHOST", "127.0.0.1.example",
                     "192.0.2.1", "10.0.0.1", "::", "0.0.0.0", "fe80::1"):
            with self.subTest(host=host):
                self.assertFalse(VERIFY.is_numeric_loopback(host))

    def test_refuses_cleartext_key_to_remote_host(self):
        status, out, err = run_main(["--base-url", "http://192.0.2.1:8080"],
                                    {"GUFO_API_KEY": KEY})
        self.assertEqual(status, 1)
        self.assertIn("cleartext", out)
        self.assertNotIn(KEY, out + err)

    def test_redirect_is_not_followed(self):
        target = FakeServer()
        with serve(target) as elsewhere:
            def redirect(handler):
                handler.send_response(307)
                handler.send_header("Location", elsewhere + "/v1/models")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
            state = FakeServer(raw={"/v1/models": redirect})
            with serve(state) as url:
                status, out, err = run_main(["--base-url", url, "--json"],
                                            {"GUFO_API_KEY": KEY})
        self.assertEqual(status, 1)
        self.assertIn("307 redirect", json.loads(out)["error"])
        self.assertEqual(target.requests, [])
        self.assertNotIn(KEY, out + err)

    def test_https_to_http_downgrade_is_rejected(self):
        handler = VERIFY.RejectRedirects()
        request = urllib.request.Request("https://gufo.example/v1/models")
        request.add_header("Authorization", "Bearer " + KEY)
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code), \
                    self.assertRaises(VERIFY.CheckError) as raised:
                handler.redirect_request(request, None, code, "moved", {},
                                         "http://gufo.example/v1/models")
            self.assertNotIn(KEY, str(raised.exception))

    def assert_single_object(self, stdout: str) -> dict[str, Any]:
        lines = stdout.splitlines()
        self.assertEqual(len(lines), 1, stdout)
        result = json.loads(lines[0])
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["ok"], bool)
        return result

    def assert_no_leak(self, key: str, text: str):
        self.assertNotIn(key, text)
        for start in range(0, len(key) - 15, 8):
            self.assertNotIn(key[start:start + 16], text)

    def test_malformed_key_is_rejected_without_echo(self):
        for key in (KEY + "\nX-Injected: 1", "two words", "café",
                    "tab\tkey", "del\x7fkey"):
            with self.subTest(key=key):
                with serve(FakeServer()) as url:
                    done = run_script(["--base-url", url, "--json"],
                                      {"GUFO_API_KEY": key})
                self.assertEqual(done.returncode, 1)
                result = self.assert_single_object(done.stdout)
                self.assertIn("not a valid bearer token", result["error"])
                self.assertNotIn(key, done.stdout + done.stderr)
                self.assertNotIn("Traceback", done.stderr)

    def test_printable_ascii_keys_are_accepted(self):
        for key in ("abc", "A-Z.a_z~0+9/", "dG9rZW4=", 'quote"key',
                    "back\\slash", "a=b", "=", "%41", "!#$&'()*,:;<>?@[]^`{|}"):
            with self.subTest(key=key):
                self.assertTrue(VERIFY.valid_key(key))

    def test_long_key_echoed_by_server_never_leaks(self):
        for key in (("Ab0-._~+/" * 28)[:250],
                    ('Ab0"\\{}<>%/=' * 25)[:250]):
            with self.subTest(key=key[:12]):
                self.check_echoed_key(key)

    def check_echoed_key(self, key: str):
        def echo_error(handler):
            # JSON may escape "/" as "\/"; the script must still find the key.
            body = json.dumps({"error": {"code": "invalid_api_key",
                                         "message": "bad key " + key}})
            handler.send(401, body.replace("/", "\\/").encode())

        def echo_chat(handler):
            handler.send(200, chat_body(content="key " + key + " ready"))

        def echo_models(handler):
            handler.send(200, {"data": [dict(TEXT_MODEL, id=key)]})

        for name, raw in (("error", {"/v1/models": echo_error}),
                          ("content", {"/v1/chat/completions": echo_chat}),
                          ("model id", {"/v1/models": echo_models})):
            for mode in (["--json"], []):
                with self.subTest(echo=name, mode=mode):
                    with serve(FakeServer(raw=raw)) as url:
                        done = run_script(["--base-url", url, *mode],
                                          {"GUFO_API_KEY": key})
                    output = done.stdout + done.stderr
                    self.assert_no_leak(key, output)
                    self.assertNotIn("Traceback", done.stderr)
                    if mode:
                        self.assert_single_object(done.stdout)
                    if name == "content":
                        self.assertEqual(done.returncode, 0, output)
                        shown = json.loads(done.stdout)["content"] \
                            if mode else done.stdout
                        self.assertIn(VERIFY.REDACTED, shown)

    def test_scrub_is_single_pass_and_covers_integers(self):
        state = FakeServer(chat=chat_body(content="key ted ready"))
        with serve(state) as url:
            status, out, _ = run_main(["--base-url", url, "--json"],
                                      {"GUFO_API_KEY": "ted"})
        self.assertEqual(status, 0, out)
        self.assertEqual(json.loads(out)["content"],
                         f"key {VERIFY.REDACTED} ready")
        self.assertEqual(VERIFY.scrub(VERIFY.scrub("a ted", "ted"), "ted"),
                         f"a {VERIFY.REDACTED}")
        state = FakeServer(chat=chat_body(
            usage={"prompt_tokens": 12345, "completion_tokens": 2}))
        with serve(state) as url:
            status, out, _ = run_main(["--base-url", url, "--json"],
                                      {"GUFO_API_KEY": "12345"})
        self.assertEqual(status, 1, out)
        self.assertNotIn("12345", out)
        self.assertEqual(VERIFY.scrub({"n": 12345, "b": True}, "12345"),
                         {"n": VERIFY.REDACTED, "b": True})

    def test_help_and_abbreviations(self):
        done = run_script(["-h", "--json"])
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(done.stdout.startswith("usage:"))
        for argv in (["--js"], ["--base", "http://127.0.0.1:1", "--json"]):
            with self.subTest(argv=argv):
                done = run_script(argv)
                self.assertEqual(done.returncode, 2, done.stdout)

    def test_json_is_always_one_object(self):
        for argv, env, status in (
                (["--base-url", "ftp://127.0.0.1"], {}, 2),
                (["--max-tokens", "x"], {}, 2),
                (["--base-url", "http://192.0.2.1:8080"],
                 {"GUFO_API_KEY": KEY}, 1)):
            with self.subTest(argv=argv):
                done = run_script([*argv, "--json"], env)
                self.assertEqual(done.returncode, status, done.stderr)
                self.assert_single_object(done.stdout)
        for state, status in ((FakeServer(), 0), (FakeServer(key=KEY), 1),
                              (FakeServer(chat=b"not json"), 1)):
            with self.subTest(status=status), serve(state) as url:
                done = run_script(["--base-url", url, "--json"])
                self.assertEqual(done.returncode, status, done.stderr)
                self.assertEqual(
                    self.assert_single_object(done.stdout)["ok"],
                    status == 0)

    def test_missing_key_explains_401(self):
        state = FakeServer(key=KEY)
        with serve(state) as url:
            status, out, _ = run_main(["--base-url", url])
        self.assertEqual(status, 1)
        self.assertIn("FAIL: GET /v1/models returned 401", out)
        self.assertIn("GUFO_API_KEY", out)
        self.assertEqual(sum(r["method"] == "POST" for r in state.requests), 0)

    def test_waits_while_loading(self):
        state = FakeServer(loading_polls=2)
        with serve(state) as url, mock.patch.object(VERIFY.time, "sleep"):
            status, out, _ = run_main(["--base-url", url, "--wait", "60"])
        self.assertEqual(status, 0, out)

    def test_loading_without_wait_fails_with_hint(self):
        result = self.assert_fails(FakeServer(loading_polls=1), "503 loading")
        self.assertIn("--wait", result["hint"])

    def test_unknown_model(self):
        self.assert_fails(FakeServer(), "'other' is not served "
                          "(served: gufo-local)", ["--model", "other"])

    def test_empty_text_at_length_suggests_budget(self):
        result = self.assert_fails(
            FakeServer(chat=chat_body(content="", finish="length")),
            "finish_reason length")
        self.assertIn("--max-tokens", result["hint"])

    def test_malformed_model_lists(self):
        for listing in ({"data": [None]}, {"data": [{"id": 7}]}, {"data": []},
                        {"data": {}}, [], b"null", "x",
                        {"data": [dict(TEXT_MODEL, context_length="big")]},
                        {"data": [dict(TEXT_MODEL, context_length=True)]},
                        {"data": [dict(TEXT_MODEL, architecture={
                            "input_modalities": "text"})]}):
            with self.subTest(listing=listing):
                self.assert_fails(FakeServer(models=listing),
                                  "GET /v1/models")

    def test_malformed_chat_responses(self):
        for body in ({}, {"choices": []}, {"choices": [None]},
                     {"choices": [{"message": "ready"}]},
                     chat_body(content=["ready"]), chat_body(finish=3),
                     chat_body(usage={"prompt_tokens": "12"}),
                     chat_body(usage={"completion_tokens": -1}),
                     chat_body(usage={"prompt_tokens": 1.5}),
                     chat_body(usage=[1])):
            with self.subTest(body=body):
                self.assert_fails(FakeServer(chat=body), "malformed")

    def test_non_json_bodies(self):
        self.assert_fails(FakeServer(models=b"<html>proxy</html>"),
                          "not JSON")
        self.assert_fails(FakeServer(chat=b"{\"choices\": "), "not JSON")

    def test_truncated_body(self):
        def truncated(handler):
            handler.send_response(200)
            handler.send_header("Content-Length", "1000")
            handler.end_headers()
            handler.wfile.write(b'{"data": [')
            handler.close_connection = True
        self.assert_fails(FakeServer(raw={"/v1/models": truncated}),
                          "truncated")

    def test_timeout(self):
        release = threading.Event()

        def stall(handler):
            release.wait(10)
        state = FakeServer(raw={"/v1/chat/completions": stall})
        try:
            self.assert_fails(state, "timed out after 0.5 s",
                              ["--timeout", "0.5"])
        finally:
            release.set()

    def test_connection_refused(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        status, out, _ = run_main(["--base-url", f"http://127.0.0.1:{port}"])
        self.assertEqual(status, 1)
        self.assertIn("not running", out)

    def test_usage_errors_exit_2_with_json(self):
        for argv in (["--base-url", "http://127.0.0.1:8080/v1"],
                     ["--base-url", "127.0.0.1:8080"],
                     ["--base-url", "http://127.0.0.1:99999"],
                     ["--base-url", "http://127.0.0.1:port"],
                     ["--base-url", "http://127.0.0.1:0"],
                     ["--base-url", "http://[::1"],
                     ["--base-url", "http://[not-an-ip]:8080"],
                     ["--base-url", "http://user:pw@127.0.0.1:8080"],
                     ["--base-url", "http://127.0.0.1:8080/?x=1"],
                     ["--max-tokens", "0"], ["--max-tokens", "many"],
                     ["--timeout", "nan"]):
            with self.subTest(argv=argv):
                status, out, err = run_main([*argv, "--json"])
                self.assertEqual(status, 2, out + err)
                result = json.loads(out)
                self.assertFalse(result["ok"])
                self.assertTrue(result["usage"])
                self.assertIn("usage error", err)
                self.assertNotIn("pw", out + err)

    def test_command_line_exit_status(self):
        with serve(FakeServer()) as url:
            done = run_script(["--base-url", url])
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PASS", done.stdout)
        done = run_script(["--base-url", "ftp://127.0.0.1"])
        self.assertEqual(done.returncode, 2)
        self.assertNotIn("Traceback", done.stderr)


def help_options(argv: list[str]) -> set[str]:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, (argv, done.stderr)
    return set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)",
                          done.stdout + done.stderr))


class SkillDocumentTest(unittest.TestCase):
    """Keeps the skill tied to the actual CLI and repository."""

    # Each option the skill mentions, with every command it is used with.
    OWNERS = {
        "--help": {"gufo", "serve llm"},
        "--version": {"gufo"},
        "--json": {"diagnose"},
        "--plan": {"diagnose"},
        "--benchmark": {"diagnose"},
        "--host": {"serve"},
        "--port": {"serve"},
        "--sessions": {"serve"},
        "--api-key": {"serve"},
        "--api-key-file": {"serve llm"},
        "--model": {"serve llm", "verify"},
        "--served-model-name": {"serve llm"},
        "--context": {"serve llm"},
        "--think": {"serve llm"},
        "--speculative": {"serve llm"},
        "--dflash-model": {"serve llm"},
        "--mtp-model": {"serve llm"},
        "--dspark-model": {"serve llm"},
        "--mmproj": {"serve llm"},
        "--no-vision": {"serve llm"},
        "--enable-yarn": {"serve llm"},
        "--cache-ram-bytes": {"serve llm"},
        "--max-pending": {"serve llm"},
        "--wait": {"verify"},
        "--base-url": {"verify"},
        "--preset": {"cmake"},
        "--build": {"cmake"},
        "--parallel": {"cmake"},
    }
    # The separately decided server key file; the skill checks `--help`
    # before using it, so builds without it stay correct.
    PENDING = {"--api-key-file"}
    CMAKE = {"--preset", "--build", "--parallel"}
    # Quoted messages and the source file that emits each one.
    MESSAGES = {
        "Unsupported GPU architecture: expected gfx1151":
            "src/cli/diagnose/diagnose.cpp",
        "Detected AMD Strix Halo GPU (gfx1151)":
            "src/cli/diagnose/diagnose.cpp",
        "verified to fit: --context ":
            "src/cli/serve/component_text_cache.cpp",
        "no fitting context found within the search limits":
            "src/cli/serve/component_text_cache.cpp",
        "insufficient component backing budget":
            "src/cli/serve/inference_backend.cpp",
        "pass --enable-yarn": "src/models/qwen38_flash_next/rope_scaling.cpp",
        '"loading"': "src/cli/serve/http_server.cpp",
        '"model_not_found"': "src/cli/serve/openai_chat.cpp",
        '"device_lost"': "src/cli/serve/http_server.cpp",
        "event=listening": "src/cli/serve/serve.cpp",
        "kDeviceLostExitStatus = 75": "src/cli/serve/serve.cpp",
    }

    @classmethod
    def setUpClass(cls):
        gufo = str(ARGS.gufo)
        cls.text = SKILL.read_text()
        cls.help = {
            "gufo": help_options([gufo, "--help"]),
            "diagnose": help_options([gufo, "diagnose", "--help"]),
            "serve": help_options([gufo, "serve", "--help"]),
            "serve llm": help_options([gufo, "serve", "llm", "--help"]),
            "verify": help_options([sys.executable, "-I", str(SCRIPT),
                                    "--help"]),
        }

    def test_front_matter(self):
        match = re.match(r"---\nname: ([a-z0-9-]+)\ndescription: (.+)\n---\n",
                         self.text)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), SKILL.parent.name)
        self.assertLessEqual(len(match.group(2)), 1024)

    def test_every_option_has_a_checked_owner(self):
        used = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", self.text))
        self.assertEqual(sorted(used - set(self.OWNERS)), [])
        for option, commands in self.OWNERS.items():
            for command in commands - {"cmake"}:
                with self.subTest(option=option, command=command):
                    if option in self.PENDING and \
                            option not in self.help[command]:
                        continue
                    self.assertIn(option, self.help[command])
            if "cmake" in commands:
                self.assertIn(option, self.CMAKE)

    def test_pending_option_is_gated_by_help(self):
        for option in self.PENDING:
            self.assertIn(f"`\"$GUFO\" serve llm --help` does not list "
                          f"`{option}`", self.text)

    def commands(self) -> Iterator[list[str]]:
        for block in re.findall(r"```sh\n(.*?)```", self.text, re.DOTALL):
            for line in block.replace("\\\n", " ").splitlines():
                line = line.split(" #", 1)[0].strip()
                if line:
                    yield shlex.split(line)

    def test_example_commands_use_their_own_options(self):
        checked = 0
        for words in self.commands():
            if words[0] == "$GUFO":
                rest, command = words[1:], "gufo"
                if rest and rest[0] == "diagnose":
                    rest, command = rest[1:], "diagnose"
                elif rest and rest[0] == "serve":
                    rest, command = rest[1:], "serve"
                for word in rest:
                    if word == "llm" and command == "serve":
                        command = "serve llm"
                    elif word.startswith("--"):
                        checked += 1
                        with self.subTest(command=command, option=word):
                            self.assertIn(word, self.help[command])
            elif words[0] == "python3" and words[1].endswith(
                    "tools/setup/verify_server.py"):
                for word in words[2:]:
                    if word.startswith("--"):
                        checked += 1
                        self.assertIn(word, self.help["verify"])
            elif words[0] == "cmake":
                for word in words[1:]:
                    if word.startswith("--"):
                        checked += 1
                        self.assertIn(word, self.CMAKE)
        self.assertGreaterEqual(checked, 12)

    def test_quoted_messages_exist_in_sources(self):
        for message, source in self.MESSAGES.items():
            with self.subTest(message=message):
                self.assertIn(message, (ROOT / source).read_text())
                quoted = message.strip('"').removeprefix(
                    "kDeviceLostExitStatus = ")
                self.assertIn(quoted.strip(), self.text)

    def test_repository_paths_exist(self):
        paths = re.findall(r"`((?:docs|tools)/[^`<>\s]+)`", self.text)
        self.assertGreater(len(paths), 5)
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).exists())


if __name__ == "__main__":
    unittest.main(argv=UNITTEST_ARGS)
