#!/usr/bin/env python3
"""CPU-only tests for the setup skill and its server verify script."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
from typing import Any, Iterator
import unittest
from unittest import mock

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


class FakeServer:
    """Records requests and answers like `gufo serve llm`."""

    def __init__(self, models: list[dict[str, Any]] | None = None,
                 key: str | None = None, loading_polls: int = 0,
                 content: str = "ready", finish: str = "stop"):
        self.models = [TEXT_MODEL] if models is None else models
        self.key = key
        self.loading_polls = loading_polls
        self.content = content
        self.finish = finish
        self.requests: list[dict[str, Any]] = []


def make_handler(state: FakeServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status: int, payload: dict[str, Any]):
            body = json.dumps(payload).encode()
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
            if state.loading_polls > 0:
                state.loading_polls -= 1
                return self.send(503, {"error": {
                    "message": "Loading model", "type": "unavailable_error",
                    "code": "loading"}})
            if state.key and self.headers.get("Authorization") != \
                    "Bearer " + state.key:
                return self.error(401, "invalid_api_key", "Unauthorized")
            if method == "GET" and self.path == "/v1/models":
                return self.send(200, {"object": "list",
                                       "data": state.models})
            if method == "POST" and self.path == "/v1/chat/completions":
                if body["model"] not in [m["id"] for m in state.models]:
                    return self.error(404, "model_not_found",
                                      "model not found")
                return self.send(200, {
                    "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": state.finish,
                                 "message": {"role": "assistant",
                                             "content": state.content}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2}})
            return self.error(404, "not_found", "unknown route")

        def do_GET(self):
            self.handle_any("GET")

        def do_POST(self):
            self.handle_any("POST")

    return Handler


@contextmanager
def serve(state: FakeServer) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def run_main(argv: list[str], env: dict[str, str] | None = None
             ) -> tuple[int, str]:
    out = io.StringIO()
    with mock.patch.dict(os.environ), mock.patch("sys.stdout", out):
        os.environ.pop("GUFO_API_KEY", None)
        os.environ.update(env or {})
        status = VERIFY.main(argv)
    return status, out.getvalue()


class VerifyServerTest(unittest.TestCase):
    def test_passes_with_one_chat_request(self):
        state = FakeServer()
        with serve(state) as url:
            status, out = run_main(["--base-url", url, "--json"])
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
        state = FakeServer(models=[tts, TEXT_MODEL])
        with serve(state) as url:
            status, out = run_main(["--base-url", url])
        self.assertEqual(status, 0, out)
        self.assertIn("PASS: gufo-local", out)
        self.assertEqual(state.requests[-1]["body"]["model"], "gufo-local")

    def test_bearer_key_from_environment_is_never_printed(self):
        state = FakeServer(key=KEY)
        with serve(state) as url:
            status, out = run_main(["--base-url", url],
                                   {"GUFO_API_KEY": KEY})
        self.assertEqual(status, 0, out)
        self.assertNotIn(KEY, out)
        self.assertIn("authenticated=true", out)
        self.assertTrue(all(r["auth"] == "Bearer " + KEY
                            for r in state.requests))

    def test_missing_key_explains_401(self):
        state = FakeServer(key=KEY)
        with serve(state) as url:
            status, out = run_main(["--base-url", url])
        self.assertEqual(status, 1)
        self.assertIn("FAIL: GET /v1/models returned 401", out)
        self.assertIn("GUFO_API_KEY", out)
        self.assertEqual(sum(r["method"] == "POST" for r in state.requests), 0)

    def test_waits_while_loading(self):
        state = FakeServer(loading_polls=2)
        with serve(state) as url, mock.patch.object(VERIFY.time, "sleep"):
            status, out = run_main(["--base-url", url, "--wait", "60"])
        self.assertEqual(status, 0, out)

    def test_loading_without_wait_fails_with_hint(self):
        state = FakeServer(loading_polls=1)
        with serve(state) as url:
            status, out = run_main(["--base-url", url])
        self.assertEqual(status, 1)
        self.assertIn("503 loading", out)
        self.assertIn("--wait", out)

    def test_unknown_model(self):
        with serve(FakeServer()) as url:
            status, out = run_main(["--base-url", url, "--model", "other"])
        self.assertEqual(status, 1)
        self.assertIn("'other' is not served (served: gufo-local)", out)

    def test_empty_text_at_length_suggests_budget(self):
        with serve(FakeServer(content="", finish="length")) as url:
            status, out = run_main(["--base-url", url])
        self.assertEqual(status, 1)
        self.assertIn("finish_reason length", out)
        self.assertIn("--max-tokens", out)

    def test_connection_refused(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        status, out = run_main(["--base-url", f"http://127.0.0.1:{port}"])
        self.assertEqual(status, 1)
        self.assertIn("not running", out)

    def test_refuses_cleartext_key_to_remote_host(self):
        status, out = run_main(["--base-url", "http://192.0.2.1:8080"],
                               {"GUFO_API_KEY": KEY})
        self.assertEqual(status, 1)
        self.assertIn("cleartext", out)
        self.assertNotIn(KEY, out)

    def test_usage_errors(self):
        for argv in (["--base-url", "http://127.0.0.1:8080/v1"],
                     ["--base-url", "127.0.0.1:8080"],
                     ["--max-tokens", "0"]):
            with self.subTest(argv=argv), \
                    mock.patch("sys.stderr", io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                VERIFY.main(argv)
            self.assertEqual(raised.exception.code, 2)

    def test_command_line_exit_status(self):
        env = {k: v for k, v in os.environ.items() if k != "GUFO_API_KEY"}
        with serve(FakeServer()) as url:
            done = subprocess.run(
                [sys.executable, "-I", str(SCRIPT), "--base-url", url],
                capture_output=True, text=True, timeout=30, env=env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PASS", done.stdout)


class SkillDocumentTest(unittest.TestCase):
    """Keeps the skill tied to the actual CLI and repository."""

    # Options of non-Gufo commands the skill uses.
    EXTERNAL = {"--preset", "--build", "--parallel"}

    def test_front_matter(self):
        text = SKILL.read_text()
        match = re.match(r"---\nname: ([a-z0-9-]+)\ndescription: (.+)\n---\n",
                         text)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), SKILL.parent.name)
        self.assertLessEqual(len(match.group(2)), 1024)

    def test_options_exist(self):
        cli = "\n".join(path.read_text(errors="replace")
                        for path in (ROOT / "src/cli").rglob("*")
                        if path.suffix in (".cpp", ".hpp", ".h"))
        known = set(re.findall(r'"(--[a-z][a-z0-9-]*)', cli))
        known |= set(re.findall(r'"(--[a-z][a-z0-9-]*)"', SCRIPT.read_text()))
        used = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)",
                              SKILL.read_text()))
        self.assertEqual(sorted(used - known - self.EXTERNAL), [])

    def test_repository_paths_exist(self):
        paths = re.findall(r"`((?:docs|tools)/[^`<>\s]+)`", SKILL.read_text())
        self.assertGreater(len(paths), 5)
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).exists())


if __name__ == "__main__":
    unittest.main()
