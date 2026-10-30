#!/usr/bin/env python3
"""Check a running `gufo serve llm` with one short chat request.

Standard library only. It lists `/v1/models`, sends one non-streaming
`/v1/chat/completions` request and explains failures with the server's own
error codes. The bearer key, when needed, is read from an environment variable
and never printed. Exit status: 0 pass, 1 check failed, 2 usage error.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
PROMPT = "Reply with the single word: ready"

# Hints keyed by the server's error `code`, or by HTTP status as a fallback.
CODE_HINTS = {
    "loading": "the model is still loading; rerun with --wait SECONDS",
    "device_lost": "the GPU context was lost; restart `gufo serve` "
                   "(it exits with status 75)",
    "model_not_found": "the model ID is not served; use an ID listed by "
                       "/v1/models or the server's --served-model-name",
    "context_length_exceeded": "the prompt does not fit the loaded context",
}
STATUS_HINTS = {
    401: "the server requires its --api-key; export it in this shell under "
         "the variable named by --api-key-env (default GUFO_API_KEY)",
    404: "unknown route or model; check --base-url has no trailing /v1",
    413: "request too large for --max-request-bytes or the loaded context",
    429: "admission queue full (--max-pending); retry when idle",
    503: "model or backend unavailable",
}


class CheckError(Exception):
    """A failed check with a user-facing hint."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def error_details(body: bytes) -> tuple[str, str]:
    """Returns (code, message) from an OpenAI-style error body."""
    try:
        error = json.loads(body).get("error", {})
    except (ValueError, AttributeError):
        return "", body[:200].decode("utf-8", "replace")
    if not isinstance(error, dict):
        return "", ""
    return str(error.get("code") or ""), str(error.get("message") or "")


class Client:
    def __init__(self, base_url: str, key: str | None, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout
        # Never route a local server check through an environment proxy.
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}))

    def request(self, method: str, path: str,
                payload: dict[str, Any] | None = None) -> tuple[int, bytes]:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data,
                                     method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", "Bearer " + self.key)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise CheckError(
                f"cannot reach {self.base_url}: {reason}",
                "a refused connection means `gufo serve` is not running on "
                "this host/port") from None


def fail_for(what: str, status: int, body: bytes) -> CheckError:
    code, message = error_details(body)
    hint = CODE_HINTS.get(code) or STATUS_HINTS.get(status, "")
    label = f"{status} {code}".strip()
    return CheckError(f"{what} returned {label}: {message}".rstrip(": "), hint)


def list_models(client: Client, wait: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + wait
    while True:
        status, body = client.request("GET", "/v1/models")
        if status == 503 and error_details(body)[0] == "loading" and \
                time.monotonic() < deadline:
            time.sleep(1.0)
            continue
        if status != 200:
            raise fail_for("GET /v1/models", status, body)
        try:
            models = json.loads(body)["data"]
        except (ValueError, KeyError, TypeError):
            raise CheckError("GET /v1/models returned no model list") from None
        if not isinstance(models, list) or not models:
            raise CheckError("GET /v1/models lists no models")
        return models


def select_model(models: list[dict[str, Any]], wanted: str | None) -> dict:
    ids = [str(model.get("id")) for model in models]
    if wanted:
        for model in models:
            if model.get("id") == wanted:
                return model
        raise CheckError(f"model {wanted!r} is not served (served: "
                         f"{', '.join(ids)})", CODE_HINTS["model_not_found"])
    for model in models:
        if "context_length" in model:  # The text model entry.
            return model
    raise CheckError(f"no text model is served (served: {', '.join(ids)})",
                     "start `gufo serve llm`, or pass --model")


def chat(client: Client, model: str, max_tokens: int,
         reasoning_effort: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    started = time.monotonic()
    status, body = client.request("POST", "/v1/chat/completions", payload)
    elapsed = time.monotonic() - started
    if status != 200:
        raise fail_for("POST /v1/chat/completions", status, body)
    try:
        response = json.loads(body)
        choice = response["choices"][0]
        content = choice["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise CheckError("chat response is not a Chat Completions object") \
            from None
    finish = str(choice.get("finish_reason") or "")
    if not isinstance(content, str) or not content.strip():
        hint = ("the output budget ran out before visible text; raise "
                "--max-tokens or keep thinking off" if finish == "length"
                else "")
        raise CheckError(f"chat response has no text (finish_reason "
                         f"{finish or 'missing'})", hint)
    usage = response.get("usage") or {}
    return {
        "content": content.strip(),
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(elapsed, 3),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="server root, without /v1 (default: %(default)s)")
    parser.add_argument("--model",
                        help="model ID to request (default: the text model "
                             "listed by /v1/models)")
    parser.add_argument("--api-key-env", default="GUFO_API_KEY",
                        metavar="NAME",
                        help="environment variable holding the server's "
                             "bearer key, used only if set "
                             "(default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="output budget of the request "
                             "(default: %(default)s)")
    parser.add_argument("--reasoning-effort", default="none",
                        help="Chat reasoning_effort; 'none' disables "
                             "thinking, '' omits the field "
                             "(default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-request timeout in seconds "
                             "(default: %(default)s)")
    parser.add_argument("--wait", type=float, default=0.0, metavar="SECONDS",
                        help="keep polling while the server answers 503 "
                             "loading (default: %(default)s)")
    parser.add_argument("--allow-cleartext-key", action="store_true",
                        help="send the key over plain HTTP to a non-loopback "
                             "host")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON result object")
    args = parser.parse_args(argv)
    url = urllib.parse.urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.hostname:
        parser.error("--base-url must be an http:// or https:// URL")
    if url.path.rstrip("/").endswith("/v1"):
        parser.error("--base-url is the server root; drop the trailing /v1")
    if args.max_tokens <= 0 or args.timeout <= 0 or args.wait < 0:
        parser.error("--max-tokens and --timeout must be positive, "
                     "--wait non-negative")
    args.host = url.hostname
    args.scheme = url.scheme
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    key = os.environ.get(args.api_key_env) or None
    if key and args.scheme == "http" and not is_loopback(args.host) and \
            not args.allow_cleartext_key:
        raise CheckError(
            f"refusing to send the bearer key in cleartext to {args.host}",
            "use loopback, a TLS reverse proxy, or --allow-cleartext-key")
    client = Client(args.base_url, key, args.timeout)
    model = select_model(list_models(client, args.wait), args.model)
    architecture = model.get("architecture") or {}
    result = {
        "model": model.get("id"),
        "context_length": model.get("context_length"),
        "input_modalities": architecture.get("input_modalities"),
        "authenticated": key is not None,
    }
    result.update(chat(client, str(model.get("id")), args.max_tokens,
                       args.reasoning_effort))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except CheckError as error:
        if args.json:
            print(json.dumps({"ok": False, "error": str(error),
                              "hint": error.hint}))
        else:
            print(f"FAIL: {error}")
            if error.hint:
                print(f"hint: {error.hint}")
        return 1
    if args.json:
        print(json.dumps({"ok": True, **result}))
    else:
        print(f"PASS: {result['model']} answered {result['content']!r} "
              f"in {result['seconds']} s")
        print(f"context_length={result['context_length']} "
              f"input_modalities={result['input_modalities']} "
              f"finish_reason={result['finish_reason']} "
              f"prompt_tokens={result['prompt_tokens']} "
              f"completion_tokens={result['completion_tokens']} "
              f"authenticated={str(result['authenticated']).lower()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
