#!/usr/bin/env python3
"""Check a running `gufo serve llm` with one short chat request.

Standard library only. It lists `/v1/models`, sends one non-streaming
`/v1/chat/completions` request and explains failures with the server's own
error codes. The bearer key, when needed, is read from an environment variable
and never printed. Exit status: 0 pass, 1 check failed, 2 usage error.

Every string (and integer) decoded from a response is scrubbed of the literal
key before it is truncated or printed. Keys the server echoes in another
encoding, such as base64 or percent-encoding, are out of scope: only a hostile
server would do that, and it already received the key.
"""

from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
# Non-ASCII, so it can never contain a valid (printable ASCII) key and a
# second scrub pass leaves it unchanged.
REDACTED = "\u2022\u2022\u2022"
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
    401: "the server requires its API key; export it in this shell under "
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


class UsageError(Exception):
    """An invalid command line; exit status 2."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str):
        raise UsageError(message)


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect, so a credential cannot be re-sent elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CheckError(
            f"server answered {code} redirect; redirects are not followed",
            "point --base-url at the Gufo server itself")


def is_numeric_loopback(host: str) -> bool:
    """True only for literal 127.0.0.0/8 or ::1; names are never trusted."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def valid_key(value: str) -> bool:
    """Printable ASCII without spaces or control characters (0x21-0x7e)."""
    return re.fullmatch(r"[\x21-\x7e]+", value) is not None


def scrub(value: Any, secret: str | None) -> Any:
    """Replaces the secret in every string, before truncation or output."""
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, REDACTED)
    if isinstance(value, int) and not isinstance(value, bool):
        return REDACTED if str(value) == secret else value
    if isinstance(value, list):
        return [scrub(item, secret) for item in value]
    if isinstance(value, dict):
        return {scrub(k, secret): scrub(v, secret) for k, v in value.items()}
    return value


def load_json(body: bytes, secret: str | None) -> Any:
    """Decodes a body and scrubs it, so no later step sees the secret."""
    return scrub(json.loads(body), secret)


def error_details(body: bytes, secret: str | None) -> tuple[str, str]:
    """Returns (code, message) from an OpenAI-style error body."""
    try:
        error = load_json(body, secret).get("error", {})
    except (ValueError, AttributeError):
        return "", ""
    if not isinstance(error, dict):
        return "", ""
    code, message = error.get("code"), error.get("message")
    return (code if isinstance(code, str) else "",
            message[:300] if isinstance(message, str) else "")


class Client:
    def __init__(self, base_url: str, key: str | None, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout
        # Never route a local server check through an environment proxy.
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), RejectRedirects())

    def request(self, method: str, path: str,
                payload: dict[str, Any] | None = None) -> tuple[int, bytes]:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data=data,
                                     method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", "Bearer " + self.key)
        what = f"{method} {path}"
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return response.status, response.read()
        except CheckError:
            raise
        except urllib.error.HTTPError as error:
            try:
                return error.code, error.read()
            except (OSError, http.client.HTTPException):
                return error.code, b""
        except TimeoutError:
            raise CheckError(f"{what} timed out after {self.timeout:g} s",
                             "the server is busy or stuck; check its log, "
                             "or raise --timeout") from None
        except http.client.IncompleteRead:
            raise CheckError(f"{what} response body was truncated") from None
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise CheckError(
                    f"{what} timed out after {self.timeout:g} s",
                    "the server is busy or stuck; check its log, or raise "
                    "--timeout") from None
            raise CheckError(
                f"cannot reach {self.base_url}: "
                f"{type(error.reason).__name__}",
                "a refused connection means `gufo serve` is not running on "
                "this host/port") from None
        except (OSError, http.client.HTTPException, ValueError) as error:
            # Messages can quote request data, so only the type is reported.
            raise CheckError(f"{what} failed: {type(error).__name__}") \
                from None


def fail_for(what: str, status: int, body: bytes,
             secret: str | None) -> CheckError:
    code, message = error_details(body, secret)
    hint = CODE_HINTS.get(code) or STATUS_HINTS.get(status, "")
    label = f"{status} {code}".strip()
    return CheckError(f"{what} returned {label}: {message}".rstrip(": "), hint)


def parse_json(what: str, body: bytes, secret: str | None) -> Any:
    try:
        return load_json(body, secret)
    except ValueError:
        raise CheckError(f"{what} returned a body that is not JSON") from None


def is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and \
        value >= 0


def list_models(client: Client, wait: float) -> list[dict[str, Any]]:
    deadline = time.monotonic() + wait
    while True:
        status, body = client.request("GET", "/v1/models")
        if status == 503 and error_details(body, client.key)[0] == "loading" and \
                time.monotonic() < deadline:
            time.sleep(1.0)
            continue
        if status != 200:
            raise fail_for("GET /v1/models", status, body, client.key)
        listing = parse_json("GET /v1/models", body, client.key)
        models = listing.get("data") if isinstance(listing, dict) else None
        if not isinstance(models, list) or not models:
            raise CheckError("GET /v1/models returned no model list")
        for model in models:
            if not isinstance(model, dict) or \
                    not isinstance(model.get("id"), str) or not model["id"]:
                raise CheckError("GET /v1/models lists a malformed entry")
        return models


def select_model(models: list[dict[str, Any]], wanted: str | None) -> dict:
    ids = [model["id"] for model in models]
    if wanted:
        for model in models:
            if model["id"] == wanted:
                break
        else:
            raise CheckError(f"model {wanted!r} is not served (served: "
                             f"{', '.join(ids)})",
                             CODE_HINTS["model_not_found"])
    else:
        text = [model for model in models if "context_length" in model]
        if not text:
            raise CheckError(f"no text model is served (served: "
                             f"{', '.join(ids)})",
                             "start `gufo serve llm`, or pass --model")
        model = text[0]
    context = model.get("context_length")
    if context is not None and not is_count(context):
        raise CheckError("GET /v1/models reports a malformed context_length")
    architecture = model.get("architecture")
    modalities = architecture.get("input_modalities") \
        if isinstance(architecture, dict) else None
    if modalities is not None and (
            not isinstance(modalities, list) or
            not all(isinstance(item, str) for item in modalities)):
        raise CheckError("GET /v1/models reports malformed input_modalities")
    return model


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
    what = "POST /v1/chat/completions"
    started = time.monotonic()
    status, body = client.request("POST", "/v1/chat/completions", payload)
    elapsed = time.monotonic() - started
    if status != 200:
        raise fail_for(what, status, body, client.key)
    response = parse_json(what, body, client.key)
    malformed = CheckError(f"{what} returned a malformed Chat Completions "
                           "object")
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or \
            not isinstance(choices[0], dict):
        raise malformed
    choice = choices[0]
    message = choice.get("message")
    finish = choice.get("finish_reason")
    if not isinstance(message, dict) or \
            not isinstance(finish, (str, type(None))):
        raise malformed
    content = message.get("content")
    if not isinstance(content, (str, type(None))):
        raise malformed
    usage = response.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise malformed
    for field in ("prompt_tokens", "completion_tokens"):
        if usage.get(field) is not None and not is_count(usage[field]):
            raise malformed
    finish = finish or ""
    if not content or not content.strip():
        hint = ("the output budget ran out before visible text; raise "
                "--max-tokens or keep thinking off" if finish == "length"
                else "")
        raise CheckError(f"chat response has no text (finish_reason "
                         f"{finish or 'missing'})", hint)
    return {
        "content": content.strip()[:200],
        "finish_reason": finish,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(elapsed, 3),
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = Parser(description=__doc__.splitlines()[0], allow_abbrev=False)
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
                        help="send the key over plain HTTP to a host that is "
                             "not 127.0.0.1 or ::1")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON result object for every check "
                             "or usage error; -h/--help still prints plain "
                             "help and exits 0")
    args = parser.parse_args(argv)
    try:
        url = urllib.parse.urlsplit(args.base_url)
        port = url.port
        host = url.hostname
    except ValueError:
        raise UsageError("--base-url is not a valid URL") from None
    if url.scheme not in ("http", "https") or not host:
        raise UsageError("--base-url must be an http:// or https:// URL")
    if port == 0:
        raise UsageError("--base-url port must be 1-65535")
    if url.username is not None or url.password is not None:
        raise UsageError("--base-url must not contain credentials")
    if url.query or url.fragment:
        raise UsageError("--base-url must not contain a query or fragment")
    if url.path.rstrip("/").endswith("/v1"):
        raise UsageError(
            "--base-url is the server root; drop the trailing /v1")
    if args.max_tokens <= 0 or not args.timeout > 0 or not args.wait >= 0:
        raise UsageError("--max-tokens and --timeout must be positive, "
                         "--wait non-negative")
    args.host = host
    args.scheme = url.scheme
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    key = os.environ.get(args.api_key_env) or None
    if key is not None and not valid_key(key):
        raise CheckError(
            f"the value of {args.api_key_env} is not a valid bearer token",
            "use printable ASCII without spaces or line breaks")
    if key and args.scheme == "http" and not is_numeric_loopback(args.host) \
            and not args.allow_cleartext_key:
        raise CheckError(
            f"refusing to send the bearer key in cleartext to {args.host}",
            "use http://127.0.0.1 or http://[::1], a TLS reverse proxy, or "
            "--allow-cleartext-key")
    client = Client(args.base_url, key, args.timeout)
    model = select_model(list_models(client, args.wait), args.model)
    architecture = model.get("architecture")
    result = {
        "model": model["id"],
        "context_length": model.get("context_length"),
        "input_modalities": architecture.get("input_modalities")
        if isinstance(architecture, dict) else None,
        "authenticated": key is not None,
    }
    result.update(chat(client, model["id"], args.max_tokens,
                       args.reasoning_effort))
    return result


def report(as_json: bool, ok: bool, fields: dict[str, Any],
           lines: list[str], secret: str | None) -> None:
    # Scrub values before serialisation, so escaping cannot hide the secret.
    if as_json:
        print(json.dumps(scrub({"ok": ok, **fields}, secret)))
    else:
        print("\n".join(scrub(lines, secret)))


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    if hasattr(sys.stdout, "reconfigure"):
        # The redaction mark is non-ASCII; never fail on a narrow terminal.
        sys.stdout.reconfigure(errors="backslashreplace")
    as_json = "--json" in raw
    try:
        args = parse_args(raw)
    except UsageError as error:
        if as_json:
            report(True, False, {"error": str(error), "usage": True}, [],
                   None)
        print(f"usage error: {error}", file=sys.stderr)
        return 2
    secret = os.environ.get(args.api_key_env) or None
    try:
        result = run(args)
    except CheckError as error:
        report(args.json, False, {"error": str(error), "hint": error.hint},
               [f"FAIL: {error}"] + ([f"hint: {error.hint}"]
                                     if error.hint else []), secret)
        return 1
    except Exception as error:  # Never print a traceback that may hold data.
        report(args.json, False,
               {"error": f"internal error: {type(error).__name__}",
                "hint": ""},
               [f"FAIL: internal error: {type(error).__name__}"], secret)
        return 1
    report(args.json, True, result, [
        f"PASS: {result['model']} answered {result['content']!r} "
        f"in {result['seconds']} s",
        f"context_length={result['context_length']} "
        f"input_modalities={result['input_modalities']} "
        f"finish_reason={result['finish_reason']} "
        f"prompt_tokens={result['prompt_tokens']} "
        f"completion_tokens={result['completion_tokens']} "
        f"authenticated={str(result['authenticated']).lower()}"], secret)
    return 0


if __name__ == "__main__":
    sys.exit(main())
