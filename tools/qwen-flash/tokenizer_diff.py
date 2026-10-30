#!/usr/bin/env python3
"""Offline-first Qwen release metadata comparison; never downloads weights.

Exit 0: identical semantic sections/templates/control IDs; 1: review needed;
2: invalid/incomplete input. Section fingerprints detect vocabulary/merges and
all config changes without vendoring the reference vocabulary. This is not a
rendered-template or model-quality oracle. See tokenizer_diff.md.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("qwen38_tokenizer_reference.json")
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
MAX_BYTES = 32 * 1024 * 1024
DOWNLOAD_SECONDS = 60


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def parse_json(data):
    return json.loads(data, object_pairs_hook=strict_object, parse_constant=reject_constant)


def validate_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    allowed = (host == "huggingface.co" or
               re.fullmatch(r"cdn-lfs(?:-[a-z0-9-]+)?\.(?:huggingface\.co|hf\.co)", host) or
               re.fullmatch(r"[a-z0-9.-]+\.cdn\.hf\.co", host) or
               host == "cas-bridge.xethub.hf.co")
    if (parsed.scheme != "https" or not allowed or parsed.username or parsed.password or
            parsed.port not in (None, 443)):
        raise ValueError("metadata URL must use HTTPS on HF or its approved CDN hosts")


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def fetch(url, optional=False, expected_sha=None):
    validate_url(url)
    deadline = time.monotonic() + DOWNLOAD_SECONDS
    checksums = [expected_sha] if expected_sha else []

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("metadata download exceeded overall deadline")
        return seconds

    class Redirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, response, code, message, headers, new_url):
            validate_url(new_url)
            remaining()
            if headers.get("X-Linked-Etag"):
                checksums.append(headers["X-Linked-Etag"])
            redirected = super().redirect_request(request, response, code, message, headers, new_url)
            request.timeout = remaining()
            redirected.timeout = request.timeout
            return redirected

    try:
        opener = urllib.request.build_opener(Redirects())
        with opener.open(url, timeout=remaining()) as response:
            # Xet CDN ETags identify storage objects, not file SHA256s.
            # Prefer HF's authoritative LFS/X-Linked-ETag when provided.
            if not checksums and response.headers.get("ETag"):
                checksums.append(response.headers["ETag"])
            chunks, total = [], 0
            while True:
                # HTTPResponse.read1 performs at most one underlying read;
                # refresh its socket timeout to enforce a whole-download budget.
                seconds = remaining()
                socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                if socket is not None:
                    socket.settimeout(seconds)
                chunk = response.read1(min(65536, MAX_BYTES + 1 - total))
                remaining()
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("metadata file exceeds 32 MiB")
            data = b"".join(chunks)
            length = response.headers.get("Content-Length")
            if length is not None and len(data) != int(length):
                raise ValueError("truncated metadata response")
    except urllib.error.HTTPError as error:
        if optional and error.code == 404:
            return None
        raise
    for checksum in checksums:
        checksum = checksum.strip('"')
        if re.fullmatch(r"[0-9a-f]{64}", checksum):
            actual = hashlib.sha256(data).hexdigest()
        elif re.fullmatch(r"[0-9a-f]{40}", checksum):
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        else:
            continue  # Opaque HTTP ETags are not content checksums.
        if actual != checksum:
            raise ValueError("HF metadata checksum mismatch")
    return data


def load_source(source, allow_network=False):
    directory = Path(source)
    if directory.is_dir():
        raw = {}
        for name in FILES:
            path = directory / name
            if not path.exists() and name == "chat_template.jinja":
                continue
            if path.stat().st_size > MAX_BYTES:
                raise ValueError(f"{name} exceeds 32 MiB")
            raw[name] = path.read_bytes()
        identity = {"local": str(directory.resolve())}
    else:
        if not allow_network:
            raise ValueError("not a local directory; HF access requires --allow-network")
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)@(.+)", source)
        if not match or ".." in match.group(1):
            raise ValueError("HF source must be repo-owner/repo-name@revision (no '..')")
        repo, revision = match.groups()
        revision_url = urllib.parse.quote(revision, safe="")
        metadata = parse_json(fetch(
            f"https://huggingface.co/api/models/{repo}/revision/{revision_url}?blobs=true"))
        revision = metadata["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("HF did not resolve an immutable commit")
        identity = {"repo": repo, "revision": revision}
        raw = {}
        siblings = {entry["rfilename"]: entry for entry in metadata.get("siblings", [])}
        for name in FILES:
            checksum = siblings.get(name, {}).get("lfs", {}).get("sha256")
            data = fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                         optional=name == "chat_template.jinja", expected_sha=checksum)
            if data is not None:
                raw[name] = data
    return snapshot(raw, identity)


def snapshot(raw, identity):
    sections, controls, control_flags = {}, {}, {}

    def add_control(token, token_id):
        if not isinstance(token, str) or type(token_id) is not int or token_id < 0:
            raise ValueError("invalid control token entry")
        if token in controls and controls[token] != token_id:
            raise ValueError(f"conflicting IDs for control token {token}")
        controls[token] = token_id

    documents = {}
    for name in FILES[:2]:
        document = parse_json(raw[name])
        if not isinstance(document, dict):
            raise ValueError(f"{name} must contain a JSON object")
        documents[name] = document
        # Split the potentially huge model into vocab/merges/etc. fingerprints.
        for key, value in document.items():
            if name == "tokenizer.json" and key == "model" and isinstance(value, dict):
                for field, content in value.items():
                    sections[f"{name}/model/{field}"] = digest(content)
            else:
                sections[f"{name}/{key}"] = digest(value)
    tokenizer = documents["tokenizer.json"]
    config = documents["tokenizer_config.json"]
    model = tokenizer.get("model")
    if not isinstance(model, dict):
        raise ValueError("expected Qwen tokenizer model object")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("expected Qwen tokenizer model.vocab object")
    for token, token_id in vocab.items():
        if re.fullmatch(r"<\|[^\n]+\|>|</?(?:think|tool_call|tool_response)>", token):
            add_control(token, token_id)
    added = tokenizer.get("added_tokens", [])
    decoder = config.get("added_tokens_decoder", {})
    if not isinstance(added, list) or not isinstance(decoder, dict):
        raise ValueError("invalid added-token catalogue")
    for entry in added:
        if not isinstance(entry, dict):
            raise ValueError("invalid added-token entry")
        add_control(entry["content"], entry["id"])
        control_flags.setdefault(entry["content"], {}).update({
            "tokenizer.json/" + key: entry[key]
            for key in ("special", "normalized", "lstrip", "rstrip", "single_word") if key in entry})
    for token_id, entry in decoder.items():
        if not isinstance(entry, dict):
            raise ValueError("invalid added-token decoder entry")
        if not re.fullmatch(r"0|[1-9][0-9]*", token_id):
            raise ValueError("added-token decoder ID must be canonical decimal")
        add_control(entry["content"], int(token_id))
        control_flags.setdefault(entry["content"], {}).update({
            "tokenizer_config.json/" + key: entry[key]
            for key in ("special", "normalized", "lstrip", "rstrip", "single_word") if key in entry})
    templates = {}
    if "chat_template" in config:
        templates["tokenizer_config.json/chat_template"] = digest(config["chat_template"])
    if "chat_template.jinja" in raw:
        templates["chat_template.jinja"] = digest(raw["chat_template.jinja"].decode("utf-8"))
    if not templates:
        raise ValueError("no chat template in tokenizer_config.json or chat_template.jinja")
    return {"schema": 1, "source": identity,
            "file_sha256": {name: hashlib.sha256(data).hexdigest()
                            for name, data in raw.items()},
            "sections": sections, "controls": controls,
            "control_flags": {token: flags for token, flags in control_flags.items() if flags},
            "templates": templates}


def catalogue(path):
    text = path.read_text(encoding="utf-8")
    tokens = re.findall(r'inline constexpr std::string_view\s+\w+\s*=\s*"([^"\n]+)";', text)
    if not tokens:
        raise ValueError("no control-token constants found in catalogue")
    return set(tokens)


def differences(before, after):
    return [{"key": key, "before": before.get(key), "after": after.get(key)}
            for key in sorted(before.keys() | after.keys())
            if before.get(key) != after.get(key) or (key in before) != (key in after)]


def compare(reference, candidate, known):
    if reference.get("schema") != 1:
        raise ValueError("unsupported reference snapshot schema")
    control_changes = differences(reference["controls"], candidate["controls"])
    by_token = {change["key"]: change for change in control_changes}
    for flags in differences(reference.get("control_flags", {}), candidate["control_flags"]):
        token = flags["key"]
        change = by_token.setdefault(token, {"key": token, "before": reference["controls"].get(token),
                                             "after": candidate["controls"].get(token)})
        change["flag_changes"] = differences(flags["before"] or {}, flags["after"] or {})
    control_changes = [by_token[token] for token in sorted(by_token)]
    for change in control_changes:
        change["in_cpp_catalogue"] = change["key"] in known
    sections = differences(reference["sections"], candidate["sections"])
    templates = differences(reference["templates"], candidate["templates"])
    # C++ owns spellings, not IDs. IDs must come from independent reference data.
    missing = sorted(token for token in known
                     if token in reference["controls"] and token not in candidate["controls"])
    return {"reference": reference["source"], "candidate": candidate["source"],
            "candidate_file_sha256": candidate["file_sha256"],
            "section_changes": sections, "template_changes": templates,
            "control_changes": control_changes, "missing_catalogue_tokens": missing,
            "catalogue_absent_in_reference": sorted(known - reference["controls"].keys()),
            "review_required": bool(sections or templates or control_changes or missing)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="local HF metadata directory or owner/repo@revision")
    parser.add_argument("--allow-network", action="store_true",
                        help="allow only public HF metadata fetches; default is offline")
    parser.add_argument("--reference", type=Path, default=REFERENCE,
                        help="pinned reference snapshot JSON (default: Qwen3.8 Flash-Next)")
    parser.add_argument("--control-tokens", type=Path,
                        default=ROOT / "src/models/qwen/control_tokens.hpp")
    args = parser.parse_args(argv)
    try:
        reference = parse_json(args.reference.read_text(encoding="utf-8"))
        candidate = load_source(args.source, args.allow_network)
        report = compare(reference, candidate, catalogue(args.control_tokens))
    except Exception as error:  # Invalid input/transport must never masquerade as review-needed exit 1.
        print(f"tokenizer_diff: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return int(report["review_required"])


if __name__ == "__main__":
    sys.exit(main())
