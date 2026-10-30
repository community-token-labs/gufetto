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
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("qwen38_tokenizer_reference.json")
FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
MAX_BYTES = 32 * 1024 * 1024


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def fetch(url, optional=False):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as error:
        if optional and error.code == 404:
            return None
        raise
    if len(data) > MAX_BYTES:
        raise ValueError("metadata file exceeds 32 MiB")
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
        match = re.fullmatch(r"([\w.-]+/[\w.-]+)@(.+)", source)
        if not match:
            raise ValueError("HF source must be repo-owner/repo-name@revision")
        repo, revision = match.groups()
        revision_url = urllib.parse.quote(revision, safe="")
        metadata = json.loads(fetch(
            f"https://huggingface.co/api/models/{repo}/revision/{revision_url}"))
        revision = metadata["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("HF did not resolve an immutable commit")
        identity = {"repo": repo, "revision": revision}
        raw = {}
        for name in FILES:
            data = fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
                         optional=name == "chat_template.jinja")
            if data is not None:
                raw[name] = data
    return snapshot(raw, identity)


def snapshot(raw, identity):
    sections, controls = {}, {}

    def add_control(token, token_id):
        if not isinstance(token, str) or type(token_id) is not int or token_id < 0:
            raise ValueError("invalid control token entry")
        if token in controls and controls[token] != token_id:
            raise ValueError(f"conflicting IDs for control token {token}")
        controls[token] = token_id

    documents = {}
    for name in FILES[:2]:
        document = json.loads(raw[name])
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
    for token_id, entry in decoder.items():
        if not isinstance(entry, dict):
            raise ValueError("invalid added-token decoder entry")
        add_control(entry["content"], int(token_id))
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
            "sections": sections, "controls": controls, "templates": templates}


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
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        candidate = load_source(args.source, args.allow_network)
        report = compare(reference, candidate, catalogue(args.control_tokens))
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as error:
        print(f"tokenizer_diff: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return int(report["review_required"])


if __name__ == "__main__":
    sys.exit(main())
