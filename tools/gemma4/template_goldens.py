#!/usr/bin/env python3
"""Render the Gemma 4 chat-template goldens with the GGUF's own Jinja source.

Reads tokenizer.chat_template from the target GGUF's metadata (never tensor
data), checks its SHA-256 against the qualified Unsloth revision, and renders
tests/models/gemma4/fixtures/template_cases.json the way Hugging Face
transformers does: a sandboxed Jinja2 environment with trim_blocks,
lstrip_blocks and the loopcontrols extension, bos_token "<bos>". Tool calls
are passed in OpenAI form with their arguments as a mapping, as llama.cpp
passes them to this template; an empty id is omitted. The template text
itself is not written anywhere.

    PYTHON_WITH_JINJA2 -I tools/gemma4/template_goldens.py --model TARGET.gguf \\
        --manifest MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_metadata import read_metadata  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/models/gemma4/fixtures"
QUALIFIED_SHA256 = "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b"


def raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for message in messages:
        converted = {k: v for k, v in message.items() if k != "tool_calls"}
        if "tool_calls" in message:
            converted["tool_calls"] = []
            for call in message["tool_calls"]:
                function = {"name": call["name"], "arguments": call["arguments"]}
                entry = {"type": "function", "function": function}
                if call.get("id"):
                    entry["id"] = call["id"]
                converted["tool_calls"].append(entry)
        out.append(converted)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    md, _ = read_metadata(args.model)
    source = md["tokenizer.chat_template"]
    digest = hashlib.sha256(source).hexdigest()
    if digest != QUALIFIED_SHA256:
        raise SystemExit(f"unqualified template sha256 {digest}")
    manifest = json.loads(args.manifest.read_text())
    record = next(f for f in manifest["files"] if f["file"] == args.model.name)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source.decode("utf-8"))
    spec = json.loads((FIXTURES / "template_cases.json").read_text())
    rendered = {}
    for name, case in spec["cases"].items():
        text = template.render(
            messages=openai_messages(case["messages"]),
            tools=[spec["tools"][t] for t in case.get("tools", [])] or None,
            add_generation_prompt=case.get("add_generation_prompt", True),
            enable_thinking=case.get("enable_thinking", False),
            preserve_thinking=case.get("preserve_thinking", False),
            bos_token="<bos>")
        rendered[name] = {"text": text,
                          "sha256": hashlib.sha256(text.encode()).hexdigest()}

    (FIXTURES / "template_goldens.json").write_text(json.dumps({
        "schema": "gufo.gemma4-template-goldens.v1",
        "generator": "tools/gemma4/template_goldens.py",
        "renderer": (f"jinja2 {version('jinja2')} (markupsafe "
                     f"{version('markupsafe')}), ImmutableSandboxedEnvironment"
                     ", trim_blocks, lstrip_blocks, loopcontrols; python "
                     f"{platform.python_version()}"),
        "template": {"source": "tokenizer.chat_template", "sha256": digest,
                     "repo": record["repo"], "revision": record["revision"],
                     "file": record["file"], "file_sha256": record["sha256"]},
        "cases": rendered,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rendered {len(rendered)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
