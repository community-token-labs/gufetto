#!/usr/bin/env python3
"""Regenerate harmony_goldens.json from the pinned gpt-oss chat template.

The goldens are rendered by Jinja2 with the Hugging Face apply_chat_template
environment, independently of Gufo's compiled renderer and its constants.

  python3 make_harmony_goldens.py TEMPLATE.jinja OUT.json

TEMPLATE.jinja is `tokenizer.chat_template` from the pinned GGUF; it is
byte-identical to llama.cpp's models/templates/openai-gpt-oss-120b.jinja.
Requires jinja2 (tested with 3.1.6).
"""

import hashlib
import json
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_SHA256 = "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146"
DATE = "2026-10-30"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                "days": {"type": "integer", "default": 1},
                "hourly": {"type": "boolean"},
            },
            "required": ["location"],
        },
    },
}
PING = {"type": "function", "function": {"name": "ping", "description": "Check liveness."}}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "patterns": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": ["integer", "null"]},
                "scope": {"type": "string", "nullable": True},
                "ratio": {"type": "number", "default": 0.5},
                "options": {"type": "object"},
                "anything": {},
            },
            "required": ["patterns"],
        },
    },
}

CASES = {
    "user_only": {"messages": [{"role": "user", "content": "Hello"}]},
    "system_high": {
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        "reasoning_effort": "high",
    },
    "identity_low": {
        "messages": [{"role": "user", "content": "Hi"}],
        "model_identity": "You are Gufo.",
        "reasoning_effort": "low",
    },
    "tools": {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [WEATHER, PING, SEARCH],
    },
    "tool_round_trip": {
        "messages": [
            {"role": "developer", "content": "Use tools."},
            {"role": "user", "content": "Weather in Tokyo?"},
            {
                "role": "assistant",
                "thinking": "Need the weather tool.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo, 東京", "days": 2, "hourly": False},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "{\"temp\": 21, \"note\": \"line\\nbreak \\\"q\\\"\"}"},
        ],
        "tools": [WEATHER],
    },
    "tool_then_final": {
        "messages": [
            {"role": "user", "content": "Weather in Tokyo?"},
            {
                "role": "assistant",
                "content": "Calling the tool.",
                "tool_calls": [{"function": {"name": "ping", "arguments": {}}}],
            },
            {"role": "tool", "content": "pong"},
            {"role": "assistant", "thinking": "dropped", "content": "It is up."},
            {"role": "user", "content": "Thanks"},
        ],
        "tools": [PING],
    },
    "multi_turn": {
        "messages": [
            {"role": "user", "content": "One"},
            {"role": "assistant", "thinking": "hidden", "content": "Two"},
            {"role": "user", "content": "Three"},
        ],
    },
    "training_final": {
        "messages": [
            {"role": "user", "content": "One"},
            {"role": "assistant", "thinking": "Count.", "content": "Two"},
        ],
        "add_generation_prompt": False,
    },
}


def render(template, case):
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: DATE
    kwargs = {k: v for k, v in case.items() if k not in ("messages", "add_generation_prompt")}
    return env.from_string(template).render(
        messages=case["messages"], add_generation_prompt=case.get("add_generation_prompt", True), **kwargs
    )


def main(template_path, out_path):
    with open(template_path, "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256:
        sys.exit("template does not match the pinned gpt-oss chat template")
    template = raw.decode("utf-8")
    out = {
        "template_sha256": TEMPLATE_SHA256,
        "generator": "jinja2 " + jinja2.__version__ + " (HF apply_chat_template environment)",
        "date": DATE,
        "cases": {name: {"input": case, "rendered": render(template, case)} for name, case in CASES.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
