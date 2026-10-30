#!/usr/bin/env python3
"""No model weights, HIP calls, host tuning or network; explicit --no-gpu."""
import pathlib
import os
import struct
import subprocess
import sys
import tempfile


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def main():
    binary = str(pathlib.Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="gufo-native-plan-cli-") as tmp:
        model = pathlib.Path(tmp) / "metadata.gguf"
        metadata = string("general.architecture") + struct.pack("<I", 8) + string("qwen4exp")
        metadata += string("qwen4exp.context_length") + struct.pack("<II", 4, 32768)
        raw = b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + metadata
        raw += b"\0" * (-len(raw) % 32)
        model.write_bytes(raw)

        def run(*args):
            return subprocess.run(
                [binary, "diagnose", *args], capture_output=True,
                text=True, timeout=10, check=False,
                env={**os.environ, "XDG_CACHE_HOME": str(model.parent / "identity-cache")},
            )

        base = ("--plan", "--model", str(model), "--no-gpu")
        result = run(*base)
        assert result.returncode == 0, result.stderr
        assert "UNPROVEN:" in result.stdout and "context=32768 sessions=1" in result.stdout
        assert "gguf_identity_sha256=" in result.stdout
        assert "FIT:" not in result.stdout
        result = run(*base, "--context=65536", "--sessions=4", "--enable-yarn",
                     "--kv-cache-type=q8_0", "--speculative=mtp", "--mtp-model=draft.gguf",
                     "--draft-tokens=7", "--prefill-chunk=4096", "--no-vision", "--cache-disk=off")
        assert result.returncode == 0, result.stderr
        assert "context=65536 sessions=4" in result.stdout
        for extra in [
            ("--sessions", "0"), ("--sessions", "-1"), ("--context", "4294967296"),
            ("--context", "1"), ("--draft-tokens", "0"), ("--prefill-chunk", "0"),
            ("--vision-pool-tokens", "0"), ("--no-vision", "--mmproj", "vision.gguf"),
            ("--cache-ram-bytes", "broken"), ("--cache-disk", tmp),
            ("--speculative", "mtp"), ("--speculative", "dflash2"),
            ("--kv-cache-type", "q4_0"), ("--pin-prefix-file", "x"),
            ("--json",), ("--output", "report.txt"),
        ]:
            result = run(*base, *extra)
            assert result.returncode == 2, (extra, result.stdout, result.stderr)
        assert run("--model", str(model), "--no-gpu").returncode == 2
        assert run("--plan=x", "--model", str(model), "--no-gpu").returncode == 2
        assert run("--plan", "--model", str(model), "--help").returncode == 0
        missing_context = model.parent / "no-context.gguf"
        metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
        no_context = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + metadata
        missing_context.write_bytes(no_context + b"\0" * (-len(no_context) % 32))
        result = run("--plan", "--model", str(missing_context), "--no-gpu")
        assert result.returncode == 0 and "UNPROVEN:" in result.stdout, result.stderr
        assert "no native context" in result.stdout
        # Independent spelling fixtures: reject aliases and trailing bytes.
        unsupported = model.parent / "unsupported.gguf"
        for architecture in ("qwen4", "Qwen4exp", "qwen4exp-next", "qwen4exp ", "qwen4exp\0", "llama"):
            metadata = string("general.architecture") + struct.pack("<I", 8) + string(architecture)
            metadata += string(architecture + ".context_length") + struct.pack("<II", 4, 32768)
            fixture = b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + metadata
            unsupported.write_bytes(fixture + b"\0" * (-len(fixture) % 32))
            result = run("--plan", "--model", str(unsupported), "--no-gpu", "--enable-yarn")
            assert result.returncode == 2, (architecture, result.stdout, result.stderr)
            assert "--enable-yarn family" in result.stderr, (architecture, result.stderr)
        assert model.read_bytes() == raw
        assert {p.name for p in model.parent.iterdir()} <= {"metadata.gguf", "no-context.gguf", "unsupported.gguf", "identity-cache"}
    print("model_admission_cli_test passed")


if __name__ == "__main__":
    main()
