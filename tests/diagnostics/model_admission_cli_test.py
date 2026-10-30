#!/usr/bin/env python3
"""No model weights, HIP calls, host tuning or network; explicit --no-gpu."""
import pathlib
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
            )

        base = ("--plan", "--model", str(model), "--no-gpu")
        result = run(*base)
        assert result.returncode == 0, result.stderr
        assert "UNPROVEN:" in result.stdout and "context=32768 sessions=1" in result.stdout
        assert "requested_file_sha256=" in result.stdout
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
        assert model.read_bytes() == raw
        assert list(model.parent.iterdir()) == [model]
    print("model_admission_cli_test passed")


if __name__ == "__main__":
    main()
