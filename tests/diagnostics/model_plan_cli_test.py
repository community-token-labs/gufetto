#!/usr/bin/env python3
"""Read-only CLI contract, requiring no model, network, or HIP device."""
import pathlib
import subprocess
import sys
import tempfile


def main():
    gufo = str(pathlib.Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="gufo-plan-cli-") as tmp:
        root = pathlib.Path(tmp)
        manifest = root / "models.tsv"
        manifest.write_text(
            "group\tstatus\trepo\trevision\tfile\tbytes\tsha256\tlicense\tgated\t"
            "role\twhy\tfit\tsource\n"
            f"test\tcandidate\ttest/model\t{'a' * 40}\tabsent.gguf\t3\t{'b' * 64}\t"
            "MIT\tfalse\ttarget\ttest\tFIT is untrusted prose\thttps://example.invalid\n"
        )
        initial = manifest.read_bytes()

        def run(*args):
            return subprocess.run(
                [gufo, "diagnose", *args], cwd=root, capture_output=True,
                text=True, timeout=10, check=False,
            )

        report = run("--plan", str(manifest))
        assert report.returncode == 0, report.stderr
        assert ": UNPROVEN:" in report.stdout
        assert "local=MISSING" in report.stdout
        assert report.stdout.endswith(initial.decode())
        assert ": FIT" not in report.stdout
        assert run(f"--plan={manifest}").stdout == report.stdout
        for args in [
            ("--plan",), ("--plan=",), ("--plan", "--json"),
            ("--plan", str(manifest), "--json"),
            ("--plan", str(manifest), "--fingerprint"),
            ("--plan", str(manifest), "--benchmark", "allocation"),
            ("--plan", str(manifest), "--output", str(root / "forbidden")),
            ("--plan", str(root / "missing.tsv")),
        ]:
            result = run(*args)
            assert result.returncode == 2, (args, result.stdout, result.stderr)
        assert sorted(p.name for p in root.iterdir()) == ["models.tsv"]
        assert manifest.read_bytes() == initial
    print("model_plan_cli_test passed")


if __name__ == "__main__":
    main()
