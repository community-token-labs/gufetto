#!/usr/bin/env python3
"""Fixture-only driver tests: no real models, native encoder, GPU or dependencies.

Run: python3 -B tests/tools/vision_encoder_capture_test.py
PNG construction is hand-written here, independent of vision_images/generator.
Only model pins/source paths are replaced with tiny local fixtures; production
validation, packet freezing, process lifecycle and capture parsing remain active.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/quality/vision_encoder_capture.py"
SPEC = importlib.util.spec_from_file_location("vision_encoder_capture", SCRIPT)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)
GOOD = dict(available=10 * 1024**3, rss=100 * 1024**2, disk_free=100_000_000_000)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def chunk(kind, raw):
    return struct.pack(">I", len(raw)) + kind + raw + struct.pack(">I", zlib.crc32(kind + raw))


def png(width=32, height=32, depth=8, color=2, interlace=0, filter_byte=0, extra=b"", compressed=None):
    pixels = bytes((17, 128, 250)) * width * height
    scanlines = b"".join(bytes([filter_byte]) + pixels[y * width * 3:(y + 1) * width * 3]
                         for y in range(height))
    data = zlib.compress(scanlines) if compressed is None else compressed
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, interlace))
            + extra + chunk(b"IDAT", data) + chunk(b"IEND", b"")), pixels


@unittest.skipUnless(sys.platform == "linux", "capture supervision uses Linux /proc and nice")
class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"tiny production fixture")
        self.dev = self.root / "development.gguf"
        self.dev.write_bytes(b"tiny development fixture")
        self.binary = self.root / "fake-encoder"
        self.backend()
        self.sources = []
        for name in ("driver.py", "math.hpp", "math.cpp", "encoder.cpp"):
            path = self.root / name
            path.write_text("independent fixture " + name)
            self.sources.append(path)
        for name, value in (("SOURCE_FILES", self.sources),
                            ("PRODUCTION", (self.model.stat().st_size, digest(self.model.read_bytes()))),
                            ("DEVELOPMENT", (self.dev.stat().st_size, digest(self.dev.read_bytes())))):
            mock = patch.object(DRIVER, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        raw, pixels = png()
        self.image = self.root / "tiny.png"
        self.image.write_bytes(raw)
        self.images = self.root / "images.json"
        self.images.write_text(json.dumps(dict(schema="gufo-vision-images-v1", cases=[
            dict(id="tiny", width=32, height=32, file="tiny.png", sha256=digest(raw), rgb_sha256=digest(pixels))])))
        self.plan = dict(schema="gufo-vision-tensors-v1", scope="encoder",
                         image_set_sha256=digest(self.images.read_bytes()), tensors=[])
        for stage in DRIVER.STAGES:
            self.plan["tensors"].append(dict(id="tiny/" + stage, case_id="tiny", stage=stage,
                stratum="tiny/" + stage, head=-1, layer=int(stage[5:]) if stage.startswith("layer") else -1,
                width=2560 if stage == "embedding" else 1152,
                row_indices=[0] if stage == "embedding" else [0, 3], input_sha256=digest(pixels)))
        self.plan_path = self.root / "plan.json"
        self.write_plan()
        self.out = self.root / "packet"
        self.args = argparse.Namespace(binary=self.binary, model=self.model, images=self.images,
            plan=self.plan_path, case="tiny", image=self.image, revision="a" * 40,
            toolchain="fixture-python", out_dir=self.out, development=False)

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def backend(self, mode="ok"):
        code = f'''#!{sys.executable}
import os, pathlib, struct, sys, time
assert sys.argv[1] == '--encode' and len(sys.argv) == 8
assert sys.argv[4:6] == ['32', '32']
assert pathlib.Path(sys.argv[3]).stat().st_size == 32*32*3
assert os.environ['OMP_NUM_THREADS'] == os.environ['OPENBLAS_NUM_THREADS'] == '4'
assert os.environ['OMP_DYNAMIC'] == 'FALSE'
out = pathlib.Path(sys.argv[7])
assert out.is_dir() and not list(out.iterdir())
mode = {mode!r}
if mode == 'sleep':
    time.sleep(60)
if mode == 'exit':
    sys.exit(7)
if mode == 'missing':
    sys.exit(0)
if mode == 'drift':
    (out.parent.parent / 'math.hpp').write_text('changed during run')
print('progress' * 10000, flush=True)
print('diagnostic' * 10000, file=sys.stderr, flush=True)
for line in pathlib.Path(sys.argv[6]).read_text().splitlines():
    stage, count, *indices = line.split()
    assert int(count) == len(indices)
    width = 2560 if stage == 'embedding' else 1152
    value = float('nan') if mode == 'nan' else 0.25
    with (out / (stage + '.f32le')).open('xb') as stream:
        raw = struct.pack('<f', value) * width * int(count)
        stream.write(raw[:-4] if mode == 'short' else raw + (b'extra' if mode == 'extra' else b''))
'''
        self.binary.write_text(code)
        self.binary.chmod(0o700)

    def prepare(self):
        return DRIVER.prepare(self.args)

    def run_packet(self, packet_hash):
        with patch.object(DRIVER, "resources", return_value=GOOD):
            return DRIVER.run(self.out, packet_hash)

    def test_prepare_pins_and_success(self):
        # Preserve plan order in the capture; backend selection stays stage order.
        self.plan["tensors"].reverse()
        self.write_plan()
        packet_hash = self.prepare()
        self.assertEqual(packet_hash, digest((self.out / "packet.json").read_bytes()))
        header = json.loads((self.out / "header.json").read_bytes())
        self.assertEqual(header["rows"], 57)
        self.assertEqual(header["plan_sha256"], digest((self.out / "case-plan.json").read_bytes()))
        self.assertEqual(header["producer"]["accumulator_dtype"], "f32_qk_f64_softmax_pv")
        self.assertEqual(header["producer"]["arithmetic_plan"], dict(projections="f32", qk="f32",
            softmax="f64", pv="f64", output="f32", query_block=128, full_keys=True))
        self.assertEqual(header["producer"]["production_or_development"], "production")
        self.assertEqual([line.split()[0] for line in (self.out / "selection.txt").read_text().splitlines()], DRIVER.STAGES)
        result = self.run_packet(packet_hash)
        self.assertEqual(result["status"], "referencecaptured")
        self.assertFalse(result["kernelqualified"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["peak_rss"], GOOD["rss"])
        rows = [json.loads(line) for line in (self.out / "capture.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0], header)
        self.assertEqual(len(rows), 58)
        self.assertEqual(rows[1]["tensor_id"], "tiny/embedding")
        for row in rows[1:]:
            width = 2560 if row["tensor_id"].endswith("embedding") else 1152
            with (self.out / row["file"]).open("rb") as source:
                source.seek(row["offset"])
                self.assertEqual(row["sha256"], digest(source.read(width * 4)))
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)
        for path in self.out.rglob("*"):
            self.assertEqual(path.stat().st_mode & 0o077, 0)
        before = (self.out / "completion.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "existing run outputs"):
            self.run_packet(packet_hash)
        self.assertEqual((self.out / "completion.json").read_bytes(), before)

    def test_development_is_only_explicit_and_never_relabeled(self):
        self.args.model = self.dev
        with self.assertRaises(ValueError):
            self.prepare()
        self.args.development = True
        packet_hash = self.prepare()
        result = self.run_packet(packet_hash)
        self.assertEqual(result["production_or_development"], "NON_QUALIFYING")
        self.assertEqual(json.loads((self.out / "header.json").read_bytes())["producer"]["production_or_development"],
                         "NON_QUALIFYING")
        self.args.out_dir = self.root / "another"
        self.args.model = self.model
        with self.assertRaises(ValueError):
            self.prepare()

    def test_wrong_model_sha_size_and_missing_source(self):
        self.model.write_bytes(b"x" * self.model.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare()
        self.model.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "size"):
            self.prepare()
        self.assertFalse(self.out.exists())
        with self.assertRaises(FileNotFoundError):
            DRIVER.source_identity([self.root / "absent.cpp"])

    def test_plan_validation(self):
        original = deepcopy(self.plan)
        mutations = [lambda p: p.update(image_set_sha256="0" * 64),
                     lambda p: p["tensors"].pop(),
                     lambda p: p["tensors"][0].update(width=2560),
                     lambda p: p["tensors"][0].update(input_sha256="0" * 64),
                     lambda p: p["tensors"][0].update(stage="layer0"),
                     lambda p: p["tensors"][1].update(layer=26),
                     lambda p: p["tensors"][0].update(id=p["tensors"][1]["id"])]
        for indices in ([0, 0], [3, 0], [4], [-1], [True], [0.0], []):
            mutations.append(lambda p, indices=indices: p["tensors"][0].update(row_indices=indices))
        for mutate in mutations:
            self.plan = deepcopy(original)
            mutate(self.plan)
            self.write_plan()
            with self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.out.exists())

    def test_external_hash_and_all_prepared_identity_drift(self):
        packet_hash = self.prepare()
        with patch.object(DRIVER.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "externally frozen"):
                DRIVER.run(self.out, "0" * 64)
            spawn.assert_not_called()
        self.assertFalse((self.out / "run.lock").exists())
        targets = [self.model, self.binary, self.images, self.plan_path, self.image, *self.sources,
                   *[self.out / name for name in ("pixels.rgb8", "selection.txt", "case-plan.json", "header.json")]]
        packet = json.loads((self.out / "packet.json").read_bytes())
        for path in targets:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b"drift")
                with self.assertRaises(ValueError):
                    DRIVER.verify(packet, self.out)
                path.write_bytes(original)
        self.model.write_bytes(b"changed")
        with patch.object(DRIVER.subprocess, "Popen") as spawn:
            with self.assertRaises(ValueError):
                DRIVER.run(self.out, packet_hash)
            spawn.assert_not_called()
        self.assertTrue((self.out / "failure.json").exists())
        self.assertFalse((self.out / "completion.json").exists())

    def test_guard_prelaunch_and_running_abort(self):
        self.backend("sleep")
        for metric, value in (("available", DRIVER.MIN_AVAILABLE - 1), ("disk_free", DRIVER.MIN_DISK - 1),
                              ("rss", DRIVER.MAX_RSS + 1)):
            with self.subTest(metric=metric):
                self.args.out_dir = self.root / metric
                self.out = self.args.out_dir
                packet_hash = self.prepare()
                bad = {**GOOD, metric: value}
                with patch.object(DRIVER, "resources", side_effect=[GOOD, bad]):
                    with self.assertRaisesRegex(ValueError, "resource guardrail"):
                        DRIVER.run(self.out, packet_hash)
                receipt = json.loads((self.out / "resource.json").read_bytes())
                self.assertEqual(receipt["status"], "FAILED")
                self.assertTrue(receipt["partial"])
                self.assertIsNotNone(receipt["exit_code"])
                self.assertFalse((self.out / "completion.json").exists())
                self.assertTrue((self.out / "stdout.log").exists())
        self.args.out_dir = self.root / "prelaunch"
        self.out = self.args.out_dir
        packet_hash = self.prepare()
        with patch.object(DRIVER, "resources", return_value={**GOOD, "available": 0}), \
                patch.object(DRIVER.subprocess, "Popen") as spawn:
            with self.assertRaises(ValueError):
                DRIVER.run(self.out, packet_hash)
            spawn.assert_not_called()

    def test_bad_backend_outputs_exit_and_postrun_drift(self):
        for mode in ("missing", "nan", "short", "extra", "exit", "drift"):
            with self.subTest(mode=mode):
                self.backend(mode)
                self.args.out_dir = self.root / mode
                self.out = self.args.out_dir
                packet_hash = self.prepare()
                with self.assertRaises(ValueError):
                    self.run_packet(packet_hash)
                self.assertFalse((self.out / "completion.json").exists())
                self.assertTrue((self.out / "failure.json").exists())
                self.assertFalse((self.out / "capture.jsonl").exists())

    def test_interrupt_cleanup_and_kill_escalation(self):
        self.backend("sleep")
        packet_hash = self.prepare()
        real_sleep = DRIVER.time.sleep
        interrupted = False
        def interrupt_once(seconds):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            real_sleep(seconds)
        with patch.object(DRIVER, "resources", return_value=GOOD), \
                patch.object(DRIVER.time, "sleep", side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                DRIVER.run(self.out, packet_hash)
        receipt = json.loads((self.out / "failure.json").read_bytes())
        self.assertIsNotNone(receipt["exit_code"])
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("fixture", 5), -9]
        DRIVER.stop(child)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 2)

    def test_png_strict_decode(self):
        original = self.image.read_bytes()
        self.assertEqual(DRIVER.decode_png(self.image)[:2], (32, 32))
        corrupt = bytearray(original)
        corrupt[30] ^= 1
        bad = [bytes(corrupt), original[:-1], original + b"trailing", original[:33] + original[8:],
               png(depth=16)[0], png(color=6)[0], png(interlace=1)[0], png(filter_byte=1)[0],
               png(width=31)[0], png(extra=chunk(b"tEXt", b"x"))[0],
               png(compressed=zlib.compress(b"\0" * 4000))[0],
               png(compressed=zlib.compress(b"\0" * 3104) + b"extra")[0],
               png(compressed=zlib.compress(b"\0" * 3104)[:-1])[0]]
        for raw in bad:
            self.image.write_bytes(raw)
            with self.assertRaises((ValueError, zlib.error)):
                DRIVER.decode_png(self.image)
        huge_header = chunk(b"IHDR", struct.pack(">IIBBBBB", 8192, 8192, 8, 2, 0, 0, 0))
        self.image.write_bytes(original[:8] + huge_header + original[33:])
        with patch.object(DRIVER.zlib, "decompressobj") as decoder:
            with self.assertRaisesRegex(ValueError, "area budget"):
                DRIVER.decode_png(self.image)
            decoder.assert_not_called()
        self.image.write_bytes(original)
        manifest = json.loads(self.images.read_bytes())
        for key, value in (("width", 64), ("sha256", "0" * 64), ("rgb_sha256", "0" * 64)):
            changed = deepcopy(manifest)
            changed["cases"][0][key] = value
            self.images.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "manifest dimensions/PNG/RGB"):
                self.prepare()

    def test_resource_sampling_and_exact_thresholds(self):
        status = self.root / "proc-status"
        status.write_text("State:\tR (running)\nVmRSS:\t2048 kB\n")
        self.assertEqual(DRIVER.proc_kib(status, "VmRSS"), 2 * 1024**2)
        status.write_text("State:\tZ (zombie)\n")
        self.assertEqual(DRIVER.proc_kib(status, "VmRSS"), 0)
        with self.assertRaisesRegex(ValueError, "missing resource metric"):
            DRIVER.proc_kib(status, "MemAvailable")
        with patch.object(DRIVER, "proc_kib", side_effect=[1024, 2048, 4096]), \
                patch.object(DRIVER.shutil, "disk_usage", return_value=Mock(free=8192)):
            self.assertEqual(DRIVER.resources(self.root, 123), dict(rss=3072, available=4096, disk_free=8192))
        receipt = {}
        boundary = dict(rss=DRIVER.MAX_RSS, available=DRIVER.MIN_AVAILABLE, disk_free=DRIVER.MIN_DISK)
        with patch.object(DRIVER, "resources", return_value=boundary):
            DRIVER.guard(self.root, None, receipt)
        self.assertEqual(receipt["peak_rss"], DRIVER.MAX_RSS)
        self.assertEqual(receipt["lowest_available"], DRIVER.MIN_AVAILABLE)
        self.assertEqual(receipt["lowest_disk_free"], DRIVER.MIN_DISK)

    def test_bounded_json_and_explicit_source_hash(self):
        self.plan_path.write_bytes(b" " * 129)
        self.assertEqual(DRIVER.LIMIT, 64 * 1024 * 1024)
        with self.assertRaisesRegex(ValueError, "budget"):
            DRIVER.bounded(self.plan_path, 128)
        self.plan_path.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            DRIVER.load(self.plan_path)
        sources, composite = DRIVER.source_identity(self.sources)
        expected = {p.name: digest(p.read_bytes()) for p in self.sources}
        self.assertEqual(composite, digest((json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n").encode()))
        self.assertEqual(len(sources), 4)
        result = subprocess.run([sys.executable, "-B", "-S", str(SCRIPT), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare,run", result.stdout)


if __name__ == "__main__":
    unittest.main()
