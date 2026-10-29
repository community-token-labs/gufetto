#!/usr/bin/env python3
"""Bounded stdlib-only contract tests; no GPU, models, or image dependencies."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "quality" / "vision_images.py"
SPEC = importlib.util.spec_from_file_location("vision_images", SCRIPT)
vision_images = importlib.util.module_from_spec(SPEC)
# Do not create generated files in the source tree when loading the utility.
with patch.object(sys, "dont_write_bytecode", True):
    SPEC.loader.exec_module(vision_images)


# Independent frozen contract, not derived from the generator's CASES constant.
EXPECTED = (
    ("shapes-1024", 1024, 1024,
     "Name the three large shapes from left to right, including their colors. "
     "What word appears below each shape?"),
    ("skyline-2048", 2048, 2048,
     "How many buildings are shown? List their colors from left to right. "
     "Where is the yellow sun relative to the buildings?"),
    ("board-4096", 4096, 4096,
     "Describe the four labeled quadrants. For each label, give the shape, "
     "color, and number of shapes inside it."),
    ("editor-1280", 1280, 768,
     "Read the active file name and summarize the visible pseudocode. "
     "What result and test status appear in the lower panel?"),
    ("terminal-1920", 1920, 1088,
     "Transcribe the command after the first prompt. How many tiles passed, "
     "how many failed, and what output file is reported?"),
    ("dashboard-1536", 1536, 1024,
     "Read the three summary metrics and their values. Which labeled bar "
     "is tallest, and what is the status of batch C?"),
)


class VisionImagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="gufo-vision-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "api"
        cls.manifest = vision_images.generate(cls.output)
        # A second independent process verifies CLI and byte-for-byte determinism.
        cls.repeat = cls.root / "cli"
        cls.cli = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--out-dir", str(cls.repeat)],
            capture_output=True, text=True, timeout=30, check=True,
        )

    def test_frozen_cases_and_provenance(self):
        manifest = self.manifest
        self.assertEqual(manifest["schema"], "gufo-vision-images-v1")
        self.assertEqual(len(manifest["cases"]), 6)
        source_hash = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        for case, (case_id, width, height, prompt) in zip(manifest["cases"], EXPECTED):
            with self.subTest(case=case_id):
                self.assertEqual(set(case), {
                    "id", "file", "width", "height", "sha256", "rgb_sha256",
                    "prompt", "provenance",
                })
                self.assertEqual(case["id"], case_id)
                self.assertEqual(case["file"], case_id + ".png")
                self.assertEqual((case["width"], case["height"]), (width, height))
                self.assertEqual((width % 32, height % 32), (0, 0))
                self.assertEqual(case["prompt"], prompt)
                provenance = case["provenance"]
                self.assertEqual(provenance["kind"], "synthetic")
                self.assertIs(provenance["personal_content"], False)
                self.assertEqual(provenance["source"], "tools/quality/vision_images.py")
                self.assertEqual(provenance["source_sha256"], source_hash)
                self.assertIn("no captures or external assets", provenance["description"])
                self.assertRegex(case["sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(case["rgb_sha256"], r"^[0-9a-f]{64}$")
        encoded = (self.output / "manifest.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(encoded), manifest)
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn(str(Path.home()), encoded)
        self.assertNotIn("timestamp", encoded)
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {case[0] + ".png" for case in EXPECTED} | {"manifest.json"})

    def test_png_crc_dimensions_and_rgb_hash(self):
        for case in self.manifest["cases"]:
            with self.subTest(case=case["id"]):
                data = (self.output / case["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), case["sha256"])
                self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
                offset = 8
                chunks = []
                compressed = bytearray()
                while offset < len(data):
                    self.assertGreaterEqual(len(data) - offset, 12)
                    size = struct.unpack_from(">I", data, offset)[0]
                    end = offset + 12 + size
                    self.assertLessEqual(end, len(data))
                    kind = data[offset + 4:offset + 8]
                    payload = data[offset + 8:end - 4]
                    crc = struct.unpack_from(">I", data, end - 4)[0]
                    self.assertEqual(crc, zlib.crc32(kind + payload) & 0xffffffff)
                    chunks.append(kind)
                    if kind == b"IHDR":
                        self.assertEqual(len(chunks), 1)
                        self.assertEqual(struct.unpack(">IIBBBBB", payload),
                                         (case["width"], case["height"], 8, 2, 0, 0, 0))
                    elif kind == b"IDAT":
                        compressed.extend(payload)
                    elif kind == b"IEND":
                        self.assertEqual(payload, b"")
                        self.assertEqual(end, len(data))
                    else:
                        self.fail(f"Unexpected PNG metadata chunk: {kind!r}")
                    offset = end
                self.assertEqual(chunks[0], b"IHDR")
                self.assertEqual(chunks[-1], b"IEND")
                self.assertTrue(chunks[1:-1])
                self.assertEqual(set(chunks[1:-1]), {b"IDAT"})
                decoder = zlib.decompressobj()
                raw = decoder.decompress(compressed) + decoder.flush()
                self.assertTrue(decoder.eof)
                self.assertEqual(decoder.unused_data, b"")
                stride = case["width"] * 3 + 1
                self.assertEqual(len(raw), stride * case["height"])
                digest = hashlib.sha256()
                view = memoryview(raw)
                for row in range(case["height"]):
                    start = row * stride
                    self.assertEqual(raw[start], 0, "only filter 0 is emitted")
                    digest.update(view[start + 1:start + stride])
                self.assertEqual(digest.hexdigest(), case["rgb_sha256"])
                del view, raw

    def test_determinism_and_cli(self):
        self.assertEqual(self.cli.stdout.strip(), str(self.repeat / "manifest.json"))
        self.assertEqual(self.cli.stderr, "")
        for path in self.output.iterdir():
            with self.subTest(file=path.name):
                self.assertEqual(path.read_bytes(), (self.repeat / path.name).read_bytes())

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_private_permissions_even_with_permissive_umask(self):
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        for path in self.output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0)
        # Exercise permission creation without another full-resolution render.
        for mask in (0, 0o777):
            out_dir = self.root / f"umask-{mask}"
            previous = os.umask(mask)
            try:
                with patch.object(vision_images, "CASES", ()):
                    vision_images.generate(out_dir)
            finally:
                os.umask(previous)
            self.assertEqual(stat.S_IMODE(out_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((out_dir / "manifest.json").stat().st_mode)
                             & 0o077, 0)

    def test_no_overwrite_directory_file_or_symlink(self):
        original = (self.output / "manifest.json").read_bytes()
        existing_file = self.root / "existing-file"
        existing_file.write_bytes(b"keep me")
        paths = [self.output, existing_file]
        if os.name == "posix":
            link = self.root / "linked-dir"
            link.symlink_to(self.output, target_is_directory=True)
            dangling = self.root / "dangling"
            dangling.symlink_to(self.root / "absent")
            paths.extend((link, dangling))
        for path in paths:
            with self.subTest(path=path.name):
                with self.assertRaises(FileExistsError):
                    vision_images.generate(path)
        self.assertEqual(existing_file.read_bytes(), b"keep me")
        self.assertEqual((self.output / "manifest.json").read_bytes(), original)
        cli = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--out-dir", str(self.output)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(cli.returncode, 0)
        self.assertIn("vision_images:", cli.stderr)
        self.assertEqual((self.output / "manifest.json").read_bytes(), original)

    def test_parent_must_exist(self):
        with self.assertRaises(FileNotFoundError):
            vision_images.generate(self.root / "missing-parent" / "images")
        self.assertFalse((self.root / "missing-parent").exists())


if __name__ == "__main__":
    unittest.main()
