#!/usr/bin/env python3
"""Generate a private, wholly synthetic vision prompt corpus using only stdlib.

Usage: python3 tools/quality/vision_images.py --out-dir NEW_DIRECTORY
The parent directory must exist. Existing paths (including symlinks) are refused.
Failures may leave a partial private directory; it is never silently reused.
RGB hashes cover packed, top-to-bottom RGB8 pixels, without PNG filter bytes.
PNG encoding is deterministic for a fixed Python/zlib toolchain; RGB content is
independent of compression. No desktop capture, external assets, or fonts occur.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import zlib


# Original, deliberately small 5x7 bitmap alphabet, not an external font asset.
_GLYPHS = {
    " ": "00000/00000/00000/00000/00000/00000/00000",
    "A": "01110/10001/10001/11111/10001/10001/10001",
    "B": "11110/10001/10001/11110/10001/10001/11110",
    "C": "01111/10000/10000/10000/10000/10000/01111",
    "D": "11110/10001/10001/10001/10001/10001/11110",
    "E": "11111/10000/10000/11110/10000/10000/11111",
    "F": "11111/10000/10000/11110/10000/10000/10000",
    "G": "01111/10000/10000/10111/10001/10001/01111",
    "H": "10001/10001/10001/11111/10001/10001/10001",
    "I": "11111/00100/00100/00100/00100/00100/11111",
    "J": "00111/00010/00010/00010/10010/10010/01100",
    "K": "10001/10010/10100/11000/10100/10010/10001",
    "L": "10000/10000/10000/10000/10000/10000/11111",
    "M": "10001/11011/10101/10101/10001/10001/10001",
    "N": "10001/11001/10101/10011/10001/10001/10001",
    "O": "01110/10001/10001/10001/10001/10001/01110",
    "P": "11110/10001/10001/11110/10000/10000/10000",
    "Q": "01110/10001/10001/10001/10101/10010/01101",
    "R": "11110/10001/10001/11110/10100/10010/10001",
    "S": "01111/10000/10000/01110/00001/00001/11110",
    "T": "11111/00100/00100/00100/00100/00100/00100",
    "U": "10001/10001/10001/10001/10001/10001/01110",
    "V": "10001/10001/10001/10001/10001/01010/00100",
    "W": "10001/10001/10001/10101/10101/10101/01010",
    "X": "10001/10001/01010/00100/01010/10001/10001",
    "Y": "10001/10001/01010/00100/00100/00100/00100",
    "Z": "11111/00001/00010/00100/01000/10000/11111",
    "0": "01110/10001/10011/10101/11001/10001/01110",
    "1": "00100/01100/00100/00100/00100/00100/01110",
    "2": "01110/10001/00001/00010/00100/01000/11111",
    "3": "11110/00001/00001/01110/00001/00001/11110",
    "4": "00010/00110/01010/10010/11111/00010/00010",
    "5": "11111/10000/10000/11110/00001/00001/11110",
    "6": "01110/10000/10000/11110/10001/10001/01110",
    "7": "11111/00001/00010/00100/01000/01000/01000",
    "8": "01110/10001/10001/01110/10001/10001/01110",
    "9": "01110/10001/10001/01111/00001/00001/01110",
    ".": "00000/00000/00000/00000/00000/00110/00110",
    ":": "00000/00100/00100/00000/00100/00100/00000",
    "-": "00000/00000/00000/11111/00000/00000/00000",
    "_": "00000/00000/00000/00000/00000/00000/11111",
    "=": "00000/00000/11111/00000/11111/00000/00000",
    "+": "00000/00100/00100/11111/00100/00100/00000",
    ">": "10000/01000/00100/00010/00100/01000/10000",
    "/": "00001/00010/00010/00100/01000/01000/10000",
    "(": "00010/00100/01000/01000/01000/00100/00010",
    ")": "01000/00100/00010/00010/00010/00100/01000",
    "[": "01110/01000/01000/01000/01000/01000/01110",
    "]": "01110/00010/00010/00010/00010/00010/01110",
    ",": "00000/00000/00000/00000/00110/00100/01000",
    "%": "11001/11010/00010/00100/01000/01011/10011",
}

INK = (222, 232, 244)
MUTED = (139, 159, 184)
DARK = (20, 27, 40)
PANEL = (31, 43, 61)
RED = (226, 70, 78)
GREEN = (56, 184, 128)
BLUE = (63, 132, 232)
YELLOW = (246, 197, 66)

# IDs, dimensions and prompts are frozen corpus inputs, not generated captions.
CASES = (
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


class _Canvas:
    def __init__(self, width, height, background):
        self.width = width
        self.height = height
        self.rgb = bytearray(background) * (width * height)

    def rect(self, x, y, width, height, color):
        if not (0 <= x <= x + width <= self.width
                and 0 <= y <= y + height <= self.height):
            raise ValueError("rectangle outside canvas")
        row = bytes(color) * width
        stride = self.width * 3
        for offset in range(y * stride + x * 3,
                            (y + height) * stride + x * 3, stride):
            self.rgb[offset:offset + len(row)] = row

    def circle(self, cx, cy, radius, color):
        for dy in range(-radius, radius + 1):
            half = math.isqrt(radius * radius - dy * dy)
            self.rect(cx - half, cy + dy, 2 * half + 1, 1, color)

    def triangle(self, x, y, size, color):
        for dy in range(size):
            half = dy // 2
            self.rect(x + size // 2 - half, y + dy, 2 * half + 1, 1, color)

    def text(self, x, y, value, color=INK, scale=2):
        for index, char in enumerate(value):
            for row, bits in enumerate(_GLYPHS[char].split("/")):
                # Only the tiny glyph mask is traversed pixel by pixel.
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self.rect(x + (index * 6 + column) * scale,
                                  y + row * scale, scale, scale, color)


def _scene(width, height):
    canvas = _Canvas(width, height, (239, 243, 249))
    s = width // 1024

    def box(x, y, w, h, color):
        canvas.rect(x * s, y * s, w * s, h * s, color)

    def label(x, y, value):
        canvas.text(x * s, y * s, value, DARK, 3 * s)

    if width == 1024:
        label(70, 70, "SHAPE GARDEN")
        canvas.circle(200, 420, 100, RED)
        box(410, 320, 200, 200, BLUE)
        canvas.triangle(710, 320, 200, GREEN)
        label(145, 580, "APPLE")
        label(465, 580, "RIVER")
        label(765, 580, "LEAF")
        box(70, 740, 884, 12, YELLOW)
    elif width == 2048:
        label(70, 70, "PAPER SKYLINE")
        canvas.circle(820 * s, 240 * s, 70 * s, YELLOW)
        for x, top, color in ((110, 480, RED), (330, 350, BLUE),
                              (550, 420, GREEN)):
            box(x, top, 160, 850 - top, color)
            for row in range(top + 40, 800, 80):
                for column in (x + 30, x + 100):
                    box(column, row, 30, 35, INK)
        box(60, 850, 904, 24, DARK)
    else:
        label(70, 40, "QUADRANT BOARD")
        for x, y, name in ((60, 120, "A"), (540, 120, "B"),
                            (60, 560, "C"), (540, 560, "D")):
            box(x, y, 420, 380, (216, 224, 237))
            label(x + 20, y + 20, name)
        canvas.circle(270 * s, 330 * s, 75 * s, RED)
        for x in (610, 790):
            box(x, 270, 100, 100, BLUE)
        for x in (100, 230, 360):
            canvas.triangle(x * s, 740 * s, 90 * s, GREEN)
        for x in (660, 840):
            for y in (710, 840):
                canvas.circle(x * s, y * s, 40 * s, YELLOW)
    return canvas


def _editor(width, height):
    c = _Canvas(width, height, DARK)
    c.rect(0, 0, width, 56, PANEL)
    c.text(24, 18, "TILE LAB - SYNTHETIC EDITOR")
    c.rect(0, 56, 250, 680, PANEL)
    c.text(20, 92, "FILES", MUTED)
    c.text(20, 140, "> GARDEN.TOY", GREEN)
    c.text(20, 180, "  CHECKS.TOY", MUTED)
    c.text(286, 82, "GARDEN.TOY", YELLOW)
    lines = (
        "FUNCTION COUNT_TILES(ROWS, COLS)",
        "  LET TOTAL = ROWS TIMES COLS",
        "  RETURN TOTAL",
        "END",
        "",
        "LET RESULT = COUNT_TILES(6, 7)",
        "PRINT RESULT",
    )
    for index, line in enumerate(lines):
        c.text(278, 140 + index * 42, str(index + 1), MUTED)
        c.text(326, 140 + index * 42, line, GREEN if index < 4 else INK)
    c.rect(250, 510, 1030, 226, PANEL)
    c.text(286, 542, "OUTPUT")
    c.text(286, 590, "RESULT: 42", YELLOW)
    c.text(286, 636, "TESTS: 3 PASSED / 0 FAILED", GREEN)
    c.rect(0, 736, width, 32, BLUE)
    c.text(20, 744, "TOY LANGUAGE    LINE 7    SYNTHETIC FIXTURE")
    return c


def _terminal(width, height):
    c = _Canvas(width, height, DARK)
    c.rect(0, 0, width, 76, PANEL)
    c.text(36, 24, "TILE LAB - SYNTHETIC TERMINAL", scale=3)
    lines = (
        ("LAB> TILECHECK --GRID 6X7 --MODE DEMO", INK),
        ("BUILD: PAPER-17", MUTED),
        ("INPUT: INVENTED COLOR TILES", MUTED),
        ("[1/3] PREPARE GRID ... OK", BLUE),
        ("[2/3] CHECK EDGES ... OK", BLUE),
        ("[3/3] WRITE REPORT ... OK", BLUE),
        ("TILES: 42    PASSED: 40    FAILED: 2", YELLOW),
        ("FAILED CELLS: B3, D5", RED),
        ("OUTPUT: DEMO-REPORT.TXT", GREEN),
        ("LAB> _", INK),
    )
    for index, (line, color) in enumerate(lines):
        c.text(48, 130 + index * 78, line, color, 4)
    return c


def _dashboard(width, height):
    c = _Canvas(width, height, (239, 243, 249))
    c.rect(0, 0, width, 88, DARK)
    c.text(32, 28, "PAPER FACTORY - SYNTHETIC DASHBOARD", scale=3)
    for x, name, value, color in (
            (48, "TILES", "240", BLUE),
            (544, "PASS RATE", "95%", GREEN),
            (1040, "QUEUED", "12", YELLOW)):
        c.rect(x, 128, 448, 180, PANEL)
        c.text(x + 24, 156, name, MUTED, 3)
        c.text(x + 24, 216, value, color, 5)
    c.text(48, 356, "OUTPUT BY LINE", DARK, 3)
    for x, label, value, color in (
            (100, "ALPHA", 180, BLUE), (330, "BETA", 300, GREEN),
            (560, "GAMMA", 240, RED)):
        c.rect(x, 760 - value, 140, value, color)
        c.text(x, 786, label, DARK, 3)
    c.rect(880, 360, 608, 484, PANEL)
    c.text(912, 392, "BATCH   ITEMS   STATUS", MUTED, 3)
    for y, line, color in (
            (472, "A       80      READY", GREEN),
            (564, "B       96      READY", GREEN),
            (656, "C       64      PAUSED", YELLOW)):
        c.text(912, y, line, color, 3)
    c.text(48, 940, "DEMO DATA ONLY - NO LIVE SERVICES", DARK, 3)
    return c


def _chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))


def _private_file(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _write_png(path, canvas):
    """Encode RGB8 rows with filter 0; never loop over full-resolution pixels."""
    digest = hashlib.sha256()
    with _private_file(path) as output:
        def emit(data):
            output.write(data)
            digest.update(data)

        emit(b"\x89PNG\r\n\x1a\n")
        emit(_chunk(b"IHDR", struct.pack(">IIBBBBB", canvas.width,
                                       canvas.height, 8, 2, 0, 0, 0)))
        compressor = zlib.compressobj(level=6)
        stride = canvas.width * 3
        pixels = memoryview(canvas.rgb)
        for offset in range(0, len(pixels), stride):
            data = compressor.compress(b"\0")
            data += compressor.compress(pixels[offset:offset + stride])
            if data:
                emit(_chunk(b"IDAT", data))
        emit(_chunk(b"IDAT", compressor.flush()))
        emit(_chunk(b"IEND", b""))
    return digest.hexdigest(), hashlib.sha256(canvas.rgb).hexdigest()


def generate(out_dir: Path):
    """Create six PNGs and manifest.json in a new private (0700) directory.

    Return the JSON-compatible manifest. Refuse every existing output path.
    Provenance hashes this generator's source, not machine-specific paths.
    """
    out_dir = Path(out_dir)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_dir.mkdir(mode=0o700)
    # Also produce exactly 0700 under a restrictive caller umask on POSIX.
    out_dir.chmod(0o700)
    manifest = {"schema": "gufo-vision-images-v1", "cases": []}
    renderers = (_scene, _scene, _scene, _editor, _terminal, _dashboard)
    for (case_id, width, height, prompt), render in zip(CASES, renderers):
        filename = case_id + ".png"
        canvas = render(width, height)
        png_hash, rgb_hash = _write_png(out_dir / filename, canvas)
        del canvas
        manifest["cases"].append({
            "id": case_id, "file": filename, "width": width, "height": height,
            "sha256": png_hash, "rgb_sha256": rgb_hash, "prompt": prompt,
            "provenance": {
                "kind": "synthetic", "personal_content": False,
                "source": "tools/quality/vision_images.py",
                "source_sha256": source_hash,
                "description": "Invented geometric scenes and UI content; "
                               "original bitmap glyphs; no captures or external assets.",
            },
        })
    with _private_file(out_dir / "manifest.json") as output:
        output.write((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="new private directory (parent must already exist)")
    args = parser.parse_args(argv)
    try:
        generate(args.out_dir)
    except OSError as error:
        parser.exit(1, f"vision_images: {error}\n")
    print(args.out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
