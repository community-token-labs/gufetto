#!/usr/bin/env python3
"""Independent analytic/transport fixtures for the native offline teacher.

The sparse synthetic GGUF is not a model or a qualifying reference capture.
No production weights, GPU, or external Python dependencies are used.
"""
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = Path(sys.argv.pop(1)).resolve()
TOOLS = Path(__file__).resolve().parents[2] / "tools" / "quality"
sys.path.insert(0, str(TOOLS))
from vision_reference import attention_f32


def pack_string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def fixture_inventory():
    """Independent fixed model contract; no backend/parser imports."""
    result = []
    def linear(name, ins, outs):
        result.extend(((name + ".weight", [ins, outs], 30),
                       (name + ".bias", [outs], 0)))
    def norm(name):
        result.extend(((name + ".weight", [1152], 0),
                       (name + ".bias", [1152], 0)))
    for layer in range(27):
        prefix = f"v.blk.{layer}."
        linear(prefix + "attn_qkv", 1152, 3456)
        linear(prefix + "attn_out", 1152, 1152)
        linear(prefix + "ffn_up", 1152, 4304)
        linear(prefix + "ffn_down", 4304, 1152)
        norm(prefix + "ln1")
        norm(prefix + "ln2")
    linear("mm.0", 4608, 4608)
    linear("mm.2", 4608, 2560)
    norm("v.post_ln")
    result.extend((("v.patch_embd.bias", [1152], 0),
                   ("v.patch_embd.weight", [16, 16, 3, 1152], 0),
                   ("v.patch_embd.weight.1", [16, 16, 3, 1152], 0),
                   ("v.position_embd.weight", [1152, 2304], 0)))
    return result


def sparse_model(path):
    descriptors, offsets, cursor = bytearray(), {}, 0
    for name, shape, dtype in fixture_inventory():
        offsets[name] = cursor
        descriptors += pack_string(name) + struct.pack("<I", len(shape))
        descriptors += struct.pack("<" + "Q" * len(shape), *shape)
        descriptors += struct.pack("<IQ", dtype, cursor)
        cursor += math.prod(shape) * (4 if dtype == 0 else 2)
        cursor = (cursor + 31) // 32 * 32
    header = b"GGUF" + struct.pack("<IQQ", 3, 334, 0) + descriptors
    data_start = (len(header) + 31) // 32 * 32
    with path.open("xb") as out:
        out.write(header)
        out.truncate(data_start + cursor)  # Sparse holes, not a 0.9 GB allocation.
        def value(name, index, number):
            out.seek(data_start + offsets[name] + 4 * index)
            out.write(struct.pack("<f", number))
        # Different channels and temporal slices test C,T,Y,X reconstruction.
        value("v.patch_embd.weight", 256, 2.0)  # out0, green, temporal0
        value("v.patch_embd.weight.1", 768 + 512, 3.0)  # out1, blue, temporal1
        value("v.patch_embd.bias", 2, 0.25)
        # T,C,Y,X would place the last two coefficients in the wrong order.
        value("v.patch_embd.weight", 3 * 768, 1e20)
        value("v.patch_embd.weight.1", 3 * 768, 1.0)
        value("v.patch_embd.weight", 3 * 768 + 256, -1e20)
        value("mm.2.bias", 0, 0.75)
        value("mm.2.bias", 2559, -0.5)
    return data_start, offsets


class NativeReferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def invoke(self, *args, success=True):
        proc = subprocess.run([str(BINARY), *map(str, args)], capture_output=True,
                              text=True, timeout=45)
        if success:
            self.assertEqual(proc.returncode, 0, proc.stderr)
        else:
            self.assertNotEqual(proc.returncode, 0)
        return proc

    def test_attention_vs_independent_scalar_f32(self):
        # Agreement is numerical: native F64 softmax/PV intentionally differs
        # from the independent scalar oracle's all-F32 reduction boundaries.
        for n, d in ((1, 4), (8, 4), (9, 72), (32, 4), (33, 4), (129, 4)):
            with self.subTest(n=n, d=d):
                q = [[((i + j) % 5 - 2) / 8 for j in range(d)] for i in range(n)]
                k = [[((i * 3 + j) % 7 - 3) / 8 for j in range(d)] for i in range(n)]
                v = [[((i + j * 2) % 9 - 4) / 8 for j in range(d)] for i in range(n)]
                raw = struct.pack("<III", n, 1, d)
                for matrix in (q, k, v):
                    raw += struct.pack("<" + "f" * (n * d), *(x for row in matrix for x in row))
                inp, out = self.root / f"{n}.in", self.root / f"{n}.out"
                inp.write_bytes(raw)
                self.invoke("--attention", inp, out)
                got = struct.unpack("<" + "f" * (n * d), out.read_bytes())
                expected = attention_f32(q, k, v, max_work=6000000)
                for actual, want in zip(got, (x for row in expected for x in row)):
                    self.assertLessEqual(abs(actual - want), 1e-6)
                self.invoke("--attention", inp, out, success=False)
                self.assertEqual(struct.unpack("<" + "f" * (n * d), out.read_bytes()), got)

    def test_attention_invalid_transport(self):
        valid = struct.pack("<IIIfff", 1, 1, 1, 0, 0, 1)
        for i, raw in enumerate((valid[:-1], valid + b"x", b"", struct.pack("<III", 258, 1, 1),
                                 struct.pack("<IIIfff", 1, 1, 1, float("nan"), 0, 1))):
            inp, out = self.root / f"bad{i}", self.root / f"out{i}"
            inp.write_bytes(raw)
            self.invoke("--attention", inp, out, success=False)
            self.assertFalse(out.exists())

    def test_full_depth_sparse_analytic_encoder(self):
        model = self.root / "synthetic.gguf"
        data_start, offsets = sparse_model(model)
        image = self.root / "image.rgb"
        # White pixels: all normalized channels are exactly1.
        image.write_bytes(bytes([255, 255, 255]) * (32 * 32))
        selection = self.root / "selection.txt"
        stages = ["patch"] + [f"layer{i}" for i in range(27)] + ["embedding"]
        selection.write_text("".join(f"{s} 1 0\n" for s in stages))
        out = self.root / "capture"
        out.mkdir()
        proc = self.invoke("--encode", model, image, 32, 32, selection, out)
        self.assertEqual(len(proc.stdout.splitlines()), 29)
        self.assertEqual({p.name for p in out.iterdir()}, {s + ".f32le" for s in stages})
        expected = [2.0, 3.0, 0.25, 0.0] + [0.0] * 1148
        for stage in stages[:-1]:
            self.assertEqual(struct.unpack("<1152f", (out / (stage + ".f32le")).read_bytes()),
                             tuple(expected))
        embedding = struct.unpack("<2560f", (out / "embedding.f32le").read_bytes())
        self.assertEqual(embedding, (0.75,) + (0.0,) * 2558 + (-0.5,))
        self.invoke("--encode", model, image, 32, 32, selection, out, success=False)
        # A nonfinite actual tensor is rejected before the first capture.
        with model.open("r+b") as f:
            f.seek(data_start + offsets["v.patch_embd.bias"])
            f.write(struct.pack("<f", float("inf")))
        bad = self.root / "nonfinite"
        bad.mkdir()
        self.invoke("--encode", model, image, 32, 32, selection, bad, success=False)
        self.assertFalse(list(bad.iterdir()))


if __name__ == "__main__":
    unittest.main()
