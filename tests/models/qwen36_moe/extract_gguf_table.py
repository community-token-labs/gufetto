#!/usr/bin/env python3
"""Write a tensor-free table of a GGUF artifact for loader fixtures.

Reads only the GGUF header: architecture metadata (tokenizer arrays and other
large values are skipped) and every tensor's name, type and dimensions. No
tensor payload is read, so a 22 GB artifact costs a few megabytes of I/O.

Output lines are tab-separated:
  kv  <key>  <gguf value type>  <value; arrays comma-separated>
  tensor  <name>  <ggml type id>  <dims comma-separated>
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
MAX_ARRAY = 64


def read(f, fmt):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def read_string(f):
    return f.read(read(f, "<Q")).decode("utf-8")


def read_value(f, kind):
    """Returns the value, or None when it is too large to keep."""
    if kind == STRING:
        return read_string(f)
    if kind != ARRAY:
        return read(f, SCALARS[kind])
    item, count = read(f, "<I"), read(f, "<Q")
    if item == STRING:
        values = [read_string(f) for _ in range(count)]
        return None if count > MAX_ARRAY else values
    fmt = SCALARS[item]
    data = f.read(struct.calcsize(fmt) * count)
    if count > MAX_ARRAY:
        return None
    return list(struct.unpack("<%d%s" % (count, fmt[1]), data))


def keep(key):
    return not key.startswith(("tokenizer.", "quantize.")) and \
        key != "general.license.link"


def format_value(value):
    if isinstance(value, list):
        return ",".join(format_value(v) for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value).replace("\t", " ").replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--sha256", help="artifact digest to record")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lines = ["# Tensor-free GGUF table written by extract_gguf_table.py",
             "# source: " + args.gguf.name]
    if args.sha256:
        lines.append("# sha256: " + args.sha256)
    with args.gguf.open("rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit("not a GGUF file")
        version = read(f, "<I")
        n_tensors, n_kv = read(f, "<Q"), read(f, "<Q")
        lines.append("# version: %d tensors: %d" % (version, n_tensors))
        for _ in range(n_kv):
            key = read_string(f)
            kind = read(f, "<I")
            value = read_value(f, kind)
            if value is not None and keep(key):
                lines.append("kv\t%s\t%s\t%s" % (key, kind, format_value(value)))
        for _ in range(n_tensors):
            name = read_string(f)
            dims = [read(f, "<Q") for _ in range(read(f, "<I"))]
            ggml_type = read(f, "<I")
            read(f, "<Q")  # payload offset
            lines.append("tensor\t%s\t%d\t%s" % (
                name, ggml_type, ",".join(str(d) for d in dims)))
    args.output.write_text("\n".join(lines) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print("%s: %d lines, sha256 %s" % (args.output, len(lines), digest))


if __name__ == "__main__":
    main()
