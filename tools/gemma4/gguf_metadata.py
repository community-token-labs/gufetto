"""Minimal GGUF v2/v3 metadata reader: parses the key/value section only and
never reads tensor descriptors or tensor data."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
            7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING = 8
_ARRAY = 9


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return struct.unpack(fmt, data)[0]


def _string(f: BinaryIO) -> bytes:
    n = _read(f, "<Q")
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF string")
    return data


def _value(f: BinaryIO, kind: int) -> Any:
    if kind in _SCALARS:
        return _read(f, _SCALARS[kind])
    if kind == _STRING:
        return _string(f)
    if kind == _ARRAY:
        item = _read(f, "<I")
        n = _read(f, "<Q")
        if item in _SCALARS:
            fmt = _SCALARS[item]
            size = struct.calcsize(fmt)
            data = f.read(size * n)
            if len(data) != size * n:
                raise ValueError("truncated GGUF array")
            return list(struct.unpack("<" + fmt[1:] * n, data))
        return [_value(f, item) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {kind}")


def read_metadata(path: Path) -> tuple[dict[str, Any], int]:
    """Returns the metadata (strings as raw bytes) and the bytes consumed."""
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(f, "<I")
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        _read(f, "<Q")  # tensor count
        count = _read(f, "<Q")
        metadata = {}
        for _ in range(count):
            key = _string(f).decode("utf-8")
            metadata[key] = _value(f, _read(f, "<I"))
        return metadata, f.tell()
