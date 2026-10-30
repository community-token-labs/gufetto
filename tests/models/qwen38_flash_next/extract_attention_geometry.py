#!/usr/bin/env python3
"""Compile actual early create/dispatch guards without HIP or GPU execution."""

import argparse
from pathlib import Path


def prefix(source, signature, end):
    start = source.index(signature)
    start = source.index("{", start) + 1
    stop = source.index(end, start)
    result = source[start:stop]
    if "IsSupported" not in result or "return;" in result:
        raise ValueError("production geometry guard missing or silently returns")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root / "src/models/qwen38_flash_next/kernels/rocm"
    executor = (root / "executor.cpp").read_text()
    kernels = (root / "kernels.hip.cpp").read_text()
    create = prefix(executor, "std::unique_ptr<Executor> Executor::Create(",
                    "std::unique_ptr<Executor> e(")
    attention = prefix(kernels, "void Attention(", "if (!IsKnownKvCacheType")
    indexer = prefix(kernels, "void SelectBlocks(", "// Graph grids")
    args.output.write_text(
        "// Generated verbatim production early guards; tail proxies do not run kernels.\n"
        "void* CreateGuard(const ModelProxy& model, std::string* error_msg) {\n" + create +
        "  ++create_fallthrough; return &create_fallthrough;\n}\n"
        "void AttentionGuard(unsigned heads, unsigned kv_heads, unsigned d, unsigned ratio) {\n" +
        attention + "  ++attention_fallthrough;\n}\n"
        "void IndexerGuard(unsigned heads, unsigned dim, unsigned ratio) {\n" + indexer +
        "  ++indexer_fallthrough;\n}\n")


if __name__ == "__main__":
    main()
