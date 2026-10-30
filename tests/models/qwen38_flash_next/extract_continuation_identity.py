#!/usr/bin/env python3
"""Expose current host-only identity bodies to a CPU contract test.

Only the Model/executor inputs are proxies. No copied serializer, historical
hash normalization, HIP execution, native factory or restore proof. A source
shape change fails generation rather than silently testing a stale copy.
"""
import argparse
from pathlib import Path
import re


def between(source: str, start: str, end: str) -> str:
    if source.count(start) != 1:
        raise ValueError(f"expected one start marker: {start}")
    first = source.index(start)
    last = source.index(end, first)
    return source[first:last]


def body(source: str, signature: str) -> str:
    function = between(source, signature, "\n}\n")
    return function[function.index("{"):] + "\n}"


def generate(root: Path) -> str:
    backend = (root / "src/cli/serve/inference_backend.cpp").read_text()
    engine = (root / "src/models/qwen38_flash_next/engine.cpp").read_text()
    adapter = (root / "src/models/qwen38_flash_next/continuation_adapter.cpp").read_text()
    runner = backend[backend.index("class QwenFlashNextTextRunner final") :]
    # CPU source-wiring guard only: the native factory must install the SAME
    # production validator exercised by the real-container CPU tests. This
    # cannot prove native allocations, captures or restore behavior.
    factory = between(runner, "  CreateComponentCacheResources(cache::ResourceLedger& ledger,",
                      "\n  [[nodiscard]] std::unique_ptr<TextRunnerState>")
    compact = re.sub(r"\s+", "", factory)
    required = (
        "detail::InstallQfnPackedKvIngressValidation(*resources,model_->config(),"
        "use_mtp_,max_context_,model_->CacheType());",
        "std::make_unique<models::qwen38_flash_next::ContinuationAdapter>(model_,"
        "use_mtp_?core::SessionMode::kSpeculative:core::SessionMode::kAutoregressive,"
        "max_context_,component_identity_)",
    )
    if any(compact.count(call) != 1 for call in required):
        raise ValueError("native factory identity/validator wiring needs review")
    constants = []
    for name in ("kPrefillChunkTokens", "kPrefillTailTokens"):
        matches = re.findall(rf"constexpr std::uint32_t {name} = [^;]+;", engine)
        if len(matches) != 1:
            raise ValueError(f"expected one constant: {name}")
        constants += matches
    pieces = [
        "// Generated from current production bodies; do not edit.\n",
        "namespace gufo::server {\n",
        between(backend, "bool IsSha256Hex(", "\nstd::vector<std::uint8_t> DeepSeekCompatibilityIdentity"),
        between(backend, "std::vector<std::uint8_t> QwenFlashNextCompatibilityIdentity(", "\nstd::vector<std::uint8_t> QwenCompatibilityIdentity("),
        "}\nnamespace identity_probe {\n",
        "\n".join(constants),
        "\nstd::uint32_t ModelProxy::PrefillCapacity() const noexcept\n",
        body(engine, "std::uint32_t Model::PrefillCapacity() const noexcept"),
        "\nstd::uint32_t ModelProxy::PrefillThroughCapacity() const noexcept\n",
        body(engine, "std::uint32_t Model::PrefillThroughCapacity() const noexcept"),
        "\nstd::uint32_t Capacity(ModelProxy* m, std::uint32_t proposals) {\n"
        "  struct { std::uint32_t max_batch, max_logit_rows, max_speculative; } exec{};\n"
        "  struct { std::uint32_t max_draft_tokens; } options{proposals};\n",
        between(engine, "  exec.max_batch = m->PrefillThroughCapacity();", "\n#if"),
        "\n",
        # Identity binds ordinary speculative capacity S, before diagnostic
        # widening of the output head to L. The cold-score suite covers L.
        between(engine, "  exec.max_logit_rows =\n",
                "\n  const auto capacity = DiagnosticLogitCapacity::Plan("),
        "\n  return exec.max_speculative;\n}\n",
        "cache::Identity Process(ModelProxy* model_, std::uint32_t max_context_,\n"
        "                        bool use_mtp_, std::uint32_t max_draft_tokens_) {\n"
        "  cache::Identity component_identity_;\n",
        # The proxy's get() returns its own address, not a fixed fake identity.
        between(runner, "      std::ostringstream identity;", "\n    }\n  }"),
        "\n  return component_identity_;\n}\n",
        "cache::Identity Adapter(ModelProxy* model_, core::SessionMode mode,\n"
        "                        std::uint32_t context, cache::Identity identity_) {\n",
        between(adapter, "  const auto put =", "\n  for (const auto& component : layout_.Components())"),
        "\n  return identity_;\n}\n}\n",
    ]
    # model_ in Process is a pointer proxy, whereas production owns a shared_ptr.
    # Adapt only the ownership spelling, never serialization or arithmetic.
    result = "".join(pieces)
    if result.count("model_.get()") != 1:
        raise ValueError("process model ownership spelling changed")
    return result.replace("model_.get()", "model_->get()")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(args.source_root))


if __name__ == "__main__":
    main()
