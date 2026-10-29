"""Host-test slices of the real Q8 scratch allocator and sidecar writer."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kernels = args.source_root / "src/models/qwen/hip/kernels"
    header = (kernels / "prefill_quant_gemm.hpp").read_text()
    source = (kernels / "prefill_quant_gemm.hip").read_text()
    layout = header[header.index("constexpr std::size_t kQ8ActTileTokens") :
                    header.index("template<int WaveSize>")]
    sizing = source[source.index("std::size_t QuantizedActivationBytes(") :]
    sizing = sizing[:sizing.index("\n}") + 2]
    allocator = source[source.index("void LaunchBatchedQuantGEMM(") :]
    allocator = allocator[:allocator.index("  constexpr std::size_t kWarp")] + "}\n"
    quantizer = source[source.index("void LaunchQuantizeActivationQ8_1(") :]
    quantizer = quantizer[:quantizer.index("\n}")]
    assert "QuantizeActivationToQ8_1Kernel<true>" in quantizer
    assert "LaunchZeroQ8ActTail<true>" in quantizer
    args.output.write_text(layout + "\n" + sizing + "\n" + allocator)


if __name__ == "__main__":
    main()
