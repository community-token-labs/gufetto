#!/usr/bin/env python3
"""CPU control regression using the real WeightType enum and verbatim HIP bodies.

Packed decoders and kernel launch are EXPLICIT STUBS: this checks exhaustive
switches, row geometry, scalar reads and rejection before launch, NOT native
HIP compilation/execution or packed-decoder numerical correctness.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


PRELUDE = r"""
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#define __device__
#define __forceinline__ inline
using __half = _Float16;
using hipStream_t = void*;
"""

STUBS = r"""
float __half2float(__half x) { return static_cast<float>(x); }
float Bf16ToF32(std::uint16_t x) {
  return std::bit_cast<float>(std::uint32_t{x} << 16);
}
unsigned packed_calls = 0, launches = 0;
const void* packed_row = nullptr;
WeightType packed_type{};
std::uint32_t packed_index = 0;
float PackedElement(const void* row, WeightType type, std::uint32_t index) {
  ++packed_calls;  // Explicit decoder DISPATCH stub; no packed math here.
  packed_row = row;
  packed_type = type;
  packed_index = index;
  return 1000.0F + static_cast<unsigned>(type);
}
constexpr unsigned kThreads = 256;
struct dim3 { unsigned value; explicit dim3(unsigned x) : value(x) {} };
template<class... T> void StubLaunch(T... args) {
  ((void)args, ...);
  ++launches;  // Explicit launch-count stub; no kernel/GPU work.
}
#define hipLaunchKernelGGL(kernel,grid,block,shared,stream,...) \
  StubLaunch(grid,block,shared,stream,__VA_ARGS__)
"""

CHECKS = r"""
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
  struct Valid { WeightType type; std::size_t bytes; };
  const Valid valid[] = {
    {WeightType::kF32, 1024}, {WeightType::kF16, 512},
    {WeightType::kBF16, 512}, {WeightType::kQ8_0, 272},
    {WeightType::kQ5_0, 176}, {WeightType::kQ4_K, 144},
    {WeightType::kQ6_K, 210}
  };
  const WeightType unsupported[] = {WeightType::kQ5_1, WeightType::kQ5_K};
  for (auto type : all_named) {
    unsigned covered = 0;
    for (auto item : valid) covered += item.type == type;
    for (auto rejected : unsupported) covered += rejected == type;
    Check(covered == 1, "named WeightType needs exactly one contract case");
  }
  alignas(4) std::uint8_t table[1024]{};
  const std::int32_t token = 0;
  float output[256]{};
  for (auto item : valid) {
    Check(RowBytes(item.type, 256) == item.bytes, "RowBytes supported type");
    Check(RowBytes(item.type, 512) == 2 * item.bytes, "RowBytes doubled width");
    const auto before = launches;
    EmbedTokens(table, item.type, &token, output, 1, 256, 1, nullptr);
    Check(launches == before + 1, "supported EmbedTokens must launch once");
  }
  const float f32[] = {1.25F, -2.5F};
  Check(RowElement(f32, WeightType::kF32, 1) == -2.5F, "F32 scalar read");
  const std::uint16_t bf16[] = {0x3f80, 0xc000};
  Check(RowElement(bf16, WeightType::kBF16, 1) == -2, "BF16 scalar read");
  const __half f16[] = {static_cast<__half>(2), static_cast<__half>(-3.5F)};
  Check(RowElement(f16, WeightType::kF16, 1) == -3.5F, "F16 scalar read");
  alignas(2) std::uint8_t q8[68]{};
  const __half scales[] = {static_cast<__half>(1), static_cast<__half>(0.5F)};
  std::memcpy(q8, &scales[0], 2);
  std::memcpy(q8 + 34, &scales[1], 2);
  q8[2] = 3;
  q8[36] = 0xfc;  // -4 at the first code in block two.
  Check(RowElement(q8, WeightType::kQ8_0, 0) == 3, "Q8 scalar read");
  Check(RowElement(q8, WeightType::kQ8_0, 32) == -2, "Q8 block/index read");
  for (auto type : {WeightType::kQ5_0, WeightType::kQ4_K, WeightType::kQ6_K}) {
    Check(RowElement(table, type, 17) == 1000.0F + static_cast<unsigned>(type),
          "packed decoder dispatch stub result");
    Check(packed_row == table && packed_type == type && packed_index == 17,
          "packed decoder dispatch arguments");
  }
  const auto before = launches;
  for (auto type : {WeightType::kQ5_1, WeightType::kQ5_K,
                   static_cast<WeightType>(UINT32_MAX)}) {
    Check(RowBytes(type, 256) == 0, "unsupported RowBytes must be zero");
    Check(std::isnan(RowElement(nullptr, type, 0)),
          "unsupported RowElement must be NaN, not a fabricated weight");
    bool rejected = false;
    try {
      EmbedTokens(nullptr, type, nullptr, nullptr, 1, 256, 1, nullptr);
    } catch (const std::invalid_argument& error) {
      rejected = std::string_view(error.what()) == "unsupported embedding weight type";
    }
    Check(rejected && launches == before, "unsupported type must reject before launch");
  }
  Check(packed_calls == 3, "unsupported type reached packed decoder stub");
}
"""


def function(source: str, marker: str) -> str:
    if source.count(marker) != 1:
        raise ValueError(f"expected one production function: {marker}")
    start = source.index(marker)
    body = source.index("{", start)
    depth = 0
    for end in range(body, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start:end + 1]
    raise ValueError(f"unterminated production function: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--rotation", choices=("ON", "OFF"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    kernels = args.source_root / "src/models/qwen38_flash_next/kernels/rocm"
    header = (kernels / "kernels.hpp").read_text()
    source = (kernels / "kernels.hip.cpp").read_text()
    enums = re.findall(r"enum class WeightType\s*:\s*std::uint32_t\s*\{[^}]+\};", header)
    if len(enums) != 1:
        raise ValueError("expected one production WeightType enum")
    names = re.findall(r"^\s*(k\w+)\s*(?==|,|$)", enums[0], re.MULTILINE)
    if not names:
        raise ValueError("no named production WeightType values")
    named = "const WeightType all_named[] = {" + ",".join(
        f"WeightType::{name}" for name in names
    ) + "};\n"
    bodies = "\n".join(function(source, marker) for marker in (
        "__device__ __forceinline__ float RowElement(",
        "__device__ __forceinline__ std::size_t RowBytes(",
        "void EmbedTokens(",
    ))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weight-dispatch-", dir=args.work_dir) as work:
        cpp = Path(work) / "contract.cpp"
        binary = Path(work) / "contract"
        cpp.write_text(PRELUDE + enums[0] + "\n" + named + STUBS + bodies + CHECKS)
        command = [args.cxx, "-std=c++20", "-O0", "-g0", "-Wall", "-Wextra",
                   "-Werror", "-Wswitch-enum"]
        if args.rotation == "ON":
            command.append("-DGUFO_EXPERIMENTAL_KV_Q8_ROTATION=1")
        subprocess.run(command + [str(cpp), "-o", str(binary)], check=True, timeout=30)
        subprocess.run([str(binary)], check=True, timeout=5)
    print(f"WeightType CPU contract PASS ({args.rotation}): {len(names)} named types, "
          "7 supported paths; unsupported/future NaN + zero geometry + prelaunch rejection. "
          "Packed decoders/GPU launch STUBBED; NOT native numerical proof.")


if __name__ == "__main__":
    main()
