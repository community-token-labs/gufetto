# Qwen 4 Max

**Not yet released / no measurements.** This is a readiness placeholder,
not a claim of Gufo support or confirmed architecture/weight availability.

Official metadata and lead approval of an exact validated profile are required
before enabling a new shape. The `qwen4exp` preview path is not evidence that
this release will load or produce correct output.

Start with the [tokenizer metadata diff](../../../tools/qwen-flash/tokenizer_diff.md),
then qualify the real config/tensors, rendered chat bytes, AR, any real MTP,
and continuation/cache behavior. Do not download weights as part of CPU checks.

See [quality](QUALITY.md) and [benchmarks](BENCHMARKS.md).

Every future number must name its stack: **host HIP/ROCr vs Nix ROCm 10.1**,
host/GPU, Gufo commit and binary hash, compiler/dependency versions, model
revision/quantization, command/flags, context, sampling/draft mode and cache
history. Do not combine evidence across stacks without an explicit comparison.
