# Qwen 4 Max benchmarks

**Not yet released / no measurements.** This is a readiness placeholder,
not a claim of Gufo support or confirmed architecture/weight availability.

No speed, latency, memory, or MTP acceptance measurements exist for this
release. Retain the standard speed benchmark and per-request/phase timing;
compare matched stacks/settings/cache histories, never averaged-away regressions.

Every future number must name its stack: **host HIP/ROCr vs Nix ROCm 10.1**,
host/GPU, Gufo commit and binary hash, compiler/dependency versions, model
revision/quantization, command/flags, context, sampling/draft mode and cache
history. Do not combine evidence across stacks without an explicit comparison.
