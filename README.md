# Gufetto

Gufetto ("owlet") is a community fork of [Gufo](https://github.com/gufo-org/gufo),
the local LLM inference engine for AMD Strix Halo (`gfx1151`), published by
[Community Token Labs](https://github.com/community-token-labs). It was built
largely by AI coding agents working against measured correctness and speed
gates on two Strix Halo machines. Upstream supplies the engine and the
model/API foundation; [FORK.md](FORK.md) lists what this fork adds, and
[docs/COMMUNITY-HANDOFF.md](docs/COMMUNITY-HANDOFF.md) lists unfinished work
that anyone is welcome to pick up.

Gufetto is not an official Gufo release and is not endorsed by the Gufo
maintainers. It keeps upstream's MIT [license](LICENSE) and
[notices](NOTICE). Measurements in these docs were taken on our own hardware
and toolchain (mostly ROCm/HIP 7.2.4); treat them as evidence for those
conditions, not guarantees.

## Contributing

Gufetto is meant to be carried forward by the community: people **and** AI
agents. Pick an item from [docs/COMMUNITY-HANDOFF.md](docs/COMMUNITY-HANDOFF.md)
or bring your own model, fix or kernel, open an issue or discussion, and send a
pull request. Agent-written changes are welcome on the same terms as human
ones: show the CPU checks, and for GPU or speed changes, a native Strix Halo
run (see [docs/TESTING.md](docs/TESTING.md)).

## What Gufetto adds to Gufo

Upstream base: Gufo v0.26.0 + #849 (`81ff13c3`). Details and limits are in
[FORK.md](FORK.md) and the linked docs.

**KV cache and memory**
- Experimental q8 KV storage for Qwen3.8 Flash-Next (plain and Hadamard-rotated), plus opt-in rotated q4 and K=q8/V=q4 modes. F16 stays the default.
- Startup memory measurement that refuses early, with a clear reason, instead of failing mid-load; a verified-to-fit context size.
- Faster validation when restoring q4/q8 KV.

**Prompt cache**
- Opt-in persistent SSD cache that survives restarts, with bounded, fsync'd writes.
- Persistent pinned prefixes (e.g. long system prompts) and predicted system-prompt checkpoints in RAM.
- RAM eviction ranked by measured recovery cost; exact sharing of in-progress prefixes.
- A read-only cache lookup API and portable checkpoint export/import.
- Cache miss reasons, exact cache-token traces and capture-refusal reporting.

**Speed**
- Exact q8 prefill, decode and MTP-verification optimizations.
- Query-local sparse attention in prefill.

**Server and API**
- `x-api-key` support, API key from a file, and a CORS allowlist for LAN use.
- llama.cpp-compatible token-ID prompts on `/completion`.
- A llama.cpp-compatible rerank API (endpoint and contract; no model runtime yet).
- `/metrics` with cache tiers, miss reasons and speed histograms.
- Opt-in serving and phase traces for performance diagnosis.

**Vision**
- A shared image pool, reclaimable image memory claims and Flash-Next vision defaults.
- An independent F32/F64 vision reference and a numerics harness for vision attention.

**Diagnostics and safety**
- `gufo diagnose` measures whether a model fits on the GPU through the real admission path.
- Fails loudly on unsupported Flash-Next attention geometry and on fatal device-to-host transfer errors.
- GGUF identity checks that probe file content instead of trusting file metadata.
- A CPU Q5_1 dequantization fix (it previously aborted).

**Tools and docs**
- Tested client recipes for Claude Code, OpenCode, Cline, Aider, Open WebUI and Pi ([docs/clients](docs/clients/README.md)), plus benchmark cards.
- Offline prompt-scoring and matched-KL quality tools (adapted from upstream #697).
- A Qwen 4 day-0 emitted-token comparator and offline tokenizer checks.

**Experimental, off by default**
- XDNA2 NPU projection offload.
- An expert-streaming policy for DeepSeek V4.1 (not yet wired in).

**Upstream fixes carried early**
- YaRN/MTP fix (#854), Q8_0 prefill scratch sizing (#777), input token count (#860) and context-overflow diagnostics (#269).

This fork adds experimental q8 KV storage for Qwen3.8 Flash-Next on AMD Strix Halo (gfx1151). **F16 remains the default.** q8 offers long-context memory headroom, not guaranteed faster inference or identical output.

## Build from source

Follow the [upstream build requirements](https://github.com/gufo-org/gufo/blob/c92b19c5dc7ec12f83d0c2a5603cda78cc200b3e/README.md#build-from-source), then use the [current fork build and usage commands](docs/q8-kv-cache/reproduce.md). Historical validation used ROCm 7.2.4 / Clang 22; the pinned ROCm 10.1 / Clang 24 stack remains untested for this change.

## Client setup

[Client recipes](docs/clients/README.md) cover Claude Code, OpenCode, Cline,
Aider, Open WebUI and Pi, with truthful limits, authentication caveats and
separate CPU-contract versus live-model qualification procedures.

## Experimental q8 KV cache

Ordinary attention keys and values, including the MTP predictor, use block32 8-bit storage. Indexer keys stay F16; recurrent GDN state stays F32. Model weights are unchanged.

Append one selector to your usual Flash-Next launch arguments:

```sh
--kv-cache-type q8_0          # plain q8; suggested starting point
--kv-cache-type q8_0-rotated  # experimental Hadamard-rotated alternative
```

Both modes are always compiled into the normal HIP build. Eligible plain-q8 matrix prefill uses a shared transient F16 view automatically; there is no rotation or prefill-view build switch or opt-out. See [build and usage](docs/q8-kv-cache/reproduce.md).

- **Memory:** a 512-value K or V row shrinks from 1024 to 544 bytes, not a halving of total RAM. View scratch is shared per executor: 390.625 MiB at context 200000, capped at 512 MiB for supported geometry. See [memory and capacity data](docs/q8-kv-cache/memory.md).
- **Historical capacity:** an earlier view-enabled build completed four 376832-token slots and one 1048576-token slot with YaRN and LIVE follow-up turns. These are not fresh R3, task-quality or evicted-checkpoint results; see the [measured limits](docs/q8-kv-cache/memory.md).
- **Historical speed:** plain-q8 prefill with the view ranged −0.57% to +3.57% versus same-binary F16 across four 8K/32K cells. Decode was about 1% behind at 32K, preliminarily. These are not final-R3 performance measurements. Fork main additionally carries exact MTP-verify kernel speedups (Group4 −14% at n8); see [speed](docs/q8-kv-cache/speed.md#fork-only-decode-verification-speedups-2026-10-28).
- **Historical quality:** a 144-position / 18-history screen gave mean KL 0.00137127 plain / 0.000946485 rotated, with 141/144 top-1 agreement each versus F16. Rotation raised target NLL; it is not uniformly better.

**Upgrade:** version-18 Flash-Next caches, including F16, rebuild once. Current payloads are 19 for F16/plain q8 and 20 for rotated q8; both q8 modes support disk persistence. Incompatible type/epoch identities are rejected.

q8 is lossy. The original strict quality gates were not met, and long-agent task non-inferiority is unproved. Historical measurements do not qualify the final candidate. The [validation table](docs/q8-kv-cache/reproduce.md#candidate-validation-status) still awaits final native results; native assessments are not public approval.

[Documentation index](docs/q8-kv-cache/README.md) · [Quality](docs/q8-kv-cache/quality.md) · [Memory](docs/q8-kv-cache/memory.md) · [Speed](docs/q8-kv-cache/speed.md) · [Design](docs/q8-kv-cache/design.md) · [Limitations](docs/q8-kv-cache/limitations.md)
