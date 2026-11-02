# Community handoff

This page lists the work that was still in progress when Gufetto was
published (2026-11-02). Each item has its own branch in this repository. None
of them is in `main`, and none has passed every check listed here. Pick
anything up: the "next step" column says where it stopped.

Every branch was rebased or merged onto some earlier `main`, so expect to
re-merge onto the current `main` and re-run the checks before landing. "CPU
proven" means the hosted contract suite (`cmake --build --preset pr`) and the
format check passed. "Native" means a run on Strix Halo (`gfx1151`) hardware.

## Ports of upstream Gufo pull requests

| Branch | What it is | State | Next step |
|---|---|---|---|
| `intake/pr-814-logprobs` | Upstream #814: OpenAI-style `logprobs` | CPU proven on `main` `2b0181b7` | Native run: check that compiled code is unchanged with logprobs off; logprobs requests with MTP off and on; streamed, Responses-API and constrained-output cases; upstream's functional logprobs suites |
| `intake/pr-882-mtp-early-stop` | Upstream #882: stop MTP drafting early | Ported; a review suspects a slowdown on the cache-restore path | Three-way timing: `main` vs the port vs the port plus a one-line restore fix |
| `intake/pr-858-loader-buffered-retry-on-467c` | Upstream #858: buffered retry in the weight loader | CPU proven | Native load + greedy-output smoke. The last attempt failed in our harness (it compared a log line whose text varies), not in the change |
| `intake/pr-740-qfn-mmq-arena-growth-on-467c` | Upstream #740: let the Flash-Next MMQ arena grow | CPU proven | Find a Flash-Next workload that grows past the 256 MiB arena (none of ours did), then correctness and per-request timing |
| `intake/pr-874-27b-verify-gemm` | Upstream #874: Qwen3.8-27B verification GEMM tuning | Native build stage passed | Candidate build, then the 27B correctness and timing phases |

## Our own fixes and features

| Branch | What it is | State | Next step |
|---|---|---|---|
| `topic/full-logits-cold-score` | Fix a race where a cold-slot scoring request lost a metadata lock to the idle-spill worker | CPU proven on `main` `2b0181b7`, including an 800/800 stress run; the GPU code is unchanged from a native-proven earlier head | Re-check the merge with `main`, then land |
| `topic/model-config-arith-hardening` | Validate model config fields first and use checked 64-bit arithmetic for derived sizes | CPU proven | Native load smoke on Flash-Next (Q4 + MTP) |
| `topic/gguf-arch-required` | Refuse GGUFs with a missing or unknown `general.architecture` | CPU proven, including a synthetic-header check that each architecture still routes to the right loader | Same native smoke. Note: `deepseek41` GGUFs are now refused clearly until a DeepSeek V4.1 loader dispatches before the generic config path |
| `topic/gufo-setup-skill` | An agent skill that sets Gufo up on a new machine | Reviewed | Rebase onto current `main` |
| `topic/named-memory-reservations` | Named memory reservations, groundwork for DeepSeek V4.1 | In progress | DeepSeek V4.1 already runs in other community Gufo forks (e.g. franzmoca's `feat/ds41`); build on that work rather than starting over |

## Gemma 4 (26B-A4B)

`topic/gemma4-g406-reference` holds the whole reviewed stack, one commit per
layer (G4-01 to G4-06). Every layer was reviewed:

- **G4-01–G4-03:** model config, weight inventory and tokenizer (checked
  against llama.cpp's Gemma 4 tokenizer).
- **G4-04a/b:** chat template and tool declarations, with 40 Jinja2-rendered
  goldens.
- **G4-05:** tool-call parsing and the Gemma 4 tool grammar. It rejects
  duplicate keys and integers beyond ±2^53, and keeps `-0`.
- **G4-06:** a scalar FP32 CPU reference judged by an independent float64
  numpy port of llama.cpp's Gemma 4 graph (`tools/gemma4/reference_oracle.py`).
  Logits must match within a 1e-4 tolerance, and routing and attention
  mutations are caught.

Next: the GPU runtime (G4-07 onward), then a KL gate against the G4-06
reference on the real model file. The stack has one adjacent-line merge
conflict with `main` in `cmake/Checks.cmake`. Test notes and known limits are in
`tests/models/gemma4/README.md` on that branch.

## Parked

These were paused to keep scope narrow, not because they failed:

| Branch | What it is |
|---|---|
| `topic/qwen36-35b-a3b-loader`, `topic/moe-kernels-shared-leaf` | Qwen3.6-35B-A3B loader and a shared MoE kernel leaf |
| `topic/gptoss-mxfp4-cpu-reference` | gpt-oss-120B MXFP4 CPU reference |
| `topic/rerank-qwen3-reference` | A Qwen3 reranker reference, for the rerank API that `main` already exposes without a runtime |

## How we worked

Our docs and commit messages mention `nimo-1` and `nimo-2` (our two Strix Halo
machines), "receipts" (private run records) and "coordinators" (the agents
that scheduled the machines). None of those records are needed to build or run
Gufetto.


Changes landed only after one independent review plus CPU checks. Changes touching GPU code or speed also
needed a native run. Timing claims are per request and per phase, at 5% and
10 ms, never averaged, and noisy results stay "inconclusive". CLAUDE.md and
docs/TESTING.md describe the checks. Measurements used ROCm/HIP 7.2.4; the
pinned Nix toolchain (ROCm 10.1) is a separate configuration, so don't compare
numbers across the two.
