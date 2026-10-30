---
name: gufo-setup
description: Set up the Gufo LLM engine on an AMD Strix Halo (gfx1151) Linux host - check the host, build Gufo, pick a supported model, check memory fit, start `gufo serve llm`, point a coding client at it and verify with one request. Use when the user asks to set up, install, run or connect Gufo, or to serve a local model on Strix Halo with Gufo.
---

# Set up Gufo on Strix Halo

You are helping a user run Gufo, an OpenAI/Anthropic-compatible LLM server
for AMD Strix Halo (`gfx1151`) on Linux x86-64. Work through the steps in
order and stop at the first failure you cannot explain from Gufo's own
messages. Every command and flag below exists in the Gufo CLI; do not add
flags that are not listed in `gufo <command> --help`.

## Safety rules (always apply)

1. **Read-only first.** Run read-only checks freely: `gufo diagnose`,
   `gufo diagnose --plan`, `--help`, `df -h`, `free -g`, `ls`, `GET` requests.
2. **Ask before anything that writes or costs.** Before cloning, building,
   downloading model weights, writing a manifest or client config, or
   starting a server, show the exact command and wait for the user to agree.
   Keep each such step reversible: new directories, backups of any config
   you change (`cp FILE FILE.bak-gufo`), and a server you can stop by PID.
3. **No privileged or host changes.** Never run `sudo`, change groups,
   kernel parameters, BIOS/UMA settings, power profiles or system packages.
   When `gufo diagnose` reports such a problem, explain it and give the user
   the fix to run themselves.
4. **Never handle credentials.** Do not ask for, read, print, generate or
   store API keys or Hugging Face tokens. The user exports them in their own
   shell (`GUFO_API_KEY`) or logs in themselves (`hf auth login`). Refer to
   keys only by variable name, never write a key literal into a file, and
   do not echo environment variables.
5. **Do not create `AGENTS.md`, `CLAUDE.md` or other agent instruction
   files**, in the Gufo checkout or in the user's projects.
6. **Loopback only.** Bind `--host 127.0.0.1` unless the user asks for LAN
   access; then require a key (`--api-key-file`, step 6) and a TLS reverse
   proxy. Never
   disable authentication to make a client work.
7. **Do not guess.** If a model, variant, flag or value is not in the Gufo
   docs, say so and ask. A refused startup is the answer, not something to
   work around with other settings.

## 1. Get a Gufo checkout

Use the checkout the user already has. Otherwise ask which repository to
clone (upstream is `https://github.com/gufo-org/gufo`) and, after they agree,
clone it into a new directory. The rest of this skill runs from that
checkout; call its path `GUFO_SRC`. Model and client guides live in
`docs/models/README.md` and `docs/clients/README.md` inside it.

## 2. Check the host (read-only)

If a Gufo binary already exists, run `gufo diagnose`; otherwise do this
right after step 3. Exit status 0 means no failed check, 1 means a check
failed, 2 means a usage error.

```sh
"$GUFO" diagnose
"$GUFO" diagnose --json   # machine-readable report
```

The `gpu` check must report `Detected AMD Strix Halo GPU (gfx1151)`. Any
other architecture (`Unsupported GPU architecture: expected gfx1151`) means
this host is not a supported target: stop and tell the user. Note the
`memory` check's `availableBytes` for step 5. Do not run
`gufo diagnose --benchmark`; it allocates GPU memory and is not needed.

## 3. Install Gufo (ask first)

Prefer Nix if `nix` is installed; do not install Nix yourself.

```sh
cd "$GUFO_SRC"
nix build                     # production package -> result/bin/gufo
GUFO="$GUFO_SRC/result/bin/gufo"
"$GUFO" --version
```

Without Nix, use the documented CMake path. Its prerequisites (ROCm/HIP for
gfx1151 and the listed libraries) are linked from the checkout's
`README.md`; if they are missing, list them for the user instead of
installing packages.

```sh
cmake --preset release
cmake --build --preset release --parallel 4
GUFO="$GUFO_SRC/build/release/gufo"
```

## 4. Pick a supported model and variant

Only offer models in the table in `docs/models/README.md`, and read that
model's `docs/models/<model>/README.md` before choosing files. For a coding
agent on one Strix Halo, the text models are:

| Model | Variant to download | Notes |
| --- | --- | --- |
| Qwen3.8 27B | `Qwen3.8-27B-UD-Q4_K_XL.gguf` (16.35 GiB) or `UD-Q8_K_XL` (29.30 GiB) | Optional DFlash2 draft and `mmproj-BF16.gguf` |
| Qwen3.8 Flash-Next | `UD-Q4_K_XL` (four shards) | Optional shared-Q8 MTP; pass the first shard |
| DeepSeek V4 Flash | Flash 0731 mixed IQ2/Q2/Q8 GGUF (80.76 GiB) | Needs a 128 GiB system |

If the user names a model that is not in the table, say it is not a
supported Gufo model; do not substitute another one silently. Use the exact
`hf download` command, repository, revision and file names from the model
README. Before downloading, check free disk space with `df -h`, tell the
user the size and license, and let them accept any gated license and log in
to Hugging Face themselves.

## 5. Check fit (read-only)

`gufo diagnose --plan` reads a pinned TSV manifest, verifies local files by
size and SHA-256 and prints a memory lower bound. It downloads nothing and
loads no GPU model. The format is in `docs/MODEL-SETUP.md`. Write the
manifest (ask first) next to the weights, because `file` is relative to the
manifest's directory. Columns are tab-separated:

```text
group	status	repo	revision	file	bytes	sha256	license	gated	role	why	fit	source	context	host_available_bytes
qwen27	candidate	unsloth/Qwen3.8-27B-GGUF	<40-hex revision>	Qwen3.8-27B-UD-Q4_K_XL.gguf	<bytes>	<64-hex sha256>	<license>	false	target	coding agent	unchecked	https://huggingface.co/unsloth/Qwen3.8-27B-GGUF	32768	<availableBytes>
```

Take `revision`, `bytes` and `sha256` from the model README pin or the
Hugging Face file page, not from the downloaded file itself. Use `unknown`
for a pin you cannot find; that row is then `NOT-EVALUATED`. Add
`device_available_bytes` only if you know it; never invent capacity values.

```sh
"$GUFO" diagnose --plan /path/to/models/models.tsv
```

- `DOES-NOT-FIT`: the lower bound already exceeds capacity. Pick a smaller
  variant or context; exit status is 1.
- `UNPROVEN`: within the bound, or evidence missing. It is **not** a fit
  guarantee; the real check is the server's startup admission in step 6.
- `NOT-EVALUATED`: unpinned rows, split GGUFs or families without a bound.
  It does not mean the model cannot run.
- A size or SHA-256 mismatch means a corrupt or wrong download: report it.

## 6. Start the server (ask first)

Start from the model README command and add the common client settings from
`docs/clients/README.md`: loopback, one served name, a 32768-token context
for the first session and thinking off. Keep the model's own draft and
projector options. For Qwen3.8 27B Q4 without a draft:

```sh
"$GUFO" serve --host 127.0.0.1 --port 8080 \
  llm --model "$MODEL" --served-model-name gufo-local \
  --context 32768 --think off > gufo-serve.log 2>&1 &
echo "gufo serve PID $!"
```

If the user wants authentication, start the server with
`--api-key-file PATH`, naming a file the user created with the key. Do not
read or write that file. The server takes the key only from that file or
from `--api-key`. Clients and the verify script read it from `GUFO_API_KEY`,
which the user exports in their own shell; that variable does not configure
the server. If `"$GUFO" serve llm --help` does not list `--api-key-file`,
this build predates it: tell the user that `--api-key` would show the key in
the process list (see `docs/clients/README.md`), and let them decide. Use `--speculative dflash2 --dflash-model "$DRAFT"`
(Qwen3.8 27B), `--speculative mtp --mtp-model "$MTP"` (Flash-Next) or
`--dspark-model "$DSPARK"` (DeepSeek) only as the model README shows.
`--mmproj` is auto-discovered beside the model; `--no-vision` disables it.
Stop the server with `kill <PID>`. While the model loads, the port answers
503 `loading`; wait for the `event=listening` log line or a 200 from
`GET /ready`.

## 7. Verify with one request

```sh
python3 "$GUFO_SRC/tools/setup/verify_server.py" --wait 600
```

It lists `/v1/models`, sends one short Chat Completions request with thinking
off and prints `PASS` with the loaded `context_length` and input modalities,
or `FAIL` with a hint. It reads the key from `GUFO_API_KEY` when that is set
and never prints it. It sends a key over plain HTTP only to `127.0.0.1` or
`::1`, and never follows redirects. Use `--base-url` for another port and
`--model` for a second served name. Report the actual `context_length`; clients must not be
configured with a larger window.

## 8. Configure the client (ask first)

Open the matching recipe and follow it exactly, using the served name
(`gufo-local`), the loaded context and an output budget smaller than it:

| Client | Recipe |
| --- | --- |
| Claude Code | `docs/clients/claude-code.md` (experimental; bearer `ANTHROPIC_AUTH_TOKEN`) |
| Pi | `docs/clients/pi.md` |
| OpenCode | `docs/clients/opencode.md` |
| Aider | `docs/clients/aider.md` |
| Cline | `docs/clients/cline.md` |
| Open WebUI | `docs/clients/open-webui.md` |

Back up any config file before editing it, reference keys through the
environment interpolation the recipe shows, and show the user the diff.
OpenAI clients use `http://127.0.0.1:8080/v1`; Anthropic clients use
`http://127.0.0.1:8080` without `/v1`. A working client setup is not model
qualification; `docs/clients/QUALIFICATION.md` covers that separately.

## 9. Explain failures with Gufo's messages

| Message | Meaning and next step |
| --- | --- |
| `Unsupported GPU architecture: expected gfx1151` | Not a Strix Halo GPU; unsupported host |
| `verified to fit: --context N` | Startup refused; N passed this startup's admission probe. Offer to restart with `--context N` (or fewer `--sessions`); Gufo never changes it silently |
| `no fitting context found within the search limits` | No probed context fit; free memory or choose a smaller variant |
| `insufficient component backing budget` | RAM cache budget too small; use `--cache-ram-bytes off` if RAM checkpoints are not needed, or reduce context/sessions |
| `... pass --enable-yarn` | Flash-Next `--context` above the native 262144 needs `--enable-yarn`; it changes every request, so prefer a smaller context |
| 503 `loading` | Model still loading; wait |
| Connection refused | The server is not running; read `gufo-serve.log` |
| 401 | The server has a key; the client must send it as a bearer token (`x-api-key` alone fails) |
| 404 `model_not_found` | Client sends a model ID that is not served; use `gufo-local` |
| 503 `device_lost`, exit status 75 | GPU context lost; restart `gufo serve` |
| 429 | Admission queue full (`--max-pending`); retry |

For anything else, quote the exact message to the user, look it up in
`docs/SERVER.md` and the model README, and ask before changing settings.
