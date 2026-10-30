# Setup skill for coding agents

[`gufo-setup`](gufo-setup/SKILL.md) lets a user say "set up Gufo for
<model> on my Strix Halo" to their coding agent. The agent checks the host
with `gufo diagnose`, builds Gufo, picks a supported model variant, checks
fit with the read-only `gufo diagnose --plan`, starts `gufo serve llm`,
configures the client from [the client recipes](../clients/README.md) and
verifies the server with one request through
`tools/setup/verify_server.py`. It changes no engine behavior.

## Install

The file uses the Agent Skills `SKILL.md` format: YAML front matter with
`name` and `description`, then Markdown instructions. Claude Code loads
personal skills from `~/.claude/skills/<name>/SKILL.md`. Link the skill from
a Gufo checkout so it always matches the CLI built from that checkout:

```sh
mkdir -p ~/.claude/skills
ln -s "$PWD/docs/skills/gufo-setup" ~/.claude/skills/gufo-setup
```

Agents that support the same format load it from their own skills
directory; use the same link there. Agents without skill support can read
it as plain Markdown: ask them to "read `docs/skills/gufo-setup/SKILL.md` in
the Gufo checkout and follow it". Before a checkout exists, point the agent
at the file in the Gufo repository; its first step obtains the checkout.

## Design

- **User-facing location.** `.agents/skills/` holds developer skills for
  working on Gufo itself. This skill is for Gufo users, so it lives under
  `docs/skills/` next to the model and client guides it reads. The
  directory name matches the skill `name`, as the format requires.
- **Paths, not links, inside the skill.** The skill names repository files
  as plain paths relative to the checkout, so it works unchanged when linked
  into a skills directory.
- **No invented commands.** Every `--option` in the skill must exist in the
  CLI sources or the verify script; `setup_verify_test` enforces this and
  checks that every repository path it names exists.
- **Read-only or confirmed steps.** Diagnostics, `--plan`, `--help` and
  `GET` requests run freely. Cloning, building, downloads, manifests, client
  configuration and starting a server need the user's agreement and stay
  reversible (new directories, config backups, a server stopped by PID).
  Privileged host changes are explained, never performed.
- **No credential handling.** The agent never asks for, prints or stores
  keys or tokens; the user exports `GUFO_API_KEY` or logs in to Hugging
  Face themselves. The verify script reads the key from the environment,
  never prints it, and refuses to send it over plain HTTP to a non-loopback
  host unless told to.
- **No agent instruction files.** The skill forbids creating `AGENTS.md`,
  `CLAUDE.md` or similar files in the checkout or the user's projects.
- **Refusals are answers.** Startup admission, `--plan` states and HTTP
  error codes are explained from Gufo's own messages; the agent does not
  work around a refusal with other settings.

## Verify script

```sh
python3 tools/setup/verify_server.py [--base-url http://127.0.0.1:8080] \
  [--model ID] [--wait SECONDS] [--json]
```

It uses only the Python standard library and bypasses environment proxies.
It lists `/v1/models`, selects the text model (or `--model`), sends one
non-streaming Chat Completions request with `reasoning_effort: none` and
reports the loaded `context_length` and input modalities. `--wait` keeps
polling while the server answers 503 `loading`. Exit status is 0 on pass,
1 on a failed check and 2 on a usage error; failures print the server's
error code with a hint.
