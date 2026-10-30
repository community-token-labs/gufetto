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
SKILL="$HOME/.claude/skills/gufo-setup"
mkdir -p "$HOME/.claude/skills"
if [ -e "$SKILL" ] || [ -L "$SKILL" ]; then
  echo "$SKILL already exists; inspect it, do not replace it" >&2
else
  ln -s "$PWD/docs/skills/gufo-setup" "$SKILL"
fi
```

Never overwrite an existing `gufo-setup` entry, whether it is your own copy,
another checkout's link or an edited skill. Inspect it with `ls -l` and
remove it yourself only if you no longer need it.

The link points at this checkout's absolute path. If you move or delete the
checkout, the link dangles and agents stop seeing the skill; remove the
dangling link as below and create it again from the new location.

To uninstall, remove the entry only if it is a symbolic link to a
`docs/skills/gufo-setup` directory; never delete a real directory there:

```sh
SKILL="$HOME/.claude/skills/gufo-setup"
case "$(readlink "$SKILL")" in
  */docs/skills/gufo-setup) rm "$SKILL" ;;
  *) echo "$SKILL is not a gufo-setup link; left unchanged" >&2 ;;
esac
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
- **No invented commands.** `setup_verify_test` checks every `--option` in
  the skill against the `--help` of the command it belongs to (built
  `gufo diagnose`, `gufo serve`, `gufo serve llm`, or the verify script). It
  also checks that each quoted Gufo message still exists in the sources, and
  that every repository path the skill names exists. Its single pending
  option, `--api-key-file`, is used only after the skill checks `--help`.
- **Read-only or confirmed steps.** Diagnostics, `--plan`, `--help` and
  `GET` requests run freely. Cloning, building, downloads, manifests, client
  configuration and starting a server need the user's agreement and stay
  reversible (new directories, config backups, a server stopped by PID).
  Privileged host changes are explained, never performed.
- **No credential handling.** The agent never asks for, prints or stores
  keys or tokens; the user exports `GUFO_API_KEY` or logs in to Hugging
  Face themselves. The server takes its key from `--api-key-file`;
  `GUFO_API_KEY` is client-side only. The verify script reads the key from
  the environment and never prints it. It never follows redirects, and it
  sends the key over plain HTTP only to numeric `127.0.0.1`/`::1`, unless
  told otherwise (`localhost` is not trusted).
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
error code with a hint. With `--json` every outcome, usage errors included,
prints one result object; `-h`/`--help` still prints plain help and exits 0.
Malformed responses, timeouts and transport errors become plain failures
without tracebacks.

The script accepts keys of printable ASCII (0x21-0x7e). The server accepts
any single-line key, so a key containing spaces or non-ASCII UTF-8 works
with `gufo serve` but is rejected here; choose a printable ASCII key to use
both. Every string or integer decoded from a response is scrubbed of the
literal key, shown as `•••`, before it is truncated or printed. A server
echoing the key in another encoding, such as base64 or percent-encoding, is
out of scope.
