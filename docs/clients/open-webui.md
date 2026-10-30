# Open WebUI → Gufo

Source: official [OpenAI-compatible provider guide](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible).
It identifies Chat Completions as the normal protocol and Responses support
as experimental. Read [the common contract](README.md).

1. In **Admin Panel → Settings → Connections**, add an OpenAI-compatible
   connection with URL `http://127.0.0.1:8080/v1` and Gufo's private bearer key.
2. Verify the connection: this calls `/models` with bearer authentication.
   Select the actual served `gufo-local` model, optionally allowlisting that ID.
3. Set the model/request maximum output to 4096 only for the common loaded
   32768-context example. Any client context metadata must use the actual loaded
   context; do not inherit a cloud-model 1M label or enable extended context.
   Inspect outbound payloads; UI settings don't increase server capacity.
4. Start with ordinary text chat, thinking off and no image/hosted tools.

If WebUI runs in a container, `127.0.0.1` refers to that container, not Gufo.
Use a host-only bridge for a colocated container, or a reverse proxy with TLS
for LAN access: HTTP bearer keys otherwise travel in cleartext. Do not casually
bind an unauthenticated Gufo to all interfaces. Gufo has one
shared credential, not per-WebUI-user isolation. The WebUI backend calls Gufo
without a browser `Origin`; if browsers call Gufo directly (direct
connections), restrict them with `--cors-origin <WebUI origin>`. See
[authentication](../SERVER.md#authentication-and-exposure).

## Reasoning, cache and tools

Use Gufo server `--think off`; only add verified Chat reasoning fields for
the loaded model. Leave unsupported active sampler/logprob parameters unset.
Automatic prefix caching has no advertised retention TTL. Responses must send
full history with `store:false`, no previous-response state; qualify it
separately instead of switching the existing chat connection silently.

For tool qualification choose **Native** function calling in model Advanced
Params, with a reviewed bounded client-side tool. The provider guide documents
the requirement for streamed tool `index`; the CPU fixture requires that key
to be present, numeric and equal to the expected call index.
Legacy text-tool mode is not native streaming tool coverage.

## Qualification slot

Use [the shared procedure](QUALIFICATION.md) with an approved tool that can
read/edit/test only the disposable fixture repository. Never grant a general
host shell just to test connectivity. If no such reviewed executor is installed,
mark real tool execution BLOCKED and retain a plain-chat result separately.
Record the chat export, tool schema, call/result IDs and **actual** executor
output. Replay the same task with Pi. Merely rendering a tool call or copying
a tool result into chat is not end-to-end acceptance.
