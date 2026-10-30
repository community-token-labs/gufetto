# CPU client protocol recordings

`chat.json`, `responses.json`, `messages.json` are synthetic request/backend
inputs. Corresponding `.sse` files are recorded response bodies from the real
HTTP routes using `tests/cli/http_server_test_support.hpp`'s fake backend, on
source base `fea076136676e234d87647f670ff8b3661249be5` with the client recipe
test additions. Recorded 2026-10-30 with native GNU14.2, cpu-test,
RelWithDebInfo, HIP OFF, assertions on. No production source changes.

These are NOT official client captures, real model inference, tool execution,
performance measurements or quality evidence. Backend text and contents are
public synthetic data; no credentials/auth headers are present. Generated
request/call/item IDs and creation times are left as recorded, not treated as
stable equality targets. No live-update/writer is present in the test.

`http_server_test` loads these fixtures relative to the source header, independent
of build-directory location. It reassembles each recorded SSE body under 1-byte,
7-byte and full-frame network fragmentation, validates contract semantics and
usage/cache accounting, then generates a new response via the actual route and
checks the same contract. Recorded and live events are also compared in order
as normalized JSON bytes: only generated id/call_id/item_id and created/created_at
fields are masked; counters, timings, schemas and all other fields must match.
Typed Responses/Messages require matching `event:` names and the complete
expected single-call event order. It returns the generated call ID in a second HTTP
request with a synthetic tool result and checks backend history and final SSE.
The fake backend does not touch a filesystem tool; real tools belong in the
[model-slot client qualification](../../../docs/clients/QUALIFICATION.md).

Required zero-valued numeric fields are checked for presence and type before
value comparison; a missing index/sequence/cache-creation count is not zero.

Contracts: Chat index/ID/name/argument assembly, tool_calls finish, terminal
usage and `[DONE]`; Responses typed events/sequence, function argument deltas
and completed output/call ID; Messages block indexes, partial JSON, tool_use
stop reason, message_stop and uncached/cache-read usage. The second turn checks
plain answer completion. Existing protocol tests cover reasoning, Unicode,
parallel tools, truncation, live streaming and cancellation beyond this bounded
single-tool fixture. Update fixtures only with an explicitly reviewed contract
change; do not regenerate them to hide a failing client contract.
