# Phase 2B-10B Worker Greedy

Coordinator defaults to `--logits-response full`. Use
`--logits-response worker-greedy --generate --sampling greedy` for
compact responses. Greedy is the default sampling algorithm. Worker
is started with the same arguments as v1. Baseline selects locally.

Full uses the exact v1 frame and message schema. Worker accepts v1 or v2
in the first frame and then locks the session version. Compact uses v2
with the same six little-endian uint32 header fields. The v2 HELLO
appends a uint32 capability word equal to 1 to the v1 metadata. Both
sides must agree on all bytes before OPEN. The Coordinator does not
retry with v1 after a v2 rejection.

Each v2 EXECUTE adds a uint32 response kind after the activation type
and before token positions. Coordinator requests kind 8 for each
intermediate Prefill batch and kind 9 for the final Prefill batch and
each Decode. The success body starts with zero status, processed count
and vocabulary width, all uint32. Kind 8 ends there. Kind 9 appends
one uint32 token ID. Coordinator checks the response kind, session,
request, status, count, width, exact size and ID range before using it.
Malformed messages and inference errors terminate the session.

Worker computes and checks all output rows. Kind 9 runs the same strict
greater-than, finite-only argmax as Baseline and full Coordinator, on
the last row in place. Ties pick the lowest ID. The Coordinator
retains token history, EOG/limit/context priority and text rendering.
Kind 8 has no token ID, but runs the same Stage and KV update.

For 19 prompt tokens, batch 2 and 64 generated tokens, v2 adds
73 × 4 B to the Coordinator requests and 4 B to HELLO each way.
The expected totals are 675,484 B and 3,084 B, or 678,568 B combined.
Worker sends nine 36 B status frames and 64 40 B token frames, plus
200 B session control. All 82 Logits rows are still computed and
validated. The receiver body limit for successful results in this
mode is 16 B. Application counters exclude TCP/IP and SSH overhead.

Worker response serialization timing includes finite validation and
argmax. Their subintervals are exposed in compact_responses and
excluded from Worker Stage execution. Argmax includes finite validation
of the selected row. Coordinator response parsing is outside its RTT.
TCP detailed timing and CSV remain separate from routine comparisons.

## Intel physical hosts

Use the exact build, PE/DLL checks, SHA-256, sshd, SSH forwarding, and
Baseline/Worker/Coordinator commands in the
[Phase 2B-10A procedure](phase2b-10a-stage-communication.md). For
each budget (256 and 512 MiB), run both response modes three times
with depth 1 and fresh output directories. Rotate trial order and
record it. Replace the trial's `--tcp-perf-details` with `--perf-details`
for the normal series. Collect detailed runs separately. Append
`--logits-response full` or `--logits-response worker-greedy` to
Coordinator only. Use the same executable hash and GGUF on both PCs.
Compare baseline and coordinator generated IDs and UTF-8 with cmp.
Copy Worker logs, metrics and CSV into the matching trial directory.
For a failed session retain its logs, restart Worker, and retry with
a new output prefix.

~~~bash
python tests/moe-paging/distributed/analyze-stage-communication.py \
  --run "$OUT" --connection intel-ssh --output "$OUT/communication.json"
~~~

Keep v1 Full Logits as the fixed-token reference. Last Logits is an
option for a future Coordinator-side sampling mode. Projection,
Logits row count and computation are identical in the two modes.

## Local regression notes

The local validation host is Windows ARM64 with an x86-64 MSYS2
UCRT64 executable. TCP peers are on the same host. Both directions
of v1 interoperability were exercised with the retained earlier
Windows executable, and it rejected a v2 Coordinator before EXECUTE.
Cross-build fixed-token logits were finite, with maximum absolute
differences of 1.79e-6 (new Coordinator) and 2.52e-5 (new Worker).
Same-build fixed-token comparisons remain byte-exact. The independent
Python microbenchmark measures TCP framing overhead for small and
large payloads. C++ Stage timing comes from the Stage metrics.

For legacy JSON without protocol fields, the analyzers interpret
the wire format as v1 Full Logits. The communication analyzer checks
request positions, lengths, the two peers' counters and any supplied
frame logs. It retains legacy compact estimates for v1 input and
reports actual bytes and validated rows for v2 input.
