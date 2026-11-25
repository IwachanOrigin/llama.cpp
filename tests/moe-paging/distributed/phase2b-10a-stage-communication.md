# Phase 2B-10A Stage TCP and logits

## Protocol and data path

The frame definition and channel are in qwen3moe-stage-tcp.cpp. See
stage-tcp.h for connect, listen, accept and transfer. The 24 B frame
header is six little-endian uint32 fields, in order: magic 0x53454f4d,
protocol version 1, message type, payload bytes, session and request.
The cap is 16 MiB of payload. The payload uses little-endian uint32
and IEEE 754 binary32. Token positions are encoded as uint32 and are
checked against the Worker cursor before being passed to llama_batch.

~~~mermaid
sequenceDiagram
    participant C as Coordinator
    participant W as Worker
    C->>W: TCP connect to the loopback endpoint
    C->>W: HELLO, session 0, request 0, model and stage metadata
    W-->>C: HELLO with exact matching body
    C->>W: OPEN, empty, session 0, request 0
    W-->>C: ACK, empty, new nonzero session, request 0
    loop Prefill batches then one token per additional Decode step
        C->>C: Compute local stage and serialize boundary
        C->>W: EXECUTE, session, next request ID
        W->>W: Validate, parse, compute and synchronize
        W-->>C: RESULT, matching session and request ID
        C->>C: Parse every row, retain the last row for generation
    end
    C->>C: Greedy, EOG/limit/context check before further Decode
    alt normal stop
        C->>W: CLOSE, next request ID, empty body
        W-->>C: ACK, same ID, empty body
    else exception
        C-->>W: Best-effort ERROR from the failing endpoint
        Note over C,W: Exit nonzero, close socket, failed metrics when enabled
    end
~~~

| Type | Body | Validation |
|---|---|---|
| 1 HELLO both ways | 11 words for architecture, layers, split/range, hidden, vocab, F32 type, effective context, batch and identity length, followed by identity bytes. A conditional 4-word tail includes identity mode, requested context, threads and Pager budget. | Exact byte match against locally generated metadata. The auto identity is a model content hash. A supplied label is merely compared. |
| 2 OPEN, C→W | Empty | Session and request both zero. |
| 7 ACK, W→C | Empty | OPEN returns nonzero random session and request zero. CLOSE echoes the final request. |
| 3 EXECUTE, C→W | Count, hidden width, sequence zero, type, count positions, then count × hidden FP32 scalars. Body length 16 + 4n + 4nh. | Session, consecutive request and positions, batch/context bounds, exact dimensions, exact length and finite values. |
| 4 RESULT, W→C | Status zero, count, vocabulary, followed by count × vocab FP32 scalars. Body length 12 + 4nv. | Matching frame, status, count, vocab, exact length and finite values. |
| 5 CLOSE, C→W | Empty | Active session and next request ID. |
| 6 ERROR, failing endpoint | Exception text bytes | Best effort. Receiver terminates with a peer error. It is not an alternate success payload. |

The connection is reused in stop-and-wait order. The sockets are
nonblocking on Windows (Winsock FIONBIO) and Linux (O_NONBLOCK). A select
loop operates against deadlines and transfers chunks capped at 65,536 B.
Header and payload are sent separately. Retries handle interruption
and would-block, and EOF is an error within a frame. TCP_NODELAY and
socket buffer sizes are left at OS defaults in the Stage implementation.
Defaults are 10 s for connect, accept, frame and response. The
Coordinator's RESULT first byte uses the response timeout, followed by
a fresh frame deadline for the rest. Other receives use the frame limit.

Application wire counters include HELLO, OPEN, CLOSE and ACK frames.
They exclude the TCP handshake, packet overhead and SSH encryption
overhead. With SSH forwarding, the local Stage socket and the SSH
transport socket are distinct connections.

## Conversion, buffers and selection

At the split, Coordinator obtains the boundary via llama_get_embeddings.
put_float bit-copies each scalar into an integer and appends its four
wire bytes to a fresh payload. The Worker receives into a byte vector,
then floats allocates and zero-initializes a float vector, decodes every
scalar and checks finiteness. The vector becomes batch.embd. The Stage
input upload is inside the Worker compute interval. The returned logits
use llama_get_logits_ith for each row and the same put_float loop. The
Coordinator allocates the receive body and then a float vector for all
rows. Generation last.assign copies one vocabulary row into its next
selection buffer. Fixed-token runs instead append full logits bytes to
the output artifact. Frame bodies are moved into channel::send, so that
handoff does not duplicate the body. Vector growth can relocate earlier
bytes. The timings do not separately measure those reallocations,
GGML staging or OS copies.

Worker sets batch.logits to one for every token. Thus every input row
is generated and transferred. In generation, intermediate Prefill
results are overwritten. Only the last Prefill row and each additional
Decode row drive greedy. Fixed-token mode retains all requested rows.
For h=2048, a boundary row is 8192 B. A two-token Prefill request is
16,432 B including its header. For v=151,936 a logit row is 607,744 B.
No Expert ID is on the wire. Expert artifacts are written locally.

greedy scans increasing IDs, checks each value is finite and changes
the winner only on strict greater-than. Equal maxima choose the
smallest ID. Empty logits and invalid vocabulary widths are rejected.
NaN and either infinity fail. Worker also rejects non-finite values in
rows the Coordinator would discard. Generated IDs are appended before
prioritized EOG, generation limit and context checks. After a stop
there is no extra Decode. The Coordinator detokenizes the IDs and
handles output and UTF-8 validation.

## Diagnostics and limits

--tcp-perf-details enables the existing perf CSV and JSON and adds
tcp_detail to JSON. It measures first-byte wait where applicable,
remaining frame header and payload transfer, send/recv calls, partial
calls, retry counts and the Coordinator's explicit last-logit row
copy. Serialization/deserialization timings still come from the
original CSV and include copy work where applicable. The bounded
request array reserves at most 4096 entries. An incomplete array or
failed run has complete=false and is rejected by the analysis tool.
The new timers and reservation are opt-in. The existing perf CSV format
is unchanged.

Analyze durations across hosts by request and session ID, without
subtracting absolute clocks. The RTT encloses Coordinator send,
Worker receive completion/parse/compute/encode and the RESULT return.
The tool subtracts Worker execution, parse and encoding from RTT to
leave an unexplained residual. Coordinator encoding and parsing occur
outside the RTT. The first-byte wait already contains Worker compute.
Worker receive can span the prior request's Coordinator compute, and
send/recv intervals overlap. These elapsed durations must not be added
as wire time. No SSH wire bytes are inferred.

## Design decision for 2B-10B

Implement a negotiated Worker Greedy response mode with empty
acknowledgement bodies after a small status/shape prefix on intermediate
Prefill, and uint32 token IDs at generation positions. Reuse one
strict/finite greedy helper across roles or use an identical shared
implementation. Keep validation of all generated rows in the first
change to preserve current errors, including unused Prefill rows.
Keep EOG, context, stop reason and the generation loop on Coordinator.
Validate reply kind, session, request, length, required row and token
range. Treat an explicitly selected unsupported mode as an error, using
a negotiated protocol revision or capability before OPEN. Keep Full
Logits as the compatibility and fixed-token reference. Last Logits
is a useful later response mode for Coordinator-side sampling.

In the observed 73-request schedule, the hypothetical modes keep
the current 12-byte RESULT prefix and 24-byte frame, all control
frames and the exact activation requests. Full, last/empty and
greedy/empty yield 49,837,832 B, 38,898,440 B and 3,080 B
Worker→Coordinator respectively. Coordinator→Worker stays 675,188 B.
Greedy stores at most 16 B of RESULT body instead of 1,215,500 B,
and removes the large return serialization, all-row parsing and
last-row copy. Worker still reads logits for validation and argmax.
Greedy can preserve exact results for this deterministic path.
Temperature, top-k/p/min-p, repetition penalties, grammar and seeded
sampling can depend on the full logits or Coordinator state. A
partial candidate list cannot generally replace full logits for
top-p. Keep the full response route.

Treat Projection skipping as a separate experiment. In Qwen3MoE
build_inp_out_ids is consumed in the last layer before its FFN
(src/models/qwen3moe.cpp), so changing output-row flags can affect
last-layer Expert operations and tracing. Confirm zero-output
graph behavior, KV position/update and eventual logits equivalence
before proposing a graph change. Socket tuning and vector reserve
changes have narrower savings than removing the serialized output
body. No network or model-performance speedup is assumed from
the byte reduction.

## Reproduce on Intel hosts

Use the [9B two-host steps](phase2b-9b-expert-prefetch.md) for the CPU
configuration, SSH login/tunnel, Baseline, Worker and Coordinator.
On PC-A build under UCRT64 with cmake --build build-lan --target
qwen3moe-stage-tcp test-backend-ops --parallel 4. Before distribution,
inspect the exe's imported DLL names with objdump -p and the build
configuration for dynamically loaded backends. Use only the DLLs
actually needed. Hash the exe and each resolved runtime file explicitly
with Get-FileHash on both hosts. No DLL wildcard is required.

On PC-B use administrator Windows PowerShell 5.1 to start sshd
(Start-Service sshd), verify login from PC-A and forward a loopback
port as in 9B. Use 24/24 layers, mmap, ctx 256, batch/threads 2,
TinyLFU 256 or 512, and prefetch depth 1. Keep separate directories for
ordinary runs, --tcp-perf-details runs and failures. Use --expert-trace-limit
only for separate Expert sequence comparisons. Compare generated files
with cmp, then copy Worker metrics, CSV and logs back. If a run fails,
keep its files and retry under a new prefix after checking model hash,
DLL resolution and RAM/commit.

~~~bash
python tests/moe-paging/distributed/analyze-stage-communication.py \
  --run results/prompt-a/tcp-detail --connection intel-ssh --output results/communication.json

python tests/moe-paging/distributed/benchmark-stage-transport.py loopback \
  --samples 100 --warmup 10 --repeats 3 --output results/loopback.json
~~~

For the microbenchmark separately use server --port 50054 on PC-B
and client --port 50054 on PC-A with identical settings and a dedicated
SSH loopback forward. The synthetic protocol uses v1 framing and
payload validation rather than the inference handshake. It uses
Python sockets/sendall and Python struct for CPU-only codec timing.
Those timings characterize the benchmark transport and payload sizes.
Production C++ encode/decode costs are measured with Stage diagnostics.

### Example trial commands

On PC-B, in administrator Windows PowerShell 5.1, run
`Start-Service sshd` and `Get-Service sshd`. On PC-A, verify interactive
login with `ssh user@pc-b`, then keep the following forward open.

~~~bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:50053:127.0.0.1:50053 user@pc-b
~~~

In the UCRT64 shell on each PC, set MODEL to the local GGUF, EXE to
the verified executable, and ID to the SHA-256 of the matching GGUF.
Set a fresh OUT on both hosts for each trial. Set PROMPT on PC-A to
the UTF-8 user prompt. Use the same budget and generation limit on
both hosts.

~~~bash
mkdir -p "$OUT"
common=(--ctx 256 --batch 2 --threads 2 --pager-policy tinylfu
        --pager-budget-mib 256 --expert-prefetch-depth 1
        --perf-cache-state uncontrolled --tcp-perf-details
        --accept-timeout-ms 90000 --frame-timeout-ms 30000
        --response-timeout-ms 300000)
gen=(--generate --chat --prompt-file "$PROMPT" --max-new-tokens 64)

# Run Baseline on PC-A, then start Worker on PC-B.
"$EXE" baseline "$MODEL" 24 48 0 mmap "$OUT/baseline" "$ID" "${common[@]}" "${gen[@]}" > "$OUT/baseline.log" 2>&1
"$EXE" worker "$MODEL" 24 48 50053 mmap "$OUT/worker" "$ID" "${common[@]}" > "$OUT/worker.ready" 2> "$OUT/worker.log"

# Start Coordinator on PC-A once Worker reports READY.
"$EXE" coordinator "$MODEL" 24 48 50053 mmap "$OUT/coordinator" "$ID" "${common[@]}" "${gen[@]}" > "$OUT/coordinator.log" 2>&1
cmp "$OUT/baseline.generated_tokens" "$OUT/coordinator.generated_tokens"
cmp "$OUT/baseline.generated.txt" "$OUT/coordinator.generated.txt"
~~~

Copy `worker.metrics.json`, `worker.perf.csv`, and `worker.log` from
PC-B into the matching PC-A OUT before analysis. For the ordinary
comparison replace `--tcp-perf-details` with `--perf-details` and use a
different OUT. For the separate transport benchmark open the same
kind of forward on a free port such as 50054.

~~~bash
# PC-B
python tests/moe-paging/distributed/benchmark-stage-transport.py server --port 50054
# PC-A
python tests/moe-paging/distributed/benchmark-stage-transport.py client --port 50054 --output results/transport-ssh.json
~~~

## Reference measurements

The Phase 2B-10A local run used real Qwen3-30B-A3B-Q4_K_M on a
Windows ARM64 Snapdragon X Elite host, running the UCRT64 x86-64
binary under emulation. Stage peers used TCP loopback and the OS file
cache was uncontrolled.

With a 256 MiB budget, prefetch depth 1 and prompt A, 64 generated
tokens required 10 Prefill requests and 63 further Decode requests.
Both the generated IDs and UTF-8 output matched the Baseline and the
earlier reference. The RTT sum was 57,672.155 ms, including
57,528.925 ms of Worker stage time. The residual was 143.230 ms.
Worker parsing and encoding took 0.868 and 41.872 ms, leaving
100.489 ms unexplained. Coordinator parsing took 38.989 ms outside
RTT and the explicit last-row copies moved 44,365,312 B in 7.289 ms.
The 57,598.719 ms first-byte wait includes remote compute.

The independent stop-and-wait loopback benchmark used six request
sizes, seven response sizes, ten warmup and 100 measured exchanges
per case, with three repetitions. For an 8 KiB request, trial median
RTTs were 0.0606–0.0619 ms for a 4 B response, 0.2952–0.3073 ms for
607,744 B and 0.6265–0.6591 ms for 1,215,488 B. The benchmark uses
Python sockets, so its result describes the tested payload sizes and
environment. The Intel/SSH RTT and a speedup for a future response mode
remain to be measured.
