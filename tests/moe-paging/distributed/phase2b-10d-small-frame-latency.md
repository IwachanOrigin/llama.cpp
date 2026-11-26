# Phase 2B-10D Small Stage TCP frames

## Transport behavior

The socket is a nonblocking Winsock or POSIX TCP socket with RAII
ownership. Connect and accept return a connected socket. Each transfer
uses select against the applicable deadline, tries a chunk of at most
64 KiB, advances on a partial transfer, retries would-block and
interruption, and fails on EOF, timeout or other socket errors.

Pass these options independently on Coordinator and Worker.

| Trial | Flags |
|---|---|
| A | `--tcp-nodelay 0 --tcp-small-frame-coalesce 0` |
| B | `--tcp-nodelay 1 --tcp-small-frame-coalesce 0` |
| C | `--tcp-nodelay 0 --tcp-small-frame-coalesce 1` |
| D | `--tcp-nodelay 1 --tcp-small-frame-coalesce 1` |

Both default to 0. NODELAY 0 leaves the OS setting in place. NODELAY 1
uses setsockopt on the connected socket before HELLO; getsockopt reports
the effective value in either mode. Failure is fatal. Winsock takes a
char pointer and int length. POSIX uses an int value via void pointer
and socklen_t. Baseline has no connected socket or transport summary.

Coalescing uses 256 bytes of stack storage and includes the 24-byte
header in the bound. The threshold covers 36/40-byte replies, typical
HELLO and short errors with a small bounded copy. Empty bodies use
only the header. Hidden State and Full Logits bodies remain split.
Both v1 and v2 keep the same bytes, framing and receive path. A
combined transfer can still need multiple send calls or TCP segments.

Detailed metrics retain the same CSV. Each coalesced send has separate
combined_send_ms, combined_send_bytes, coalesce_copy_ms and
coalesce_copy_bytes. Header and payload send durations are null
for that event. Combined send contributes to existing total send
time once. Counts and timings are application observations.

## Cause investigation

The supplied Phase 2B-10C Wi-Fi/SSH totals showed 52–54 ms median
Coordinator payload receive for Greedy, with all 73 payload receives
above 40 ms for each tested budget. The actual zip was unavailable.
A split write can leave a small body behind an unacknowledged header;
Nagle and delayed ACK are a hypothesis. SSH channel scheduling,
SSH's network socket, buffering, radio contention/retries and OS
scheduling can also contribute. Stage NODELAY affects only the
Coordinator-to-local-SSH and remote-sshd-to-Worker sockets. The
encrypted SSH TCP connection between them is a distinct connection.

## Phase 2B-10C wired follow-up

User-supplied single diagnostic trials reused revision 43ad571fc,
the same executable and four DLLs, and SSH forwarding. PC-A used
1 Gbps Ethernet with Wi-Fi disconnected. ICMP showed 0% loss over
100 probes, with 0/2/6 ms minimum/mean/maximum RTT. The workload used
TinyLFU 512 MiB, prefetch depth 1, ctx 256, batch 2, two threads,
Prompt A and 64 generated tokens.

| Metric | Wi-Fi Full | Wired Full | Wi-Fi Greedy | Wired Greedy |
|---|---:|---:|---:|---:|
| Generation (s) | 31.682 | 28.725 | 35.108 | 32.001 |
| RTT minus Worker Stage (s) | 2.639 | 0.838 | 5.153 | 3.957 |
| Payload receive median (ms) | 14.042 | 5.690 | 53.835 | 48.949 |
| Payload receive min / p95 / max (ms) | 5.764 / 31.293 / 40.027 | 4.434 / 10.382 / 22.365 | 43.974 / 61.379 / 155.970 | 41.590 / 57.245 / 58.596 |
| Payload receive sum (s) | 1.122 | 0.469 | 4.052 | 3.657 |

On Ethernet, all 73 Greedy payload receives exceeded 40 ms and all
73 Full receives were below 40 ms. The Greedy Worker spent 5.377 ms
in header sends and 1.545 ms in payload sends, compared with 3,656.569
ms of Coordinator payload receive. The transport change reduced the
observed tail, while the plateau persisted. Wi-Fi is less likely to
be the primary cause. The shared endpoint TCP behavior, split writes
and SSH forwarding remain candidates. The trial did not include a
packet capture or an A-D comparison.

The supplied results report byte-identical generated IDs and text,
Worker Expert IDs for 24 layers, matching Pager counters and
unchanged application bytes. The archive phase2b10c-wired.zip was
unavailable in the workspace for independent review. Treat 28.725 s
as the provisional wired Full reference. Set the formal reference
using repeated, matched runs with detailed diagnostics kept separate.

## Reproduce on two Intel hosts

1. Build `qwen3moe-stage-tcp` and `test-backend-ops` in build-lan
   under UCRT64. Use objdump -p to list PE imports and inspect CMake
   for dynamically loaded backends. Distribute the exe and resolved
   runtime DLLs only. Save SHA-256 of each named file on both PCs.
2. On both PCs save `netsh wlan show interfaces`, `ipconfig /all`,
   `Get-NetAdapter`, `Get-NetAdapterStatistics`, OS version,
   `ssh -V` and `netstat -s -p tcp` before and after the trial.
   From PC-A run `ping -n 100 <PC-B IPv4>`. Record band, channel,
   PHY, signal, link rate, TCP retransmissions and loss where
   available. Redact SSID, addresses and hostnames before sharing.
   ICMP RTT and system TCP counters are separate from Stage metrics.
3. In administrator Windows PowerShell 5.1 on PC-B run
   `Start-Service sshd` and `Get-Service sshd`. Verify PC-A can
   log in using `ssh user@pc-b`, then start both loopback forwards.

~~~bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 \
  -L 127.0.0.1:50054:127.0.0.1:50054 user@pc-b
~~~

4. Use Ethernet plus SSH as the primary test path. Confirm the route and Ethernet link speed on both PCs. Start the model-free benchmark server on PC-B, wait for READY,
   then run the client on PC-A. The arguments following the port
   are pattern, NODELAY, coalescing, response delay in ms, warmup,
   measured samples, output path and connection label.

~~~bash
# PC-B
"$EXE" transport-bench server 50054 small 0 0 0 10 100 "$OUT/server.json" wired-ssh
# PC-A
"$EXE" transport-bench client 50054 small 0 0 0 10 100 "$OUT/client.json" wired-ssh
~~~

   Restart server per trial. Compare all four settings before
   changing one endpoint at a time. Run A first to look for the 40 ms plateau, then compare B, C and D in rotating repetitions. Use `full`, `full2`,
   `mixed-full` and `mixed-greedy` to cover real body sizes and
   the 73-request schedule. Mixed warmup and sample counts are
   schedule counts. Use a representative nonzero delay such as
   50 or 150 ms with 0 warmup and 1 schedule, and subtract the
   measured server sleep by request ID for diagnostic comparison.
   Raw results include local endpoints, directional bytes and
   client RTT histogram. Keep failed logs separately. No failed
   benchmark enters the successful JSON collection.
5. Use the Baseline, Worker and Coordinator commands in
   [Phase 2B-10B](phase2b-10b-worker-greedy.md) and its linked
   Phase 2B-10A procedure on port 50053. Use Worker Greedy at 256 MiB
   with Phase 2B-10D flags on both processes, and keep a Full
   reference. Begin with the control, then the most promising
   settings from the benchmark. Capture detailed diagnostics in
   separate runs. Check generated Token IDs and text with cmp.
6. Save run order, revision, model/binary hashes, CLI, logs,
   metrics.json, perf.csv, both benchmark JSONs and network
   snapshots. Use distinct prefixes and archive them together.
   If a Stage run fails, keep its logs and restart the Worker.
   Compare using analyze-stage-communication.py with `--connection
   wired-ssh`. Compare matching repetitions with
   analyze-stage-performance.py. Local transport settings are
   recorded separately for each role.

If packet capture is available, collect loopback traffic (e.g.
Npcap loopback capture) and physical traffic separately. Filter
loopback by tcp.port 50053 or 50054 and physical by the SSH
connection. Compare Header-to-Payload gap, the ACK around it and
retransmissions under A–D. Inspect the timing on each host with
its own clock. Encrypted SSH segments cannot identify Stage
headers or bodies directly. Keep existing firewall and SSH
settings. Wi-Fi and direct TCP are optional comparisons, and direct
listeners should only be used on an authorized isolated LAN.


## Local measurement notes

The native host was Windows ARM64 and the executable was UCRT64
x86-64 running under emulation. For 8 KiB requests and 40-byte
responses, three 100-request trials per setting yielded pooled p50
RTTs of 0.0603, 0.0621, 0.0643 and 0.0599 ms in A/B/C/D order.
Worker send calls were 600, 600, 300 and 300, respectively. These
counts refer to calls to send, not segments. With the 73-request
Greedy pattern and a 50 ms simulated compute delay, actual server
sleep p50 was 50.47–51.51 ms and paired RTT minus measured sleep
p50 was 0.193–0.239 ms. Payload receive did not show the 40 ms
plateau reported on the Intel Wi-Fi/SSH path.

Keep default A for compatibility. On Intel rehearse A against C
first to isolate the split-write hypothesis, then complete A–D
in rotating order. If behavior diverges from loopback, try
Worker-only coalescing and Worker-only NODELAY, followed by
Wi-Fi/SSH or isolated direct TCP when available.
