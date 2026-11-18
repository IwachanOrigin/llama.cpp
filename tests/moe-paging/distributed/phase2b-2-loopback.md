# Phase 2B-2 loopback stage execution

The CPU-only test executable uses the Phase 2B-1 stage loader. The coordinator
loads layers [0, split) and the worker loads [split, layers). Each keeps its
own Context, KV and Expert Pager (8 MiB budget in this fixture program).
The baseline role is a third, separate invocation which exits before the
two staged processes are launched. The workload is fixed to tokens
1, 19, 7, 42, 51, sent as a three-token prefill and two one-token decode steps.
This is a validation tool with fixed Context and batching settings.

## Reproduce on Windows

Build in a configured CPU-only test build with UCRT64 executables on PATH.

~~~powershell
cmake --build build --target qwen3moe-stage-tcp
ctest --test-dir build -V -R '^test-qwen3moe-stage-tcp$'
~~~

The test uses the existing fixture generator and GGUF splitter. To preserve
logs, supply generated fixtures and a new output directory.

~~~powershell
python tests/moe-paging/distributed/test-stage-tcp.py build/bin/qwen3moe-stage-tcp.exe model.gguf split-00001-of-00002.gguf run-logs
~~~

To start the same tool manually, run these commands in separate terminals.
Use port 0 on the worker for OS assignment and copy the printed READY port
to the coordinator. A nonzero worker port is also supported.

~~~powershell
build/bin/qwen3moe-stage-tcp.exe worker model.gguf 1 2 0 mmap worker-out -
build/bin/qwen3moe-stage-tcp.exe coordinator model.gguf 1 2 50053 mmap coordinator-out -
~~~

Replace 50053 with the READY port. The listener and connector use IPv4
127.0.0.1. The optional final identity label (or dash for metadata-only
mode) is a SHA-256 digest string supplied by the caller. The harness hashes
the size and contents of each shard in order. When supplied, both sides
must agree on this label. The C++ tool does not rehash the GGUF. Metadata
agreement alone does not establish weight identity.

## Protocol v1

Each 24-byte header contains six little-endian u32 fields in this order.
Magic 0x53454f4d, version, type, body byte length, session ID and request ID.
Frame bodies are limited to 16 MiB. The per-frame deadline applies across
partial reads or writes. A single synchronous request may be in flight.

| Code | Message | Body and state |
|---|---|---|
| 1 | HELLO | Architecture tag, total layers, prefix end, suffix begin, suffix end, hidden size, vocabulary size, F32 tag, effective context size, batch limit and hash label length as u32, then label bytes. Both peers verify equality. Session and request are zero. |
| 2 | OPEN_SESSION | Empty with zero IDs. Worker assigns a nonzero session ID and acknowledges request zero. |
| 3 | EXECUTE_STAGE | u32 token count, hidden size, sequence ID (zero), F32 tag (one), then count u32 absolute positions and count times hidden size IEEE-754 binary32 values. Request IDs start at one and increase by one. |
| 4 | STAGE_RESULT | u32 status (zero), row count, vocabulary size and row-major logits as binary32. Echoes the request and session IDs. |
| 5 | CLOSE_SESSION | Empty, using the next request ID. Worker acknowledges and exits. |
| 6 | ERROR | Diagnostic text followed by termination. |
| 7 | ACK | Empty, with the assigned or echoed session ID and request ID. |

Positions must start at zero and continue without gaps. The worker validates
the complete activation frame before calling decode. Worker input owns the
decoded floats through decode and synchronization. The coordinator serializes
the boundary into an independent frame before another local decode can
overwrite it. An error or ambiguous disconnect abandons both contexts.
Each process releases its local resources on exit. A fresh process pair can
open a new session without prior KV. The explicit worker fault options
after-receive and after-decode close the connection at those test points.

Per-frame logs show header and body bytes. The totals count bytes supplied
to or obtained from the socket, including control frames, excluding TCP/IP
headers. Round-trip times include Worker execution. The comparison harness
checks finite logits and boundary data and reports maximum absolute and
relative error if bytes differ. It also checks Expert ID recordings and
Stage inventories. Worker stdout provides READY; stderr holds individual
protocol and computation logs.

Physical hosts, POSIX execution, authentication and encryption are later work.
The POSIX and Winsock paths are isolated in stage-tcp.h. For a trusted LAN
prototype, add explicit peer and bind configuration, authenticated model
identity and bounded startup suitable for the target machine. Validate
Windows-to-Windows correctness and resource use before mixed-OS runs.

