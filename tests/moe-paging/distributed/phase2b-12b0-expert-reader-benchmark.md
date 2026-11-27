# Phase 2B-12B-0 Expert Reader benchmark

## Scope and baseline

Intel 12A-W (revision 58885b67d6670e91f8af71f8d705b6660ec49bab)
reported Depth 0/1 at 30.267/29.986 s across the two stages. At
Depth 1, service p50 was 166.2/171.8 us and lead p50 was 49.0/61.7
us. Win32 file API time made up 99.04%/98.04% of provider time.
That identifies the API interval to investigate; the calls can be
satisfied by the file cache. These are supplied aggregate values.
phase2b12a-wired.zip was unavailable for independent inspection.

The production provider retains a llama_file per GGUF shard.
llama_external_experts::read_expert (src/llama-model.cpp) takes
that shard's mutex and calls tensor_source::read, which translates
the key to an absolute offset. llama_file::read_exact_at
(src/llama-mmap.cpp) bounds the range, seeks, reads directly to
the slot and checks the final position. On Windows the buffered
path uses synchronous ReadFile and the file's shared position.
The production loader disallows unbuffered direct I/O with external
Experts. The benchmark's shared mode uses the same llama_file
operations and lock scope; the provider/cache lookup, leases and
admission are outside the benchmark.

## Implementation

The existing qwen3moe-stage-tcp executable has a model-free
reader-bench subcommand. It parses GGUF metadata with no tensor
allocation and validates name, Expert ID, tensor stride, absolute
offset, actual shard size and every selected request range.
The offline converter uses the existing Expert I/O validator
to reconcile real reads, demand misses, delivery IDs and offsets.
It replays actual successful Reader calls in provider-entry order,
with file-emission order as the stable tie-break. Cache Hits do
not become reads. Prefill/Decode is joined via demand/Stage request
IDs. Unconsumed async entries have unknown demand phase. Prefill
and Decode are selectable. L0/v1 and L1/v2 are supported, including
multi-shard models specified in original shard order.

| Mode | Method |
|---|---|
| shared | One current llama_file per shard, one mutex per shard |
| independent | Each lane owns one llama_file per shard and has independent file position |
| overlapped | Windows read-only FILE_FLAG_OVERLAPPED, one file per shard, explicit 64-bit offset and one event per lane |
| mapping | One persistent read-only llama_mmap(file, prefetch=0) per shard, direct memcpy into destination |

R0 and R1 both use the same synchronous file operations for one
lane; the shard mutex overhead is expected to be small. For
OVERLAPPED each lane keeps the operation, event and destination
alive until immediate or pending completion has been checked.
On a timeout the thread requests cancellation with CancelIoEx
and drains the operation with GetOverlappedResult(TRUE), including
the race where it has completed. Any failure stops new dispatch;
remaining lanes finish or time out and are joined before buffers
and handles are released. The process returns nonzero. This is
one waiter thread per outstanding request, separate from an IOCP
or single-thread multiplexing design. A zero timeout exercises
the immediate poll path and may race with completion.

Windows mapping uses PAGE_READONLY/FILE_MAP_READ via llama_mmap,
whole shard per view. This avoids window-granularity arithmetic
and measures persistent mappings, with one memcpy per read.
An in-page fault due to a modified/truncated backing file can
terminate the mapping process; use immutable read-only inputs.
Virtual mapped bytes and Working Set/Private Commit are separately
reported. Peak WS is a process lifetime peak; reported page faults
are not Hard Faults; process read counters are not SSD traffic.

The queue is a bounded pull schedule over a finite manifest.
One lane owns one aligned destination. There are at most N
active service intervals for N lanes; the input backlog is
not an OS queue. Latency uses one lane with one request at a time.
Throughput begins with the whole selected backlog available;
its queue and end-to-end times are relative to trial release.
Paced uses the observed provider-entry relative schedule as a
release-time proxy. It does not reproduce Router, demand
priority, CPU scheduling or Prefetch submission timestamps.
Reported peak active requests can include shard-mutex wait and
does not measure simultaneous physical I/O.

Verify mode compares every byte against the current synchronous
Reader through a separate reference handle/buffer per lane.
Measure mode avoids that work and explicitly reports verified=false.
Buffers are 64 B aligned, prefaulted and bounded by lanes times
aligned maximum slice (double in verify mode), plus 63 B alignment
headroom. The manifest, results and optional detail also have
fixed record caps. For the 1,290,240 B maximum slice, future Depth
2/4 would consume about 2.46/4.92 MiB of payload and reduce the
256/512 MiB estimates to 205/413 or 203/411 residents. These
exclude implementation-dependent new metadata; benchmark memory
is separate from Pager allocation.

## Local and Intel commands

Build with MSYS2 UCRT64.

~~~bash
cmake --build build-lan --target qwen3moe-stage-tcp --parallel 4
~~~

Run on each PC independently. Deploy the existing qwen3moe-stage-tcp
binary into build-lan/deploy-12b0 on PC-A and /c/moe-worker/bin-12b0
on PC-B. Inspect PE imports with objdump -p; resolve runtime DLLs
from the build runtime or /ucrt64/bin. Do not search old deploy
directories. Preserve earlier packages. Save SHA-256 of the exact
exe and resolved libraries and compare both PCs. No SSH or Stage
port is used.

On each host set EXE to the deployed executable. Use the actual
local L1 directory and GGUF. PC-B needs its Worker trace plus
Coordinator metrics/CSV for request phase and session correlation.
Keep original trace files out of the result archive; the manifest
records the origin SHA-256 and per-request slice data. Retain the
full manifest with its generated JSON. Use distinct destinations
for each host and trial.

~~~bash
MODEL=/c/work/models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf
TRACE=/c/moe-stage-test/phase2b12a-wired/L1
OUT=/c/moe-stage-test/phase2b12b0-reader/first-pass
KIT=tests/moe-paging/distributed
# PC-A, then separately PC-B with its own EXE and original trace folder.
bash "$KIT/run-expert-reader-bench.sh" PC-A "$EXE" "$MODEL" "$TRACE" "$OUT" verify
bash "$KIT/run-expert-reader-bench.sh" PC-A "$EXE" "$MODEL" "$TRACE" "$OUT" screen
# On PC-B substitute PC-B for PC-A in both commands.
~~~

The script refuses to reuse a result directory. It saves commands,
cache label, exit status and binary hashes. For L0 use the L0 input.
For phase or synthetic variants:

~~~bash
python "$KIT/prepare-expert-reader-bench.py" --gguf "$MODEL" --run "$TRACE" \
  --role worker --phase decode --output "$OUT/worker-decode.json"
python "$KIT/prepare-expert-reader-bench.py" --gguf "$MODEL" \
  --pattern scattered --samples 1024 --output "$OUT/scattered.json"
"$EXE" reader-bench "$OUT/worker-decode.json" shared 1 latency 1024 1024 \
  warm-subset measure "$OUT/latency.json" 30000
~~~

CLI position order is MANIFEST MODE READERS WORKLOAD LIMIT WARMUP
CACHE verify|measure OUTPUT TIMEOUT_MS [DETAIL_LIMIT]. LIMIT=0
runs the entire manifest. Warmup is a count of requests and
cycles if longer than the selected list; warm-subset denotes a
selected repeated prefix, warm-repeat the repeated full schedule.
Neither guarantees residency, and uncontrolled does not mean cold.
For normal performance use detail limit 0. For request diagnostics use
a capped nonzero limit. Results include timing distributions, bytes,
thread/buffer/mapping metrics, process CPU and memory data and
the manifest SHA-256. Paced replay uses provider-entry timestamps; precise issue/demand scheduling is deferred.

Rotate trial order after screening. Repeat baseline and two or
three finalists three times under the same trace, warmup, workload
and limits. Compare one host/role per report.

~~~bash
python "$KIT/analyze-expert-reader-bench.py" run1.json run2.json run3.json --output summary.json
python -m zipfile -c phase2b12b0-results.zip "$OUT"
python -m zipfile -t phase2b12b0-results.zip
~~~

Collect tail latency and demand behavior before deciding. The local
benchmark data can provide per-size service distributions (or
paired request samples) for the 12A simulation, but CPU/memory
contention and warm state must be modeled or observed separately.
The simulation was not changed. For 12B-1 consider an isolated
Windows Reader backend only after two-host repeatability, CPU-load
measurements and timeout/cancellation validation. No Pager Reader,
cache policy or default Prefetch behavior was changed.

## Windows ARM64 local results

The x86-64 UCRT64 executable ran under emulation. The local 12A
L1 manifests contained 37,740 Coordinator and 37,846 Worker calls
(about 33.62 and 33.61 GiB requested). All eight mode/count
combinations completed the entire trace in initial uncontrolled
screening. A separate 256-call byte check covered all modes and
real offsets above 4 GiB. The full Worker trace (36,087,054,336 B)
was byte-checked in separate two-lane OVERLAPPED and mapping runs.
A zero-timeout poll of the full Coordinator input produced a
timeout/drain and the subsequent verification run succeeded.
A four-lane 256-call detailed verification observed 72 adjacent
request-index inversions in completion order while all data matched.

These throughput runs were rotated by trial order with a 1,024-call
warmup and a 1,024-call measured prefix. The table reports medians
of three trials on the Worker trace.

| Mode / lanes | MiB/s | Service p50 / p95 / p99 (us) | Peak WS (MiB) |
|---|---:|---:|---:|
| shared / 1 | 8554 | 102 / 169 / 230 | 90.6 |
| independent / 1 | 8639 | 102 / 164 / 208 | 90.6 |
| independent / 2 | 10337 | 171 / 288 / 352 | 90.6 |
| overlapped / 1 | 8591 | 103 / 159 / 224 | 90.6 |
| overlapped / 2 | 10733 | 166 / 285 / 356 | 90.6 |
| overlapped / 4 | 12749 | 278 / 522 / 680 | 90.6 |
| mapping / 1 | 33354 | 25 / 40 / 60 | 1008 |
| mapping / 2 | 55922 | 27 / 47 / 61 | 1010 |

Single-request warm-subset latency p95 for shared, independent,
OVERLAPPED and mapping was 173.1 / 173.7 / 155.3 / 40.8 us in the
512-request, one-lane Worker check. Mapping's first measured call
after warmup is not a first-touch measurement. Its whole-file
mapping reserved about 17.28 GiB of address space; the full-trace
mapping peak exceeded 5.2 GiB WS in the screening. In one warm
Worker repeat, mapping/2 initialization took 0.177 ms and its
Private Commit was 120.4 MiB versus 84.6 MiB for shared/1. The
same repeat reported user/kernel CPU time of 15.625/15.625 ms for
mapping/2 and 78.125/187.500 ms for OVERLAPPED/4. Process time
granularity limits interpretation of such short trials.

Keep shared/1 as control. Prioritize mapping/1 for Intel's
tail-latency check with working-set limits, and independent/2
and OVERLAPPED/2 for the throughput/latency tradeoff. Higher
OVERLAPPED concurrency raised tail latency on the warm subset.
Record per-phase latency, memory pressure and performance under
compute contention before selection. The uncontrolled screening
order and ARM emulation limit conclusions about Intel ranking.
