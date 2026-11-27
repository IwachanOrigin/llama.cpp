# Phase 2B-12A Expert Prefetch latency investigation

## Why measure Prefetch

The supplied Intel 11A-W run (a8cf45e24c40028b7383c1147d55481a67bb110f)
reports 30.052 s elapsed, 13.962 s of aggregate Prefetch wait and
2.759 s of measured overlap. Only 222 of 59,330 deliveries were
already ready (0.374%). The 59.35 GiB of requested bytes represents
Reader activity. Windows can satisfy the reads from its OS cache.
The source ZIP was unavailable for independent review.

The first unused projection took 824.232 ms; the following eight
averaged 16.702 ms. First-touch work can move to the final Prefill
when early projection is removed. Keep 11B pending and revisit for
longer prompts or larger batches.

## Confirmed software path

- src/llama-graph.cpp around 2375 creates selected_experts. Around
  2468–2490 and 2596, separate Up, Gate and Down matmuls share the
  selected IDs. Down consumes the gated product; readiness of its
  IDs does not establish a safe scheduler sequence across nodes.
- ggml/src/ggml-cpu/ggml-cpu.c around 1696: workers have built the
  token-to-expert row lists and crossed a barrier. Nonempty expert
  IDs are visited in increasing order. Worker 0 acquires the current
  lease, then issues only the next nonempty expert of this tensor,
  then publishes current weights at the barrier. Other workers
  execute the same expert. Worker 0 records compute through the
  final barrier and releases the lease. finish cancels/drains at
  tensor scope exit. IDs in a later Router are not yet known.
- src/llama-expert-cache.cpp try_prefetch samples current resident/
  loading state under the cache mutex; it does not update the
  frequency sketch. acquire performs demand-order TinyLFU admission
  and reserves the resident or transient slot before consuming a job.
- prefetch_loop waits on the cache condition variable. After taking
  a job it releases the cache mutex during Reader service. It
  reacquires it to publish readiness and wake waiters. Acquire
  waits through the same mutex/condition variable; it swaps the
  ready payload pointer and keeps the cache slot identity. Failure
  recovers through the demand Reader. finish drains an in-flight
  request before its job scope can be destroyed.
- src/llama-model.cpp:1449 and read_expert around 1534: key is
  (tensor_source_id, expert_id), offset is tensor base + id * stride.
  The model retains per-shard file handles and a per-shard mutex.
  The Reader holds that shard mutex across seek/read into a slot;
  Windows ReadFile is synchronous, with no OVERLAPPED request.
  Expert paging's direct-to-slot path uses buffered file I/O in
  this mmap configuration. FILE_FLAG_NO_BUFFERING is a separate
  loader setting and is incompatible with the external Expert
  path as configured. mmap ordinary weights and file-cache reads
  can coexist. There is no extra copy from the staging slot into
  resident storage.

~~~mermaid
sequenceDiagram
    participant W as Worker 0
    participant R as Prefetch Reader
    Note over W: Row list available after MUL_MAT_ID grouping
    W->>W: Acquire current Expert and apply demand policy
    W->>R: Submit next same-tensor Expert
    W->>W: Publish current lease, compute across workers
    R->>R: Take job, unlock cache, read under shard mutex
    par Computation
      W->>W: End compute barrier, release, acquire next
    and Completion
      R->>W: Lock cache, publish ready, notify
    end
    W->>W: Reacquire lock if waiting, swap payload and continue
~~~

## Instrumentation and limitations

Use existing --expert-trace-limit 200000 and
--expert-io-trace-limit 200000 with --perf-details. Depth 1 adds
a bounded .prefetch-timing.jsonl sidecar. Old .expert-io.jsonl
and .prefetch-delivery.jsonl formats remain unchanged. Its index
is the demand event index, followed by job ID, submit, start,
end, ready, demand entry, wait begin/end, and previous expert's
compute begin/end. All timestamps use the local steady clock.
Zero wait timestamps mean no condition wait. The timing sidecar
covers successful delivered jobs; failed/cancelled work remains
accounted in Reader traces and counters. It does not timestamp
Router readiness, provider entry/exit around acquire, release
or cancelled jobs. Queue includes notification and cache-mutex
reacquisition. Publish includes cache-mutex reacquisition and
overlap/counter bookkeeping. Resume includes wakeup and lock
reacquisition. Compute includes the completion barrier.

Reader service from start to end includes the provider and its
trace callback. Provider I/O trace divides provider service,
shard mutex wait, and Win32 read calls; calls are not necessarily
physical storage access. Demand wait and compute intervals
carry their own boundaries. Overlap is per job with its previous
expert, and the reported background Reader utilization divides
its service sum by Stage execution wall time for this one-reader,
single-context run. Do not add these intervals to Stage time.
With I/O tracing off there are no extra clocks or trace allocation.
With it on, per-demand delivery record storage is bounded by the
Expert trace limit. The analyzer rejects trace drops, missing
deliveries and bad interval ordering.

## Replay and design evaluation

The analyzer replays sequential demand TinyLFU using its existing
policy simulator. It opens the demand list only within a single
tensor run within the same Stage request. It schedules later
nonresident keys after current demand, checks current cache state,
uses a bounded staging horizon and never reorders demand. Reader
service and prior-expert compute use matched demand samples when
available and medians otherwise. It omits unrelated Stage graph time and runtime dispatch
jitter. Compute-only is an ideal envelope requiring costless
readiness. The two-reader cases assume unchanged individual service
time, ignoring bandwidth, same-shard serialization and OS policy,
so they are sensitivity cases. Actual latency of two readers sharing
the current seek/read handle would still serialize.

For equal-budget estimates reserve one maximum-size payload per
staging job and approximate additional management at 64 B/job,
retaining the existing metadata estimate. Equal-residency should
be assessed separately by raising the total budget. Q4_K and Q6_K
slices are 884,736 B and 1,290,240 B respectively in the target.
Each source has contiguous expert IDs in one GGUF tensor, while
neighboring candidates are often nonadjacent. Do not coalesce
distant requests without measuring byte amplification. At the
observed 206/414 residents for 256/512 MiB Depth 1, Depth 0 has
207/415 as a layout estimate, Depth 2 205/413, Depth 4 203/411
and Depth 8 199/407. Actual future structures and alignment may
alter these counts.

A: place submission earlier within ggml_compute_forward_mul_mat_id
once IDs are grouped, while keeping the one-slot scope and measuring
the first-demand conflict. B: bounded multi-slot queue in
llama-expert-cache.cpp, reducing residents at fixed budget and
preserving acquire-driven admission. C: optimize Reader service or
notifications only after measuring queue, read and wake components.
D: independent per-shard positional handles or OVERLAPPED I/O would
require a separate ownership, cancellation and memory design with
real concurrent benchmarks. E: adjacent tensor lookahead needs
certified scheduler order, shared Top-K and safe drain handoff, and
must preserve the exact demand sequence. F: larger resident working
set can reduce reads without timing changes; compare at matched
budget and model memory pressure. None of these changes are enabled
in runtime by this investigation.

## Intel Ethernet + SSH measurements

Build in UCRT64 on PC-A, reusing the 10F Stage invocation and 10D
deployment and SSH instructions. Inspect PE imports and hash each
resolved DLL and executable; retain the prior package. Use separate
PC-A build-lan/deploy-12a and PC-B /c/moe-worker/bin-12a packages.
Use the existing TCP port 50053 loopback tunnel to PC-B (172.17.67.37).
Use revision and binary hashes from the same build, and collect logs
and .metrics.json, .perf.csv, .experts.jsonl, .expert-io.jsonl,
.prefetch-delivery.jsonl and .prefetch-timing.jsonl as present.
For each L0/L1 trial use a new directory under
/c/moe-stage-test/phase2b12a-wired and both Stage configs below.
Run Baseline on PC-A, then start Worker on PC-B and wait for READY. Run Coordinator on PC-A. Leave projection diagnostics off.

~~~bash
MODEL=/c/work/models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf
PROMPT=/c/moe-stage-test/phase2b6-intel/user.txt
DEPTH=1 # Set 0 for L0 on both hosts
common=(--ctx 256 --batch 2 --threads 2 --pager-policy tinylfu
        --pager-budget-mib 512 --expert-prefetch-depth "$DEPTH"
        --expert-trace-limit 200000 --expert-io-trace-limit 200000
        --perf-details --perf-cache-state uncontrolled
        --accept-timeout-ms 90000 --response-timeout-ms 300000)
transport=(--tcp-nodelay 0 --tcp-small-frame-coalesce 1)
gen=(--generate --chat --prompt-file "$PROMPT" --max-new-tokens 64)
# PC-B
"$EXE" worker "$MODEL" 24 48 50053 mmap "$OUT/worker" auto "${common[@]}" "${transport[@]}" >"$OUT/worker.ready" 2>"$OUT/worker.log"
# PC-A
"$EXE" baseline "$MODEL" 24 48 0 mmap "$OUT/baseline" auto "${common[@]}" "${gen[@]}" >"$OUT/baseline.log" 2>&1
"$EXE" coordinator "$MODEL" 24 48 50053 mmap "$OUT/coordinator" auto "${common[@]}" "${transport[@]}" "${gen[@]}" --logits-response worker-greedy >"$OUT/coordinator.log" 2>&1
cmp "$OUT/baseline.generated_tokens" "$OUT/coordinator.generated_tokens"
cmp "$OUT/baseline.generated.txt" "$OUT/coordinator.generated.txt"
~~~

Set EXE to the deployed executable and create OUT on each machine
before starting. Start Worker after Baseline if the Baseline would
exceed the Worker accept timeout. Collect both stages under the same
trial directory on PC-A. Repeat L1 without --expert-io-trace-limit
when measuring diagnostic perturbation. Analyze each role independently.

~~~bash
python tests/moe-paging/distributed/analyze-expert-prefetch-latency.py \
  --run "$OUT" --role worker --gguf "$MODEL" --output "$OUT/worker-latency.json"
python -m zipfile -c phase2b12a-wired.zip "$OUT"
python -m zipfile -t phase2b12a-wired.zip
~~~

## Local result and priority for 12B

The 256 MiB, Depth 1, 64-token real Qwen3 test captured 90,399
Baseline, 45,258 Coordinator and 45,141 Worker demands with no
drops. The run was Windows ARM64 using x86-64 emulation with
loopback TCP. The 16-token same-build I/O-trace ON/OFF pair
and the 64-token retained 10D comparison matched generated bytes,
Expert IDs/sequence, counters and wire bytes. Ready/Wait and
timing are scheduling dependent.

| 64-token diagnostic | Coordinator | Worker |
|---|---:|---:|
| Reader queue p50/p95 (us) | 3.10 / 8.50 | 3.30 / 8.60 |
| Reader service p50/p95 (us) | 166.70 / 1653.70 | 163.70 / 1656.20 |
| Available lead p50/p95 (us) | 638.20 / 831.50 | 675.10 / 827.20 |
| Publish p50/p95 (us) | 0.20 / 0.40 | 0.10 / 0.50 |
| Resume when waiting p50/p95 (us) | 12.60 / 19.50 | 13.00 / 19.20 |
| Previous compute p50/p95 (us) | 616.40 / 800.10 | 660.10 / 800.80 |
| Ready / consumed | 23088 / 33569 | 23747 / 33409 |
| Prefill wait / overlap (s) | 2.299 / 2.928 | 2.293 / 2.952 |
| Decode wait / overlap (s) | 4.786 / 6.126 | 4.235 / 6.287 |

Provider service for all reads was 18.192 / 17.843 s; Win32 API
intervals accounted for 18.092 / 17.743 s, while shard-lock waits
totaled about 0.008 s on each role. The API interval cannot
distinguish cached copying and physical I/O. Prefill had higher
service p50 (286 / 260 us) than Decode (158 / 156 us), although
Decode contributed more total wait. The local ready share (~69% /
71%) differs sharply from the supplied Intel share of 0.374%;
collect the same timestamps on Intel before choosing a larger
queue or inferring what accounts for Intel's lower ready share.

| Modeled depth / Readers | Residents | Coordinator wait (s) | Worker wait (s) |
|---|---:|---:|---:|
| 0 / 1 | 207 | sync reads | sync reads |
| 1 / 1 | 206 | 7.261 | 6.669 |
| 2 / 1 | 205 | 4.786 | 4.358 |
| 4 / 1 | 203 | 4.092 | 3.755 |
| 8 / 1 | 199 | 4.079 | 3.745 |
| 2 / 2 | 205 | 2.997 | 2.636 |
| 4 / 2 | 203 | 1.888 | 1.669 |

Depth 1 replay matched read counts of 37,740 / 37,846 and
modeled wait within roughly 3% of measured 7.086 / 6.529 s.
That agreement applies to this trace and does not establish
cross-host or higher-depth predictive validity. In a single
Reader replay, Depth 2 captures most of the incremental wait
reduction, then improvement flattens from Depth 4 to 8. At Depth
8 modeled reads rise to 38,310 / 37,924. The two-Reader sensitivity
requires independent I/O and unchanged service latency; it is
not evidence for a two-Reader rollout.

For 12B, collect Intel L0/L1 first. Test a one-slot earlier
submission prototype at the known grouped-ID boundary and
control for conflict with the current demand. If the service
tail continues to exceed lead time, compare one Reader at Depth
2 under equal-budget and equal-resident controls. Retain the
single-reader architecture during that experiment. Freeze
demand order and require exact TinyLFU decisions at equal resident
count, no leaked pins, bounded slot accounting and no regressions
under failure, cancel or tensor transition. Use real Intel traces
to decide whether Depth 4 or separately owned positional I/O
deserves follow-up. Do not use the modeled times as Intel speed
estimates.

The real GGUF inventory has 120 Q4_K and 24 Q6_K sources.
The same-expert gate/up/down slices are never adjacent in the
6,144 checked triplets. Their contiguous covering span is about
86x the required bytes at the median. This supports retaining
individual reads for cross-tensor candidates rather than merging
them into one physical range.

### 256 MiB equal-budget Depth 0 check

The single 64-token diagnostic runs took 124.578 s at Depth 0
and 109.672 s at Depth 1. At Depth 0 the two distributed stages
had 207 resident slots each versus 206 at Depth 1. Managed bytes
per distributed stage were 268,401,551 and 268,401,807 respectively.
The Depth 0 read counts were 37,697 / 37,815, consistent with the
replayed counts. Depth 1 had 37,740 / 37,846. Stage wall times
were 60.982 / 63.488 s at Depth 0 and 52.274 / 57.293 s at Depth 1.
Under uncontrolled cache conditions these single runs describe the
tested environment. They do not isolate the effect of Prefetch
from the changed resident count or run-to-run timing.

The earlier archived 512 MiB runs reported 415 residents at Depth 0
(536,792,207 managed bytes) and 414 at Depth 1 (536,792,463 managed
bytes), each with one transient slot. The Depth 1 staging slab held
1,290,240 payload bytes and 320 management bytes. The new timing
record array uses up to 14,400,000 diagnostic bytes for 200,000
access slots per role, accounted separately from the Pager budget.
No larger runtime Prefetch depth or extra Reader was implemented.
