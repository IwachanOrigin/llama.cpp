# Phase 2B-8A External Expert Pager capacity evaluation

## What a hit means

In src/llama-model.cpp the model builds a source per Expert tensor, holding
the shard, byte offset, Expert stride and count. Each layer's gate, up and
down weights are different sources. src/llama-expert-cache.h defines the
cache key as (tensor_source_id, expert_id). IDs are local to the model, so
use the source names to compare between Stage processes.

In llama_expert_cache::acquire, a READY entry can be pinned and moved to
MRU. A miss reserves an empty slot or evicts the oldest eligible slot.
LOADING, pinned or waiter-protected slots cannot be evicted. A lease keeps
the slot pinned through the CPU operation and release makes it eligible
when its reference count reaches zero. Failed reads never publish READY
data. All slots fit the largest Expert stride in the Stage. The effective
slot count includes allocation and provider overhead and should be read
from the metrics. Use it for reuse-distance comparisons.

The Stage's ordinary weights use the selected mmap or buffered loader.
External Expert weights go through read_exact_at on a shard handle in
both modes. The operating system can satisfy those calls from its own
file cache. The reader counters measure application requests.

The CPU MUL_MAT_ID loop groups token rows by Expert for a given weight
tensor. One request can therefore service several token rows in a prefill
batch. Decode at top-8 with 48 layers and 3 source tensors per layer uses
up to 1,152 different keys per token. Each 24-layer Stage has up to 576.
The slot limit has to be compared with the number of distinct intervening
keys between two visits to the same source and Expert. A token count or
the Expert ID alone does not predict reuse. With serial, successful Stage
execution, a reuse distance d hits an LRU cache with C slots when d < C.

## Timing runs on Intel PC-A and PC-B

Use the same build-lan executable and model revision on both hosts.
In UCRT64 Bash, choose a fresh run directory and set the same budget on
all three processes. Start with ascending one-run checks at 64, 128,
256, 512 and 1024 MiB. A run counts only if both peers finish successfully
and both generated files match the Baseline. Check available RAM, commit
headroom, per-process Working Set and paging on both hosts before going
higher. Stop increasing capacity if RAM or commit is insufficient or if
repeat observations show significant paging. OS hard faults can include
mapped files. Reader byte counters cannot diagnose pagefile activity.

Once smoke checks pass, perform three repetitions per safe budget.
Interleave capacities, reverse or rotate the order between rounds, and
record the actual order in runs.txt. Give each repetition a fresh output
directory. Label the cache uncontrolled unless the procedure establishes
another condition. Do not clear system caches in the runner. Use the
requested Japanese user prompt in UTF-8 user.txt with no System message.

~~~bash
cmake --build build-lan --target qwen3moe-stage-tcp --parallel 4
stage=./build-lan/bin/qwen3moe-stage-tcp.exe
model=/c/moe-stage-test/models/Qwen3-30B-A3B-Q4_K_M.gguf
budget=64
trial=01
dir="results/budget-$budget/run-$trial"
mkdir -p "$dir"
common=(--ctx 256 --batch 2 --threads 2 --pager-budget-mib "$budget" \
  --accept-timeout-ms 240000 --frame-timeout-ms 30000 --response-timeout-ms 300000 \
  --perf-details --perf-cache-state uncontrolled)
gen=(--generate --chat --prompt-file user.txt --max-new-tokens 64 --sampling greedy)
~~~

On PC-A run Baseline to completion:

~~~bash
"$stage" baseline "$model" 24 48 0 mmap "$dir/baseline" auto \
  "${common[@]}" "${gen[@]}" 2>"$dir/baseline.log"
~~~

On PC-B run Worker and wait for READY:

~~~bash
"$stage" worker "$model" 24 48 50053 mmap "$dir/worker" auto \
  "${common[@]}" >"$dir/worker.ready" 2>"$dir/worker.log"
~~~

On PC-A keep the SSH loopback forwarding open, then run Coordinator:

~~~bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 user@intel-pc-b
~~~

~~~bash
"$stage" coordinator "$model" 24 48 50053 mmap "$dir/coordinator" auto \
  "${common[@]}" "${gen[@]}" 2>"$dir/coordinator.log"
cmp "$dir/baseline.generated_tokens" "$dir/coordinator.generated_tokens"
cmp "$dir/baseline.generated.txt" "$dir/coordinator.generated.txt"
~~~

Collect Worker JSON, CSV and log under the matching PC-A run directory.
Record all attempted directories in actual execution order, including
failures. For Windows-native Python, convert paths with cygpath.

~~~bash
args=()
while IFS= read -r run; do args+=(--run "$(cygpath -m "$run")"); done < runs.txt
python tests/moe-paging/distributed/analyze-pager-capacity.py \
  "${args[@]}" --output "$(cygpath -m results/capacity-summary.json)"
~~~

The script uses Phase 2B-7 validation, rejects traced samples from timing
statistics, records excluded attempts, and returns nonzero if there are
exclusions. Successful runs are grouped by budget and role and report
median, min, max and differences from the smallest measured budget.
All generated bytes must agree across budgets. Record binary and GGUF
hashes and environment notes alongside the result. Startup auto hashes
read the model before inference and may influence cache residency.

## Diagnostic trace

Use a separate run and fresh prefixes. Append --expert-trace-limit 200000
to each role. The callback stores events in a preallocated bounded array.
A JSONL trace uses the existing version-1 record format and adds
request ranges in its header. A successful run publishes the trace and
a truncation count; truncated traces are rejected by the reader. Since
Worker does not know phase, pair its trace with the Coordinator CSV
from the same session. The CSV schedule supplies the phase. The analyzer
checks serial LRU at the observed capacity against every hit/miss in the
trace, and reports unique keys and bytes per request, per-source Expert
counts, layer and weight-type totals, reuse distances and misses that
would have hit absent eviction since the preceding use.

~~~bash
d=$(cygpath -m "$dir")
python tests/moe-paging/distributed/analyze-pager-capacity.py \
  --trace "$d/worker.experts.jsonl" --coordinator-csv "$d/coordinator.perf.csv" \
  --output "$d/worker.reuse.json"
~~~

## Candidate policies for 2B-8B

These are experiments to prioritize after comparing traced access
sequences. None requires changing routing, accumulation order, weight
bytes or quantization. Verify token and text equality after any change.

| Candidate | Effect to test | Memory / CPU cost | Integration and priority |
|---|---|---|---|
| Current LRU | Use the measured capacity knee as a baseline | Existing budget / existing lookup | Existing implementation; first control |
| Layer-local partitions | Protect reuse across successive tokens from pressure by other layers | Same aggregate slots / small per-layer counters and allocation | Can retain leases and reader; first alternative when global reuse distance is too large |
| Frequency-based admission | Keep frequently reused keys and bypass one-use keys | Bounded frequency table and statistics / per-access update | Can preserve existing reader and lease interfaces; compare with held-out generations before deployment |
| Cross-decode reuse | Retain selected keys until the next token where useful | Same budget for selective retention, more for a full working set / track repeated keys | Evaluate by layer and source; test together with partition/admission |
| Group gate/up/down reads | Reduce reader calls or seeks when requests are nearby | Buffer for grouped slices or slots / grouping and range checks | Depends on GGUF offsets and file layout; higher complexity, profile after replacement-policy experiment |
| Prefetch selected Experts | Hide some synchronous reader waiting | Bounded staging/pins / scheduling and cancellation | Preserve routing and compute order; only investigate after measuring useful lead time and queue pressure |
| Capacity-aware eviction | Adapt to capacity and access skew | Bounded policy metadata / selection overhead | Higher complexity; compare to the simpler partition/admission baselines first |

The existing analyze-expert-trace.py can also replay layer partitions,
selective admission and Belady for non-Worker traces. Belady is an offline
upper bound on hits for mandatory admission; the frequency ranking uses
the whole observed trace and can overstate online gains. Compare hit rate
and requested bytes separately from measured Stage and generation times.


## Local trace result

The real-model 16-token smoke run used 19 prompt tokens in batches of two,
followed by 15 extra decode steps. At 64 MiB the full model held 51 slots
and the two Stages held 52 each. The serial trace and observed counters
agreed at every lookup. Minimum reuse distances were 1,146 for Baseline
and 570 for both Stages. The first observed hits would need 1,147 and
571 slots respectively. For the recorded 1,290,240-byte maximum slice,
the raw slot payloads alone are about 1,411 and 703 MiB. Actual admission
also needs Pager metadata and other process memory.

Each additional decode needed 1,152 unique full-model keys (1,097,072,640
bytes) or 576 per Stage (548,536,320 bytes). At 64 MiB, Baseline decode
had 14,826 repeat accesses that missed after eviction. The two Stages
had 7,584 and 7,242. First-use misses were counted separately.

The first next experiment is to measure the per-Stage global-LRU transition
around the observed 571-slot threshold, with actual slot counts and RAM
headroom recorded. Equal partitioning produced zero simulated hits with
416 total slots or fewer on this access sequence. Compare it again at
larger budgets, and evaluate frequency-based admission on held-out
prompts. The frequency oracle is a shortlist tool, since it sees future
workload frequency. Pursue read grouping or prefetch only after the
reader and Stage timings have been compared over repeated runs.

At 832 total slots per Stage, decode replay gave 3,805 and 3,866
global-LRU hits for Coordinator and Worker. Equal layer partitions
gave 3,908 and 3,816. The offline frequency oracle gave 4,391 and
3,909. The layer partition improvement is not consistent across Stages.
The production policy remains LRU.

On the local host, budgets 64 through 512 MiB completed as full
Baseline/Coordinator/Worker triplets. Baseline also completed at
1024 MiB. The local two-process 1024 MiB test was skipped because
available commit headroom was about 3,529 MiB, below the local test
guard of 4,096 MiB (two budgets plus 2 GiB for ordinary weights,
overhead and headroom). This does not measure the two-host capacity.
