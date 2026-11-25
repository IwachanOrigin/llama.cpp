# Phase 2B-9B Expert prefetch

## Demand and prefetch paths

Stage TCP accepts --expert-prefetch-depth 0 or 1. The default is 0.
The setting is local to each role and leaves the wire protocol unchanged.
LRU and TinyLFU both support depth 1 when their model-owned cache fits.

After row grouping in ggml_compute_forward_mul_mat_id, CPU worker 0
acquires the current Expert as before. Before publishing its data at
the existing barrier, worker 0 finds the next nonempty row group in
the actual Expert loop and issues that one key through an optional
Provider interface. It makes no request after the final group.
The current lease remains valid through the existing end-of-compute
barrier. The loop and all dot-product inputs keep their original order.

The cache uses one persistent C++ thread and one aligned, contiguous
payload region for prefetch, plus bounded state and a condition
variable. The job transitions empty → queued → reading → ready/failed.
All state changes hold the cache mutex. The I/O thread unlocks before
calling the existing synchronous Reader, which independently locks the
shard. It publishes readiness under the cache mutex. Demand waits on
the condition variable with the cache mutex released. A scope is the
GGML output tensor identity for the synchronous graph operation.
Different in-flight graphs cannot consume or cancel one another's
job. They can skip a busy prefetch slot. The caller must serialize
executions of the same graph and preserve the ordinary model lifetime
contract. Multi-context contention has cache-level coverage only.

Issuing prefetch performs a read-only resident lookup. No MRU, pin,
frequency, admission or eviction changes follow from that lookup.
Demand still performs normal cache lookup, admission and reservation.
On demand miss the cache consumes only a matching scope/key and
slice length. It waits for a matching running job as necessary.
If a read failed, it performs a single normal demand read. On a hit,
or a conflicting demand in the same scope, it logically discards
that scope's outstanding result.

On delivery the cache swaps the reserved, unpinned target slot's
payload pointer with the prefetch payload pointer. This works for a
resident and for the fixed-index TinyLFU bypass slot. Slot metadata,
generations, aligned backing allocation and indexes keep their
identity. The new prefetch payload comes from the former destination,
so the I/O worker can use it while the current lease pins its delivered
payload. There is no memcpy and no extra allocation per request.

Finish can cancel queued work, discard ready data, or mark a reading
job for discard and drain it. This is logical cancellation of
synchronous OS I/O. The provider joins the thread before destroying
shard locks or handles. It holds neither the cache mutex nor the lease
record mutex across join or shard I/O.

## Memory and metrics

The allocator subtracts one aligned maximum slice and the aligned
prefetch state from the existing cache budget before computing the
resident slot count. Too-small budgets fail model loading. TinyLFU
keeps its separate bypass slot. OS thread stacks, runtime thread
resources and opt-in diagnostics are outside Pager-managed bytes.
The in-memory provider and cache types contain additional optional
dispatch pointers, accounted even at depth 0. OFF starts no I/O
thread, reserves no prefetch payload and performs no extra Reader calls.

Metrics use an expert_prefetch_depth configuration field and prefetch
object. Issued means queued. reads_started counts actual Reader calls.
reads_completed counts successful calls and reads_failed counts errors.
consumed_ready/waited count successful deliveries, distinguished by
whether the demand blocked. cancelled_before_read has no Reader bytes.
discarded_bytes counts successful bytes abandoned before demand.
fallback_reads counts matching failed prefetches retried synchronously.
demand_reads and demand_bytes_read exclude all prefetch calls and bytes.
Pager disk_reads and bytes_read keep accounting for all actual Reader
calls and successful bytes. Demand misses remain distinct.

overlap_ns sums the same-process intersection of the background Reader
interval and the current Expert's worker-0 interval from just after the
publish barrier through the completion barrier. It therefore includes
some barrier and loop overhead and does not add per-worker intervals.
Reader time is cumulative service duration, including overlapped work.
Demand wait_ns measures waiting for matching prefetch completion.
The overlap counter itself runs only when prefetch is enabled.
When compute finishes first, its interval is held until the Reader
completion timestamp is published. This avoids treating publication
delay as additional overlap.

## Diagnostics

Retain the demand-only .experts.jsonl and .admission.jsonl. With prefetch
enabled, demand misses and their logical bytes remain in the demand
header for Python policy replay. reader_calls and reader_bytes give the
physical application Reader totals. Replay uses the actual resident slots.
For --expert-io-trace-limit, .expert-io.jsonl becomes v2 when prefetch is
enabled. A trailing nonzero id identifies an asynchronous Reader call.
The .prefetch-delivery.jsonl sidecar links demand event index to job id
and indicates whether the caller waited. The analyzer verifies keys,
offsets, delivery uniqueness, unmatched misses and actual Reader totals.
Depth 0 preserves v1. Older performance metrics are explicitly treated
as depth 0. Compare one depth per analyzer input set.

## Intel PC-A and PC-B

Use the build, executable and DLL distribution, SHA-256 validation,
OpenSSH service, SSH login and tunnel instructions in
[Phase 2B-9A](phase2b-9a-expert-io-analysis.md) and
[Phase 2B-8B-2](phase2b-8b-2-pager-tinylfu.md). Build the Stage target
and tests with UCRT64, then use the same verified binary and GGUF on both
PCs. Add --expert-prefetch-depth 0 or 1 to the common flags on all three
roles, alongside --pager-policy tinylfu and --pager-budget-mib 256 or 512.
Use a new directory per prompt, depth, budget and trial. Retain CPU-only
mmap, ctx 256, batch 2, threads 2, chat and greedy, with 24/24 split.
Verify one 16-token smoke test, then run the four configurations three
times at 64 tokens, rotating order and recording actual order. Inspect
RAM and commit before starting. Keep the OS cache label uncontrolled.
Compare .generated_tokens and .generated.txt against the same prompt's
OFF reference and against the same trial's Baseline.

Copy Worker metrics, CSV and log into the matching PC-A trial folder.
Run analyze-stage-performance.py for each trial and keep OFF and ON
capacity summary inputs separate. In a separate diagnostic run add
--expert-io-trace-limit 200000 to all roles and copy the event and
delivery sidecars with the metrics. Run analyze-expert-io.py with
--gguf, --trace and --output as in 9A. After failure or a capped trace,
save the failed folder and rerun in a fresh directory. A discarded
synchronous syscall may delay shutdown until that call returns.

## Changed files

- Cache and loader: src/llama-expert-cache.cpp, src/llama-expert-cache.h,
  src/llama-model.cpp, src/llama-model.h, src/llama.cpp, src/llama-ext.h.
- CPU bridge: ggml/src/ggml-cpu/external-expert.h and
  ggml/src/ggml-cpu/ggml-cpu.c.
- Stage and trace: qwen3moe-stage-tcp.cpp and stage-expert-trace.h in this directory.
- Analysis: analyze-expert-io.py, analyze-stage-performance.py and
  analyze-pager-capacity.py in this directory.
- Tests: tests/test-expert-cache.cpp, test-stage-tcp.py and
  test-expert-io-analysis.py.

The equal-residency test uses 255 MiB OFF and 256 MiB ON when the
measured resident count is 206 in both runs. Check the resident count
in metrics before reusing this comparison on other models or builds.
