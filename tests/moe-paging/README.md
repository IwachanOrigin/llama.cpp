# CPU MoE paging tests

Run from the repository root with CMake, a C/C++ toolchain, and Python 3.9 or newer.
On Windows use the MSYS2 UCRT64 shell and its native CMake, Python, and compiler.

```sh
cmake -S . -B build -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DGGML_BACKEND_DL=OFF
cmake --build build --config Release --target test-qwen3moe-lazy test-external-expert
ctest --test-dir build -C Release -R '^test-(qwen3moe-(lazy|external)|external-expert)$' --output-on-failure -V
```

`test-qwen3moe-lazy` requires the CPU backend, llama tools, and Python.
It regenerates the Phase 1A fixture with `random.Random(681)` in a temporary directory,
then runs `llama-gguf-split --split-max-tensors 13` to create two shards.
The GGUF v3 fixture has two layers, 32 experts per layer, top-2 routing, Q8_0 expert weights,
F32 non-expert weights, and 128 token IDs. No model download is needed.
The generator can also be run directly as `python tests/moe-paging/make_fixture.py model.gguf`.

Both layouts run with CPU mmap, no extra CPU weight buffers, four threads, requested context
128, and batch and microbatch sizes of 32. The prompt is the fixed token sequence
`1 19 7 42 51 33 91 3`. It is followed by 32 single-token decode calls with token
`1 + (17 * step + 13) % 127`. The test compares 160 Expert IDs and 5,120 finite
FP32 logits byte for byte against the single-file `off` run from the same binary.
It also checks that exactly six tensors use lazy mapping in `on`, and zero in
`auto` and `off`. Do not add the generated GGUFs or traces to Git.

`test-external-expert` uses the built-in CPU backend and runs one- and four-thread
graphs. It writes deterministic Q8_0 slices to a small file in the test build directory
and directly reads each requested Expert into one reusable slot. A short slice at EOF
tests error propagation. The test compares with a resident graph, checks which slices
were read, repeats allocation and execution on the same graph, and verifies that the
next node does not execute after either an I/O failure or an explicit abort.

The proof of concept requires an internal flag and a dedicated zero-byte CPU buffer.
Weights have a null data pointer and are consumed only as the first input of CPU
`MUL_MAT_ID` on its ordinary kernel path. The test rejects mismatched buffer and flag
pairs, mixed CPU extra buffers, copies, and views. It also tests scheduler rejection
of nested views and explicit placement on a mock GPU. The capacity printed by the test
is the GGML graph buffer plus the separately reported weight slot. It does not include
graph metadata, the CPU plan workspace, stdio, or the OS file cache.

`test-qwen3moe-external` uses the same fixture and reference. Set
`llama_model_params.external_experts = true` and `n_gpu_layers = 0` to load a named
Qwen3MoE GGUF in this mode. The load modes mmap and buffered I/O are covered,
with one model-owned slot shared by all external Expert tensors. The test checks
load counters, the zero-sized CPU buffer and null data pointer during scheduler
execution, Expert IDs, and logits. It renames the fixture shards after model
destruction to check that the files can be released on Windows.
It also runs two independent llama contexts concurrently against the same mmap
model and compares both traces against the same reference. The contexts and all
their work must finish before destroying the model.

## Phase 1B-3A cache tests

```sh
cmake --build build --config Release --target test-expert-cache
ctest --test-dir build -C Release -R '^test-expert-cache$' --output-on-failure -V
```

The test uses two small files with deterministic Expert byte patterns. Three
source IDs map to those files so two tensor sources can share a shard. Tests
trigger reader entry and completion with condition variables and use deadlines
for all concurrent scenarios.

The cache is constructed once with a byte budget and maximum slice size. Its
accounting includes the cache object, padded slot metadata (including intrusive
LRU links), and one 64-byte-aligned payload per slot. The reader fills this
payload directly. The cache owns no additional read buffer, dynamic key index,
or heap-allocated lease record. A lease is a move-only value held by its caller.
Allocator and OS bookkeeping, file handles owned by the reader, the OS file
cache, ggml workspace, and KV cache are outside this standalone cache's budget.

The cache mutex protects state, statistics, wait registrations, and LRU links.
The reader is called after unlocking the cache; it may take its shard I/O mutex
to serialize seek and read. It must release the shard mutex before returning
and must not acquire from the same cache from the reader callback. On return,
the cache mutex is reacquired to publish success or failure. A successful
lease pins a slot without holding either mutex. Waiting callers for the
same load see its result before the slot is eligible for eviction or reuse.
Release may run on a different thread. All acquires and leases must finish
before cache destruction.
Each CPU execution team releases its current Expert before waiting for another
key, so it cannot block on capacity that only its own release would free.

`cache_misses` and `disk_reads` count reader starts, including failures.
`bytes_read` counts bytes from complete reads. A successful coalesced request
is a hit. I/O time measures reader wall time (including a shard-lock wait),
and cache wait time measures condition-variable waits. Both timers aggregate
across threads.

## Qwen3MoE LRU integration

For a named CPU Qwen3MoE GGUF, enable `external_experts` and provide a nonzero
`llama_model_params.external_expert_cache_size` in bytes. A zero budget selects
the original uncached single slot. The loader assigns source IDs in Expert
tensor creation order and owns one cache and one set of shard handles per model.

The total budget accounts for the cache allocation, source records, the
zero-byte GGML buffer descriptor, the retained file wrappers and filename
capacity, per-shard mutexes, and the bounded pool of 64 provider lease records.
The reader writes directly into each reserved slot. Runtime allocator and
stdio bookkeeping, OS file cache, and normal GGML model and graph storage are
outside this accounting.

The CPU team receives a provider record pointer and generation. Each record
owns one move-only cache lease. The team leader passes the same token back
after all workers finish reading. In LRU mode, record admission, cache
operations, and shard I/O use separate mutexes; the record mutex is released
before the cache is entered, and the cache mutex is released before taking a
shard mutex. Release drops the cache pin before making the record reusable,
without nesting those mutexes. Idle records are reused without per-request
heap allocation. The test holds all 64 records on one Expert, then checks
that the next request waits and resumes with a new record generation.

`test-qwen3moe-lru` checks one, two and four slots, one and four CPU threads,
single and split GGUF, mmap and buffered loading, and two simultaneous
contexts. It also exercises the actual model provider and temporarily truncates
a buffered fixture to check read failure and recovery.

To time the uninstrumented graph with the same decoder inputs:

```sh
python tests/moe-paging/run.py build/bin/test-qwen3moe-lazy build/bin/llama-gguf-split bench
```

The benchmark sweeps 1, 2, 4, 8, 16, 32 and 64 slots using the exact budget
reported by the loader for each capacity. A separate traced run computes
reuse distance as the number of distinct Expert keys since the last use,
then checks the first-hit boundary against the measured counters. Tracing
is disabled for the timed runs. The benchmark alternates run order and
reports eight-run medians. Cache
counters cover prefill and decode. `disk_reads` denotes reader calls, not
physical storage operations. The LRU capacity timer counts waits for a slot;
the single-slot timing measures acquisition of the slot mutex.

## Phase 1B-4A real-model measurements

Build the CPU benchmark and provide a local Qwen3MoE GGUF. For a split model,
pass the first shard. The benchmark reads local paths and uses zero GPU layers.

```sh
cmake --build build --config Release --target bench-expert-paging
python tests/moe-paging/bench-matrix.py --bench build/bin/bench-expert-paging --model /path/to/model.gguf \
  --output measurements.json --threads 4 --ctx 512 --batch 128 --ubatch 64 --prefill 128 --decode 32
```

Append `.exe` to the executable path on Windows. The matrix defaults to Pager
off, uncached single slot, and 1/16/64/128/256 LRU slots, with separate runs for
one and two contexts. `--modes`, `--contexts`, `--io mmap|buffered`, and
`--repeats` select other matrices. Before running each case, it uses
`no_alloc` and the model's exact slot-sizing function. The runner skips
cases whose separate ordinary-weight, Pager, KV, scheduler workspace, logits
and headroom allowances exceed available physical memory or an optional lower
`--max-planned-mib` ceiling. `--headroom-mib` defaults to 1024. Each execution
checks currently available RAM again before loading. The mmap estimate counts
all ordinary weights as a conservative allowance. External Expert weights are
represented only by Pager capacity. The total GGUF file size, model backend
buffer sizes (which can be address spans under mmap), and measured residency
are reported separately. The estimate is a preflight check, not an OS memory
reservation. Compare measured residency against the allowance on the target
host before scaling up. Cases record `ok`, `skipped`, `error`, `timeout` or
`mismatch`. Missing Pager-off comparisons are marked separately.

For a specific byte budget or a standalone estimate:

```sh
build/bin/bench-expert-paging --model /path/to/model.gguf --output plan.json --pager lru --slots 64 --plan
build/bin/bench-expert-paging --model /path/to/model.gguf --output sample.json --pager lru --budget 1GiB --io mmap
```

The matrix writes individual runs, medians, preflight decisions, and
correctness results. Timings use fresh processes and alternate the case order
every round. The default `warm` mode runs an untimed process for each case.
`--warmup N` additionally runs N prefill/decode workloads inside each process,
then clears KV while keeping the Pager cache. `--cache-state operator-cold`
accepts one case with one repeat and no in-process warmup. It labels an
operator-prepared cold run; metadata preflight still reads the GGUF headers.
The tooling leaves OS caches under the operator's control.

The prefill input repeats token IDs 1, 19, 7, 42, 51, 33, 91, 3 reduced
modulo vocabulary size. Decode feeds fixed single tokens, one per step.
Timed prefill requests one logits row per batch. Separate correctness runs
validate all logits for finiteness and compare Expert IDs and FP32 logits
to Pager off per context. Relative error uses max(abs(reference), 1e-30)
as its denominator. Compare traces only within the same matrix and binary.
`--no-verify` omits the correctness runs.

### Interpreting the JSON

- Metadata time covers GGUF headers, device setup, hyperparameters and vocabulary.
  Weight time covers tensor setup and loading. Deferred mmap faults during
  inference contribute to inference time. Context initialization is timed separately.
- Throughput is aggregate tokens divided by wall time across the requested contexts.
  `--sample-decode-memory` enables a separate memory-oriented run with a
  resident-memory query per decode step. Its decode wall time includes those
  queries. The default timing run samples only at phase boundaries, and
  individual results also include per-context compute times.
- Windows resident and lifetime peak metrics are Working Set and Peak Working Set.
  Linux uses VmRSS and VmHWM from `/proc/self/status`. The decode-only sampled peak
  is the highest token-boundary reading. The lifetime peak may include earlier
  phases. Resident mapped pages can count toward process residency; the full
  virtual size of mappings and system file cache are separate measurements.
- Model, KV and scheduler compute buffer capacities describe allocated buffers.
  CPU backend-private scratch is captured in process memory. Expert slot payload
  excludes alignment and metadata; Pager peak includes managed overhead.
- Prefill and decode Pager statistics are deltas after warmup. Pinned slots is an
  end-of-phase sample. `disk_reads` counts reader invocations. Windows process
  I/O counters track read requests and transferred bytes. Linux `/proc/self/io`
  separates interface traffic from bytes attributed to storage by the kernel.
  Process I/O can include unrelated reads, such as procfs sampling.
- `cache_wait_ns` already includes the capacity and lease-record wait
  counters. These are accumulated request times and can overlap across
  contexts. Keep them separate from wall time and do not add the breakdowns
  to the aggregate. On Windows, the host and executable architectures are
  recorded separately. Translation is inferred from IsWow64Process2 and the
  executable target (x86-64 running on ARM64 requires translation).

For Linux sanitizer runs, configure separate builds with both
`LLAMA_SANITIZE_ADDRESS` and `GGML_SANITIZE_ADDRESS` enabled, optionally
combining with `LLAMA_SANITIZE_UNDEFINED` and `GGML_SANITIZE_UNDEFINED`.
Configure a separate build with `LLAMA_SANITIZE_THREAD` and `GGML_SANITIZE_THREAD`
(and `GGML_OPENMP=OFF` if needed). Run the five paging CTests in each build.
