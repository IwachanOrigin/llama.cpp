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

## Expert access tracing and offline replay

Run the trace separately from timing measurements. Use the same input options,
I/O mode and batching as the timed run. The access trace requires Pager enabled
and zero in-process warmup, so prefill starts with an empty Expert cache.
`--trace-prefix` is optional and changes prefill to request every logits row.
Use it to reproduce the earlier synthetic 438-access run, or omit it to match
the current benchmark's normal last-row-per-batch workload.

PowerShell example for a local single-file or first-shard GGUF:

```powershell
$bench = 'build/bin/bench-expert-paging.exe'
$model = 'D:/models/Qwen3-30B-A3B-Q4_K_M.gguf'
$workload = @('--model', $model, '--threads', '4', '--ctx', '512', '--batch', '128', '--ubatch', '64', '--prefill', '128', '--decode', '32', '--io', 'mmap', '--contexts', '1')
& $bench @workload --pager lru --slots 1024 --plan --output plan.json
if ($LASTEXITCODE) { throw 'Could not size cache' }
$plan = Get-Content plan.json -Raw | ConvertFrom-Json
$required = $plan.model_buffer_bytes_including_pager + $plan.kv_bytes + $plan.workspace_bytes + 1GB + 48MB
if ($required -gt $plan.available_physical_bytes) { throw 'Insufficient memory for this trace configuration' }
& $bench @workload --pager lru --budget $plan.budget_bytes --required-available-bytes $required --expert-access-trace expert.jsonl --expert-trace-limit 1000000 --output trace-run.json
if ($LASTEXITCODE) { throw 'Trace execution failed' }
python tests/moe-paging/analyze-expert-trace.py expert.jsonl --output analysis.json
python tests/moe-paging/test-expert-trace.py
```

The plan uses the exact managed Pager budget. Choose workload values from the
timed run you want to explain. If 1024 slots exceed the memory limit, capture a
smaller serial LRU run or the legacy `--pager single` path and replay the same
trace at larger capacities. Trace storage is outside the Pager budget and is
preallocated up to the requested event count. The example allows 48 MiB for
one million records on a 64-bit build. Use a fresh output trace path.

The JSONL header contains the source ID to tensor name map, total and dropped
events, observed cache counters, and configured cache size. Each subsequent
line is `[sequence, source_id, expert_id, bytes, slot, decode, hit]`.
Layer and gate/up/down are derived from the tensor name. The event sequence
is serialized by the existing Pager lock, at miss reservation and successful
hit pin. Context IDs are unavailable at that interface. Under concurrency,
I/O completion, pinning, and LRU order can diverge from reservation order.
Serial LRU replay must reproduce observed outcomes and counters. The analyzer
reports parallel differences without requiring equality, and rejects truncated
traces. A failed inference does not produce a successful trace.

Replay starts empty for each policy and preserves the prefill cache in
decode. The capacity comparison uses the same fixed maximum Expert size for
all policies. Reported slot payload excludes management bytes. The header
records the real runtime's managed bytes for its observed capacity. Read
counts represent reader calls. Selective admission ranks keys by full-trace
frequency. Frequency-proportional layer allocation also uses the full trace.
These are retrospective comparisons. Belady is the equal-slot optimum for
mandatory insertion on each miss. The alternate policies run only offline.

## Distributed CPU inference design

The official Qwen3-30B-A3B configuration specifies 48 layers, hidden size
2,048, expert intermediate size 768, top-8 routing, 128 experts per layer,
four KV heads and head dimension 128. The three expert matrices are gate
and up (2,048 by 768) and down (768 by 2,048). The reported 1,152 weight
requests per decode token equal 48 layers times eight experts times three
matrices. [Official model configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)

These are payload estimates for one sequence and FP32 activations. They
exclude headers, acknowledgements, retransmits, and initial model placement.
Multiply bytes per token by the desired tokens per second to size sustained
link bandwidth.

| Placement | Payload per token | Dependency and ownership |
|---|---:|---|
| Remote Expert weights | 1.02 GiB at the supplied observed read demand and all misses remote | Remote hosts store Expert weights. The coordinator retains local compute and KV. Without request coalescing, up to 1,152 reads can be on the token's path. |
| Remote Expert compute, one input and output per selected expert | 6 MiB with every routed Expert remote | For each of 48 layers, 8 inputs and 8 outputs of 8 KiB each. Group dispatches by host and wait at the layer's merge. Hosts hold Expert weights and execute gate/up, activation and down. The coordinator holds attention, KV and routing. |
| Remote Expert compute, one host per layer with remote weighted aggregation | 768 KiB | Each of 48 layers exchanges an 8 KiB activation and 8 KiB aggregate. Top-k IDs and weights or remote router computation are additional. Match local accumulation order if bitwise comparison is required. |
| Contiguous Transformer stages | 8 KiB per boundary, one way | A stage holds all weights for its layers, including router and Experts, and retains those layers' KV. A single token passes through stages in dependency order. Logits/sampling placement may require a return message. |

At the supplied 32/12.51 tokens/s, the first three rows imply respectively
about 22.4 Gbit/s, 129 Mbit/s and 16.1 Mbit/s. The last row implies about
0.168 Mbit/s per boundary. The first row's source number is logical demand
from the supplied run, including reads served by the OS page cache. At 1 Gbit/s,
moving 1.02 GiB alone takes about 8.8 seconds per token. For one dependent
exchange per layer at 1 ms RTT, the Expert-compute round trips alone add at
least 48 ms per token. For contiguous two-stage placement, the boundary
activation can travel in one direction per token, with one return
dependency if sampling is coordinated on the first host.

The KV payload for FP16 K and V is 2 x 4 x 128 x 2 = 2,048 bytes per
token per layer, or 96 KiB per token over 48 layers. A 24-layer stage keeps
48 KiB per token, roughly 192 MiB for 4,096 slots before allocator and graph
overhead. Extending Context or adding sequences raises this independently of
the per-token link volume. Keep stage KV local and give every request a session
and token position. On a lost peer, invalidate that session on the affected
stages and replay the committed prefix, unless all stages can prove a common
checkpoint. A remote weight fetch can be retried under an immutable model
identity. A remote expert-compute request can be retried if it is stateless.
Do not blindly retry a layer stage after it has appended KV.

## Existing GGML RPC

The repository's [RPC device implementation](../../ggml/src/ggml-rpc/ggml-rpc.cpp)
already includes device discovery, buffer allocation and transfer, remote
graph compute/recompute and a protocol handshake. Its supports-op callback
currently returns true without asking the remote device. Use a representative
CPU graph as a capability test. Initial buffer transfers can still be large
if the coordinator loads and uploads all weights. RPC buffer handles are tied
to an endpoint/device and graph reuse depends on that connection. Existing
RPC neither defines an Expert slice or lease request nor turns a local
CPU_ExternalExpert provider into a remote one. Its GGML graph messages should
not be assumed to preserve local pointers or file handles across processes.

The smallest useful next phase is a two-host CPU split at one contiguous
layer boundary, with ordinary weight buffers for each stage, fixed model
identity, one sequence, local KV on each host and an FP32 activation boundary.
Start with a synthetic model and compare per-token logits and network bytes
in an isolated lab. Then measure a real split model and load placement.
Evaluate RPC's device/graph path for that prototype before designing a
separate stage service. A worker-local GGUF and explicit reconnect/session
semantics may need a separate layer. Expert-level remote compute comes next
if placement balance or communication measurements warrant it. Remote
weight fetch is useful as a comparison, but the measured miss traffic
makes it a poor first target for ordinary LAN.

For Windows/Linux interoperability, start with matching revisions, identical
model and shard hashes, known CPU features and token/graph shapes, and fixed
little-endian scalar encodings for any new cross-platform protocol. Validate
tolerances separately from same-build bitwise comparisons. The [RPC
documentation](../../tools/rpc/README.md) limits the current server to
isolated testing. No cross-host or mixed-OS run was performed in this phase.
