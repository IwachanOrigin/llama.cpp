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

Phase 1B-3B can give each model one shared cache, assign stable IDs to its
registered tensor sources, and use a shard-locked reader for
`llama_file::read_exact_at()`. The current model loader and single-slot CPU
provider still use the Phase 1B-2A path.
