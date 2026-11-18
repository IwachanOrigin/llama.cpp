# Phase 2B-1 local stage execution

The internal staging API in `src/llama-ext.h` loads a named Qwen3MoE GGUF with
the half-open interval passed to `llama_model_load_stage`. A full interval
uses the full graph. Global layer IDs are preserved. For partial ranges the
loader creates tensors for owned layers and required endpoints, disables
whole-file mmap prefetch, and totals progress over selected tensors. mmap can
reserve file address space even when pages are outside the stage's weight
inventory. The operating system can cache file pages independently of the
stage allocation.

Each stage model owns its Pager and its contexts have independent KV caches.
The normal KV layer filter keeps only owned layers and uses the existing
global-to-local mapping. If the GGUF ties output to embeddings, the final
stage loads a local copy of the tied weight for output projection. It uses
the incoming activations at the graph entry.

Use a default CPU context with one sequence. The first stage accepts IDs.
Its embeddings output returns every boundary row, even when only the last
logits row is requested. A subsequent stage consumes a contiguous F32
`[hidden_size, tokens]` buffer through `llama_batch.embd` with `token == nullptr`.
Pass the same positions and sequence 0 to both stages. Each context constructs
its own attention mask and applies RoPE. Keep the stage models and contexts
alive across calls for the sequence. Consume output before reusing a context.
The final stage returns ordinary logits. Use compatible context lengths and
batching across stages. Context ownership and failure recovery across
processes are planned for Phase 2B-2.

Build and run the regression in an existing CPU test build:

```powershell
cmake --build build --target test-qwen3moe-stage
ctest --test-dir build -V -R '^test-qwen3moe-stage$'
```

The test runner generates the fixture and shards using the existing paging
fixture scripts. It tests both mmap and buffered I/O, compares finite
boundary/logits bytes and Expert IDs after a three-token prefill and two
decode steps, and measures normal tensor bytes, Expert bytes/cache, KV and
scheduler workspace. With a local fixture on Windows, run each role in a
fresh process to sample Working Set. For the two-layer fixture:

```powershell
foreach ($role in 'full','prefix','suffix') {
    & build/bin/test-qwen3moe-stage.exe --working-set model.gguf $role 2
}
```

This last probe decodes one token. The suffix uses a fixed F32 activation,
so its resident-page sample measures an isolated execution rather than the
same token path as the full graph. Correctness comparison uses activations
from the prefix in the CTest.
