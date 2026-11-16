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
