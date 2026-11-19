# Phase 2B-5 greedy generation

MSYS2 UCRT64 Bash, from the repository root.

~~~bash
cmake --build build --target qwen3moe-stage-tcp test-backend-ops
ctest --test-dir build --output-on-failure -R 'test-(expert-cache|external-expert|qwen3moe-(lazy|external|lru|stage|stage-tcp|stage-lan-options))$'
build/bin/test-backend-ops.exe test -b CPU -o MUL_MAT_ID
python tests/moe-paging/test-expert-trace.py
~~~

Set the model path and common options on both PCs.

~~~bash
stage=./build/bin/qwen3moe-stage-tcp.exe
model=/c/moe-stage-test/models/Qwen3-30B-A3B-Q4_K_M.gguf
mkdir -p results
common=(--ctx 256 --batch 2 --threads 2 --pager-budget-mib 64 \
  --accept-timeout-ms 180000 --frame-timeout-ms 30000 --response-timeout-ms 300000)
~~~

Run the baseline on PC-A first, using a fresh result prefix.

~~~bash
prompt=/c/moe-stage-test/prompts/test01.txt
generation=(--generate --prompt-file "$prompt" --max-new-tokens 16 --sampling greedy)
"$stage" baseline "$model" 24 48 0 mmap results/baseline auto \
  "${common[@]}" "${generation[@]}" 2>results/baseline.generate.log
~~~

On PC-B, start Worker and wait for READY.

~~~bash
"$stage" worker "$model" 24 48 50053 mmap results/worker auto \
  "${common[@]}" >results/worker.ready 2>results/worker.generate.log
~~~

On PC-A, establish the tunnel according to [the two-host guide](phase2b-3-two-windows-hosts.md).

~~~bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 user@intel-pc-b
~~~

In a separate PC-A terminal run Coordinator and compare both files and stop reasons.

~~~bash
"$stage" coordinator "$model" 24 48 50053 mmap results/coordinator auto \
  "${common[@]}" "${generation[@]}" 2>results/coordinator.generate.log
cmp results/baseline.generated_tokens results/coordinator.generated_tokens
cmp results/baseline.generated.txt results/coordinator.generated.txt
grep '^generation ' results/{baseline,coordinator}.generate.log
~~~

Check each process exit before comparing. After the 16-token case, restart
Worker and repeat with --max-new-tokens 32 and fresh prefixes.

The prompt is UTF-8 plain text. An optional UTF-8 BOM is removed.
Tokenization applies the GGUF vocabulary's BOS/EOS policy once and treats
text resembling special tokens literally. The fixture-only --prompt-tokens
bypasses tokenization for tests. That fixture has no tokenizer and emits no text.

The .generated_tokens file contains little-endian u32 IDs, including EOG
if selected. The .generated.txt file contains detokenized generation only.
A cutoff inside a multibyte character is represented with U+FFFD. At the
context boundary, generation can select a token from the last available
logits without evaluating a new position. Greedy chooses the lowest ID
among ties. Logits must be finite.

Worker processes ordinary Stage requests. The 16 MiB limit still covers
all prefill logits rows. Generation retains no full logits artifact. If
IDs diverge, compare the first differing little-endian u32 value.
A failed TCP session publishes no generation artifacts.

Generation prefill and decode times include local execution and, when
distributed, transport and Worker execution. Decode time ends at the last
token selection, before detokenization and CLOSE. It counts one fewer
forward pass than selected tokens. Per-process logs also record socket bytes,
Working Set, Pager reads, hits, misses, I/O time and pinned slots.

