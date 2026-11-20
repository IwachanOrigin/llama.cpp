# Phase 2B-6: Qwen3 chat prompts

Build from the repository root in MSYS2 UCRT64 Bash.

~~~bash
cmake --build build --target qwen3moe-stage-tcp test-backend-ops
ctest --test-dir build --output-on-failure -R 'test-(expert-cache|external-expert|qwen3moe-(lazy|external|lru|stage|stage-tcp|stage-lan-options))$'
~~~

Use the CPU build and SSH loopback forwarding described in
[the two-host guide](phase2b-3-two-windows-hosts.md). Start the baseline first
on PC-A. On PC-B start a new worker for each coordinator invocation.
Set the local model path and common options on each machine.

~~~bash
stage=./build/bin/qwen3moe-stage-tcp.exe
model=/c/moe-stage-test/models/Qwen3-30B-A3B-Q4_K_M.gguf
mkdir -p results
common=(--ctx 256 --batch 2 --threads 2 --pager-budget-mib 64 \
  --accept-timeout-ms 180000 --frame-timeout-ms 30000 --response-timeout-ms 300000)
~~~

On PC-A, save the user message in user.txt and, for the second run, the
system message in system.txt. Both files are UTF-8. For the system run,
append --system-file system.txt to the chat array and change tag to
system-user. Use fresh output prefixes for each run.

~~~bash
tag=user-only
chat=(--generate --chat --prompt-file user.txt --max-new-tokens 64 --sampling greedy)
"$stage" baseline "$model" 24 48 0 mmap "results/$tag-baseline" auto \
  "${common[@]}" "${chat[@]}" 2>"results/$tag-baseline.log"
~~~

For the second run, set these variables on PC-A before repeating the
baseline and coordinator commands, and start a fresh worker.

~~~bash
tag=system-user
chat=(--generate --chat --prompt-file user.txt --system-file system.txt --max-new-tokens 64 --sampling greedy)
~~~

Start the worker on PC-B and wait for READY.

~~~bash
"$stage" worker "$model" 24 48 50053 mmap results/worker auto \
  "${common[@]}" 2>results/worker.log
~~~

Keep the tunnel open on PC-A.

~~~bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 user@intel-pc-b
~~~

Run Coordinator from the PC-A terminal that defined the variables. Check
the three process exit statuses before comparing their files.

~~~bash
"$stage" coordinator "$model" 24 48 50053 mmap "results/$tag-coordinator" auto \
  "${common[@]}" "${chat[@]}" 2>"results/$tag-coordinator.log"
cmp "results/$tag-baseline.generated_tokens" "results/$tag-coordinator.generated_tokens"
cmp "results/$tag-baseline.generated.txt" "results/$tag-coordinator.generated.txt"
grep -E '^(chat_enabled|generation |stage \[)' "results/$tag-baseline.log" "results/$tag-coordinator.log"
~~~

The tool uses the GGUF default chat template and the existing common
Jinja renderer with an Assistant generation prompt. Missing templates,
substitution, rendering failures and missing Assistant continuations
produce errors. Raw generation is selected by omitting --chat.

Jinja marks template text and message text as separate spans. Usual
inputs retain whole-prompt tokenization. When a message contains a
literal control-token spelling, it is tokenized as text in a separate
span, which can change BPE segmentation at the span boundary. The
template's control tokens still use special-token parsing. A configured
leading BOS is inserted once. Generation starts at the rendered
Assistant prefix without an automatic final EOS.

Thinking uses the common renderer's default setting. There is no added
Thinking switch or tag filtering. A /no_think string is passed as
message content. The resulting model behavior is not forced by the tool.

Normal logs record template source, prompt token count after rendering,
generated count, stop reason, timing and Pager statistics. For explicit
diagnostics, --rendered-prompt-file writes UTF-8 prompt text to a new file.
That file contains the user and system messages. It is a rendering
diagnostic, not a serialization of the protected token spans.

