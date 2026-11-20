# Phase 2B-7 Stage TCP profiling

Use MSYS2 UCRT64 Bash from the checkout. Use existing build-lan with the
CPU-only options for both hosts.

~~~bash
cmake -S . -B build-lan -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DGGML_NATIVE=OFF
cmake --build build-lan --target qwen3moe-stage-tcp test-expert-cache \
  test-external-expert test-qwen3moe-lazy test-qwen3moe-stage test-backend-ops --parallel 4
ctest --test-dir build-lan --output-on-failure \
  -R 'test-(expert-cache|external-expert|qwen3moe-(lazy|external|lru|stage|stage-tcp|stage-lan-options))$'
~~~

## Intel PC-A and PC-B

On both hosts set the local model path and the same configuration. Use a
fresh output path per run. cache_state is a label for the operator's observed
procedure; it does not manipulate the OS cache. Auto identity streams the full
file before model load and may warm the cache. Keep hash mode and startup order
consistent across runs. Record host load, storage, revision and SSH setup.
If CMake cannot read Git metadata, set LLAMA_BUILD_COMMIT in the configure
command with the checkout revision and record any local changes separately.

~~~bash
stage=./build-lan/bin/qwen3moe-stage-tcp.exe
model=/c/moe-stage-test/models/Qwen3-30B-A3B-Q4_K_M.gguf
dir=results/user-only-01
mkdir -p "$dir"
common=(--ctx 256 --batch 2 --threads 2 --pager-budget-mib 64 \
  --accept-timeout-ms 240000 --frame-timeout-ms 30000 --response-timeout-ms 300000 \
  --perf-details --perf-cache-state uncontrolled)
~~~

On PC-A, place the requested Japanese user text in UTF-8 user.txt. Execute
Baseline, and wait for it to finish.

~~~bash
gen=(--generate --chat --prompt-file user.txt --max-new-tokens 64 --sampling greedy)
"$stage" baseline "$model" 24 48 0 mmap "$dir/baseline" auto \
  "${common[@]}" "${gen[@]}" 2>"$dir/baseline.log"
~~~

On PC-B start Worker, and wait for READY.

~~~bash
"$stage" worker "$model" 24 48 50053 mmap "$dir/worker" auto \
  "${common[@]}" 2>"$dir/worker.log"
~~~

On PC-A start the existing SSH tunnel, then run Coordinator in another shell.

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

Check normal exit on both peers. Copy the Worker JSON and CSV to the run
directory on PC-A. Keep the two generated output pairs beside their JSON.

~~~bash
report_dir=$(cygpath -m "$dir")
python tests/moe-paging/distributed/analyze-stage-performance.py \
  --baseline "$report_dir/baseline.metrics.json" \
  --coordinator "$report_dir/coordinator.metrics.json" \
  --worker "$report_dir/worker.metrics.json" > "$dir/comparison.md"
~~~

Repeat with UTF-8 system.txt and append --system-file system.txt to gen,
using a new directory. For each input do one verification and then three
comparable repetitions. Pass each triplet in order with repeated role
arguments to compute median/minimum/maximum. Also run a corresponding
unprofiled case with fresh prefixes (remove --perf-details), compare both
generated files and use the legacy generation logs to gauge overhead.
Alternate trial order if comparing durations. Avoid treating one sample
or uncontrolled caches as an estimate of speedup.

## Schema version 1

Each role writes PREFIX.metrics.json and PREFIX.perf.csv only when
--perf-details is supplied. Rows accumulate in memory and are saved on exit.
Failures after profiling initialization are marked failed, with partial
counters. Baseline has no TCP. Worker receives the same frames as before
and cannot determine prefill versus decode locally. The analyzer joins its
blank-phase rows to Coordinator by session and request ID and validates
positions, token counts and bytes without comparing host timestamps.

JSON contains settings, session ID, optional content identity, prompt-token
SHA-256 (over little-endian IDs), initialization, generation, Stage and Pager
summaries, TCP and memory. The prompt itself does not appear in metrics.
Generation is present on generating Baseline and Coordinator only.
Phase summaries are local on Baseline and Coordinator. Durations use a
monotonic clock. Field suffixes identify ms, ns and bytes.

- hash_ms is null when local hashing is skipped. model_load_wall_ms includes
  gguf_metadata_ms and weights_load_ms. Context initialization also includes
  KV and initial graph allocation. Worker accept wall includes standby.
  Handshake follows connect or accept. elapsed_before_metrics_write_ms starts
  at profiling initialization and includes earlier phases and output save.
- prefill_ms covers execution of the tokenized prompt, including transfers
  and Worker execution on Coordinator. Prompt rendering and tokenization
  are measured separately. prefill_tokens_per_sec divides prompt tokens by
  prefill seconds. decode_steps counts only the extra forwards.
  decode_step_wall_ms sums those forward steps including exchange, encoding
  and decoding on Coordinator. decode_tokens_per_sec divides decode_steps
  by that duration; zero-step throughput is null.
- generation_elapsed_ms includes prefill and the generation loop, through
  final selection. It excludes initialization, preprocessing, detokenization
  and CLOSE. generation_tokens_per_sec uses all generated tokens over that
  interval. The legacy generation log retains its prior decode-loop-only
  rate with the first prefill-derived token included in the numerator.
- stage_execution_wall_ms covers synchronized llama_decode and includes
  in-stage Pager activity. Pager durations are accumulated counters and
  should not be subtracted to estimate pure CPU time.
- TCP bytes count socket payload including Stage framing and handshake.
  They exclude TCP/IP and SSH headers. Send and receive times include waits
  inside transfer calls. Coordinator receive may include Worker computation.
  Worker receive can include idle time before a request. CSV serialization
  and deserialization are activation/logit payload construction and parsing,
  excluding TCP transfer and small frame-header handling. RTT starts before
  Coordinator sends and ends once the response frame arrives. Its residual
  after subtracting Worker execution includes transfer, payload processing,
  framing and scheduling. It is not one-way latency or pure network time.
- Pager totals and deltas report reader calls and requested bytes, not SSD
  I/O. Rates per input token use processed tokens in the phase, including
  prompt tokens. cache_wait_ns already includes lease and capacity waiting.
- On success, process memory is sampled before model/context disposal.
  Windows reports Working Set, Peak Working Set, PrivateUsage and the API's
  PeakPagefileUsage as separate fields. Linux uses VmRSS and VmHWM and does
  not fabricate commit values. Failure memory may be sampled after unwinding.
  Logical tensor payloads, KV buffer bytes and Pager allocations are not
  equivalent to resident memory.

CSV column order is fixed by the header. Each row represents a parsed
Worker request or a Baseline/Coordinator batch attempt. Missing fields are
blank and measured zeros remain numeric. A partial request can appear with
only the values recorded before failure. Fixed-input runs use the fixed phase
on Baseline/Coordinator and do not populate generation-only fields.

