# Phase 2B-8B-2 TinyLFU-inspired Expert admission

## Policy and ownership

Stage TCP accepts --pager-policy lru or --pager-policy tinylfu. The
default is lru. The Stage loader passes the policy to the model-owned
Expert cache. The CLI and metrics record the local policy. The protocol
is unchanged. Use the same policy and budget on the three roles for a
performance comparison. The analyzer treats missing policy in old
metrics as LRU and rejects mixed settings within a run.

A key is (tensor_source_id, expert_id). Gate, up, and down each have
their own source within each layer. On a resident hit, observe the key,
pin the slot and move it to MRU. On a miss with a free resident slot,
observe and admit. Otherwise use the oldest resident without pins or
waiters, observe once, and replace it only when the candidate estimate
exceeds the victim estimate. A tie bypasses. The rejected candidate
still incurs one full reader request.

TinyLFU uses the same BLAKE2b-128 digest as the Python reference, on two
uint64 words encoded little-endian, with no key or salt. The four
little-endian 32-bit digest words index four distinct rows of width
next_power_of_two(max(16, 4 * resident_slots)). Each byte counter
saturates at 15. The estimate is the minimum of the four counters.
Before observing access max(16, 10 * resident_slots) + 1, halve each
counter using integer division and reset the interval count. Repeat the
same cycle thereafter. The C++ code uses a single fixed-size BLAKE2b
compression, with unsigned 64-bit arithmetic and no dependency.

## Bounded allocation and concurrency

The cache allocation includes 64-byte-aligned slot metadata and
payload, the TinyLFU state rounded up to 64 bytes and four byte-counter
rows. The provider accounts for its existing separate bookkeeping
before passing the remaining budget to the cache. BLAKE2b uses a fixed local stack and no allocation. Allocator-internal
bookkeeping and the OS file cache are outside the Pager requested-byte
metric. TinyLFU initialization fails if a resident slot, a transient
slot and the sketch do not fit. LRU retains the original slot layout and
capacity formula, with no sketch or transient payload.

The last slot is reserved at initialization. Bypass sets it LOADING
under the cache mutex, performs shard I/O with the mutex unlocked, and
publishes a generation-tagged, pinned lease only on success. It stays
outside the resident LRU. The last release clears the transient slot
and notifies waiters. Failed transient reads clear it and notify too.
Transient requests are not coalesced, including concurrent requests
for the same key. Existing same-key resident-loading coalescing remains.

If no resident victim is eligible, wait on the existing condition
variable. A full resident miss with a busy transient slot also waits
before frequency observation. After wakeup recheck the resident hit
and victim. This means some misses that would eventually admit also
wait for the transient slot. It avoids updating the frequency twice
and avoids reserving two slots for one request. The mutex is released
while waiting. The CPU MUL_MAT_ID worker 0 acquires one lease, shares
its pointer with the other workers through the compute barrier, and
releases after the final barrier before acquiring the next Expert.
No path takes a second lease while holding the first in that operator.
The model and cache must outlive all context executions. The provider
does not hold its lease-record mutex while waiting for the cache or
releasing a slot. The cache unlocks before the reader takes the shard
mutex. No new lock is introduced.

## Statistics and verification

In successful serial TinyLFU runs, hits + misses equals frequency updates, and
admitted + bypassed misses equals reader calls. The miss counters count
attempts, including I/O failure. bytes_read counts successful bytes.
Admission comparisons require an eligible victim. Transient waits count
condition-variable wait iterations and are already included in capacity
and aggregate cache waits. Peak slot use measures pinned slots.
The final in-use gauge also includes a transient slot still loading.
The timing suite records I/O, Stage, prefill, decode and resident memory
using the existing metrics. It does not instrument frequency-update,
aging, or comparison time with clocks on every access.

With --expert-trace-limit, TinyLFU writes the existing v1 .experts.jsonl
layout with mode tinylfu, plus .admission.jsonl. Each sidecar row holds
event index, candidate and victim estimates, victim key, bypass and
aging. Candidate -1 means no comparison. The final slot is transient.
For a complete serial trace, verify every event with:

~~~bash
python tests/moe-paging/distributed/simulate-pager-policies.py \
  --verify-tinylfu results/A/tinylfu/256/run-01/coordinator.experts.jsonl \
  --output results/A/tinylfu/256/run-01/coordinator.replay.json
~~~

Keep diagnostic runs separate from timing runs. Replay assumes serial,
successful execution and does not reconstruct concurrent pin ordering.

## Intel PC-A and PC-B

Start from the [Phase 2B-3 LAN steps](phase2b-3-two-windows-hosts.md)
and [Phase 2B-8A commands](phase2b-8a-pager-capacity.md). Build on PC-A
under UCRT64, then copy the same executable and needed UCRT64 DLLs to
PC-B. Compare SHA-256 values of the executable and DLLs on both PCs
using Get-FileHash. Use the same verified GGUF on both PCs.

~~~bash
cmake --build build-lan --target qwen3moe-stage-tcp test-backend-ops test-expert-cache --parallel 4
~~~

On PC-B, use an elevated Windows PowerShell 5.1 session to start the
already configured OpenSSH service with Start-Service sshd. Verify
Get-Service sshd shows Running. On PC-A first verify
ssh user@intel-pc-b and then keep the tunnel open:

~~~bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:50053:127.0.0.1:50053 user@intel-pc-b
~~~

Use a fresh directory per prompt, policy, budget and trial. Save the
UTF-8 user prompt separately and set the variables on each host. Set
policy/budget to lru/256, tinylfu/256, lru/512, tinylfu/512 or lru/768
(the budget value itself is numeric). Start with one check per condition
at 16 tokens, then set count=64. Check RAM and commit on both hosts.

~~~bash
stage=./build-lan/bin/qwen3moe-stage-tcp.exe
model=/c/path/to/Qwen3-30B-A3B-Q4_K_M.gguf
policy=tinylfu
budget=256
count=64
run="results/A/$policy/$budget/run-01"
mkdir -p "$run"
common=(--ctx 256 --batch 2 --threads 2 --pager-policy "$policy" --pager-budget-mib "$budget"
        --perf-details --perf-cache-state uncontrolled --accept-timeout-ms 240000
        --frame-timeout-ms 30000 --response-timeout-ms 300000)
gen=(--generate --chat --prompt-file user.txt --sampling greedy --max-new-tokens "$count")
~~~

PC-A Baseline, PC-B Worker and then PC-A Coordinator after READY:

~~~bash
"$stage" baseline "$model" 24 48 0 mmap "$run/baseline" auto "${common[@]}" "${gen[@]}" 2>"$run/baseline.log"
"$stage" worker "$model" 24 48 50053 mmap "$run/worker" auto "${common[@]}" >"$run/worker.ready" 2>"$run/worker.log"
"$stage" coordinator "$model" 24 48 50053 mmap "$run/coordinator" auto "${common[@]}" "${gen[@]}" 2>"$run/coordinator.log"
cmp "$run/baseline.generated_tokens" "$run/coordinator.generated_tokens"
cmp "$run/baseline.generated.txt" "$run/coordinator.generated.txt"
~~~

Copy Worker metrics, CSV and logs back into the matching PC-A folder.
For example, when PC-B runs with its SSH home as the working directory,
run the following on PC-A after the Worker has closed:

~~~bash
scp "user@intel-pc-b:$run/worker.metrics.json" "user@intel-pc-b:$run/worker.perf.csv" \
  "user@intel-pc-b:$run/worker.log" "$run/"
~~~

In each condition, also cmp both generated files against the LRU
reference for that prompt. Record actual run order, failures and
available commit. On failure, stop only the processes for that trial,
preserve its results, and start again using a new trial directory.
Use analyze-pager-capacity.py with repeated --run arguments and --output.
Its by_policy object separates policies and budgets. The legacy
by_budget_mib view continues to show LRU trials. Reader requests and
bytes measure application I/O, with OS file caching uncontrolled.
