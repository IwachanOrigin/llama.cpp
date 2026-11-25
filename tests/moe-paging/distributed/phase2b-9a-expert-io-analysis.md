# Phase 2B-9A External Expert I/O

## Code path and lock order

The graph in src/llama-graph.cpp, llm_graph_context::build_moe_ffn,
creates ffn_moe_topk through ggml_argsort_top_k and shares the selected
IDs across up, gate and down. In the CPU non-tiled external path,
ggml_compute_forward_mul_mat_id groups activation rows by Expert before
the first acquisition; the selected IDs for that tensor and batch are
already known. The layer's down projection needs the gate/up output as
its input. Routing for the next layer depends on the current layer
output, and routing for the next decoded token depends on the generated
token and preceding state.

~~~mermaid
sequenceDiagram
    participant GGML as CPU MUL_MAT_ID
    participant Provider as tensor_source / lru_state
    participant Cache as llama_expert_cache
    participant File as llama_file / shard
    GGML->>GGML: Group routed rows by Expert ID
    GGML->>Provider: Worker 0 acquire(expert, nb02)
    Provider->>Cache: acquire((source ID, Expert ID), stride)
    Note over Cache: Cache mutex, hit/pin/MRU or miss/observe/admission/reservation
    alt miss
        Cache->>Provider: read_expert with cache mutex released
        Provider->>File: Lock shard, read_exact_at(offset + expert * stride, slot)
        File-->>Provider: Buffered read directly to selected slot
        Provider-->>Cache: Unlock shard and return
        Note over Cache: Relock cache, publish READY and pin or recover on failure
    end
    Cache-->>GGML: Generation-tagged lease via provider record
    GGML->>GGML: Barrier, quantized dot products use slot pointer
    GGML->>GGML: Barrier, worker 0 releases lease
    GGML->>Provider: release_lease
    Provider->>Cache: Unpin, free transient slot or allow resident eviction
~~~

llama_model_loader::llama_tensor_weight calculates the absolute offset
within the source file from the GGUF data base plus its relative tensor
offset. llama_model::load_tensors registers the shard, offset, Expert
count and nb[2] stride. The model retains each llama_file and its shard
mutex. The provider reserves the lease record separately. It holds
neither that mutex nor the cache mutex when entering shard I/O. The
CPU worker holds a single pin per operation iteration; both barriers
protect the slot pointer for other workers. Existing cache io_wait_ns
times the reader callback after cache unlock and before retrying the
cache mutex; it includes the shard wait and reader work, including
error logging when applicable.

## File I/O and memory copies

- Windows Stage mmap and buffered modes use the retained handle from
  ggml_fopen, _fileno and _get_osfhandle. read_exact_at checks the range,
  seeks, reads, and checks the resulting position. The buffered path
  uses synchronous ReadFile with a null OVERLAPPED. For a successful
  Expert slice the source path has four SetFilePointerEx calls (seek,
  tell in read_raw, tell in read_raw_unsafe, final tell) and one
  ReadFile unless the request exceeds MAXDWORD. Win32 short reads
  fail immediately. The distinct no-buffering handle and aligned
  bounce path are outside the two tested Stage load modes.
- Linux Stage mmap/buffered keeps a FILE* and uses fseeko, ftello and
  fread in the same seek/read/check sequence. A short read or stream
  error is reported without a retry in the buffered branch. libc can
  implement these calls with internal buffering or multiple syscalls.
  Its kernel-call count is unavailable in this diagnostic. The
  separate O_DIRECT path uses lseek/read and retries EINTR, with
  fallback for some alignment errors. No pread or mmap data access is
  used by the tested Expert reader. Linux is source-reviewed only.
- The buffered reader on Windows writes to the final resident or
  transient slot, and the quantized dot kernel reads that slot.
  No explicit pager memcpy, decompressed weight cache, or GGML weight
  staging buffer is involved. Activation preparation is separate.
  File cache hits, kernel copies and physical SSD transfers are
  outside these measurements. mmap controls ordinary tensors.

## Diagnostic and offline simulation

--expert-io-trace-limit N adds a bounded array of reader records
(maximum 200,000 records), and enables the existing access trace with
at least N entries. Successful runs produce the unchanged
.experts.jsonl plus .expert-io.jsonl, indexed by cache access event.
The sidecar includes shard and absolute shard offset, requested bytes,
Reader start/end on a process-monotonic clock, shard mutex acquisition
time, measured ReadFile count/returned bytes/time when available, and
success. Successful Reader timing includes the measured ReadFile and
shard acquisition intervals. The residual also includes the seek/tell
calls, C++ checks, unlock, and sampling overhead. The existing
io_wait_ns also includes callback overhead. These intervals overlap.
ReadFile counters exclude seek/tell API calls. Actual syscall counts,
device bytes and OS transfer counts are unavailable.

A decode failure saves an incomplete I/O sidecar. A record cap is
reported separately for reads and the access trace. Analysis rejects
incomplete data. New clocks and the bounded event arrays are active
only with the diagnostic. No per-read log write occurs in normal
inference. The public-internal callback can run concurrently across
contexts; the Stage consumer indexes records only for its synchronous
single-context CPU execution. A concurrent consumer would need its own
synchronization and a request correlation key. The provider stores two
extra diagnostic pointers in its accounted bookkeeping.

The Python analyzer reads only metadata and tensor descriptions from
each GGUF shard. It independently checks diagnostic offsets and misses,
joins Worker requests with the Coordinator CSV and reports distributions
and dimensions by phase, layer, tensor, Expert and slot kind. Supply
shards in the model loader's order. Legacy access traces support layout
and locality analysis with timing fields unavailable.

The simulation combines only misses in one Stage request and one
tensor, where CPU row grouping provides the routed selection. It holds
the observed demand miss set fixed and never combines different
requests, tensors or shards. Using the eventual misses is a conditional
opportunity estimate: an implementation must respect TinyLFU admission,
slots available at demand time and existing arithmetic order. It would
need a multi-slice provider operation or staging, not a larger length
on the present fixed-size single-Expert callback. A merged span needs
temporary storage up to the largest merged range and scatter copies
into destination slots if the existing slots are preserved. The tool
reports those bytes. Read amplification is span bytes divided by
required miss bytes. The cross-tensor gate/up/down envelope is a
layout calculation only and is excluded from the simulation.

## Implementation choices for 2B-9B

| Candidate | Evidence and tradeoff | Priority |
| --- | --- | --- |
| Remove redundant file-position queries | Reader residual is measurable but combines several small costs. Test a buffered exact-read path on both OSes only if the residual justifies it. Preserves budget and demand order. | Secondary |
| Direct read to final slot | Current behavior. No additional separate copy to eliminate. | Keep |
| Merge slices in one tensor | Zero-gap merges can cut reader calls, but still transfer all required bytes and add staging/scatter with existing slots. Needs a new provider entry point and careful admission handling. | Secondary |
| Batch provider API | Can expose same-tensor Expert IDs after row grouping. Multiple live pins can reduce useful residency or block a one-slot configuration. Extra memory is bounded only with an explicit request/slot cap. | Enabler |
| One selected Expert ahead on dedicated reader thread | Overlap one known synchronous read with CPU work. Portable, queue depth one and one budgeted aligned slice. Keep lookup/observe/admission at actual demand, use a read-only lookup to avoid speculative hits becoming frequency updates. Track stale results, extra bytes, waits and cancel/drain before model destruction. | First prototype |
| Overlapped I/O / IOCP, pread pool or io_uring | Can pipeline more requests, at cost of OS-specific lifetime, completion, cancellation and ordering. A single buffered read may have the same latency. Test the portable overlap first. | Defer |
| mmap direct reference | No explicit copy to remove before the current final slot. Mapping may avoid buffered transfer into that slot, but introduces page faults and OS-controlled resident pages. Stable views need model-owned mappings; Windows offsets obey allocation granularity and Linux offsets obey page alignment. A mapped view limit cannot enforce a physical Working Set limit. | Defer |
| Repack Expert weights | Same-Expert gate/up/down spans are large on this file. A repack can place them together but needs format/version validation, disk space, and new loaders. Additional readers and on-disk byte ordering must preserve quantized slices exactly. | Defer |

In the prototype start with one same-tensor Expert selected after
row grouping. The observed interval between Reader events includes CPU
work and other operations and is not a directly usable overlap time.
Use separate demand and prefetch counters, compare byte-for-byte with
prefetch off, and measure overlap, discarded prefetch bytes and wall
time. If scratch storage cannot fit, leave overlap off. A prefetch must
not prevent demand from proceeding or reuse storage before the lease
or pending read completes. No cross-layer or next-token prediction is
needed. Compare Stage communication separately on both Intel hosts.

## Reproduce on Intel PC-A and PC-B

Use the [Phase 2B-8B-2 LAN procedure](phase2b-8b-2-pager-tinylfu.md)
for building in UCRT64, copying the binary and matching DLLs, verifying
SHA-256, starting sshd from elevated PowerShell, verifying login and
keeping the SSH loopback tunnel open. Use the same GGUF identity on
both machines. In those commands, give each test its own run directory
and append --expert-io-trace-limit 200000 to the shared flags. Use LRU
256, TinyLFU 256, TinyLFU 512 and LRU 768 with the requested Prompt A
and optionally B. Start at 16 tokens, then use 64 when RAM and commit
headroom permit. Keep normal performance runs in separate directories.

Compare generated files with cmp as before. Recover Worker .metrics.json,
.perf.csv, .experts.jsonl, .expert-io.jsonl and .log into the PC-A run
directory. Keep the Coordinator .perf.csv alongside them. For a capped
or failed trace inspect the header. If either dropped count is nonzero,
rerun in a fresh directory with an increased cap within 200,000, or
shorten generation while recording the changed condition. Use:

~~~bash
python tests/moe-paging/distributed/analyze-expert-io.py --gguf "$model" \
    --trace "$run/baseline.experts.jsonl" --trace "$run/coordinator.experts.jsonl" \
    --trace "$run/worker.experts.jsonl" --output "$run/io-analysis.json"
~~~

## Observed layout and measurement boundaries

The 17.28 GiB single-file Qwen3-30B-A3B-Q4_K_M has 144 Expert tensors,
128 Experts per tensor and 32-byte GGUF alignment. Gate and up are
Q4_K in all 48 layers. The down tensors split evenly between Q4_K
and Q6_K. Expert slices are 884,736 B (120 tensors) or 1,290,240 B
(24 tensors). All 6,144 same-Expert triplets occupy disjoint slices
in three tensor-major ranges. A triplet needs a median 2,856,960 B
of payload. The median covering file interval is 228,636,672 B
and the median per-triplet amplification is 86.06x.

In the initial 256 MiB TinyLFU trace at 16 generated tokens,
ReadFile occupied 99.63–99.68% of summed Reader time. The median
shard mutex acquisition was 0.10 us, and median residual 2.10–2.40 us.
The median time from one Reader ending to the next in the same
request and tensor was 593–689 us, including more than CPU dot
products alone. The comparison host is Windows ARM64 running an
x86-64 UCRT64 binary. These diagnostic durations are not Intel timing
estimates.

The physical Intel Phase 2B-8B-3 input supplied aggregate figures.
Dividing its aggregate wait by its Reader calls yields about
269 us/call at LRU 256, 251 at TinyLFU 256, 264 at TinyLFU 512 and
327 at LRU 768. That arithmetic illustrates why total bytes or
call count alone cannot explain the wait difference. Locality,
OS cache state, page residency and device latency are competing
explanations. Per-role Intel Reader traces and OS/device counters
would be needed to distinguish them. Do not sum Reader time with
Stage time or interpret returned ReadFile bytes as SSD transfers.
