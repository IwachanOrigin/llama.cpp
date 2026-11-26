# Phase 2B-11A Qwen3MoE logits projection investigation

## Executed graph and Worker entry point

Coordinator evaluates layers [0,24) and sends F32 hidden state.
Worker maps it to batch.embd, passes the original positions, and
fills batch.logits with one per token for every response kind.
The Worker calls stage::compute, llama_decode and llama_synchronize.
Non-output stages terminate at stage_output; only the stage that
owns the final layer runs the LM head.

~~~mermaid
flowchart TD
    A["Worker stage_input: F32 [2048, tokens]"] --> B["Layers 24–46, then layer 47 Attention"]
    B --> C["Layer 47: GET_ROWS on attention result and residual"]
    B --> K["K/V update nodes explicitly expanded"]
    C --> D["Residual ADD → RMSNorm → Router → Expert FFN → residual ADD"]
    D --> E["l_out-47: F32 [2048, outputs]"]
    E --> F["result_norm: RMSNorm and output_norm.weight MUL"]
    F --> G["result_output: MUL_MAT with output.weight"]
    G --> H["F32 [151936, outputs] → backend readback"]
    H --> I["All rows validated; greedy on final row when requested"]
~~~

Source landmarks for the investigated checkout:

- qwen3moe-stage-tcp.cpp worker and stage::compute set batch flags,
  run and synchronize, then obtain rows using llama_get_logits_ith.
- src/models/qwen3moe.cpp:76–94 binds stage_input and creates out_ids.
  Lines 129–166 gather both branches after Attention and before the
  FFN residual, Router, Experts and final residual. Lines 169–190
  create stage_output or result_norm and result_output.
- src/llama-graph.cpp:366–396 fills out_ids from ubatch.output,
  with the identity path for all outputs. At 2887–2904 out_ids is
  an I32 tensor sized to n_outputs. At 3351–3370 attention input
  and cache writes are explicitly expanded. At 4382–4391 build_logits
  expands the LM head; src/llama-model.cpp:4063 calls it.
- src/llama-graph.cpp:1467 marks t_logits and t_embd when present.
  Marking an output does not replace dependency expansion.
- src/llama-context.cpp:2366 tracks output rows, 2429–2439 queues
  backend readback, and 2572 updates output bookkeeping. At 4724
  the public accessor synchronizes before retrieving a host row.
  src/llama-graph.h:877 and llama-graph.cpp:396 include n_outputs
  in graph reuse criteria. llama-context.cpp:2971 selects distinct
  previous graph slots for zero versus nonzero outputs.

Changing batch.logits does not move layer 47 Attention or its K/V
updates. It changes the rows entering layer 47 FFN and Router,
and can change the top-k IDs, Expert demand reads, cache admission
and prefetch. Exact reader deltas depend on cache state. Norm and
projection also inherit the changed row count. Zero outputs are
represented in decode bookkeeping and skip host readback, but
that alone does not establish that this Stage graph and the
zero-sized MoE path preserve all needed computation.

## Output weight and CPU route

The real GGUF has a separate output.weight, Q6_K, shape
[2048,151936], 311,164,928 elements, 255,252,480 B. The tensor
starts at absolute GGUF offset 5,969,408, equal to the file's data
section start. Its 256-element quantized blocks are 210 B and
the vocabulary row stride is 1,680 B. token_embd.weight is Q4_K
at offset 261,230,080. The loader's embedding tie fallback is
unused here. output_norm.weight is F32 [2048], 8,192 B at
261,221,888. The output stage owns the head as an ordinary mapped
CPU weight outside Expert Pager. CPU buffers and extra repack are
controlled by Stage setup. Runtime diagnostic records input/output
shapes, types, strides and buffer names.

The ordinary GGML CPU MUL_MAT dispatch is used. The generic Q6_K
dot table pairs Q6_K with Q8_K; when the RHS is F32 the CPU path
can convert it to the dot type. The selected route on this x86-64 build is the generic dot kernel. The Stage test uses two threads.
The bundled llamafile dispatcher has no Q6_K entry. The build disables AVX/AVX2, and objdump shows the x86 Q6_K/Q8_K wrapper relocating directly to ggml_vec_dot_q6_K_q8_K_generic. Intel binaries may select a different implementation.
Normalized F32 input has strides 4/8192 bytes on the first two
axes, and FP32 logits use 4/607744. Per output row the logical cost
is 311,164,928 MACs and 607,744 output bytes. Skipping 18 rows
avoids 5,600,968,704 logical MACs and 10,939,392 bytes of materialized
FP32 logits. Neither quantity is a measure of physical read traffic
or elapsed execution time.

## Diagnostics and benchmark

On Worker append --projection-perf-details. This implies
--perf-details and can coexist with --tcp-perf-details. It uses
the existing scheduler evaluation callback; result_norm and
result_output introduce synchronized subgraph boundaries. The
interval after result_norm and through result_output includes
head graph dispatch, backend compute and synchronization. Hence
it is a wall-time segment with diagnostic overhead, not a pure
kernel timer or a multi-thread service-time sum. With diagnostics
disabled the callback and graph boundaries stay at their original
settings. Host readback is inside llama_decode and is not timed
separately. The later accessor timer measures synchronized host
row lookup only. Non-sampled finite scans and sampled strict greedy
(including its finite check) are disjoint recorded intervals.
Records are JSON at projection_detail, up to 4096, with dropped
count. Join by request ID and position to perf.csv; Worker phase
can be empty by design, so derive Prefill/Decode from Coordinator.
Absence of a segment or readback timing means unmeasured.

For isolated tests, use the existing CPU operator benchmark:
`test-backend-ops perf -b CPU --test-file projection-ops.txt`.
The run used the exported-test format with MUL_MAT op 28 and
these two leading cases:

~~~text
28 0 151936 1 1 1 0 2 14 2048 151936 1 1 210 1680 255252480 255252480 0 2048 1 1 1 4 8192 8192 8192 P1
28 0 151936 2 1 1 0 2 14 2048 151936 1 1 210 1680 255252480 255252480 0 2048 2 1 1 4 8192 16384 16384 P2
~~~

Operator inputs are generated by the harness in the matching
quantization/shape. It uses its default CPU thread selection and
does not benchmark the GGUF mmap or control OS cache residency.
The Windows ARM64/x86-64 emulator yielded means P1 14.656 ms,
P2 30.758 ms, P4 62.621 ms and P8 139.167 ms. These are diagnostic
operator results for that host and should not be subtracted from
the Intel generation time.

## Candidate designs and 11B validation

A: Add an explicit local Stage head policy carried through context
graph parameters and reuse keys. Keep all batch output flags and
Transformer nodes. For STATUS_ONLY, explicitly expand and retain
the normalized hidden root, omit construction/expansion of the head
and provide no logits tensor or accessor call. Keep GREEDY_TOKEN
and Full unchanged initially. Retain the current path by default.

B: Retain pre-norm l_out-47 instead and skip normalization too.
Verify callback outputs and any consumer of normalized embeddings
before adopting it. C: Gather only required rows after all final
FFN work. For contiguous last-row selection, test a 2D view with
an explicit offset and stride; benchmark the view or GET_ROWS
and scheduler compatibility separately. D: The existing early
out_ids gather can prune final-layer Expert work and is a separate
optimization with different Pager semantics.

Preserve explicit K/V write expansion, full l_out dependency,
POSITION/KV bookkeeping and scheduler error behavior. Graph
identity must distinguish headless/headful execution even at
equal token and output counts. A simple removal of t_logits can
prune otherwise unused compute. The proven pooled embedding
head omission is a structural precedent, not a validation of
zero-output decode or Stage cache transitions.

Skipping the head intentionally removes finite validation of
uncomputed logits. Finite hidden state cannot catch corrupt
weights, projection overflow or other head-local failures.
Keep Full and a strict diagnostic/reference path, and require
finite validation for all projected sampling rows. Treat this
different fault detection contract as an explicit opt-in.

11B touch points are Stage response dispatch, context graph
parameters and reuse, Qwen3MoE and llm_graph_context::build_logits.
Start with A and an internally explicit mode, without a wire
change. Test headless/full transitions and repeated identical
batch shapes, future selected logits byte equivalence, KV state,
all Expert access IDs/order, read counts, Pin/Lease and graph
dependencies. Cover partial and full Prefill batches, 1-token
prompt, EOG/context, split GGUF, v1 fixed-token and nonfinite
failure behavior. Repeat on the real model and compare separate
ordinary versus diagnostic runs on Intel Ethernet+SSH. Use the
same clocks and request IDs for intervals. Do not sum Expert I/O
service time and overlapped compute.

The improvement budget is the measured intermediate-prefill
head segment plus disjoint validation/readback saved, minus
unknown extra selection/graph overhead. Do not scale the whole
generation or projection timings by 18/82. Compare actual sums
for those nine P2 requests. At 10E scale the two stages already
issue 66,649 Expert Reader calls (~59.35 GiB requested), making
Expert wait/overlap a competing investigation if head savings
are small. Read requests do not measure physical disk bytes.

## Local regression and measurement procedure

Use the 10F Coordinator/Worker flags, the 10D launch procedures
and a 256 MiB budget for the local comparison. Append
--projection-perf-details on Worker only in the diagnostic run.
Use the same prompt and compare generated files, request CSV,
expert traces and Pager counters to an otherwise identical
baseline and distributed run. For Intel Ethernet+SSH repeat
the same diagnostic, also capturing separate ordinary timing
runs. A full head segment should be compared with the owned
stage's wall time. Capture phase by request ID from Coordinator.

In the 16-generated-token trial the diagnostic and control runs
matched generated IDs, UTF-8, Worker Expert IDs, per-role Expert
access sequence, Pager counters, request schedule and bytes.
The Worker recorded 18 intermediate-Prefill rows (940.967 ms),
one final-Prefill row (41.313 ms) and 15 Decode rows (640.990 ms).
The intermediate validation scans measured 1.741 ms total and
the host row accessor 0.027 ms. The Worker Prefetch wait counter
was 3.672 s and its Reader-related I/O counter 10.067 s. These
counters can overlap other work. They cannot be added to the
projection segment to reconstruct total generation time.

The additional isolated P1/P2 runs used
TEST_ARG_HARDWARE_CONCURRENCY=2,4,8 to request 1,2,4 backend
threads. P1/P2 means were 91.933/174.033 ms, 46.241/94.141 ms,
and 23.379/47.176 ms. Input data are synthetic and model
diagnostics or other local work can compete for CPU resources.
No controlled cold-cache experiment was run.

### 64-token diagnostic result

| Worker phase | Requests | Rows | Projection segment (ms) | Host accessor (ms) | Finite scan only (ms) | Greedy including finite scan (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Intermediate Prefill | 9 | 18 | 724.455 | 0.0146 | 1.209 | 0 |
| Final Prefill | 1 | 1 | 36.154 | 0.0016 | 0 | 1.057 |
| Decode | 63 | 63 | 2605.629 | 0.0989 | 0 | 33.035 |

Runtime reported CPU_Mapped output.weight and CPU activation/output
buffers. The middle nine requests account for about 0.726 s of
measured candidate work (head segment, finite scans and accessor),
or about 0.64% of this 112.935 s diagnostic generation. Untimed
readback and new graph selection/scheduling costs are unknown.
This calculation does not predict a realized saving. Worker Reader
I/O counter was 17.677 s and Prefetch wait 6.503 s, with 37,846
Reader calls and 36,087,054,336 requested bytes. They make wait
reduction worth profiling separately. Results are Windows ARM64
with x86-64 emulation and cannot be applied to the Intel timing.

All 73 requests matched the existing non-diagnostic control
schedule; generated IDs and UTF-8, Expert artifacts and access
order, Pager counters and wire byte counts were also equal.
Use the 16-token same-build paired run as the primary check of the
diagnostic toggle. The 64-token control is the retained 10D run.
