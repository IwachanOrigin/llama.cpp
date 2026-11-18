# Phase 2A memory and network budget

The local checkout only has the real-model measurement summary. The 18.55 GB GGUF was unavailable. The stage inventory tool reports exact tensor payload bytes for a supplied file, grouped by layer and quantization layout. Run it on the real shards before committing to a split. [Model configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)

Each 24-layer stage has 24 x 128 x 3 x 2048 x 768 = 14,495,514,624 Expert parameters. All-Q4_K would represent 7.59 GiB per stage at 144 B per 256 parameters. Q4_K_M mixes types, so neither stage's actual expert bytes can be obtained from that approximation. The largest per-stage Expert slice controls slot size. Compute the managed Pager budget with fixed-slot overhead, per-source and per-shard metadata, plus lease storage. For example an explicit 512 MiB budget limits that stage's Pager to 512 MiB.

Ordinary tensors are attention Q/K/V/O, per-layer router and norms, plus embedding on stage A and output norm/projection on stage B. Based on configured dimensions, F32 attention projection payload is 1.6875 GiB per stage and F32 router payload is 24 MiB. Each endpoint matrix would occupy about 1.16 GiB in F32. This puts the modeled ordinary matrices at about 2.87 GiB per stage before norms/alignment, with actual quantized payloads determined by the GGUF. Real stage KV at 4096 is 192 MiB per stage with FP16 K/V and 24 layers, since 2 x 4 x 128 x 2 B per layer and token yields 48 KiB per token per stage. A 128-token batch with one send and one receive staging buffer consumes 2 MiB, before diagnostic logits. Profile graph descriptors, scheduler work buffers, backend-private scratch, and prefill peak separately on each stage. Allow initially 2 GiB of additional host headroom, then confirm from the actual OS and workload. The modeled ordinary matrices, 512 MiB Pager, 192 MiB KV, 2 MiB boundary and 2 GiB reserve total about 5.56 GiB per stage before the unmeasured graph/workspace costs. Two 16 GB PCs look plausible under these assumptions. This is a conditional estimate, and available RAM and stage peaks still need checking.

Both processes may hold a copy of the GGUF on disk. Parse local metadata and construct only owned tensors, create mappings for used ranges and exclude unowned and external Experts from prefetch. Ordinary tensor Lazy Loader facilities can help with I/O and mapping, while explicit range ownership determines what allocates. Worker-local loading transfers zero initial weight payload over TCP. A whole-file address mapping is virtual memory. Working Set is resident process pages and can include mapped data. Private commit, Pager bytes and OS file-cache occupancy are distinct. Measure each independently.

## Theoretical link budget

At 1 Gbit/s, serialization time uses decimal link rate and excludes Ethernet/IP/TCP overhead. In the illustrative Worker frame, 48 B per request and 24 B per reply cover length, version, session/step, position, count/type and status. Sequence zero is implicit for Phase 2B.

| Exchange | Application bytes | Link serialization | With 1 ms / 10 ms RTT |
|---|---:|---:|---:|
| Decode, one hidden and one sampled I32 token | 8,268 | 0.066 ms | 1.066 / 10.066 ms |
| Decode returning 151,936 FP32 logits | 616,008 | 4.93 ms | 5.93 / 14.93 ms |
| Prefill block of 128, one sampled token returned | 1,048,652 | 8.39 ms | 9.39 / 18.39 ms |

For prefill N tokens in batches of B, send 8192N bytes of activations and ceil(N/B) requests. The first prototype waits for a reply per block before reusing buffers and advancing position. CPU execution and queueing add to the table. The Worker builds or reuses its graph locally and sends no graph description. Ordinary RPC adds graph bytes on UID cache misses, tensor/control message headers, and optional initial tensor uploads. Link volumes for a complete Transformer graph are unmeasured. Keeping KV local transfers zero KV in the normal path. A peer failure invalidates the session and replays tokens. A future full checkpoint migration would transfer about 192 MiB for one full 24-layer stage at the example context length.

The synthetic inventory test produced identical results for a single GGUF and its two shards. Stage A has 3,478,016 B of ordinary tensors and stage B 3,480,064 B, including a local copy of tied embedding/output weights. Each stage has 53,477,376 B of Expert weight payload and 557,056 B maximum Expert slice.
