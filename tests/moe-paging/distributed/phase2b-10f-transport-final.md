# Phase 2B-10F Transport results and recommended configuration

Phase 2B-10E used revision 971ec18c30a814c4b1f60694023bda927485b2cc
and executable SHA-256
54c506d467653f28ddc5607e6d92d0e0bdbf5202d206baf89c05e6d0d10535ee.
The supplied aggregate is the evidence for this document.
phase2b10e2-wired-final.zip was unavailable in the workspace.

The runs used i7-12700 and i7-10750H hosts with 16 GB each, wired
Ethernet and OpenSSH forwarding, Qwen3-30B-A3B-Q4_K_M, 24/24 layers,
ctx 256, batch/threads 2, mmap, 512 MiB TinyLFU per Stage, depth 1,
Prompt A, 64 greedy tokens and uncontrolled OS cache.

| Regular trials (s) | Trial 1 | Trial 2 | Trial 3 | Median |
|---|---:|---:|---:|---:|
| E1 Full | 29.527 | 30.359 | 29.892 | 29.892 |
| E2 Greedy split writes | 34.165 | 32.977 | 33.066 | 33.066 |
| E3 Greedy coalesced | 29.265 | 29.220 | 29.189 | 29.220 |

E3 was 11.63% shorter than E2 and 2.25% shorter than E1. Its
three trials were faster than all three E1 trials. These are results
for the tested hosts and workload, based on three trials per mode.

| Metric | E1 | E2 | E3 |
|---|---:|---:|---:|
| Prefill median (s) | 10.364 | 10.837 | 10.002 |
| Decode median (s) | 19.490 | 22.357 | 19.218 |
| Decode tokens/sec median | 3.232 | 2.818 | 3.278 |
| RTT residual median (s) | 0.850 | 3.963 | 0.323 |
| Detailed diagnosis payload receive median (ms) | 6.009 | 51.008 | 0.0098 |
| Detailed diagnosis payload receive sum (ms) | 486.407 | 3683.483 | 0.673 |
| Detailed diagnosis payload receives at least 40 ms | 0/73 | 73/73 | 0/73 |
| Peak Coordinator WS median (MiB) | 839.73 | 837.45 | 837.43 |
| Peak Worker WS median (MiB) | 1079.52 | 1078.07 | 1078.39 |

Coalescing removed the observed small-body waits associated with split
writes. NODELAY alone did not remove the delay in the prior transport
comparison. No packet capture was performed to establish a root cause. Nagle,
Delayed ACK and SSH scheduling remain possible contributors.

For this wired greedy deployment, use
`--logits-response worker-greedy --tcp-nodelay 0 --tcp-small-frame-coalesce 1`
on Coordinator and `--tcp-nodelay 0 --tcp-small-frame-coalesce 1`
on Worker. Use `--pager-policy tinylfu --pager-budget-mib 512 --expert-prefetch-depth 1`
on both. The CLI defaults remain Full, OS-default NODELAY,
split sends and the existing Pager/prefetch defaults.

Worker bytes fell from 49,837,832 to 3,084 (99.9938%), while
Coordinator bytes changed from 675,188 to 675,484. Combined Stage
bytes fell from 50,513,020 to 678,568 (98.66%); TCP/IP and SSH
overhead are outside these counters. All modes executed 73 requests
and produced 82 logits rows. E3 returned 64 token IDs and zero rows.

The supplied 12-run record includes 36 successful processes,
24 identical generated-ID files and 24 identical UTF-8 files,
identical Worker Expert IDs (24 layers), all 876 matched requests,
identical model and binary hashes, budget adherence, unchanged
Pager counters and zero timeouts, transport or session errors.
Both stages called the Expert reader 66,649 times in aggregate
(about 59.35 GiB of application requests). No prefetch discard,
failed prefetch read or final pin remained.

E3 Trial 2 spent 14.61452 s in Coordinator Stage and 14.25082 s
in Worker Stage, about 98.8% of 29.22013 s elapsed. These
include Pager I/O and waits. See the
[Projection investigation](phase2b-11a-logits-projection-investigation.md)
for the next diagnostic target.
