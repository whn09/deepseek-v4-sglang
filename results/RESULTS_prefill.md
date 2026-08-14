# Prefill results — DeepSeek-V4-Flash, DeepEP v2 on EFA, 2× p6-b300

Measured 2026-08-14. Question asked: **does 2-node prefill input throughput exceed
2× single-node, and does it admit higher concurrency?**

## Held constant across every row

| | |
|---|---|
| model | `DeepSeek-V4-Flash`, MXFP4 routed experts (auto-detected), `--moe-runner-backend deep_gemm` |
| KV | `fp8_e4m3` (model-forced), `--swa-full-tokens-ratio 0.1`, `page_size 256` |
| per-rank chunked prefill | **8192** (`--chunked-prefill-size` = 8192×TP, divided by dp_size) |
| per-rank concurrency slots | **32** (`--max-running-requests` = 32×TP → 256 on 1 node, 512 on 2) |
| workload | `bench_serving --dataset-name random`, **ISL 4096 / OSL 8**, `--random-range-ratio 1.0`, `--flush-cache`, 2 waves per row |
| transport | `GDAKI=1` → `NCCL_GIN_TYPE=5` (EFA GDA), `EP_NIC_NAME=rdmap101s0` @400 Gb/s |
| graphs | `--disable-cuda-graph` (prefill graph is force-disabled by the deepep_v2 handler anyway) |

OSL is 8, not 0, so decode contributes ~0 to the wall clock while TTFT/ITL still
exist. Read **input token throughput**; it is the quantity the question is about.

Both `--chunked-prefill-size` and `--max-running-requests` are GLOBAL budgets that
DP attention divides by `dp_size`. Pinning the PER-RANK values is what makes node
count the only variable — otherwise the 2-node shape would silently get half the
concurrency slots per rank and "2 nodes serve more concurrency" would be
untestable for a purely configural reason.

## Input token throughput (tok/s)

| shape | a2a backend | SM | c256 | c512 | c1024 |
|---|---|---|---|---|---|
| 1 node, 8 GPU | `deepep_v2` `direct` (NVLink) | 20 | 162.8k | 166.0k | — |
| 1 node, 8 GPU | `deepep_v2` `direct` (NVLink) | 55 | 162.9k | 168.5k | **169.1k** |
| 2 node, 16 GPU | `deepep_v2` `hybrid` (EFA) | 20 | 196.9k | 212.8k | 220.4k |
| 2 node, 16 GPU | `deepep_v2` `hybrid` (EFA) | 24 | — | 238.9k | 256.3k |
| 2 node, 16 GPU | `deepep_v2` `hybrid` (EFA) | 55 | 204.4k | 256.9k | **274.4k** |
| 2 node, 16 GPU | `none` (all-gather, EP=16) | — | 67.5k | 72.7k | 72.5k |
| 2 node, 16 GPU | `megamoe` | — | cannot boot — see below |

p99 TTFT at the peak of each shape: 1 node c1024 **23.2 s**; 2 node c1024 **14.3 s**;
`a2a=none` c1024 **56.0 s**.

## Answers

**1. 2 nodes is 1.62×, not >2×.** 274.4k / 169.1k = 1.62, i.e. 81% per-node scaling
efficiency. The hypothesis that 2-node input throughput beats 2× single-node does
**not** hold at this shape.

**2. Higher concurrency: yes, and the latency evidence is stronger than the
throughput evidence.** One node saturates at c512 (c512→c1024 buys 0.4%) and its
p99 TTFT is already 23.2 s. At the *same* c=1024 two nodes deliver 1.62× the
throughput at **62% of the p99 TTFT** (14.3 s vs 23.2 s).

**3. DeepEP v2 is what makes 2 nodes worth having at all.** Against the only
non-DeepEP a2a that crosses nodes here, DeepEP v2 is **3.78×** (274.4k vs 72.7k).
And 2 nodes without it (72.7k) is only **43% of ONE node with it** (169.1k) — on
EFA, adding the second node is a net loss unless the a2a backend is DeepEP v2.

## The knob that mattered most: per-rank chunk, not SM

Single node, SM 20, c256 — the first configuration measured used the cookbook's
capacity of 1024 and was chunk-bound by a factor of 4:

| per-rank chunk (= capacity) | input tok/s |
|---|---|
| 1024 (cookbook value) | 41.4k |
| 2048 | 70.6k |
| 4096 | 130.8k |
| **8192** | **162.8k** |
| 16384 | server did not stay up — clean `exit 0`, `kill_process_tree`, no error line, no host OOM (3.9 TB free), cause not established |

Comparing the two shapes at capacity 1024 would have pitted two crippled configs
against each other. 8192 is both near the single-node knee and a normal production
value.

SM is the weaker knob and it acts **only** on the cross-node shape:

- 2 node: 20 → 220.4k, 24 → 256.3k, 55 → 274.4k (monotonic up to the clamp)
- 1 node: 20 → 166.0k, 55 → 168.5k (**SM-insensitive**; intra-node a2a is not SM-bound)

`kMaxSM` for v2-on-EFA is `(256−2×17)/4 = 55`, so 55 is the ceiling, and for
cross-node prefill the ceiling is where you want to be. This is the opposite of
the decode guidance (where ~24 is enough) — the knee is phase- and
topology-dependent, not a constant.

## `megamoe` cannot cross nodes in this build

```
torch/distributed/_symmetric_memory/__init__.py:2004  rendezvous
RuntimeError: Failed to send fd: No such file or directory
```

megamoe rendezvouses through `torch.distributed._symmetric_memory`, which passes
POSIX fds over a unix socket — intra-node only. Cross-node symmetric memory needs
FABRIC handles, which this path does not use. Rank 0's scheduler dies right after
the DeepGEMM warmup and the server tears itself down. `20_launch_node.sh` now warns
before launching it with `NNODES>1`. Keep megamoe as the **single-node** attribution
baseline only.

DeepEP v1 is also unavailable for comparison here: the image uninstalls
`sgl-deep-ep` and installs the v2 EFA fork over the same `deep_ep` module name.
A v1 comparison needs the `ep-benchmarks-efa` image, not this one.

## Operational notes worth keeping

- **First 2-node boot on a cold host costs ~11 min**, nearly all of it TileLang JIT
  in `_prewarm_mhc_kernels` (`deepseek_v4.py:2907`). It looks exactly like a hang:
  every rank prints `MHC prenorm prewarm: 22 n_splits buckets`, then GPU memory is
  allocated and utilization sits at 0% for minutes. `py-spy dump` distinguishes the
  two instantly — a barrier-blocked rank shows `(idle)` in
  `parallel_state.barrier`, a compiling rank shows `(active)` in
  `tilelang/engine/lower.py`. Warm the cache by booting once per host before timing
  anything.
- deep_gemm JIT also is not in the server's warmup set: the first bench row came
  out at 6.3k tok/s and the identical rerun at 20.7k. `92_prefill_sweep.sh` fires a
  discarded warmup row for this reason.
- The `NCCL WARN Call to ibv_query_port_speed failed ... errno 93` lines (16 of
  them, one per rank) are benign on EFA.
