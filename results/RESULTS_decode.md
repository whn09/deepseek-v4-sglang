# Decode results — DeepSeek-V4-Flash, DeepEP v2 on EFA, 1 vs 2 × p6-b300

Measured 2026-08-14, `NUM_SMS=20` on both shapes (SM was deliberately not swept —
see "What was not measured"). Question asked: **what does the second node do to
decode?**

## Held constant

| | |
|---|---|
| model | `DeepSeek-V4-Flash`, MXFP4 routed experts, `--moe-runner-backend deep_gemm`, 43 layers / 256 experts / topk 6 / hidden 4096 |
| a2a | `deepep_v2`, `direct` on 1 node (NVLink) / `hybrid` on 2 nodes (EFA), `NCCL_GIN_TYPE=5` |
| SM | `SGLANG_DEEPEP_V2_NUM_SMS=20` |
| per-rank dispatch capacity | 2048 (`--chunked-prefill-size` = 2048×TP) — **but see "The capacity knob": this is the wrong value for decode, and 256 is −16% on step time** |
| per-rank concurrency slots | **128** (`--max-running-requests` = 128×TP → 1024 on 1 node, 2048 on 2) |
| decode graph | `--cuda-graph-max-bs-decode 128`, `--mem-fraction-static 0.70`, `cuda graph: True` in every decode batch |
| workload | `bench_serving --dataset-name random`, ISL 1024 / OSL 1024, `--random-range-ratio 1.0`, `--flush-cache`, 2 waves |

Per-rank slots are pinned at 128, not the total, for the same reason as the
prefill run: both `--max-running-requests` and `--chunked-prefill-size` are global
budgets that DP attention divides by `dp_size`.

## The headline: server-side decode step time at matched per-rank batch

This is the comparable measurement. The server logs its own
`gen throughput (token/s)` per DP rank; `running_req / gen_throughput` is a real
per-rank decode step time, and it is sampled at the *same per-rank batch* on both
shapes (51 samples per point, `94_server_decode_rate.sh`).

| req/rank | 1 node step | 2 node step | 2n / 1n | 1 node decode tok/s (×8) | 2 node decode tok/s (×16) | 2n / 1n |
|---|---|---|---|---|---|---|
| 8 | 12.97 ms | 21.74 ms | 1.68× | 4.9k | 5.9k | 1.19× |
| 16 | 13.68 ms | 22.06 ms | 1.61× | 9.4k | 11.6k | 1.24× |
| 32 | 14.43 ms | 22.85 ms | 1.58× | 17.7k | 22.4k | 1.26× |
| 64 | 15.30 ms | 24.46 ms | 1.60× | 33.5k | 41.9k | 1.25× |
| 128 | 16.65 ms | 25.95 ms | 1.56× | 61.5k | 78.9k | **1.28×** |

Linear fits over the five points:

```
1 node (direct/NVLink):  step = 12.72 + 0.0307 x batch   ms
2 node (hybrid/EFA):     step = 21.46 + 0.0351 x batch   ms
```

**Crossing EFA costs a fixed +8.7 ms per decode step (+69% of the floor) and
almost nothing per token (slope +14%).** That is the same shape the standalone
DeepEP b300 work found — a batch-insensitive floor plus a small slope — reproduced
here at the whole-model level rather than per a2a op.

Sanity cross-check: 8.7 ms spread over 43 MoE layers is **~202 µs per layer** of
extra dispatch+combine. The standalone b300 micro-benchmark puts a *hybrid*
dispatch+combine step at 320–343 µs at 128 tokens; intra-node `direct` is cheaper
than that, so a ~200 µs delta is the right order. The two measurements agree.

**So the second node buys 1.25–1.28× the decode throughput for 2× the hardware
(≈63% per-node efficiency), and every individual token gets ~1.6× slower.** Decode
scales far worse than prefill did (prefill was 1.62×).

⚠️ **This table is at capacity 2048, and that turned out to be the wrong capacity
for decode.** The +8.7 ms floor is not an EFA latency floor — it is largely a
fixed-size payload, and it shrinks to +4.9 ms at capacity 256. See the next
section; the 1.28× figure is specific to cap 2048.

## The capacity knob: 2048 → 256 recovers half of the EFA penalty

The tables above were all taken at **per-rank dispatch capacity 2048**, which is
this stack's biggest decode mistake. Re-running the 2-node ladder at **capacity
256** (the floor — per-rank chunk must be a multiple of `page_size=256`) with
nothing else changed:

| req/rank | 2 node cap 2048 | 2 node **cap 256** | Δ step | 2 node cap256 decode tok/s (×16) |
|---|---|---|---|---|
| 8 | 21.74 ms | **17.57 ms** | −19.2% | 7.3k |
| 16 | 22.06 ms | **18.00 ms** | −18.4% | 14.2k |
| 32 | 22.85 ms | **18.87 ms** | −17.4% | 27.1k |
| 64 | 24.46 ms | **20.28 ms** | −17.1% | 50.5k |
| 128 | 25.95 ms | **21.69 ms** | **−16.4%** | **94.4k** |

```
2 node cap2048 (hybrid/EFA):  step = 21.46 + 0.0351 x batch   ms
2 node cap256  (hybrid/EFA):  step = 17.59 + 0.0341 x batch   ms
```

**The whole win is in the floor (−3.9 ms, −18%); the slope is unchanged (−3%).**
That is exactly the signature expected, because the floor *is* the fixed-size
dispatch. Aggregate at 128 req/rank: **80.1k → 94.4k decode tok/s (+17.9%)**, and
the EFA crossing penalty drops from +8.7 ms to **+4.9 ms** over the 1-node
cap2048 floor.

### Matched-capacity scaling: 1.33×

The 1-node cap-256 point was caught in the last minute of machine time — one rung
only, 128 req/rank, 25 samples (`serverrate-n1-sm20-cap256-isl1.txt`):

| 128 req/rank | 1 node | 2 node | 2n / 1n |
|---|---|---|---|
| step, cap 2048 | 16.65 ms | 25.95 ms | 1.56× slower/token |
| step, **cap 256** | **14.40 ms** | **21.69 ms** | 1.51× slower/token |
| decode tok/s, cap 2048 | 61.5k | 78.9k | **1.28×** |
| decode tok/s, **cap 256** | **71.1k** | **94.4k** | **1.33×** |

So capacity 256 helps *both* shapes (1 node −13.5% step / +15.6% tok/s; 2 nodes
−16.4% / +17.9%) and it helps the 2-node shape slightly more, because the fixed
payload it shrinks is the part that crosses EFA. **Matched-capacity scaling is
1.33×, not the 1.53× a cap256-vs-cap2048 comparison would have claimed.** The
crossing penalty at 128 req/rank goes 9.3 ms → **7.3 ms**.

Only one rung was measured on 1-node cap256, so there is no linear fit for it —
the floor/slope split is known for the 2-node shape only.

### Why capacity is a decode perf knob at all

`SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` sizes the receive slab
(`[num_local_experts, capacity × num_ranks, hidden]`), but sglang passes the
**fixed cap** as the collective dispatch argument on *every* call — never the real
token count (`sglang/srt/layers/moe/token_dispatcher/deepep_v2.py:376-383`):

```python
# num_max_tokens_per_rank is a COLLECTIVE dispatch arg (ElasticBuffer requires
# the same value on all ranks). Keep it at the fixed buffer cap ...
# Do NOT derive it from the local hidden_states.shape[0]: under ragged DP load
# (or TP attention) the ranks would disagree on this collective arg.
num_max_tokens = self.num_max_dispatch_tokens_per_rank
```

So a 128-token decode step declares 2048 tokens/rank on the wire. The check:
2048 tok × 4096 hidden × 1 B (fp8 dispatch) = 8.4 MB, ÷ ~50 GB/s per-GPU EFA =
**168 µs**, against the measured ~202 µs/layer of extra cost. **This is the
explanation for the batch-independent 2-node floor** — it is not an EFA latency
floor, it is a fixed-size payload.

In DeepEP's standalone bench `--num-tokens` is simultaneously the token count
*and* the buffer capacity; in sglang they are separate. Our cap-2048 decode rows
were therefore the equivalent of `--num-tokens 2048`, not 128.

### The A/B is protocol-clean

The cap-256 ladder was run at ISL=1 (so prefill cannot contaminate the wall
clock), while the original cap-2048 ladder was ISL=1024. Control point, cap 2048
at ISL=1, 128 req/rank: **25.57 ms** vs 25.95 ms at ISL=1024 — protocol is worth
1.5%, so the −16.4% is the capacity effect. (`results/ctrl-cap2048-isl1.log`,
`results/serverrate-n2-sm20-cap256-isl1.txt`.)

## Client-side view, and why it is not a decode measurement

What an OpenAI-compatible client actually sees on this mixed ISL 1024 / OSL 1024
workload:

| conc | req/rank (1n / 2n) | 1 node out tok/s | 1 node TPOT p99 | 2 node out tok/s | 2 node TPOT p99 |
|---|---|---|---|---|---|
| 64 | 8 / 4 | 4521 | 13.6 ms | 2929 | 21.2 ms |
| 128 | 16 / 8 | 8136 | 15.3 ms | 5249 | 24.9 ms |
| 256 | 32 / 16 | 14066 | 17.4 ms | 10448 | 23.7 ms |
| 512 | 64 / 32 | 18069 | 27.4 ms | 18676 | 26.4 ms |
| 1024 | 128 / 64 | **18368** | 54.3 ms | 19267 | 51.7 ms |
| 2048 | — / 128 | — | — | **19596** | 103.0 ms |

⚠️ **Both shapes flatten at ~18–19.6k output tok/s, and the same ceiling on 8 and
on 16 GPUs is the tell that it is not a decode ceiling** — a real one would move
with the GPU count. Server-side, those same rows were decoding at 61.5k and 78.9k
tok/s (and 94.4k at cap 256, while the client still read 18.7k).

### Correction: the ceiling is **not** prefill interleave — root cause unknown

An earlier version of this file (and the corresponding memory) attributed the
ceiling to prefill/decode interleave in the client's wall-clock denominator
(DP0 logged 262 `Prefill batch` prints against 256 `Decode batch` prints). **That
explanation is insufficient — it was falsified by measurement.** What was
actually established:

- **It reproduces at ISL=1.** `ctrl-cap2048-isl1.json`: 2048 prompts,
  `total_input_tokens 2048` (one token each, so there is essentially no prefill
  work), and the client still read **18,696 out tok/s** while the server was
  stepping at 25.57 ms → 80.1k tok/s. A ~4× gap with prefill removed.
- **It is not one client's fault.** Four independent `bench_serving` processes
  against the same endpoint summed to **20,472 tok/s** — the same ceiling, split
  four ways.
- **It is not tokenizer fan-out.** `--tokenizer-worker-num 8
  --detokenizer-worker-num 8` booted correctly and made it *worse*: 13.2k vs 17.4k
  at conc 2048 (whole ladder: 6581/10570/11258/15497/13178 vs
  6595/11880/17110/17836/17404).
- **The GPUs are starved, not slow.** Bucketing DP0's scheduler log by second
  shows **60-second stretches with no batch prints at all**, and DP0's last
  scheduler line lands ~2 minutes before the row finishes.
- **Prefill admission is expensive even at ISL=1**: prefill lines read
  `#new-seq: 1, #new-token: 256` at ~1410 input tok/s → **~182 ms per admitted
  request, one sequence at a time**. That is where `p99_ttft_ms 23860` comes from,
  and it is a plausible next place to look — but it was not proven to be the
  ceiling.

So: the client number is real as "what this endpoint delivered", the server number
is real as "what decode cost", the ~4× between them is **unexplained**, and only
the server number is comparable across node counts. Do not publish a mechanism for
the client ceiling until one is measured.

The 2-node ladder was measured twice (once before the server-side instrumentation
existed, once after) and reproduces to ≤5.5% on every row — `decode-sweep-*-n2-sm20.txt`
vs `-v2.txt`.

## Tuning audit against the standalone DeepEP tuning list

Every b300-applicable item from `ep-benchmarks-efa/.../docs/调优改动清单_zh.md` was
already in force in this stack — none of it needed adding, but it needed checking,
because three of the four are silent when absent:

| knob | applies on b300 | actual value here | evidence |
|---|---|---|---|
| `kMaxParts` 4→1 (only source-level win) | yes | **1** | `deep_ep/include/deep_ep/common/gin_resource_alloc.cuh:77` — the image ships the `:parts1` tree |
| `prefer_overlap_with_compute=False` (decode +30% if True) | yes, load-bearing | **False** | hardcoded, `sglang/srt/layers/moe/token_dispatcher/deepep_v2.py:187` |
| `allow_multiple_reduction` (−39% combine) | yes | **True** | sglang never passes it → `elastic.py:141` default |
| `allow_hybrid_mode` (2.8× worse if off) | yes | **True** | `--deepep-v2-mode hybrid` |
| `EP_NUM_SUB_PARTS=1` | **no-op on sm_103** (arch gate) | unset | correct to leave unset |
| `--num-qps` | closed axis | auto | `_resolve_num_sms_qps()` always uses `get_theoretical_num_qps(num_sms)` |

## What was not measured

- **The 1-node cap-256 ladder below 128 req/rank** — only the 128 rung was caught
  before the instances were reclaimed, so cap256 has a floor/slope fit on 2 nodes
  and a single point on 1 node. The matched-capacity ratio (1.33×) is therefore
  established at 128 req/rank only. Never quote 94.4k / 61.5k = 1.53×: that races
  cap 256 against cap 2048.
- **Capacity 1024**, which is the value PR #29525 itself uses, and anything between
  256 and 2048 — so 256 is "better than 2048", not "optimal".
- **A decode row aligned to PR #29525's own protocol** (ISL=1 / OSL=1024 / 128
  prompts / conc 128) for side-by-side with their H20/B200 table.
- **`NUM_SMS=55` on 2-node decode at cap 256.** The floor moved, so the earlier
  reasoning for skipping the SM sweep (below) is now weaker than it was.
- **SM sweep on decode.** Skipped by decision. The standalone b300 regressions put
  12→55 SM at only −10% at 128 tok/rank because the floor dominates, and the floor
  is what the second node inflates here.
- **`a2a=none` decode.** The prefill run already showed `none` is 3.78× worse
  cross-node; the decode question was 1 node vs 2, not a2a attribution.
- **PD disaggregation**, out of scope by request.
