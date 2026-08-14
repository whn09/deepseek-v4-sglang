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
| per-rank dispatch capacity | 2048 (`--chunked-prefill-size` = 2048×TP) |
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
tok/s. The denominator is the problem: `output_throughput` is total output tokens
over total wall clock, and on DP0 that wall clock contained **262 `Prefill batch`
prints against 256 `Decode batch` prints** (1 node; 259 vs 256 on 2 nodes). A
2048-prompt row still has 2.1M input tokens to chew, prefill runs ~8× faster per
token than decode so it *looks* negligible per request, but the scheduler
interleaves continuously and the client's wall clock absorbs all of it.

Both tables are real; they answer different questions. Read the client table as
"what one endpoint delivers on a mixed workload at this concurrency", and the
server table as "what decode costs". Only the second is comparable across node
counts.

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

- **SM sweep on decode.** Skipped by decision. The standalone b300 regressions put
  12→55 SM at only −10% at 128 tok/rank because the floor dominates, and the floor
  is what the second node inflates here.
- **`a2a=none` decode.** The prefill run already showed `none` is 3.78× worse
  cross-node; the decode question was 1 node vs 2, not a2a attribution.
- **PD disaggregation**, out of scope by request.
