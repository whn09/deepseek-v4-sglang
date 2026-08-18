# deepseek-v4-sglang

DeepSeek-V4-Flash on SGLang with **DeepEP v2 (ElasticBuffer) MoE all-to-all over AWS EFA**,
on 2× `p6-b300` (B300, sm_103, 8 GPU/node, 16× EFA @400 Gb/s), and — via
`source env_p5.sh` + `ARCH=sm90` — on 2× `p5.48xlarge` (H100, sm_90, 32× EFA
@100 Gb/s, no GDAKI). See **Results** below for both.

The thing under test is [sglang PR #29525](https://github.com/sgl-project/sglang/pull/29525)
— `[Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend` — pinned at
`b2d379cc72793bb220d6d79062d6331284c843ba`. The PR's own numbers are H20/B200 on
NVLink or IB; the open question is what it does on EFA.

## Quick start

```bash
bash sync.sh                       # laptop -> both hosts
# on EACH host (nvme is ephemeral, no broadcast between ranks):
bash 00_download_model.sh          # 149 GB, ~5 min at Xet speed
# build ONCE, then ECR-sync -- never build per node (see "Image distribution"):
ssh P6-B300-1 'cd deepseek-v4-sglang && bash 10_build_image.sh && bash 11_sync_image_ecr.sh push'
ssh P6-B300-2 'cd deepseek-v4-sglang && bash 11_sync_image_ecr.sh pull'
bash 11_sync_image_ecr.sh verify   # one image id on every host, or stop
# 2-node run, leader first:
ssh P6-B300-1 'cd deepseek-v4-sglang && bash 20_launch_node.sh 0'
ssh P6-B300-2 'cd deepseek-v4-sglang && bash 20_launch_node.sh 1'
bash 90_smoke_test.sh              # on the leader
bash 91_bench.sh
```

## Results

| doc | shape | question it answers | headline |
|---|---|---|---|
| [`results/RESULTS_prefill.md`](results/RESULTS_prefill.md) | 2× p6-b300, GDAKI GIN | does 2 nodes beat 2× one node on input throughput? | no — **1.62×** (274.4k vs 169.1k tok/s at c1024, p99 TTFT 14.3 s vs 23.2 s). Per-rank chunk, not SM, is the strongest knob |
| [`results/RESULTS_decode.md`](results/RESULTS_decode.md) | 2× p6-b300, GDAKI GIN | what does the second node cost decode? | server step **16.65 → 25.95 ms** at 128 req/rank (1.56×), aggregate only 1.28×. `CAPACITY` 2048→256 recovers half of it |
| [`docs/p5_48xlarge_实测报告_zh.md`](docs/p5_48xlarge_实测报告_zh.md) (中文) | 2× p5.48xlarge, **proxy GIN** | does any of this run on Hopper/EFA gen-1, and is DeepEP v2 worth it there? | yes, once FP4 is dequantised at load time. prefill **137.5k tok/s** at c512 (TTFT mean 7 915 ms) = **3.43× `a2a=none`**; decode step **33.78 ms** = **21–29% SLOWER** than `a2a=none` |

The two shapes are **not** directly comparable: p5 caps `CAPACITY` at 4096 (80 GB
memory ceiling, see that report's §4) against b300's 8192, and p5 has no GDAKI, so
arch + a2a capacity + GIN type all differ. The reusable finding is the *sign* of
each effect, not the ratio between shapes.

## Why DeepSeek-V4-Flash and not Kimi-K3

The PR wires DeepEP v2 into `deepseek_v2.DeepseekV2MoE` — it adds `is_deepep_v2()`
to that file's existing backend gates and nothing else. `deepseek_v4.py:1483`
instantiates exactly that class, so V4-Flash is on the patched path for free.

`kimi_k3.py` is not: it defines its own `KimiK3MoE`, imports only
`DeepseekV2AttentionMLA` and `MoEGate` from deepseek_v2, and gates the a2a path on
`_a2a_backend.is_megamoe() or _a2a_backend.is_deepep()` (lines 491 and 1969) —
both **False** for `deepep_v2`. The PR does not touch the file, so `--moe-a2a-backend
deepep_v2` on K3 silently runs the dense path. Wiring K3 would mean handling its
latent MoE (`routed_expert_hidden_size`), the fused `_front_w`, and `situ`
activation as well.

This is *not* an FP4 problem. The on-disk Kimi-K3 has no `quantization_config` at
all (`dtype: bfloat16`). And dispatch dtype ≠ weight dtype: DeepEP v2 moves BF16
or FP8 payloads (`use_fp8_dispatch`; the only FP4 mention in the fork is
`deep_ep/buffers/elastic.py:305  # TODO: consider FP4`), while the GEMM dtype is
the MoE runner's business. FP4 weights would dispatch FP8 and still work.

## Quantization: the experts really are FP4

`config.json` advertises `quantization_config: {quant_method: fp8, fmt: e4m3,
scale_fmt: ue8m0, weight_block_size: [128,128]}` — but that covers the dense and
attention weights. The routed experts are separate and **MXFP4**:

```
config.json                     expert_dtype: fp4
layers.0.attn.wo_a.weight       F8_E4M3  [8192, 4096]
layers.0.ffn.experts.0.w1.weight  I8     [2048, 2048]   # 2 fp4 per byte -> K=4096
layers.0.ffn.experts.0.w1.scale F8_E8M0  [2048,  128]   # 4096/128 = 32/scale
```

One E8M0 scale per 32 values is exactly `deep_gemm`'s fp4-expert recipe —
`moe_runner/deep_gemm.py` sets `recipe_a, recipe_b = (1,128), (1,32)` when
`quant_info.is_fp4_experts`. sglang auto-detects the layout
(`model_config.py::try_detect_fp4_experts`), so **no env var is needed** and
`--moe-runner-backend deep_gemm` is a valid pairing: FP8 activations at 1×128,
FP4 weights at 1×32.

Two things follow, and both matter for reading a result:

- **This exact pairing is code-supported but PR-unvalidated.** The PR's accuracy
  table ran `DeepSeek-V4-Flash-FP8` (FP8 experts); that repo 401s for us. So if
  output quality is bad, suspect `deep_gemm` + fp4 experts before blaming EFA, and
  cross-check with `A2A=megamoe` on the same weights.
- **`SGLANG_DSV4_FP4_DEQUANT=1` is blocked by one assert, not by design** — and on
  pre-Blackwell GPUs it is the *only* route, so the assert has to be widened.
  Stock, it asserts `get_moe_runner_backend().is_auto()`, and the deepep_v2 handler
  resolves `auto` → `deep_gemm` before quant methods are built, so it always fires.
  The three branches it guards are the *specialised mxfp4* runners (marlin /
  humming / flashinfer_mxfp4), and after dequant `is_fp4_expert=False` takes none
  of them — i.e. the assert's text is wider than its intent. `ARCH=sm90` widens it
  at build time (`SGLANG_FP4_DEQUANT_ANY_RUNNER=1`, Dockerfile section 6b) and
  `20_launch_node.sh` turns the env var on automatically below compute_cap 10.
  **MEASURED end to end on 2× p5.48xlarge** — see
  [`docs/p5_48xlarge_实测报告_zh.md`](docs/p5_48xlarge_实测报告_zh.md).
  b300 images are byte-identical without `ARCH=sm90`.

`wo_a` is already `F8_E4M3`, so the PR's `SGLANG_OPT_FP8_WO_A_GEMM=0` caveat
("Blackwell with a BF16 `wo_a` checkpoint") does **not** apply.

## What "on EFA" requires

**Single node proves nothing about EFA.** With `NNODES=1` the ElasticBuffer runs
`--deepep-v2-mode direct` and the entire a2a stays on NVLink — that is what the
PR's B200×8 row measures. Only `NNODES=2` (→ `hybrid`) puts dispatch/combine on
the fabric. `20_launch_node.sh` derives the mode from `NNODES` so this cannot be
got wrong by accident.

Two transport routes, selected by `GDAKI`:

| `GDAKI` | `NCCL_GIN_TYPE` | what runs |
|---|---|---|
| `1` (default) | `5` = `EFA_GDA` | GPU writes a 64 B WQE and rings the MMIO doorbell itself; zero CPU |
| `0` | `2` = `PROXY` | CPU proxy thread posts the sends |

`NCCL_GIN_TYPE=5` exists only in `amazon-contributing/upstream-to-nccl` (NVIDIA's
enum stops at 4 = GPI), which is why the image builds NCCL from source. IBGDA can
never run on EFA — different doorbell semantics — so route B is the EFA-native
equivalent, not a variant of it.

## The image

`Dockerfile` (built by `10_build_image.sh`) starts from
`lmsysorg/sglang:nightly-dev-20260811-d59c1ddf` and layers:

1. purge the base's `sgl-deep-ep`, **both** libnccl providers (apt `libnccl2`
   2.28.3 — on HOLD — and pip `nvidia-nccl-cu13` 2.29.7)
2. EFA installer **1.50.0** (dev channel; first version whose libfabric exposes
   `cntr_open_ext`, asserted at build time)
3. GDRCopy 2.5.2
4. amazon NCCL fork from source, asserting `NCCL_GIN_TYPE_EFA_GDA` exists
5. aws-ofi-nccl with `--enable-platform-aws`, recording `OFI_GDAKI=1|0`
6. the private DeepEP v2 EFA fork, with `kMaxParts` patched 4→1
7. sglang PR #29525 fetched over `refs/pull/29525/head`

`/etc/deepep-build-env` records what the build actually decided; `env_common.sh`'s
`require_epv2_image` reads it and refuses to launch a proxy-only image under
`GDAKI=1`, rather than letting GIN init fail 4 minutes into a weight load.

Two base-image gotchas worth knowing, both already handled:

- `apt-get remove libnccl2` silently no-ops without `--allow-change-held-packages`
  (the packages are `hi`). The real gate is
  `! ldconfig -p | grep -q 'libnccl\.so\.2'`.
- The published image bakes the expired GitHub Actions token that built it into
  `.git/config` as `http.https://github.com/.extraheader`, so every fetch 401s and
  git then prompts, failing with the misleading *"could not read Username for
  https://github.com"*. The build unsets it.

`kMaxParts=1` is the one source-level win on b300 (decode dispatch 2.3× on the
DeepEP microbenchmarks) and must be patched **before** the build — it reaches the
host `.so` through `kScaleoutSlotRoundingReserve`, so it cannot be flipped at
runtime. `DEEPEP_KMAXPARTS=4` builds the stock comparison image.

## Launch contract (from `server_args.py`'s deepep_v2 handler)

Violating any of these makes the server refuse to boot, or boot into a fallback:

- runner must resolve to `deep_gemm` or `triton` (`auto` → `triton` if the
  dispatcher dtype is bf16, else `deep_gemm`)
- `--enable-two-batch-overlap` / `--enable-single-batch-overlap` rejected
- `--enforce-shared-experts-fusion` rejected
- per-rank chunked-prefill budget (CLI value ÷ `dp_size` under DP attention) must
  be ≤ `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`. `20_launch_node.sh`
  sets `CHUNKED=CAPACITY*TP`, the largest legal value.
- decode CUDA graph only on deep_gemm+fp8; prefill graph always off
- `ep_size` forced to `tp_size`

### What the resolution actually produces

`12_preflight_args.sh` runs the exact argv through `ServerArgs` in a **CPU-only**
container (~20 s) instead of discovering a rejection after 149 GB has loaded on 16
ranks. Because no accelerator is visible it needs `--device cuda` explicitly, or
`_handle_missing_default_values()` dies in `get_device()`. Output for the run under
test (`deepep_v2` / `hybrid` / decode, tp=dp=ep=16):

```
Auto-detected DSV4 routed-expert layout: is_fp4_experts=True
Hybrid SWA model detected. architectures=['DeepseekV4ForCausalLM']
Use dsv4 attention backend for DeepseekV4ForCausalLM, setting page_size to 256.
Setting KV cache dtype to fp8_e4m3 for DeepseekV4ForCausalLM.
Setting max_running_requests to 256 for DeepseekV4ForCausalLM.
DP attention is enabled. chunked prefill size is adjusted from 16384 to 1024.
DeepEP v2 MoE is enabled. The expert parallel size is adjusted ... [16].
  page_size = 256      chunked_prefill_size = 1024   kv_cache_dtype = fp8_e4m3
  ep=tp=dp = 16        disable_shared_experts_fusion = True
  graph.decode = full max_bs=128        graph.prefill = disabled
```

Three things to read carefully there:

- **`--chunked-prefill-size` is global; the resolved field is per-rank.** 16384 → 1024
  is `/ dp_size`, not a clamp. That per-rank number must be ≤ capacity **and** a
  multiple of `page_size`; 1024 = capacity = 4×256. ✓
- **`page_size` is 256**, chosen by the dsv4 attention backend — not the usual 1/64.
- **The fp4-expert layout is auto-detected**, so no `SGLANG_DSV4_FP4_EXPERTS` needed.

New knobs: `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (default 128),
`SGLANG_DEEPEP_V2_NUM_SMS` (0 = let DeepEP derive it), `--deepep-v2-mode
{direct,hybrid}`, `--deepep-v2-dispatcher-output-dtype {auto,bf16,fp8}`.

`NCCL_CUMEM_ENABLE=1` is set in the image: without it the PR dies with
`nccl.cu:104: Communicator does not support symmetric memory!`.

The PR also requires **NCCL ≥ 2.30** (that is where the GIN headers the elastic
backend needs land). The image builds `2.31.1` from the amazon fork, so this holds
— worth re-asserting after any `NCCL_REF` bump, since the base image's own
providers were 2.28.3/2.29.7.

## Cross-check against the sglang cookbook

The [cookbook's DeepSeek-V4 matrix](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
has no `deepep_v2` rows (the PR is unmerged), but its b300 rows pin the
model-level knobs. What it says vs. what this kit does:

| knob | cookbook (b300, V4-Flash) | here | why |
|---|---|---|---|
| per-rank dispatch capacity | `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` | `..._V2_...=1024` | same number, v2's env name |
| `num_sms` | `--deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'` | `SGLANG_DEEPEP_V2_NUM_SMS=20` | v1-only flag. 96 is unreachable for v2-on-EFA anyway: the GIN fork clamps at `kMaxSM=(256−2×17)/4=55`. Sweep 20/32/55. |
| `--swa-full-tokens-ratio` | `0.1` | `0.1` (`SWA_RATIO`) | **was missing**; server default is 0.8, and V4 has `sliding_window: 128` |
| `--kv-cache-dtype` | not passed (auto) | `auto` (`KV_DTYPE`) | was pinned to `fp8_e4m3`, which the preflight shows is **redundant, not wrong**: V4 forces it (`Setting KV cache dtype to fp8_e4m3 for DeepseekV4ForCausalLM`). `auto` lets the model declare it. |
| `--tp` | `4` (Flash fits in 4 GPUs) | `16` (2×8) | tp4 is single-node, i.e. NVLink-only. EP=16 needs 16 ranks, which is the point. |
| runner | `flashinfer_mxfp4` / `megamoe` / `humming` | `deep_gemm` | deepep_v2 accepts only `deep_gemm`/`triton`; deep_gemm has the fp4-expert recipe |
| spec decode | `DSPARK` / EAGLE | off | keeps the a2a measurement clean; MTP+deepep_v2 is untested |
| `SGLANG_DSV4_FP4_EXPERTS` | `0` only on h200 (sm90 has no FP4) | unset → auto-detect | b300 is sm103; the fp4 path is the fast one. On sm90 the cookbook's `=0` is not enough by itself — that only *stops* the fp4 path, and the weights are still MXFP4; the working route is `SGLANG_DSV4_FP4_DEQUANT=1` (auto-set by `20_launch_node.sh`) on an `ARCH=sm90` image |

Also in the cookbook and deliberately skipped: `--reasoning-parser deepseek-v4`,
`--tool-call-parser deepseekv4` (serving ergonomics, irrelevant to a2a numbers),
and `--cuda-graph-max-bs-decode` (this build accepts `--cuda-graph-max-bs` as its
alias — checked against `--help`, not assumed).

## Files

| | |
|---|---|
| `env_common.sh` | paths, IPs, topology, NIC autodetect, preflight guards |
| `00_download_model.sh` | weights → host NVMe (re-run after every instance restart) |
| `10_build_image.sh` | builds out of `$BUILD_CTX_HOST` (see below) |
| `11_sync_image_ecr.sh` | `push` / `pull` / `verify` — one image digest on every rank |
| `12_preflight_args.sh` | resolve the argv through `ServerArgs`, CPU-only, no server |
| `20_launch_node.sh` | `bash 20_launch_node.sh 0\|1`; `PHASE=decode\|prefill` |
| `90_smoke_test.sh` | health + "did it really take the deepep_v2 and GDA paths" |
| `91_bench.sh` | `sglang.bench_serving`, results tagged with the knobs |
| `Dockerfile` | copied into the build context by `10_build_image.sh` |
| `sync.sh` / `push.sh` | laptop → hosts, and laptop → GitHub via the jump host |

`10_build_image.sh` builds from `/home/ubuntu/work/deepep-v2-efa-gdaki` rather
than this repo because two required artifacts cannot be committed: the 627 MB EFA
dev-channel tarball and the private DeepEP fork. Both are already staged there by
the `ep-benchmarks-efa` kit.

### Image distribution — build once, ECR-sync

`579019700964.dkr.ecr.us-west-2.amazonaws.com/deepseek-v4-sglang:pr29525`.

**Do not build the image separately on each node.** The Dockerfile pins the
amazon NCCL fork to a *branch* (`staging`) and aws-ofi-nccl to `master`, and
apt/pip resolve latest at build time — so two builds hours apart can put two
different NCCL/libfabric stacks into one job. That failure mode is a hang or a
plausible-but-wrong number, not an error message. `11_sync_image_ecr.sh verify`
compares the local `$IMAGE` id (not the ECR tag, which a stale local build can
shadow) across hosts and exits non-zero on a mismatch.

## Host notes

- `P6-B300-1` `172.31.57.229` · `P6-B300-2` `172.31.61.182` — **re-check after any
  instance restart**, or NCCL bootstrap just times out.
- `/opt/dlami/nvme` is ephemeral: weights and caches are gone after a stop/start.
- `/dev/gdrdrv` has no udev rule and DKMS must be rebuilt after a kernel upgrade;
  `build_gdr_args` warns instead of failing, since a missing `--device` makes
  `docker run` itself fail.
- b300 exposes 2× `ibp19{8,9}s0f0` at 100 Gb/s alongside the 16× `rdmap*` at
  400 Gb/s. Naming a slow one in `EP_NIC_NAME` makes DeepEP's
  `get_theoretical_num_sms()` mis-size the kernel; `detect_ep_nic` picks the
  fastest.
