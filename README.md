# deepseek-v4-sglang

DeepSeek-V4-Flash on SGLang with **DeepEP v2 (ElasticBuffer) MoE all-to-all over AWS EFA**,
on 2× `p6-b300` (B300, sm_103, 8 GPU/node, 16× EFA @400 Gb/s).

The thing under test is [sglang PR #29525](https://github.com/sgl-project/sglang/pull/29525)
— `[Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend` — pinned at
`b2d379cc72793bb220d6d79062d6331284c843ba`. The PR's own numbers are H20/B200 on
NVLink or IB; the open question is what it does on EFA.

## Quick start

```bash
bash sync.sh                       # laptop -> both hosts
# on EACH host (nvme is ephemeral, no broadcast between ranks):
bash 00_download_model.sh          # 149 GB, ~5 min at Xet speed
bash 10_build_image.sh             # CPU-only, ~6 min, safe while GPUs are busy
# 2-node run, leader first:
ssh P6-B300-1 'cd deepseek-v4-sglang && bash 20_launch_node.sh 0'
ssh P6-B300-2 'cd deepseek-v4-sglang && bash 20_launch_node.sh 1'
bash 90_smoke_test.sh              # on the leader
bash 91_bench.sh
```

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

V4-Flash also happens to be *already* FP8 on disk — `fmt: e4m3`, `scale_fmt: ue8m0`,
`weight_block_size: [128,128]` — which is what `--moe-runner-backend deep_gemm
--kv-cache-dtype fp8_e4m3` wants, so there is no quantization step.

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

New knobs: `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (default 128),
`SGLANG_DEEPEP_V2_NUM_SMS` (0 = let DeepEP derive it), `--deepep-v2-mode
{direct,hybrid}`, `--deepep-v2-dispatcher-output-dtype {auto,bf16,fp8}`.

`NCCL_CUMEM_ENABLE=1` is set in the image: without it the PR dies with
`nccl.cu:104: Communicator does not support symmetric memory!`.

## Files

| | |
|---|---|
| `env_common.sh` | paths, IPs, topology, NIC autodetect, preflight guards |
| `00_download_model.sh` | weights → host NVMe (re-run after every instance restart) |
| `10_build_image.sh` | builds out of `$BUILD_CTX_HOST` (see below) |
| `20_launch_node.sh` | `bash 20_launch_node.sh 0\|1`; `PHASE=decode\|prefill` |
| `90_smoke_test.sh` | health + "did it really take the deepep_v2 and GDA paths" |
| `91_bench.sh` | `sglang.bench_serving`, results tagged with the knobs |
| `Dockerfile` | copied into the build context by `10_build_image.sh` |

`10_build_image.sh` builds from `/home/ubuntu/work/deepep-v2-efa-gdaki` rather
than this repo because two required artifacts cannot be committed: the 627 MB EFA
dev-channel tarball and the private DeepEP fork. Both are already staged there by
the `ep-benchmarks-efa` kit.

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
