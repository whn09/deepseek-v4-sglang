#!/bin/bash
# HOST-side launcher: one SGLang server rank-group per node, with PR #29525's
# `--moe-a2a-backend deepep_v2` (DeepEP v2 / ElasticBuffer) over EFA.
#
#   host1: bash 20_launch_node.sh 0
#   host2: bash 20_launch_node.sh 1
#
# Single node (NVLink-only reference, matches the PR's own B200x8 row):
#   NNODES=1 bash 20_launch_node.sh 0
#
# WHAT IS BEING MEASURED: with one node the ElasticBuffer runs `direct` mode and
# the a2a never leaves NVLink, so a single-node number says nothing about EFA.
# Only NNODES=2 (mode `hybrid`) puts the dispatch/combine on the fabric.
#
# EXPERTS ARE MXFP4, and this is a deliberate deviation from the PR. The public
# deepseek-ai/DeepSeek-V4-Flash checkpoint carries `expert_dtype: fp4`: routed
# expert weights are stored as I8 (two fp4 per byte) with F8_E8M0 scales at one
# scale per 32 values -- exactly deep_gemm's `recipe_b=(1,32)` fp4-expert path
# (moe_runner/deep_gemm.py, `quant_info.is_fp4_experts`). sglang auto-detects the
# layout (model_config.py `try_detect_fp4_experts`), so no env is needed. The
# PR's own accuracy table used DeepSeek-V4-Flash-FP8 (FP8 experts), which is not
# publicly readable, so this combination is code-supported but PR-unvalidated:
# if generation comes out garbage, suspect deep_gemm+fp4 before blaming EFA, and
# confirm with A2A=megamoe on the same weights.
#
# ON HOPPER (p5/p5en, sm_90) THERE IS NO FP4 AT ALL, so the above path cannot
# run and SGLANG_DSV4_FP4_DEQUANT=1 stops being an escape hatch and becomes the
# only route: it walks each expert through cast_e2m1fn_to_e4m3fn() at load time
# and leaves plain 128x128 block-scaled FP8, which Hopper deep_gemm serves. Its
# one obstacle -- an `assert runner.is_auto()` that the deepep_v2 handler has
# already made false -- is widened at BUILD time by
# SGLANG_FP4_DEQUANT_ANY_RUNNER=1 (Dockerfile section 6b, ARCH=sm90). This
# script turns the env var on automatically below when the GPUs are pre-Blackwell
# and refuses to launch if the image was not built for it, because the failure
# mode otherwise is a load-time assert 40 GB into weight loading.
# Set FP4_DEQUANT=0 to force the stock behaviour.
#
# Env knobs (defaults in env_common.sh):
#   PHASE=decode|prefill  which of the PR's two published configs to launch.
#   NUM_SMS=20            SGLANG_DEEPEP_V2_NUM_SMS; 0 lets DeepEP derive it.
#   CAPACITY=1024         SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.
#   MODE=hybrid|direct    override the topology chosen from NNODES.
#   A2A=deepep_v2|megamoe the backend under test, or the in-image baseline.
#   KV_DTYPE=auto         --kv-cache-dtype; auto is what the cookbook uses for V4.
#   SWA_RATIO=0.1         --swa-full-tokens-ratio (server default 0.8).
#   RUNNING_PER_RANK=32   per-rank concurrency slots; MAX_RUNNING=this*TP.
#   MEM_FRACTION          --mem-fraction-static; see the 80 GB note below.
#   GDAKI=1  TP  NNODES  PORT  NCCL_DEBUG  EXTRA_ARGS  EXTRA_ENV
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NODE_RANK="${1:?node rank (0=leader, 1=worker)}"
NAME="${NAME:-dsv4-epv2}"
PHASE="${PHASE:-decode}"
NUM_SMS="${NUM_SMS:-20}"
CAPACITY="${CAPACITY:-1024}"
MODE="${MODE:-$([[ "$NNODES" -gt 1 ]] && echo hybrid || echo direct)}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
A2A="${A2A:-deepep_v2}"
# The sglang cookbook's DeepSeek-V4 rows never pass --kv-cache-dtype, and V4
# carries its own compressed-KV + DSA indexer path, so `auto` (= let the model
# config decide) is the faithful default. The PR's own accuracy table ran kv fp8
# on the FP8-expert checkpoint; set KV_DTYPE=fp8_e4m3 to reproduce that exactly.
# Either way the deepep_v2 decode-graph gate keys off the DISPATCHER dtype and the
# runner, not the KV dtype, so this does not silently drop CUDA graphs.
KV_DTYPE="${KV_DTYPE:-auto}"
# Server default is 0.8; every b300 V4 row in the cookbook that sets it uses 0.1,
# which buys KV headroom on the SWA layers (config.json: sliding_window=128).
SWA_RATIO="${SWA_RATIO:-0.1}"

# --chunked-prefill-size is a GLOBAL budget that _handle_data_parallelism divides
# by dp_size under DP attention; server_args then refuses to boot if the per-rank
# result exceeds CAPACITY (server_args.py, the deepep_v2 handler). dp_size == TP
# here, so this is the largest legal value. Must stay a multiple of page_size.
CHUNKED="${CHUNKED:-$((CAPACITY * TP))}"

# --max-running-requests is ALSO a global budget that DP attention divides by
# dp_size. V4 auto-sets it to 256, so at dp=8 each rank admits 32 and at dp=16
# each rank admits only 16 -- i.e. the total concurrency ceiling would stay 256
# no matter how many nodes you add, and "2 nodes serve more concurrency" would be
# untestable for a purely configural reason. Pinning PER-RANK slots instead makes
# node count the only variable: 32*TP -> 256 on one node, 512 on two.
#
# BUT KNOW WHAT THIS PINS AWAY: with slots fixed, "how much concurrency can this
# shape hold" is no longer being measured -- it is being asserted. MEASURED on p5
# (docs/p5_48xlarge_实测报告_zh.md §6d): at RUNNING_PER_RANK=32 the runs used only
# 1.7-12.2% of the KV pool, and the pool itself is 2.26-2.95x LARGER per rank on
# two nodes than on one -- superlinear, because ep16 halves the local expert count
# and the 15.1 GB/GPU it frees all becomes KV. So the 1.5-1.7x throughput scaling
# and a ~2.9x capacity scaling coexist; they measure different things. To measure
# the capacity axis instead, raise this until TPOT p99 breaks or the scheduler
# starts retracting, and move client concurrency with it (= RUNNING_PER_RANK*TP).
RUNNING_PER_RANK="${RUNNING_PER_RANK:-32}"
MAX_RUNNING="${MAX_RUNNING:-$((RUNNING_PER_RANK * TP))}"

require_idle_gpus
require_epv2_image "$IMAGE"

# --- MXFP4 experts vs. pre-Blackwell GPUs -----------------------------------
# compute_cap is "9.0" on H100/H200, "10.3" on b300. Anything below 10 has no
# FP4 arithmetic, so the routed experts must be dequantised to FP8 at load time.
SM_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
SM_MAJOR="${SM_CAP%%.*}"
FP4_DEQUANT="${FP4_DEQUANT:-$([[ -n "$SM_MAJOR" && "$SM_MAJOR" -lt 10 ]] && echo 1 || echo 0)}"
if [[ "$FP4_DEQUANT" == "1" ]]; then
    echo "=== compute_cap=$SM_CAP (pre-Blackwell): SGLANG_DSV4_FP4_DEQUANT=1 ==="
    # Fail HERE, not 40 GB into weight loading. An image built without
    # SGLANG_FP4_DEQUANT_ANY_RUNNER=1 still has the `assert runner.is_auto()`,
    # which deepep_v2 has already made false.
    built="$(docker run --rm --entrypoint printenv "$IMAGE" \
                SGLANG_FP4_DEQUANT_ANY_RUNNER 2>/dev/null || true)"
    [[ "$built" == "1" ]] || {
        echo "ERROR: $IMAGE was built with SGLANG_FP4_DEQUANT_ANY_RUNNER=${built:-<unset>}." >&2
        echo "       Rebuild it with: ARCH=sm90 bash 10_build_image.sh" >&2
        exit 1
    }
    EXTRA_ENV="${EXTRA_ENV:-} SGLANG_DSV4_FP4_DEQUANT=1"
fi
build_cache_args
build_gdr_args
detect_ep_nic

for dev in /dev/infiniband; do
    [[ -e "$dev" ]] || { echo "MISSING $dev on this host." >&2; exit 1; }
done
[[ -d "$HOST_MODEL_DIR/$MODEL_NAME" ]] \
    || { echo "MISSING $HOST_MODEL_DIR/$MODEL_NAME -- run 00_download_model.sh" >&2; exit 1; }

# NCCL_GIN_TYPE, from the amazon NCCL fork's nccl_device/core.h:
#   0 NONE   2 PROXY   3 GDAKI(IB/DOCA)   4 GPI   5 EFA_GDA
# 5 exists only in amazon-contributing/upstream-to-nccl; NVIDIA's stops at 4.
GIN_ARGS=()
if [[ "$GDAKI" == "1" ]]; then
    GIN_TYPE=5
    echo "=== Route B: NCCL_GIN_TYPE=5 (EFA GDA) ==="
    # FI_EFA_USE_HW_CNTR=1 arms the hardware counter the counting-event path
    # reads; without it GIN init fails in a way that looks like a host prereq.
    GIN_ARGS+=(-e FI_EFA_USE_HW_CNTR=1 -e OFI_NCCL_GIN_STRONG_SIGNAL=1 -e NCCL_RMA_DISABLE=1)
else
    GIN_TYPE=2
    echo "=== Route A: NCCL_GIN_TYPE=2 (proxy GIN) ==="
fi

# A2A backend. deepep_v2 is the thing under test; megamoe is the attribution
# baseline that works in THIS image -- DeepEP v1 does not, because the v2 fork
# installs over the same `deep_ep` module name and 10_build_image.sh uninstalls
# sgl-deep-ep. So a v1 comparison needs the other kit's image, not this one.
case "$A2A" in
    deepep_v2)
        A2A_ARGS=(--moe-a2a-backend deepep_v2 --deepep-v2-mode "$MODE"
                  --moe-runner-backend deep_gemm)
        A2A_ENV=(-e SGLANG_DEEPEP_V2_NUM_SMS="$NUM_SMS"
                 -e SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$CAPACITY")
        ;;
    none)
        # The only NON-DeepEP a2a that actually crosses nodes in this image, and
        # the cleanest possible control: EP geometry, weights, runner and per-rank
        # chunk all stay identical, and the ONLY thing that changes is that the
        # ElasticBuffer is replaced by sglang's all-gather StandardDispatcher.
        # Keeping --ep $TP is deliberate -- dropping it would switch EP-MoE for
        # TP-MoE and change two variables at once.
        A2A_ARGS=(--moe-a2a-backend none --moe-runner-backend deep_gemm)
        A2A_ENV=()
        ;;
    megamoe)
        # MEASURED: megamoe CANNOT run across nodes in this build. Its rendezvous
        # is torch.distributed._symmetric_memory, which passes POSIX fds over a
        # unix socket, so at nnodes=2 rank 0 dies with
        #   _symmetric_memory/__init__.py:2004 rendezvous
        #   RuntimeError: Failed to send fd: No such file or directory
        # right after the DeepGEMM warmup. Cross-node symmetric memory needs
        # FABRIC handles, which this path does not use. Keep it for the
        # SINGLE-NODE attribution baseline only.
        # Runner is megamoe's own; passing --moe-runner-backend would fight it.
        # Token budget from the cookbook's b300/V4 megamoe rows.
        A2A_ARGS=(--moe-a2a-backend megamoe)
        A2A_ENV=(-e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320)
        [[ "$NNODES" -gt 1 ]] && echo "WARNING: megamoe is intra-node only; this will die in symmetric-memory rendezvous." >&2
        ;;
    *) echo "A2A must be deepep_v2, none or megamoe" >&2; exit 1 ;;
esac

# --mem-fraction-static bounds the WEIGHTS+KV pool only; the DeepEP v2
# ElasticBuffer is cudaMalloc'd outside it, so on an 80 GB card the two collide.
# MEASURED on p5.48xlarge (79.18 GB usable, weights 27.18 GB/GPU after the FP4->FP8
# dequant): at 0.80 the KV pool takes 35 GB (max_total_num_tokens=4.8M), leaves
# 15.30 GB, and CAPACITY=8192 dies asking for 16.00 GiB. The buffer scales with
# CAPACITY (~2 MiB/token/rank), so prefill needs the lower fraction, not the lower
# capacity -- 4.8M KV tokens is already ~2x what ISL 4096 x conc 512 can use.
# b300 (288 GB) never hits this, which is why the default is per-arch.
MEM_TOTAL_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
case "$PHASE" in
    decode)
        # --cuda-graph-max-bs is accepted but deprecated in this build ("use
        # --cuda-graph-max-bs-decode instead"), and it only ever set the decode
        # phase anyway.
        MEM_FRACTION="${MEM_FRACTION:-0.70}"
        PHASE_ARGS=(--mem-fraction-static "$MEM_FRACTION" --cuda-graph-max-bs-decode 128) ;;
    prefill)
        # The deepep_v2 handler already forces the prefill graph off (the
        # contiguous extend path needs a host readback), verified by
        # 12_preflight_args.sh -> `graph.prefill = disabled`. So only the decode
        # graph needs dropping here; --disable-piecewise-cuda-graph would be
        # redundant AND deprecated ("use --cuda-graph-backend-prefill=disabled").
        MEM_FRACTION="${MEM_FRACTION:-$([[ -n "$MEM_TOTAL_GB" && "$MEM_TOTAL_GB" -lt 120000 ]] && echo 0.65 || echo 0.80)}"
        PHASE_ARGS=(--mem-fraction-static "$MEM_FRACTION" --disable-cuda-graph) ;;
    *) echo "PHASE must be decode or prefill" >&2; exit 1 ;;
esac

MULTINODE_ARGS=()
if [[ "$NNODES" -gt 1 ]]; then
    MULTINODE_ARGS=(--nnodes "$NNODES" --node-rank "$NODE_RANK"
                    --dist-init-addr "${MASTER_IP}:${DIST_PORT}"
                    --enable-dp-lm-head)
fi

# EXTRA_ENV: whitespace-separated NAME=VALUE list for one-off knobs. Word
# splitting is INTENTIONAL (that is how it becomes separate -e pairs), so values
# must not contain spaces.
EXTRA_ENV_ARGS=""
for kv in ${EXTRA_ENV:-}; do EXTRA_ENV_ARGS="${EXTRA_ENV_ARGS} -e ${kv}"; done

# Unmapping a large model volume can outlast a single `rm -f`, leaving the
# container Exited-but-present and the next `docker run` failing on the name.
for _ in $(seq 1 30); do
    docker inspect "$NAME" >/dev/null 2>&1 || break
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    sleep 2
done

set -x
docker run -d --name "$NAME" \
    --gpus all --privileged --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --device=/dev/infiniband ${GDR_ARGS[@]+"${GDR_ARGS[@]}"} \
    --shm-size=200g \
    -v "$HOST_MODEL_DIR/$MODEL_NAME:$MODEL_PATH:ro" \
    "${CACHE_ARGS[@]}" \
    -e NCCL_GIN_TYPE="$GIN_TYPE" \
    "${GIN_ARGS[@]}" \
    -e NCCL_DEBUG="$NCCL_DEBUG" \
    "${A2A_ENV[@]}" \
    ${EP_NIC_NAME:+-e EP_NIC_NAME=$EP_NIC_NAME} \
    ${EXTRA_ENV_ARGS} \
    -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1 \
    --entrypoint python3 \
    "$IMAGE" \
    -m sglang.launch_server \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --trust-remote-code \
      --host 0.0.0.0 --port "$PORT" \
      --tp "$TP" --dp "$TP" --ep "$TP" --enable-dp-attention \
      "${A2A_ARGS[@]}" \
      ${KV_DTYPE:+--kv-cache-dtype "$KV_DTYPE"} \
      --swa-full-tokens-ratio "$SWA_RATIO" \
      --chunked-prefill-size "$CHUNKED" \
      --max-running-requests "$MAX_RUNNING" \
      "${PHASE_ARGS[@]}" \
      ${MULTINODE_ARGS[@]+"${MULTINODE_ARGS[@]}"} \
      ${EXTRA_ARGS:-}
set +x

echo "launched '$NAME' (a2a=$A2A phase=$PHASE mode=$MODE tp=$TP nnodes=$NNODES rank=$NODE_RANK"
echo "          num_sms=$NUM_SMS capacity=$CAPACITY chunked=$CHUNKED kv=$KV_DTYPE swa=$SWA_RATIO gin=$GIN_TYPE mem_frac=$MEM_FRACTION"
echo "          max_running=$MAX_RUNNING (= $RUNNING_PER_RANK per rank))"
echo "follow:  docker logs -f $NAME"
echo "health:  curl -s localhost:${PORT}/health_generate"
