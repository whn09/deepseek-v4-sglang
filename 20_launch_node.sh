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
# Do NOT reach for SGLANG_DSV4_FP4_DEQUANT=1 -- it asserts the runner is `auto`,
# and the deepep_v2 handler always resolves `auto` to a concrete runner first.
#
# Env knobs (defaults in env_common.sh):
#   PHASE=decode|prefill  which of the PR's two published configs to launch.
#   NUM_SMS=20            SGLANG_DEEPEP_V2_NUM_SMS; 0 lets DeepEP derive it.
#   CAPACITY=1024         SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.
#   MODE=hybrid|direct    override the topology chosen from NNODES.
#   A2A=deepep_v2|megamoe the backend under test, or the in-image baseline.
#   KV_DTYPE=auto         --kv-cache-dtype; auto is what the cookbook uses for V4.
#   SWA_RATIO=0.1         --swa-full-tokens-ratio (server default 0.8).
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

require_idle_gpus
require_epv2_image "$IMAGE"
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
    megamoe)
        # Runner is megamoe's own; passing --moe-runner-backend would fight it.
        # Token budget from the cookbook's b300/V4 megamoe rows.
        A2A_ARGS=(--moe-a2a-backend megamoe)
        A2A_ENV=(-e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320)
        ;;
    *) echo "A2A must be deepep_v2 or megamoe" >&2; exit 1 ;;
esac

# Phase configs, from the PR's own repro commands.
case "$PHASE" in
    decode)
        PHASE_ARGS=(--mem-fraction-static 0.70 --cuda-graph-max-bs 128) ;;
    prefill)
        # The deepep_v2 handler already forces the prefill graph off (the
        # contiguous extend path needs a host readback); passing these makes it
        # explicit and also drops the decode graph, matching the PR's prefill row.
        PHASE_ARGS=(--mem-fraction-static 0.80 --disable-cuda-graph --disable-piecewise-cuda-graph) ;;
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
      "${PHASE_ARGS[@]}" \
      ${MULTINODE_ARGS[@]+"${MULTINODE_ARGS[@]}"} \
      ${EXTRA_ARGS:-}
set +x

echo "launched '$NAME' (a2a=$A2A phase=$PHASE mode=$MODE tp=$TP nnodes=$NNODES rank=$NODE_RANK"
echo "          num_sms=$NUM_SMS capacity=$CAPACITY chunked=$CHUNKED kv=$KV_DTYPE swa=$SWA_RATIO gin=$GIN_TYPE)"
echo "follow:  docker logs -f $NAME"
echo "health:  curl -s localhost:${PORT}/health_generate"
