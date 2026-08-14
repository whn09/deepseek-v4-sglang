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
# Env knobs (defaults in env_common.sh):
#   PHASE=decode|prefill  which of the PR's two published configs to launch.
#   NUM_SMS=20            SGLANG_DEEPEP_V2_NUM_SMS; 0 lets DeepEP derive it.
#   CAPACITY=1024         SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.
#   MODE=hybrid|direct    override the topology chosen from NNODES.
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
    -e SGLANG_DEEPEP_V2_NUM_SMS="$NUM_SMS" \
    -e SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$CAPACITY" \
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
      --moe-a2a-backend deepep_v2 --deepep-v2-mode "$MODE" \
      --moe-runner-backend deep_gemm \
      --kv-cache-dtype fp8_e4m3 \
      --chunked-prefill-size "$CHUNKED" \
      "${PHASE_ARGS[@]}" \
      ${MULTINODE_ARGS[@]+"${MULTINODE_ARGS[@]}"} \
      ${EXTRA_ARGS:-}
set +x

echo "launched '$NAME' (phase=$PHASE mode=$MODE tp=$TP nnodes=$NNODES rank=$NODE_RANK"
echo "          num_sms=$NUM_SMS capacity=$CAPACITY chunked=$CHUNKED gin=$GIN_TYPE)"
echo "follow:  docker logs -f $NAME"
echo "health:  curl -s localhost:${PORT}/health_generate"
