#!/bin/bash
# a2a low-latency microbench runner (bench/bench_a2a_ll.py).
#
# One invocation = one (eplib, backend, shape, nnodes, cap-policy) point, sweeping
# --num-tokens inside the process. Run it on EVERY node of the job; rank 0 writes
# the JSON.
#
#   ELIB=deepep_v1 SHAPE=dsv4 bash 96_a2a_microbench.sh            # 1 node
#   ELIB=uccl      SHAPE=dsv3 NNODES=2 NODE_RANK=0 bash 96_...      # on P5EN-1
#   ELIB=uccl      SHAPE=dsv3 NNODES=2 NODE_RANK=1 bash 96_...      # on P5EN-2
#
# Knobs: ELIB SHAPE NNODES NODE_RANK CAP_POLICY CAPACITY NUM_SMS TOKENS SPLIT
#
# The GPUs must be IDLE -- stop the sglang servers first. This is a full-GPU
# benchmark, not a probe, so it does not fall under the
# "no probes in live containers" rule; it falls under "do not contend".
set -euo pipefail
cd "$(dirname "$0")"

# Capture the caller's NNODES BEFORE sourcing: env_common.sh defaults it to 2
# (the server launchers are multi-node by default), so a plain
# `NNODES="${NNODES:-1}"` after the source silently inherits 2 and the run hangs
# on NCCL bootstrap waiting for a rank that was never started -- with an `-n2`
# stamp on the output file, which is at least honest about what it tried to do.
_NNODES_REQ="${NNODES:-}"
source env_common.sh

ELIB="${ELIB:?set ELIB=deepep_v1|uccl|deepep_v2}"
SHAPE="${SHAPE:-dsv4}"
NNODES="${_NNODES_REQ:-1}"
NODE_RANK="${NODE_RANK:-0}"
CAP_POLICY="${CAP_POLICY:-bench-native}"   # bench-native | sglang
CAPACITY="${CAPACITY:-1024}"               # only read when CAP_POLICY=sglang
NUM_SMS="${NUM_SMS:-0}"                    # v2 only; 0 = DeepEP's theoretical
TOKENS="${TOKENS:-4 8 16 32 128}"
SPLIT="${SPLIT:-1}"
PORT="${A2A_PORT:-18361}"
NAME="a2a-bench"

# --- model shapes -----------------------------------------------------------
# DSV4-Flash read from its own config.json: hidden_size 4096, num_experts_per_tok
# 6, n_routed_experts 256, and all 43 decoder layers are MoE (deepseek_v4.py
# builds DeepseekV2MoE unconditionally; there is no first_k_dense_replace).
# DSV3 is the shape every published DeepEP/UCCL-EP decode number uses.
case "$SHAPE" in
    dsv4) HIDDEN=4096 TOPK=6 EXPERTS=256 ;;
    dsv3) HIDDEN=7168 TOPK=8 EXPERTS=256 ;;
    *)    echo "unknown SHAPE=$SHAPE" >&2; exit 1 ;;
esac

# --- library -> image + backend --------------------------------------------
# `v1` is deep_ep.Buffer's low-latency path, which BOTH the DeepEP fork and
# UCCL-EP's deep_ep_wrapper implement -- that is what makes deepep_v1 vs uccl a
# same-code comparison. `v2` is ElasticBuffer and exists only in the fork.
case "$ELIB" in
    deepep_v1) BACKEND=v1; USE_IMAGE="${IMAGE}" ;;
    deepep_v2) BACKEND=v2; USE_IMAGE="${IMAGE}" ;;
    uccl)      BACKEND=v1; USE_IMAGE="${IMAGE%%:*}-uccl:${IMAGE##*:}" ;;
    *)         echo "unknown ELIB=$ELIB" >&2; exit 1 ;;
esac

# NCCL_GIN_TYPE is consumed ONLY by deepep_v2. v1 goes through NVSHMEM and
# UCCL-EP through its own EFA transport, so stamping gda/proxy on those arms
# would claim a mechanism that is not in the datapath -- same rule as
# 92/93_*_sweep.sh.
GIN_ARGS=()
if [[ "$BACKEND" == "v2" ]]; then
    if [[ "${GDAKI:-1}" == "1" ]]; then
        GIN_TYPE=5; GIN=gda
        GIN_ARGS+=(-e FI_EFA_USE_HW_CNTR=1 -e OFI_NCCL_GIN_STRONG_SIGNAL=1 -e NCCL_RMA_DISABLE=1)
    else
        GIN_TYPE=2; GIN=proxy
    fi
    GIN_ARGS+=(-e NCCL_GIN_TYPE="$GIN_TYPE")
else
    GIN=nogin
fi

# Spelled as if/fi, not `[[ ]] && x || y` inside a command substitution: a false
# test there returns 1 and `set -e` aborts the script mid-sourcing. Same trap as
# the IMAGE_TAG_SUFFIX note in env_common.sh.
if [[ "$CAP_POLICY" == "sglang" ]]; then CAPTAG="cap${CAPACITY}"; else CAPTAG="capnative"; fi

# Largest capacity any buffer in this process will be built with: under
# bench-native the capacity IS the token count, so it is max(TOKENS).
MAX_CAP="$CAPACITY"
if [[ "$CAP_POLICY" != "sglang" ]]; then
    MAX_CAP=0
    for t in $TOKENS; do (( t > MAX_CAP )) && MAX_CAP=$t; done
fi
# Same trap as 20_launch_node.sh:208 -- deep_ep/buffers/legacy.py asserts
# `nvshmem_qp_depth >= (capacity + 1) * 2` with no message, on ONE node too
# (v1's low-latency path is NVSHMEM even with no remote peer). Default is 1024,
# so bench-native (cap <= 128) never tripped it but CAP_POLICY=sglang at 1024
# demands 2050. Unread on the v2 and UCCL-EP arms, so setting it unconditionally
# keeps the three arms' env identical.
V1_QP_DEPTH=1024
while (( V1_QP_DEPTH < (MAX_CAP + 1) * 2 )); do V1_QP_DEPTH=$((V1_QP_DEPTH * 2)); done
SPLIT_ARGS=()
if [[ "$SPLIT" == "1" ]]; then SPLIT_ARGS+=(--split); fi
STEM="a2a-${ELIB}-${SHAPE}-n${NNODES}-${GIN}-${CAPTAG}-sm${NUM_SMS}"
OUT="results/${STEM}.json"
LOG="results/${STEM}.log"
mkdir -p results

require_idle_gpus
build_gdr_args
detect_ep_nic

MASTER="${MASTER_IP:?set MASTER_IP}"
if [[ "$NNODES" == "1" ]]; then MASTER=127.0.0.1; fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "=== $STEM  (image=$USE_IMAGE backend=$BACKEND rank=$NODE_RANK/$NNODES) ==="
set -x
docker run --rm --name "$NAME" \
    --gpus all --privileged --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --device=/dev/infiniband ${GDR_ARGS[@]+"${GDR_ARGS[@]}"} \
    --shm-size=32g \
    -v "$PWD/bench:/bench:ro" \
    -v "$PWD/results:/results" \
    "${GIN_ARGS[@]}" \
    ${EP_NIC_NAME:+-e EP_NIC_NAME=$EP_NIC_NAME} \
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    -e NVSHMEM_QP_DEPTH="$V1_QP_DEPTH" \
    -e MASTER_ADDR="$MASTER" -e MASTER_PORT="$PORT" \
    -e NNODES="$NNODES" -e NODE_RANK="$NODE_RANK" \
    -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1 \
    --entrypoint python3 \
    "$USE_IMAGE" /bench/bench_a2a_ll.py \
      --backend "$BACKEND" --eplib "$ELIB" --shape "$SHAPE" \
      --hidden "$HIDDEN" --num-topk "$TOPK" --num-experts "$EXPERTS" \
      --num-tokens $TOKENS \
      --cap-policy "$CAP_POLICY" --capacity "$CAPACITY" \
      --num-sms "$NUM_SMS" \
      ${SPLIT_ARGS[@]+"${SPLIT_ARGS[@]}"} \
      --out "/results/${STEM}.json" 2>&1 | tee "$LOG"
set +x

echo "=== wrote $OUT ==="
