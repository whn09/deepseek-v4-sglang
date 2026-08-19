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
#   A2A=deepep_v2|deepep|none|megamoe   the backend under test, or a baseline.
#   DEEPEP_MODE=normal|low_latency|auto v1's kernel family (A2A=deepep only);
#                         defaults from PHASE.
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

# A2A backend. deepep_v2 is the thing under test; none/megamoe are attribution
# baselines; deepep is the UCCL-EP comparison arm.
#
# CORRECTION (2026-08-19): this comment used to claim DeepEP *v1* cannot run in
# this image, because the v2 fork installs over the same `deep_ep` module name and
# 10_build_image.sh uninstalls sgl-deep-ep. That is WRONG and it was tested: the
# EFA fork's `deep_ep` exports BOTH `Buffer` (v1: dispatch/combine/
# low_latency_dispatch/low_latency_combine/internode_*/get_dispatch_layout/...) and
# `ElasticBuffer` (v2), and `import sglang.srt.layers.moe.token_dispatcher.deepep`
# succeeds. So `A2A=deepep` runs out of this image with no rebuild -- which is what
# makes the UCCL-EP comparison a clean control (see the deepep) arm below).
case "$A2A" in
    deepep_v2)
        A2A_ARGS=(--moe-a2a-backend deepep_v2 --deepep-v2-mode "$MODE"
                  --moe-runner-backend deep_gemm)
        A2A_ENV=(-e SGLANG_DEEPEP_V2_NUM_SMS="$NUM_SMS"
                 -e SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$CAPACITY")
        ;;
    deepep)
        # DeepEP **v1** (`Buffer`), and the ONLY backend UCCL-EP can drive: its
        # deep_ep_wrapper reimplements the v1 API, not v2's `ElasticBuffer`. So the
        # UCCL-EP comparison runs `A2A=deepep` on BOTH sides -- DeepEP v1 out of
        # $IMAGE, UCCL-EP out of the 13_build_uccl_image.sh overlay of the same
        # image -- and the EP library is then the only variable. Pointing UCCL-EP
        # at our deepep_v2 numbers instead would change the transport AND the
        # algorithm generation at once, and the v2 ElasticBuffer is a large part of
        # why those numbers look the way they do.
        #
        # MODE here is v1's own --deepep-mode, which is a different axis from
        # --deepep-v2-mode (that one is a topology: direct vs hybrid). v1's is
        # which KERNEL FAMILY runs: `normal` = the internode contiguous-layout
        # dispatch (prefill), `low_latency` = the masked-layout one (decode).
        # `auto` allocates both buffers and switches per batch, which doubles the
        # buffer and makes the two phases share one sizing -- so it is NOT the
        # default here; each phase gets its matching kernel, mirroring how the
        # deepep_v2 arms were run as separate servers.
        DEEPEP_MODE="${DEEPEP_MODE:-$([[ "$PHASE" == "prefill" ]] && echo normal || echo low_latency)}"
        # v1 hard-asserts num_max_dispatch_tokens_per_rank <= 1024
        # (deepep.py: internode_ll dispatch uses FINISHED_SUM_TAG=1024), so the
        # prefill arm's CAPACITY=4096 CANNOT be passed through -- it dies on an
        # assert, not a warning. Clamp instead of failing: for `normal` mode this
        # env var only sizes the low-latency buffer, which normal mode does not
        # use, while the thing that actually sets the prefill message geometry is
        # --chunked-prefill-size (still CAPACITY*TP above, so the per-rank chunk
        # stays 4096 and matches the deepep_v2 prefill arm token-for-token).
        V1_CAPACITY="$CAPACITY"
        if [[ "$V1_CAPACITY" -gt 1024 ]]; then
            V1_CAPACITY=1024
            echo "=== A2A=deepep: clamping v1 dispatch capacity ${CAPACITY} -> 1024 (v1 asserts <=1024) ==="
        fi
        # deep_gemm to match the deepep_v2 arms; the masked grouped-GEMM consumes
        # v1's low-latency output layout directly. On Hopper this makes
        # runner.is_auto() false, i.e. it needs the same
        # SGLANG_FP4_DEQUANT_ANY_RUNNER=1 image the deepep_v2 arms need -- already
        # asserted above.
        # SECOND v1-only sizing assert, and it fires at the FIRST DECODE STEP, not at
        # startup -- the server loads all 46 shards, finishes the DeepGEMM warmup,
        # reports healthy, and then dies on the first low-latency dispatch with a
        # bare `AssertionError` and no message:
        #   deep_ep/buffers/legacy.py:609
        #     assert self.nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2
        # At V1_CAPACITY=1024 that demands 2050 and NVSHMEM's default QP depth is
        # 1024. It applies even on ONE node, because v1's low-latency path goes
        # through NVSHMEM regardless of whether any peer is remote. Rounded up to a
        # power of two because NVSHMEM sizes its rings that way.
        # No-op on the UCCL-EP arm: that image has no deep_ep and its wrapper never
        # loads NVSHMEM, so the variable is simply unread -- which is why it is safe
        # to set unconditionally here and keeps the two arms' env identical.
        V1_QP_DEPTH=1024
        while (( V1_QP_DEPTH < (V1_CAPACITY + 1) * 2 )); do V1_QP_DEPTH=$((V1_QP_DEPTH * 2)); done
        A2A_ARGS=(--moe-a2a-backend deepep --deepep-mode "$DEEPEP_MODE"
                  --moe-runner-backend deep_gemm)
        A2A_ENV=(-e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$V1_CAPACITY"
                 -e NVSHMEM_QP_DEPTH="$V1_QP_DEPTH")
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
    *) echo "A2A must be deepep_v2, deepep, none or megamoe" >&2; exit 1 ;;
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
        # Was hardcoded at 128. Exposed while chasing a server-killing hang, and
        # the honest state of that investigation is: NOT this knob.
        #
        # What happened (2026-08-18): at GRAPH_MAX_BS=128, the c=2048 rung of a
        # decode ladder died when both nodes were at 125-126 req/rank. DeepEP v2's
        # ElasticBuffer dispatch, called from eager_runner._execute_decode, sat in
        # its CPU-side wait for the full num_cpu_timeout_secs=300 (elastic.py's
        # default; sglang never overrides it) with all-zero received counts, threw
        #   Dispatch CPU wait exception (elastic/buffer.hpp:1113):
        #   CPU side received count (scaleup: 7): 0 0 0 ...
        # killed all 8 schedulers on node 2 (exit 3) and SIGQUIT'd node 1. The KV
        # pool was at 4% full / 16% swa, so it was neither the capacity nor the
        # slots wall.
        #
        # The obvious hypothesis -- "the first batch to exceed GRAPH_MAX_BS falls
        # to the eager path and hangs there" -- was TESTED AND FALSIFIED: at
        # GRAPH_MAX_BS=320, per-rank batches of 320 AND 384 both ran the eager path
        # (`cuda graph: False`) for entire rungs with no hang. Raising this knob is
        # therefore NOT a known fix; the relaunch was confounded with it. Treat the
        # hang as an intermittent DeepEP-v2/proxy-GIN dispatch failure that has not
        # reproduced in 4 subsequent rungs up to 384 req/rank. 300 s of silence
        # before the throw is the signature; grep it with 95_stress_evidence.sh.
        #
        # Capturing more graphs is cheap here, though: max_total_num_tokens was
        # 3,070,720 at both 128 and 320, so decode-graph depth cost no KV pool.
        GRAPH_MAX_BS="${GRAPH_MAX_BS:-128}"
        PHASE_ARGS=(--mem-fraction-static "$MEM_FRACTION" --cuda-graph-max-bs-decode "$GRAPH_MAX_BS") ;;
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

# Private IPs are reassigned on every instance stop/start, and a stale MASTER_IP
# is the single most common multi-node failure here. Catch it in a second instead
# of minutes: rank 0 must OWN the address it is about to bind, and any other rank
# must be able to reach it. Both hosts get their env_common.sh overwritten by
# sync.sh, so "I fixed it on the host" does not survive.
if [[ "$NNODES" -gt 1 ]]; then
    if [[ "$NODE_RANK" == "0" ]]; then
        if ! hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "$MASTER_IP"; then
            echo "ERROR: MASTER_IP=$MASTER_IP is not an address of this host." >&2
            echo "       This host has: $(hostname -I)" >&2
            echo "       Re-run with MASTER_IP=<this host's private ip>, and update" >&2
            echo "       P5_1_IP / B300_1_IP in env_common.sh so it survives a sync." >&2
            exit 1
        fi
    elif ! (exec 3<>"/dev/tcp/$MASTER_IP/$DIST_PORT") 2>/dev/null; then
        # Not fatal: the leader may simply not be listening yet, which is normal
        # when both nodes are started together. Warn, because the other cause of
        # this is a stale IP and that will hang the NCCL bootstrap instead.
        echo "WARNING: cannot reach ${MASTER_IP}:${DIST_PORT} yet. Fine if rank 0 is" >&2
        echo "         still booting; a hang here means MASTER_IP is stale." >&2
    fi
fi

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
