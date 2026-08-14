#!/bin/bash
# Resolve the exact server argv through ServerArgs WITHOUT starting a server.
#
#   bash 12_preflight_args.sh
#
# Why this exists: the deepep_v2 handler rejects illegal combinations (capacity vs
# per-rank chunk, runner whitelist, TBO/SBO, forced shared-expert fusion) at
# server-args time, but the real launch reaches that point only AFTER loading
# 149 GB of weights on 16 ranks. This does the same resolution in ~20 s.
#
# It runs CPU-ONLY (no --gpus, no --device=/dev/infiniband) precisely so it cannot
# touch a GPU that another workload is using -- which also means torch sees no
# accelerator, so `--device cuda` has to be passed explicitly or
# _handle_missing_default_values() dies in get_device(). The model config IS read
# from disk, which is the point: page_size, KV dtype and the fp4-expert layout are
# all model-declared and only appear after that read.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

CAPACITY="${CAPACITY:-1024}"
NUM_SMS="${NUM_SMS:-20}"

run_case() {
    local label="$1"; shift
    echo
    echo "################ $label"
    ssh_local_docker "$@"
}

ssh_local_docker() {
    docker run --rm --net=none \
        -v "$HOST_MODEL_DIR/$MODEL_NAME:$MODEL_PATH:ro" \
        -e SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$CAPACITY" \
        -e SGLANG_DEEPEP_V2_NUM_SMS="$NUM_SMS" \
        --entrypoint python3 "$IMAGE" -c '
import sys
from sglang.srt.server_args import prepare_server_args
sa = prepare_server_args(sys.argv[1:])
print("RESOLVED OK")
for k in ("moe_a2a_backend","moe_runner_backend","deepep_v2_mode",
          "deepep_v2_dispatcher_output_dtype","page_size","chunked_prefill_size",
          "ep_size","tp_size","dp_size","kv_cache_dtype","swa_full_tokens_ratio",
          "max_running_requests","disable_shared_experts_fusion"):
    print(f"  {k} = {getattr(sa, k, '"'"'<absent>'"'"')}")
g = sa.cuda_graph_config
print(f"  graph.decode  = {g.decode.backend} max_bs={g.decode.max_bs}")
print(f"  graph.prefill = {g.prefill.backend}")
' "$@" 2>&1 | grep -vE "_pytree|KernelPreference|ScaleCalculation|UserWarning|warnings.warn|cpp_extension" | tail -25
}

COMMON=(--model-path "$MODEL_PATH" --device cuda --trust-remote-code
        --host 0.0.0.0 --port "$PORT"
        --tp "$TP" --dp "$TP" --ep "$TP" --enable-dp-attention
        --kv-cache-dtype auto --swa-full-tokens-ratio 0.1
        --chunked-prefill-size "$((CAPACITY * TP))")
MULTI=(--nnodes 2 --node-rank 0 --dist-init-addr "${MASTER_IP}:${DIST_PORT}" --enable-dp-lm-head)

run_case "deepep_v2 / hybrid / decode (the run under test)" \
    "${COMMON[@]}" --moe-a2a-backend deepep_v2 --deepep-v2-mode hybrid \
    --moe-runner-backend deep_gemm --mem-fraction-static 0.70 \
    --cuda-graph-max-bs-decode 128 "${MULTI[@]}"

run_case "deepep_v2 / hybrid / prefill" \
    "${COMMON[@]}" --moe-a2a-backend deepep_v2 --deepep-v2-mode hybrid \
    --moe-runner-backend deep_gemm --mem-fraction-static 0.80 \
    --disable-cuda-graph --disable-piecewise-cuda-graph "${MULTI[@]}"

run_case "megamoe baseline (attribution, same weights)" \
    "${COMMON[@]}" --moe-a2a-backend megamoe --mem-fraction-static 0.70 \
    --cuda-graph-max-bs-decode 128 "${MULTI[@]}"

echo
echo "NOTE: 'chunked prefill size is adjusted from N to N/dp_size' is expected --"
echo "      the CLI value is the GLOBAL budget and the resolved field is PER-RANK."
echo "      That per-rank number must be <= capacity ($CAPACITY) and a multiple of"
echo "      page_size, or the launch is refused."
