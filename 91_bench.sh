#!/bin/bash
# Throughput / latency benchmark via sglang.bench_serving.
#
#   bash 91_bench.sh                                     # 1k/1k, 32 concurrent
#   ISL=4096 OSL=512 NUM_PROMPTS=128 CONCURRENCY=64 bash 91_bench.sh
#
# Raw output is tee'd to results/<TAG>.log with the JSON summary alongside, so a
# sweep leaves an auditable trail. TAG records the knobs that change the answer
# (num_sms, capacity, GIN route, node count) -- a bare throughput number with no
# topology attached is not comparable to anything.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

ENDPOINT="${ENDPOINT:-localhost:$PORT}"
HOST="${ENDPOINT%%:*}"
BPORT="${ENDPOINT##*:}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
CONCURRENCY="${CONCURRENCY:-32}"
NUM_SMS="${NUM_SMS:-20}"
PHASE="${PHASE:-decode}"
NAME="${NAME:-dsv4-bench}"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR_HOST/results}"
GIN=$([[ "$GDAKI" == "1" ]] && echo gda || echo proxy)
TAG="${TAG:-n${NNODES}-${PHASE}-sm${NUM_SMS}-${GIN}-isl${ISL}-osl${OSL}-c${CONCURRENCY}}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${TAG}.log"
JSON="$RESULTS_DIR/${TAG}.json"

echo "bench: ${ENDPOINT}  isl=${ISL} osl=${OSL} n=${NUM_PROMPTS} conc=${CONCURRENCY}"
echo "log  : ${LOG}"

{
  echo "### tag=${TAG}"
  echo "### endpoint=${ENDPOINT} nnodes=${NNODES} tp=${TP} phase=${PHASE}"
  echo "### num_sms=${NUM_SMS} gin=${GIN} image=${IMAGE}"
  echo "### isl=${ISL} osl=${OSL} num_prompts=${NUM_PROMPTS} concurrency=${CONCURRENCY}"
  echo "### started=$(date -u +%FT%TZ)"
} > "$LOG"

# --flush-cache: the random dataset is seeded, so a second run replays the same
# prompts and hits the radix cache from the first -- inflating output throughput
# and collapsing TTFT.
#
# --tokenizer must point at the local weights: the server reports its model_path
# as /models/..., which bench_serving would otherwise try to resolve as an HF
# repo id ("Repo id must be in the form 'repo_name' or 'namespace/repo_name'").
docker run --rm --name "$NAME" --net=host \
    -v "$HOST_MODEL_DIR/$MODEL_NAME:$MODEL_PATH:ro" \
    -v "$RESULTS_DIR:/results" \
    --entrypoint python3 "$IMAGE" \
    -m sglang.bench_serving \
    --backend sglang-oai \
    --host "$HOST" --port "$BPORT" \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --flush-cache \
    --output-file "/results/${TAG}.json" 2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
echo "### finished=$(date -u +%FT%TZ) rc=${rc}" >> "$LOG"
[[ -f "$JSON" ]] && echo "json : ${JSON}"
exit "$rc"
