#!/bin/bash
# Decode-dominated concurrency sweep: OUTPUT token throughput and per-token
# latency, as a function of how many requests are decoding at once.
#
#   bash 93_decode_sweep.sh                          # default ladder
#   CONCS="256 1024 2048" bash 93_decode_sweep.sh
#
# Same protocol as 92_prefill_sweep.sh -- run it twice with only NNODES changed --
# but the quantity to read is `out_tok/s`, and the honest companion is TPOT, not
# TTFT. Throughput alone cannot answer "is the second node worth it" for decode:
# decode is a per-step all-reduce/a2a-bound loop, so adding ranks can raise
# aggregate tok/s while making every individual token slower. Both columns or
# neither.
#
# THE DECODE-SPECIFIC TRAP: what a decode step costs depends on the PER-RANK
# batch, and DP attention gives each rank concurrency/dp_size requests. So at a
# fixed total concurrency the two shapes are NOT running the same kernel:
#   c=1024 -> 128 req/rank on 1 node (dp=8), but only 64 req/rank on 2 nodes.
# That is a real property of the deployment, not an artifact, so the ladder is
# reported against total concurrency. But it means the 2-node shape needs the
# ladder extended one rung further (c=2048 -> 128 req/rank) before its per-rank
# batch matches what the single node was doing at its own peak. Compare peaks,
# and read the per-rank column when a row looks anomalous.
#
# ⚠️ MEASURED, AND IT INVALIDATES `out_tok/s` AS A DECODE METRIC AT HIGH
# CONCURRENCY. The intuition for ISL 1024 / OSL 1024 was "4 prefill chunks per
# request against 1024 sequential decode steps, so the wall clock is decode".
# It is not: prefill runs ~8x faster per token than decode, but the scheduler
# interleaves continuously and a 2048-prompt row still has 2.1M input tokens to
# chew. On the single-node run DP0 logged 262 `Prefill batch` lines against 256
# `Decode batch` lines, and the client-side out_tok/s flattened at ~18k on BOTH
# 8 and 16 GPUs -- the same number twice is the tell, because a real decode
# ceiling would move with the GPU count. Server-side, the same run was doing
# 61.7k tok/s at 128 req/rank.
# So read `out_tok/s` as "what a client of this endpoint sees on a mixed
# workload", and get the decode cost itself from 94_server_decode_rate.sh, which
# reads the server's own per-rank `gen throughput` and divides by the running
# batch. Both are real; only the second one is comparable across node counts.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONCS="${CONCS:-64 128 256 512 1024}"
# 2 waves, not 3: a decode row is ~1024 sequential steps and costs 20-100x what a
# prefill row does, so the ladder has to stay inside a session.
REQ_MULT="${REQ_MULT:-2}"
MAX_PROMPTS="${MAX_PROMPTS:-2048}"

# NNODES is only an env default -- it does NOT know what server is listening, and
# mislabelling a whole ladder as the other topology is the one error this
# comparison cannot survive. Ask the server what it actually is.
ACTUAL_TP="$(curl -s "localhost:${PORT}/get_server_info" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tp_size","?"))' 2>/dev/null || echo "?")"
if [[ "$ACTUAL_TP" != "$TP" ]]; then
    echo "REFUSING: server reports tp_size=$ACTUAL_TP but NNODES=$NNODES implies TP=$TP." >&2
    echo "  Set NNODES to match the running server (NNODES=1 for a single-node server)." >&2
    exit 1
fi

A2A="${A2A:-deepep_v2}"
GIN=$([[ "$GDAKI" == "1" ]] && echo gda || echo proxy)
# Must match what 20_launch_node.sh was given. It is in the filename because the
# PR states "smaller capacity is better for decode" and because a fixed-capacity
# buffer is a candidate explanation for the batch-independent 2-node floor -- so
# two capacities are two different experiments, and `rm -f "$JSON"` below would
# otherwise have one delete the other.
CAPACITY="${CAPACITY:-1024}"
SUMMARY="${SUMMARY:-$SCRIPT_DIR_HOST/results/decode-sweep-${A2A}-n${NNODES}-sm${NUM_SMS:-20}-cap${CAPACITY}.txt}"
mkdir -p "$(dirname "$SUMMARY")"

{
  echo "### decode sweep  a2a=$A2A nnodes=$NNODES tp=$TP sm=${NUM_SMS:-20} isl=$ISL osl=$OSL"
  echo "### started=$(date -u +%FT%TZ)"
  printf '%-6s %-9s %-8s %-12s %-11s %-11s %-11s %-9s\n' \
      conc req/rank prompts out_tok/s tpot_ms_mean tpot_ms_p99 ttft_ms_p99 req/s
} | tee "$SUMMARY"

# deep_gemm JIT-compiles per GEMM shape on first use and the fp4-expert path is
# not in the server's warmup set (measured: 6.3k vs 20.7k tok/s on an otherwise
# identical rerun). The decode shapes are a different set from the prefill ones,
# so this warmup is needed again even on a server that has served prefill.
if [[ "${WARMUP:-1}" == "1" ]]; then
    echo "### warmup (JIT compile; discarded)" | tee -a "$SUMMARY"
    ISL="$ISL" OSL=64 NUM_PROMPTS=16 CONCURRENCY=16 PHASE=decode TAG=warmup-decode \
        CAPACITY="$CAPACITY" bash ./91_bench.sh >/dev/null 2>&1 || true
fi

for C in $CONCS; do
    N=$((C * REQ_MULT))
    [[ "$N" -gt "$MAX_PROMPTS" ]] && N="$MAX_PROMPTS"
    [[ "$N" -lt "$C" ]] && N="$C"
    PER_RANK=$((C / TP))

    JSON="$SCRIPT_DIR_HOST/results/${A2A}-n${NNODES}-decode-sm${NUM_SMS:-20}-cap${CAPACITY}-${GIN}-isl${ISL}-osl${OSL}-c${C}.json"
    # Or a failed run silently reports the previous sweep's numbers.
    rm -f "$JSON"

    ISL="$ISL" OSL="$OSL" NUM_PROMPTS="$N" CONCURRENCY="$C" PHASE=decode \
        CAPACITY="$CAPACITY" bash ./91_bench.sh >/dev/null 2>&1

    python3 - "$JSON" "$C" "$PER_RANK" "$N" <<'PY' | tee -a "$SUMMARY"
import json, sys
path, c, per_rank, n = sys.argv[1:5]
try:
    raw = open(path).read()
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = json.loads([l for l in raw.splitlines() if l.strip()][-1])
except Exception as e:
    print(f"{c:<6} {per_rank:<9} {n:<8} FAILED ({type(e).__name__}: {e})")
    raise SystemExit(0)
print("%-6s %-9s %-8s %-12.1f %-11.2f %-11.2f %-11.1f %-9.2f" % (
    c, per_rank, n,
    d.get("output_throughput", 0.0),
    d.get("mean_tpot_ms", 0.0), d.get("p99_tpot_ms", 0.0),
    d.get("p99_ttft_ms", 0.0), d.get("request_throughput", 0.0)))
PY
done

echo "### finished=$(date -u +%FT%TZ)" | tee -a "$SUMMARY"
echo "summary: $SUMMARY"
