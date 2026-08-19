#!/bin/bash
# Prefill-dominated concurrency sweep: what is the INPUT token throughput, and how
# far can concurrency be pushed before it stops buying anything?
#
#   bash 92_prefill_sweep.sh                       # default ladder
#   ISL=8192 CONCS="32 128 512" bash 92_prefill_sweep.sh
#
# The question this answers: does 2 nodes give MORE than 2x the single-node input
# throughput, and does it admit a higher concurrency? So the sweep must be run
# twice with only NNODES changed -- every other knob (per-rank chunk, per-rank
# concurrency slots, ISL/OSL, num_sms) is held per-rank constant by construction:
#   chunked-prefill-size = CAPACITY*TP    -> CAPACITY tokens/rank either way
#   max-running-requests = RUNNING_PER_RANK*TP -> same slots/rank either way
# The published rows used CAPACITY=8192 and RUNNING_PER_RANK=32. CAPACITY matters
# more than anything else measured here: at the cookbook's 1024 the single node
# reads 41.4k tok/s and at 8192 it reads 162.8k, so a comparison run at the
# default would be two crippled configs racing each other.
# so the only thing that changes is how many ranks there are, and whether the a2a
# crosses EFA (NNODES=2 -> hybrid) or stays on NVLink (NNODES=1 -> direct).
#
# OSL is deliberately tiny, not 0: this is a prefill measurement, so decode must
# contribute as little as possible, but bench_serving needs at least a few output
# tokens for TTFT/ITL to exist. Read "Input token throughput" from each row --
# with OSL=8 vs ISL=4096 the decode share of the wall clock is ~0.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

ISL="${ISL:-4096}"
OSL="${OSL:-8}"
CONCS="${CONCS:-16 32 64 128 256 512}"
# 3 waves per concurrency level: enough for the steady state to dominate the
# ramp-in, few enough that the whole ladder is minutes not hours. Capped so a
# high-concurrency row does not blow up into millions of input tokens.
REQ_MULT="${REQ_MULT:-3}"
MAX_PROMPTS="${MAX_PROMPTS:-512}"

# Every filename and log header in this sweep is stamped with $NNODES, but NNODES
# is just an env default (2) -- it does NOT know what server is actually listening.
# Getting this wrong mislabels a whole ladder as the other topology, which is the
# one error this comparison cannot survive. So ask the server what it really is.
ACTUAL_TP="$(curl -s "localhost:${PORT}/get_server_info" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tp_size","?"))' 2>/dev/null || echo "?")"
if [[ "$ACTUAL_TP" != "$TP" ]]; then
    echo "REFUSING: server reports tp_size=$ACTUAL_TP but NNODES=$NNODES implies TP=$TP." >&2
    echo "  Set NNODES to match the running server (NNODES=1 for a single-node server)." >&2
    exit 1
fi

# Must match what 20_launch_node.sh was given -- and per-rank chunk (= CAPACITY)
# is the strongest knob measured here, so it belongs in the filename. Without it
# the `rm -f "$JSON"` below lets a run at one chunk size delete another's result.
CAPACITY="${CAPACITY:-1024}"
GIN=$([[ "$GDAKI" == "1" ]] && echo gda || echo proxy)
# $A2A belongs in the filename for the same reason $CAPACITY does, and MEASURED the
# hard way: an A2A=none control run silently overwrote the deepep_v2 summary on the
# host (the per-row .json/.log names from 91_bench.sh DO carry $A2A, so only the
# summary was lost). 93_decode_sweep.sh already stamps it; this one did not.
A2A="${A2A:-deepep_v2}"
# ...and ${GIN} for the third time, found on p5en: a GDAKI=0 control ladder wrote the
# GDAKI=1 ladder's summary name and deleted it. On p5/b300 the GIN type was fixed by
# the hardware so this could not bite; p5en is the first box where both arms are real.
SUMMARY="${SUMMARY:-$SCRIPT_DIR_HOST/results/prefill-sweep-${A2A}-n${NNODES}-cap${CAPACITY}-${GIN}.txt}"
mkdir -p "$(dirname "$SUMMARY")"

{
  echo "### prefill sweep  a2a=$A2A nnodes=$NNODES tp=$TP isl=$ISL osl=$OSL"
  echo "### started=$(date -u +%FT%TZ)"
  printf '%-6s %-8s %-12s %-12s %-12s %-12s %-10s\n' \
      conc prompts in_tok/s out_tok/s ttft_ms_mean ttft_ms_p99 req/s
} | tee "$SUMMARY"

# deep_gemm JIT-compiles per GEMM shape on first use, and the fp4-expert path is
# not in the server's own warmup set. Measured cost of skipping this: the first
# row came out at 6.3k tok/s and the identical rerun at 20.7k -- a 3.3x artifact
# that would look exactly like a concurrency effect.
if [[ "${WARMUP:-1}" == "1" ]]; then
    echo "### warmup (JIT compile; discarded)" | tee -a "$SUMMARY"
    ISL="$ISL" OSL="$OSL" NUM_PROMPTS=16 CONCURRENCY=16 PHASE=prefill TAG=warmup \
        CAPACITY="$CAPACITY" bash ./91_bench.sh >/dev/null 2>&1 || true
fi

for C in $CONCS; do
    N=$((C * REQ_MULT))
    [[ "$N" -gt "$MAX_PROMPTS" ]] && N="$MAX_PROMPTS"
    [[ "$N" -lt "$C" ]] && N="$C"

    JSON="$SCRIPT_DIR_HOST/results/${A2A}-n${NNODES}-prefill-sm${NUM_SMS:-20}-cap${CAPACITY}-${GIN}-isl${ISL}-osl${OSL}-c${C}.json"
    # Or a failed run silently reports the previous sweep's numbers.
    rm -f "$JSON"

    ISL="$ISL" OSL="$OSL" NUM_PROMPTS="$N" CONCURRENCY="$C" PHASE=prefill \
        CAPACITY="$CAPACITY" bash ./91_bench.sh >/dev/null 2>&1

    python3 - "$JSON" "$C" "$N" <<'PY' | tee -a "$SUMMARY"
import json, sys
path, c, n = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    # bench_serving writes one object; older builds append one per line.
    raw = open(path).read()
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = json.loads([l for l in raw.splitlines() if l.strip()][-1])
except Exception as e:
    print(f"{c:<6} {n:<8} FAILED ({type(e).__name__}: {e})")
    raise SystemExit(0)
print("%-6s %-8s %-12.1f %-12.1f %-12.1f %-12.1f %-10.2f" % (
    c, n,
    d.get("input_throughput", 0.0), d.get("output_throughput", 0.0),
    d.get("mean_ttft_ms", 0.0), d.get("p99_ttft_ms", 0.0),
    d.get("request_throughput", 0.0)))
PY
done

echo "### finished=$(date -u +%FT%TZ)" | tee -a "$SUMMARY"
echo "summary: $SUMMARY"
