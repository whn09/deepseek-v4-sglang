#!/bin/bash
# Sequential driver for 96_a2a_microbench.sh.
#
#   ELIBS="deepep_v1 deepep_v2" SHAPES="dsv4 dsv3" POLICIES="bench-native" \
#     NNODES=1 bash 97_a2a_sweep.sh
#
# For a 2-node sweep, launch with the SAME ELIBS/SHAPES/POLICIES order on both
# nodes (only NODE_RANK differs): each 96_ invocation reuses one fixed port and
# blocks on NCCL bootstrap, so identical iteration order is what pairs the ranks
# up. A mismatched order deadlocks on the first differing config rather than
# producing wrong numbers.
#
# Runs are never parallel within a node: two a2a benchmarks on the same GPUs
# would measure each other (see `feedback_benchmark_real_topology`).
set -uo pipefail
cd "$(dirname "$0")"

ELIBS="${ELIBS:-deepep_v1}"
SHAPES="${SHAPES:-dsv4 dsv3}"
POLICIES="${POLICIES:-bench-native}"
TOKENS="${TOKENS:-4 8 16 32 128}"
CAPACITY="${CAPACITY:-1024}"
NUM_SMS="${NUM_SMS:-0}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-1200}"

mkdir -p results
SUMMARY="results/a2a-sweep-n${NNODES}-r${NODE_RANK}.txt"
: > "$SUMMARY"

for pol in $POLICIES; do
  for elib in $ELIBS; do
    for shape in $SHAPES; do
      echo "################ $elib $shape $pol n$NNODES r$NODE_RANK ################"
      NNODES="$NNODES" NODE_RANK="$NODE_RANK" \
      ELIB="$elib" SHAPE="$shape" CAP_POLICY="$pol" CAPACITY="$CAPACITY" \
      NUM_SMS="$NUM_SMS" TOKENS="$TOKENS" SPLIT=1 \
        timeout "$PER_RUN_TIMEOUT" bash 96_a2a_microbench.sh
      rc=$?
      echo "$elib $shape $pol n$NNODES rc=$rc" | tee -a "$SUMMARY"
      # Let the container tear down and the bootstrap port drain before the next
      # config claims it; a TIME_WAIT socket here shows up as a bootstrap hang.
      docker rm -f a2a-bench >/dev/null 2>&1
      sleep 15
    done
  done
done
echo "=== sweep done; see $SUMMARY ==="
