#!/bin/bash
# Extract the SERVER-SIDE decode rate from a running server's log, as a function
# of the per-rank running batch.
#
#   bash 94_server_decode_rate.sh                    # after a 93_decode_sweep.sh run
#   NAME=dsv4-epv2 DP=0 bash 94_server_decode_rate.sh
#
# WHY THIS EXISTS: bench_serving's `output_throughput` is total output tokens over
# total wall clock, and at ISL 1024 / OSL 1024 that wall clock is NOT decode. On a
# measured single-node run DP0 logged 262 `Prefill batch` lines against 256
# `Decode batch` lines -- the scheduler interleaves continuously, so prefilling
# the 2.1M input tokens of a 2048-prompt row is a large share of the denominator.
# The visible symptom is that 8 GPUs and 16 GPUs both "saturate" at ~18-20k
# output tok/s, which is not a coincidence and not a decode ceiling.
#
# The server's own `gen throughput (token/s)` is per-DP-rank and is measured over
# decode intervals, so `rate / running_req` is a real per-rank step time and is
# directly comparable across node counts. Report BOTH: the client number is what
# a user of the endpoint sees, the server number is what decode actually costs.
#
# One rank is enough (DP attention routes round-robin and the ranks agree to a few
# percent), and the aggregate is rate x dp_size.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-dsv4-epv2}"
DP="${DP:-0}"
OUT="${OUT:-}"

docker logs "$NAME" 2>&1 \
  | grep "DP${DP} TP${DP} " \
  | grep -E "Decode batch|Prefill batch" \
  | python3 -c '
import re, sys, statistics
from collections import defaultdict

dec, pre = defaultdict(list), 0
for line in sys.stdin:
    if "Prefill batch" in line:
        pre += 1
        continue
    rr = re.search(r"#running-req: (\d+)", line)
    gt = re.search(r"gen throughput \(token/s\): ([\d.]+)", line)
    if rr and gt:
        dec[int(rr.group(1))].append(float(gt.group(1)))

if not dec:
    sys.exit("no Decode batch lines with a gen throughput -- wrong DP rank or no run yet")

# Batches within a couple of requests of each other are the same ladder rung
# (requests finish at slightly different steps), so bucket to the nearest power of
# two rather than reporting 126/127/128 as three rows.
buckets = defaultdict(list)
for rr, vals in dec.items():
    b = 1 << max(0, (rr - 1).bit_length())
    buckets[b].extend(vals)

print("# decode prints=%d prefill prints=%d" % (sum(len(v) for v in dec.values()), pre))
print("# a prefill count comparable to the decode count means the client-side")
print("# output_throughput for those rows is NOT a decode measurement.")
print("%9s %15s %9s %8s" % ("req/rank", "gen tok/s/rank", "step ms", "samples"))
for b in sorted(buckets):
    v = statistics.median(buckets[b])
    print("%9d %15.1f %9.2f %8d" % (b, v, b / v * 1000, len(buckets[b])))
' 2>&1 | { if [[ -n "$OUT" ]]; then tee "$OUT"; else cat; fi; }

[[ -n "$OUT" ]] && echo "wrote $OUT"
