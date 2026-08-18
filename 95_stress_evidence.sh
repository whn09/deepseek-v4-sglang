#!/bin/bash
# Which wall did a stress ladder actually hit -- the latency wall or the KV
# capacity wall? The client-side numbers cannot tell them apart: a huge p99 TTFT
# looks the same whether requests are queueing behind max_running_requests, or
# the KV pool is full and the scheduler is retracting.
#
#   bash 95_stress_evidence.sh                      # parse the live container
#   SINCE=15m bash 95_stress_evidence.sh            # only the last 15 minutes
#   LOGFILE=/path/to/server.log bash 95_stress_evidence.sh
#
# The server prints everything needed on its own `Decode batch` / `Prefill batch`
# lines, so this reads them instead of probing the running process (probing a live
# container has broken a benchmark here before).
#   full/swa token usage -- KV pool occupancy, 0..1. THIS is the capacity wall.
#                       V4 prints TWO pools because it is a hybrid SWA model
#                       (config sliding_window=128, --swa-full-tokens-ratio 0.1):
#                       the full-attention layers and the sliding-window layers
#                       have separate pools and either can fill first, so a single
#                       `token usage` regex would silently read only one of them.
#   #queue-req       -- requests admitted by the client but not by the scheduler,
#                       i.e. the max_running_requests wall.
#   #retracted-reqs  -- the scheduler gave up on an already-running request
#                       because the pool could not hold it (retraction_policy).
#                       Non-zero means the capacity wall was genuinely hit.
#   gen throughput   -- with #running-req, gives the real per-rank step time.
#
# Reported as MAX over the window, not mean: the question is whether the wall was
# ever touched, and a mean over a ramp-in hides exactly that. Only DP rank 0's
# lines are parsed by default (every rank prints; they are near-identical and
# mixing them would double-count) -- set DP=all to pool every rank.
#
# The log goes through a temp FILE rather than a pipe because the parser is fed by
# a heredoc: `cmd | python3 - <<PY` would have the heredoc win stdin and the log
# would silently parse as empty.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-dsv4-epv2}"
SINCE="${SINCE:-}"
LOGFILE="${LOGFILE:-}"
DP="${DP:-0}"

TMP="$(mktemp -t stressev.XXXXXX)"
trap 'rm -f "$TMP"' EXIT

if [[ -n "$LOGFILE" ]]; then
    if [[ "$LOGFILE" == *.gz ]]; then gzcat "$LOGFILE" > "$TMP" 2>/dev/null || zcat "$LOGFILE" > "$TMP"
    else cat "$LOGFILE" > "$TMP"; fi
else
    if [[ -n "$SINCE" ]]; then docker logs --since "$SINCE" "$NAME" > "$TMP" 2>&1
    else docker logs "$NAME" > "$TMP" 2>&1; fi
fi

python3 - "$TMP" "$DP" <<'PY'
import re, sys

path, dp = sys.argv[1], sys.argv[2]
rank_tag = None if dp == "all" else f"DP{dp} "

pat = {
    "run":   re.compile(r"#running-req:\s*(\d+)"),
    # Decode lines print `#full token:` / `#swa token:` (absolute counts) beside
    # the two usage fractions; prefill lines print `#new-token` / `#cached-token`
    # and no pool count at all. A plain `#token:` regex therefore matched nothing
    # and silently reported 0 for the whole ladder.
    "tok":   re.compile(r"#full token:\s*(\d+)"),
    "swatok": re.compile(r"#swa token:\s*(\d+)"),
    "full":  re.compile(r"full token usage:\s*([0-9.]+)"),
    "swa":   re.compile(r"swa token usage:\s*([0-9.]+)"),
    "queue": re.compile(r"#queue-req:\s*(\d+)"),
    "pend":  re.compile(r"#pending-token:\s*(\d+)"),
    # NOT on the batch line, and NOT hyphenated. Retraction is its own log line,
    #   "KV cache pool is full. Retract requests. #retracted_reqs: 1, ..."
    # so the hyphenated `#retracted-reqs:` this used to look for never matched and
    # the capacity wall silently read as 0 retractions even in the run that hit it.
    "retr":  re.compile(r"#retracted_reqs:\s*(\d+)"),
    "gen":   re.compile(r"gen throughput \(token/s\):\s*([0-9.]+)"),
    "graph": re.compile(r"cuda graph:\s*(\w+)"),
}
NUM = ("run", "tok", "swatok", "full", "swa", "queue", "pend", "retr", "gen")

dec = pre = 0
mx = {k: 0.0 for k in NUM}
graph_false = 0
peak = None          # worst decode line by batch size, for the step-time reading
slots = None         # per-rank concurrency slots, read from the server's own args
crash = []           # the a2a wall: a dispatch that never completed
retr_events = 0      # how many times the pool filled, not just the peak count

# The slots ceiling has to come from the log too, because "queueing" only means
# the SLOTS wall if the batch is actually near that ceiling. It was not: 126
# running against 512 slots/rank, with #pending-token at 105k -- the limiter was
# the chunked-prefill token budget, and calling that a slots wall would send
# someone to raise exactly the wrong knob.
args_run = re.compile(r"max_running_requests=(\d+)")
args_dp = re.compile(r"dp_size=(\d+)")
_mr = _dp = None

for line in open(path, errors="replace"):
    if _mr is None:
        m = args_run.search(line)
        if m:
            _mr = int(m.group(1))
    if _dp is None:
        m = args_dp.search(line)
        if m:
            _dp = int(m.group(1))
    # A DeepEP v2 ElasticBuffer dispatch that sits in its CPU-side wait for the
    # full num_cpu_timeout_secs (300 s, and sglang does not override it) with
    # all-zero received counts is not slowness -- it is a hang, and it kills every
    # scheduler on the node. MEASURED at 126 req/rank with the KV pool at 4%.
    if "Dispatch CPU wait" in line or "Combine CPU wait" in line:
        crash.append(line.strip()[:200])
    # Matched BEFORE the batch-line filter below, because retraction is logged on
    # its own "KV cache pool is full." line -- leaving it in the batch-only loop is
    # what made the capacity wall invisible.
    m = pat["retr"].search(line)
    if m:
        mx["retr"] = max(mx["retr"], float(m.group(1)))
        retr_events += 1
    is_dec = "Decode batch" in line
    is_pre = "Prefill batch" in line
    if not (is_dec or is_pre):
        continue
    if rank_tag and rank_tag not in line:
        continue
    dec += is_dec
    pre += is_pre
    v = {}
    for k, r in pat.items():
        m = r.search(line)
        if m:
            v[k] = m.group(1)
    for k in NUM:
        if k in v:
            try:
                mx[k] = max(mx[k], float(v[k]))
            except ValueError:
                pass
    if is_dec and v.get("graph", "").lower().startswith("f"):
        graph_false += 1
    if is_dec and "run" in v and "gen" in v:
        try:
            run, gen = float(v["run"]), float(v["gen"])
        except ValueError:
            continue
        if gen > 0 and (peak is None or run > peak[0]):
            peak = (run, gen, run / gen * 1000.0, v.get("full"), v.get("swa"))

print("lines parsed        decode=%d  prefill=%d   (rank filter: %s)"
      % (dec, pre, rank_tag.strip() if rank_tag else "all ranks pooled"))
if not dec and not pre:
    print("NO BATCH LINES -- wrong container, or the window has no traffic.")
    raise SystemExit(0)

worst_pool = max(mx["full"], mx["swa"])
print("max #running-req    %d      <- per-rank batch actually reached" % mx["run"])
print("max #full token     %s   <- absolute tokens in the full-attn pool"
      % format(int(mx["tok"]), ","))
print("max #swa token      %s   <- and in the sliding-window pool"
      % format(int(mx["swatok"]), ","))
print("max FULL tok usage  %.3f   <- CAPACITY wall if this approaches 1.0" % mx["full"])
print("max SWA  tok usage  %.3f   <- sliding-window pool, fills separately" % mx["swa"])
print("max #queue-req      %s   <- SLOTS wall (max_running_requests)" % format(int(mx["queue"]), ","))
print("max #pending-token  %s" % format(int(mx["pend"]), ","))
print("retraction events   %d (peak #retracted_reqs %d)   <- non-zero = pool overran"
      % (retr_events, int(mx["retr"])))
tail = "   <- eager decode; batch exceeded --cuda-graph-max-bs-decode" if graph_false else ""
print("decode lines w/ cuda graph FALSE: %d of %d%s" % (graph_false, dec, tail))
if peak:
    run, gen, step, full, swa = peak
    print("at the largest decode batch: %d req/rank, gen %.1f tok/s/rank => step %.2f ms, "
          "full usage %s, swa usage %s"
          % (run, gen, step, full if full is not None else "n/a",
             swa if swa is not None else "n/a"))
if _mr and _dp:
    slots = _mr // _dp
    print("slots/rank           %d   (max_running_requests=%d / dp_size=%d)" % (slots, _mr, _dp))
print()

# Ordered by how much each wall costs: a hang loses the server, a retraction loses
# work, queueing only loses latency.
if crash:
    print("VERDICT: a2a wall -- DeepEP v2 dispatch/combine never completed, which is")
    print("         neither the capacity nor the slots wall. Worst pool was only %.3f." % worst_pool)
    print("         Largest batch before it: %d req/rank." % mx["run"])
    print("         %s" % crash[0])
    print("         Intermittent: did NOT reproduce up to 384 req/rank, and raising")
    print("         --cuda-graph-max-bs-decode was tested and is NOT a known fix.")
elif mx["retr"] > 0:
    print("VERDICT: capacity wall -- the scheduler retracted running requests.")
elif worst_pool >= 0.90:
    print("VERDICT: at the capacity wall (worst pool %.3f) without retracting yet." % worst_pool)
elif mx["queue"] > 0 and slots and mx["run"] < 0.9 * slots:
    print("VERDICT: prefill-admission wall -- %s queued, but the batch peaked at %d of"
          % (format(int(mx["queue"]), ","), mx["run"]))
    print("         %d slots/rank and the pool at only %.3f, so neither ceiling was the"
          % (slots, worst_pool))
    print("         limiter. #pending-token peaked at %s: admission is bounded by"
          % format(int(mx["pend"]), ","))
    print("         --chunked-prefill-size. Raising RUNNING_PER_RANK will do nothing.")
elif mx["queue"] > 0:
    print("VERDICT: slots wall -- %s queued at %d req/rank (slots/rank=%s) with the fuller"
          % (format(int(mx["queue"]), ","), mx["run"], slots if slots else "?"))
    print("         KV pool at only %.3f. Raise RUNNING_PER_RANK, not the memory fraction."
          % worst_pool)
else:
    print("VERDICT: neither wall reached (worst pool %.3f, no queueing)." % worst_pool)
PY
