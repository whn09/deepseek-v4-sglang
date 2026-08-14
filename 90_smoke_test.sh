#!/bin/bash
# Post-launch checks: is the server up, did it really take the DeepEP v2 path, and
# did it really take the EFA GDA path? Run on the leader node.
#
#   bash 90_smoke_test.sh
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-dsv4-epv2}"
ENDPOINT="${ENDPOINT:-localhost:$PORT}"

echo "=== 1. health ==="
curl -s --max-time 30 "http://${ENDPOINT}/health_generate" && echo " OK" || echo " NOT READY"

echo
echo "=== 2. one generation ==="
curl -s --max-time 120 "http://${ENDPOINT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0]["text"] if "choices" in d else d)' \
  2>/dev/null || echo "generation FAILED"

echo
echo "=== 3. DeepEP v2 actually selected ==="
# The a2a backend is resolved at server-args time, so these lines appear once per
# launch. Absence means the run silently fell back to another dispatcher.
docker logs "$NAME" 2>&1 | grep -E "DeepEP v2 MoE is (enabled|using)|resolved --moe-runner-backend" | head -5
echo "  num_sms env:  $(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" 2>/dev/null | grep DEEPEP_V2 | tr '\n' ' ')"

echo
echo "=== 4. EFA GDA (route B) actually used ==="
echo "  NCCL_GIN_TYPE: $(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" 2>/dev/null | grep '^NCCL_GIN_TYPE=' )"
# A GIN init that falls back logs it; a GIN init that fails outright kills the
# rank, so an empty result here plus a healthy server means route B held.
docker logs "$NAME" 2>&1 | grep -iE "gin|gda|gdrcopy|ofi" | grep -iE "fail|error|fallback|disabl" | head -10 \
  || echo "  no GIN/OFI failures logged"

echo
echo "=== 5. transport errors ==="
for pat in "Failed to initialize GDRCopy" "NCCL WARN" "ncclInternalError" "Communicator does not support symmetric memory"; do
    n=$(docker logs "$NAME" 2>&1 | grep -c "$pat")
    echo "  ${pat}: ${n}"
done
