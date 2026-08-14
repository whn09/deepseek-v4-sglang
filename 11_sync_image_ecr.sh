#!/bin/bash
# Distribute the image built by 10_build_image.sh to the other node(s) via ECR.
#
#   bash 11_sync_image_ecr.sh push          # on the BUILDER node
#   bash 11_sync_image_ecr.sh pull          # on every other node
#   bash 11_sync_image_ecr.sh verify        # compare digests across both hosts
#
# WHY NOT JUST BUILD ON EACH NODE: the Dockerfile pins the amazon NCCL fork to a
# BRANCH (staging) and aws-ofi-nccl to master, and apt/pip resolve latest at build
# time. Two builds hours apart therefore produce two different NCCL/libfabric
# stacks. Running mismatched GIN builds in one job is a debugging trap -- the
# symptom is a hang or a plausible-but-wrong number, not an error. One digest on
# every rank, always.
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

ACTION="${1:-push}"
HOSTS_STR="${HOSTS:-P6-B300-1 P6-B300-2}"

case "$ACTION" in
  push)
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
        || { echo "ERROR: $IMAGE not present here. Build it first." >&2; exit 1; }
    aws ecr describe-repositories --region "$ECR_REGION" --repository-names "$ECR_REPO" >/dev/null 2>&1 \
        || aws ecr create-repository --region "$ECR_REGION" --repository-name "$ECR_REPO" >/dev/null
    ecr_login
    docker tag "$IMAGE" "$ECR_IMAGE"
    docker push "$ECR_IMAGE"
    echo "pushed $ECR_IMAGE"
    ;;
  pull)
    ecr_login
    docker pull "$ECR_IMAGE"
    # Re-tag to the local name the launchers use, so 20_launch_node.sh needs no
    # per-node branching.
    docker tag "$ECR_IMAGE" "$IMAGE"
    echo "pulled and tagged as $IMAGE"
    docker run --rm --entrypoint cat "$IMAGE" /etc/deepep-build-env
    ;;
  verify)
    # The local IMAGE tag is what the launchers run, so compare THAT, not the ECR
    # tag: a stale local build can shadow a freshly pulled one.
    for h in $HOSTS_STR; do
        printf '%-12s ' "$h"
        ssh "$h" "docker images --no-trunc --format '{{.ID}}' '$IMAGE' | head -1" 2>/dev/null || echo "(unreachable)"
    done
    n=$(for h in $HOSTS_STR; do ssh "$h" "docker images --no-trunc --format '{{.ID}}' '$IMAGE' | head -1" 2>/dev/null; done | sort -u | wc -l | tr -d ' ')
    [[ "$n" == "1" ]] && echo "OK: one image id on all hosts" \
                      || { echo "MISMATCH: $n distinct image ids -- pull again before running." >&2; exit 1; }
    ;;
  *) echo "usage: $0 push|pull|verify" >&2; exit 1 ;;
esac
