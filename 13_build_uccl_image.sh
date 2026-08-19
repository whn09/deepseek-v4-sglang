#!/bin/bash
# Build the UCCL-EP arm as a thin overlay on the DeepEP image. CPU + nvcc only,
# no GPU touched, so it is safe to run while a server holds the GPUs.
#
#   bash 13_build_uccl_image.sh
#   ARCH=sm103 bash 13_build_uccl_image.sh
#   UCCL_REF=<sha> bash 13_build_uccl_image.sh
#
# Unlike 10_build_image.sh this needs NO build context -- Dockerfile.uccl-ep only
# clones from the internet and edits site-packages -- so it builds out of an empty
# dir instead of $BUILD_CTX_HOST. That is deliberate: passing the 600 MB EFA
# context here would just be uploaded and ignored.
#
# The resulting image can run ONLY `A2A=deepep` (v1). UCCL-EP's deep_ep_wrapper
# implements v1's `Buffer`, not v2's `ElasticBuffer`, and this image deletes
# DeepEP outright -- so `A2A=deepep_v2` will fail at import. Run the DeepEP arm
# of the comparison at `A2A=deepep` too, out of the BASE image, or the two arms
# differ by algorithm generation as well as by transport.
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

BASE_IMAGE="${BASE_IMAGE:-$IMAGE}"
UCCL_IMAGE="${UCCL_IMAGE:-${BASE_IMAGE%%:*}-uccl:${BASE_IMAGE##*:}}"
UCCL_REPO="${UCCL_REPO:-https://github.com/uccl-project/uccl.git}"
UCCL_REF="${UCCL_REF:-main}"
LOG="${LOG:-/tmp/build_uccl_ep.log}"

ARCH="${ARCH:-sm90}"
case "$ARCH" in
  sm90|h100|h200|p5|p5en) TORCH_ARCH="9.0" ;;
  sm103|b300)             TORCH_ARCH="10.3" ;;
  sm100|b200)             TORCH_ARCH="10.0" ;;
  *) echo "ERROR: unknown ARCH=$ARCH" >&2; exit 1 ;;
esac

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
    echo "ERROR: base image $BASE_IMAGE not present on this host." >&2
    echo "       Pull it first (see env_p5en.sh ECR_* vars) -- this overlay exists" >&2
    echo "       so that both arms share one byte-identical sglang/NCCL/libfabric." >&2
    exit 1
}

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
cp Dockerfile.uccl-ep "$CTX/Dockerfile"

echo "building $UCCL_IMAGE"
echo "  base : $BASE_IMAGE"
echo "  arch : $ARCH (TORCH_CUDA_ARCH_LIST=$TORCH_ARCH)"
echo "  uccl : $UCCL_REPO @ $UCCL_REF"
echo "  log  : $LOG"

# --no-cache on the clone layer only would be nicer, but UCCL_REF=main means the
# whole point is to re-clone; the base layers are already cached by digest.
DOCKER_BUILDKIT=1 docker build --progress=plain \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "TORCH_CUDA_ARCH_LIST=$TORCH_ARCH" \
    --build-arg "UCCL_REPO=$UCCL_REPO" \
    --build-arg "UCCL_REF=$UCCL_REF" \
    -t "$UCCL_IMAGE" "$CTX" 2>&1 | tee "$LOG"

echo
echo "=== what the build decided ==="
docker run --rm --entrypoint cat "$UCCL_IMAGE" /etc/uccl-build-env
# Same file as the base image: it must be UNCHANGED, that is the whole control.
docker run --rm --entrypoint cat "$UCCL_IMAGE" /etc/deepep-build-env
echo
echo "run with:  IMAGE=$UCCL_IMAGE A2A=deepep bash 20_launch_node.sh <rank>"
