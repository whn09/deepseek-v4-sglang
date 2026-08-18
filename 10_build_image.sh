#!/bin/bash
# Build the sglang + DeepEP-v2-on-EFA image on THIS host. CPU only (~6 min on
# 192 cores), so it is safe to run while something else holds the GPUs.
#
#   bash 10_build_image.sh
#   DEEPEP_KMAXPARTS=4 IMAGE=sglang-epv2-efa:stock bash 10_build_image.sh
#   ARCH=sm90 IMAGE=sglang-epv2-efa:pr29525-sm90 bash 10_build_image.sh
#
# ARCH=sm90 is the p5.48xlarge (H100) build. It changes three things at once and
# they are not independent, which is why it is one switch and not three env vars:
#   TORCH_CUDA_ARCH_LIST / NVCC_GENCODE  -> compute_90 (the default targets 10.3)
#   SGLANG_FP4_DEQUANT_ANY_RUNNER=1      -> Hopper has no FP4, so the MXFP4 routed
#                                           experts must be dequantised to FP8 at
#                                           load time, and the assert guarding
#                                           that path has to accept the deepep_v2
#                                           runner. See Dockerfile section 6b.
# Note the image tag is NOT derived from ARCH: a wrong-arch image fails at the
# first kernel launch, not at `docker run`, so name it yourself and keep the two
# tags apart.
#
# WHY IT BUILDS OUT OF ANOTHER DIRECTORY ($BUILD_CTX_HOST): the Dockerfile needs
# two artifacts that cannot live in this repo --
#   aws-efa-installer-1.50.0.tar.gz   627 MB, and not on the public release S3
#                                     (it is the dev channel build; 1.50.0 is the
#                                     first version whose libfabric+efa.ko can do
#                                     GDAKI at all)
#   deepep-src/                       the PRIVATE AWS EFA fork of DeepEP v2
# Both are already staged in $BUILD_CTX_HOST by the ep-benchmarks-efa kit, so this
# script copies the Dockerfile there and builds in place rather than duplicating
# 600 MB. Staging instructions are in the Dockerfile's own ARG comments.
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

DEEPEP_KMAXPARTS="${DEEPEP_KMAXPARTS:-1}"
LOG="${LOG:-$BUILD_CTX_HOST/build_sglang_epv2.log}"

ARCH="${ARCH:-sm103}"
case "$ARCH" in
  sm103|sm100|b300|b200) ARCH_ARGS=() ;;   # Dockerfile defaults
  sm90|h100|h200|p5)
      ARCH_ARGS=(
        --build-arg "TORCH_CUDA_ARCH_LIST=9.0"
        --build-arg "NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90"
        --build-arg "SGLANG_FP4_DEQUANT_ANY_RUNNER=1"
      ) ;;
  *) echo "ERROR: unknown ARCH=$ARCH (want sm90 or sm103)" >&2; exit 1 ;;
esac

[[ -d "$BUILD_CTX_HOST" ]] || {
    echo "ERROR: build context $BUILD_CTX_HOST not found." >&2
    echo "       It must contain aws-efa-installer-1.50.0.tar.gz and deepep-src/." >&2
    exit 1
}
for f in aws-efa-installer-1.50.0.tar.gz deepep-src/deep_ep/buffers/elastic.py; do
    [[ -e "$BUILD_CTX_HOST/$f" ]] || { echo "ERROR: missing $BUILD_CTX_HOST/$f" >&2; exit 1; }
done

cp Dockerfile "$BUILD_CTX_HOST/Dockerfile.sglang-epv2"

echo "building $IMAGE  (arch=$ARCH kMaxParts=$DEEPEP_KMAXPARTS)"
echo "log: $LOG"
cd "$BUILD_CTX_HOST"
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -f Dockerfile.sglang-epv2 \
    --build-arg "DEEPEP_KMAXPARTS=$DEEPEP_KMAXPARTS" \
    "${ARCH_ARGS[@]+"${ARCH_ARGS[@]}"}" \
    -t "$IMAGE" . 2>&1 | tee "$LOG"

echo
echo "=== what the build decided ==="
docker run --rm --entrypoint cat "$IMAGE" /etc/deepep-build-env
docker run --rm --entrypoint cat "$IMAGE" /opt/sglang-pr.sha
