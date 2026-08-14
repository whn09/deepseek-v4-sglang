#!/bin/bash
# Build the sglang + DeepEP-v2-on-EFA image on THIS host. CPU only (~6 min on
# 192 cores), so it is safe to run while something else holds the GPUs.
#
#   bash 10_build_image.sh
#   DEEPEP_KMAXPARTS=4 IMAGE=sglang-epv2-efa:stock bash 10_build_image.sh
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

[[ -d "$BUILD_CTX_HOST" ]] || {
    echo "ERROR: build context $BUILD_CTX_HOST not found." >&2
    echo "       It must contain aws-efa-installer-1.50.0.tar.gz and deepep-src/." >&2
    exit 1
}
for f in aws-efa-installer-1.50.0.tar.gz deepep-src/deep_ep/buffers/elastic.py; do
    [[ -e "$BUILD_CTX_HOST/$f" ]] || { echo "ERROR: missing $BUILD_CTX_HOST/$f" >&2; exit 1; }
done

cp Dockerfile "$BUILD_CTX_HOST/Dockerfile.sglang-epv2"

echo "building $IMAGE  (kMaxParts=$DEEPEP_KMAXPARTS)"
echo "log: $LOG"
cd "$BUILD_CTX_HOST"
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -f Dockerfile.sglang-epv2 \
    --build-arg "DEEPEP_KMAXPARTS=$DEEPEP_KMAXPARTS" \
    -t "$IMAGE" . 2>&1 | tee "$LOG"

echo
echo "=== what the build decided ==="
docker run --rm --entrypoint cat "$IMAGE" /etc/deepep-build-env
docker run --rm --entrypoint cat "$IMAGE" /opt/sglang-pr.sha
