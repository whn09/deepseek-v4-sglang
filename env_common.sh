#!/bin/bash
# Shared config for the DeepSeek-V4-Flash + SGLang DeepEP-v2-on-EFA scripts.
# Sourced by the host launchers; keep it side-effect free.

# ---- paths (host) ----
NVME="${NVME:-/opt/dlami/nvme}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-$NVME/models}"
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$NVME/cache}"
SCRIPT_DIR_HOST="${SCRIPT_DIR_HOST:-/home/ubuntu/deepseek-v4-sglang}"

# The image build needs two artifacts that are NOT in this repo (627 MB and a
# private fork), so it runs out of a separate context directory on the host --
# see 10_build_image.sh and the README.
BUILD_CTX_HOST="${BUILD_CTX_HOST:-/home/ubuntu/work/deepep-v2-efa-gdaki}"

# ---- model ----
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash}"
HF_REPO="${HF_REPO:-deepseek-ai/DeepSeek-V4-Flash}"
MODEL_PATH="${MODEL_PATH:-/models/$MODEL_NAME}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$HF_REPO}"

# The GPU generation, asked once here because THREE separate defaults key off it
# (image tag, GDAKI, MEM_FRACTION) and each one had drifted independently.
# "9.0" on H100/H200, "10.3" on b300; empty where there is no nvidia-smi (e.g.
# reading logs on a laptop), in which case every default below stays Blackwell.
# The trailing `|| true` is REQUIRED, not defensive. This file is sourced by
# scripts running under `set -euo pipefail` (sync.sh among them), and on a machine
# with no nvidia-smi the pipeline fails, pipefail propagates it, and set -e aborts
# the caller *during sourcing* -- so `bash sync.sh | tail` printed nothing, exited
# 0 (tail's status), and silently stopped syncing. Failure here must degrade to an
# empty SM_CAP, never abort the caller.
SM_CAP="${SM_CAP:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)}"
SM_MAJOR="${SM_CAP%%.*}"
IS_PRE_BLACKWELL=$([[ -n "$SM_MAJOR" && "$SM_MAJOR" -lt 10 ]] && echo 1 || echo 0)

# Locally built: lmsysorg/sglang nightly + EFA 1.50.0 + GDRCopy + amazon NCCL
# (GIN) + aws-ofi-nccl + the DeepEP v2 EFA fork + sglang PR #29525.
#
# ARCH-SUFFIXED, because 10_build_image.sh tags `ARCH=sm90` builds
# `:pr29525-sm90` and the sm90 image is NOT interchangeable (it carries
# SGLANG_FP4_DEQUANT_ANY_RUNNER=1 and a sm_90 deep_gemm). Without the suffix
# every script on p5 dies with `pull access denied for sglang-epv2-efa` --
# which reads like a credentials problem, not a wrong-tag problem, and inside a
# sweep it costs a whole ladder: 91_bench.sh still creates its .log, so all five
# rungs "complete" in one second having written rc=125 stubs over their own
# output names.
#
# Spelled as an if/fi rather than `$([[ ... ]] && echo -sm90)`: an && list whose
# test is false returns 1, and inside a command substitution under `set -e` that
# aborts the sourcing script -- the same silent-sync failure as above, but firing
# on b300 (where the test is false) instead of on the laptop.
IMAGE_TAG_SUFFIX=""
if [[ "$IS_PRE_BLACKWELL" == "1" ]]; then IMAGE_TAG_SUFFIX="-sm90"; fi
IMAGE="${IMAGE:-sglang-epv2-efa:pr29525${IMAGE_TAG_SUFFIX}}"

# ---- image distribution ----
# Build ONCE, then ECR-sync to the other nodes -- do not build per node. Two
# independent builds are not the same image: the amazon NCCL fork is pinned to a
# BRANCH (staging), aws-ofi-nccl to master, and apt/pip resolve latest, so a
# rebuild hours later can ship a different NCCL into one rank of the same job.
ECR_REGION="${ECR_REGION:-us-west-2}"          # the p6-b300 instances live here
ECR_ACCOUNT="${ECR_ACCOUNT:-579019700964}"
ECR_REPO="${ECR_REPO:-deepseek-v4-sglang}"
ECR_TAG="${ECR_TAG:-${IMAGE##*:}}"
ECR_REGISTRY="${ECR_REGISTRY:-${ECR_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com}"
ECR_IMAGE="${ECR_IMAGE:-${ECR_REGISTRY}/${ECR_REPO}:${ECR_TAG}}"

ecr_login() {
    aws ecr get-login-password --region "$ECR_REGION" \
      | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null
}

# ---- JIT / autotune caches to persist on the host ----
# "container path=host subdir". DeepEP v2 JITs its kernels on first use and the
# cubin cache key includes the arch and every EP_* knob, so this is safe across
# rebuilds; wipe $HOST_CACHE_DIR if one ever misbehaves.
CACHE_MOUNTS=(
    "/root/.deep_ep=deepep_jit"              # DeepEP v2 cubin cache
    "/root/.cache/deep_gemm=deep_gemm"
    "/root/.cache/torch=torch"
    "/root/.cache/flashinfer=flashinfer"
    "/root/.cache/tvm-ffi=tvm-ffi"
    "/root/.cache/sglang=sglang"
    "/root/.triton=triton"
    "/root/.nv/ComputeCache=nv_compute"
)

build_cache_args() {
    local entry cpath hsub
    CACHE_ARGS=()
    for entry in "${CACHE_MOUNTS[@]}" "$@"; do
        cpath="${entry%%=*}"; hsub="${entry#*=}"
        mkdir -p "$HOST_CACHE_DIR/$hsub"
        CACHE_ARGS+=(-v "$HOST_CACHE_DIR/$hsub:$cpath")
    done
}

# ---- cluster ----
# Re-assigned on every instance restart -- re-check with `ssh P6-B300-N hostname -I`
# before a multi-node run, or NCCL bootstrap silently times out.
B300_1_IP="${B300_1_IP:-172.31.57.229}"  # P6-B300-1 (as of 2026-08-14)
B300_2_IP="${B300_2_IP:-172.31.61.182}"  # P6-B300-2 (as of 2026-08-14)
P5_1_IP="${P5_1_IP:-172.31.44.207}"      # P5-1 (as of 2026-08-18)
P5_2_IP="${P5_2_IP:-172.31.34.46}"       # P5-2 (as of 2026-08-18)

# Keyed off the arch for the same reason IMAGE and GDAKI are: this used to default
# unconditionally to the b300 leader, so every p5 launch needed MASTER_IP exported
# by hand and forgetting it did not fail cleanly -- rank 0 tried to bind an address
# it does not own and died with
#   zmq.error.ZMQError: Cannot assign requested address (addr='tcp://<b300 ip>:29601')
# which names ZMQ, not the config. Worse, `bash sync.sh` overwrites this file on
# the hosts, so a correct IP set on the host is silently reverted to the local
# copy's. 20_launch_node.sh now checks rank 0 actually owns this address.
if [[ "$IS_PRE_BLACKWELL" == "1" ]]; then
    MASTER_IP="${MASTER_IP:-$P5_1_IP}"
else
    MASTER_IP="${MASTER_IP:-$B300_1_IP}"
fi

PORT="${PORT:-30000}"
DIST_PORT="${DIST_PORT:-29600}"

# ---- topology ----
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
# The deepep_v2 handler forces ep_size == tp_size, and the PR's config runs
# --tp == --dp == --ep with DP attention.
TP="${TP:-$((NNODES * GPUS_PER_NODE))}"

# ---- EFA / GIN ----
# 1 -> NCCL_GIN_TYPE=5 (EFA GDA / GDAKI, zero-CPU device puts)
# 0 -> NCCL_GIN_TYPE=2 (proxy GIN)
#
# DEFAULTS OFF ON PRE-BLACKWELL, because EFA GDA needs the GPU to write a WQE and
# ring an MMIO doorbell itself, which p5/p5en (sm_90) cannot do -- there GIN is
# always proxy. The default mattered beyond the launch: 92/93_*_sweep.sh put
# `gda` vs `proxy` in every output filename from THIS variable rather than from
# the server, so a stale GDAKI=1 silently labelled a whole proxy-GIN p5 campaign
# as `gda`. Kept overridable so an sm_90 host can still be asked to try.
GDAKI="${GDAKI:-$([[ "$IS_PRE_BLACKWELL" == "1" ]] && echo 0 || echo 1)}"

# Fills GDR_ARGS with the /dev/gdrdrv device flag when the host has the module
# loaded. Without it aws-ofi-nccl logs "NET/OFI Failed to initialize GDRCopy:
# Failed to open gdr handle" and GIN falls back to a slower path. Conditional
# because a missing --device makes `docker run` fail outright, and the module is
# not loaded on every host (DKMS has to be rebuilt after a kernel upgrade, and
# there is no udev rule).
build_gdr_args() {
    GDR_ARGS=()
    if [[ -c /dev/gdrdrv ]]; then
        GDR_ARGS+=(--device=/dev/gdrdrv)
    else
        echo "WARN: /dev/gdrdrv missing -- GIN will run without GDRCopy." >&2
        echo "      Check: lsmod | grep gdrdrv ; sudo mknod /dev/gdrdrv c \$(awk '/gdrdrv/{print \$1}' /proc/devices) 0" >&2
    fi
}

# EP_NIC_NAME: pick the FASTEST rdmap* on THIS host rather than trusting the name
# baked into the image. p6-b300 also exposes two 100 Gb/s ibp19{8,9}s0f0 ports;
# naming one of those makes DeepEP's get_theoretical_num_sms() derive the wrong
# line rate and mis-size the kernel, and they also fail ce_probe with errno=95.
# The device name is PCI-derived, so it differs per instance type.
detect_ep_nic() {
    [[ -n "${EP_NIC_NAME:-}" ]] && return 0
    local n p r best="" best_rate=0
    for n in /sys/class/infiniband/rdmap*; do
        [[ -d "$n" ]] || continue
        for p in "$n"/ports/*/rate; do
            [[ -r "$p" ]] || continue
            r=$(awk '{print $1}' "$p")
            if [[ "${r:-0}" -gt "$best_rate" ]]; then best_rate=$r; best=$(basename "$n"); fi
        done
    done
    if [[ -n "$best" ]]; then
        EP_NIC_NAME="$best"
        echo "=== EP_NIC_NAME=$EP_NIC_NAME (auto, ${best_rate} Gb/s) ==="
    else
        echo "WARN: no rdmap* device found; falling back to the image's EP_NIC_NAME." >&2
    fi
}

# Refuse to start on a node that is already busy. Two distinct hazards: an
# unrelated workload gets OOMed, and our own number comes out of a contended node
# (a concurrent transfer_engine_bench saturates the same EFA rails the a2a needs).
require_idle_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [[ "${used:-0}" -le 1024 ]] && return 0
    echo "REFUSING TO START: ${used} MiB already resident on the busiest GPU." >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2 || true
    docker ps --format '  {{.Names}}\t{{.Status}}' >&2 || true
    echo "Stop the other workload first, or set ALLOW_BUSY_GPU=1 to override." >&2
    [[ "${ALLOW_BUSY_GPU:-0}" == "1" ]] || exit 1
}

# Assert the image is the one this kit expects, before a multi-minute weight load.
# Both checks catch failures that do NOT show up at startup:
#   - OFI_GDAKI=0 means aws-ofi-nccl was configured proxy-only, and
#     NCCL_GIN_TYPE=5 then fails inside GIN init with a message that looks like a
#     missing host prereq.
#   - a base image without the PR has no 'deepep_v2' in MoeA2ABackend, and
#     --moe-a2a-backend deepep_v2 dies on an argparse choice error.
require_epv2_image() {
    local img="${1:-$IMAGE}"
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "ERROR: image '$img' not found. Build it: bash 10_build_image.sh" >&2
        exit 1
    fi
    local gdaki
    gdaki=$(docker run --rm --entrypoint sh "$img" -c '. /etc/deepep-build-env; echo ${OFI_GDAKI:-0}' 2>/dev/null || echo 0)
    if [[ "$GDAKI" == "1" && "$gdaki" != "1" ]]; then
        echo "ERROR: '$img' has a proxy-only aws-ofi-nccl (OFI_GDAKI=$gdaki)." >&2
        echo "       NCCL_GIN_TYPE=5 will fail GIN init. Rebuild, or run GDAKI=0." >&2
        exit 1
    fi
    if ! docker run --rm --entrypoint test "$img" \
            -f /sgl-workspace/sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py; then
        echo "ERROR: '$img' does not contain sglang PR #29525 (no deepep_v2.py)." >&2
        exit 1
    fi
}
