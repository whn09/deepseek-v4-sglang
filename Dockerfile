# syntax=docker/dockerfile:1.10
# =============================================================================
# SGLang PR #29525 (`deepep_v2` / ElasticBuffer MoE A2A backend) on AWS EFA
# =============================================================================
#
# Goal: run https://github.com/sgl-project/sglang/pull/29525 -- which adds
# `--moe-a2a-backend deepep_v2` on top of DeepEP v2's ElasticBuffer -- on
# p6-b300 over EFA. The PR itself was validated on IB/RoCE; EFA needs the whole
# NCCL-GIN comm stack that `Dockerfile` (this directory) builds, because DeepEP
# v2 on EFA runs over NCCL GIN, not NVSHMEM. See docs/调优改动清单_zh.md.
#
# ---------------------------------------------------------------------------
# WHY THIS IS A SECOND DOCKERFILE AND NOT A BUILD-ARG ON THE FIRST
# ---------------------------------------------------------------------------
# `Dockerfile` starts from nvcr.io/nvidia/pytorch (torch 2.12) and only needs to
# run DeepEP's own `tests/elastic/test_ep.py`. SGLang cannot be layered onto it:
# `sgl-kernel` ships as a compiled wheel built against one torch minor version,
# so a 2.13-built sgl-kernel on NGC's torch 2.12 dies on undefined symbols. The
# base has to be an SGLang image, and then the comm stack gets rebuilt on top --
# which is what this file does, step for step the same as `Dockerfile`, with the
# handful of base-image differences called out inline (marked "DIFF vs NGC").
#
# ---------------------------------------------------------------------------
# BASE IMAGE PIN -- do not float this
# ---------------------------------------------------------------------------
# nightly-dev-20260811-d59c1ddf is chosen for two reasons, both checked:
#   1. It carries `sglang-kernel 0.4.6.post1`, the exact version the PR's own
#      speed table was measured on.
#   2. Its sglang tree is d59c1ddf (2026-08-11 00:15 UTC), i.e. NEWER than the
#      PR's merge base: PR head b2d379cc is "Merge origin/main into
#      epv2-deepep-v2-backend" from 2026-08-10 09:17 UTC. So checking the PR
#      out over this tree rewinds main by ~15 h and adds the PR -- the smallest
#      drift any published nightly offers.
# Floating to :dev or a newer nightly re-opens the sgl-kernel-vs-PR skew.
ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-20260811-d59c1ddf
FROM ${BASE_IMAGE}

# ---- SGLang PR under test -------------------------------------------------
# Pinned to the PR head SHA, not to `pull/29525/head`, so a force-push upstream
# cannot silently change what we measured. Bump both together.
ARG SGLANG_PR=29525
ARG SGLANG_PR_SHA=b2d379cc72793bb220d6d79062d6331284c843ba
ARG SGLANG_SRC=/sgl-workspace/sglang

# ---- EFA userspace --------------------------------------------------------
# 1.50.0 is the first version with all three counting-event prereqs
# (efa.ko 3.3.0g + libfabric 2.6.0amzn1.0 + rdma-core 64.0amzn0), i.e. the only
# one GDAKI (route B) can be built on. It is not on the public release S3, so it
# is a staged tarball bind-mounted from the build context -- never COPYd:
#   aws s3 cp --no-sign-request --region us-west-2 \
#     s3://aws-efa-installer-dev/aws-efa-installer-latest.tar.gz \
#     ./aws-efa-installer-1.50.0.tar.gz
ARG EFA_INSTALLER_TARBALL=aws-efa-installer-1.50.0.tar.gz
ARG GDRCOPY_VERSION=2.5.2

# ---- NCCL -----------------------------------------------------------------
# The amazon fork, not NVIDIA's: `NCCL_GIN_TYPE_EFA_GDA = 5` exists only here
# (NVIDIA's nccl_device/core.h enumerates NONE=0, PROXY=2, GDAKI=3, GPI=4), so
# route B is impossible with an NVIDIA NCCL no matter what the host driver does.
# Source, not wheel: the DeepEP fork's csrc/kernels/backend/nccl.cu sets
# ginContextCount / ginSignalCount / ginQueueDepth / ginExclusiveContexts on
# ncclDevCommRequirements and links via EP_NCCL_ROOT_DIR.
ARG NCCL_REPO=https://github.com/amazon-contributing/upstream-to-nccl.git
ARG NCCL_REF=staging

ARG OFI_NCCL_REPO=https://github.com/aws/aws-ofi-nccl.git
ARG OFI_NCCL_REF=master
ARG OFI_NCCL_PREFIX=/opt/aws-ofi-nccl

# ---- DeepEP v2, AWS EFA fork ---------------------------------------------
# Xuan-1998/DeepEP is PRIVATE, so it comes from the build context, not a clone:
#   bash scripts/fetch_deepep_src.sh     # -> ./deepep-src
ARG DEEPEP_SRC=deepep-src

# kMaxParts 4 -> 1 is the one source-level tuning win that survived on b300
# (docs/调优改动清单_zh.md 改动 1): decode step -18.9% on p5en, and on b300 one
# image serves both prefill and decode. It cannot be an env var because
# kScaleoutSlotRoundingReserve feeds the HOST-side .so, which does not go through
# the JIT -- patching only the device half desynchronises the workspace layout.
# Set to 4 to build the stock geometry for an A/B.
ARG DEEPEP_KMAXPARTS=1

# p6-b300 is sm_103 -- NOT sm_100. A b200 (sm_100) cubin on b300 fails with
# "no kernel image is available", which reads like a driver bug. Single-arch on
# purpose: it halves nvcc time, and the image is not portable either way.
ARG TORCH_CUDA_ARCH_LIST="10.3"
ARG NVCC_GENCODE="-gencode=arch=compute_103,code=sm_103"

ARG CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 0. Purge every NCCL and DeepEP-v1 the base image ships.
#
#    DeepEP v2's check_nccl_so() byte-compares the libnccl in /proc/self/maps
#    against ${NCCL_HOME}/lib/libnccl.so*, and asserts on a duplicate. This base
#    has TWO providers, at two DIFFERENT versions, and they are easy to miss
#    because only one is on ldconfig:
#      - apt libnccl2 2.28.3 -> /lib/x86_64-linux-gnu/libnccl.so.2  (ldconfig)
#      - nvidia-nccl-cu13 2.29.7 -> dist-packages/nvidia/nccl/lib/... (what torch
#        actually resolves, via libtorch_cuda.so's rpath $ORIGIN/../../nvidia)
#    Removing only the apt one leaves torch loading the wheel's 2.29.7 and the
#    assert fires at `import deep_ep` with a confusing version pair.
#
#    libnccl2 / libnccl-dev are on apt HOLD (`hi`) in this base, so the remove
#    silently no-ops without --allow-change-held-packages -- and then the
#    ldconfig assertion below is what catches it.
#
#    sgl-deep-ep is DeepEP **v1** and installs as the module `deep_ep`, the same
#    name v2 uses, so it must go too (the PR says as much).
#
#    nvidia-nvshmem-cu13 3.4.5 is deliberately KEPT: the fork's setup.py still
#    links -l:libnvshmem_host.so.3 -l:libnvshmem_device.a unconditionally, so
#    metadata generation dies with "Cannot find package: nvshmem" without it --
#    even though V2 on EFA uses zero NVSHMEM at runtime.
# -----------------------------------------------------------------------------
RUN set -eux; \
    pip uninstall -y sgl-deep-ep 2>/dev/null || true; \
    pip uninstall -y nvidia-nccl-cu13 nvidia-nccl-cu12 nvidia-nccl 2>/dev/null || true; \
    apt-get update -y; \
    apt-get remove -y --allow-change-held-packages libnccl2 libnccl-dev || true; \
    ldconfig; \
    echo "--- deep_ep must be gone (v1 uninstalled) ---"; \
    test ! -e /usr/local/lib/python3.12/dist-packages/deep_ep; \
    echo "--- libnccl.so.2 must be gone entirely at this point ---"; \
    ! ldconfig -p | grep -q 'libnccl\.so\.2'; \
    test ! -e /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2; \
    echo OK
# `import torch` is deliberately NOT verified here: between this step and step 3
# there is no libnccl.so.2 on the system at all, so it would fail with
# ImportError: libnccl.so.2. That is expected; step 3 checks it.

# -----------------------------------------------------------------------------
# 1. Toolchain + EFA installer, userspace only (--skip-kmod).
#    efa.ko / efa_nv_peermem / gdrdrv / nvidia all live on the HOST and are
#    reached with --device=/dev/infiniband --device=/dev/gdrdrv. CE prereq #1
#    (efa.ko exporting comp_cntr) can ONLY come from running this same installer
#    WITHOUT --skip-kmod on the host -- no image can supply it.
#
#    DIFF vs NGC: this base lacks libtool (NGC has it), and has none of the
#    libnl/libudev/hwloc/numa headers the EFA installer's rdma-core wants.
# -----------------------------------------------------------------------------
RUN --mount=type=bind,target=/ctx \
    set -eux; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends \
      git ca-certificates curl wget \
      build-essential pkg-config autoconf automake libtool \
      libnuma-dev libhwloc-dev \
      libnl-3-dev libnl-route-3-dev libudev-dev \
      python3-dev; \
    test -f "/ctx/${EFA_INSTALLER_TARBALL}" \
      || { echo "ERROR: ${EFA_INSTALLER_TARBALL} is not in the build context." >&2; \
           echo "  aws s3 cp --no-sign-request --region us-west-2 \\" >&2; \
           echo "    s3://aws-efa-installer-dev/aws-efa-installer-latest.tar.gz \\" >&2; \
           echo "    ./aws-efa-installer-1.50.0.tar.gz" >&2; exit 1; }; \
    cd /tmp; tar -xf "/ctx/${EFA_INSTALLER_TARBALL}"; \
    cd aws-efa-installer; head -5 ChangeLog.md; \
    ./efa_installer.sh --disable-ngc -y --skip-kmod --skip-limit-conf --no-verify; \
    ldconfig; \
    cd /; rm -rf /tmp/aws-efa-installer*; rm -rf /var/lib/apt/lists/*; \
    dpkg-query -W -f='${Package} ${Version}\n' 'libfabric*' 'rdma-core' 2>/dev/null || true; \
    /opt/amazon/efa/bin/fi_info --version | head -1; \
    echo "--- CE prereq #2, decided right here at build time ---"; \
    grep -q cntr_open_ext /opt/amazon/efa/include/rdma/fi_ext_efa.h \
      && echo "CE#2 OK: fi_ext_efa.h has cntr_open_ext -> aws-ofi-nccl can enable GDAKI" \
      || { echo "CE#2 ABSENT: this installer cannot do GDAKI (route B)" >&2; exit 1; }
ENV EFA_PREFIX=/opt/amazon/efa
ENV PATH=/opt/amazon/efa/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
# Recorded so "which EFA stack is in this image?" is one command:
#   docker run --rm IMAGE cat /etc/deepep-build-env
RUN set -eux; \
    { echo "LIBFABRIC_PREFIX=/opt/amazon/efa"; \
      echo "EFA_LIBFABRIC=$(dpkg-query -W -f='${Version}' libfabric1-aws 2>/dev/null || echo unknown)"; \
      echo "EFA_RDMA_CORE=$(dpkg-query -W -f='${Version}' rdma-core 2>/dev/null || echo unknown)"; \
    } > /etc/deepep-build-env; cat /etc/deepep-build-env

# -----------------------------------------------------------------------------
# 2. GDRCopy userspace (libgdrapi). The gdrdrv kernel module is the HOST's; if
#    /dev/gdrdrv is missing, GIN init fails with "Failed to open gdr handle".
# -----------------------------------------------------------------------------
RUN set -eux; \
    cd /tmp; \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz; \
    tar xf v${GDRCOPY_VERSION}.tar.gz; cd gdrcopy-${GDRCOPY_VERSION}; \
    make -j"$(nproc)" lib lib_install CUDA=${CUDA_HOME} PREFIX=/usr/local; \
    ldconfig; test -f /usr/local/lib/libgdrapi.so; \
    cd /; rm -rf /tmp/gdrcopy* /tmp/v${GDRCOPY_VERSION}.tar.gz

# -----------------------------------------------------------------------------
# 3. GIN-capable NCCL from source, and the first point where torch works again.
# -----------------------------------------------------------------------------
ENV NCCL_HOME=/opt/nccl/build
RUN set -eux; \
    git clone "${NCCL_REPO}" /opt/nccl; \
    cd /opt/nccl; git checkout "${NCCL_REF}"; \
    git rev-parse HEAD > /opt/nccl.sha; \
    echo "NCCL ${NCCL_REF} @ $(cat /opt/nccl.sha) -- $(git log -1 --format=%s)"; \
    make -j"$(nproc)" src.build CUDA_HOME="${CUDA_HOME}" NVCC_GENCODE="${NVCC_GENCODE}"; \
    test -f ${NCCL_HOME}/include/nccl_device.h \
      || { echo "ERROR: nccl_device.h missing -- ${NCCL_REF} is not GIN-capable" >&2; exit 1; }; \
    grep -q 'NCCL_GIN_TYPE_EFA_GDA' src/include/nccl_device/core.h \
      || { echo "ERROR: no NCCL_GIN_TYPE_EFA_GDA=5 -- GDAKI impossible with this NCCL" >&2; exit 1; }; \
    echo ${NCCL_HOME}/lib > /etc/ld.so.conf.d/00-nccl-src.conf; ldconfig; \
    test -e ${NCCL_HOME}/lib/libnccl.so || ln -sf libnccl.so.2 ${NCCL_HOME}/lib/libnccl.so; \
    echo "--- exactly one libnccl.so.2, and it must be ours ---"; \
    test "$(ldconfig -p | grep -c 'libnccl\.so\.2')" = 1; \
    ldconfig -p | grep 'libnccl\.so\.2' | grep -q "${NCCL_HOME}/lib"; \
    python3 -c "import torch, ctypes; ctypes.CDLL('libnccl.so.2'); \
print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version(), '| resolved')"
ENV LD_LIBRARY_PATH=${NCCL_HOME}/lib:${LD_LIBRARY_PATH}
# LIBRARY_PATH, not just LD_LIBRARY_PATH: the fork's setup.py passes
#   extra_link_args += ['-l:libnccl.so', '-Wl,-rpath,{nccl_root}/lib']
# but -- unlike its NVSHMEM branch -- never adds {nccl_root}/lib to library_dirs,
# so no -L ever reaches the linker and g++ dies with `cannot find -l:libnccl.so`.
# LIBRARY_PATH is the documented link-time equivalent of -L; rpath is runtime only.
ENV LIBRARY_PATH=${NCCL_HOME}/lib:${LIBRARY_PATH:-}

# -----------------------------------------------------------------------------
# 4. aws-ofi-nccl -- this is what actually implements GIN on EFA.
#
#    --enable-gdaki and --with-nccl are DELIBERATELY NOT PASSED: both were
#    dropped in 1.21.x and now only produce "unrecognized options" warnings.
#    GDAKI auto-enables from configure.ac when all three of
#      ac_cv_have_decl_FI_EFA_GDA_OPS=yes, have_fi_efa_comp_cntr=1,
#      have_device_interface=cuda
#    hold, so the authoritative test is the configure NOTICE -- not a strings/nm
#    grep on the .so.
# -----------------------------------------------------------------------------
RUN set -eux; \
    . /etc/deepep-build-env; \
    git clone --recursive "${OFI_NCCL_REPO}" /tmp/aws-ofi-nccl; \
    cd /tmp/aws-ofi-nccl; git checkout "${OFI_NCCL_REF}"; \
    git submodule sync && git submodule update --init --recursive; \
    test -f 3rd-party/efa-dp-direct/CUDA/README.md \
      || { echo "ERROR: efa-dp-direct submodule empty -- GDAKI can never enable" >&2; exit 1; }; \
    ./autogen.sh; \
    { ./configure \
        --prefix=${OFI_NCCL_PREFIX} \
        --with-libfabric="${LIBFABRIC_PREFIX}" \
        --with-cuda=${CUDA_HOME} \
        --with-gdrcopy=/usr/local \
        --enable-platform-aws \
        --disable-tests \
      2>&1 | tee /var/log/ofi-configure.log \
      || { tail -n 160 config.log 2>/dev/null; exit 1; }; }; \
    make -j"$(nproc)"; make install; \
    test -f ${OFI_NCCL_PREFIX}/lib/libnccl-net-ofi.so; \
    strings ${OFI_NCCL_PREFIX}/lib/libnccl-net-ofi.so | grep -oE 'ncclGinPlugin_v[0-9]+' | sort -u || true; \
    grep -E 'GDAKI support (enabled|disabled)' /var/log/ofi-configure.log || true; \
    if grep -q 'GDAKI support enabled' /var/log/ofi-configure.log; then \
      echo "OFI_GDAKI=1" >> /etc/deepep-build-env; \
      echo "OK: GDAKI built in -- NCCL_GIN_TYPE=5 available (route B)"; \
    else \
      echo "OFI_GDAKI=0" >> /etc/deepep-build-env; \
      echo "WARN: proxy-only GIN -- NCCL_GIN_TYPE=2 (route A) only"; \
      grep -E 'FI_EFA_GDA_OPS|comp_cntr|device interface' /var/log/ofi-configure.log || true; \
    fi; \
    echo ${OFI_NCCL_PREFIX}/lib > /etc/ld.so.conf.d/00-aws-ofi-nccl.conf; ldconfig; \
    cd /; rm -rf /tmp/aws-ofi-nccl
ENV LD_LIBRARY_PATH=${OFI_NCCL_PREFIX}/lib:${LD_LIBRARY_PATH}

# -----------------------------------------------------------------------------
# 5. DeepEP V2, AWS EFA fork.
#
#    DIFF vs NGC: CCCL lives at ${CUDA_HOME}/include/cccl here, not
#    ${CUDA_HOME}/targets/x86_64-linux/include/cccl. DeepEP includes <cuda/...>,
#    so the wrong one silently means "not found" much later in nvcc.
#
#    envs.py needs no patching: the fork already reads NIC rate from
#    /sys/class/infiniband (_get_sysfs_rdma_gbs) instead of upstream's ibstat,
#    which is blind to EFA.
# -----------------------------------------------------------------------------
ENV CPATH=${CUDA_HOME}/include/cccl:${CPATH:-}
ENV EP_NCCL_ROOT_DIR=${NCCL_HOME}

COPY ${DEEPEP_SRC} /opt/deepep
RUN set -eux; \
    test -d ${CUDA_HOME}/include/cccl/cuda \
      || { echo "ERROR: CCCL not at ${CUDA_HOME}/include/cccl -- fix CPATH" >&2; exit 1; }; \
    ls /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib/libnvshmem_device.a >/dev/null; \
    cd /opt/deepep; \
    git config --global --add safe.directory /opt/deepep; \
    git rev-parse HEAD > /opt/deepep.sha; cat /opt/deepep.sha; \
    test -f csrc/kernels/backend/nccl.cu \
      || { echo "ERROR: not a V2 (NCCL GIN) tree" >&2; exit 1; }; \
    test -f third-party/fmt/CMakeLists.txt \
      || { echo "ERROR: third-party/fmt submodule not populated in the context" >&2; exit 1; }; \
    KP=deep_ep/include/deep_ep/common/gin_resource_alloc.cuh; \
    if [ "${DEEPEP_KMAXPARTS}" != "4" ]; then \
      sed -i "s/^static constexpr int kMaxParts = 4;/static constexpr int kMaxParts = ${DEEPEP_KMAXPARTS};/" "$KP"; \
      grep -q "kMaxParts = ${DEEPEP_KMAXPARTS};" "$KP" \
        || { echo "ERROR: kMaxParts patch did not apply -- upstream moved the line" >&2; exit 1; }; \
    fi; \
    grep -n 'kMaxParts = ' "$KP" | head -2; \
    echo "kMaxParts=${DEEPEP_KMAXPARTS}" >> /etc/deepep-build-env; \
    if [ "${TORCH_CUDA_ARCH_LIST}" = "9.0" ]; then DIS_PTX=0; else DIS_PTX=1; fi; \
    echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} DISABLE_AGGRESSIVE_PTX_INSTRS=${DIS_PTX}"; \
    DISABLE_AGGRESSIVE_PTX_INSTRS=${DIS_PTX} \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    EP_NCCL_ROOT_DIR=${NCCL_HOME} \
      pip install --no-cache-dir --no-build-isolation -v .; \
    cd /tmp; \
    python3 -c "import importlib.metadata as m; print('deep_ep', m.version('deep_ep'))"; \
    python3 -c "import glob; \
p=glob.glob('/usr/local/lib/python3.12/dist-packages/deep_ep/buffers/elastic.py'); \
assert p, 'ElasticBuffer source missing from the installed wheel'; print('elastic.py', p[0])"
# `import deep_ep` cannot be verified at build time: it runs check_nccl_so() and
# init_jit(), which need a CUDA device. Verified by scripts/check_sglang_epv2.sh
# on a GPU node instead.

# -----------------------------------------------------------------------------
# 6. The PR under test, checked out over the image's own sglang tree.
#
#    The base installs sglang editable from ${SGLANG_SRC}/python (a SHALLOW clone
#    with a normal `origin`, so fetching a PR ref works), so a checkout is almost
#    enough -- but PEP 660 editable installs can carry a static module map, and
#    this PR ADDS files (token_dispatcher/deepep_v2.py). Re-running the editable
#    install with --no-deps is cheap and removes that doubt.
#
#    The reinstall is best-effort: setuptools_scm derives the version through
#    python/tools/get_version_tag.py, which can come up empty on a detached HEAD
#    in a shallow clone. That is harmless (fallback_version), but if pip fails for
#    any other reason the existing finder still maps the `sglang` package at
#    package level, so new submodules resolve anyway. The real gate is the import
#    + enum assertion below -- baseline `MoeA2ABackend` does NOT contain
#    'deepep_v2' (checked: none/deepep/mooncake/nixl/mori/... only), so that
#    assertion fails loudly if the PR code is not actually live.
# -----------------------------------------------------------------------------
RUN set -eux; \
    cd ${SGLANG_SRC}; \
    git config --global --add safe.directory ${SGLANG_SRC}; \
    echo "base sglang: $(git log -1 --format='%h %ad %s')"; \
    : 'The published image bakes the GitHub Actions token that built it into'; \
    : '.git/config as http.https://github.com/.extraheader. It is long expired,'; \
    : 'so every fetch gets a 401 and git then asks for a username, failing with'; \
    : 'the misleading "could not read Username for https://github.com". Drop it'; \
    : 'and anonymous HTTPS works (sgl-project/sglang is public).'; \
    git config --unset-all 'http.https://github.com/.extraheader' || true; \
    ! git config --get-regexp '^http\..*extraheader$'; \
    GIT_TERMINAL_PROMPT=0 git fetch --depth 200 origin "refs/pull/${SGLANG_PR}/head"; \
    git checkout --detach "${SGLANG_PR_SHA}"; \
    git rev-parse HEAD > /opt/sglang-pr.sha; \
    echo "PR #${SGLANG_PR} @ $(cat /opt/sglang-pr.sha) -- $(git log -1 --format=%s)"; \
    test -f python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py \
      || { echo "ERROR: PR tree has no deepep_v2.py -- wrong SHA?" >&2; exit 1; }; \
    pip install --no-cache-dir --no-deps --no-build-isolation -e python/ \
      || echo "WARN: editable reinstall failed; falling back to the existing finder"; \
    cd /tmp; \
    python3 -c "import sglang, sglang.srt.layers.moe.token_dispatcher.deepep_v2 as d; \
print('sglang', sglang.__file__); print('deepep_v2 module OK')"; \
    python3 -c "from sglang.srt.layers.moe.utils import MoeA2ABackend as B; \
vs=[m.value for m in B]; assert 'deepep_v2' in vs, vs; print('MoeA2ABackend:', vs)"

# -----------------------------------------------------------------------------
# 6b. sm_90 ONLY: let SGLANG_DSV4_FP4_DEQUANT=1 coexist with the deepep_v2 runner.
#
# DSV4's routed experts are MXFP4 (config.json `expert_dtype: fp4`; the
# `quantization_config` block is the FP8 128x128 layout of the DENSE/attention
# weights). Hopper has no FP4, so the only way to serve this checkpoint on
# p5.48xlarge is SGLANG_DSV4_FP4_DEQUANT=1, which walks the experts through
# cast_e2m1fn_to_e4m3fn() and lands on plain 128x128 block-scaled FP8
# (fp8.py, `scale_param.format_ue8m0 = False; self.is_fp4_expert = False`) --
# exactly the DeepSeek FP8 blockwise layout Hopper deep_gemm already serves.
#
# The single obstacle is an assert that the MoE runner is still `auto`. It cannot
# hold under deepep_v2: server_args.py's deepep_v2 handler resolves auto ->
# deep_gemm (or triton with --deepep-v2-dispatcher-output-dtype bf16) BEFORE any
# quant method is built. The assert's intent is to exclude the specialised mxfp4
# runners (marlin / humming / flashinfer_mxfp4), which are the three branches
# immediately below it -- not the post-dequant FP8 path, which by then has
# is_fp4_expert=False and no FP4 left to run. So we widen it to deep_gemm/triton.
#
# Default 0: a b300 build is byte-identical to before this ARG existed.
# The `$` anchor is load-bearing -- fp8.py has two other `.is_auto():` lines
# (both `moe_runner_backend.is_auto():`, with a colon) that must not be touched;
# the count check below is what proves the sed hit exactly the intended one.
# -----------------------------------------------------------------------------
ARG SGLANG_FP4_DEQUANT_ANY_RUNNER=0
ENV SGLANG_FP4_DEQUANT_ANY_RUNNER=${SGLANG_FP4_DEQUANT_ANY_RUNNER}
RUN set -eux; \
    if [ "${SGLANG_FP4_DEQUANT_ANY_RUNNER}" != "1" ]; then \
      echo "fp4-dequant assert left stock (b300/sm_10x build)"; \
    else \
      f=${SGLANG_SRC}/python/sglang/srt/layers/quantization/fp8.py; \
      n=$(grep -c 'get_moe_runner_backend()\.is_auto()$' "$f"); \
      test "$n" = 1 || { echo "ERROR: expected 1 anchor line, found $n" >&2; exit 1; }; \
      sed -i 's/get_moe_runner_backend()\.is_auto()$/get_moe_runner_backend().is_auto()\n                    or get_moe_runner_backend().is_deep_gemm()\n                    or get_moe_runner_backend().is_triton()/' "$f"; \
      grep -n -A4 'is not compatible with SGLANG_DSV4_FP4_DEQUANT' "$f" | head -8; \
      test "$(grep -c 'get_moe_runner_backend()\.is_deep_gemm()$' "$f")" = 1; \
      python3 -c "import py_compile,sys; py_compile.compile(sys.argv[1], doraise=True)" "$f"; \
      python3 -c "from sglang.srt.layers.moe.utils import MoeRunnerBackend as R; \
[getattr(R, m) for m in ('is_auto','is_deep_gemm','is_triton')]; print('runner predicates OK')"; \
      echo "fp4-dequant assert widened to deep_gemm/triton (sm_90 build)"; \
    fi

# -----------------------------------------------------------------------------
# 7. Runtime env: AWS EFA + NCCL GIN + what PR #29525 requires.
# -----------------------------------------------------------------------------
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_EFA_USE_HW_CNTR=1
ENV NVIDIA_GDRCOPY=enabled
ENV NCCL_OFI_RDMA_GDR_FLUSH_DISABLE=0
ENV NCCL_NET_PLUGIN=${OFI_NCCL_PREFIX}/lib/libnccl-net-ofi.so
ENV NCCL_GIN_PLUGIN=${OFI_NCCL_PREFIX}/lib/libnccl-net-ofi.so
# EFA GDA cannot do strong signals, so NCCL's own symmetric kernels must not ask
# for them: sym_kernels.cc hard-sets ginStrongSignalsRequired under the GIN
# branch and ncclDevCommCreate then fails with ncclInternalError.
ENV NCCL_SYM_GIN_KERNELS_ENABLE=0
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth
# Route A (2 = PROXY) by default; the launcher sets 5 (EFA_GDA) when GDAKI=1, so
# route A/B runs are always distinguishable in the logs.
ENV NCCL_GIN_TYPE=2
#
# PR #29525 requirement: DeepEP v2 needs NCCL symmetric memory, and SGLang sets
# NCCL_CUMEM_ENABLE=int(enable_symm_mem) only when the variable is UNSET.
# Without this an otherwise correct launch dies with
#   nccl.cu:104: Communicator does not support symmetric memory!
# Setting it here (rather than relying on --enable-symm-mem) avoids also turning
# on NVLS and a 4 GB symmetric-memory prealloc.
ENV NCCL_CUMEM_ENABLE=1
#
# p6-b300 exposes 8x rdmap*@400Gb/s AND 2x ibp19{8,9}s0f0@100Gb/s. EP_NIC_NAME
# must name a 400G device: get_theoretical_num_sms() derives per-GPU line rate
# from ONE device's sysfs rate, so a 100G pick mis-sizes the kernel -- and those
# two also fail ce_probe with errno=95. The name is PCI-derived and therefore
# instance-specific, so the launcher auto-detects and passes -e; this default
# only matters for a hand-rolled `docker run`.
ARG EP_NIC_NAME=rdmap101s0
ENV EP_NIC_NAME=${EP_NIC_NAME}

WORKDIR ${SGLANG_SRC}
CMD ["/bin/bash"]
