# p5en.48xlarge (H200 141 GB / sm_90, us-east-2) overrides. Source BEFORE any script:
#
#   source env_p5en.sh; bash 20_launch_node.sh 0
#
# p5en IS NOT p5, and the difference is not just the GPU:
#
#   p5.48xlarge    H100 80 GB    32 x 100 Gb/s EFA gen-1   = 3200 Gb/s = 50 GB/s per GPU
#   p5en.48xlarge  H200 141 GB   16 x 200 Gb/s EFA         = 3200 Gb/s = 50 GB/s per GPU
#
# So the FABRIC is the same aggregate and the same per-GPU rate; what changes is
# 61 GB more HBM per GPU (which is what lifts the prefill CAPACITY ceiling, see
# below) and the EFA generation (which is what makes GDA a question at all).
#
# GDAKI=1 IS LIVE HERE AS OF 2026-08-19, and getting there took ONE 82 KB deb.
# MEASURED from inside the container after the fix, all 16 devices:
#   hw_ver=0xEFA2  device_caps=0x0000007F  CAPS_COMP_CNTR=YES
#   ibv_query_comp_cntr_caps=ok (max_counters=512 max_value=2147483647
#                                attach_ops=0x37)
# versus p5.48xlarge's 0xEFA1 / 0x3F with bit 6 CLEAR. That is the whole story:
# EFA gen-2 silicon has counting events, gen-1 does not.
#
# THE FIX, if these instances are ever rebuilt (a fresh p5en DLAMI is NOT ready):
#   1. Extract ONE file from the 1.50.0 dev installer -- the full 627 MB tarball is
#      not needed, and neither are its libfabric/rdma-core, because the workload's
#      userspace is the IMAGE's (already 2.6.0amzn1.0, see below):
#        tar xzf aws-efa-installer-1.50.0.tar.gz \
#          aws-efa-installer/DEBS/UBUNTU2604/x86_64/efa-driver/   # 82 KB, DKMS src
#   2. sudo dpkg -i efa_3.3.0-1.amzn1_amd64.deb        # DKMS rebuilds for the kernel
#   3. sudo modprobe -r efa_nv_peermem; sudo modprobe -r efa; sudo modprobe efa
#      NO REBOOT NEEDED. `modprobe -r efa` prints "FATAL: Module ib_uverbs is in
#      use" -- IGNORE IT, that is modprobe declining to also remove ib_uverbs;
#      efa itself unloads and reloads. Verify by VERSION, not by exit code:
#        cat /sys/module/efa/version   # must read 3.3.0g, not 3.1.0g
#      Checking only the on-disk module is the trap: dpkg leaves the NEW module on
#      disk and the OLD one live, and check_gdaki_prereqs.sh's srcversion compare
#      is what catches it.
#   4. Re-run check_gdaki_prereqs.sh; CE #1 must flip to PASS.
# CE #4 (nvidia PeerMappingOverride) is still unset and was NOT needed.
#
# Everything below documents WHY only that one deb was required.
#
# THE ORIGINAL STATE WAS A SOFTWARE FACT, NOT A HARDWARE ONE -- the opposite of p5.
# On p5 the silicon itself lacks counting events (hw_ver=0xEFA1, device_caps=0x3F
# with CAPS_COMP_CNTR clear), so no installer could ever fix it. p5en is on AWS's
# OWN validated GDAKI list (device_tests_v1.21, 2026-08-10: p5en + p6-b200 +
# p6-b300, on efa.ko 3.3.0g). What blocks it on a FRESH p5en DLAMI is the stack
# version, measured 2026-08-19 with check_gdaki_prereqs.sh:
#
#   CE #1  efa.ko 3.1.0g          comp_cntr_strings=0   FAIL  <- host kernel module
#   CE #2  libfabric 2.4.0amzn5.0 no cntr_open_ext      FAIL  <- 2.5+ required
#   CE #3  rdma-core 63.0-1                             pass
#   CE #4  PeerMappingOverride    unset                 warn
#
# i.e. exactly the public EFA installer 1.49.0 stack (all four lines above are the
# BEFORE state, kept because a rebuilt instance starts here again).
#
# ONLY CE #1 ACTUALLY BLOCKS US, and it is the one a container cannot fix.
# CE #2/#3 look like host failures above but are irrelevant at runtime, because
# the workload's libfabric is the IMAGE's, not the host's -- and the
# :pr29525-sm90 image already carries libfabric 2.6.0amzn1.0 with
# OFI_GDAKI=1 baked in (verified: `. /etc/deepep-build-env` -> GDAKI=1), because
# 10_build_image.sh installs the 1.50.0 stack INSIDE the image. So the aws-ofi-nccl
# configure gate ("GDAKI support disabled ... building proxy-only GIN") was already
# passed at build time on p5, even though p5 itself could never use the result.
# What is left is purely the HOST kernel module: efa.ko must export counting
# events, and 3.1.0g has zero comp_cntr symbols. Fix = dev EFA installer 1.50.0
# on the host (efa.ko 3.3.0g); everything else is already in place.
#
# Default 1 because CE #1 now passes on both hosts. Set GDAKI=0 explicitly for the
# proxy-GIN control arm -- that comparison is the point of running here at all, and
# it is the only apples-to-apples GDA-vs-proxy measurement we can make (on p5 the
# hardware forced proxy; on b300 nobody wants proxy).
# If GIN init ever fails with "not supported on this platform", the live efa.ko has
# gone back to 3.1.0g (a kernel upgrade rebuilds DKMS; a reboot reloads it) -- check
# `cat /sys/module/efa/version` before suspecting anything else.
export GDAKI="${GDAKI:-1}"

# Same sm_90 image as p5: MXFP4 experts still have to be dequantised at load time
# (H200 is Hopper, no FP4), and the tag must not collide with a b300 :pr29525.
export IMAGE="${IMAGE:-sglang-epv2-efa:pr29525-sm90}"
export ECR_REGION=us-east-2
export ECR_TAG="${ECR_TAG:-pr29525-sm90}"

# Re-check after any stop/start -- private IPs are reassigned and NCCL bootstrap
# then times out silently. 20_launch_node.sh verifies rank 0 owns this address.
export MASTER_IP="${MASTER_IP:-172.31.9.118}"   # P5EN-1 (rank 0), as of 2026-08-19.
                                                # P5EN-2 = 172.31.6.47

# NOT overridden on purpose:
#   EP_NIC_NAME -- detect_ep_nic picks the fastest live rdmap* (here 200 Gb/s;
#     the p5 name rdmap113s0 does not exist on p5en, the devices are rdmap85s0,
#     rdmap86..88, rdmap110..113, rdmap135..138, rdmap160..163).
#   FP4_DEQUANT -- 20_launch_node.sh derives it from compute_cap (9.0 here), the
#     one check that cannot go stale.
#   MEM_FRACTION -- and this is the interesting one. env_common's per-arch default
#     keys off `MEM_TOTAL_GB < 120000`, and H200 reports 143771 MiB, so p5en takes
#     the *b300* branch (prefill 0.80, not p5's 0.65). That is correct: the p5
#     prefill CAPACITY=4096 ceiling was a MEMORY limit (the masked grouped-GEMM
#     transient wants ~2 GiB per 1024 capacity and 80 GB could not spare 16 GiB
#     for 8192), and 141 GB can. So p5en should be able to run b300's
#     CAPACITY=8192 prefill row -- worth measuring both 4096 (p5-comparable) and
#     8192 (b300-comparable), since that is the one config axis where p5 was
#     hardware-limited and p5en may not be.
