# p5.48xlarge (H100 / sm_90, us-east-2) overrides. Source BEFORE any script:
#
#   source env_p5.sh; ARCH=sm90 bash 10_build_image.sh
#   source env_p5.sh; bash 20_launch_node.sh 0
#
# env_common.sh reads every one of these as ${VAR:-default}, so exporting here is
# enough -- do not edit env_common.sh, the b300 defaults have to stay intact.
#
# WHY THESE FOUR AND ONLY THESE FOUR:
#
# GDAKI=0 is not a tuning choice, it is the hardware. All 32 EFA devices on
# p5.48xlarge report hw_ver=0xEFA1 (EFA gen-1) with device_caps=0x3F, i.e.
# EFADV_DEVICE_ATTR_CAPS_COMP_CNTR (bit 6) CLEAR, while
# ibv_query_comp_cntr_caps() itself succeeds returning max_counters=0 -- so the
# efa.ko/rdma-core CE ABI is fine and the silicon simply has no counting events.
# Counting events are GDAKI's one hard prerequisite, so NCCL_GIN_TYPE=5 fails GIN
# init outright (nccl_ofi_gin_gdaki_createContext:660 "not supported on this
# platform") rather than falling back. Route A (proxy GIN, =2) is the only route.
# Verify with: bash ../ep-benchmarks-efa/deepep-v2-efa-gdaki-b200/scripts/check_gdaki_prereqs.sh
#
# IMAGE gets its own tag because sm_90 changes the compiled arch AND relaxes the
# fp4-dequant assert. A b300 :pr29525 image would run here up to the first kernel
# launch and then fail somewhere unhelpful, so the tags must not collide.
#
# ECR_REGION: these nodes are us-east-2; the b300 fleet is us-west-2. The repo may
# not exist here yet (11_sync_image_ecr.sh creates it).
#
# NOT overridden on purpose: EP_NIC_NAME (detect_ep_nic picks a live rdmap*; the
# baked b300 default rdmap101s0 does not exist here) and FP4_DEQUANT
# (20_launch_node.sh derives it from compute_cap, which is the check that cannot
# go stale).
export GDAKI=0
export IMAGE="${IMAGE:-sglang-epv2-efa:pr29525-sm90}"
export ECR_REGION=us-east-2
export ECR_TAG="${ECR_TAG:-pr29525-sm90}"
# MASTER_IP, not B300_1_IP: the leader address is the only thing the kit reads out
# of that pair (12_preflight_args.sh:58 and 20_launch_node.sh's --dist-init-addr),
# and the b300-flavoured names would just mislead the next reader.
# Re-check after any stop/start -- private IPs are reassigned and NCCL bootstrap
# then times out silently.
export MASTER_IP=172.31.44.207   # P5-1 (rank 0), as of 2026-08-18.  P5-2 = 172.31.34.46
                                 # (2026-08-17 they were .6.202 / .10.234 -- they DID move)
