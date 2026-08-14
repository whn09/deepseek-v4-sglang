#!/bin/bash
# Download DeepSeek-V4-Flash weights to the host NVMe. Run on EVERY node: a
# multi-node SGLang loads shards locally on each rank, there is no broadcast.
#
#   bash 00_download_model.sh
#
# /opt/dlami/nvme is EPHEMERAL on these instances -- a stop/start wipes it, so
# this has to be re-run after every restart.
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

# HF_XET_HIGH_PERFORMANCE, not HF_HUB_ENABLE_HF_TRANSFER: the latter is
# deprecated and these repos route through Xet, which ignores it. ~6 GB/s
# measured on p6-b300, i.e. ~30 s/10 GB.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# The DLAMI's python has no huggingface_hub; the venv's CLI does.
HF_BIN="${HF_BIN:-/opt/pytorch/bin/hf}"
[[ -x "$HF_BIN" ]] || { echo "ERROR: $HF_BIN not found (set HF_BIN=)" >&2; exit 1; }

DEST="$HOST_MODEL_DIR/$MODEL_NAME"
mkdir -p "$HOST_MODEL_DIR"
echo "downloading $HF_REPO -> $DEST"
df -h "$HOST_MODEL_DIR" | tail -1

"$HF_BIN" download "$HF_REPO" --local-dir "$DEST"

du -sh "$DEST"
python3 - "$DEST" <<'EOF'
import json, sys, glob, os
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
q = cfg.get("quantization_config", {})
print("arch      :", cfg.get("architectures"))
print("layers    :", cfg.get("num_hidden_layers"), "experts:", cfg.get("n_routed_experts"),
      "topk:", cfg.get("num_experts_per_tok"))
print("quant     :", q.get("quant_method"), q.get("fmt"), "scale_fmt:", q.get("scale_fmt"),
      "block:", q.get("weight_block_size"))
print("shards    :", len(glob.glob(os.path.join(d, "*.safetensors"))))
EOF
