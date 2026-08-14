#!/bin/bash
# Push this directory to the b300 hosts. Edit locally, sync, run there.
#
#   bash sync.sh            # both hosts
#   bash sync.sh P6-B300-2  # one host
#
# --exclude 'results/' is LOAD-BEARING: --delete without it wipes the measurement
# logs that only exist on the hosts. Pull them back with 99_fetch_results.sh
# style rsync, never by syncing the other direction over this.
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

HOSTS=("${@:-P6-B300-1 P6-B300-2}")
read -r -a HOSTS <<< "${HOSTS[*]}"

for h in "${HOSTS[@]}"; do
    echo "=== $h:$SCRIPT_DIR_HOST ==="
    ssh "$h" "mkdir -p '$SCRIPT_DIR_HOST/results'"
    rsync -av --delete \
        --exclude 'results/' \
        --exclude '.git/' \
        --exclude '.DS_Store' \
        ./ "$h:$SCRIPT_DIR_HOST/"
done
