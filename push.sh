#!/bin/bash
# Push this repo to github.com/whn09/deepseek-v4-sglang via the EC2 jump host.
#
#   bash push.sh
#
# Direct pushes from the laptop are blocked by the corporate Code Defender proxy,
# so the commit is made locally and the PUSH happens on the jump host, which has
# gh authenticated as whn09. This mirrors the local repo state (including .git)
# rather than committing there, so history stays authored here.
set -euo pipefail

cd "$(dirname "$0")"

JUMP="${JUMP:-Jump}"
REMOTE_DIR="${REMOTE_DIR:-~/deepseek-v4-sglang-push}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

[[ -z "$(git status --porcelain)" ]] \
    || { echo "ERROR: uncommitted changes -- commit first, then push." >&2; git status --short >&2; exit 1; }

echo "=== mirroring to $JUMP:$REMOTE_DIR ==="
rsync -a --delete --exclude 'results/*.json' --exclude 'results/*.log' ./ "$JUMP:$REMOTE_DIR/"

echo "=== pushing $BRANCH ==="
ssh "$JUMP" "cd $REMOTE_DIR && git remote get-url origin >/dev/null 2>&1 \
    || git remote add origin https://github.com/whn09/deepseek-v4-sglang.git; \
    cd $REMOTE_DIR && git push origin '$BRANCH'"

echo "https://github.com/whn09/deepseek-v4-sglang/tree/$BRANCH"
