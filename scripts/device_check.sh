#!/bin/bash
# Emit a ladder schedule, copy it to a Trainium node, and run the nkibench layer plus the bf16 check.
#
#   scripts/device_check.sh l5            # needs: python with torch+numpy, kubectl access
#
# Environment (defaults match the hackathon setup in AGENTS.md):
#   ENV_FILE  file exporting AWS credentials     (default: ../.env relative to the repo root's parent)
#   POD       pod to use                         (default: seat-270)
#   PYTHON    interpreter with torch             (default: python3)
set -euo pipefail
LEVEL_NAME=${1:?usage: device_check.sh l4|l5|l6|l7}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
POD=${POD:-seat-270}
PYTHON=${PYTHON:-python3}
ENV_FILE=${ENV_FILE:-$ROOT/../.env}
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
N=${LEVEL_NAME#l}
OUT=$(mktemp -d)/kernel_$LEVEL_NAME.py

"$PYTHON" "$ROOT/examples/matmul_ladder.py" "$LEVEL_NAME" "$OUT" >/dev/null
ENTRY=$(grep -m1 '^def ' "$OUT" | sed 's/def \([a-zA-Z0-9_]*\).*/\1/')
kubectl cp -c app "$OUT" "$POD:/tmp/$(basename "$OUT")"
kubectl cp -c app "$ROOT/scripts/sim_dtypes.py" "$POD:/tmp/sim_dtypes.py"
echo "== nkibench level $N =="
kubectl exec "$POD" -c app -- bash -c "cd /workspace/projects/02-kernel-agent && python nkibench.py --level $N --check /tmp/$(basename "$OUT")"
echo "== dtype check (float32 + bfloat16) =="
kubectl exec "$POD" -c app -- python /tmp/sim_dtypes.py "/tmp/$(basename "$OUT")" "$ENTRY"
