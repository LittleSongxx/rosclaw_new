#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/_cmu_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_cmu_common.sh"

echo "[ROSClaw] Stopping CMU ARE + RViz container..."
if cmu_compose ps --status running --services | grep -qx "rosclaw"; then
  cmu_compose exec -T rosclaw bash -lc 'pkill -TERM -x rviz 2>/dev/null || true'
else
  echo "[ROSClaw] rosclaw service is not running."
fi
cmu_compose stop --timeout 20 rosclaw

echo "[ROSClaw] Cleaning leftover RViz processes, if any..."
pkill -TERM -x rviz 2>/dev/null || true

if cmu_compose ps --status running --services | grep -qx "rosclaw"; then
  echo "[ROSClaw] rosclaw did not stop cleanly; forcing it down..."
  cmu_compose kill rosclaw >/dev/null 2>&1 || true
fi

echo "[ROSClaw] Current compose status:"
cmu_compose ps
echo "[ROSClaw] Done. The seekdb helper service is left running."
