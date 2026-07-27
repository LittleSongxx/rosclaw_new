#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/_cmu_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_cmu_common.sh"

MODE="${DASHBOARD_MODE:-sidecar}"

if [[ "${MODE}" == "sidecar" ]]; then
  cmu_compose --profile dashboard up -d cmu-dashboard
  echo "[ROSClaw] CMU dashboard is running at: http://localhost:${CMU_DASHBOARD_PORT:-18770}"
  echo "[ROSClaw] Stop it with: docker compose -f ${CMU_COMPOSE_FILE} --profile dashboard stop cmu-dashboard"
  exit 0
fi

if ! cmu_compose ps --status running --services | grep -qx "rosclaw"; then
  echo "[ROSClaw] The rosclaw simulation container is not running."
  echo "[ROSClaw] Start it first with: ./scripts/start_cmu_rviz.sh"
  exit 2
fi

exec docker compose -f "${CMU_COMPOSE_FILE}" exec \
  -e CMU_OUTPUT_DIR="${CMU_OUTPUT_DIR:-practice_data/app_runs}" \
  -e CMU_DASHBOARD_HOST="${CMU_DASHBOARD_HOST:-0.0.0.0}" \
  -e CMU_DASHBOARD_PORT="${CMU_DASHBOARD_PORT:-18770}" \
  -e CMU_DASHBOARD_MAX_POINTS="${CMU_DASHBOARD_MAX_POINTS:-2000}" \
  rosclaw rosclaw app cmu-dashboard \
    --host "${CMU_DASHBOARD_HOST:-0.0.0.0}" \
    --port "${CMU_DASHBOARD_PORT:-18770}" \
    --max-points "${CMU_DASHBOARD_MAX_POINTS:-2000}" \
    "$@"
