#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/_cmu_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_cmu_common.sh"

if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ROSClaw] cmu-chat requires DEEPSEEK_API_KEY or OPENAI_API_KEY in this terminal."
  echo "[ROSClaw] Example: export DEEPSEEK_API_KEY='...'"
  exit 2
fi

if ! cmu_compose ps --status running --services | grep -qx "rosclaw"; then
  echo "[ROSClaw] The rosclaw simulation container is not running."
  echo "[ROSClaw] Start it first with: ./scripts/start_cmu_rviz.sh"
  exit 2
fi

exec docker compose -f "${CMU_COMPOSE_FILE}" exec \
  -e DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}" \
  -e DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-}" \
  -e DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
  -e OPENAI_MODEL="${OPENAI_MODEL:-}" \
  -e CMU_PLACES="${CMU_PLACES:-docker/ros1/places.campus.yaml}" \
  -e CMU_OUTPUT_DIR="${CMU_OUTPUT_DIR:-practice_data/app_runs}" \
  -e CMU_NAV_TIMEOUT="${CMU_NAV_TIMEOUT:-120}" \
  -e CMU_READINESS_TIMEOUT="${CMU_READINESS_TIMEOUT:-60}" \
  -e CMU_NAV_TOLERANCE="${CMU_NAV_TOLERANCE:-1.5}" \
  -e CMU_SPEED="${CMU_SPEED:-2.0}" \
  -e CMU_MAX_RELATIVE_M="${CMU_MAX_RELATIVE_M:-20}" \
  -e CMU_CHAT_PROGRESS_INTERVAL="${CMU_CHAT_PROGRESS_INTERVAL:-3}" \
  -e CMU_MAX_SEQUENCE_STEPS="${CMU_MAX_SEQUENCE_STEPS:-8}" \
  -e CMU_CIRCLE_SEGMENTS="${CMU_CIRCLE_SEGMENTS:-12}" \
  -e CMU_MAX_CIRCLE_RADIUS="${CMU_MAX_CIRCLE_RADIUS:-6}" \
  -e CMU_TASK_PREEMPT_POLICY="${CMU_TASK_PREEMPT_POLICY:-preempt}" \
  -e CMU_EXPLORATION_ON_MANUAL="${CMU_EXPLORATION_ON_MANUAL:-pause}" \
  rosclaw rosclaw app cmu-chat "$@"
