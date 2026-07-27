# Shared setup for the CMU ARE helper scripts.
#
# Sourced, not executed. Resolves the repo root, loads .env, and exports the
# compose file for the ROS1 CMU ARE stack (kept separate from the repo-root
# docker-compose.yml, which runs the mainline ROSClaw runtime).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Callers may override to point at a different stack definition.
CMU_COMPOSE_FILE="${CMU_COMPOSE_FILE:-docker/ros1/docker-compose.ros1-are.yml}"

if [[ ! -f "${CMU_COMPOSE_FILE}" ]]; then
  echo "[ROSClaw] Compose file not found: ${CMU_COMPOSE_FILE}" >&2
  exit 2
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Wrapper so every script talks to the same stack.
cmu_compose() {
  docker compose -f "${CMU_COMPOSE_FILE}" "$@"
}
