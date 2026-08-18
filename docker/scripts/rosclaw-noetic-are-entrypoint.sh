#!/usr/bin/env bash
set -e

ORIG_ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
source /opt/rosclaw/ros1_ws/devel/setup.bash
set -- "${ORIG_ARGS[@]}"

export PYTHONPATH="/opt/rosclaw/src:${PYTHONPATH:-}"
export GAZEBO_MODEL_PATH="/opt/rosclaw/third_party/ros1/are/src/vehicle_simulator/mesh:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI="${GAZEBO_MODEL_DATABASE_URI:-}"
export ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9090}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export DISABLE_ROS1_EOL_WARNINGS="${DISABLE_ROS1_EOL_WARNINGS:-1}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export NO_AT_BRIDGE="${NO_AT_BRIDGE:-1}"
export ROSCLAW_XDG_RUNTIME_DIR="${ROSCLAW_XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p "${ROSCLAW_XDG_RUNTIME_DIR}" /opt/rosclaw/ros1_ws/src/vehicle_simulator/log
chmod 700 "${ROSCLAW_XDG_RUNTIME_DIR}" 2>/dev/null || true

if [[ -d /mnt/wslg/runtime-dir ]]; then
  export DISPLAY="${DISPLAY:-:0}"
  export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
  export XDG_RUNTIME_DIR="${ROSCLAW_XDG_RUNTIME_DIR}"
fi

exec "$@"
