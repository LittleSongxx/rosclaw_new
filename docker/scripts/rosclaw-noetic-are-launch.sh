#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
source /opt/rosclaw/ros1_ws/devel/setup.bash

export PYTHONPATH="/opt/rosclaw/src:${PYTHONPATH:-}"
export GAZEBO_MODEL_PATH="/opt/rosclaw/third_party/ros1/are/src/vehicle_simulator/mesh:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_MODEL_DATABASE_URI="${GAZEBO_MODEL_DATABASE_URI:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export DISABLE_ROS1_EOL_WARNINGS="${DISABLE_ROS1_EOL_WARNINGS:-1}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export NO_AT_BRIDGE="${NO_AT_BRIDGE:-1}"
export ROSCLAW_XDG_RUNTIME_DIR="${ROSCLAW_XDG_RUNTIME_DIR:-/tmp/runtime-root}"
mkdir -p "${ROSCLAW_XDG_RUNTIME_DIR}" /opt/rosclaw/ros1_ws/src/vehicle_simulator/log
chmod 700 "${ROSCLAW_XDG_RUNTIME_DIR}" 2>/dev/null || true

WORLD="${CMU_ARE_WORLD:-campus}"
HEADLESS="${HEADLESS:-true}"
USE_RVIZ="${USE_RVIZ:-false}"
START_ARIADNE2="${START_ARIADNE2:-true}"
ARIADNE2_ACTIVE="${ARIADNE2_ACTIVE:-false}"
ARIADNE2_USE_RVIZ="${ARIADNE2_USE_RVIZ:-false}"
SPAWN_CAMERA="${SPAWN_CAMERA:-false}"
ARIADNE2_LAYERED_PROJECTION="${ARIADNE2_LAYERED_PROJECTION:-false}"
ARIADNE2_PROJECTION_Z_BELOW="${ARIADNE2_PROJECTION_Z_BELOW:-1.0}"
ARIADNE2_PROJECTION_Z_ABOVE="${ARIADNE2_PROJECTION_Z_ABOVE:-1.6}"
ARIADNE2_PROJECTION_MIN_KNOWN_CELLS="${ARIADNE2_PROJECTION_MIN_KNOWN_CELLS:-80}"
ARIADNE2_PROJECTION_ROBOT_CHECK_RADIUS="${ARIADNE2_PROJECTION_ROBOT_CHECK_RADIUS:-1.5}"
ARIADNE2_PROJECTION_MIN_ROBOT_FREE_CELLS="${ARIADNE2_PROJECTION_MIN_ROBOT_FREE_CELLS:-4}"
ARIADNE2_PROJECTION_OVERLAY_RADIUS="${ARIADNE2_PROJECTION_OVERLAY_RADIUS:-8.0}"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9090}"

if [[ "${HEADLESS}" == "true" ]]; then
  GAZEBO_GUI="false"
else
  GAZEBO_GUI="true"
fi

if [[ "${USE_RVIZ}" == "true" || "${ARIADNE2_USE_RVIZ}" == "true" ]]; then
  export DISPLAY="${DISPLAY:-:0}"
  export XDG_RUNTIME_DIR="${ROSCLAW_XDG_RUNTIME_DIR}"
  export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
fi

echo "[ROSClaw] Launch world=${WORLD} headless=${HEADLESS} cmu_rviz=${USE_RVIZ} ariadne2_rviz=${ARIADNE2_USE_RVIZ} spawn_camera=${SPAWN_CAMERA}"
echo "[ROSClaw] ARiADNE2 local_height_overlay=${ARIADNE2_LAYERED_PROJECTION} z_below=${ARIADNE2_PROJECTION_Z_BELOW} z_above=${ARIADNE2_PROJECTION_Z_ABOVE} min_known_cells=${ARIADNE2_PROJECTION_MIN_KNOWN_CELLS} robot_check_radius=${ARIADNE2_PROJECTION_ROBOT_CHECK_RADIUS} min_robot_free_cells=${ARIADNE2_PROJECTION_MIN_ROBOT_FREE_CELLS} overlay_radius=${ARIADNE2_PROJECTION_OVERLAY_RADIUS}"
echo "[ROSClaw] Display=${DISPLAY:-<unset>} Wayland=${WAYLAND_DISPLAY:-<unset>} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-<unset>}"

roslaunch vehicle_simulator "system_${WORLD}.launch" \
  gazebo_gui:="${GAZEBO_GUI}" \
  launch_rviz:="${USE_RVIZ}" \
  spawn_camera:="${SPAWN_CAMERA}" &
SIM_PID=$!

# Southbound ROS1 access is exposed only through the fixed rosbridge endpoint;
# the host rosclawd owns the WebSocket client and Agents never run rospy.
roslaunch rosbridge_server rosbridge_websocket.launch port:="${ROSBRIDGE_PORT}" &
ROSBRIDGE_PID=$!
roslaunch rosapi rosapi.launch &
ROSAPI_PID=$!

cleanup() {
  kill "${SIM_PID}" 2>/dev/null || true
  if [[ -n "${ARIADNE_PID:-}" ]]; then
    kill "${ARIADNE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ROSBRIDGE_PID:-}" ]]; then
    kill "${ROSBRIDGE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ROSAPI_PID:-}" ]]; then
    kill "${ROSAPI_PID}" 2>/dev/null || true
  fi
  wait || true
}
trap cleanup INT TERM EXIT

if [[ "${START_ARIADNE2}" == "true" ]]; then
  sleep "${ARIADNE2_START_DELAY:-12}"
  roslaunch ariadne2 "ariadne2_${WORLD}.launch" \
    launch_rviz:="${ARIADNE2_USE_RVIZ}" \
    start_active:="${ARIADNE2_ACTIVE}" \
    layered_projection:="${ARIADNE2_LAYERED_PROJECTION}" \
    projection_z_below:="${ARIADNE2_PROJECTION_Z_BELOW}" \
    projection_z_above:="${ARIADNE2_PROJECTION_Z_ABOVE}" \
    projection_min_known_cells:="${ARIADNE2_PROJECTION_MIN_KNOWN_CELLS}" \
    projection_robot_check_radius:="${ARIADNE2_PROJECTION_ROBOT_CHECK_RADIUS}" \
    projection_min_robot_free_cells:="${ARIADNE2_PROJECTION_MIN_ROBOT_FREE_CELLS}" \
    projection_overlay_radius:="${ARIADNE2_PROJECTION_OVERLAY_RADIUS}" &
  ARIADNE_PID=$!
fi

wait
