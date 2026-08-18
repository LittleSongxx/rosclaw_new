# CMU ARE simulation on ROSClaw

This branch keeps the ROS1 Noetic, Gazebo, CMU ARE, local-planner, terrain and
ARiADNE2 sources as a simulation-only Robot Pack. The host `rosclawd` process
owns the rosbridge WebSocket; an Agent or CLI never imports `rospy` and never
publishes an arbitrary ROS topic.

## Validated local layout

The current validated setup is split deliberately between the checkout, the
host Python environment and a ROS1 Docker container:

| Part | Location / value |
| --- | --- |
| ROSClaw checkout | `/home/song/code/Agent/robot-claw/rosclaw_new` (`feature/cmu-are-sim`) |
| Large ARE asset root | `/home/song/code_docker/ros_ws/ARE` (read-only bind mounts) |
| Host Python | Conda environment `rosclaw` |
| ROS1 base image | `my_ros_noetic:v3` |
| Derived simulation image | `rosclaw/noetic-are:latest` |
| Compose container | `rosclaw-cmu-are-sim` |
| rosbridge endpoint | `ws://127.0.0.1:19090` |

The existing `ros_noetic` container is the source of the `my_ros_noetic:v3`
base image. `rosclaw sim cmu-are launch` builds/starts the branch-specific
derived container so that the migrated workspace and launch scripts are
isolated from any legacy container state.

Use the host Conda interpreter for every ROSClaw command; do not run the
Python control plane inside the ROS1 container:

```bash
cd /home/song/code/Agent/robot-claw/rosclaw_new
conda activate rosclaw
export CMU_ARE_ASSET_ROOT=/home/song/code_docker/ros_ws/ARE
export CMU_ARE_ROS_BASE_IMAGE=my_ros_noetic:v3
export ROSBRIDGE_PORT=19090
export ROSCLAW_CMU_ARE_ROSBRIDGE_URL=ws://127.0.0.1:19090
```

## Contract checks

From the repository root, using the `rosclaw` Conda environment:

```bash
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are check \
  --contract-only --json
```

The contract check validates the embodiment safety card, registered places,
Robot Pack identity, and the out-of-band asset manifest. It can pass while
`launchable` is false when large assets are absent.

The full validated preflight uses the external ARE workspace and the local
rosbridge endpoint:

```bash
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are check \
  --asset-root /home/song/code_docker/ros_ws/ARE \
  --rosbridge-url ws://127.0.0.1:19090 \
  --timeout 8 --json
```

## Launch with audited local assets

The checkpoint, planner path files and Gazebo mesh bundle stay outside Git.
After placing them under an audited root matching the manifest, run:

```bash
export CMU_ARE_ASSET_ROOT=/home/song/code_docker/ros_ws/ARE
export CMU_ARE_ROS_BASE_IMAGE=my_ros_noetic:v3
export ROSBRIDGE_PORT=19090
export ROSCLAW_CMU_ARE_ROSBRIDGE_URL=ws://127.0.0.1:19090
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are check \
  --asset-root "$CMU_ARE_ASSET_ROOT" \
  --rosbridge-url "$ROSCLAW_CMU_ARE_ROSBRIDGE_URL" --timeout 8 --json
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are launch \
  --asset-root "$CMU_ARE_ASSET_ROOT" \
  --rosbridge-url "$ROSCLAW_CMU_ARE_ROSBRIDGE_URL" --json
```

`launch` refuses to invoke Docker until the required files and hashes pass.
The Compose stack mounts those assets read-only at the ROS package paths used by
ARiADNE2, the local planner and Gazebo. No automatic download is performed.
The launcher also translates the external ARE workspace layout into the
container mount paths; invoking Compose directly without those generated
per-file source variables can create empty placeholder bind mounts, so use the
CLI launch command for the audited setup.

To inspect the ROS1 graph after startup:

```bash
docker ps --filter name=rosclaw-cmu-are-sim
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are check \
  --asset-root /home/song/code_docker/ros_ws/ARE \
  --rosbridge-url ws://127.0.0.1:19090 --timeout 8 --json
```

The check requires `/gazebo`, `/vehicleSimulator`, `/localPlanner` and
`/pathFollower`, plus the CMU ARE command/observation topic edges. It is safe
to run without a daemon; action commands require the host `rosclawd` socket.

## Structured SHADOW actions

The public simulation namespace is deliberately separate from capability Apps:

```bash
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are navigate \
  --place inspection_a --json
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are navigate \
  --x 3.0 --y 2.0 --z 0.0 --json
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are explore start --json
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are explore pause --json
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are explore resume --json
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are explore stop --json
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are stop --json
```

These commands submit `cmu_are.navigate_to_waypoint`,
`cmu_are.exploration_control` and `cmu_are.stop` to the daemon-owned SHADOW
executor. Navigation is task-verified only after fresh odometry reaches the
bounded map-frame target and zero velocity is observed. Exploration requires
an ARiADNE2 lifecycle-state change. Stop requires a zero-velocity observation.

Each action writes the legacy trace files and a signed-runtime canonical
receipt under `ROSCLAW_HOME/practice_data/app_runs/<action_id>/`, plus
`cmu_are_manifest.json` with controlled paths and SHA-256 artifact digests.

Only `SHADOW` is registered for this pack. A successful navigation means that
fresh simulated odometry reached the bounded target and a zero-velocity stop
was observed; it is simulation evidence, not hardware or `REAL` support.

The old `rosclaw app cmu-*` commands, natural-language parser, dashboard and
Nav2 adapter are intentionally not part of this branch's public entrypoint.

## Asset inventory

The exact file sizes, SHA-256 values, source-relative paths and mount targets
are recorded in [`docs/assets/cmu-are-assets.yaml`](assets/cmu-are-assets.yaml).
The current external workspace contains the checkpoint, planner path files and
the vehicle-simulator mesh directory. Those large files are intentionally not
committed to `rosclaw_new`.

## Evidence boundary

With the external assets present and the Docker smoke path passing, this branch
is eligible for `H2_SIMULATION_VERIFIED` candidate evidence. Without those
assets, contract/fixture checks can still run but Docker launch is blocked. The
branch never declares a REAL executor, hardware support or production
privilege separation.
