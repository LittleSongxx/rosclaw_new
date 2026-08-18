# CMU ARE simulation on ROSClaw

This branch keeps the ROS1 Noetic, Gazebo, CMU ARE, local-planner, terrain and
ARiADNE2 sources as a simulation-only Robot Pack. The host `rosclawd` process
owns the rosbridge WebSocket; an Agent or CLI never imports `rospy` and never
publishes an arbitrary ROS topic.

## Contract checks

From the repository root:

```bash
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are check \
  --contract-only --json
```

The contract check validates the embodiment safety card, registered places,
Robot Pack identity, and the out-of-band asset manifest. It can pass while
`launchable` is false when large assets are absent.

## Launch with audited local assets

The checkpoint, planner path files and Gazebo mesh bundle stay outside Git.
After placing them under an audited root matching the manifest, run:

```bash
export CMU_ARE_ASSET_ROOT=/path/to/audited/third_party
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are check --json
PYTHONPATH=src python -m rosclaw.entrypoint sim cmu-are launch --json
```

`launch` refuses to invoke Docker until the required files and hashes pass.
The Compose stack mounts those assets read-only at the ROS package paths used by
ARiADNE2, the local planner and Gazebo. No automatic download is performed.

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

The old `rosclaw app cmu-*` commands, natural-language parser, dashboard and
Nav2 adapter are intentionally not part of this branch's public entrypoint.
