# ROSClaw Mobility App

This app turns ROSClaw from a runtime base into a runnable mobile-robot
application demo.

## Mock Natural-Language Movement

```bash
PYTHONPATH=src python -m rosclaw.cli app move "前进 1 米"
PYTHONPATH=src python -m rosclaw.cli app move "move to x=2.0" --json
```

The command runs:

1. deterministic instruction parsing
2. ROSClaw Runtime startup
3. EventBus command and response events
4. mock mobile-base PID execution
5. Practice/EpisodeRecorder artifact creation
6. app artifacts under `practice_data/app_runs/<episode_id>/`

The app writes `summary.json`, `trajectory.json`, `trajectory.csv`, and
`events.jsonl`.

## Mock Behavior Tree Patrol

```bash
PYTHONPATH=src python -m rosclaw.cli app patrol "去 A 点巡检，再去 B 点，最后返回起点"
```

The command generates a BehaviorTree.CPP-compatible XML tree:

```xml
<root BTCPP_format="4">
  <BehaviorTree ID="ROSClawPatrol">
    <Sequence name="patrol_sequence">
      <Navigate target="A" ... />
      <Inspect target="A" />
      <Navigate target="B" ... />
      <Inspect target="B" />
      <Navigate target="home" ... />
    </Sequence>
  </BehaviorTree>
</root>
```

The Python runner executes the same task structure in a mock 2D navigation
simulation and writes `bt.xml`, `bt.json`, `timeline.jsonl`, `trajectory.json`,
and `summary.json`.

## Docker: ROS1 Noetic + CMU ARE + ARiADNE2

The real simulation route now uses ROS1 Noetic, Gazebo 11, CMU Autonomous
Exploration Development Environment, and the vendored ARiADNE2 planner.  The
main compose file starts a long-running container managed by Docker Desktop.

Build and start the stack:

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml up -d --build rosclaw seekdb
```

Verify the ROS1 bridge and graph:

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-check
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rostopic list
```

Send natural-language waypoint commands into the running CMU ARE simulation:

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-go "向上走 3 米"
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-go "去 inspection_a"
```

The bridge publishes `/way_point`, `/speed`, and `/stop`, observes
`/state_estimation`, `/cmd_vel`, and `/path`, then writes `summary.json`,
`odom_trace.jsonl`, `cmd_vel.jsonl`, `path_trace.jsonl`, `waypoints.jsonl`, and
`ros_topics.txt` under `practice_data/app_runs/<episode_id>/`.

Control ARiADNE2 exploration without restarting the planner:

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore start
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore pause
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore resume
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore stop
```

`pause` publishes `/stop=1` and `/speed=0`, and the patched vendored ARiADNE2
node stops issuing new waypoints while keeping its in-memory graph and policy
state.  `resume` reactivates the same process.

For the recommended two-terminal RViz demo, keep the simulator running in
Terminal A:

```bash
./scripts/start_cmu_rviz.sh
```

Then send commands from Terminal B into the same running container:

```bash
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-go "向上走 3 米"
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore pause
docker compose -f docker/ros1/docker-compose.ros1-are.yml exec rosclaw rosclaw app cmu-explore resume
```

`HEADLESS=true` keeps Gazebo headless while the CMU RViz is visible.
`ARIADNE2_USE_RVIZ=false` is the default, so ARiADNE2 does not open a second
RViz window. Use
`HEADLESS=false ./scripts/start_cmu_rviz.sh` only when the Gazebo GUI window is
also needed.
ROS1 EOL warning dialogs are disabled by default with
`DISABLE_ROS1_EOL_WARNINGS=1`.

For LLM-only interactive control, export a local API key and use:

```bash
cp .env.example .env
# Fill DEEPSEEK_API_KEY in .env, then:
./scripts/cmu_chat.sh
```

Inside the prompt, type natural commands such as `向上走20米`, `右上走5米`,
`前进3米`, `先向上走5米，再向右走3米`, `以半径2米转一圈`, `开始探索`,
or `暂停`. The older `cmu-go` command remains deterministic and does not
require an LLM key.

`cmu-chat` is bounded but broader now: it supports named-place navigation,
explicit map coordinates, screen-direction and robot-frame relative moves,
multi-step waypoint tasks, circle waypoint trajectories, and ARiADNE2
exploration control (`start/pause/resume/stop`). It still does not directly
take over `/cmd_vel` or execute raw speed curves, so it does not fight CMU's
`pathFollower`.

The console reports task phases instead of a generic "任务完成": start,
progress, and success/failure/cancelled. A new movement command preempts the
currently running movement task. If manual movement is requested during
exploration, ROSClaw pauses ARiADNE2 first and waits for an explicit resume.

For a read-only Web task console:

```bash
./scripts/cmu_dashboard.sh
```

Open `http://localhost:18770` to inspect CMU task artifacts, status, timeline
events, final error, and 2D trajectory replay. The dashboard reads
`practice_data/app_runs` and does not send robot control commands.

`start` and `resume` republish `/speed=${CMU_SPEED}`. This prevents a previous
`pause` from leaving the latched `/speed=0` command active.

ARiADNE2 uses `ARIADNE2_LAYERED_PROJECTION=false` by default for the campus demo,
which restores the original full-height 2D projection. The global current-height
projection is not safe as a default because it can remove far-field multi-level
structure and make waypoints jump to falsely connected regions. If enabled for
experiments, ROSClaw now keeps the full-height projection as the base map and
only overlays known current-height cells near the robot within
`ARIADNE2_PROJECTION_OVERLAY_RADIUS`.

Camera simulation is optional and off by default. Keep `SPAWN_CAMERA=false` for
the standard RViz demo; the default RViz layout does not include an Image
display.

Stop the simulator and RViz from another terminal:

```bash
./scripts/stop_cmu_rviz.sh
```
