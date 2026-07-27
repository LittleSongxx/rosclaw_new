"""CLI surface for the imperative ROSClaw demo apps.

The upstream ``rosclaw app`` namespace is a declarative Capability App system
(``app install`` / ``app run`` over YAML manifests, gated by ``rosclawd``).
The commands in this module are the imperative, demo-grade counterparts that
drive ROS1 CMU ARE and ROS2 Nav2 directly. They register into the *same*
``rosclaw app`` subparser via the upstream ``app_handler`` convention, so both
families coexist without touching the declarative code paths.

Command names here are deliberately prefixed (``cmu-*``, ``nav2-*``) or chosen
to avoid collision with the upstream verbs (``install``, ``list``, ``init``,
``add``, ``validate``, ``run``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def cmd_app_move(args: argparse.Namespace) -> int:
    """Run natural-language movement on the mock mobile base."""
    from rosclaw.apps.mobility import InstructionParseError, run_language_move

    try:
        result = run_language_move(
            args.instruction,
            robot_id=args.robot_id,
            output_dir=args.output_dir,
            kp=args.kp,
            ki=args.ki,
            kd=args.kd,
            tolerance_m=args.tolerance,
        )
    except InstructionParseError as exc:
        print(f"[ROSClaw] Could not parse movement instruction: {exc}")
        return 2
    except Exception as exc:
        print(f"[ROSClaw] App move failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(include_trajectory=False), indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print("ROSClaw App — Natural Language Move")
        print("=" * 60)
        print(f"Instruction: {result.instruction}")
        print(f"Robot:       {result.robot_id}")
        print(f"Episode:     {result.episode_id}")
        print(f"Intent:      {result.intent.mode} target_x={result.intent.target_x:.3f}m")
        print(f"Status:      {result.status}")
        print(f"Steps:       {result.steps}")
        print(f"Final x:     {result.final_x:.4f}m")
        print(f"Final error: {result.final_error:.4f}m")
        print(f"Artifacts:   {result.artifact_dir}")
        print(f"Practice:    {result.practice_artifact_dir}")
        print("=" * 60)
    return 0 if result.ok else 1


def cmd_app_patrol(args: argparse.Namespace) -> int:
    """Generate and execute a template behavior-tree patrol app."""
    from rosclaw.apps.behavior_tree import run_patrol_behavior_tree

    try:
        result = run_patrol_behavior_tree(
            args.instruction,
            robot_id=args.robot_id,
            output_dir=args.output_dir,
            tolerance_m=args.tolerance,
        )
    except Exception as exc:
        print(f"[ROSClaw] App patrol failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(include_traces=False), indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print("ROSClaw App — Behavior Tree Patrol")
        print("=" * 60)
        print(f"Instruction: {result.instruction}")
        print(f"Robot:       {result.robot_id}")
        print(f"Episode:     {result.episode_id}")
        print(f"Status:      {result.status}")
        print(f"Waypoints:   {', '.join(w.name for w in result.plan.waypoints)}")
        print(f"Return home: {'yes' if result.plan.return_home else 'no'}")
        print(f"BT XML:      {Path(result.artifact_dir) / 'bt.xml'}")
        print(f"Timeline:    {Path(result.artifact_dir) / 'timeline.jsonl'}")
        print(f"Trajectory:  {Path(result.artifact_dir) / 'trajectory.json'}")
        print(f"Practice:    {result.practice_artifact_dir}")
        print("=" * 60)
    return 0 if result.ok else 1


def cmd_app_nav2_check(_args: argparse.Namespace) -> int:
    """Check whether ROS2/Nav2 Python packages are importable."""
    from rosclaw.apps.nav2_bridge import check_nav2_available

    ok, reason = check_nav2_available()
    print("=" * 60)
    print("ROSClaw App — Nav2 Bridge Check")
    print("=" * 60)
    print(f"Available: {'yes' if ok else 'no'}")
    print(f"Detail:    {reason}")
    print("=" * 60)
    return 0 if ok else 1


def cmd_app_nav2_go(args: argparse.Namespace) -> int:
    """Send one real NavigateToPose goal through the optional Nav2 bridge."""
    from rosclaw.apps.nav2_bridge import (
        Nav2Bridge,
        Nav2UnavailableError,
        load_places_yaml,
        write_nav2_artifact,
    )

    try:
        places = load_places_yaml(args.places)
    except Exception as exc:
        print(f"[ROSClaw] Could not load places file: {exc}")
        return 1

    place = places.get(args.place)
    if place is None:
        print(f"[ROSClaw] Unknown place '{args.place}'. Known: {', '.join(sorted(places))}")
        return 1

    feedback_events = []

    def on_feedback(event):
        feedback_events.append(event)
        if args.verbose:
            print(f"[Nav2] feedback: {event}")

    try:
        bridge = Nav2Bridge(feedback_callback=on_feedback)
        try:
            result = bridge.navigate_to_place(place, timeout_sec=args.timeout)
        finally:
            bridge.close()
    except Nav2UnavailableError as exc:
        print(f"[ROSClaw] Nav2 bridge unavailable: {exc}")
        return 2
    except Exception as exc:
        print(f"[ROSClaw] Nav2 navigation failed: {exc}")
        return 1

    artifact_dir = write_nav2_artifact(place=place, result=result, output_dir=args.output_dir)

    print("=" * 60)
    print("ROSClaw App — Nav2 NavigateToPose")
    print("=" * 60)
    print(f"Place:      {place.name}")
    print(f"Target:     x={place.x:.3f}, y={place.y:.3f}, theta={place.theta:.3f}")
    print(f"Status:     {result.get('status')}")
    print(f"Feedback:   {len(feedback_events)} event(s)")
    print(f"Artifacts:  {artifact_dir}")
    print("=" * 60)
    return 0 if result.get("status") == "success" else 1


def cmd_app_nav2_demo(args: argparse.Namespace) -> int:
    """Run the full ROSClaw -> Nav2 TurtleBot simulation demo."""
    from rosclaw.apps.nav2_bridge import (
        Nav2Bridge,
        Nav2UnavailableError,
        PlacePose,
        launch_humble_nav2_simulation,
        load_places_yaml,
        make_nav2_artifact_dir,
        resolve_place_query,
        stop_process_group,
        write_nav2_artifact,
    )

    try:
        places = load_places_yaml(args.places)
        place = resolve_place_query(args.instruction, places)
        initial = places.get(args.initial_place) or PlacePose(
            name=args.initial_place,
            x=0.0,
            y=0.0,
            theta=0.0,
            frame_id=place.frame_id,
        )
        spawn_place = args.spawn_place or args.initial_place
        spawn = places.get(spawn_place) or initial
        if args.spawn_x is not None or args.spawn_y is not None or args.spawn_theta is not None:
            spawn = PlacePose(
                name=spawn.name,
                x=args.spawn_x if args.spawn_x is not None else spawn.x,
                y=args.spawn_y if args.spawn_y is not None else spawn.y,
                theta=args.spawn_theta if args.spawn_theta is not None else spawn.theta,
                frame_id=spawn.frame_id,
                aliases=spawn.aliases,
            )
    except Exception as exc:
        print(f"[ROSClaw] Could not resolve Nav2 demo target: {exc}")
        return 1

    artifact_dir = make_nav2_artifact_dir(
        place=place,
        output_dir=args.output_dir,
        app="nav2_demo",
    )
    launch_log_path = artifact_dir / "nav2_launch.log"
    launch_process = None
    if not args.no_launch:
        print("[ROSClaw] Starting Humble TurtleBot3/Nav2 simulation...")
        print(
            "[ROSClaw] TurtleBot spawn pose: "
            f"{spawn.name} x={spawn.x:.2f}, y={spawn.y:.2f}, theta={spawn.theta:.2f}"
        )
        launch_process = launch_humble_nav2_simulation(
            headless=args.headless,
            use_rviz=args.use_rviz,
            turtlebot_model=args.turtlebot_model,
            log_path=launch_log_path,
            spawn_pose=spawn,
            gazebo_master_uri=args.gazebo_master_uri,
        )
        print(f"[ROSClaw] Launch log: {launch_log_path}")

    feedback_events = []

    def on_feedback(event):
        feedback_events.append(event)
        if args.verbose:
            print(f"[Nav2] feedback: {event}")

    result = None
    topics = []
    readiness = {}
    localization = {}
    try:
        bridge = Nav2Bridge(feedback_callback=on_feedback)
        try:
            print(f"[ROSClaw] Waiting for Nav2 action server ({args.timeout:.0f}s timeout)...")
            if not bridge.wait_for_nav2(timeout_sec=args.timeout):
                raise TimeoutError("Nav2 NavigateToPose action server is not available")
            readiness_timeout = args.readiness_timeout or args.timeout
            print(f"[ROSClaw] Waiting for Gazebo robot readiness ({readiness_timeout:.0f}s timeout)...")
            readiness = bridge.wait_for_simulation_ready(timeout_sec=readiness_timeout)
            if not readiness.get("ready"):
                missing = ", ".join(readiness.get("missing_topics", [])) or "none"
                raise TimeoutError(
                    "simulation is not ready "
                    f"(missing_topics={missing}, "
                    f"odom_samples={readiness.get('odom_samples')}, "
                    f"tf_odom_base_link={readiness.get('tf_odom_base_link')})"
                )
            print(
                "[ROSClaw] Publishing initial pose: "
                f"{initial.name} x={initial.x:.2f}, y={initial.y:.2f}, theta={initial.theta:.2f}"
            )
            bridge.set_initial_pose(initial)
            localization_timeout = args.localization_timeout
            print(f"[ROSClaw] Waiting for localization ({localization_timeout:.0f}s timeout)...")
            localization = bridge.wait_for_localization(
                initial_pose=initial,
                timeout_sec=localization_timeout,
            )
            if not localization.get("ready"):
                raise TimeoutError(
                    "localization is not ready "
                    f"(new_amcl_pose_samples={localization.get('new_amcl_pose_samples')}, "
                    f"tf_map_base_link={localization.get('tf_map_base_link')})"
                )
            topics = bridge.list_topics()
            print(
                "[ROSClaw] Sending NavigateToPose: "
                f"{place.name} x={place.x:.2f}, y={place.y:.2f}, theta={place.theta:.2f}"
            )
            result = bridge.navigate_to_place(place, timeout_sec=args.timeout)
            result["readiness"] = readiness
            result["localization"] = localization
            topics = bridge.list_topics()
        finally:
            bridge.close()
    except Nav2UnavailableError as exc:
        print(f"[ROSClaw] Nav2 bridge unavailable: {exc}")
        return 2
    except Exception as exc:
        result = {
            "status": "error",
            "place": place.name,
            "error": str(exc),
            "feedback": feedback_events,
            "pose_trace": [],
            "readiness": readiness,
            "localization": localization,
        }
        print(f"[ROSClaw] Nav2 demo failed: {exc}")
    finally:
        if launch_process is not None and args.stop_launch:
            stop_process_group(launch_process)

    artifact_dir = write_nav2_artifact(
        place=place,
        result=result or {"status": "error", "feedback": feedback_events},
        output_dir=args.output_dir,
        artifact_dir=artifact_dir,
        app="nav2-demo",
        instruction=args.instruction,
        ros2_topics=topics,
        launch_log_path=launch_log_path if not args.no_launch else None,
    )

    print("=" * 60)
    print("ROSClaw App — Humble Nav2 TurtleBot Demo")
    print("=" * 60)
    print(f"Instruction: {args.instruction}")
    print(f"Target:      {place.name} ({place.x:.3f}, {place.y:.3f}, {place.theta:.3f})")
    print(f"Status:      {(result or {}).get('status')}")
    if readiness:
        print(f"Sim ready:   {readiness.get('ready')} ({readiness.get('elapsed_sec')}s)")
    if localization:
        print(f"Localized:   {localization.get('ready')} ({localization.get('elapsed_sec')}s)")
    print(f"Feedback:    {len((result or {}).get('feedback', []))} event(s)")
    print(f"Pose trace:  {len((result or {}).get('pose_trace', []))} sample(s)")
    print(f"Artifacts:   {artifact_dir}")
    print("=" * 60)
    return 0 if (result or {}).get("status") == "success" else 1


def cmd_app_cmu_check(_args: argparse.Namespace) -> int:
    """Check whether ROS1 CMU ARE packages are available."""
    from rosclaw.apps.cmu_are_bridge import check_cmu_are_available

    ok, reason = check_cmu_are_available()
    print("=" * 60)
    print("ROSClaw App — ROS1 CMU ARE Check")
    print("=" * 60)
    print(f"Available: {'yes' if ok else 'no'}")
    print(f"Detail:    {reason}")
    print("=" * 60)
    return 0 if ok else 1


def cmd_app_cmu_launch(args: argparse.Namespace) -> int:
    """Launch CMU ARE and optionally ARiADNE2."""
    from rosclaw.apps.cmu_are_bridge import launch_cmu_are_simulation, stop_process_group

    print("=" * 60)
    print("ROSClaw App — Launch ROS1 CMU ARE")
    print("=" * 60)
    print(f"World:       {args.world}")
    print(f"Headless:    {args.headless}")
    print(f"RViz:        {args.use_rviz}")
    print(f"ARiADNE2:    {args.ariadne2}")
    print(f"ARiADNE2 RViz:{args.ariadne2_rviz}")
    print(f"Start active:{args.ariadne2_active}")
    print("=" * 60)

    process = launch_cmu_are_simulation(
        world=args.world,
        headless=args.headless,
        use_rviz=args.use_rviz,
        log_path=args.log_path,
        include_ariadne2=args.ariadne2,
        ariadne2_active=args.ariadne2_active,
        ariadne2_use_rviz=args.ariadne2_rviz,
    )
    try:
        return process.wait()
    except KeyboardInterrupt:
        print("\n[ROSClaw] Stopping CMU ARE launch...")
        stop_process_group(process)
        return 0


def cmd_app_cmu_go(args: argparse.Namespace) -> int:
    """Send a waypoint to a running CMU ARE local planner."""
    from rosclaw.apps.cmu_are_bridge import (
        CmuAreParseError,
        CmuAreUnavailableError,
        run_cmu_go,
    )

    try:
        result = run_cmu_go(
            args.instruction,
            places_path=args.places,
            output_dir=args.output_dir,
            timeout_sec=args.timeout,
            readiness_timeout_sec=args.readiness_timeout,
            tolerance_m=args.tolerance,
            speed=args.speed,
            use_llm=args.use_llm,
        )
    except CmuAreParseError as exc:
        print(f"[ROSClaw] Could not parse CMU ARE instruction: {exc}")
        return 2
    except CmuAreUnavailableError as exc:
        print(f"[ROSClaw] CMU ARE bridge unavailable: {exc}")
        return 2
    except Exception as exc:
        print(f"[ROSClaw] CMU ARE navigation failed: {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print("ROSClaw App — CMU ARE Target Navigation")
        print("=" * 60)
        print(f"Instruction: {result.instruction}")
        print(f"Intent:      {result.intent.type}")
        if result.intent.x is not None and result.intent.y is not None:
            print(f"Target:      x={result.intent.x:.3f}, y={result.intent.y:.3f}")
        print(f"Status:      {result.status}")
        if result.distance_to_goal is not None:
            print(f"Goal error:  {result.distance_to_goal:.3f}m")
        print(f"Odom:        {result.odom_trace_count} sample(s)")
        print(f"Cmd_vel:     {result.cmd_vel_count} sample(s)")
        print(f"Artifacts:   {result.artifact_dir}")
        print("=" * 60)
    return 0 if result.ok else 1


def cmd_app_cmu_explore(args: argparse.Namespace) -> int:
    """Control a running ARiADNE2 planner without killing its process."""
    from rosclaw.apps.cmu_are_bridge import CmuAreBridge, CmuAreUnavailableError

    explore_command = args.explore_command
    try:
        bridge = CmuAreBridge()
        bridge.publish_exploration_control(explore_command, speed=args.speed)
    except CmuAreUnavailableError as exc:
        print(f"[ROSClaw] CMU ARE bridge unavailable: {exc}")
        return 2
    except Exception as exc:
        print(f"[ROSClaw] Exploration control failed: {exc}")
        return 1

    print("=" * 60)
    print("ROSClaw App — ARiADNE2 Exploration Control")
    print("=" * 60)
    print(f"Command: {explore_command}")
    if explore_command == "pause":
        print("Effect:  /stop=1, /speed=0, ARiADNE2 keeps in-memory planner state")
    elif explore_command == "resume":
        print(f"Effect:  /stop=0, /speed={args.speed}, same ARiADNE2 process resumes planning")
    elif explore_command == "start":
        print(f"Effect:  /stop=0, /speed={args.speed}, ARiADNE2 starts publishing exploration waypoints")
    elif explore_command == "stop":
        print("Effect:  /stop=1, /speed=0, ARiADNE2 stops publishing waypoints")
    print("=" * 60)
    return 0


def cmd_app_cmu_demo(args: argparse.Namespace) -> int:
    """Launch CMU ARE, execute one language waypoint command, and stop."""
    from rosclaw.apps.cmu_are_bridge import CmuAreParseError, CmuAreUnavailableError, run_cmu_demo

    try:
        result = run_cmu_demo(
            args.instruction,
            places_path=args.places,
            output_dir=args.output_dir,
            world=args.world,
            headless=args.headless,
            use_rviz=args.use_rviz,
            timeout_sec=args.timeout,
            readiness_timeout_sec=args.readiness_timeout,
            tolerance_m=args.tolerance,
            speed=args.speed,
            stop_launch=args.stop_launch,
        )
    except CmuAreParseError as exc:
        print(f"[ROSClaw] Could not parse CMU ARE demo instruction: {exc}")
        return 2
    except CmuAreUnavailableError as exc:
        print(f"[ROSClaw] CMU ARE demo unavailable: {exc}")
        return 2
    except Exception as exc:
        print(f"[ROSClaw] CMU ARE demo failed: {exc}")
        return 1

    print("=" * 60)
    print("ROSClaw App — CMU ARE Demo")
    print("=" * 60)
    print(f"Instruction: {result.instruction}")
    print(f"Status:      {result.status}")
    if result.intent.x is not None and result.intent.y is not None:
        print(f"Target:      x={result.intent.x:.3f}, y={result.intent.y:.3f}")
    if result.distance_to_goal is not None:
        print(f"Goal error:  {result.distance_to_goal:.3f}m")
    print(f"Artifacts:   {result.artifact_dir}")
    print("=" * 60)
    return 0 if result.ok else 1


def cmd_app_cmu_chat(args: argparse.Namespace) -> int:
    """Interactive LLM-only CMU ARE control shell."""
    from rosclaw.apps.cmu_are_bridge import (
        CmuAreBridge,
        CmuAreParseError,
        CmuChatTaskManager,
        CmuAreUnavailableError,
        load_cmu_places,
    )

    if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("[ROSClaw] cmu-chat requires DEEPSEEK_API_KEY or OPENAI_API_KEY.")
        print("[ROSClaw] Example: export DEEPSEEK_API_KEY='...' && ./scripts/cmu_chat.sh")
        return 2

    print("=" * 60)
    print("ROSClaw App — CMU ARE LLM Chat")
    print("=" * 60)
    print("直接输入：向上走20米 / 右上走5米 / 前进3米 / 半径2米转一圈 / 开始探索 / 暂停")
    print("输入 exit、quit、退出 或 Ctrl-D 结束。")
    print("=" * 60)

    try:
        places = load_cmu_places(args.places)
        bridge = CmuAreBridge()
        readiness = bridge.wait_for_topics(timeout_sec=args.readiness_timeout)
        if not readiness["ready"]:
            missing = ", ".join(readiness["missing_topics"])
            print(f"仿真环境还没准备好：{missing}")
            return 2
        manager = CmuChatTaskManager(
            bridge=bridge,
            places=places,
            ros_topics=readiness["seen_topics"],
            output_dir=args.output_dir,
            timeout_sec=args.timeout,
            readiness_timeout_sec=args.readiness_timeout,
            tolerance_m=args.tolerance,
            speed=args.speed,
            max_relative_m=args.max_relative_m,
            progress_interval_sec=args.progress_interval,
            max_sequence_steps=args.max_sequence_steps,
            circle_segments=args.circle_segments,
            max_circle_radius_m=args.max_circle_radius,
            exploration_on_manual=args.exploration_on_manual,
        )
    except CmuAreUnavailableError as exc:
        print(f"仿真环境还没准备好：{exc}")
        return 2
    except Exception as exc:
        print(f"启动交互控制台失败：{exc}")
        return 1

    def print_events() -> bool:
        printed = False
        events = manager.drain_events()
        if events and prompt_visible:
            print()
        for event in events:
            printed = True
            print(event.message)
            result = event.result
            if result is None:
                continue
            if result.intent.x is not None and result.intent.y is not None:
                print(f"目标点：x={result.intent.x:.3f}, y={result.intent.y:.3f}")
            if result.distance_to_goal is not None:
                print(f"最终误差：{result.distance_to_goal:.3f}m")
            print(f"记录：{result.artifact_dir}")
        return printed

    prompt_visible = False
    while True:
        try:
            if print_events():
                prompt_visible = False
            if not prompt_visible:
                sys.stdout.write("rosclaw> ")
                sys.stdout.flush()
                prompt_visible = True
            ready, _, _ = select.select([sys.stdin], [], [], max(0.2, args.progress_interval))
            if not ready:
                continue
            instruction = sys.stdin.readline()
            prompt_visible = False
            if instruction == "":
                print()
                for event in manager.cancel_current(reason="控制台退出，已取消当前任务。"):
                    print(event.message)
                return 0
            instruction = instruction.strip()
        except EOFError:
            print()
            for event in manager.cancel_current(reason="控制台退出，已取消当前任务。"):
                print(event.message)
            return 0
        except KeyboardInterrupt:
            print("\n[ROSClaw] 已退出。")
            for event in manager.cancel_current(reason="控制台退出，已取消当前任务。"):
                print(event.message)
            return 0

        if not instruction:
            continue
        if instruction.lower() in {"exit", "quit", "q"} or instruction in {"退出", "结束"}:
            print("[ROSClaw] 已退出。")
            for event in manager.cancel_current(reason="控制台退出，已取消当前任务。"):
                print(event.message)
            return 0

        try:
            events = manager.submit(instruction)
        except CmuAreParseError as exc:
            print(f"我还不能安全理解这条指令：{exc}")
            continue
        except CmuAreUnavailableError as exc:
            print(f"仿真环境还没准备好：{exc}")
            return 2
        except Exception as exc:
            print(f"执行时出错：{exc}")
            continue

        for event in events:
            print(event.message)


def cmd_app_cmu_dashboard(args: argparse.Namespace) -> int:
    """Start the CMU ARE task dashboard."""
    from rosclaw.apps.cmu_dashboard import run_cmu_dashboard

    print("=" * 60)
    print("ROSClaw App — CMU Web Task Dashboard")
    print("=" * 60)
    print(f"URL:       http://localhost:{args.port}")
    print(f"Artifacts: {args.output_dir}")
    print("Metadata:  dashboard_state.json + .trash/")
    print(f"ROS live:  {'yes' if args.connect_ros else 'no'}")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    try:
        run_cmu_dashboard(
            host=args.host,
            port=args.port,
            output_dir=args.output_dir,
            max_points=args.max_points,
            connect_ros=args.connect_ros,
        )
    except KeyboardInterrupt:
        print("\n[ROSClaw] Dashboard stopped.")
        return 0
    except Exception as exc:
        print(f"[ROSClaw] CMU dashboard failed: {exc}")
        return 1
    return 0


def add_demo_app_subparsers(commands: Any) -> None:
    """Register the imperative demo-app commands on an ``app`` subparser.

    Called after :func:`rosclaw.app.cli.add_app_subparsers` so the declarative
    Capability App verbs keep priority on name collisions.
    """

    move = commands.add_parser("move", help="Natural-language mock mobile-base movement")
    move.add_argument("instruction", help="Movement instruction, e.g. '前进 1 米'")
    move.add_argument("--robot-id", default="mock_mobile_base", help="Robot profile")
    move.add_argument(
        "--output-dir", default="practice_data/app_runs", help="Artifact directory"
    )
    move.add_argument("--kp", type=float, default=2.0, help="PID proportional gain")
    move.add_argument("--ki", type=float, default=0.1, help="PID integral gain")
    move.add_argument("--kd", type=float, default=0.5, help="PID derivative gain")
    move.add_argument(
        "--tolerance", type=float, default=0.05, help="Success tolerance in meters"
    )
    move.add_argument("--json", action="store_true", help="Output machine-readable summary")
    move.set_defaults(app_handler=cmd_app_move)

    patrol = commands.add_parser("patrol", help="Natural-language behavior-tree patrol")
    patrol.add_argument(
        "instruction", help="Patrol instruction, e.g. '去 A 点巡检，再去 B 点，最后返回起点'"
    )
    patrol.add_argument("--robot-id", default="mock_mobile_base", help="Robot profile")
    patrol.add_argument(
        "--output-dir", default="practice_data/app_runs", help="Artifact directory"
    )
    patrol.add_argument(
        "--tolerance", type=float, default=0.05, help="Navigation tolerance in meters"
    )
    patrol.add_argument("--json", action="store_true", help="Output machine-readable summary")
    patrol.set_defaults(app_handler=cmd_app_patrol)

    _add_nav2_subparsers(commands)
    _add_cmu_subparsers(commands)


def _add_nav2_subparsers(commands: Any) -> None:
    """ROS2 / Nav2 demo commands."""

    nav2_check = commands.add_parser(
        "nav2-check", help="Check whether ROS2/Nav2 Python packages are importable"
    )
    nav2_check.set_defaults(app_handler=cmd_app_nav2_check)

    nav2_go = commands.add_parser(
        "nav2-go", help="Send one real NavigateToPose goal through the Nav2 bridge"
    )
    nav2_go.add_argument("place", help="Place name defined in the places YAML file")
    nav2_go.add_argument(
        "--places",
        default=os.environ.get("NAV2_PLACES", "docker/ros1/places.campus.yaml"),
        help="YAML file mapping place names to map coordinates",
    )
    nav2_go.add_argument(
        "--timeout",
        type=float,
        default=_env_float("NAV2_NAV_TIMEOUT", 120.0),
        help="Navigation timeout",
    )
    nav2_go.add_argument(
        "--output-dir",
        default=os.environ.get("NAV2_OUTPUT_DIR", "practice_data/app_runs"),
        help="Artifact directory",
    )
    nav2_go.add_argument("--verbose", action="store_true", help="Print Nav2 feedback events")
    nav2_go.set_defaults(app_handler=cmd_app_nav2_go)

    nav2_demo = commands.add_parser(
        "nav2-demo", help="Run the full ROSClaw -> Nav2 TurtleBot simulation demo"
    )
    nav2_demo.add_argument("instruction", help="Target instruction, e.g. '去 inspection_a'")
    nav2_demo.add_argument(
        "--places",
        default=os.environ.get("NAV2_PLACES", "docker/ros1/places.campus.yaml"),
        help="YAML file mapping place names to map coordinates",
    )
    nav2_demo.add_argument(
        "--initial-place", default="home", help="Place used as the initial AMCL pose"
    )
    nav2_demo.add_argument(
        "--spawn-place", default=None, help="Place used to spawn the robot (defaults to initial)"
    )
    nav2_demo.add_argument("--spawn-x", type=float, default=None, help="Override spawn x")
    nav2_demo.add_argument("--spawn-y", type=float, default=None, help="Override spawn y")
    nav2_demo.add_argument("--spawn-theta", type=float, default=None, help="Override spawn theta")
    nav2_demo.add_argument("--turtlebot-model", default="waffle", help="TURTLEBOT3_MODEL value")
    nav2_demo.add_argument(
        "--gazebo-master-uri", default=None, help="Override GAZEBO_MASTER_URI"
    )
    nav2_demo.add_argument(
        "--headless", action="store_true", default=True, help="Run Gazebo headless"
    )
    nav2_demo.add_argument(
        "--gui", dest="headless", action="store_false", help="Run Gazebo with GUI"
    )
    nav2_demo.add_argument("--use-rviz", action="store_true", help="Launch RViz")
    nav2_demo.add_argument(
        "--no-launch",
        dest="no_launch",
        action="store_true",
        help="Assume a simulation is already running",
    )
    nav2_demo.add_argument(
        "--timeout",
        type=float,
        default=_env_float("NAV2_NAV_TIMEOUT", 180.0),
        help="Navigation timeout",
    )
    nav2_demo.add_argument(
        "--readiness-timeout",
        type=float,
        default=_env_float("NAV2_READINESS_TIMEOUT", 120.0),
        help="Nav2 stack readiness timeout",
    )
    nav2_demo.add_argument(
        "--localization-timeout",
        type=float,
        default=_env_float("NAV2_LOCALIZATION_TIMEOUT", 60.0),
        help="AMCL localization timeout",
    )
    nav2_demo.add_argument(
        "--output-dir",
        default=os.environ.get("NAV2_OUTPUT_DIR", "practice_data/app_runs"),
        help="Artifact directory",
    )
    nav2_demo.add_argument(
        "--no-stop-launch",
        dest="stop_launch",
        action="store_false",
        help="Leave launched simulation running",
    )
    nav2_demo.add_argument("--verbose", action="store_true", help="Print Nav2 feedback events")
    nav2_demo.add_argument("--json", action="store_true", help="Output machine-readable summary")
    nav2_demo.set_defaults(app_handler=cmd_app_nav2_demo, stop_launch=True)


def _add_cmu_subparsers(commands: Any) -> None:
    """ROS1 CMU ARE / ARiADNE2 demo commands."""

    default_places = os.environ.get("CMU_PLACES", "docker/ros1/places.campus.yaml")
    default_output = os.environ.get("CMU_OUTPUT_DIR", "practice_data/app_runs")

    cmu_check = commands.add_parser(
        "cmu-check", help="Check ROS1 CMU ARE bridge availability"
    )
    cmu_check.set_defaults(app_handler=cmd_app_cmu_check)

    cmu_launch = commands.add_parser("cmu-launch", help="Launch ROS1 CMU ARE simulation")
    cmu_launch.add_argument("--world", default="campus", help="CMU ARE world name")
    cmu_launch.add_argument(
        "--headless", action="store_true", default=True, help="Run Gazebo headless"
    )
    cmu_launch.add_argument(
        "--gui", dest="headless", action="store_false", help="Run Gazebo with GUI"
    )
    cmu_launch.add_argument("--use-rviz", action="store_true", help="Launch RViz")
    cmu_launch.add_argument(
        "--ariadne2", action="store_true", help="Also launch ARiADNE2 planner"
    )
    cmu_launch.add_argument(
        "--ariadne2-active",
        action="store_true",
        help="Start ARiADNE2 in active exploration mode",
    )
    cmu_launch.add_argument(
        "--ariadne2-rviz", action="store_true", help="Launch ARiADNE2's own RViz window"
    )
    cmu_launch.add_argument("--log-path", default=None, help="Optional launch log path")
    cmu_launch.set_defaults(app_handler=cmd_app_cmu_launch)

    cmu_go = commands.add_parser(
        "cmu-go", help="Send a natural-language waypoint to CMU ARE"
    )
    cmu_go.add_argument(
        "instruction", help="Target instruction, e.g. '去 inspection_a' or '向上走 3 米'"
    )
    cmu_go.add_argument(
        "--places",
        default=default_places,
        help="YAML file mapping place names to CMU ARE map coordinates",
    )
    cmu_go.add_argument(
        "--timeout",
        type=float,
        default=_env_float("CMU_NAV_TIMEOUT", 120.0),
        help="Navigation timeout",
    )
    cmu_go.add_argument(
        "--readiness-timeout",
        type=float,
        default=_env_float("CMU_READINESS_TIMEOUT", 60.0),
        help="CMU ARE topic readiness timeout",
    )
    cmu_go.add_argument(
        "--tolerance",
        type=float,
        default=_env_float("CMU_NAV_TOLERANCE", 1.5),
        help="Success tolerance in meters",
    )
    cmu_go.add_argument(
        "--speed", type=float, default=_env_float("CMU_SPEED", 2.0), help="Published /speed value"
    )
    cmu_go.add_argument(
        "--use-llm", action="store_true", help="Enable constrained LLM fallback parser"
    )
    cmu_go.add_argument("--output-dir", default=default_output, help="Artifact directory")
    cmu_go.add_argument("--json", action="store_true", help="Output machine-readable summary")
    cmu_go.set_defaults(app_handler=cmd_app_cmu_go)

    cmu_explore = commands.add_parser("cmu-explore", help="Control ARiADNE2 exploration")
    cmu_explore.add_argument(
        "explore_command",
        choices=["start", "pause", "resume", "stop"],
        help="Exploration control command",
    )
    cmu_explore.add_argument(
        "--speed",
        type=float,
        default=_env_float("CMU_SPEED", 2.0),
        help="Published /speed value for start/resume",
    )
    cmu_explore.set_defaults(app_handler=cmd_app_cmu_explore)

    cmu_chat = commands.add_parser("cmu-chat", help="Interactive LLM control for CMU ARE")
    cmu_chat.add_argument(
        "--places",
        default=default_places,
        help="YAML file mapping place names to CMU ARE map coordinates",
    )
    cmu_chat.add_argument(
        "--timeout",
        type=float,
        default=_env_float("CMU_NAV_TIMEOUT", 120.0),
        help="Navigation timeout",
    )
    cmu_chat.add_argument(
        "--readiness-timeout",
        type=float,
        default=_env_float("CMU_READINESS_TIMEOUT", 60.0),
        help="CMU ARE topic readiness timeout",
    )
    cmu_chat.add_argument(
        "--tolerance",
        type=float,
        default=_env_float("CMU_NAV_TOLERANCE", 1.5),
        help="Success tolerance in meters",
    )
    cmu_chat.add_argument(
        "--speed", type=float, default=_env_float("CMU_SPEED", 2.0), help="Published /speed value"
    )
    cmu_chat.add_argument("--output-dir", default=default_output, help="Artifact directory")
    cmu_chat.add_argument(
        "--max-relative-m",
        type=float,
        default=_env_float("CMU_MAX_RELATIVE_M", 20.0),
        help="Maximum single relative move in meters",
    )
    cmu_chat.add_argument(
        "--progress-interval",
        type=float,
        default=_env_float("CMU_CHAT_PROGRESS_INTERVAL", 3.0),
        help="Seconds between running-task progress messages",
    )
    cmu_chat.add_argument(
        "--max-sequence-steps",
        type=int,
        default=_env_int("CMU_MAX_SEQUENCE_STEPS", 8),
        help="Maximum LLM-generated waypoint steps",
    )
    cmu_chat.add_argument(
        "--circle-segments",
        type=int,
        default=_env_int("CMU_CIRCLE_SEGMENTS", 12),
        help="Maximum waypoint segments for circle tasks",
    )
    cmu_chat.add_argument(
        "--max-circle-radius",
        type=float,
        default=_env_float("CMU_MAX_CIRCLE_RADIUS", 6.0),
        help="Maximum circle task radius in meters",
    )
    cmu_chat.add_argument(
        "--exploration-on-manual",
        default=os.environ.get("CMU_EXPLORATION_ON_MANUAL", "pause"),
        choices=["pause", "keep"],
        help="What to do with ARiADNE2 exploration when a manual navigation task starts",
    )
    cmu_chat.set_defaults(app_handler=cmd_app_cmu_chat)

    cmu_dashboard = commands.add_parser(
        "cmu-dashboard", help="Web dashboard for CMU ARE task artifacts"
    )
    cmu_dashboard.add_argument(
        "--host",
        default=os.environ.get("CMU_DASHBOARD_HOST", "127.0.0.1"),
        help="Dashboard bind host (default loopback; the dashboard has no auth)",
    )
    cmu_dashboard.add_argument(
        "--port",
        type=int,
        default=_env_int("CMU_DASHBOARD_PORT", 18770),
        help="Dashboard HTTP port",
    )
    cmu_dashboard.add_argument("--output-dir", default=default_output, help="Artifact directory")
    cmu_dashboard.add_argument(
        "--max-points",
        type=int,
        default=_env_int("CMU_DASHBOARD_MAX_POINTS", 2000),
        help="Maximum trajectory points sent to browser",
    )
    cmu_dashboard.add_argument(
        "--no-ros",
        dest="connect_ros",
        action="store_false",
        help="Do not attempt live ROS1 topic status",
    )
    cmu_dashboard.set_defaults(app_handler=cmd_app_cmu_dashboard, connect_ros=True)

    cmu_demo = commands.add_parser(
        "cmu-demo", help="Launch CMU ARE and run one waypoint command"
    )
    cmu_demo.add_argument("instruction", help="Target instruction, e.g. '去 inspection_a'")
    cmu_demo.add_argument(
        "--places",
        default=default_places,
        help="YAML file mapping place names to CMU ARE map coordinates",
    )
    cmu_demo.add_argument("--world", default="campus", help="CMU ARE world name")
    cmu_demo.add_argument(
        "--headless", action="store_true", default=True, help="Run Gazebo headless"
    )
    cmu_demo.add_argument(
        "--gui", dest="headless", action="store_false", help="Run Gazebo with GUI"
    )
    cmu_demo.add_argument("--use-rviz", action="store_true", help="Launch RViz")
    cmu_demo.add_argument(
        "--timeout",
        type=float,
        default=_env_float("CMU_NAV_TIMEOUT", 120.0),
        help="Navigation timeout",
    )
    cmu_demo.add_argument(
        "--readiness-timeout",
        type=float,
        default=_env_float("CMU_READINESS_TIMEOUT", 90.0),
        help="CMU ARE topic readiness timeout",
    )
    cmu_demo.add_argument(
        "--tolerance",
        type=float,
        default=_env_float("CMU_NAV_TOLERANCE", 1.5),
        help="Success tolerance in meters",
    )
    cmu_demo.add_argument(
        "--speed", type=float, default=_env_float("CMU_SPEED", 2.0), help="Published /speed value"
    )
    cmu_demo.add_argument("--output-dir", default=default_output, help="Artifact directory")
    cmu_demo.add_argument(
        "--no-stop-launch",
        dest="stop_launch",
        action="store_false",
        help="Leave launched simulation running",
    )
    cmu_demo.add_argument("--json", action="store_true", help="Output machine-readable summary")
    cmu_demo.set_defaults(app_handler=cmd_app_cmu_demo, stop_launch=True)


__all__ = ["add_demo_app_subparsers"]
