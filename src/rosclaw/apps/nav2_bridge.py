"""Optional ROS2 Nav2 bridge for the ROSClaw mobility app.

The bridge is import-safe on machines without ROS2.  It only imports rclpy and
Nav2 message packages when a real bridge instance is created.
"""

from __future__ import annotations

import json
import math
import os
import socket
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


class Nav2UnavailableError(RuntimeError):
    """Raised when ROS2/Nav2 Python dependencies are unavailable."""


@dataclass(frozen=True)
class PlacePose:
    """Named Nav2 target pose."""

    name: str
    x: float
    y: float
    theta: float = 0.0
    frame_id: str = "map"
    aliases: tuple[str, ...] = ()


def check_nav2_available() -> tuple[bool, str]:
    """Return whether the Python side of ROS2/Nav2 is importable."""

    try:
        import rclpy  # noqa: F401
        from geometry_msgs.msg import PoseStamped  # noqa: F401
        from nav2_msgs.action import NavigateToPose  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, "rclpy, geometry_msgs, and nav2_msgs are importable"


def load_places_yaml(path: str | Path) -> dict[str, PlacePose]:
    """Load a simple place-name to Nav2 pose map."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "places" in data and isinstance(data["places"], dict):
        data = data["places"]

    places: dict[str, PlacePose] = {}
    for name, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"place '{name}' must be a mapping")
        places[str(name)] = PlacePose(
            name=str(name),
            x=float(value["x"]),
            y=float(value["y"]),
            theta=float(value.get("theta", value.get("yaw", 0.0))),
            frame_id=str(value.get("frame_id", "map")),
            aliases=tuple(str(alias) for alias in value.get("aliases", [])),
        )
    return places


def resolve_place_query(query: str, places: dict[str, PlacePose]) -> PlacePose:
    """Resolve a natural-language place query to a configured pose."""

    text = query.strip()
    lowered = text.lower()
    if text in places:
        return places[text]
    if lowered in places:
        return places[lowered]

    for place in places.values():
        names = (place.name, *place.aliases)
        for name in names:
            if not name:
                continue
            name_lower = name.lower()
            if lowered == name_lower or name_lower in lowered or name in text:
                return place
    known = ", ".join(sorted(places))
    raise ValueError(f"could not resolve place from {query!r}; known places: {known}")


def make_nav2_artifact_dir(
    *,
    place: PlacePose,
    output_dir: str | Path = "practice_data/app_runs",
    app: str = "nav2",
) -> Path:
    """Create a stable artifact directory for one Nav2 app run."""

    safe_place = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in place.name)
    episode_id = f"app_{app}_{safe_place}_{int(time.time() * 1000)}"
    artifact_dir = Path(output_dir).expanduser() / episode_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def write_nav2_artifact(
    *,
    place: PlacePose,
    result: dict[str, Any],
    output_dir: str | Path = "practice_data/app_runs",
    artifact_dir: str | Path | None = None,
    app: str = "nav2-go",
    instruction: str | None = None,
    ros2_topics: list[str] | None = None,
    launch_log_path: str | Path | None = None,
) -> Path:
    """Write a ROSClaw app artifact for a real Nav2 navigation attempt."""

    if artifact_dir is None:
        artifact_dir = make_nav2_artifact_dir(place=place, output_dir=output_dir, app="nav2")
    artifact_dir = Path(artifact_dir).expanduser()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "episode_id": artifact_dir.name,
        "app": app,
        "instruction": instruction,
        "place": {
            "name": place.name,
            "x": place.x,
            "y": place.y,
            "theta": place.theta,
            "frame_id": place.frame_id,
            "aliases": list(place.aliases),
        },
        "result": result,
        "feedback_count": len(result.get("feedback", [])),
        "pose_trace_count": len(result.get("pose_trace", [])),
        "timestamp": time.time(),
        "launch_log": str(launch_log_path) if launch_log_path else None,
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    with open(artifact_dir / "feedback.jsonl", "w", encoding="utf-8") as f:
        for event in result.get("feedback", []):
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    with open(artifact_dir / "pose_trace.jsonl", "w", encoding="utf-8") as f:
        for event in result.get("pose_trace", []):
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    (artifact_dir / "goal.json").write_text(
        json.dumps(summary["place"], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if ros2_topics is not None:
        (artifact_dir / "ros2_topics.txt").write_text(
            "\n".join(sorted(ros2_topics)) + "\n",
            encoding="utf-8",
        )
    return artifact_dir


def launch_humble_nav2_simulation(
    *,
    headless: bool = True,
    use_rviz: bool = False,
    turtlebot_model: str = "waffle",
    log_path: str | Path | None = None,
    spawn_pose: PlacePose | None = None,
    gazebo_master_uri: str | None = None,
) -> subprocess.Popen:
    """Launch the Humble TurtleBot3/Nav2 Gazebo Classic demo."""

    env = os.environ.copy()
    env["TURTLEBOT3_MODEL"] = turtlebot_model
    if gazebo_master_uri is None:
        gazebo_master_uri = _default_gazebo_master_uri()
    if gazebo_master_uri:
        env["GAZEBO_MASTER_URI"] = gazebo_master_uri
    gazebo_model_paths = [
        "/opt/ros/humble/share/turtlebot3_gazebo/models",
        "/usr/share/gazebo-11/models",
    ]
    existing_model_path = env.get("GAZEBO_MODEL_PATH", "")
    env["GAZEBO_MODEL_PATH"] = ":".join([*gazebo_model_paths, existing_model_path])
    env.setdefault("GAZEBO_MODEL_DATABASE_URI", "")
    command = (
        "source /opt/ros/humble/setup.bash && "
        f"export TURTLEBOT3_MODEL={turtlebot_model} && "
        "export GAZEBO_MODEL_DATABASE_URI=${GAZEBO_MODEL_DATABASE_URI:-} && "
        "export GAZEBO_MODEL_PATH="
        "/opt/ros/humble/share/turtlebot3_gazebo/models:"
        "/usr/share/gazebo-11/models:"
        "$GAZEBO_MODEL_PATH && "
        "ros2 launch nav2_bringup tb3_simulation_launch.py "
        f"headless:={'True' if headless else 'False'} "
        f"use_rviz:={'True' if use_rviz else 'False'} "
        "autostart:=True"
    )
    stdout = subprocess.DEVNULL
    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stdout = open(log_file, "w", encoding="utf-8")  # noqa: SIM115 - kept alive by Popen
    pose = spawn_pose or PlacePose(name="spawn", x=-2.0, y=-0.5, theta=0.0)
    return subprocess.Popen(
        [
            "bash",
            "-lc",
            command
            + " "
            + f"x_pose:={pose.x:.6f} "
            + f"y_pose:={pose.y:.6f} "
            + f"yaw:={pose.theta:.6f}",
        ],
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def stop_process_group(process: subprocess.Popen, *, timeout_sec: float = 10.0) -> None:
    """Stop a launch process and its children with SIGINT, then terminate."""

    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.2)
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return


class Nav2Bridge:
    """Thin ROS2 action-client wrapper around Nav2 NavigateToPose."""

    def __init__(
        self,
        *,
        node_name: str = "rosclaw_nav2_bridge",
        action_name: str = "navigate_to_pose",
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        ok, reason = check_nav2_available()
        if not ok:
            raise Nav2UnavailableError(reason)

        import rclpy
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose

        self._rclpy = rclpy
        self._NavigateToPose = NavigateToPose
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._client = ActionClient(self._node, NavigateToPose, action_name)
        self._feedback_callback = feedback_callback
        self._pose_trace: list[dict[str, Any]] = []
        self._subscriptions: list[Any] = []
        self._tf_buffer: Any | None = None
        self._tf_listener: Any | None = None
        self._create_pose_subscriptions()

    def close(self) -> None:
        self._node.destroy_node()

    def wait_for_nav2(self, *, timeout_sec: float = 120.0) -> bool:
        """Wait for Nav2's NavigateToPose action server."""

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._client.wait_for_server(timeout_sec=1.0):
                return True
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
        return False

    def list_topics(self) -> list[str]:
        """List currently visible ROS2 topics."""

        return [name for name, _types in self._node.get_topic_names_and_types()]

    @property
    def pose_trace(self) -> list[dict[str, Any]]:
        return list(self._pose_trace)

    def wait_for_simulation_ready(
        self,
        *,
        timeout_sec: float = 120.0,
        required_topics: tuple[str, ...] = ("/odom", "/scan", "/tf"),
        require_odom_sample: bool = True,
        require_odom_tf: bool = True,
    ) -> dict[str, Any]:
        """Wait until the spawned TurtleBot is publishing motion data.

        Nav2's action server can become visible before Gazebo has spawned the
        robot entity.  A goal sent during that window is commonly rejected
        because the base_link/odom transform does not exist yet.
        """

        started_at = time.time()
        status: dict[str, Any] = {
            "ready": False,
            "required_topics": list(required_topics),
            "seen_topics": [],
            "missing_topics": list(required_topics),
            "odom_samples": self._pose_source_count("odom"),
            "tf_odom_base_link": False,
            "elapsed_sec": 0.0,
        }
        deadline = started_at + timeout_sec
        while time.time() < deadline:
            topics = set(self.list_topics())
            missing = [topic for topic in required_topics if topic not in topics]
            self._rclpy.spin_once(self._node, timeout_sec=0.2)
            odom_samples = self._pose_source_count("odom")
            tf_ok = True
            if require_odom_tf:
                tf_ok = self._can_transform("odom", "base_link")
            status = {
                "ready": (
                    not missing
                    and (not require_odom_sample or odom_samples > 0)
                    and (not require_odom_tf or tf_ok)
                ),
                "required_topics": list(required_topics),
                "seen_topics": sorted(topics),
                "missing_topics": missing,
                "odom_samples": odom_samples,
                "tf_odom_base_link": tf_ok,
                "elapsed_sec": round(time.time() - started_at, 3),
            }
            if status["ready"]:
                return status
            time.sleep(0.2)
        return status

    def wait_for_localization(
        self,
        *,
        initial_pose: PlacePose | None = None,
        timeout_sec: float = 60.0,
        require_amcl_pose: bool = True,
        require_map_tf: bool = True,
        republish_interval_sec: float = 3.0,
    ) -> dict[str, Any]:
        """Wait until AMCL/map localization is available after initial pose."""

        started_at = time.time()
        start_amcl_samples = self._pose_source_count("amcl_pose")
        next_republish = started_at + republish_interval_sec
        target_frame = initial_pose.frame_id if initial_pose else "map"
        status: dict[str, Any] = {
            "ready": False,
            "amcl_pose_samples": start_amcl_samples,
            "new_amcl_pose_samples": 0,
            "tf_map_base_link": False,
            "elapsed_sec": 0.0,
        }
        deadline = started_at + timeout_sec
        while time.time() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.2)
            now = time.time()
            if initial_pose is not None and now >= next_republish:
                self.set_initial_pose(initial_pose, repetitions=1, interval_sec=0.0)
                next_republish = now + republish_interval_sec

            amcl_samples = self._pose_source_count("amcl_pose")
            new_amcl_samples = max(0, amcl_samples - start_amcl_samples)
            tf_ok = True
            if require_map_tf:
                tf_ok = self._can_transform(target_frame, "base_link")
            status = {
                "ready": (
                    (not require_amcl_pose or new_amcl_samples > 0)
                    and (not require_map_tf or tf_ok)
                ),
                "amcl_pose_samples": amcl_samples,
                "new_amcl_pose_samples": new_amcl_samples,
                "tf_map_base_link": tf_ok,
                "target_frame": target_frame,
                "elapsed_sec": round(time.time() - started_at, 3),
            }
            if status["ready"]:
                return status
            time.sleep(0.2)
        return status

    def set_initial_pose(
        self,
        pose: PlacePose,
        *,
        repetitions: int = 8,
        interval_sec: float = 0.25,
    ) -> None:
        """Publish AMCL initial pose on /initialpose."""

        from geometry_msgs.msg import PoseWithCovarianceStamped

        publisher = self._node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        for _ in range(repetitions):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = pose.frame_id
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.pose.pose.position.x = pose.x
            msg.pose.pose.position.y = pose.y
            qz, qw = _yaw_to_quaternion_z_w(pose.theta)
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            msg.pose.covariance[0] = 0.25
            msg.pose.covariance[7] = 0.25
            msg.pose.covariance[35] = 0.06853891945200942
            publisher.publish(msg)
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            time.sleep(interval_sec)

    def navigate_to_place(
        self,
        place: PlacePose,
        *,
        timeout_sec: float = 120.0,
    ) -> dict[str, Any]:
        """Send one NavigateToPose goal and wait for the result."""

        from geometry_msgs.msg import PoseStamped

        if not self.wait_for_nav2(timeout_sec=timeout_sec):
            raise TimeoutError("Nav2 NavigateToPose action server is not available")

        goal_msg = self._NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = place.frame_id
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose.position.x = place.x
        pose.pose.position.y = place.y
        pose.pose.position.z = 0.0
        qz, qw = _yaw_to_quaternion_z_w(place.theta)
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal_msg.pose = pose

        feedback_events: list[dict[str, Any]] = []

        def _on_feedback(msg: Any) -> None:
            feedback = getattr(msg, "feedback", msg)
            event = {
                "distance_remaining": getattr(feedback, "distance_remaining", None),
                "navigation_time": str(getattr(feedback, "navigation_time", "")),
            }
            feedback_events.append(event)
            if self._feedback_callback is not None:
                self._feedback_callback(event)

        send_future = self._client.send_goal_async(goal_msg, feedback_callback=_on_feedback)
        self._rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=timeout_sec)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"status": "rejected", "place": place.name, "feedback": feedback_events}

        result_future = goal_handle.get_result_async()
        self._rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=timeout_sec)
        result = result_future.result()
        raw_status = getattr(result, "status", None)
        status = "timeout" if result is None else ("success" if raw_status == 4 else "failed")
        return {
            "status": status,
            "place": place.name,
            "x": place.x,
            "y": place.y,
            "theta": place.theta,
            "feedback": feedback_events,
            "pose_trace": self.pose_trace,
            "raw_status": raw_status,
        }

    def _create_pose_subscriptions(self) -> None:
        try:
            from geometry_msgs.msg import PoseWithCovarianceStamped
            from nav_msgs.msg import Odometry

            self._subscriptions.append(self._node.create_subscription(
                PoseWithCovarianceStamped,
                "/amcl_pose",
                lambda msg: self._record_pose("amcl_pose", msg.pose.pose),
                10,
            ))
            self._subscriptions.append(self._node.create_subscription(
                Odometry,
                "/odom",
                lambda msg: self._record_pose("odom", msg.pose.pose),
                10,
            ))
        except Exception:
            self._subscriptions = []

    def _record_pose(self, source: str, pose: Any) -> None:
        orientation = pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        self._pose_trace.append({
            "timestamp": time.time(),
            "source": source,
            "x": pose.position.x,
            "y": pose.position.y,
            "theta": yaw,
        })

    def _pose_source_count(self, source: str) -> int:
        return sum(1 for sample in self._pose_trace if sample.get("source") == source)

    def _ensure_tf_listener(self) -> None:
        if self._tf_buffer is not None and self._tf_listener is not None:
            return
        from tf2_ros import Buffer, TransformListener

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

    def _can_transform(self, target_frame: str, source_frame: str) -> bool:
        try:
            from rclpy.time import Time

            self._ensure_tf_listener()
            return bool(self._tf_buffer.can_transform(target_frame, source_frame, Time()))
        except Exception:
            return False


def _yaw_to_quaternion_z_w(theta: float) -> tuple[float, float]:
    half = theta / 2.0
    return math.sin(half), math.cos(half)


def _default_gazebo_master_uri() -> str:
    """Pick a low-collision Gazebo master URI for host-network Docker runs."""

    env_value = os.environ.get("GAZEBO_MASTER_URI")
    if env_value:
        return env_value
    for port in range(11350, 11370):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return f"http://127.0.0.1:{port}"
    return "http://127.0.0.1:11350"
