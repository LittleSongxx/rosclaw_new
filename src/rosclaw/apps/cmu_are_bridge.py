"""ROS1 CMU Autonomous Exploration bridge for ROSClaw.

This module is intentionally import-safe outside ROS1.  It parses a constrained
mobility command, then uses the CMU ARE public ROS topics instead of replacing
the simulator or local planner.

The safety envelope (goal geofence, largest relative move, speed ceiling) is
declared in the ``cmu_are`` embodiment card and resolved through
:mod:`rosclaw.apps.cmu_safety`. The module-level ``DEFAULT_CMU_*`` constants
remain the fallback when no card is present.
"""

from __future__ import annotations

import json
import math
import os
import threading
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml

if TYPE_CHECKING:
    from rosclaw.apps.cmu_safety import CmuSafetyLimits


class CmuAreUnavailableError(RuntimeError):
    """Raised when ROS1/CMU ARE dependencies are unavailable."""


class CmuAreParseError(ValueError):
    """Raised when a language command cannot be grounded safely."""


@dataclass(frozen=True)
class CmuPlace:
    """Named target in the CMU ARE map frame."""

    name: str
    x: float
    y: float
    z: float = 0.0
    frame_id: str = "map"
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CmuIntent:
    """Grounded command for the CMU ARE bridge."""

    type: str
    instruction: str
    x: Optional[float] = None
    y: Optional[float] = None
    z: float = 0.0
    frame_id: str = "map"
    place: Optional[str] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    command: Optional[str] = None
    source: str = "deterministic"


@dataclass
class CmuRunResult:
    """Result and artifact metadata for one CMU ARE command."""

    episode_id: str
    instruction: str
    intent: CmuIntent
    status: str
    artifact_dir: str
    duration_sec: float
    final_pose: Optional[dict[str, float]]
    distance_to_goal: Optional[float]
    odom_trace_count: int
    cmd_vel_count: int
    path_count: int
    waypoint_count: int

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "intent": asdict(self.intent),
            "status": self.status,
            "artifact_dir": self.artifact_dir,
            "duration_sec": self.duration_sec,
            "final_pose": self.final_pose,
            "distance_to_goal": self.distance_to_goal,
            "odom_trace_count": self.odom_trace_count,
            "cmd_vel_count": self.cmd_vel_count,
            "path_count": self.path_count,
            "waypoint_count": self.waypoint_count,
        }


@dataclass(frozen=True)
class CmuChatTurn:
    """One LLM chat turn result for interactive CMU ARE control."""

    status: str
    message: str
    result: CmuRunResult | None = None
    question: str | None = None


@dataclass(frozen=True)
class CmuChatTask:
    """Grounded executable task for the interactive CMU ARE chat console."""

    kind: str
    instruction: str
    intents: tuple[CmuIntent, ...] = ()
    command: Optional[str] = None
    say: str = ""
    source: str = "llm"
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "instruction": self.instruction,
            "intents": [asdict(intent) for intent in self.intents],
            "command": self.command,
            "say": self.say,
            "source": self.source,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class CmuChatTaskEvent:
    """Console-facing event emitted by the asynchronous CMU chat task manager."""

    phase: str
    status: str
    message: str
    task_id: Optional[str] = None
    result: CmuRunResult | None = None


DEFAULT_CMU_MAX_RELATIVE_M = 20.0
DEFAULT_CMU_CHAT_PROGRESS_INTERVAL = 3.0
DEFAULT_CMU_MAX_SEQUENCE_STEPS = 8
DEFAULT_CMU_CIRCLE_SEGMENTS = 12
DEFAULT_CMU_MAX_CIRCLE_RADIUS = 6.0
DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE = 100.0  # Safety limit for absolute coordinates in meters


_NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres|米)?")
_COORD_RE = re.compile(
    r"(?:x|X)\s*[=:：]\s*([-+]?\d+(?:\.\d+)?).*?(?:y|Y)\s*[=:：]\s*([-+]?\d+(?:\.\d+)?)"
)

_EXPLORE_COMMANDS = {
    "start": ("start", "开始探索", "启动探索", "自主探索", "explore", "start exploration"),
    "pause": ("pause", "暂停", "暂停待命", "待命", "停下待命", "pause"),
    "resume": ("resume", "继续", "继续探索", "恢复探索", "resume"),
    "stop": ("stop", "停止探索", "结束探索", "stop"),
}

_UNSUPPORTED_MOTION_HINT = (
    "我可以执行地点/坐标导航、相对移动、多步 waypoint 任务、圆形 waypoint 轨迹和探索控制；"
    "但不会直接接管 /cmd_vel、底层速度曲线或未校验的 ROS topic。"
)
_UNSUPPORTED_MOTION_PATTERNS = (
    "速度曲线",
    "cmd_vel",
    "/cmd_vel",
    "ros topic",
    "topic",
    "twist",
    "底层速度",
)


def check_cmu_are_available() -> tuple[bool, str]:
    """Check whether ROS1 Python messages needed by this bridge are importable."""

    try:
        import rospy  # noqa: F401
        import rospkg
        from geometry_msgs.msg import PointStamped, TwistStamped  # noqa: F401
        from nav_msgs.msg import Odometry, Path as RosPath  # noqa: F401
        from std_msgs.msg import Float32, Int8, String  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)

    try:
        rospack = rospkg.RosPack()
        packages = [
            "vehicle_simulator",
            "local_planner",
            "terrain_analysis",
            "terrain_analysis_ext",
            "sensor_scan_generation",
            "velodyne_description",
            "velodyne_gazebo_plugins",
            "ariadne2",
        ]
        missing = [pkg for pkg in packages if not _rospack_has(rospack, pkg)]
        if missing:
            return False, "missing ROS packages: " + ", ".join(missing)
    except Exception as exc:  # noqa: BLE001
        return False, f"rospack failed: {exc}"
    return True, "ROS1, CMU ARE, and ARiADNE2 packages are discoverable"


def load_cmu_places(path: str | Path) -> dict[str, CmuPlace]:
    """Load named CMU ARE targets from YAML."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = data.get("places", data)
    places: dict[str, CmuPlace] = {}
    for name, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"place '{name}' must be a mapping")
        places[str(name)] = CmuPlace(
            name=str(name),
            x=float(value["x"]),
            y=float(value["y"]),
            z=float(value.get("z", 0.0)),
            frame_id=str(value.get("frame_id", "map")),
            aliases=tuple(str(alias) for alias in value.get("aliases", [])),
        )
    return places


def resolve_cmu_place(query: str, places: dict[str, CmuPlace]) -> CmuPlace:
    """Resolve a natural-language place query against the allowlisted places."""

    text = query.strip()
    lowered = text.lower()
    if text in places:
        return places[text]
    if lowered in places:
        return places[lowered]

    for place in places.values():
        for alias in (place.name, *place.aliases):
            if not alias:
                continue
            alias_lower = alias.lower()
            if lowered == alias_lower or alias_lower in lowered or alias in text:
                return place
    known = ", ".join(sorted(places))
    raise CmuAreParseError(f"could not resolve place from {query!r}; known places: {known}")


def _resolve_limits(
    limits: "CmuSafetyLimits | None",
    *,
    workspace_boundaries: dict[str, Any] | None = None,
    max_relative_m: float | None = None,
) -> "CmuSafetyLimits":
    """Return the effective safety envelope for a navigation entry point.

    Resolves the ``cmu_are`` embodiment card on first use. An explicit
    ``workspace_boundaries`` or ``max_relative_m`` from the caller (CLI flag)
    takes precedence over the card.
    """
    from rosclaw.apps.cmu_safety import resolve_cmu_safety_limits

    if limits is None:
        return resolve_cmu_safety_limits(
            workspace_boundaries=workspace_boundaries,
            max_relative_move_m=max_relative_m,
        )
    return limits.with_overrides(
        workspace_boundaries=workspace_boundaries,
        max_relative_move_m=max_relative_m,
    )


def _absolute_intent(
    *,
    instruction: str,
    x: float,
    y: float,
    z: float = 0.0,
    frame_id: str = "map",
    source: str,
    max_absolute_coordinate_m: float | None = None,
) -> CmuIntent:
    """Build an ``absolute`` intent, enforcing the coordinate safety cap.

    The cap comes from the ``cmu_are`` embodiment card
    (``operational_limits.max_absolute_coordinate_m``) when the caller resolved
    one, otherwise from :data:`DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE`.

    Raises:
        CmuAreParseError: A coordinate lies outside ±max.
    """
    max_abs = (
        max_absolute_coordinate_m
        if max_absolute_coordinate_m is not None
        else DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE
    )
    for axis, value in (("x", x), ("y", y), ("z", z)):
        if abs(value) > max_abs:
            raise CmuAreParseError(
                f"absolute {axis} coordinate {value:.3f}m exceeds safety limit ±{max_abs:.3f}m"
            )
    return CmuIntent(
        type="absolute",
        instruction=instruction,
        x=x,
        y=y,
        z=z,
        frame_id=frame_id,
        source=source,
    )


def parse_cmu_instruction(
    instruction: str,
    *,
    places: dict[str, CmuPlace] | None = None,
    current_pose: dict[str, float] | None = None,
    max_relative_m: float = DEFAULT_CMU_MAX_RELATIVE_M,
    use_llm: bool = False,
    max_absolute_coordinate_m: float | None = None,
) -> CmuIntent:
    """Ground a constrained Chinese/English command into a place, relative move, or control."""

    text = instruction.strip()
    if not text:
        raise CmuAreParseError("empty CMU ARE instruction")

    command = _parse_explore_command(text)
    if command:
        return CmuIntent(type="explore_control", instruction=instruction, command=command)

    coord_match = _COORD_RE.search(text)
    if coord_match:
        # The absolute-coordinate cap applies to the deterministic path too, not
        # just to LLM-produced intents: a typed "x=1e6, y=-5e5" is the same
        # unbounded goal.
        return _absolute_intent(
            instruction=instruction,
            x=float(coord_match.group(1)),
            y=float(coord_match.group(2)),
            z=0.0,
            frame_id="map",
            source="deterministic",
            max_absolute_coordinate_m=max_absolute_coordinate_m,
        )

    relative = _parse_relative_move(text, max_relative_m=max_relative_m, current_pose=current_pose)
    if relative:
        dx, dy = relative
        base = current_pose or {"x": 0.0, "y": 0.0}
        return CmuIntent(
            type="relative",
            instruction=instruction,
            x=float(base.get("x", 0.0)) + dx,
            y=float(base.get("y", 0.0)) + dy,
            dx=dx,
            dy=dy,
            source="deterministic",
        )

    if places:
        try:
            place = resolve_cmu_place(text, places)
            return CmuIntent(
                type="place",
                instruction=instruction,
                x=place.x,
                y=place.y,
                z=place.z,
                frame_id=place.frame_id,
                place=place.name,
                source="deterministic",
            )
        except CmuAreParseError:
            pass

    if use_llm:
        llm_intent = _parse_with_llm(text)
        return _validate_llm_intent(
            llm_intent,
            instruction=instruction,
            places=places or {},
            current_pose=current_pose,
            max_relative_m=max_relative_m,
            max_absolute_coordinate_m=max_absolute_coordinate_m,
        )

    raise CmuAreParseError(
        "instruction must be a known place, x/y coordinate, relative move, or exploration command"
    )


def launch_cmu_are_simulation(
    *,
    world: str = "campus",
    headless: bool = True,
    use_rviz: bool = False,
    log_path: str | Path | None = None,
    include_ariadne2: bool = False,
    ariadne2_active: bool = False,
    ariadne2_use_rviz: bool = False,
) -> subprocess.Popen:
    """Launch vendored CMU ARE Gazebo/RViz simulation in ROS1 Noetic."""

    env = os.environ.copy()
    env.setdefault("GAZEBO_MODEL_DATABASE_URI", "")
    env.setdefault("DISABLE_ROS1_EOL_WARNINGS", "1")
    env["GAZEBO_MODEL_PATH"] = _append_env_path(
        env.get("GAZEBO_MODEL_PATH", ""),
        [
            "/opt/rosclaw/third_party/ros1/are/src/vehicle_simulator/mesh",
            str(Path.cwd() / "third_party/ros1/are/src/vehicle_simulator/mesh"),
        ],
    )
    launch_file = f"system_{world}.launch"
    command = (
        "source /opt/ros/noetic/setup.bash && "
        "source /opt/rosclaw/ros1_ws/devel/setup.bash && "
        "roslaunch vehicle_simulator "
        f"{launch_file} gazebo_gui:={'false' if headless else 'true'} "
        f"launch_rviz:={'true' if use_rviz else 'false'}"
    )
    if include_ariadne2:
        command += (
            " & sleep 12 && "
            "roslaunch ariadne2 "
            f"ariadne2_{world}.launch launch_rviz:={'true' if ariadne2_use_rviz else 'false'} "
            f"start_active:={'true' if ariadne2_active else 'false'}"
        )

    stdout: int | Any = subprocess.DEVNULL
    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stdout = open(log_file, "w", encoding="utf-8")  # noqa: SIM115 - owned by Popen

    return subprocess.Popen(
        ["bash", "-lc", command],
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def stop_process_group(process: subprocess.Popen, *, timeout_sec: float = 10.0) -> None:
    """Stop a launch process and its children."""

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


class CmuAreBridge:
    """Thin ROS1 publisher/subscriber facade for CMU ARE."""

    def __init__(self, *, node_name: str = "rosclaw_cmu_are_bridge") -> None:
        ok, reason = check_cmu_are_available()
        if not ok:
            raise CmuAreUnavailableError(reason)

        import rospy
        from geometry_msgs.msg import PointStamped, TwistStamped
        from nav_msgs.msg import Odometry, Path as RosPath
        from std_msgs.msg import Float32, Int8, String

        self.rospy = rospy
        self.PointStamped = PointStamped
        self.Float32 = Float32
        self.Int8 = Int8
        self.String = String
        self._pose: Optional[dict[str, float]] = None
        self._odom_trace: list[dict[str, Any]] = []
        self._cmd_trace: list[dict[str, Any]] = []
        self._path_trace: list[dict[str, Any]] = []
        self._waypoint_trace: list[dict[str, Any]] = []

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

        self.waypoint_pub = rospy.Publisher("/way_point", PointStamped, queue_size=5)
        self.speed_pub = rospy.Publisher("/speed", Float32, queue_size=5, latch=True)
        self.stop_pub = rospy.Publisher("/stop", Int8, queue_size=5, latch=True)
        self.explore_pub = rospy.Publisher(
            "/rosclaw/exploration_control", String, queue_size=5, latch=True
        )
        self._subs = [
            rospy.Subscriber("/state_estimation", Odometry, self._odom_callback, queue_size=25),
            rospy.Subscriber("/cmd_vel", TwistStamped, self._cmd_callback, queue_size=25),
            rospy.Subscriber("/path", RosPath, self._path_callback, queue_size=10),
            rospy.Subscriber("/way_point", PointStamped, self._waypoint_callback, queue_size=25),
        ]

    @property
    def current_pose(self) -> Optional[dict[str, float]]:
        return dict(self._pose) if self._pose else None

    @property
    def odom_trace(self) -> list[dict[str, Any]]:
        return list(self._odom_trace)

    @property
    def cmd_trace(self) -> list[dict[str, Any]]:
        return list(self._cmd_trace)

    @property
    def path_trace(self) -> list[dict[str, Any]]:
        return list(self._path_trace)

    @property
    def waypoint_trace(self) -> list[dict[str, Any]]:
        return list(self._waypoint_trace)

    def list_topics(self) -> list[str]:
        topics = self.rospy.get_published_topics()
        return sorted(topic for topic, _type in topics)

    def wait_for_topics(
        self,
        *,
        required_topics: tuple[str, ...] = ("/state_estimation", "/registered_scan", "/terrain_map"),
        timeout_sec: float = 60.0,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        seen: set[str] = set()
        while time.time() < deadline and not self.rospy.is_shutdown():
            seen = set(self.list_topics())
            if all(topic in seen for topic in required_topics):
                return {"ready": True, "seen_topics": sorted(seen), "missing_topics": []}
            time.sleep(0.5)
        missing = [topic for topic in required_topics if topic not in seen]
        return {"ready": False, "seen_topics": sorted(seen), "missing_topics": missing}

    def wait_for_pose(self, *, timeout_sec: float = 60.0) -> Optional[dict[str, float]]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline and not self.rospy.is_shutdown():
            if self._pose is not None:
                return dict(self._pose)
            time.sleep(0.1)
        return None

    def _check_workspace_boundaries(
        self,
        *,
        x: float,
        y: float,
        z: float,
        workspace_boundaries: dict[str, Any],
    ) -> None:
        """Validate coordinates against workspace boundaries.

        Raises:
            CmuAreParseError: If coordinates violate workspace boundaries.
        """
        from rosclaw.apps.cmu_safety import CmuGeofenceError, check_within_workspace

        try:
            check_within_workspace(
                x=x, y=y, z=z, workspace_boundaries=workspace_boundaries
            )
        except CmuGeofenceError as exc:
            raise CmuAreParseError(str(exc)) from exc

    def publish_waypoint(
        self,
        *,
        x: float,
        y: float,
        z: float = 0.0,
        frame_id: str = "map",
        speed: float | None = None,
    ) -> None:
        if speed is not None:
            self.publish_speed(speed)
        self.publish_stop(False)
        msg = self.PointStamped()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(z)
        self.waypoint_pub.publish(msg)
        self._waypoint_trace.append(_point_msg_to_dict(msg))

    def publish_speed(self, speed: float) -> None:
        msg = self.Float32()
        msg.data = float(speed)
        self.speed_pub.publish(msg)

    def publish_stop(self, stopped: bool) -> None:
        msg = self.Int8()
        msg.data = 1 if stopped else 0
        self.stop_pub.publish(msg)

    def publish_exploration_control(self, command: str, *, speed: float = 2.0) -> None:
        command = command.strip().lower()
        if command not in {"start", "pause", "resume", "stop"}:
            raise ValueError(f"unsupported exploration command: {command}")

        control_msg = self.String()
        control_msg.data = command
        speed_msg = self.Float32()
        speed_msg.data = 0.0 if command in {"pause", "stop"} else float(speed)
        stop_msg = self.Int8()
        stop_msg.data = 1 if command in {"pause", "stop"} else 0

        publishers = [self.explore_pub, self.stop_pub, self.speed_pub]
        self._wait_for_publisher_connections(publishers, timeout_sec=3.0)
        time.sleep(0.3)

        # ROS1 short-lived CLI publishers can exit before subscribers complete
        # the TCPROS handshake. Repeating for a short window makes control
        # commands reliable without keeping a daemon-side bridge process.
        deadline = time.time() + 3.0
        while time.time() < deadline and not self.rospy.is_shutdown():
            self.speed_pub.publish(speed_msg)
            self.stop_pub.publish(stop_msg)
            self.explore_pub.publish(control_msg)
            time.sleep(0.1)
        time.sleep(0.2)

    def _wait_for_publisher_connections(self, publishers: list[Any], *, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline and not self.rospy.is_shutdown():
            if all(pub.get_num_connections() > 0 for pub in publishers):
                return
            time.sleep(0.05)

    def navigate_to_intent(
        self,
        intent: CmuIntent,
        *,
        speed: float = 2.0,
        timeout_sec: float = 120.0,
        tolerance_m: float = 1.5,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        workspace_boundaries: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if intent.x is None or intent.y is None:
            raise ValueError("navigation intent must include x and y")

        # Validate against workspace boundaries if provided
        if workspace_boundaries:
            self._check_workspace_boundaries(
                x=float(intent.x),
                y=float(intent.y),
                z=float(intent.z),
                workspace_boundaries=workspace_boundaries,
            )

        start = time.time()
        self.publish_waypoint(
            x=float(intent.x),
            y=float(intent.y),
            z=float(intent.z),
            frame_id=intent.frame_id,
            speed=speed,
        )
        status = "timeout"
        final_pose: Optional[dict[str, float]] = None
        distance: Optional[float] = None

        rate = self.rospy.Rate(5)
        while time.time() - start < timeout_sec and not self.rospy.is_shutdown():
            if cancel_event is not None and cancel_event.is_set():
                status = "cancelled"
                self.publish_stop(True)
                break
            pose = self.current_pose
            if pose:
                final_pose = pose
                distance = math.hypot(float(intent.x) - pose["x"], float(intent.y) - pose["y"])
                event = {
                    "timestamp": time.time(),
                    "x": pose["x"],
                    "y": pose["y"],
                    "distance_to_goal": distance,
                }
                if progress_callback:
                    progress_callback(event)
                if distance <= tolerance_m:
                    status = "success"
                    break
            rate.sleep()

        return {
            "status": status,
            "duration_sec": time.time() - start,
            "final_pose": final_pose,
            "distance_to_goal": distance,
        }

    def _odom_callback(self, msg: Any) -> None:
        pose = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        theta = _quaternion_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        item = {
            "timestamp": time.time(),
            "stamp": msg.header.stamp.to_sec() if msg.header.stamp else None,
            "x": float(pose.x),
            "y": float(pose.y),
            "z": float(pose.z),
            "theta": theta,
        }
        self._pose = {"x": item["x"], "y": item["y"], "z": item["z"], "theta": item["theta"]}
        self._append_limited(self._odom_trace, item)

    def _cmd_callback(self, msg: Any) -> None:
        item = {
            "timestamp": time.time(),
            "linear_x": float(msg.twist.linear.x),
            "linear_y": float(msg.twist.linear.y),
            "angular_z": float(msg.twist.angular.z),
        }
        self._append_limited(self._cmd_trace, item)

    def _path_callback(self, msg: Any) -> None:
        item = {
            "timestamp": time.time(),
            "poses": len(msg.poses),
            "frame_id": msg.header.frame_id,
        }
        self._append_limited(self._path_trace, item)

    def _waypoint_callback(self, msg: Any) -> None:
        self._append_limited(self._waypoint_trace, _point_msg_to_dict(msg))

    @staticmethod
    def _append_limited(items: list[dict[str, Any]], item: dict[str, Any], *, limit: int = 10000) -> None:
        items.append(item)
        if len(items) > limit:
            del items[: len(items) - limit]


def run_cmu_go(
    instruction: str,
    *,
    places_path: str | Path = "docker/ros1/places.campus.yaml",
    output_dir: str | Path = "practice_data/app_runs",
    timeout_sec: float = 120.0,
    readiness_timeout_sec: float = 60.0,
    tolerance_m: float = 1.5,
    speed: float = 2.0,
    use_llm: bool = False,
    workspace_boundaries: dict[str, Any] | None = None,
    limits: "CmuSafetyLimits | None" = None,
) -> CmuRunResult:
    """Parse and execute one target-navigation command against a running CMU ARE sim.

    ``limits`` carries the resolved embodiment-card safety envelope. When
    omitted it is resolved from the ``cmu_are`` card, and ``workspace_boundaries``
    (if given) overrides the card's geofence.
    """

    limits = _resolve_limits(limits, workspace_boundaries=workspace_boundaries)
    workspace_boundaries = limits.workspace_boundaries
    speed = limits.clamp_speed(speed)

    places = load_cmu_places(places_path)
    bridge = CmuAreBridge()
    readiness = bridge.wait_for_topics(timeout_sec=readiness_timeout_sec)
    if not readiness["ready"]:
        raise CmuAreUnavailableError("CMU ARE topics not ready: " + ", ".join(readiness["missing_topics"]))
    current_pose = bridge.wait_for_pose(timeout_sec=readiness_timeout_sec)
    intent = parse_cmu_instruction(
        instruction,
        places=places,
        current_pose=current_pose,
        use_llm=use_llm,
        max_relative_m=limits.max_relative_move_m,
        max_absolute_coordinate_m=limits.max_absolute_coordinate_m,
    )
    if intent.type == "explore_control":
        if not intent.command:
            raise CmuAreParseError("exploration control intent missing command")
        bridge.publish_exploration_control(intent.command, speed=speed)
        navigation = {
            "status": "success",
            "duration_sec": 0.0,
            "final_pose": bridge.current_pose,
            "distance_to_goal": None,
        }
    else:
        navigation = bridge.navigate_to_intent(
            intent,
            speed=speed,
            timeout_sec=timeout_sec,
            tolerance_m=tolerance_m,
            workspace_boundaries=workspace_boundaries,
        )

    artifact_dir = make_cmu_artifact_dir(output_dir=output_dir, app="cmu_go")
    return write_cmu_artifact(
        artifact_dir=artifact_dir,
        instruction=instruction,
        intent=intent,
        status=navigation["status"],
        duration_sec=float(navigation["duration_sec"]),
        final_pose=navigation.get("final_pose"),
        distance_to_goal=navigation.get("distance_to_goal"),
        bridge=bridge,
        ros_topics=readiness["seen_topics"],
    )


def run_cmu_chat_turn(
    instruction: str,
    *,
    places_path: str | Path = "docker/ros1/places.campus.yaml",
    output_dir: str | Path = "practice_data/app_runs",
    timeout_sec: float = 120.0,
    readiness_timeout_sec: float = 60.0,
    tolerance_m: float = 1.5,
    speed: float = 2.0,
    bridge: CmuAreBridge | None = None,
    places: dict[str, CmuPlace] | None = None,
    ros_topics: list[str] | None = None,
    max_relative_m: float = DEFAULT_CMU_MAX_RELATIVE_M,
    workspace_boundaries: dict[str, Any] | None = None,
    limits: "CmuSafetyLimits | None" = None,
) -> CmuChatTurn:
    """Use an LLM-only parser to execute or clarify one interactive command."""

    limits = _resolve_limits(
        limits,
        workspace_boundaries=workspace_boundaries,
        max_relative_m=max_relative_m,
    )
    workspace_boundaries = limits.workspace_boundaries
    max_relative_m = limits.max_relative_move_m
    speed = limits.clamp_speed(speed)

    resolved_places = places or load_cmu_places(places_path)
    active_bridge = bridge or CmuAreBridge()
    if ros_topics is None:
        readiness = active_bridge.wait_for_topics(timeout_sec=readiness_timeout_sec)
        if not readiness["ready"]:
            raise CmuAreUnavailableError(
                "CMU ARE topics not ready: " + ", ".join(readiness["missing_topics"])
            )
        ros_topics = readiness["seen_topics"]

    current_pose = active_bridge.wait_for_pose(timeout_sec=readiness_timeout_sec)
    unsupported = describe_unsupported_cmu_command(instruction)
    if unsupported:
        return CmuChatTurn(status="unsupported", message=unsupported, question=unsupported)

    llm_data = _parse_with_llm_chat(
        instruction,
        places=resolved_places,
        current_pose=current_pose,
        max_relative_m=max_relative_m,
    )
    status = str(llm_data.get("status", "action")).lower()
    if status == "clarify":
        question = str(llm_data.get("question", "")).strip()
        if not question:
            raise CmuAreParseError("LLM requested clarification without a question")
        return CmuChatTurn(status="clarify", message=question, question=question)
    if status != "action":
        raise CmuAreParseError(f"LLM returned unsupported chat status {status!r}")

    message = str(llm_data.get("say", "")).strip() or "收到，正在执行。"
    intent = _validate_llm_intent(
        llm_data,
        instruction=instruction,
        places=resolved_places,
        current_pose=current_pose,
        max_relative_m=max_relative_m,
        max_absolute_coordinate_m=limits.max_absolute_coordinate_m,
    )
    if intent.type == "explore_control":
        if not intent.command:
            raise CmuAreParseError("exploration control intent missing command")
        active_bridge.publish_exploration_control(intent.command, speed=speed)
        navigation = {
            "status": "success",
            "duration_sec": 0.0,
            "final_pose": active_bridge.current_pose,
            "distance_to_goal": None,
        }
    else:
        navigation = active_bridge.navigate_to_intent(
            intent,
            speed=speed,
            timeout_sec=timeout_sec,
            tolerance_m=tolerance_m,
            workspace_boundaries=workspace_boundaries,
        )

    artifact_dir = make_cmu_artifact_dir(output_dir=output_dir, app="cmu_chat")
    result = write_cmu_artifact(
        artifact_dir=artifact_dir,
        instruction=instruction,
        intent=intent,
        status=navigation["status"],
        duration_sec=float(navigation["duration_sec"]),
        final_pose=navigation.get("final_pose"),
        distance_to_goal=navigation.get("distance_to_goal"),
        bridge=active_bridge,
        ros_topics=ros_topics,
    )
    result_message = build_cmu_result_message(result, prefix=message)
    return CmuChatTurn(status="action", message=result_message, result=result)


class CmuChatTaskManager:
    """Asynchronous task runner for the interactive CMU ARE chat shell."""

    def __init__(
        self,
        *,
        bridge: CmuAreBridge,
        places: dict[str, CmuPlace],
        ros_topics: list[str],
        output_dir: str | Path = "practice_data/app_runs",
        timeout_sec: float = 120.0,
        readiness_timeout_sec: float = 60.0,
        tolerance_m: float = 1.5,
        speed: float = 2.0,
        max_relative_m: float = DEFAULT_CMU_MAX_RELATIVE_M,
        progress_interval_sec: float = DEFAULT_CMU_CHAT_PROGRESS_INTERVAL,
        max_sequence_steps: int = DEFAULT_CMU_MAX_SEQUENCE_STEPS,
        circle_segments: int = DEFAULT_CMU_CIRCLE_SEGMENTS,
        max_circle_radius_m: float = DEFAULT_CMU_MAX_CIRCLE_RADIUS,
        exploration_on_manual: str = "pause",
        workspace_boundaries: dict[str, Any] | None = None,
        limits: "CmuSafetyLimits | None" = None,
    ) -> None:
        # The embodiment card supplies the safety envelope; explicit kwargs from
        # the CLI still win. Only the geofence and relative-move cap are treated
        # as overrides here, since the rest are already explicit defaults.
        self.limits = _resolve_limits(limits, workspace_boundaries=workspace_boundaries)

        self.bridge = bridge
        self.places = places
        self.ros_topics = ros_topics
        self.output_dir = output_dir
        self.timeout_sec = timeout_sec
        self.readiness_timeout_sec = readiness_timeout_sec
        self.tolerance_m = tolerance_m
        self.speed = self.limits.clamp_speed(speed)
        self.max_relative_m = max_relative_m
        self.progress_interval_sec = progress_interval_sec
        self.max_sequence_steps = max_sequence_steps
        self.circle_segments = circle_segments
        self.max_circle_radius_m = max_circle_radius_m
        self.exploration_on_manual = exploration_on_manual
        self.workspace_boundaries = self.limits.workspace_boundaries

        self._lock = threading.Lock()
        self._events: list[CmuChatTaskEvent] = []
        self._current_thread: threading.Thread | None = None
        self._current_cancel: threading.Event | None = None
        self._current_task_id: str | None = None
        self._current_task_kind: str | None = None
        self._task_counter = 0
        self._exploration_active = False
        self._bridge_command_lock = threading.Lock()

    def submit(self, instruction: str) -> list[CmuChatTaskEvent]:
        """Parse a user instruction, apply preemption policy, and start a task if needed."""

        current_pose = self.bridge.current_pose
        if current_pose is None:
            current_pose = self.bridge.wait_for_pose(timeout_sec=min(self.readiness_timeout_sec, 5.0))
        task = parse_cmu_chat_task(
            instruction,
            places=self.places,
            current_pose=current_pose,
            max_relative_m=self.max_relative_m,
            max_sequence_steps=self.max_sequence_steps,
            circle_segments=self.circle_segments,
            max_circle_radius_m=self.max_circle_radius_m,
            max_absolute_coordinate_m=self.limits.max_absolute_coordinate_m,
        )
        if task.kind == "clarify":
            question = str((task.metadata or {}).get("question", task.say or "请再说清楚一点。"))
            return [CmuChatTaskEvent(phase="clarify", status="clarify", message=question)]
        if task.kind == "cancel":
            return self.cancel_current(reason=task.say or "已取消当前任务。")

        initial_events: list[CmuChatTaskEvent] = []
        if task.kind in {"navigation", "sequence"}:
            cancel_events = self._cancel_current_locked(
                reason="收到新任务，已取消上一条移动任务。", preempt_only=True
            )
            initial_events.extend(cancel_events)
            if self._exploration_active and self.exploration_on_manual == "pause":
                with self._bridge_command_lock:
                    self.bridge.publish_exploration_control("pause", speed=self.speed)
                self._exploration_active = True
                initial_events.append(
                    CmuChatTaskEvent(
                        phase="progress",
                        status="paused_exploration",
                        message="正在暂停自主探索，切换到人工移动任务。",
                    )
                )
        elif task.kind == "explore_control":
            cancel_events = self._cancel_current_locked(
                reason="收到探索控制指令，已取消当前移动任务。", preempt_only=True
            )
            initial_events.extend(cancel_events)

        task_id = self._next_task_id()
        cancel_event = threading.Event()
        start_event = CmuChatTaskEvent(
            phase="start",
            status="running",
            message=_task_start_message(task),
            task_id=task_id,
        )
        with self._lock:
            self._current_task_id = task_id
            self._current_task_kind = task.kind
            self._current_cancel = cancel_event
        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, task, cancel_event),
            name=f"rosclaw-cmu-chat-{task_id}",
            daemon=True,
        )
        with self._lock:
            self._current_thread = thread
        thread.start()
        return [*initial_events, start_event]

    def drain_events(self) -> list[CmuChatTaskEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def cancel_current(self, *, reason: str = "已取消当前任务。") -> list[CmuChatTaskEvent]:
        return self._cancel_current_locked(reason=reason, preempt_only=False)

    def _next_task_id(self) -> str:
        with self._lock:
            self._task_counter += 1
            return f"cmu_chat_task_{self._task_counter:04d}"

    def _emit(self, event: CmuChatTaskEvent) -> None:
        with self._lock:
            self._events.append(event)

    def _cancel_current_locked(
        self,
        *,
        reason: str,
        preempt_only: bool,
    ) -> list[CmuChatTaskEvent]:
        with self._lock:
            cancel_event = self._current_cancel
            task_id = self._current_task_id
            thread = self._current_thread
            kind = self._current_task_kind
        if cancel_event is None or task_id is None:
            return [] if preempt_only else [CmuChatTaskEvent(phase="end", status="idle", message="当前没有正在执行的移动任务。")]
        if kind not in {"navigation", "sequence"} and preempt_only:
            return []
        cancel_event.set()
        try:
            with self._bridge_command_lock:
                self.bridge.publish_stop(True)
        except Exception:  # noqa: BLE001
            pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        return [
            CmuChatTaskEvent(
                phase="end",
                status="cancelled",
                message=reason,
                task_id=task_id,
            )
        ]

    def _run_task(self, task_id: str, task: CmuChatTask, cancel_event: threading.Event) -> None:
        started = time.time()
        artifact_dir = make_cmu_artifact_dir(output_dir=self.output_dir, app=f"cmu_chat_{task_id}")
        events: list[dict[str, Any]] = [
            {"timestamp": time.time(), "phase": "start", "task": task.to_dict()}
        ]
        result: CmuRunResult | None = None
        status = "failed"
        final_pose: Optional[dict[str, float]] = None
        distance_to_goal: Optional[float] = None
        error: str | None = None
        try:
            if task.kind == "explore_control":
                if not task.command:
                    raise CmuAreParseError("exploration control task missing command")
                self._emit(
                    CmuChatTaskEvent(
                        phase="progress",
                        status="running",
                        message=_task_progress_message(task),
                        task_id=task_id,
                    )
                )
                events.append(
                    {
                        "timestamp": time.time(),
                        "phase": "progress",
                        "status": "running",
                        "message": _task_progress_message(task),
                    }
                )
                with self._bridge_command_lock:
                    self.bridge.publish_exploration_control(task.command, speed=self.speed)
                self._exploration_active = task.command in {"start", "resume", "pause"}
                if task.command == "stop":
                    self._exploration_active = False
                status = "success"
                final_pose = self.bridge.current_pose
                distance_to_goal = None
            else:
                intents = task.intents
                if not intents:
                    raise CmuAreParseError("navigation task has no waypoints")
                last_progress_at = 0.0
                for index, intent in enumerate(intents, start=1):
                    if cancel_event.is_set():
                        status = "cancelled"
                        break

                    def progress_callback(event: dict[str, Any], *, step: int = index) -> None:
                        nonlocal last_progress_at
                        now = time.time()
                        if now - last_progress_at < self.progress_interval_sec:
                            return
                        last_progress_at = now
                        message = _task_progress_message(
                            task,
                            step=step,
                            total=len(intents),
                            distance_to_goal=event.get("distance_to_goal"),
                        )
                        events.append(
                            {
                                "timestamp": now,
                                "phase": "progress",
                                "status": "running",
                                "step": step,
                                "distance_to_goal": event.get("distance_to_goal"),
                                "message": message,
                            }
                        )
                        self._emit(
                            CmuChatTaskEvent(
                                phase="progress",
                                status="running",
                                message=message,
                                task_id=task_id,
                            )
                        )

                    navigation = self.bridge.navigate_to_intent(
                        intent,
                        speed=self.speed,
                        timeout_sec=self.timeout_sec,
                        tolerance_m=self.tolerance_m,
                        progress_callback=progress_callback,
                        cancel_event=cancel_event,
                        workspace_boundaries=self.workspace_boundaries,
                    )
                    events.append(
                        {
                            "timestamp": time.time(),
                            "phase": "waypoint_result",
                            "step": index,
                            "intent": asdict(intent),
                            "navigation": navigation,
                        }
                    )
                    status = str(navigation["status"])
                    final_pose = navigation.get("final_pose")
                    distance_to_goal = navigation.get("distance_to_goal")
                    if status != "success":
                        break
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            status = "failed"
        finally:
            if cancel_event.is_set() and status not in {"success", "failed"}:
                status = "cancelled"
            if task.intents:
                result_intent = task.intents[-1]
            else:
                result_intent = CmuIntent(
                    type="explore_control" if task.kind == "explore_control" else task.kind,
                    instruction=task.instruction,
                    command=task.command,
                    source=task.source,
                )
            result = write_cmu_artifact(
                artifact_dir=artifact_dir,
                instruction=task.instruction,
                intent=result_intent,
                status=status,
                duration_sec=time.time() - started,
                final_pose=final_pose,
                distance_to_goal=distance_to_goal,
                bridge=self.bridge,
                ros_topics=self.ros_topics,
                task=task.to_dict(),
                task_events=events,
                error=error,
            )
            message = _task_end_message(task, result, error=error)
            events.append(
                {
                    "timestamp": time.time(),
                    "phase": "end",
                    "status": status,
                    "message": message,
                    "result": result.to_dict(),
                    "error": error,
                }
            )
            _write_jsonl(Path(result.artifact_dir) / "task_events.jsonl", events)
            self._emit(
                CmuChatTaskEvent(
                    phase="end",
                    status=status,
                    message=message,
                    task_id=task_id,
                    result=result,
                )
            )
            with self._lock:
                if self._current_task_id == task_id:
                    self._current_task_id = None
                    self._current_cancel = None
                    self._current_thread = None
                    self._current_task_kind = None


def run_cmu_demo(
    instruction: str,
    *,
    places_path: str | Path = "docker/ros1/places.campus.yaml",
    output_dir: str | Path = "practice_data/app_runs",
    world: str = "campus",
    headless: bool = True,
    use_rviz: bool = False,
    timeout_sec: float = 120.0,
    readiness_timeout_sec: float = 90.0,
    tolerance_m: float = 1.5,
    speed: float = 2.0,
    stop_launch: bool = True,
) -> CmuRunResult:
    """Launch the CMU ARE sim, run one command, and write artifacts."""

    artifact_dir = make_cmu_artifact_dir(output_dir=output_dir, app="cmu_demo")
    launch_log = artifact_dir / "cmu_launch.log"
    process = launch_cmu_are_simulation(
        world=world,
        headless=headless,
        use_rviz=use_rviz,
        log_path=launch_log,
    )
    try:
        result = run_cmu_go(
            instruction,
            places_path=places_path,
            output_dir=output_dir,
            timeout_sec=timeout_sec,
            readiness_timeout_sec=readiness_timeout_sec,
            tolerance_m=tolerance_m,
            speed=speed,
        )
        _copy_file_if_exists(launch_log, Path(result.artifact_dir) / "cmu_launch.log")
        return result
    finally:
        if stop_launch:
            stop_process_group(process)


def parse_cmu_chat_task(
    instruction: str,
    *,
    places: dict[str, CmuPlace],
    current_pose: dict[str, float] | None,
    max_relative_m: float = DEFAULT_CMU_MAX_RELATIVE_M,
    max_sequence_steps: int = DEFAULT_CMU_MAX_SEQUENCE_STEPS,
    circle_segments: int = DEFAULT_CMU_CIRCLE_SEGMENTS,
    max_circle_radius_m: float = DEFAULT_CMU_MAX_CIRCLE_RADIUS,
    max_absolute_coordinate_m: float | None = None,
) -> CmuChatTask:
    """Parse an interactive LLM command into an executable task."""

    text = instruction.strip()
    if not text:
        raise CmuAreParseError("empty CMU ARE instruction")
    if _is_cancel_instruction(text):
        return CmuChatTask(kind="cancel", instruction=instruction, say="已收到取消指令。", source="deterministic")

    unsupported = describe_unsupported_cmu_command(instruction)
    if unsupported:
        return CmuChatTask(
            kind="clarify",
            instruction=instruction,
            say=unsupported,
            source="deterministic",
            metadata={"question": unsupported},
        )

    data = _parse_with_llm_chat(
        instruction,
        places=places,
        current_pose=current_pose,
        max_relative_m=max_relative_m,
        max_sequence_steps=max_sequence_steps,
        max_circle_radius_m=max_circle_radius_m,
    )
    status = str(data.get("status", "action")).lower()
    if status == "clarify":
        question = str(data.get("question", "")).strip() or "这条指令还不够明确，你想让我怎么移动？"
        return CmuChatTask(
            kind="clarify",
            instruction=instruction,
            say=question,
            metadata={"question": question},
        )
    if status != "action":
        raise CmuAreParseError(f"LLM returned unsupported chat status {status!r}")

    task = _validate_llm_task(
        data,
        instruction=instruction,
        places=places,
        current_pose=current_pose,
        max_relative_m=max_relative_m,
        max_sequence_steps=max_sequence_steps,
        circle_segments=circle_segments,
        max_circle_radius_m=max_circle_radius_m,
        max_absolute_coordinate_m=max_absolute_coordinate_m,
    )
    return task


def _validate_llm_task(
    data: dict[str, Any],
    *,
    instruction: str,
    places: dict[str, CmuPlace],
    current_pose: dict[str, float] | None,
    max_relative_m: float,
    max_sequence_steps: int,
    circle_segments: int,
    max_circle_radius_m: float,
    max_absolute_coordinate_m: float | None = None,
) -> CmuChatTask:
    if not isinstance(data, dict):
        raise CmuAreParseError("LLM returned non-object task")
    say = str(data.get("say", "")).strip()
    typ = str(data.get("type", "")).lower()
    if typ == "cancel":
        return CmuChatTask(kind="cancel", instruction=instruction, say=say or "已收到取消指令。")
    if typ == "explore_control":
        command = str(data.get("command", "")).lower()
        if command not in {"start", "pause", "resume", "stop"}:
            raise CmuAreParseError(f"LLM returned invalid exploration command {command!r}")
        return CmuChatTask(
            kind="explore_control",
            instruction=instruction,
            command=command,
            say=say,
            metadata={"command": command},
        )
    if typ in {"place", "absolute", "relative"}:
        intent = _validate_llm_intent(
            data,
            instruction=instruction,
            places=places,
            current_pose=current_pose,
            max_relative_m=max_relative_m,
            max_absolute_coordinate_m=max_absolute_coordinate_m,
        )
        return CmuChatTask(
            kind="navigation",
            instruction=instruction,
            intents=(intent,),
            say=say,
            metadata={"intent_type": intent.type},
        )
    if typ == "sequence":
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            raise CmuAreParseError("sequence task must include non-empty steps")
        if len(raw_steps) > max_sequence_steps:
            raise CmuAreParseError(
                f"sequence has {len(raw_steps)} steps, exceeds limit {max_sequence_steps}"
            )
        intents: list[CmuIntent] = []
        pose = dict(current_pose or {"x": 0.0, "y": 0.0, "theta": 0.0})
        for index, step in enumerate(raw_steps, start=1):
            if not isinstance(step, dict):
                raise CmuAreParseError(f"sequence step {index} must be an object")
            step_data = dict(step)
            step_data.setdefault("status", "action")
            intent = _validate_llm_intent(
                step_data,
                instruction=f"{instruction} / step {index}",
                places=places,
                current_pose=pose,
                max_relative_m=max_relative_m,
                max_absolute_coordinate_m=max_absolute_coordinate_m,
            )
            if intent.type == "explore_control":
                raise CmuAreParseError("sequence steps cannot include exploration control")
            intents.append(intent)
            if intent.x is not None and intent.y is not None:
                pose["x"] = intent.x
                pose["y"] = intent.y
                if intent.dx is not None or intent.dy is not None:
                    dx = float(intent.dx or 0.0)
                    dy = float(intent.dy or 0.0)
                    if dx or dy:
                        pose["theta"] = math.atan2(dy, dx)
        return CmuChatTask(
            kind="sequence",
            instruction=instruction,
            intents=tuple(intents),
            say=say,
            metadata={"steps": len(intents), "mode": "sequence"},
        )
    if typ == "circle":
        intents = _circle_task_to_intents(
            data,
            instruction=instruction,
            current_pose=current_pose,
            max_circle_radius_m=max_circle_radius_m,
            circle_segments=circle_segments,
        )
        return CmuChatTask(
            kind="sequence",
            instruction=instruction,
            intents=intents,
            say=say,
            metadata={
                "steps": len(intents),
                "mode": "circle",
                "radius_m": float(data.get("radius", data.get("radius_m", 0.0))),
            },
        )
    raise CmuAreParseError(f"LLM returned unsupported intent type {typ!r}")


def _circle_task_to_intents(
    data: dict[str, Any],
    *,
    instruction: str,
    current_pose: dict[str, float] | None,
    max_circle_radius_m: float,
    circle_segments: int,
) -> tuple[CmuIntent, ...]:
    radius = abs(float(data.get("radius", data.get("radius_m", 0.0))))
    if radius <= 0:
        raise CmuAreParseError("circle radius must be positive")
    if radius > max_circle_radius_m:
        raise CmuAreParseError(
            f"circle radius {radius:.3f}m exceeds safety limit {max_circle_radius_m:.3f}m"
        )
    segments = int(data.get("segments", circle_segments) or circle_segments)
    segments = max(6, min(segments, circle_segments))
    base = current_pose or {"x": 0.0, "y": 0.0}
    cx = float(data.get("center_x", float(base.get("x", 0.0))))
    cy = float(data.get("center_y", float(base.get("y", 0.0))))
    start_angle = math.atan2(float(base.get("y", cy)) - cy, float(base.get("x", cx)) - cx)
    if math.hypot(float(base.get("x", cx)) - cx, float(base.get("y", cy)) - cy) < 0.1:
        start_angle = 0.0
    clockwise = bool(data.get("clockwise", False))
    direction = -1.0 if clockwise else 1.0
    intents: list[CmuIntent] = []
    for index in range(1, segments + 1):
        angle = start_angle + direction * 2.0 * math.pi * index / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        intents.append(
            CmuIntent(
                type="absolute",
                instruction=f"{instruction} / circle {index}/{segments}",
                x=x,
                y=y,
                source="llm",
            )
        )
    return tuple(intents)


def _is_cancel_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered or word in text for word in ("取消", "中止", "停下", "急停", "cancel", "abort"))


def make_cmu_artifact_dir(
    *, output_dir: str | Path = "practice_data/app_runs", app: str = "cmu"
) -> Path:
    episode_id = f"app_{app}_{int(time.time() * 1000)}"
    artifact_dir = Path(output_dir).expanduser() / episode_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def write_cmu_artifact(
    *,
    artifact_dir: str | Path,
    instruction: str,
    intent: CmuIntent,
    status: str,
    duration_sec: float,
    final_pose: Optional[dict[str, float]],
    distance_to_goal: Optional[float],
    bridge: CmuAreBridge,
    ros_topics: list[str] | None = None,
    task: dict[str, Any] | None = None,
    task_events: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> CmuRunResult:
    artifact_path = Path(artifact_dir).expanduser()
    artifact_path.mkdir(parents=True, exist_ok=True)
    result = CmuRunResult(
        episode_id=artifact_path.name,
        instruction=instruction,
        intent=intent,
        status=status,
        artifact_dir=str(artifact_path),
        duration_sec=duration_sec,
        final_pose=final_pose,
        distance_to_goal=distance_to_goal,
        odom_trace_count=len(bridge.odom_trace),
        cmd_vel_count=len(bridge.cmd_trace),
        path_count=len(bridge.path_trace),
        waypoint_count=len(bridge.waypoint_trace),
    )
    summary = result.to_dict()
    if task is not None:
        summary["task"] = task
    if error:
        summary["error"] = error
    (artifact_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if task_events is not None:
        _write_jsonl(artifact_path / "task_events.jsonl", task_events)
    _write_jsonl(artifact_path / "odom_trace.jsonl", bridge.odom_trace)
    _write_jsonl(artifact_path / "cmd_vel.jsonl", bridge.cmd_trace)
    _write_jsonl(artifact_path / "path_trace.jsonl", bridge.path_trace)
    _write_jsonl(artifact_path / "waypoints.jsonl", bridge.waypoint_trace)
    if ros_topics is not None:
        (artifact_path / "ros_topics.txt").write_text(
            "\n".join(sorted(ros_topics)) + "\n",
            encoding="utf-8",
        )
    return result


def describe_unsupported_cmu_command(instruction: str) -> str | None:
    """Return a user-facing capability boundary message for clearly unsupported motion."""

    lowered = instruction.strip().lower()
    if not lowered:
        return None
    if any(pattern in lowered for pattern in _UNSUPPORTED_MOTION_PATTERNS):
        return _UNSUPPORTED_MOTION_HINT
    return None


def build_cmu_result_message(
    result: CmuRunResult,
    *,
    prefix: str = "",
    error: str | None = None,
    use_llm: bool = True,
) -> str:
    """Build a concise Chinese status message from real execution facts."""

    facts = build_cmu_result_facts(result, error=error)
    fallback = _default_cmu_result_message(facts)
    if not use_llm:
        return fallback
    try:
        polished = _polish_cmu_result_message(facts, fallback=fallback, prefix=prefix)
    except Exception:  # noqa: BLE001 - chat UX should never fail because polishing failed
        return fallback
    return polished or fallback


def build_cmu_result_facts(
    result: CmuRunResult,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """Create a structured, non-hallucinatory summary for result messaging."""

    intent = result.intent
    facts: dict[str, Any] = {
        "instruction": result.instruction,
        "intent_type": intent.type,
        "status": result.status,
        "ok": result.ok,
        "artifact_dir": result.artifact_dir,
        "duration_sec": round(result.duration_sec, 3),
    }
    if error:
        facts["error"] = error
    if intent.command:
        facts["command"] = intent.command
        facts["command_label"] = _explore_command_label(intent.command)
    if intent.place:
        facts["place"] = intent.place
    if intent.x is not None and intent.y is not None:
        facts["target"] = {"x": round(intent.x, 3), "y": round(intent.y, 3)}
    if intent.dx is not None or intent.dy is not None:
        dx = float(intent.dx or 0.0)
        dy = float(intent.dy or 0.0)
        facts["relative"] = {
            "dx": round(dx, 3),
            "dy": round(dy, 3),
            "distance_m": round(math.hypot(dx, dy), 3),
            "direction": _relative_direction_label(dx, dy),
        }
    if result.distance_to_goal is not None:
        facts["distance_to_goal_m"] = round(float(result.distance_to_goal), 3)
    if result.final_pose is not None:
        facts["final_pose"] = {
            key: round(float(value), 3)
            for key, value in result.final_pose.items()
            if isinstance(value, (int, float))
        }
    return facts


def _task_start_message(task: CmuChatTask) -> str:
    if task.say:
        text = task.say.strip()
        if text.startswith("开始"):
            return text
    if task.kind == "explore_control":
        return "开始" + _explore_command_label(task.command or "探索控制")
    if task.kind == "sequence" and (task.metadata or {}).get("mode") == "circle":
        radius = (task.metadata or {}).get("radius_m")
        return f"开始执行半径 {_format_distance(radius)} 的圆形轨迹。"
    if task.kind == "sequence":
        return f"开始执行 {len(task.intents)} 步移动任务。"
    if task.intents:
        return _intent_start_message(task.intents[0])
    return "开始执行任务。"


def _intent_start_message(intent: CmuIntent) -> str:
    if intent.type == "place":
        return f"开始前往 {intent.place or '目标地点'}。"
    if intent.type == "relative":
        dx = float(intent.dx or 0.0)
        dy = float(intent.dy or 0.0)
        return f"开始{_relative_direction_label(dx, dy)}移动 {_format_distance(math.hypot(dx, dy))}。"
    if intent.x is not None and intent.y is not None:
        return f"开始前往坐标 x={intent.x:.2f}, y={intent.y:.2f}。"
    return "开始移动。"


def _task_progress_message(
    task: CmuChatTask,
    *,
    step: int | None = None,
    total: int | None = None,
    distance_to_goal: Any = None,
) -> str:
    if task.kind == "explore_control":
        return f"正在下发{_explore_command_label(task.command or '探索控制')}指令。"
    prefix = "正在执行移动任务"
    if step is not None and total is not None and total > 1:
        prefix = f"正在执行第 {step}/{total} 个目标点"
    elif task.kind == "sequence":
        prefix = "正在执行轨迹任务"
    if distance_to_goal is not None:
        return f"{prefix}，距离当前目标约 {_format_distance(distance_to_goal)}。"
    return prefix + "。"


def _task_end_message(task: CmuChatTask, result: CmuRunResult, *, error: str | None = None) -> str:
    if result.status == "cancelled":
        return "当前任务已取消，车辆已进入待命状态。"
    if task.kind == "explore_control":
        return build_cmu_result_message(result, error=error, use_llm=True)
    if task.kind == "sequence":
        mode = (task.metadata or {}).get("mode")
        label = "圆形轨迹" if mode == "circle" else "多步移动任务"
        if result.ok:
            message = f"{label}成功"
            if result.distance_to_goal is not None:
                message += f"，最终误差 {_format_distance(result.distance_to_goal)}"
            return message + f"。记录：{result.artifact_dir}"
        reason = error or f"状态为 {result.status}"
        if result.distance_to_goal is not None:
            reason += f"，最终误差 {_format_distance(result.distance_to_goal)}"
        return f"{label}失败：{reason}。记录：{result.artifact_dir}"
    return build_cmu_result_message(result, error=error, use_llm=True) + f" 记录：{result.artifact_dir}"


def _parse_explore_command(text: str) -> Optional[str]:
    lowered = text.lower()
    for command, aliases in _EXPLORE_COMMANDS.items():
        for alias in aliases:
            if alias.lower() in lowered or alias in text:
                return command
    return None


def _explore_command_label(command: str) -> str:
    labels = {
        "start": "开始探索",
        "pause": "暂停探索",
        "resume": "继续探索",
        "stop": "停止探索",
    }
    return labels.get(command, command)


def _relative_direction_label(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "向上" if dx >= 0 else "向下"
    return "向左" if dy >= 0 else "向右"


def _default_cmu_result_message(facts: dict[str, Any]) -> str:
    ok = bool(facts.get("ok"))
    status = str(facts.get("status", "unknown"))
    intent_type = str(facts.get("intent_type", "unknown"))
    error = str(facts.get("error", "")).strip()
    failure = error or f"状态为 {status}"

    if intent_type == "explore_control":
        command = str(facts.get("command", ""))
        if ok:
            if command == "start":
                return "已成功开始探索，ARiADNE2 正在接管目标点规划。"
            if command == "pause":
                return "已成功暂停探索，车辆已进入待命状态。"
            if command == "resume":
                return "已成功继续探索，ARiADNE2 将从当前状态恢复规划。"
            if command == "stop":
                return "已成功停止探索，车辆已进入安全停止状态。"
            label = str(facts.get("command_label", command))
            return f"已成功执行{label}。"
        label = str(facts.get("command_label", command or "探索控制"))
        return f"{label}失败：{failure}。"

    if intent_type == "relative":
        relative = facts.get("relative", {})
        direction = str(relative.get("direction", "相对方向")) if isinstance(relative, dict) else "相对方向"
        distance = relative.get("distance_m") if isinstance(relative, dict) else None
        distance_text = _format_distance(distance)
        if ok:
            message = f"已成功{direction}移动 {distance_text}"
            if "distance_to_goal_m" in facts:
                message += f"，最终误差 {_format_distance(facts['distance_to_goal_m'])}"
            return message + "。"
        return f"{direction}移动 {distance_text} 失败：{failure}。"

    if intent_type == "place":
        place = str(facts.get("place", "目标地点"))
        if ok:
            message = f"已成功到达 {place}"
            if "distance_to_goal_m" in facts:
                message += f"，最终误差 {_format_distance(facts['distance_to_goal_m'])}"
            return message + "。"
        return f"前往 {place} 失败：{failure}。"

    target = facts.get("target")
    if isinstance(target, dict):
        target_text = f"x={target.get('x')}, y={target.get('y')}"
    else:
        target_text = "目标点"
    if ok:
        message = f"已成功到达{target_text}"
        if "distance_to_goal_m" in facts:
            message += f"，最终误差 {_format_distance(facts['distance_to_goal_m'])}"
        return message + "。"
    return f"前往{target_text}失败：{failure}。"


def _format_distance(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "未知距离"
    text = f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{text} 米"


def _polish_cmu_result_message(
    facts: dict[str, Any],
    *,
    fallback: str,
    prefix: str = "",
) -> str:
    payload = {
        "facts": facts,
        "fallback_message": fallback,
        "llm_pre_execution_message": prefix,
    }
    system_prompt = (
        "你是 ROSClaw CMU 仿真控制台的中文状态播报器。"
        "你只能基于用户给出的 JSON facts 改写一句简短状态消息，不能增加事实，不能承诺未发生的动作。"
        "如果 facts.ok 为 false，必须保留失败原因。"
        "如果是探索控制，必须明确是开始、暂停、继续或停止探索。"
        "如果是相对移动，必须保留方向、距离和已有最终误差。"
        "返回 exactly one JSON object: {\"message\":\"...\"}。"
    )
    data = _llm_api_request(
        system_prompt,
        json.dumps(payload, ensure_ascii=False, default=str),
        max_tokens=160,
    )
    message = str(data.get("message", "")).strip()
    if not message:
        return fallback
    if len(message) > 160:
        return fallback
    return message


def _parse_relative_move(
    text: str,
    *,
    max_relative_m: float,
    current_pose: dict[str, float] | None = None,
) -> Optional[tuple[float, float]]:
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    value = abs(float(match.group(1)))
    if value > max_relative_m:
        raise CmuAreParseError(
            f"relative move {value:.3f}m exceeds safety limit {max_relative_m:.3f}m"
        )
    lowered = text.lower()
    direction_patterns = (
        ("up_left", ("左上", "向左上", "往左上", "up left", "upper left")),
        ("up_right", ("右上", "向右上", "往右上", "up right", "upper right")),
        ("down_left", ("左下", "向左下", "往左下", "down left", "lower left")),
        ("down_right", ("右下", "向右下", "往右下", "down right", "lower right")),
        ("forward_left", ("左前", "向左前", "往左前", "forward left")),
        ("forward_right", ("右前", "向右前", "往右前", "forward right")),
        ("backward_left", ("左后", "向左后", "往左后", "backward left")),
        ("backward_right", ("右后", "向右后", "往右后", "backward right")),
        ("forward", ("前进", "向前", "往前", "forward")),
        ("backward", ("后退", "向后", "往后", "backward", "back")),
        ("up", ("向上", "往上", "上走", "up")),
        ("down", ("向下", "往下", "下走", "down")),
        ("right", ("向右", "往右", "右走", "right")),
        ("left", ("向左", "往左", "左走", "left")),
    )
    for direction, patterns in direction_patterns:
        if any(pattern in lowered or pattern in text for pattern in patterns):
            return _direction_to_delta(direction, value, max_relative_m=max_relative_m, current_pose=current_pose)
    return None


def _llm_api_request(system_prompt: str, instruction: str, *, max_tokens: int = 220) -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise CmuAreParseError("LLM requested but no DEEPSEEK_API_KEY/OPENAI_API_KEY is set")

    import urllib.request

    provider = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "openai"
    base_url = (
        os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        if provider == "deepseek"
        else os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    )
    model = (
        os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
        if provider == "deepseek"
        else os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    raw = urllib.request.urlopen(request, timeout=30).read()
    body = json.loads(raw)
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def _parse_with_llm(instruction: str) -> dict[str, Any]:
    system_prompt = (
        "You convert robot mobility commands into one JSON object only. "
        "Allowed schemas: "
        '{"type":"place","place":"entrance_a"}, '
        '{"type":"relative","direction":"up|down|left|right","distance":3.0}, '
        '{"type":"explore_control","command":"start|pause|resume|stop"}. '
        "Use screen directions: up/down/left/right. "
        "Do not output free-form text."
    )
    return _llm_api_request(system_prompt, instruction, max_tokens=160)


def _parse_with_llm_chat(
    instruction: str,
    *,
    places: dict[str, CmuPlace],
    current_pose: dict[str, float] | None,
    max_relative_m: float,
    max_sequence_steps: int = DEFAULT_CMU_MAX_SEQUENCE_STEPS,
    max_circle_radius_m: float = DEFAULT_CMU_MAX_CIRCLE_RADIUS,
) -> dict[str, Any]:
    place_names = ", ".join(sorted(places))
    current = (
        f"x={current_pose.get('x', 0.0):.2f}, y={current_pose.get('y', 0.0):.2f}"
        if current_pose
        else "unknown"
    )
    system_prompt = (
        "You are a Chinese robot mobility command parser for a ROS1 CMU ARE demo. "
        "Return exactly one JSON object. Do not output markdown. "
        "If the command is clear, use status=action and include a short Chinese say field. "
        "If the command is ambiguous, use status=clarify with a Chinese question and do not include an action. "
        "Allowed action schemas: "
        '{"status":"action","type":"place","place":"inspection_a","say":"收到，正在前往 inspection_a。"}, '
        '{"status":"action","type":"relative","direction":"up","distance":3.0,"say":"收到，向上移动3米。"}, '
        '{"status":"action","type":"relative","direction":"forward","distance":3.0,"say":"收到，前进3米。"}, '
        '{"status":"action","type":"absolute","x":1.0,"y":2.0,"say":"收到，前往坐标点。"}, '
        '{"status":"action","type":"sequence","steps":[{"type":"relative","direction":"up","distance":5.0},{"type":"relative","direction":"right","distance":3.0}],"say":"开始执行多步移动任务。"}, '
        '{"status":"action","type":"circle","radius":2.0,"clockwise":false,"say":"开始执行圆形轨迹。"}, '
        '{"status":"action","type":"explore_control","command":"start|pause|resume|stop","say":"收到。"}, '
        '{"status":"action","type":"cancel","say":"收到，取消当前任务。"}, '
        '{"status":"clarify","question":"你想去哪个地点？"}. '
        f"Known places are: {place_names}. "
        "Relative directions may be screen directions up, down, left, right, up_left, up_right, down_left, down_right, "
        "or robot-frame directions forward, backward, forward_left, forward_right, backward_left, backward_right. "
        "For Chinese screen commands, 上=up, 下=down, 左=left, 右=right. "
        "For Chinese robot-frame commands, 前进=forward, 后退=backward, 左前=forward_left, 右前=forward_right. "
        f"Relative distance must be <= {max_relative_m:.1f} meters. "
        f"Sequence steps must be <= {max_sequence_steps}. "
        f"Circle radius must be <= {max_circle_radius_m:.1f} meters; circle means a waypoint polygon around current pose, not direct cmd_vel control. "
        f"Absolute coordinates must be within ±{DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE:.1f} meters. "
        f"Current robot pose is {current}. "
        "Do not reject a target only because it may be unreachable; execution will report success or failure. "
        "Never invent unknown places, ROS topics, cmd_vel commands, speed curves, or raw velocity control."
    )
    return _llm_api_request(system_prompt, instruction, max_tokens=220)


def _validate_llm_intent(
    data: dict[str, Any],
    *,
    instruction: str,
    places: dict[str, CmuPlace],
    current_pose: dict[str, float] | None,
    max_relative_m: float,
    max_absolute_coordinate_m: float | None = None,
) -> CmuIntent:
    if not isinstance(data, dict):
        raise CmuAreParseError("LLM returned non-object intent")
    typ = data.get("type")
    if typ == "place":
        place_name = str(data.get("place", ""))
        if place_name not in places:
            raise CmuAreParseError(f"LLM returned unknown place {place_name!r}")
        place = places[place_name]
        return CmuIntent(
            type="place",
            instruction=instruction,
            x=place.x,
            y=place.y,
            z=place.z,
            frame_id=place.frame_id,
            place=place.name,
            source="llm",
        )
    if typ == "relative":
        if "direction" in data:
            dx, dy = _direction_to_delta(
                str(data.get("direction", "")),
                float(data.get("distance", 0.0)),
                max_relative_m=max_relative_m,
                current_pose=current_pose,
            )
        else:
            dx = float(data.get("dx", 0.0))
            dy = float(data.get("dy", 0.0))
        distance = math.hypot(dx, dy)
        if distance > max_relative_m:
            raise CmuAreParseError(
                f"LLM relative move {distance:.3f}m exceeds safety limit {max_relative_m:.3f}m"
            )
        base = current_pose or {"x": 0.0, "y": 0.0}
        return CmuIntent(
            type="relative",
            instruction=instruction,
            x=float(base.get("x", 0.0)) + dx,
            y=float(base.get("y", 0.0)) + dy,
            dx=dx,
            dy=dy,
            source="llm",
        )
    if typ == "explore_control":
        command = str(data.get("command", "")).lower()
        if command not in {"start", "pause", "resume", "stop"}:
            raise CmuAreParseError(f"LLM returned invalid exploration command {command!r}")
        return CmuIntent(
            type="explore_control",
            instruction=instruction,
            command=command,
            source="llm",
        )
    if typ == "absolute":
        # Extract and validate coordinates
        x_val = data.get("x")
        y_val = data.get("y")

        # Check for missing required fields
        if x_val is None:
            raise CmuAreParseError("absolute intent missing required field 'x'")
        if y_val is None:
            raise CmuAreParseError("absolute intent missing required field 'y'")

        try:
            x = float(x_val)
            y = float(y_val)
            z = float(data.get("z", 0.0))
        except (ValueError, TypeError) as e:
            raise CmuAreParseError(f"absolute intent has invalid coordinate values: {e}")

        return _absolute_intent(
            instruction=instruction,
            x=x,
            y=y,
            z=z,
            frame_id=str(data.get("frame_id", "map")),
            source="llm",
            max_absolute_coordinate_m=max_absolute_coordinate_m,
        )
    raise CmuAreParseError(f"LLM returned unsupported intent type {typ!r}")


def _direction_to_delta(
    direction: str,
    distance: float,
    *,
    max_relative_m: float,
    current_pose: dict[str, float] | None = None,
) -> tuple[float, float]:
    value = abs(float(distance))
    if value <= 0:
        raise CmuAreParseError("relative move distance must be positive")
    if value > max_relative_m:
        raise CmuAreParseError(
            f"relative move {value:.3f}m exceeds safety limit {max_relative_m:.3f}m"
        )
    normalized = direction.strip().lower().replace("-", "_").replace(" ", "_")
    screen_vectors = {
        "up": (1.0, 0.0),
        "向上": (1.0, 0.0),
        "上": (1.0, 0.0),
        "down": (-1.0, 0.0),
        "向下": (-1.0, 0.0),
        "下": (-1.0, 0.0),
        "left": (0.0, 1.0),
        "向左": (0.0, 1.0),
        "左": (0.0, 1.0),
        "right": (0.0, -1.0),
        "向右": (0.0, -1.0),
        "右": (0.0, -1.0),
        "up_left": (1.0, 1.0),
        "左上": (1.0, 1.0),
        "up_right": (1.0, -1.0),
        "右上": (1.0, -1.0),
        "down_left": (-1.0, 1.0),
        "左下": (-1.0, 1.0),
        "down_right": (-1.0, -1.0),
        "右下": (-1.0, -1.0),
    }
    if normalized in screen_vectors:
        return _scale_unit_vector(*screen_vectors[normalized], distance=value)
    robot_vectors = {
        "forward": (1.0, 0.0),
        "前": (1.0, 0.0),
        "前进": (1.0, 0.0),
        "backward": (-1.0, 0.0),
        "back": (-1.0, 0.0),
        "后": (-1.0, 0.0),
        "后退": (-1.0, 0.0),
        "forward_left": (1.0, 1.0),
        "left_forward": (1.0, 1.0),
        "左前": (1.0, 1.0),
        "forward_right": (1.0, -1.0),
        "right_forward": (1.0, -1.0),
        "右前": (1.0, -1.0),
        "backward_left": (-1.0, 1.0),
        "left_backward": (-1.0, 1.0),
        "左后": (-1.0, 1.0),
        "backward_right": (-1.0, -1.0),
        "right_backward": (-1.0, -1.0),
        "右后": (-1.0, -1.0),
    }
    if normalized in robot_vectors:
        local_x, local_y = _scale_unit_vector(*robot_vectors[normalized], distance=value)
        theta = float((current_pose or {}).get("theta", 0.0))
        dx = math.cos(theta) * local_x - math.sin(theta) * local_y
        dy = math.sin(theta) * local_x + math.cos(theta) * local_y
        return (dx, dy)
    raise CmuAreParseError(f"unsupported relative direction {direction!r}")


def _scale_unit_vector(x: float, y: float, *, distance: float) -> tuple[float, float]:
    norm = math.hypot(x, y)
    if norm <= 0:
        raise CmuAreParseError("relative direction vector cannot be zero")
    return (distance * x / norm, distance * y / norm)


def _rospack_has(rospack: Any, package: str) -> bool:
    try:
        rospack.get_path(package)
    except Exception:  # noqa: BLE001
        return False
    return True


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _point_msg_to_dict(msg: Any) -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "frame_id": msg.header.frame_id,
        "x": float(msg.point.x),
        "y": float(msg.point.y),
        "z": float(msg.point.z),
    }


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def _append_env_path(current: str, paths: list[str]) -> str:
    values = [path for path in paths if path]
    values.extend(part for part in current.split(":") if part)
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return ":".join(deduped)


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
