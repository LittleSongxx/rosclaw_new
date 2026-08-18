"""Daemon-owned CMU ARE ROS1 adapter over rosbridge.

This module deliberately contains no ``rospy`` import.  The only southbound
surface is a small allow-listed set of CMU ARE topic operations.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from rosclaw.connectors.ros.transport import RosbridgeTransport

REQUIRED_TOPICS = {
    "/way_point": "geometry_msgs/PointStamped",
    "/speed": "std_msgs/Float32",
    "/stop": "std_msgs/Int8",
    "/rosclaw/exploration_control": "std_msgs/String",
    "/rosclaw/exploration_state": "std_msgs/String",
    "/state_estimation": "nav_msgs/Odometry",
    "/cmd_vel": "geometry_msgs/TwistStamped",
    "/registered_scan": "sensor_msgs/PointCloud2",
    "/terrain_map": "sensor_msgs/PointCloud2",
}


class CmuAreTransportError(RuntimeError):
    """Raised when the rosbridge worker cannot satisfy a bounded operation."""


@dataclass(frozen=True)
class CmuAreConnection:
    connection_id: str
    generation: int


class CmuAreRosbridgeAdapter:
    """Bounded CMU ARE topic adapter owned by rosclawd."""

    def __init__(
        self,
        transport: RosbridgeTransport,
        *,
        daemon_owner_id: str = "daemon_cmu_are_sim",
    ) -> None:
        if not daemon_owner_id.startswith("daemon_"):
            raise ValueError("daemon_owner_id must identify rosclawd")
        self.transport = transport
        self.daemon_owner_id = daemon_owner_id
        self._lock = threading.RLock()
        self._connection: CmuAreConnection | None = None
        self._generation_counter = 0
        self._transport_marker: int | None = None
        self._advertised: set[str] = set()
        self.odom_trace: list[dict[str, Any]] = []
        self.cmd_trace: list[dict[str, Any]] = []
        self.path_trace: list[dict[str, Any]] = []
        self.waypoint_trace: list[dict[str, Any]] = []
        self.last_check: dict[str, Any] | None = None

    @property
    def connection(self) -> CmuAreConnection | None:
        return self._connection

    def connect(self) -> CmuAreConnection:
        with self._lock:
            result = self.transport.connect()
            if not result.ok:
                self._invalidate()
                raise CmuAreTransportError(result.error or "rosbridge connection failed")
            transport_marker = self._current_transport_marker()
            if self._connection is None or (
                self._transport_marker is not None
                and transport_marker is not None
                and transport_marker != self._transport_marker
            ):
                self._generation_counter += 1
                self._connection = CmuAreConnection(
                    connection_id=f"cmu_rosbridge_{uuid.uuid4().hex}",
                    generation=self._generation_counter,
                )
            self._transport_marker = transport_marker
            return self._connection

    def _invalidate(self) -> None:
        with self._lock:
            self._connection = None
            self._transport_marker = None
            self._advertised.clear()

    def _current_transport_marker(self) -> int | None:
        websocket = getattr(self.transport, "_ws", None)
        return id(websocket) if websocket is not None else None

    def _transport_is_connected(self) -> bool | None:
        websocket = getattr(self.transport, "_ws", None)
        if websocket is not None:
            return bool(getattr(websocket, "connected", False))
        connected = getattr(self.transport, "connected", None)
        return connected if isinstance(connected, bool) else None

    def _is_observation_timeout(self, error: str | None) -> bool:
        message = str(error or "").casefold()
        disconnect_markers = (
            "broken pipe",
            "connection closed",
            "connection refused",
            "connection reset",
            "failed to connect",
            "not connected",
            "remote host was lost",
            "transport is closed",
        )
        if any(marker in message for marker in disconnect_markers):
            return False
        if self._transport_is_connected() is False:
            return False
        return any(
            marker in message
            for marker in (
                "no message received",
                "no mock response queued",
                "timed out",
                "timeout",
            )
        )

    def _send(self, operation: str, callback) -> Any:
        self.connect()
        try:
            return callback()
        except Exception as exc:  # noqa: BLE001 - convert and invalidate
            self._invalidate()
            raise CmuAreTransportError(f"CMU ARE {operation} failed: {exc}") from exc

    def check(self, *, timeout_sec: float = 5.0) -> dict[str, Any]:
        result = self._send(
            "topic discovery",
            lambda: self.transport.call_service("/rosapi/topics", {}, timeout_sec=timeout_sec),
        )
        if not result.ok:
            self._invalidate()
            raise CmuAreTransportError(result.error or "rosapi topic discovery failed")
        values = (result.data or {}).get("values", {})
        topics = values.get("topics", []) if isinstance(values, dict) else []
        types = values.get("types", []) if isinstance(values, dict) else []
        topic_types = {
            str(topic): str(message_type)
            for topic, message_type in zip(topics, types, strict=False)
        }
        missing = sorted(topic for topic in REQUIRED_TOPICS if topic not in topic_types)
        report = {
            "ok": not missing,
            "connection": self.connection.__dict__ if self.connection else None,
            "topics": topic_types,
            "required_topics": REQUIRED_TOPICS,
            "missing_topics": missing,
        }
        self.last_check = report
        return report

    def _advertise(self, topic: str, message_type: str, *, latch: bool = False) -> None:
        if topic in self._advertised:
            return
        result = self.transport.advertise(topic, message_type, latch=latch, queue_size=5)
        if not result.ok:
            raise CmuAreTransportError(result.error or f"failed to advertise {topic}")
        self._advertised.add(topic)

    def _publish(
        self, topic: str, message_type: str, message: dict[str, Any], *, latch: bool = False
    ) -> None:
        def send() -> Any:
            self._advertise(topic, message_type, latch=latch)
            result = self.transport.publish(topic, message)
            if not result.ok:
                raise CmuAreTransportError(result.error or f"failed to publish {topic}")
            return result

        self._send(topic, send)

    def publish_waypoint(self, target: dict[str, Any]) -> None:
        self._publish(
            "/way_point",
            "geometry_msgs/PointStamped",
            {
                "header": {"frame_id": str(target["frame_id"])},
                "point": {
                    "x": float(target["x"]),
                    "y": float(target["y"]),
                    "z": float(target.get("z", 0.0)),
                },
            },
        )
        self.waypoint_trace.append(
            {
                "timestamp": time.time(),
                "frame_id": target["frame_id"],
                "x": float(target["x"]),
                "y": float(target["y"]),
                "z": float(target.get("z", 0.0)),
            }
        )

    def publish_speed(self, speed_mps: float) -> None:
        self._publish(
            "/speed",
            "std_msgs/Float32",
            {"data": float(speed_mps)},
            latch=True,
        )

    def publish_stop(self, stopped: bool = True) -> None:
        self._publish(
            "/stop",
            "std_msgs/Int8",
            {"data": 1 if stopped else 0},
            latch=True,
        )

    def publish_exploration_control(self, command: str) -> None:
        if command not in {"start", "pause", "resume", "stop"}:
            raise CmuAreTransportError(f"unsupported exploration command: {command}")
        self._publish(
            "/rosclaw/exploration_control",
            "std_msgs/String",
            {"data": command},
            latch=True,
        )

    def _subscribe_once(
        self, topic: str, message_type: str, timeout_sec: float
    ) -> dict[str, Any] | None:
        result = self._send(
            topic,
            lambda: self.transport.subscribe_once(
                topic,
                msg_type=message_type,
                timeout_sec=timeout_sec,
            ),
        )
        if not result.ok:
            if self._is_observation_timeout(result.error):
                return None
            # A broken receive invalidates the connection generation.  An
            # executor must never continue using observations from the old
            # rosbridge session after this point.
            self._invalidate()
            raise CmuAreTransportError(result.error or f"failed to observe {topic}")
        data = result.data or {}
        message = data.get("msg") if isinstance(data, dict) else None
        return message if isinstance(message, dict) else None

    @staticmethod
    def _parse_odom(message: dict[str, Any]) -> dict[str, float] | None:
        pose = message.get("pose", {}).get("pose", {})
        twist = message.get("twist", {}).get("twist", {})
        position = pose.get("position", {})
        linear = twist.get("linear", {})
        try:
            values = {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position.get("z", 0.0)),
                "linear_x": float(linear.get("x", 0.0)),
                "linear_y": float(linear.get("y", 0.0)),
                "timestamp": time.time(),
            }
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values.values()):
            return None
        return values

    def read_odom(self, *, timeout_sec: float = 2.0) -> dict[str, float] | None:
        message = self._subscribe_once("/state_estimation", "nav_msgs/Odometry", timeout_sec)
        if message is None:
            return None
        parsed = self._parse_odom(message)
        if parsed is not None:
            self.odom_trace.append(parsed)
        return parsed

    def read_cmd_vel(self, *, timeout_sec: float = 1.0) -> dict[str, float] | None:
        message = self._subscribe_once("/cmd_vel", "geometry_msgs/TwistStamped", timeout_sec)
        if message is None:
            return None
        twist = message.get("twist", {})
        linear = twist.get("linear", {})
        angular = twist.get("angular", {})
        try:
            parsed = {
                "linear_x": float(linear.get("x", 0.0)),
                "linear_y": float(linear.get("y", 0.0)),
                "angular_z": float(angular.get("z", 0.0)),
                "timestamp": time.time(),
            }
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in parsed.values()):
            return None
        self.cmd_trace.append(parsed)
        return parsed

    def read_exploration_state(self, *, timeout_sec: float = 1.0) -> str | None:
        """Read one ARiADNE2 lifecycle state when the simulator exposes it.

        Different historical ARE launch files used different names.  The
        adapter accepts only these fixed, known topics and never accepts a
        caller-provided topic string.
        """

        topics = (
            "/rosclaw/exploration_state",
            "/exploration_state",
            "/ariadne2/state",
        )
        deadline = time.monotonic() + max(0.0, timeout_sec)
        for index, topic in enumerate(topics):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            topics_left = len(topics) - index
            message = self._subscribe_once(
                topic,
                "std_msgs/String",
                max(0.05, remaining / topics_left),
            )
            if message is None:
                continue
            for key in ("data", "state", "status", "mode"):
                value = message.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().casefold()
        return None

    def navigate(
        self,
        *,
        target: dict[str, Any],
        speed_mps: float,
        tolerance_m: float,
        timeout_sec: float,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        start = self.read_odom(timeout_sec=min(3.0, timeout_sec))
        if start is None:
            return {"status": "degraded", "error": "no fresh odometry before dispatch"}
        self.publish_speed(speed_mps)
        self.publish_stop(False)
        self.publish_waypoint(target)
        deadline = time.monotonic() + timeout_sec
        final = start
        distance: float | None = None
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                self.publish_stop(True)
                return {
                    "status": "cancelled",
                    "start_pose": start,
                    "final_pose": final,
                    "distance_to_goal": distance,
                }
            current = self.read_odom(timeout_sec=min(1.0, max(0.1, deadline - time.monotonic())))
            if current is None:
                continue
            final = current
            distance = math.hypot(
                float(target["x"]) - current["x"], float(target["y"]) - current["y"]
            )
            if distance <= tolerance_m:
                self.publish_stop(True)
                stopped = self.wait_for_stop(timeout_sec=min(3.0, timeout_sec))
                return {
                    "status": "success" if stopped else "degraded",
                    "start_pose": start,
                    "final_pose": final,
                    "distance_to_goal": distance,
                    "stop_confirmed": stopped,
                }
        self.publish_stop(True)
        stopped = self.wait_for_stop(timeout_sec=2.0)
        return {
            "status": "timeout",
            "start_pose": start,
            "final_pose": final,
            "distance_to_goal": distance,
            "stop_confirmed": stopped,
        }

    def wait_for_stop(self, *, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            command = self.read_cmd_vel(timeout_sec=min(0.5, deadline - time.monotonic()))
            if command is None:
                # A quiet command stream after a latched stop is not enough to
                # claim task verification, but it is useful as a degraded stop
                # signal when the simulator does not publish zero Twist frames.
                continue
            if (
                abs(command["linear_x"]) <= 1e-3
                and abs(command["linear_y"]) <= 1e-3
                and abs(command["angular_z"]) <= 1e-3
            ):
                return True
        return False

    def exploration_control(
        self,
        command: str,
        *,
        speed_mps: float,
        timeout_sec: float = 3.0,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + max(0.1, timeout_sec)
        state_deadline = (
            deadline if command in {"start", "resume"} else started + max(0.05, timeout_sec * 0.6)
        )
        before = self.read_exploration_state(timeout_sec=min(0.5, timeout_sec / 3))
        if cancel_event is not None and cancel_event.is_set():
            return {
                "status": "cancelled",
                "command": command,
                "state_before": before,
                "state_after": before,
                "state_changed": False,
            }
        if command in {"start", "resume"}:
            self.publish_speed(speed_mps)
            self.publish_stop(False)
        else:
            self.publish_stop(True)
            self.publish_speed(0.0)
        self.publish_exploration_control(command)
        after: str | None = before
        while time.monotonic() < state_deadline:
            if cancel_event is not None and cancel_event.is_set():
                self.publish_stop(True)
                return {
                    "status": "cancelled",
                    "command": command,
                    "state_before": before,
                    "state_after": after,
                    "state_changed": False,
                }
            remaining = state_deadline - time.monotonic()
            observed = self.read_exploration_state(timeout_sec=min(0.5, remaining))
            if observed is not None:
                after = observed
            if before is not None and after is not None and before != after:
                break
        state_changed = before is not None and after is not None and before != after
        remaining = max(0.0, deadline - time.monotonic())
        stop_confirmed = command in {"pause", "stop"} and self.wait_for_stop(timeout_sec=remaining)
        return {
            "status": "success" if state_changed else "timeout",
            "command": command,
            "stop_confirmed": stop_confirmed,
            "state_before": before,
            "state_after": after,
            "state_changed": state_changed,
        }

    def stop(self, *, timeout_sec: float = 2.0) -> dict[str, Any]:
        self.publish_exploration_control("stop")
        self.publish_speed(0.0)
        self.publish_stop(True)
        stopped = self.wait_for_stop(timeout_sec=max(0.1, timeout_sec))
        return {
            "status": "success" if stopped else "timeout",
            "stop_confirmed": stopped,
        }


__all__ = [
    "CmuAreConnection",
    "CmuAreRosbridgeAdapter",
    "CmuAreTransportError",
    "REQUIRED_TOPICS",
]
