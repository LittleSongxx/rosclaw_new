"""Daemon-owned SHADOW executor for the CMU ARE simulator.

The executor is intentionally small and boring: it validates an immutable
``ActionEnvelope``, calls the allow-listed rosbridge adapter, and persists a
bounded set of evidence files.  It never imports ``rospy`` and it never
registers a REAL executor.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.integrations.cmu_are.adapter import (
    CmuAreRosbridgeAdapter,
    CmuAreTransportError,
)
from rosclaw.integrations.cmu_are.contracts import (
    CMU_ARE_BODY_ID,
    CMU_ARE_EXPLORE_SCHEMA,
    CMU_ARE_NAV_SCHEMA,
    CMU_ARE_STOP_SCHEMA,
    CmuAreContractError,
    CmuAreSafetyContract,
    body_snapshot_hash,
    load_safety_contract,
)
from rosclaw.kernel import (
    AcknowledgementStage,
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceDomain,
    EvidenceLevel,
    ExecutionMode,
)

CMU_NAVIGATE_CAPABILITY = "cmu_are.navigate_to_waypoint"
CMU_EXPLORE_CAPABILITY = "cmu_are.exploration_control"
CMU_STOP_CAPABILITY = "cmu_are.stop"
CMU_CAPABILITIES = frozenset({CMU_NAVIGATE_CAPABILITY, CMU_EXPLORE_CAPABILITY, CMU_STOP_CAPABILITY})
CMU_MANIFEST_SCHEMA = "rosclaw.cmu_are.manifest.v1"

_FORBIDDEN_ARGUMENT_NAMES = frozenset(
    {
        "ros_topic",
        "ros_topics",
        "device_path",
        "driver",
        "cmd_vel",
        "/cmd_vel",
        "topic",
        "topic_name",
        "driver_name",
    }
)
_NAV_KEYS = frozenset(
    {"schema_version", "target", "speed_mps", "tolerance_m", "timeout_sec", "expected_effect"}
)
_EXPLORE_KEYS = frozenset(
    {"schema_version", "command", "speed_mps", "timeout_sec", "expected_effect"}
)
_STOP_KEYS = frozenset({"schema_version", "timeout_sec", "expected_effect"})
_ACTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _safe_child(root: Path, child: str) -> Path:
    candidate = (root / child).resolve()
    candidate.relative_to(root)
    return candidate


class CmuAreShadowExecutor:
    """Execute one CMU ARE action in SHADOW mode only."""

    def __init__(
        self,
        adapter: CmuAreRosbridgeAdapter | Any | None = None,
        *,
        safety: CmuAreSafetyContract | None = None,
        home: str | Path | None = None,
        pack_version: str = "0.1.0",
        asset_manifest: str | Path | None = None,
        connection_factory: Callable[[], CmuAreRosbridgeAdapter] | None = None,
    ) -> None:
        self.safety = safety or load_safety_contract()
        self.home = resolve_home(str(home) if home is not None else None).resolve()
        self.pack_version = pack_version
        self.asset_manifest = (
            Path(asset_manifest).expanduser().resolve()
            if asset_manifest is not None
            else self._default_asset_manifest()
        )
        self._connection_factory = connection_factory
        self.adapter = adapter
        self.worker_id = os.environ.get("ROSCLAW_CMU_ARE_WORKER_ID", "cmu-are-shadow")
        self.cancel_event = threading.Event()
        self._cancel_lock = threading.Lock()
        self._active_action_id: str | None = None

    @staticmethod
    def _default_asset_manifest() -> Path:
        return Path(__file__).resolve().parents[4] / "docs/assets/cmu-are-assets.yaml"

    @property
    def expected_body_snapshot_hash(self) -> str:
        return body_snapshot_hash(
            safety=self.safety,
            asset_manifest=self.asset_manifest,
            pack_version=self.pack_version,
        )

    def _get_adapter(self) -> Any:
        if self.adapter is not None:
            return self.adapter
        if self._connection_factory is not None:
            self.adapter = self._connection_factory()
            return self.adapter
        from rosclaw.connectors.ros.transport import RosbridgeEndpoint, RosbridgeTransport

        endpoint = os.environ.get("ROSCLAW_CMU_ARE_ROSBRIDGE_URL", "ws://127.0.0.1:9090")
        transport = RosbridgeTransport(endpoint=RosbridgeEndpoint.from_url(endpoint))
        self.adapter = CmuAreRosbridgeAdapter(transport)
        return self.adapter

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        validation_error = self._validate_action(action)
        if validation_error is not None:
            code, message = validation_error
            return self._failed_result(action, code, message)

        started_at = _iso_now()
        try:
            artifact_dir = self._artifact_directory(action)
        except (OSError, ValueError) as exc:
            return self._failed_result(action, "CMU_ARE_ARTIFACT_PATH_INVALID", str(exc))
        cancel_event = self._begin_action(action.action_id)
        try:
            adapter = self._get_adapter()
        except Exception as exc:  # noqa: BLE001 - daemon boundary must fail closed
            self._finish_action(action.action_id, cancel_event)
            return self._failed_result(
                action,
                "CMU_ARE_ADAPTER_INIT_FAILED",
                str(exc),
            )
        generation_before: Any = None
        try:
            connect = getattr(adapter, "connect", None)
            if callable(connect):
                connect()
            generation_before = self._connection_details(adapter).get("generation")
            if action.capability_id == CMU_NAVIGATE_CAPABILITY:
                result = self._navigate(action, adapter, cancel_event=cancel_event)
            elif action.capability_id == CMU_EXPLORE_CAPABILITY:
                result = self._explore(action, adapter, cancel_event=cancel_event)
            else:
                result = self._stop(action, adapter)
        except CmuAreTransportError as exc:
            result = {
                "status": "failed",
                "error": str(exc),
                "error_code": "CMU_ARE_CONNECTION_FAILED",
            }
        except (CmuAreContractError, TimeoutError, OSError, ValueError) as exc:
            result = {
                "status": "failed",
                "error": str(exc),
                "error_code": "CMU_ARE_EXECUTION_FAILED",
            }
        finally:
            self._finish_action(action.action_id, cancel_event)

        generation_after = self._connection_details(adapter).get("generation")
        if (
            generation_before is not None
            and generation_after is not None
            and generation_before != generation_after
        ):
            result = {
                "status": "failed",
                "error": "rosbridge connection generation changed during action",
                "error_code": "CMU_ARE_CONNECTION_GENERATION_CHANGED",
                "generation_before": generation_before,
                "generation_after": generation_after,
            }

        finished_at = _iso_now()
        evidence, final_state, verification = self._classify(action, result)
        errors_from_artifacts: dict[str, Any] | None
        try:
            artifacts = self._write_artifacts(
                action=action,
                adapter=adapter,
                result=result,
                artifact_dir=artifact_dir,
                started_at=started_at,
                finished_at=finished_at,
                evidence=evidence,
                verification=verification,
            )
        except (OSError, ValueError) as exc:
            artifacts = []
            errors_from_artifacts = {
                "code": "CMU_ARE_ARTIFACT_WRITE_FAILED",
                "message": str(exc),
            }
        else:
            errors_from_artifacts = None
        errors: list[dict[str, Any]] = []
        if result.get("error"):
            errors.append(
                {
                    "code": result.get("error_code", "CMU_ARE_EXECUTION_FAILED"),
                    "message": str(result["error"]),
                }
            )
        elif final_state is ActionState.FAILED:
            errors.append(
                {
                    "code": result.get("error_code", "CMU_ARE_EXECUTION_FAILED"),
                    "message": f"adapter reported terminal status {result.get('status')!r}",
                }
            )
        if errors_from_artifacts is not None:
            errors.append(errors_from_artifacts)
        return ActionExecutionResult(
            final_state=final_state,
            evidence_level=evidence,
            evidence_domain=EvidenceDomain.SHADOW,
            policy_decision={
                "allowed": not errors,
                "policy": "cmu-are-sim/shadow-v1",
                "reason": (
                    "bounded CMU ARE contract and observation predicate passed"
                    if evidence is EvidenceLevel.TASK_VERIFIED
                    else "command was accepted but task-level observation is unavailable"
                ),
            },
            authorization_decision={"authorized": False, "required": False, "mode": "SHADOW"},
            dispatch_result={
                "accepted": result.get("dispatched") is True,
                "adapter": "cmu_are_rosbridge",
                "connection": self._connection_details(adapter),
                "status": result.get("status"),
            },
            driver_ack={
                "acknowledged": result.get("dispatched") is True,
                "stage": (
                    AcknowledgementStage.PROTOCOL_ACKNOWLEDGED.value
                    if result.get("dispatched") is True
                    else AcknowledgementStage.REQUEST_ACCEPTED.value
                ),
            },
            observations=self._observations(result),
            verification_result=verification,
            artifacts=artifacts,
            errors=errors,
            artifact_directory=str(artifact_dir),
            acknowledgement_stage=(
                AcknowledgementStage.PROTOCOL_ACKNOWLEDGED
                if result.get("dispatched") is True
                else AcknowledgementStage.REQUEST_ACCEPTED
            ),
        )

    def cancel(self) -> None:
        """Request cooperative cancellation for an in-flight navigation."""

        with self._cancel_lock:
            if self._active_action_id is not None:
                self.cancel_event.set()

    def _begin_action(self, action_id: str) -> threading.Event:
        with self._cancel_lock:
            # A worker instance may execute multiple actions over its lifetime.
            # Never let a previous cancellation leak into the next action.
            self.cancel_event = threading.Event()
            self._active_action_id = action_id
            return self.cancel_event

    def _finish_action(self, action_id: str, cancel_event: threading.Event) -> None:
        with self._cancel_lock:
            if self._active_action_id == action_id and self.cancel_event is cancel_event:
                self._active_action_id = None

    def _validate_action(self, action: ActionEnvelope) -> tuple[str, str] | None:
        if _ACTION_ID_RE.fullmatch(action.action_id) is None:
            return "CMU_ARE_ACTION_ID_INVALID", "action_id must be a bounded path-safe identifier"
        if action.execution_mode is not ExecutionMode.SHADOW:
            return "CMU_ARE_SHADOW_ONLY", "CMU ARE simulation executor accepts SHADOW actions only"
        if action.body_id != CMU_ARE_BODY_ID:
            return "CMU_ARE_BODY_MISMATCH", f"expected body_id={CMU_ARE_BODY_ID!r}"
        if action.body_snapshot_hash != self.expected_body_snapshot_hash:
            return "CMU_ARE_BODY_SNAPSHOT_MISMATCH", "body snapshot hash is stale or malformed"
        if action.capability_id not in CMU_CAPABILITIES:
            return "CMU_ARE_CAPABILITY_UNSUPPORTED", action.capability_id
        args = action.arguments
        if not isinstance(args, dict):
            return "CMU_ARE_ARGUMENTS_INVALID", "action arguments must be an object"
        forbidden = sorted(
            key
            for key in args
            if key in _FORBIDDEN_ARGUMENT_NAMES or str(key).casefold() in _FORBIDDEN_ARGUMENT_NAMES
        )
        if forbidden:
            return "CMU_ARE_SOUTHBAND_FIELD_FORBIDDEN", ", ".join(forbidden)
        if action.capability_id == CMU_NAVIGATE_CAPABILITY:
            return self._validate_navigation(args)
        if action.capability_id == CMU_EXPLORE_CAPABILITY:
            return self._validate_exploration(args)
        return self._validate_stop(args)

    def _validate_navigation(self, args: dict[str, Any]) -> tuple[str, str] | None:
        if set(args) - _NAV_KEYS:
            return "CMU_ARE_ARGUMENTS_INVALID", "navigation arguments contain unknown fields"
        if args.get("schema_version") != CMU_ARE_NAV_SCHEMA:
            return "CMU_ARE_SCHEMA_INVALID", "navigation schema version is unsupported"
        target = args.get("target")
        if not isinstance(target, dict) or set(target) - {"frame_id", "x", "y", "z"}:
            return "CMU_ARE_TARGET_INVALID", "target must contain only frame_id/x/y/z"
        if set(target) != {"frame_id", "x", "y", "z"}:
            return "CMU_ARE_TARGET_INVALID", "target requires frame_id, x, y, and z"
        if target.get("frame_id") != "map":
            return "CMU_ARE_TARGET_INVALID", "only the map frame is accepted"
        try:
            self.safety.validate_goal(x=target["x"], y=target["y"], z=target["z"])
            self.safety.clamp_speed(args["speed_mps"])
            self.safety.validate_tolerance(args["tolerance_m"])
            self.safety.validate_timeout(args["timeout_sec"])
        except (KeyError, TypeError, ValueError, CmuAreContractError) as exc:
            return "CMU_ARE_ARGUMENTS_INVALID", str(exc)
        expected = args.get("expected_effect")
        if expected is not None and expected != {
            "kind": "navigate_to_waypoint",
            "final_frame": "map",
            "stop_required": True,
        }:
            return "CMU_ARE_EXPECTED_EFFECT_INVALID", "navigation expected_effect is not exact"
        return None

    def _validate_exploration(self, args: dict[str, Any]) -> tuple[str, str] | None:
        if set(args) - _EXPLORE_KEYS:
            return "CMU_ARE_ARGUMENTS_INVALID", "exploration arguments contain unknown fields"
        if args.get("schema_version") != CMU_ARE_EXPLORE_SCHEMA:
            return "CMU_ARE_SCHEMA_INVALID", "exploration schema version is unsupported"
        if args.get("command") not in {"start", "pause", "resume", "stop"}:
            return "CMU_ARE_COMMAND_INVALID", "exploration command must be start/pause/resume/stop"
        try:
            self.safety.clamp_speed(args["speed_mps"])
            self.safety.validate_timeout(args["timeout_sec"])
        except (KeyError, TypeError, ValueError, CmuAreContractError) as exc:
            return "CMU_ARE_ARGUMENTS_INVALID", str(exc)
        expected = args.get("expected_effect")
        if expected is not None and expected != {
            "kind": "exploration_control",
            "command": args["command"],
        }:
            return "CMU_ARE_EXPECTED_EFFECT_INVALID", "exploration expected_effect is not exact"
        return None

    @staticmethod
    def _validate_stop(args: dict[str, Any]) -> tuple[str, str] | None:
        if set(args) - _STOP_KEYS:
            return "CMU_ARE_ARGUMENTS_INVALID", "stop arguments contain unknown fields"
        if args.get("schema_version") != CMU_ARE_STOP_SCHEMA:
            return "CMU_ARE_SCHEMA_INVALID", "stop schema version is unsupported"
        try:
            timeout = args["timeout_sec"]
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise CmuAreContractError("timeout_sec must be numeric")
            timeout = float(timeout)
            if not math.isfinite(timeout):
                raise CmuAreContractError("timeout_sec must be finite")
        except (KeyError, TypeError, ValueError, CmuAreContractError) as exc:
            return "CMU_ARE_ARGUMENTS_INVALID", f"timeout_sec is invalid: {exc}"
        if not 0 < timeout <= 60:
            return "CMU_ARE_ARGUMENTS_INVALID", "timeout_sec must be in (0, 60]"
        expected = args.get("expected_effect")
        if expected is not None and expected != {
            "kind": "stop",
            "zero_velocity_required": True,
        }:
            return "CMU_ARE_EXPECTED_EFFECT_INVALID", "stop expected_effect is not exact"
        return None

    def _navigate(
        self,
        action: ActionEnvelope,
        adapter: Any,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        args = action.arguments
        try:
            result = adapter.navigate(
                target=args["target"],
                speed_mps=float(args["speed_mps"]),
                tolerance_m=float(args["tolerance_m"]),
                timeout_sec=float(args["timeout_sec"]),
                cancel_event=cancel_event or self.cancel_event,
            )
        except TypeError as exc:
            # Small test doubles and older adapters may not expose cooperative
            # cancellation.  Only fall back when the signature rejects that
            # keyword; transport/type errors are still handled fail-closed by
            # the outer execution boundary.
            if "cancel_event" not in str(exc):
                raise
            result = adapter.navigate(
                target=args["target"],
                speed_mps=float(args["speed_mps"]),
                tolerance_m=float(args["tolerance_m"]),
                timeout_sec=float(args["timeout_sec"]),
            )
        return {**(result if isinstance(result, dict) else {}), "dispatched": True}

    def _explore(
        self,
        action: ActionEnvelope,
        adapter: Any,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        args = action.arguments
        try:
            result = adapter.exploration_control(
                str(args["command"]),
                speed_mps=float(args["speed_mps"]),
                timeout_sec=float(args["timeout_sec"]),
                cancel_event=cancel_event or self.cancel_event,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            result = adapter.exploration_control(
                str(args["command"]), speed_mps=float(args["speed_mps"])
            )
        return {**(result if isinstance(result, dict) else {}), "dispatched": True}

    def _stop(self, action: ActionEnvelope, adapter: Any) -> dict[str, Any]:
        try:
            result = adapter.stop(timeout_sec=float(action.arguments["timeout_sec"]))
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            result = adapter.stop()
        return {**(result if isinstance(result, dict) else {}), "dispatched": True}

    def _classify(
        self, action: ActionEnvelope, result: dict[str, Any]
    ) -> tuple[EvidenceLevel, ActionState, dict[str, Any]]:
        if result.get("error") or result.get("status") in {"failed", "error"}:
            return (
                EvidenceLevel.REQUESTED,
                ActionState.FAILED,
                {
                    "success": False,
                    "predicate": "adapter operation completed without transport error",
                },
            )
        if action.capability_id == CMU_NAVIGATE_CAPABILITY:
            target = action.arguments["target"]
            final = result.get("final_pose")
            distance = result.get("distance_to_goal")
            tolerance = float(action.arguments["tolerance_m"])
            verified = (
                result.get("status") == "success"
                and result.get("stop_confirmed") is True
                and self._is_finite_pose(final)
                and isinstance(distance, (int, float))
                and not isinstance(distance, bool)
                and math.isfinite(float(distance))
                and float(distance) <= tolerance
            )
            state = (
                ActionState.TIMED_OUT
                if result.get("status") == "timeout"
                else ActionState.CANCELLED
                if result.get("status") == "cancelled"
                else ActionState.COMPLETED
            )
            return (
                (EvidenceLevel.TASK_VERIFIED if verified else EvidenceLevel.DISPATCH_CONFIRMED),
                state,
                {
                    "success": verified,
                    "predicate": "fresh odometry within target tolerance and zero velocity observed",
                    "target": target,
                    "distance_to_goal_m": distance,
                    "stop_confirmed": result.get("stop_confirmed"),
                },
            )
        if action.capability_id == CMU_EXPLORE_CAPABILITY:
            verified = result.get("state_changed") is True
            state = (
                ActionState.TIMED_OUT
                if result.get("status") == "timeout"
                else ActionState.CANCELLED
                if result.get("status") == "cancelled"
                else ActionState.COMPLETED
            )
            return (
                (EvidenceLevel.TASK_VERIFIED if verified else EvidenceLevel.DISPATCH_CONFIRMED),
                state,
                {
                    "success": verified,
                    "predicate": "ARiADNE2 exploration state changed after control command",
                    "state_before": result.get("state_before"),
                    "state_after": result.get("state_after"),
                },
            )
        verified = result.get("stop_confirmed") is True
        state = (
            ActionState.TIMED_OUT
            if result.get("status") == "timeout"
            else ActionState.CANCELLED
            if result.get("status") == "cancelled"
            else ActionState.COMPLETED
        )
        return (
            (EvidenceLevel.TASK_VERIFIED if verified else EvidenceLevel.DISPATCH_CONFIRMED),
            state,
            {
                "success": verified,
                "predicate": "zero velocity observation received after stop command",
                "stop_confirmed": result.get("stop_confirmed"),
            },
        )

    @staticmethod
    def _is_finite_pose(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        coordinates = (value.get("x"), value.get("y"))
        return all(
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            and math.isfinite(float(coordinate))
            for coordinate in coordinates
        )

    def _artifact_directory(self, action: ActionEnvelope) -> Path:
        root = (self.home / "practice_data" / "app_runs").resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = _safe_child(root, action.action_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _write_artifacts(
        self,
        *,
        action: ActionEnvelope,
        adapter: Any,
        result: dict[str, Any],
        artifact_dir: Path,
        started_at: str,
        finished_at: str,
        evidence: EvidenceLevel,
        verification: dict[str, Any],
    ) -> list[str]:
        traces = {
            "odom_trace.jsonl": getattr(adapter, "odom_trace", []),
            "cmd_vel.jsonl": getattr(adapter, "cmd_trace", []),
            "path_trace.jsonl": getattr(adapter, "path_trace", []),
            "waypoints.jsonl": getattr(adapter, "waypoint_trace", []),
        }
        events = [
            {"at": started_at, "event": "action_started", "action_id": action.action_id},
            {
                "at": finished_at,
                "event": "action_finished",
                "status": result.get("status"),
                "evidence_level": evidence.value,
            },
        ]
        files: dict[str, Any] = {
            "summary.json": {
                "schema_version": "rosclaw.cmu_are.summary.v1",
                "action_id": action.action_id,
                "capability_id": action.capability_id,
                "body_id": action.body_id,
                "status": result.get("status"),
                "result": result,
                "verification": verification,
            },
            "task_events.jsonl": events,
            "ros_topics.txt": self._topic_lines(adapter),
        }
        for name, rows in traces.items():
            files[name] = rows if isinstance(rows, list) else []

        artifact_paths: list[str] = []
        artifact_digests: list[dict[str, Any]] = []
        for name, value in files.items():
            path = _safe_child(artifact_dir, name)
            if name.endswith(".json"):
                path.write_text(
                    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            elif name.endswith(".jsonl"):
                rows = value if isinstance(value, list) else []
                path.write_text(
                    "".join(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
                    ),
                    encoding="utf-8",
                )
            else:
                path.write_text(str(value), encoding="utf-8")
            artifact_paths.append(path.as_uri())
            artifact_digests.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": f"sha256:{_sha256_bytes(path.read_bytes())}",
                }
            )

        manifest_payload = {
            "schema_version": CMU_MANIFEST_SCHEMA,
            "action_id": action.action_id,
            "trace_id": action.parent_trace_id or f"trace_{action.action_id}",
            "capability_id": action.capability_id,
            "body_id": action.body_id,
            "body_snapshot_hash": action.body_snapshot_hash,
            "world": os.environ.get("CMU_ARE_WORLD", "campus"),
            "pack_version": self.pack_version,
            "safety_card_hash": self.safety.digest,
            "asset_manifest_hash": (
                f"sha256:{_sha256_bytes(self.asset_manifest.read_bytes())}"
                if self.asset_manifest.is_file()
                else None
            ),
            "worker_id": self.worker_id,
            "worker_generation": self._connection_details(adapter).get("generation"),
            "input_digest": f"sha256:{_sha256_bytes(_json_bytes(action.to_dict()))}",
            "started_at": started_at,
            "finished_at": finished_at,
            "status": result.get("status"),
            "artifact_paths": [Path(uri).name for uri in artifact_paths],
            "artifact_digests": artifact_digests,
            "observation_summary": self._observations(result),
            "verification_summary": verification,
            "errors": ([{"message": result["error"]}] if result.get("error") else []),
        }
        manifest_path = _safe_child(artifact_dir, "cmu_are_manifest.json")
        manifest_path.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifact_paths.append(manifest_path.as_uri())
        return artifact_paths

    @staticmethod
    def _topic_lines(adapter: Any) -> str:
        check = getattr(adapter, "last_check", None)
        if isinstance(check, dict):
            topics = check.get("topics", {})
            if isinstance(topics, dict):
                return "".join(f"{topic} {kind}\n" for topic, kind in sorted(topics.items()))
        return ""

    @staticmethod
    def _connection_details(adapter: Any) -> dict[str, Any]:
        connection = getattr(adapter, "connection", None)
        if connection is None:
            return {}
        if hasattr(connection, "__dict__"):
            return dict(connection.__dict__)
        return {}

    @staticmethod
    def _observations(result: dict[str, Any]) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for key in (
            "start_pose",
            "final_pose",
            "distance_to_goal",
            "stop_confirmed",
            "state_before",
            "state_after",
        ):
            if key in result:
                observations.append({"kind": key, "value": result[key]})
        return observations

    def _failed_result(
        self, action: ActionEnvelope, code: str, message: str
    ) -> ActionExecutionResult:
        return ActionExecutionResult(
            final_state=ActionState.BLOCKED,
            evidence_level=EvidenceLevel.REQUESTED,
            evidence_domain=EvidenceDomain.SHADOW,
            policy_decision={
                "allowed": False,
                "policy": "cmu-are-sim/shadow-v1",
                "reason": message,
            },
            dispatch_result={"accepted": False},
            errors=[{"code": code, "message": message}],
        )


__all__ = [
    "CMU_ARE_BODY_ID",
    "CMU_CAPABILITIES",
    "CMU_EXPLORE_CAPABILITY",
    "CMU_MANIFEST_SCHEMA",
    "CMU_NAVIGATE_CAPABILITY",
    "CMU_STOP_CAPABILITY",
    "CmuAreShadowExecutor",
]
