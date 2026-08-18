from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rosclaw.integrations.cmu_are.executor import (
    CMU_EXPLORE_CAPABILITY,
    CMU_NAVIGATE_CAPABILITY,
    CMU_STOP_CAPABILITY,
    CmuAreShadowExecutor,
)
from rosclaw.kernel import ActionEnvelope, ActionState, EvidenceLevel, ExecutionMode


class _FakeAdapter:
    connection = SimpleNamespace(connection_id="fake", generation=1)
    odom_trace = [{"x": 0.0, "y": 0.0}]
    cmd_trace = [{"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}]
    path_trace = []
    waypoint_trace = []

    def navigate(self, **kwargs):
        return {
            "status": "success",
            "start_pose": {"x": 0.0, "y": 0.0},
            "final_pose": kwargs["target"],
            "distance_to_goal": 0.1,
            "stop_confirmed": True,
        }

    def exploration_control(self, command, **_kwargs):
        return {
            "status": "success",
            "command": command,
            "state_before": "paused",
            "state_after": "active",
            "state_changed": True,
        }

    def stop(self, **_kwargs):
        return {"status": "success", "stop_confirmed": True}

    def emergency_stop(self):
        return {"status": "success", "stop_confirmed": True}


def _action(executor: CmuAreShadowExecutor, **arguments) -> ActionEnvelope:
    return ActionEnvelope(
        actor_id="pytest",
        agent_framework="pytest",
        session_id="cmu-test",
        body_id="cmu_are_sim",
        body_snapshot_hash=executor.expected_body_snapshot_hash,
        capability_id=CMU_NAVIGATE_CAPABILITY,
        arguments={
            "schema_version": "cmu_are.navigation.v1",
            "target": {"frame_id": "map", "x": 1.0, "y": 2.0, "z": 0.0},
            "speed_mps": 1.0,
            "tolerance_m": 1.5,
            "timeout_sec": 5.0,
            **arguments,
        },
        execution_mode=ExecutionMode.SHADOW,
    )


def test_shadow_executor_writes_legacy_artifacts_and_manifest(tmp_path: Path) -> None:
    executor = CmuAreShadowExecutor(_FakeAdapter(), home=tmp_path)
    result = executor(_action(executor))
    assert result.final_state is ActionState.COMPLETED
    assert result.evidence_level is EvidenceLevel.TASK_VERIFIED
    assert (Path(result.artifact_directory) / "summary.json").is_file()
    manifest = json.loads((Path(result.artifact_directory) / "cmu_are_manifest.json").read_text())
    assert manifest["schema_version"] == "rosclaw.cmu_are.manifest.v1"
    assert len(manifest["artifact_digests"]) == 7
    assert all(item["sha256"].startswith("sha256:") for item in manifest["artifact_digests"])


def test_southbound_fields_and_stale_snapshot_fail_closed(tmp_path: Path) -> None:
    executor = CmuAreShadowExecutor(_FakeAdapter(), home=tmp_path)
    forbidden = executor(_action(executor, ros_topic="/cmd_vel"))
    assert forbidden.final_state is ActionState.BLOCKED
    assert forbidden.errors[0]["code"] == "CMU_ARE_SOUTHBAND_FIELD_FORBIDDEN"
    stale = _action(executor)
    stale.body_snapshot_hash = "sha256:" + "0" * 64
    result = executor(stale)
    assert result.final_state is ActionState.BLOCKED
    assert result.errors[0]["code"] == "CMU_ARE_BODY_SNAPSHOT_MISMATCH"


def _control_action(
    executor: CmuAreShadowExecutor,
    capability: str,
    arguments: dict[str, object],
) -> ActionEnvelope:
    return ActionEnvelope(
        actor_id="pytest",
        agent_framework="pytest",
        session_id="cmu-test",
        body_id="cmu_are_sim",
        body_snapshot_hash=executor.expected_body_snapshot_hash,
        capability_id=capability,
        arguments=arguments,
        execution_mode=ExecutionMode.SHADOW,
    )


def test_exploration_and_stop_require_observation(tmp_path: Path) -> None:
    executor = CmuAreShadowExecutor(_FakeAdapter(), home=tmp_path)
    explore = _control_action(
        executor,
        CMU_EXPLORE_CAPABILITY,
        {
            "schema_version": "cmu_are.exploration.v1",
            "command": "pause",
            "speed_mps": 1.0,
            "timeout_sec": 5.0,
        },
    )
    stop = _control_action(
        executor,
        CMU_STOP_CAPABILITY,
        {"schema_version": "cmu_are.stop.v1", "timeout_sec": 5.0},
    )
    assert executor(explore).evidence_level is EvidenceLevel.TASK_VERIFIED
    assert executor(stop).evidence_level is EvidenceLevel.TASK_VERIFIED


def test_cancellation_does_not_leak_to_the_next_action(tmp_path: Path) -> None:
    class CancellingAdapter(_FakeAdapter):
        calls = 0

        def navigate(self, **kwargs):
            type(self).calls += 1
            if type(self).calls == 1:
                kwargs["cancel_event"].set()
                return {"status": "cancelled", "stop_confirmed": False}
            return super().navigate(**kwargs)

    executor = CmuAreShadowExecutor(CancellingAdapter(), home=tmp_path)
    first = executor(_action(executor))
    second = executor(_action(executor))
    assert first.final_state is ActionState.CANCELLED
    assert second.evidence_level is EvidenceLevel.TASK_VERIFIED


def test_connection_generation_change_fails_closed(tmp_path: Path) -> None:
    class ReconnectingAdapter(_FakeAdapter):
        connection = SimpleNamespace(connection_id="fake", generation=1)

        def navigate(self, **kwargs):
            type(self).connection = SimpleNamespace(connection_id="new", generation=2)
            return super().navigate(**kwargs)

    executor = CmuAreShadowExecutor(ReconnectingAdapter(), home=tmp_path)
    result = executor(_action(executor))
    assert result.final_state is ActionState.FAILED
    assert result.errors[0]["code"] == "CMU_ARE_CONNECTION_GENERATION_CHANGED"


def test_shadow_executor_exposes_verified_emergency_stop(tmp_path: Path) -> None:
    executor = CmuAreShadowExecutor(_FakeAdapter(), home=tmp_path)
    result = executor.emergency_stop()
    assert result["acknowledged"] is True
    assert result["physical_stop_observed"] is True
    assert result["execution_mode"] == "SHADOW"
    assert result["verification_source"] == "/cmd_vel"
