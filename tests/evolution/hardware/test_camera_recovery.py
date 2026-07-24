"""Camera recovery state machine tests (PR-EVO-HW-2 §五B/§九)."""

from __future__ import annotations

import pytest

from rosclaw.evolution.hardware.camera_recovery import (
    RECOVERY_PROCEDURES,
    run_recovery,
)


def _ok_handlers(*fail_at: str):
    calls: list[str] = []

    def make(name):
        def handler(ctx):
            calls.append(name)
            if name in fail_at:
                return False, f"injected failure at {name}"
            if name == "new_camera_session":
                ctx["camera_session_id"] = "cam_test_2"
            return True, ""

        return handler

    table: dict = {}
    for steps in RECOVERY_PROCEDURES.values():
        for name in steps:
            table.setdefault(name, make(name))
    table["_calls"] = calls
    return table


def test_procedure_b_full_path_recovers() -> None:
    table = _ok_handlers()
    result = run_recovery("B_reset_fresh_frames", handlers=table)
    assert result.recovered
    assert [s.name for s in result.steps] == RECOVERY_PROCEDURES["B_reset_fresh_frames"]
    assert all(s.ok for s in result.steps)
    assert result.camera_session_id == "cam_test_2"
    assert result.ttr_s >= 0
    assert not result.manual_replug_required
    assert "never a REAL permit" in result.note


def test_failed_reset_marks_manual_replug() -> None:
    table = _ok_handlers("hardware_reset")
    result = run_recovery("B_reset_fresh_frames", handlers=table)
    assert not result.recovered
    assert result.manual_replug_required
    # Fail-fast: steps after the failure never ran.
    assert [s.name for s in result.steps] == ["stop_node", "wait_1s", "hardware_reset"]


def test_step_durations_recorded() -> None:
    table = _ok_handlers()
    result = run_recovery("C_backoff_sync_check", handlers=table)
    assert result.recovered
    assert all(s.duration_s >= 0 for s in result.steps)
    assert result.ttr_s >= max(s.duration_s for s in result.steps)


def test_unknown_procedure_rejected() -> None:
    with pytest.raises(ValueError, match="unknown recovery procedure"):
        run_recovery("D_magic", handlers={})


def test_procedural_memory_schema() -> None:
    table = _ok_handlers()
    result = run_recovery("A_immediate_restart", handlers=table)
    memory = result.to_procedural_memory(
        body_id="d435i_01",
        device={"name": "Intel RealSense D435I", "serial": "231122070092", "firmware": "5.17.0.10"},
        pre_fault_frame_age_ms=2100.0,
        robot_id="rh56_rps_robot",
        evidence_refs=["evt_cam_1"],
    )
    assert memory["memory_type"] == "procedural"
    assert memory["failure_type"] == "camera_no_fresh_frame"
    assert memory["outcome"] == "success"
    md = memory["metadata"]
    assert md["recovered"] is True
    assert md["ttr_s"] >= 0
    assert md["steps"][0]["name"] == "stop_node"
    assert md["device"]["serial"] == "231122070092"
    assert md["pre_fault_frame_age_ms"] == 2100.0
    assert "stop_node" in md["recovery_hint"]
