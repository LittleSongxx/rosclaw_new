from __future__ import annotations

from rosclaw.feedback.contracts import ErrorState, FeedbackFrame
from rosclaw.feedback.controllers.cerebellum import G1CerebellumConfig, G1CerebellumController
from rosclaw.feedback.profiles.g1_cerebellum import build_g1_cerebellum_runtime
from rosclaw.feedback.replay import RecordedLatencyClock


def _frame() -> FeedbackFrame:
    signals = {
        "torso_roll": 0.45,
        "torso_pitch": -0.10,
        "com_y_relative": 0.02,
        "contact_phase": 0.40,
        "ball_lateral_error_m": 0.10,
    }
    return FeedbackFrame(
        sequence=0,
        timestamp_ns=1,
        observation_timestamp_ns=1,
        phase=0.40,
        reference=dict.fromkeys(signals, 0.0),
        actual={**signals, "support_slip_m": 0.0, "contact_detected": 0.0},
        error=ErrorState(
            value={name: -value for name, value in signals.items()},
            derivative=dict.fromkeys(signals, 0.0),
            integral=dict.fromkeys(signals, 0.0),
            timestamp_ns=1,
        ),
    )


def test_cerebellum_combines_balance_and_skill_before_projection() -> None:
    controller = G1CerebellumController(
        G1CerebellumConfig(
            phase_modulation_enabled=True,
            lateral_modulation_enabled=True,
            recovery_modulation_enabled=True,
        )
    )

    residual = controller.compute(_frame(), {})

    assert residual["joint:left_hip_roll_joint"] < 0.0
    assert residual["joint:right_hip_yaw_joint"] < 0.0
    assert residual["skill:kick_phase_rate"] > 0.0


def test_cerebellum_keeps_uncalibrated_phase_modulation_transparent_by_default() -> None:
    residual = G1CerebellumController().compute(_frame(), {})

    assert "skill:kick_phase_rate" not in residual
    assert "joint:right_hip_yaw_joint" not in residual


def test_default_cerebellum_runtime_does_not_advertise_disabled_skill_outputs() -> None:
    runtime = build_g1_cerebellum_runtime(body_hash="sha256:" + "1" * 64)

    assert "skill:kick_phase_rate" not in runtime.spec.output_limits
    assert "contact_phase" not in runtime.spec.observation_signals


def test_cerebellum_profile_projects_composite_outputs_once() -> None:
    runtime = build_g1_cerebellum_runtime(
        body_hash="sha256:" + "1" * 64,
        config=G1CerebellumConfig(
            phase_modulation_enabled=True,
            lateral_modulation_enabled=True,
            recovery_modulation_enabled=True,
        ),
        compute_clock_ns=RecordedLatencyClock((100_000,)),
    )
    frame = _frame()

    command = runtime.tick(
        timestamp_ns=20_000_000,
        observation_timestamp_ns=20_000_000,
        phase=frame.phase,
        reference=frame.reference,
        actual=frame.actual,
        base_action={},
    )

    assert command.deadline_met
    assert abs(command.projected["skill:kick_phase_rate"]) <= 0.08
    assert abs(command.projected["joint:left_hip_roll_joint"]) <= 0.08
