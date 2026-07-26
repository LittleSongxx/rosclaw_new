from __future__ import annotations

from rosclaw.feedback.contracts import ErrorState, FeedbackFrame
from rosclaw.feedback.controllers.kick_skill import G1KickSkillFeedbackController
from rosclaw.feedback.profiles.g1_skill import build_g1_kick_skill_runtime
from rosclaw.feedback.replay import RecordedLatencyClock


def _frame(*, phase: float, contact: bool, lateral: float = 0.1) -> FeedbackFrame:
    actual = {
        "contact_phase": 0.52,
        "ball_lateral_error_m": lateral,
        "contact_detected": float(contact),
        "torso_roll": 0.2,
        "torso_pitch": -0.1,
    }
    return FeedbackFrame(
        sequence=1,
        timestamp_ns=1,
        observation_timestamp_ns=1,
        phase=phase,
        reference={
            "contact_phase": 0.48,
            "ball_lateral_error_m": 0.0,
            "torso_roll": 0.0,
            "torso_pitch": 0.0,
        },
        actual=actual,
        error=ErrorState(
            value={
                "contact_phase": -0.04,
                "ball_lateral_error_m": -lateral,
                "torso_roll": -0.2,
                "torso_pitch": 0.1,
            },
            derivative={
                "contact_phase": 0.0,
                "ball_lateral_error_m": 0.0,
                "torso_roll": 0.0,
                "torso_pitch": 0.0,
            },
            integral={
                "contact_phase": 0.0,
                "ball_lateral_error_m": 0.0,
                "torso_roll": 0.0,
                "torso_pitch": 0.0,
            },
            timestamp_ns=1,
        ),
    )


def test_skill_feedback_separates_precontact_replan_and_recovery() -> None:
    controller = G1KickSkillFeedbackController()
    precontact = controller.compute(_frame(phase=0.4, contact=False), {})
    recovery = controller.compute(_frame(phase=0.55, contact=True), {})

    assert precontact["skill:kick_phase_rate"] < 0.0
    assert precontact["joint:right_hip_yaw_joint"] < 0.0
    assert "skill:kick_phase_rate" not in recovery
    assert recovery["joint:waist_roll_joint"] < 0.0
    assert recovery["joint:waist_pitch_joint"] > 0.0


def test_skill_runtime_projects_phase_and_joint_directives() -> None:
    runtime = build_g1_kick_skill_runtime(
        body_hash="sha256:" + "1" * 64,
        compute_clock_ns=RecordedLatencyClock((100_000,)),
    )
    frame = _frame(phase=0.4, contact=False, lateral=1.0)
    command = runtime.tick(
        timestamp_ns=20_000_000,
        observation_timestamp_ns=20_000_000,
        phase=frame.phase,
        reference=frame.reference,
        actual=frame.actual,
        base_action={},
    )

    assert command.projected["joint:right_hip_yaw_joint"] == -0.035
    assert command.saturation_count == 2
    assert command.deadline_met
