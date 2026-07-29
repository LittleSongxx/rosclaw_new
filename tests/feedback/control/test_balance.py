from __future__ import annotations

from rosclaw.feedback.contracts import ErrorState, FeedbackFrame
from rosclaw.feedback.controllers.balance import G1BalanceReflexController


def _frame(phase: float, *, roll: float = 0.45, pitch: float = -0.1) -> FeedbackFrame:
    actual = {
        "torso_roll": roll,
        "torso_pitch": pitch,
        "com_y_relative": 0.04,
        "support_slip_m": 0.02,
        "left_hip_roll_joint": 0.1,
        "left_ankle_roll_joint": 0.0,
        "left_hip_pitch_joint": -0.2,
        "left_ankle_pitch_joint": 0.1,
    }
    return FeedbackFrame(
        sequence=0,
        timestamp_ns=1,
        observation_timestamp_ns=1,
        phase=phase,
        reference={"torso_roll": 0.0, "torso_pitch": 0.0, "com_y_relative": 0.0},
        actual=actual,
        error=ErrorState(
            value={"torso_roll": -roll, "torso_pitch": -pitch, "com_y_relative": -0.04},
            derivative={"torso_roll": -0.2, "torso_pitch": 0.1, "com_y_relative": 0.0},
            integral={"torso_roll": 0.0, "torso_pitch": 0.0, "com_y_relative": 0.0},
            timestamp_ns=1,
        ),
    )


def test_g1_balance_reflex_is_phase_gated_and_counteracts_roll() -> None:
    controller = G1BalanceReflexController()
    inactive = controller.compute(_frame(0.1), {})
    active = controller.compute(_frame(0.4), {})

    assert inactive == {}
    assert active["joint:left_hip_roll_joint"] < 0.0
    assert active["joint:left_ankle_roll_joint"] > 0.0
    assert active["joint:left_hip_pitch_joint"] > 0.0


def test_g1_balance_reflex_stays_transparent_below_trigger_and_latches() -> None:
    controller = G1BalanceReflexController()

    assert controller.compute(_frame(0.4, roll=0.2), {}) == {}
    assert controller.compute(_frame(0.4, roll=0.45), {})
    assert controller.compute(_frame(0.41, roll=0.2), {})
    controller.reset()
    assert controller.compute(_frame(0.4, roll=0.2), {}) == {}


def test_com_trigger_requires_transient_motion_not_only_kick_pose() -> None:
    controller = G1BalanceReflexController()
    frame = _frame(0.4, roll=0.2)
    actual = dict(frame.actual, com_y_relative=0.14)
    quasi_static = FeedbackFrame(
        sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        observation_timestamp_ns=frame.observation_timestamp_ns,
        phase=frame.phase,
        reference=frame.reference,
        actual=actual,
        error=frame.error,
    )
    assert controller.compute(quasi_static, {}) == {}

    transient_error = ErrorState(
        value=frame.error.value,
        derivative={**frame.error.derivative, "com_y_relative": -0.8},
        integral=frame.error.integral,
        timestamp_ns=frame.error.timestamp_ns,
    )
    transient = FeedbackFrame(
        sequence=frame.sequence,
        timestamp_ns=frame.timestamp_ns,
        observation_timestamp_ns=frame.observation_timestamp_ns,
        phase=frame.phase,
        reference=frame.reference,
        actual=actual,
        error=transient_error,
    )
    assert controller.compute(transient, {})


def test_slip_term_moves_support_joints_toward_measured_state() -> None:
    controller = G1BalanceReflexController()
    base = {
        "joint:left_hip_roll_joint": -0.2,
        "joint:left_ankle_roll_joint": 0.2,
        "joint:left_hip_pitch_joint": 0.3,
        "joint:left_ankle_pitch_joint": -0.3,
    }
    with_slip = controller.compute(_frame(0.4), base)
    without_slip_frame = _frame(0.4)
    actual = dict(without_slip_frame.actual)
    actual["support_slip_m"] = 0.0
    without_slip_frame = FeedbackFrame(
        sequence=without_slip_frame.sequence,
        timestamp_ns=without_slip_frame.timestamp_ns,
        observation_timestamp_ns=without_slip_frame.observation_timestamp_ns,
        phase=without_slip_frame.phase,
        reference=without_slip_frame.reference,
        actual=actual,
        error=without_slip_frame.error,
    )
    without_slip = controller.compute(without_slip_frame, base)

    assert with_slip["joint:left_hip_roll_joint"] > without_slip["joint:left_hip_roll_joint"]
