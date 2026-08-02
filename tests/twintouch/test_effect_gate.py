"""PR-TT-3 effect gate tests — including the v3 left-no-motion replay.

The v3 hole is replayed from its recorded evidence shape
(t1 run prac_20260731T002911Z_t1lattice): left hand commanded to a
silent slave id — no telemetry at all — while the visual channel kept
reporting an OK, perfectly STABLE hand.  The old pipeline judged that
hand "settled" 10/10 poses.  This gate must never.
"""

from __future__ import annotations

from rosclaw.twintouch.effect_gate import (
    COMMAND_HOLD,
    COMMAND_MOVE,
    EFFECT_CONFIRMED,
    EFFECT_UNKNOWN,
    EXTERNAL_ONLY,
    INTERNAL_ONLY,
    NO_EFFECT,
    WRONG_BODY_EFFECT,
    EffectThresholds,
    JointCommand,
    PhysicalActionEffectReceipt,
    TelemetryPoint,
    VisualSample,
    bimanual_effect_gate,
    evaluate_effect,
)

TH = EffectThresholds()


def _move_command(joints=("index",), target=400, before=1000) -> JointCommand:
    return JointCommand(
        action_id="act_1",
        body_id="rh56_left_01",
        command_type=COMMAND_MOVE,
        targets=dict.fromkeys(joints, target),
        issued_at_s=0.0,
        window_s=3.0,
    )


def _telemetry_series(joint_values: list[dict[str, int]], dt: float = 0.5) -> list[TelemetryPoint]:
    return [TelemetryPoint(ts_s=i * dt, angle_actual=dict(v)) for i, v in enumerate(joint_values)]


def _visual(x: float, y: float = 0.0, z: float = 0.2, ok: bool = True) -> VisualSample:
    return VisualSample(ok=ok, centroid_3d=(x, y, z) if ok else None)


# ------------------------------------------------------------- move verdicts


def test_move_confirmed_when_servo_follows_and_camera_sees_it():
    command = _move_command(target=400)
    telemetry = _telemetry_series([{"index": 1000}, {"index": 830}, {"index": 620}, {"index": 405}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(0.30),
        visual_after=_visual(0.35),  # 0.05 m — clearly moved
        other_before=_visual(-0.30),
        other_after=_visual(-0.30),  # other body static
        thresholds=TH,
    )
    assert receipt.verdict == EFFECT_CONFIRMED
    assert receipt.internal_followed is True
    assert receipt.external_moved is True
    assert receipt.validate() == []


def test_no_effect_when_servo_static_and_camera_static():
    """Static telemetry + static visual = the command changed NOTHING.
    A static hand is not a settled hand; it is an uneffected command."""
    command = _move_command(target=400)
    telemetry = _telemetry_series([{"index": 1000}, {"index": 1000}, {"index": 1000}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(0.30),
        visual_after=_visual(0.301),  # sub-jitter
        other_before=_visual(-0.30),
        other_after=_visual(-0.30),
        thresholds=TH,
    )
    assert receipt.verdict == NO_EFFECT
    assert receipt.internal_followed is False
    assert receipt.external_moved is False
    assert receipt.validate() == []


def test_wrong_body_when_target_static_but_other_body_moved():
    """The swapped-slave shape: commanded body never moved, the OTHER
    body's ROI shows the motion."""
    command = _move_command(target=400)
    telemetry = _telemetry_series([{"index": 1000}, {"index": 1000}, {"index": 1000}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(0.30),
        visual_after=_visual(0.30),
        other_before=_visual(-0.30),
        other_after=_visual(-0.24),  # the OTHER body moved
        thresholds=TH,
    )
    assert receipt.verdict == WRONG_BODY_EFFECT
    assert receipt.other_body_moved is True
    assert receipt.validate() == []


def test_internal_only_when_servo_follows_but_camera_blind():
    command = _move_command(target=400)
    telemetry = _telemetry_series([{"index": 1000}, {"index": 700}, {"index": 410}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(0.30),
        visual_after=_visual(0.30),  # camera saw nothing
        other_before=_visual(-0.30),
        other_after=_visual(-0.30),
        thresholds=TH,
    )
    assert receipt.verdict == INTERNAL_ONLY


def test_external_only_when_camera_moves_without_servo():
    command = _move_command(target=400)
    telemetry = _telemetry_series([{"index": 1000}, {"index": 1000}, {"index": 999}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(0.30),
        visual_after=_visual(0.34),
        other_before=_visual(-0.30),
        other_after=_visual(-0.30),
        thresholds=TH,
    )
    assert receipt.verdict == EXTERNAL_ONLY
    assert any("disturbed" in r for r in receipt.reasons)


def test_unknown_when_channels_missing():
    command = _move_command(target=400)
    receipt = evaluate_effect(
        command,
        [],  # no telemetry
        visual_before=_visual(0.30, ok=False),
        visual_after=_visual(0.30, ok=False),
        other_before=_visual(-0.30),
        other_after=_visual(-0.30),
        thresholds=TH,
    )
    assert receipt.verdict == EFFECT_UNKNOWN
    assert any("telemetry" in r for r in receipt.reasons)


def test_wrong_body_survives_missing_own_visual():
    """Servo provably static + other body provably moved is too specific
    to bury in UNKNOWN even when the own-body camera is blind."""
    command = _move_command(target=400)
    telemetry = _telemetry_series([{"index": 1000}, {"index": 1000}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(0.30, ok=False),
        visual_after=_visual(0.30, ok=False),
        other_before=_visual(-0.30),
        other_after=_visual(-0.22),
        thresholds=TH,
    )
    assert receipt.verdict == WRONG_BODY_EFFECT


# ------------------------------------------------- v3 no-motion hole replay


def test_v3_replay_missing_telemetry_stable_visual_is_never_settled():
    """The recorded v3 shape (prac_20260731T002911Z_t1lattice): commanded
    to a silent slave — NO telemetry — visual OK and perfectly stable.
    The old pipeline: settled=True 10/10.  This gate: UNKNOWN, and the
    bimanual gate BLOCKS."""
    command = _move_command(target=150)  # 'rock' — never executed
    receipt = evaluate_effect(
        command,
        [],  # silent slave: zero telemetry, exactly like the v3 run
        visual_before=_visual(0.30),
        visual_after=_visual(0.30),  # static hand is trivially stable
        other_before=_visual(-0.30),
        other_after=_visual(-0.30),
        thresholds=TH,
    )
    assert receipt.verdict == EFFECT_UNKNOWN
    assert receipt.verdict != EFFECT_CONFIRMED

    ok_receipt = PhysicalActionEffectReceipt(
        action_id="act_r",
        body_id="rh56_right_01",
        command_type=COMMAND_MOVE,
        commanded_joints=("index",),
        internal_delta_max_raw=600,
        internal_followed=True,
        external_displacement_m=0.05,
        external_moved=True,
        other_body_displacement_m=0.001,
        other_body_moved=False,
        verdict=EFFECT_CONFIRMED,
    )
    gate = bimanual_effect_gate(receipt, ok_receipt)
    assert gate.proceed is False
    assert any("left" in v for v in gate.violations)


def test_v3_replay_static_telemetry_variant_is_no_effect_and_blocks():
    """Variant: reads work but the actuator ignored the command (power,
    stall) — static telemetry.  NO_EFFECT, still never confirmed."""
    command = _move_command(target=150)
    telemetry = _telemetry_series([{"index": 1000}, {"index": 1000}, {"index": 1000}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(0.30),
        visual_after=_visual(0.30),
        other_before=_visual(-0.30),
        other_after=_visual(-0.30),
        thresholds=TH,
    )
    assert receipt.verdict == NO_EFFECT
    gate = bimanual_effect_gate(
        receipt,
        PhysicalActionEffectReceipt(
            action_id="act_r",
            body_id="rh56_right_01",
            command_type=COMMAND_MOVE,
            commanded_joints=("index",),
            internal_delta_max_raw=600,
            internal_followed=True,
            external_displacement_m=0.05,
            external_moved=True,
            other_body_displacement_m=0.001,
            other_body_moved=False,
            verdict=EFFECT_CONFIRMED,
        ),
    )
    assert gate.proceed is False


# ------------------------------------------------------------- hold verdicts


def test_hold_confirmed_by_stability():
    command = JointCommand(
        action_id="act_h",
        body_id="rh56_right_01",
        command_type=COMMAND_HOLD,
        targets={"index": 1000},
        issued_at_s=0.0,
        window_s=5.0,
    )
    telemetry = _telemetry_series([{"index": 1000}, {"index": 998}, {"index": 1001}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(-0.30),
        visual_after=_visual(-0.301),
        other_before=_visual(0.30),
        other_after=_visual(0.35),  # the ACTIVE hand moved — fine
        thresholds=TH,
    )
    assert receipt.verdict == EFFECT_CONFIRMED


def test_hold_disturbed_is_external_only():
    """The passive hand drifted/was pushed while 'holding' — the
    PEER_DISTURBANCE precursor the Contact Supervisor must see."""
    command = JointCommand(
        action_id="act_h",
        body_id="rh56_right_01",
        command_type=COMMAND_HOLD,
        targets={"index": 1000},
        issued_at_s=0.0,
        window_s=5.0,
    )
    telemetry = _telemetry_series([{"index": 1000}, {"index": 999}, {"index": 1000}])
    receipt = evaluate_effect(
        command,
        telemetry,
        visual_before=_visual(-0.30),
        visual_after=_visual(-0.26),  # passive hand visibly moved
        other_before=_visual(0.30),
        other_after=_visual(0.30),
        thresholds=TH,
    )
    assert receipt.verdict == EXTERNAL_ONLY


# ------------------------------------------------------------ bimanual gate


def test_bimanual_gate_proceeds_only_when_both_confirmed():
    ok_left = PhysicalActionEffectReceipt(
        action_id="a_l",
        body_id="rh56_left_01",
        command_type=COMMAND_MOVE,
        commanded_joints=("index",),
        internal_delta_max_raw=590,
        internal_followed=True,
        external_displacement_m=0.05,
        external_moved=True,
        other_body_displacement_m=0.0,
        other_body_moved=False,
        verdict=EFFECT_CONFIRMED,
    )
    ok_right = PhysicalActionEffectReceipt(
        action_id="a_r",
        body_id="rh56_right_01",
        command_type=COMMAND_HOLD,
        commanded_joints=("index",),
        internal_delta_max_raw=3,
        internal_followed=True,
        external_displacement_m=0.002,
        external_moved=False,  # hold confirmed by stillness
        other_body_displacement_m=0.05,
        other_body_moved=True,  # the ACTIVE hand moved — attribution ok
        verdict=EFFECT_CONFIRMED,
    )
    gate = bimanual_effect_gate(ok_left, ok_right)
    assert gate.proceed is True
    assert gate.violations == ()

    wrong_body = PhysicalActionEffectReceipt(
        action_id="a_l2",
        body_id="rh56_left_01",
        command_type=COMMAND_MOVE,
        commanded_joints=("index",),
        internal_delta_max_raw=1,
        internal_followed=False,
        external_displacement_m=0.0,
        external_moved=False,
        other_body_displacement_m=0.06,
        other_body_moved=True,
        verdict=WRONG_BODY_EFFECT,
    )
    gate2 = bimanual_effect_gate(wrong_body, ok_right)
    assert gate2.proceed is False
    assert any("WRONG body" in v for v in gate2.violations)
