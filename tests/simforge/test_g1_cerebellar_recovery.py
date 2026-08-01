from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rosclaw.simforge.g1_cerebellar_recovery import (
    G1CerebellarRecoveryConfig,
    G1CerebellarRecoveryController,
    evaluate_g1_cerebellar_recovery_regime,
)
from rosclaw.simforge.g1_recovery_evolution import (
    G1MomentumUnloadingEvolution,
    G1RecoveryEvolutionDecision,
)
from rosclaw.simforge.g1_recovery_quality import (
    compare_g1_momentum_unloading,
    compare_g1_naturalness,
    compare_g1_recovery,
    measure_g1_recovery_quality,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters

_HASH_A = "sha256:" + "1" * 64
_HASH_B = "sha256:" + "2" * 64


def _controller(
    config: G1CerebellarRecoveryConfig | None = None,
) -> G1CerebellarRecoveryController:
    return G1CerebellarRecoveryController(
        body_hash=_HASH_A,
        motion_hash=_HASH_B,
        regime_commitment="sha256:" + "3" * 64,
        regime_eligible=True,
        regime_reasons=(),
        standing_pose=np.zeros(29, dtype=np.float64),
        config=config,
    )


def test_recovery_is_transparent_until_contact_and_kick_foot_landing() -> None:
    controller = _controller()
    target = np.linspace(-0.2, 0.2, 29)

    no_contact = controller.adapt_target(
        target=target,
        policy_frame=500,
        timestamp_sec=10.0,
        ball_contact_detected=False,
        left_support=True,
        right_support=True,
    )
    contact_without_landing = controller.adapt_target(
        target=target,
        policy_frame=500,
        timestamp_sec=10.02,
        ball_contact_detected=True,
        left_support=True,
        right_support=False,
    )

    assert not no_contact.active
    assert not contact_without_landing.active
    np.testing.assert_array_equal(no_contact.target, target)
    np.testing.assert_array_equal(contact_without_landing.target, target)


def test_recovery_smoothly_blends_qualified_pose_after_landing() -> None:
    controller = _controller()
    target = np.ones(29, dtype=np.float64)
    controller.adapt_target(
        target=target,
        policy_frame=300,
        timestamp_sec=6.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
    )

    halfway = controller.adapt_target(
        target=target,
        policy_frame=470,
        timestamp_sec=9.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
    )
    complete = controller.adapt_target(
        target=target,
        policy_frame=520,
        timestamp_sec=10.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
    )

    assert halfway.active
    assert halfway.blend_fraction == 0.5
    assert complete.blend_fraction == 1.0
    assert complete.settling_fraction == 0.0
    # Non-roll joints receive only the 30% standing-pose blend.
    assert complete.target[0] == 0.7
    # Left hip roll also receives the bounded -0.05 rad posture bias.
    assert complete.target[1] == pytest.approx(0.65)
    receipt = controller.build_receipt(strict_replay=True)
    assert receipt.controller_hash == controller.controller_hash
    assert receipt.config_hash == controller.config_hash
    assert receipt.contact_latched
    assert receipt.kick_foot_landing_latched
    assert receipt.activation_policy_frame == 470
    assert receipt.peak_blend_fraction == 1.0
    assert receipt.strict_replay


def test_recovery_uses_a_second_smoothstep_for_upright_settling() -> None:
    controller = _controller(
        G1CerebellarRecoveryConfig(
            start_policy_frame=100,
            blend_frames=100,
            standing_pose_blend=0.20,
            roll_posture_bias_rad=0.0,
            settling_start_policy_frame=200,
            settling_blend_frames=100,
            settling_standing_pose_blend=0.40,
            settling_roll_posture_bias_rad=0.0,
            settling_waist_pitch_bias_rad=0.09,
        )
    )
    target = np.ones(29, dtype=np.float64)

    halfway = controller.adapt_target(
        target=target,
        policy_frame=250,
        timestamp_sec=5.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
    )

    assert halfway.blend_fraction == 1.0
    assert halfway.settling_fraction == 0.5
    assert halfway.target[0] == pytest.approx(0.70)
    assert halfway.target[14] == pytest.approx(0.745)
    receipt = controller.build_receipt(strict_replay=True)
    assert receipt.settling_activation_policy_frame == 250
    assert receipt.peak_settling_fraction == 0.5


def test_recovery_causally_smooths_only_upper_body_after_landing() -> None:
    controller = _controller(
        G1CerebellarRecoveryConfig(
            target_smoothing_alpha=0.70,
            target_smoothing_start_policy_frame=400,
            target_smoothing_joint_group="upper_body",
        )
    )
    before = np.zeros(29, dtype=np.float64)
    after = np.ones(29, dtype=np.float64)
    controller.adapt_target(
        target=before,
        policy_frame=399,
        timestamp_sec=7.98,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
    )
    effect = controller.adapt_target(
        target=after,
        policy_frame=400,
        timestamp_sec=8.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
    )

    np.testing.assert_array_equal(effect.target[:12], after[:12])
    np.testing.assert_allclose(effect.target[12:], 0.70)
    assert effect.active
    assert effect.smoothing_active
    assert effect.blend_fraction == 0.0
    receipt = controller.build_receipt(strict_replay=True)
    assert receipt.smoothing_activation_policy_frame == 400
    assert receipt.peak_smoothing_residual_rms_rad > 0.0


def test_recovery_quality_detects_lower_wobble_and_preserved_goal() -> None:
    baseline_trace = _trajectory(wobble_scale=1.0)
    candidate_trace = _trajectory(wobble_scale=0.35)
    baseline = measure_g1_recovery_quality(baseline_trace)
    candidate = measure_g1_recovery_quality(candidate_trace)
    result = {
        "success": True,
        "goal_crossed": True,
        "target_zone_hit": True,
        "target_error_m": 0.2,
        "ball_speed_mps": 6.0,
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "support_foot_slip_m": 0.01,
    }

    comparison = compare_g1_recovery(
        baseline=baseline,
        candidate=candidate,
        baseline_result=result,
        candidate_result=result,
    )

    assert comparison.passed
    assert comparison.tail_wobble_reduction > 0.5
    assert comparison.tail_angular_velocity_reduction > 0.5
    assert candidate.terminal_bilateral_support
    assert candidate.settling_time_sec is not None
    assert candidate.post_contact_pelvis_path_length_m > 0.0
    assert candidate.post_contact_pelvis_displacement_m >= 0.0
    assert candidate.post_contact_backward_reversal_m >= 0.0
    assert candidate.post_contact_lateral_peak_return_m >= 0.0


def test_recovery_quality_measures_reversal_in_the_ball_travel_frame() -> None:
    trajectory = _trajectory(wobble_scale=0.35)
    contact = int(np.flatnonzero(trajectory["contact_impulse"] > 0.0)[0])
    progress = np.linspace(0.0, 1.0, len(trajectory["time"]) - contact)
    trajectory["pelvis_pose"][contact:, 0] = np.where(
        progress <= 0.5,
        progress,
        0.5 - 0.8 * (progress - 0.5),
    )
    trajectory["pelvis_pose"][contact:, 1] = np.where(
        progress <= 0.5,
        1.2 * progress,
        0.6 - 1.6 * (progress - 0.5),
    )

    quality = measure_g1_recovery_quality(trajectory)

    assert quality.post_contact_forward_peak_advance_m == pytest.approx(0.5, abs=0.01)
    assert quality.post_contact_backward_reversal_m == pytest.approx(0.4, abs=0.01)
    assert quality.post_contact_lateral_peak_return_m == pytest.approx(0.8, abs=0.01)


def test_momentum_unloading_gate_requires_measured_motion_and_replay_gains() -> None:
    parent = replace(
        measure_g1_recovery_quality(_trajectory(wobble_scale=1.0)),
        post_contact_pelvis_path_length_m=4.0,
        post_contact_pelvis_displacement_m=2.0,
        post_contact_pelvis_max_excursion_m=2.1,
        post_contact_support_transition_count=20,
        settling_time_sec=5.0,
    )
    candidate = replace(
        measure_g1_recovery_quality(_trajectory(wobble_scale=0.35)),
        post_contact_pelvis_path_length_m=1.8,
        post_contact_pelvis_displacement_m=0.5,
        post_contact_pelvis_max_excursion_m=0.6,
        post_contact_support_transition_count=12,
        settling_time_sec=3.8,
    )
    parent_result = _successful_result(target_error_m=0.09, ball_speed_mps=5.0)
    candidate_result = _successful_result(target_error_m=0.14, ball_speed_mps=5.6)

    promoted = compare_g1_momentum_unloading(
        parent=parent,
        candidate=candidate,
        parent_result=parent_result,
        candidate_result=candidate_result,
        parent_strict_replay=True,
        candidate_strict_replay=True,
    )
    replay_rejected = compare_g1_momentum_unloading(
        parent=parent,
        candidate=candidate,
        parent_result=parent_result,
        candidate_result=candidate_result,
        parent_strict_replay=True,
        candidate_strict_replay=False,
    )

    assert promoted.passed
    assert promoted.pelvis_path_reduction == pytest.approx(0.55)
    assert promoted.pelvis_displacement_reduction == pytest.approx(0.75)
    assert promoted.tail_wobble_reduction > 0.5
    assert not replay_rejected.passed
    assert replay_rejected.reasons == ("strict_replay_missing",)

    evolution = G1MomentumUnloadingEvolution.evaluate(
        body_hash=_HASH_A,
        kick_prior_hash=_HASH_B,
        recovery_controller_hash="sha256:" + "6" * 64,
        recovery_config_hash="sha256:" + "7" * 64,
        regime_commitment="sha256:" + "4" * 64,
        parent=ShotParameters(),
        candidate=ShotParameters(stance_offset_y=-0.08, swing_amplitude=1.125),
        parent_metrics=parent,
        candidate_metrics=candidate,
        comparison=promoted,
    )
    selected, selected_receipt = evolution.route(regime_commitment="sha256:" + "4" * 64)
    fallback, fallback_receipt = evolution.route(regime_commitment="sha256:" + "5" * 64)

    assert evolution.decision is G1RecoveryEvolutionDecision.SIM_CHAMPION
    assert selected.policy_hash == evolution.candidate.policy_hash
    assert selected_receipt.used_candidate
    assert selected_receipt.recovery_controller_hash == evolution.recovery_controller_hash
    assert selected_receipt.recovery_config_hash == evolution.recovery_config_hash
    assert fallback.policy_hash == evolution.parent.policy_hash
    assert not fallback_receipt.used_candidate
    assert fallback_receipt.fallback_reason == "out_of_evidence_regime"
    assert fallback_receipt.rollback_target_hash == evolution.parent.policy_hash
    assert replace(evolution, recovery_config_hash="sha256:" + "8" * 64).candidate_hash != (
        evolution.candidate_hash
    )


def test_naturalness_gate_requires_jerk_gains_without_stability_regression() -> None:
    parent = replace(
        measure_g1_recovery_quality(_trajectory(wobble_scale=1.0)),
        post_contact_joint_acceleration_rms_rad_s2=20.0,
        post_contact_joint_jerk_rms_rad_s3=750.0,
        post_contact_leg_joint_jerk_rms_rad_s3=1000.0,
        post_contact_waist_joint_jerk_rms_rad_s3=680.0,
        post_contact_arm_joint_jerk_rms_rad_s3=400.0,
        tail_joint_jerk_rms_rad_s3=1.25,
        post_contact_pelvis_path_length_m=1.86,
        post_contact_pelvis_displacement_m=0.50,
        post_contact_support_transition_count=49,
        post_contact_backward_reversal_m=0.60,
        post_contact_lateral_peak_return_m=0.85,
        tail_wobble_index=0.124,
        settling_time_sec=4.26,
    )
    candidate = replace(
        parent,
        post_contact_joint_acceleration_rms_rad_s2=19.7,
        post_contact_joint_jerk_rms_rad_s3=735.0,
        post_contact_leg_joint_jerk_rms_rad_s3=992.0,
        post_contact_waist_joint_jerk_rms_rad_s3=655.0,
        post_contact_arm_joint_jerk_rms_rad_s3=365.0,
        tail_joint_jerk_rms_rad_s3=1.04,
        post_contact_pelvis_path_length_m=1.875,
        post_contact_pelvis_displacement_m=0.52,
        post_contact_backward_reversal_m=0.49,
        post_contact_lateral_peak_return_m=0.61,
        tail_wobble_index=0.115,
        settling_time_sec=4.30,
    )
    result = _successful_result(target_error_m=0.13, ball_speed_mps=5.6)

    promoted = compare_g1_naturalness(
        parent=parent,
        candidate=candidate,
        parent_result=result,
        candidate_result=result,
        parent_strict_replay=True,
        candidate_strict_replay=True,
    )
    rejected = compare_g1_naturalness(
        parent=parent,
        candidate=replace(candidate, post_contact_arm_joint_jerk_rms_rad_s3=410.0),
        parent_result=result,
        candidate_result=result,
        parent_strict_replay=True,
        candidate_strict_replay=True,
    )

    assert promoted.passed
    assert promoted.arm_joint_jerk_reduction == pytest.approx(0.0875)
    assert promoted.tail_joint_jerk_reduction > 0.15
    assert promoted.backward_reversal_reduction > 0.15
    assert promoted.lateral_peak_return_reduction > 0.25
    assert not rejected.passed
    assert "arm_jerk_reduction_below_gate" in rejected.reasons


def test_recovery_regime_gate_rejects_uncalibrated_dynamics() -> None:
    config = G1CerebellarRecoveryConfig()

    eligible, reasons = evaluate_g1_cerebellar_recovery_regime(
        support_friction=1.0,
        control_latency_ms=0.0,
        disturbance_n=80.0,
        config=config,
    )
    low_grip, low_grip_reasons = evaluate_g1_cerebellar_recovery_regime(
        support_friction=0.8,
        control_latency_ms=0.0,
        disturbance_n=0.0,
        config=config,
    )
    medium_push, medium_push_reasons = evaluate_g1_cerebellar_recovery_regime(
        support_friction=1.0,
        control_latency_ms=0.0,
        disturbance_n=60.0,
        config=config,
    )

    assert eligible and not reasons
    assert not low_grip
    assert low_grip_reasons == ("support_friction_below_calibrated_range",)
    assert not medium_push
    assert medium_push_reasons == ("disturbance_below_calibrated_recovery_range",)

    blocked = G1CerebellarRecoveryController(
        body_hash=_HASH_A,
        motion_hash=_HASH_B,
        regime_commitment="sha256:" + "4" * 64,
        regime_eligible=False,
        regime_reasons=low_grip_reasons,
        standing_pose=np.zeros(29, dtype=np.float64),
    )
    effect = blocked.adapt_target(
        target=np.ones(29, dtype=np.float64),
        policy_frame=520,
        timestamp_sec=10.0,
        ball_contact_detected=True,
        left_support=True,
        right_support=True,
    )
    receipt = blocked.build_receipt(strict_replay=True)
    assert not effect.active
    np.testing.assert_array_equal(effect.target, np.ones(29))
    assert not receipt.regime_eligible
    assert receipt.regime_reasons == low_grip_reasons


def _trajectory(*, wobble_scale: float) -> dict[str, np.ndarray]:
    time = np.arange(0.0, 8.02, 0.02)
    count = len(time)
    contact_index = 50
    elapsed = np.maximum(0.0, time - time[contact_index])
    envelope = np.exp(-0.45 * elapsed)
    roll = wobble_scale * 0.12 * envelope * np.sin(2.0 * elapsed)
    pitch = wobble_scale * 0.06 * envelope * np.sin(1.5 * elapsed)
    quaternion = np.zeros((count, 4), dtype=np.float64)
    # Small roll/pitch fixture; the exact yaw-free quaternion is sufficient for
    # the metric's deterministic Euler conversion.
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    quaternion[:, 0] = cr * cp
    quaternion[:, 1] = sr * cp
    quaternion[:, 2] = cr * sp
    quaternion[:, 3] = -sr * sp
    pelvis = np.zeros((count, 7), dtype=np.float64)
    pelvis[:, 0] = wobble_scale * 0.04 * envelope * np.sin(1.8 * elapsed)
    pelvis[:, 1] = wobble_scale * 0.05 * envelope * np.cos(1.6 * elapsed)
    pelvis[:, 2] = 0.78
    pelvis[:, 3] = 1.0
    joint_velocity = np.repeat(
        (wobble_scale * 0.20 * envelope * np.sin(1.7 * elapsed))[:, None],
        29,
        axis=1,
    )
    impulse = np.zeros(count, dtype=np.float64)
    impulse[contact_index:] = 1.0
    ball_velocity = np.zeros((count, 3), dtype=np.float64)
    ball_velocity[contact_index:, 0] = 1.0
    return {
        "time": time,
        "torso_quaternion": quaternion,
        "pelvis_pose": pelvis,
        "joint_velocity": joint_velocity,
        "com_y_relative": wobble_scale * 0.04 * envelope * np.sin(1.4 * elapsed),
        "left_foot_contact": np.ones(count, dtype=bool),
        "right_foot_contact": np.ones(count, dtype=bool),
        "contact_impulse": impulse,
        "ball_velocity": ball_velocity,
    }


def _successful_result(*, target_error_m: float, ball_speed_mps: float) -> dict[str, object]:
    return {
        "success": True,
        "goal_crossed": True,
        "target_zone_hit": True,
        "target_error_m": target_error_m,
        "ball_speed_mps": ball_speed_mps,
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "support_foot_slip_m": 0.02,
    }
