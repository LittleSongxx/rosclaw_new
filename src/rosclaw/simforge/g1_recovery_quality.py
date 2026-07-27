"""Trajectory-derived post-kick stability metrics and matched A/B gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class G1RecoveryQuality:
    contact_time_sec: float
    evaluation_start_sec: float
    evaluation_duration_sec: float
    tail_window_sec: float
    torso_angular_velocity_rms_rad_s: float
    torso_tilt_rms_rad: float
    com_lateral_velocity_rms_m_s: float
    pelvis_planar_velocity_rms_m_s: float
    joint_velocity_rms_rad_s: float
    tail_torso_angular_velocity_rms_rad_s: float
    tail_torso_tilt_rms_rad: float
    tail_com_lateral_velocity_rms_m_s: float
    tail_pelvis_planar_velocity_rms_m_s: float
    tail_joint_velocity_rms_rad_s: float
    tail_wobble_index: float
    bilateral_support_fraction: float
    terminal_bilateral_support: bool
    terminal_stable_duration_sec: float
    settling_time_sec: float | None
    schema_version: str = "rosclaw.g1_goalforge.recovery_quality.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1RecoveryComparison:
    passed: bool
    goal_outcome_preserved: bool
    safety_preserved: bool
    terminal_bilateral_support: bool
    tail_angular_velocity_reduction: float
    tail_tilt_reduction: float
    tail_pelvis_velocity_reduction: float
    tail_joint_velocity_reduction: float
    tail_wobble_reduction: float
    reasons: tuple[str, ...]
    schema_version: str = "rosclaw.g1_goalforge.recovery_comparison.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def measure_g1_recovery_quality(
    trajectory: Mapping[str, np.ndarray],
    *,
    evaluation_delay_sec: float = 1.0,
    tail_window_sec: float = 2.0,
    stable_dwell_sec: float = 0.5,
) -> G1RecoveryQuality:
    """Measure motion after contact instead of reusing the loose fall gate.

    The wobble index is a deterministic comparison score, not a physical unit:
    angular velocity + planar pelvis velocity + half torso tilt + one quarter
    whole-body joint velocity.  Its components are always reported separately.
    """

    if evaluation_delay_sec < 0.0 or tail_window_sec <= 0.0 or stable_dwell_sec <= 0.0:
        raise ValueError("recovery metric windows must be positive")
    required = {
        "time",
        "torso_quaternion",
        "pelvis_pose",
        "joint_velocity",
        "com_y_relative",
        "left_foot_contact",
        "right_foot_contact",
        "contact_impulse",
    }
    missing = required.difference(trajectory)
    if missing:
        raise ValueError(f"recovery trajectory is missing fields: {sorted(missing)}")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    quaternion = np.asarray(trajectory["torso_quaternion"], dtype=np.float64)
    pelvis = np.asarray(trajectory["pelvis_pose"], dtype=np.float64)
    joint_velocity = np.asarray(trajectory["joint_velocity"], dtype=np.float64)
    com_y = np.asarray(trajectory["com_y_relative"], dtype=np.float64)
    left_support = np.asarray(trajectory["left_foot_contact"], dtype=bool)
    right_support = np.asarray(trajectory["right_foot_contact"], dtype=bool)
    impulse = np.asarray(trajectory["contact_impulse"], dtype=np.float64)
    count = len(time)
    expected = {
        "torso_quaternion": (count, 4),
        "pelvis_pose": (count, 7),
        "joint_velocity": (count, 29),
        "com_y_relative": (count,),
        "left_foot_contact": (count,),
        "right_foot_contact": (count,),
        "contact_impulse": (count,),
    }
    actual_shapes = {
        "torso_quaternion": quaternion.shape,
        "pelvis_pose": pelvis.shape,
        "joint_velocity": joint_velocity.shape,
        "com_y_relative": com_y.shape,
        "left_foot_contact": left_support.shape,
        "right_foot_contact": right_support.shape,
        "contact_impulse": impulse.shape,
    }
    invalid = [name for name, shape in expected.items() if actual_shapes[name] != shape]
    if count < 3 or time.ndim != 1 or invalid:
        raise ValueError(f"recovery trajectory shapes are invalid: {invalid}")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("recovery trajectory time must be strictly increasing")
    if not all(
        np.all(np.isfinite(value))
        for value in (time, quaternion, pelvis, joint_velocity, com_y, impulse)
    ):
        raise ValueError("recovery trajectory must contain only finite values")
    contact_indices = np.flatnonzero(np.diff(impulse, prepend=0.0) > 1e-9)
    if not len(contact_indices):
        raise ValueError("recovery trajectory does not contain a ball-contact impulse")
    contact_index = int(contact_indices[0])
    contact_time = float(time[contact_index])
    evaluation_start = min(float(time[-1]), contact_time + evaluation_delay_sec)
    evaluation = time >= evaluation_start - 1e-12
    tail_start = max(evaluation_start, float(time[-1]) - tail_window_sec)
    tail = time >= tail_start - 1e-12
    if not np.any(evaluation) or not np.any(tail):
        raise ValueError("recovery trajectory is shorter than its metric windows")

    roll, pitch = _roll_pitch(quaternion)
    roll = np.unwrap(roll)
    pitch = np.unwrap(pitch)
    roll_rate = np.gradient(roll, time)
    pitch_rate = np.gradient(pitch, time)
    angular_speed = np.hypot(roll_rate, pitch_rate)
    tilt = np.hypot(roll, pitch)
    com_velocity = np.gradient(com_y, time)
    pelvis_velocity = np.linalg.norm(
        np.gradient(pelvis[:, :2], axis=0) / np.gradient(time)[:, None],
        axis=1,
    )
    joint_speed = np.sqrt(np.mean(np.square(joint_velocity), axis=1))
    bilateral = left_support & right_support

    tail_angular = _rms(angular_speed[tail])
    tail_tilt = _rms(tilt[tail])
    tail_com_velocity = _rms(com_velocity[tail])
    tail_pelvis_velocity = _rms(pelvis_velocity[tail])
    tail_joint_velocity = _rms(joint_speed[tail])
    wobble = tail_angular + tail_pelvis_velocity + 0.50 * tail_tilt + 0.25 * tail_joint_velocity
    stable = (
        (angular_speed <= 0.35) & (tilt <= 0.35) & (pelvis_velocity <= 0.25) & (joint_speed <= 0.55)
    )
    first_evaluation_index = int(np.flatnonzero(evaluation)[0])
    final_stable_start = len(time)
    for index in range(len(time) - 1, first_evaluation_index - 1, -1):
        if not stable[index]:
            break
        final_stable_start = index
    terminal_duration = (
        float(time[-1] - time[final_stable_start]) if final_stable_start < len(time) else 0.0
    )
    settling_time = (
        max(0.0, float(time[final_stable_start] - contact_time))
        if terminal_duration + 1e-12 >= stable_dwell_sec
        else None
    )
    return G1RecoveryQuality(
        contact_time_sec=contact_time,
        evaluation_start_sec=evaluation_start,
        evaluation_duration_sec=float(time[-1] - evaluation_start),
        tail_window_sec=float(time[-1] - tail_start),
        torso_angular_velocity_rms_rad_s=_rms(angular_speed[evaluation]),
        torso_tilt_rms_rad=_rms(tilt[evaluation]),
        com_lateral_velocity_rms_m_s=_rms(com_velocity[evaluation]),
        pelvis_planar_velocity_rms_m_s=_rms(pelvis_velocity[evaluation]),
        joint_velocity_rms_rad_s=_rms(joint_speed[evaluation]),
        tail_torso_angular_velocity_rms_rad_s=tail_angular,
        tail_torso_tilt_rms_rad=tail_tilt,
        tail_com_lateral_velocity_rms_m_s=tail_com_velocity,
        tail_pelvis_planar_velocity_rms_m_s=tail_pelvis_velocity,
        tail_joint_velocity_rms_rad_s=tail_joint_velocity,
        tail_wobble_index=wobble,
        bilateral_support_fraction=float(np.mean(bilateral[evaluation])),
        terminal_bilateral_support=bool(bilateral[-1]),
        terminal_stable_duration_sec=terminal_duration,
        settling_time_sec=settling_time,
    )


def compare_g1_recovery(
    *,
    baseline: G1RecoveryQuality,
    candidate: G1RecoveryQuality,
    baseline_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    minimum_wobble_reduction: float = 0.10,
) -> G1RecoveryComparison:
    if not 0.0 <= minimum_wobble_reduction < 1.0:
        raise ValueError("minimum_wobble_reduction must be in [0, 1)")
    goal_preserved = bool(
        candidate_result.get("success")
        and candidate_result.get("goal_crossed") == baseline_result.get("goal_crossed")
        and candidate_result.get("target_zone_hit") == baseline_result.get("target_zone_hit")
        and float(candidate_result.get("target_error_m", math.inf))
        <= float(baseline_result.get("target_error_m", math.inf)) + 1e-9
        and float(candidate_result.get("ball_speed_mps", 0.0))
        >= float(baseline_result.get("ball_speed_mps", 0.0)) - 1e-9
    )
    safety_preserved = bool(
        not candidate_result.get("post_kick_fall")
        and not candidate_result.get("joint_limit_violation")
        and not candidate_result.get("torque_limit_violation")
        and float(candidate_result.get("support_foot_slip_m", math.inf))
        <= max(0.08, float(baseline_result.get("support_foot_slip_m", 0.0)) + 0.005)
    )
    angular_reduction = _reduction(
        baseline.tail_torso_angular_velocity_rms_rad_s,
        candidate.tail_torso_angular_velocity_rms_rad_s,
    )
    tilt_reduction = _reduction(
        baseline.tail_torso_tilt_rms_rad,
        candidate.tail_torso_tilt_rms_rad,
    )
    pelvis_reduction = _reduction(
        baseline.tail_pelvis_planar_velocity_rms_m_s,
        candidate.tail_pelvis_planar_velocity_rms_m_s,
    )
    joint_reduction = _reduction(
        baseline.tail_joint_velocity_rms_rad_s,
        candidate.tail_joint_velocity_rms_rad_s,
    )
    wobble_reduction = _reduction(
        baseline.tail_wobble_index,
        candidate.tail_wobble_index,
    )
    reasons = []
    if not goal_preserved:
        reasons.append("goal_outcome_regressed")
    if not safety_preserved:
        reasons.append("safety_regressed")
    if not candidate.terminal_bilateral_support:
        reasons.append("terminal_bilateral_support_missing")
    if angular_reduction < 0.05:
        reasons.append("tail_angular_velocity_reduction_below_5pct")
    # Tilt is a posture term rather than an oscillation term.  A quiet,
    # bounded lean is acceptable; angular/pelvis/joint velocity must still
    # improve and the composite wobble gate remains mandatory.
    if candidate.tail_torso_tilt_rms_rad > 0.35:
        reasons.append("tail_tilt_exceeds_stable_bound")
    if pelvis_reduction < 0.05:
        reasons.append("tail_pelvis_velocity_reduction_below_5pct")
    if joint_reduction < 0.0:
        reasons.append("tail_joint_velocity_regressed")
    if wobble_reduction < minimum_wobble_reduction:
        reasons.append("tail_wobble_reduction_below_gate")
    return G1RecoveryComparison(
        passed=not reasons,
        goal_outcome_preserved=goal_preserved,
        safety_preserved=safety_preserved,
        terminal_bilateral_support=candidate.terminal_bilateral_support,
        tail_angular_velocity_reduction=angular_reduction,
        tail_tilt_reduction=tilt_reduction,
        tail_pelvis_velocity_reduction=pelvis_reduction,
        tail_joint_velocity_reduction=joint_reduction,
        tail_wobble_reduction=wobble_reduction,
        reasons=tuple(reasons),
    )


def _roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = (quaternion_wxyz[:, index] for index in range(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def _reduction(baseline: float, candidate: float) -> float:
    if baseline <= 1e-12:
        return 0.0 if candidate <= baseline + 1e-12 else -math.inf
    return (baseline - candidate) / baseline


__all__ = [
    "G1RecoveryComparison",
    "G1RecoveryQuality",
    "compare_g1_recovery",
    "measure_g1_recovery_quality",
]
