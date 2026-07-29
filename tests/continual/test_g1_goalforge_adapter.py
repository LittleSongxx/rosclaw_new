from __future__ import annotations

import numpy as np

from rosclaw.continual.contracts import SkillPhase
from rosclaw.continual.g1_goalforge import (
    G1_CONTINUAL_ACTIONS,
    G1_CONTINUAL_OBSERVATIONS,
    adapt_goalforge_episode,
    build_g1_policy_lineage,
)
from rosclaw.simforge.backends.unitree_mujoco_backend import GoalForgeEpisode
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GoalForgeResult,
    GoalForgeStatus,
    ShotParameters,
    hash_json,
)
from rosclaw.simforge.tasks.g1_goalforge.scenario import GoalForgeScenario
from tests.continual.helpers import digest


def test_goalforge_adapter_emits_version_pinned_self_bound_transitions() -> None:
    lineage = build_g1_policy_lineage(
        body_hash=digest("qualified-body"),
        kick_prior_hash=digest("kick-prior"),
        motion_hash=digest("motion"),
        backend_commit="abc123",
        torque_guard_scale=0.85,
    )
    episode = GoalForgeEpisode(
        scenario=_scenario(),
        parameters=ShotParameters(),
        result=_result(),
        receipt=None,
        artifact_root=None,
        trajectory=_trace(),
    )

    adapted = adapt_goalforge_episode(
        episode,
        policy=lineage.policy(2),
        strict_replay=True,
    )

    trajectory = adapted.trajectory
    assert trajectory.strict_replay
    assert tuple(trajectory.segments[0].observation) == G1_CONTINUAL_OBSERVATIONS
    assert tuple(trajectory.segments[0].residual_action) == G1_CONTINUAL_ACTIONS
    assert [segment.phase for segment in trajectory.segments] == [
        SkillPhase.STAND,
        SkillPhase.PREPARE,
        SkillPhase.WEIGHT_TRANSFER,
        SkillPhase.SWING,
        SkillPhase.CONTACT,
        SkillPhase.COMPLETE,
    ]
    assert all(segment.policy.version == 2 for segment in trajectory.segments)
    assert len(set(adapted.self_state_hashes)) == len(adapted.self_state_hashes)
    assert trajectory.has_critical_cost
    assert trajectory.segments[-1].cost.joint_limit == 1.0


def _scenario() -> GoalForgeScenario:
    return GoalForgeScenario(
        scenario_id="adapter-episode",
        partition=Partition.DEVELOPMENT,
        seed=1,
        seed_commitment=hash_json({"seed": 1}),
        generation=0,
        ball_x_m=1.0,
        ball_y_m=0.0,
        ball_velocity_x_mps=0.0,
        ball_velocity_y_mps=0.0,
        target_y_m=0.0,
        target_z_m=0.2,
        ball_mass_kg=0.41,
        ball_ground_friction=0.05,
        restitution=0.55,
        support_ground_friction=1.0,
        control_latency_ms=0.0,
        observation_noise_m=0.0,
        joint_zero_bias_rad=0.02,
        disturbance_n=35.0,
    )


def _result() -> GoalForgeResult:
    return GoalForgeResult(
        status=GoalForgeStatus.JOINT_LIMIT_EXCEEDED,
        success=False,
        physics_executed=True,
        contact_observed=True,
        kick_foot_contacted=True,
        goal_crossed=False,
        target_zone_hit=False,
        target_error_m=float("inf"),
        ball_speed_mps=4.0,
        ball_contact_time_sec=5.0,
        contact_impulse_ns=2.0,
        support_foot_slip_m=0.02,
        com_margin_min_m=0.03,
        torso_roll_peak_rad=0.3,
        torso_pitch_peak_rad=0.2,
        peak_torque_scale=0.8,
        joint_limit_violation=True,
        torque_limit_violation=False,
        actuator_saturation=False,
        post_kick_fall=False,
        post_kick_stability_time_sec=1.0,
        final_pelvis_height_m=0.75,
        physics_steps=7000,
        finite_state=True,
        robustness=-0.1,
    )


def _trace() -> dict[str, np.ndarray]:
    count = 7
    torque = np.zeros((count, 29), dtype=np.float64)
    torque[:, 0] = np.linspace(0.0, 20.0, count)
    return {
        "time": np.arange(count, dtype=np.float64) * 0.1,
        "joint_torque": torque,
        "torso_quaternion": np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        "com_y_relative": np.linspace(0.0, 0.04, count),
        "support_foot_slip": np.linspace(0.0, 0.02, count),
        "ball_lateral_error_m": np.linspace(0.1, 0.0, count),
        "policy_phase": np.asarray([0.0, 0.1, 0.3, 0.45, 0.5, 0.7, 1.0]),
        "contact_impulse": np.asarray([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0]),
    }
