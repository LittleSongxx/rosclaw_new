from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from rosclaw.simforge.backends.unitree_mujoco_backend import GoalForgeEpisode
from rosclaw.simforge.g1_muscle_memory import (
    G1_MUSCLE_MEMORY_ACTIONS,
    G1_MUSCLE_MEMORY_OBSERVATIONS,
)
from rosclaw.simforge.g1_muscle_memory_training import (
    G1MuscleMemoryHoldoutSummary,
    G1MuscleMemoryTrainingConfig,
    _artifact_from_genome,
    _build_g1_muscle_memory_holdout_cases,
    _naturalness_preserved,
    _observation_rows,
    _reduction,
    build_g1_muscle_memory_cases,
)
from rosclaw.simforge.g1_recovery_quality import G1RecoveryQuality
from rosclaw.simforge.models import Partition


def test_training_config_rejects_unbounded_search() -> None:
    with pytest.raises(ValueError, match="population"):
        G1MuscleMemoryTrainingConfig(population_size=5)
    with pytest.raises(ValueError, match="std"):
        G1MuscleMemoryTrainingConfig(initial_std=0.04)


def test_training_and_private_holdout_are_partitioned() -> None:
    training = build_g1_muscle_memory_cases()
    holdout = _build_g1_muscle_memory_holdout_cases()

    assert all(case.scenario.partition is Partition.DEVELOPMENT for case in training)
    assert all(case.scenario.partition is Partition.HOLDOUT for case in holdout)
    assert {case.scenario.scenario_commitment for case in training}.isdisjoint(
        case.scenario.scenario_commitment for case in holdout
    )
    assert [case.feedback_enabled for case in training] == [False, False, True]
    assert training[0].scenario.ball_velocity_x_mps == 0.0
    assert training[0].scenario.ball_launch_delay_sec == 0.0
    assert holdout[0].scenario.ball_velocity_x_mps == 0.0


def test_sparse_genome_builds_bounded_content_addressed_actor() -> None:
    artifact = _artifact_from_genome(
        np.zeros(33, dtype=np.float64),
        body_hash="sha256:" + "a" * 64,
        motion_hash="sha256:" + "b" * 64,
        parent_config_hash="sha256:" + "c" * 64,
        dataset_hash="sha256:" + "d" * 64,
        observation_mean=(0.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS),
        observation_scale=(1.0,) * len(G1_MUSCLE_MEMORY_OBSERVATIONS),
        training_episode_count=3,
        training_seed=7,
    )

    assert len(artifact.weights) == len(G1_MUSCLE_MEMORY_ACTIONS)
    assert all(len(row) == len(G1_MUSCLE_MEMORY_OBSERVATIONS) for row in artifact.weights)
    assert artifact.activation_ceiling == "SIM_ONLY"
    assert artifact.artifact_hash.startswith("sha256:")
    assert len(artifact.artifact_hash) == 71


def test_holdout_summary_cannot_disclose_case_rows() -> None:
    summary = G1MuscleMemoryHoldoutSummary(
        suite_hash="e" * 64,
        case_count=3,
        passed_count=3,
        strict_replay_count=3,
        mean_score=0.1,
        minimum_score=0.0,
        qualified=True,
    )

    payload = asdict(summary)
    assert payload["case_rows_disclosed"] is False
    assert payload["recovery_equivalence_margin"] == pytest.approx(0.03)
    assert "cases" not in payload


def test_quality_reduction_uses_a_physical_scale_floor() -> None:
    assert _reduction(0.0, 0.01, scale_floor=0.05) == pytest.approx(-0.2)
    assert _reduction(0.10, 0.05, scale_floor=0.05) == pytest.approx(0.5)


def test_training_consumes_the_exact_recorded_online_observation() -> None:
    rows = np.arange(32, dtype=np.float64).reshape(2, 16)
    episode = cast(
        GoalForgeEpisode,
        SimpleNamespace(
            trajectory={
                "time": np.asarray((0.0, 0.02)),
                "recovery_proprioception": rows,
            }
        ),
    )

    observed = _observation_rows(episode)

    assert np.array_equal(observed, rows)


def test_naturalness_guard_rejects_reward_hacking_tail_wobble() -> None:
    parent = cast(
        G1RecoveryQuality,
        SimpleNamespace(
            settling_time_sec=4.72,
            post_contact_lateral_peak_return_m=0.65,
            post_contact_pelvis_path_length_m=1.67,
            tail_wobble_index=0.16,
            post_contact_leg_joint_jerk_rms_rad_s3=893.0,
            post_contact_support_transition_count=38,
        ),
    )
    candidate = cast(
        G1RecoveryQuality,
        SimpleNamespace(
            settling_time_sec=5.44,
            post_contact_lateral_peak_return_m=0.68,
            post_contact_pelvis_path_length_m=1.77,
            tail_wobble_index=0.28,
            post_contact_leg_joint_jerk_rms_rad_s3=842.0,
            post_contact_support_transition_count=38,
        ),
    )

    assert not _naturalness_preserved(parent, candidate)
