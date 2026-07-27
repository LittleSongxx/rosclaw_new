from __future__ import annotations

import json

import pytest

from rosclaw.simforge.g1_candidate_evaluation import (
    CandidateEvaluationCounts,
    CandidateEvaluationRow,
    _evaluation_scenarios,
    _gate,
    _load_checkpoint,
)


def test_candidate_evaluation_supports_resumable_disjoint_shards() -> None:
    counts = CandidateEvaluationCounts(recent=2, anchor=0, boundary=0, self_partition=0)

    first = _evaluation_scenarios(counts, suite_shard="recent-a")
    second = _evaluation_scenarios(counts, suite_shard="recent-b")

    assert len(first) == len(second) == 2
    assert {scenario.scenario_commitment for _, scenario in first}.isdisjoint(
        scenario.scenario_commitment for _, scenario in second
    )
    assert all(scenario.ball_launch_delay_sec > 0.0 for _, scenario in first + second)


def test_candidate_evaluation_rejects_empty_or_negative_counts() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CandidateEvaluationCounts(0, 0, 0, 0)
    with pytest.raises(ValueError, match="non-negative"):
        CandidateEvaluationCounts(-1, 1, 1, 1)


def test_partition_shard_gate_defers_anchor_check_instead_of_crashing() -> None:
    gate = _gate(
        rows=(),
        paired={"target_error_improvement_ci95": [0.1, 0.2]},
        replay_checks={"recent": True},
        counts=CandidateEvaluationCounts(1, 0, 0, 0),
        training_seed_count=8,
    )

    assert gate["anchor_success_delta"] is None
    assert gate["checks"]["historical_mean_degradation_lt_3pct"] is False
    assert gate["checks"]["critical_skill_degradation_lte_5pct"] is False
    assert gate["decision"] == "REJECTED"


def test_candidate_evaluation_row_rejects_non_finite_evidence() -> None:
    values = {
        "scenario_id": "scenario",
        "scenario_commitment": "sha256:" + "0" * 64,
        "replay_partition": "recent",
        "arm": "candidate_v3",
        "policy_version": 3,
        "policy_version_hash": "sha256:" + "1" * 64,
        "status": "SUCCESS",
        "success": True,
        "contact": True,
        "goal_crossed": True,
        "penalized_target_error_m": 0.1,
        "conditional_target_error_m": 0.1,
        "ball_speed_mps": 5.0,
        "fall": False,
        "joint_violation": False,
        "torque_violation": False,
        "actuator_saturation": False,
        "com_margin_min_m": float("inf"),
        "support_slip_m": 0.01,
        "tracking_rms_rad": 0.1,
        "energy_proxy": 1.0,
        "action_drift_rms": 0.0,
        "trajectory_hash": "sha256:" + "2" * 64,
        "inference_receipt_hash": None,
        "version_switch_count": 0,
    }

    with pytest.raises(ValueError, match="com_margin_min_m"):
        CandidateEvaluationRow(**values)


def test_checkpoint_migrates_only_legacy_unobserved_com_margin(tmp_path) -> None:
    path = tmp_path / "legacy.checkpoint"
    values = {
        "scenario_id": "scenario",
        "scenario_commitment": "sha256:" + "0" * 64,
        "replay_partition": "recent",
        "arm": "candidate_v3",
        "policy_version": 3,
        "policy_version_hash": "sha256:" + "1" * 64,
        "status": "NO_CONTACT",
        "success": False,
        "contact": False,
        "goal_crossed": False,
        "penalized_target_error_m": 2.0,
        "conditional_target_error_m": None,
        "ball_speed_mps": 0.0,
        "fall": False,
        "joint_violation": False,
        "torque_violation": False,
        "actuator_saturation": False,
        "com_margin_min_m": float("inf"),
        "support_slip_m": 0.0,
        "tracking_rms_rad": 0.1,
        "energy_proxy": 1.0,
        "action_drift_rms": 0.0,
        "trajectory_hash": "sha256:" + "2" * 64,
        "inference_receipt_hash": None,
        "version_switch_count": 0,
    }
    path.write_text(
        json.dumps({"identity": "expected", "rows": [values], "replay_checks": {}}),
        encoding="utf-8",
    )

    rows, replay = _load_checkpoint(path, expected_identity="expected")

    assert rows[0].com_margin_min_m == -1.0
    assert replay == {}
