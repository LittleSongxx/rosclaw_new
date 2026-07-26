from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_feedback_holdout import (
    FeedbackHoldoutCase,
    FeedbackHoldoutRegime,
    G1FeedbackHoldoutValidation,
    HistoricalMotionCase,
    run_g1_feedback_holdout,
)


def test_holdout_gate_requires_rescue_and_exact_history() -> None:
    regime = FeedbackHoldoutRegime("rescue", 1.0, 0.0, 0.0, 80.0)
    case = FeedbackHoldoutCase(
        regime=regime,
        scenario_commitment="sha256:" + "1" * 64,
        baseline_status="JOINT_LIMIT_EXCEEDED",
        feedback_status="SUCCESS",
        baseline_success=False,
        feedback_success=True,
        baseline_roll_peak_rad=0.7,
        feedback_roll_peak_rad=0.4,
        baseline_fall=True,
        feedback_fall=False,
        baseline_joint_violation=True,
        feedback_joint_violation=False,
        baseline_torque_violation=False,
        feedback_torque_violation=False,
        correction_applied=True,
        transparent_when_inactive=True,
        strict_replay=True,
    )
    history = HistoricalMotionCase(
        scenario_id="historical",
        scenario_commitment="sha256:" + "2" * 64,
        baseline_status="SUCCESS",
        feedback_status="SUCCESS",
        correction_applied=False,
        result_exact=True,
        physical_trajectory_exact=True,
    )
    result = G1FeedbackHoldoutValidation(
        body_hash="sha256:" + "3" * 64,
        kick_prior_hash="sha256:" + "4" * 64,
        holdout_cases=(case,),
        historical_cases=(history,),
        baseline_success_rate=0.0,
        feedback_success_rate=1.0,
        rescue_count=1,
        deadline_miss_count=0,
        simulation_episode_count=5,
    )

    assert result.passed


def test_holdout_rejects_evidence_inside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        run_g1_feedback_holdout(
            asset_root=tmp_path / "missing",
            output_path=tmp_path / "evidence.json",
            source_checkout=tmp_path,
        )
