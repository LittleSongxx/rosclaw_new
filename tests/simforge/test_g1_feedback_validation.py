from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_feedback_validation import (
    G1FeedbackABCase,
    G1FeedbackMetrics,
    G1FeedbackValidation,
    run_g1_feedback_validation,
)


def _metrics(*, success: bool, fall: bool = False) -> G1FeedbackMetrics:
    return G1FeedbackMetrics(
        status="SUCCESS" if success else "POST_KICK_FALL",
        success=success,
        target_error_m=0.1,
        support_foot_slip_m=0.01,
        com_margin_min_m=0.02,
        torso_roll_peak_rad=0.2,
        torso_pitch_peak_rad=0.2,
        post_kick_fall=fall,
        joint_limit_violation=False,
        torque_limit_violation=False,
        robustness=0.02,
    )


def test_feedback_validation_requires_rescue_without_nominal_regression() -> None:
    nominal = G1FeedbackABCase(
        scenario_id="nominal",
        scenario_commitment="sha256:" + "1" * 64,
        disturbance_n=0.0,
        baseline=_metrics(success=True),
        feedback=_metrics(success=True),
        feedback_receipt={},
        trajectory_strict_replay=True,
    )
    rescued = G1FeedbackABCase(
        scenario_id="disturbed",
        scenario_commitment="sha256:" + "2" * 64,
        disturbance_n=80.0,
        baseline=_metrics(success=False, fall=True),
        feedback=_metrics(success=True),
        feedback_receipt={},
        trajectory_strict_replay=True,
    )
    result = G1FeedbackValidation(
        body_hash="sha256:" + "3" * 64,
        kick_prior_hash="sha256:" + "4" * 64,
        backend_commit="abc",
        cases=(nominal, rescued),
        deadline_compliance_rate=1.0,
        baseline_success_rate=0.5,
        feedback_success_rate=1.0,
        rescue_count=1,
        nominal_no_regression=True,
    )

    assert result.passed
    assert result.to_dict()["claims"]["real_hardware"] is False


def test_feedback_validation_rejects_evidence_inside_source_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        run_g1_feedback_validation(
            asset_root=tmp_path / "missing-assets",
            output_path=tmp_path / "evidence.json",
            source_checkout=tmp_path,
            disturbances_n=(0.0,),
        )
