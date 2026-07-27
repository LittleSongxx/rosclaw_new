from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rosclaw.simforge.g1_causal_feedback_validation import (
    AttitudeWindowMetrics,
    G1CausalFeedbackValidation,
    _attitude_error,
    run_g1_causal_feedback_validation,
)
from rosclaw.simforge.g1_feedback_validation import G1FeedbackMetrics


def _episode_metrics(*, success: bool) -> G1FeedbackMetrics:
    return G1FeedbackMetrics(
        status="SUCCESS" if success else "POST_KICK_FALL",
        success=success,
        target_error_m=0.1,
        support_foot_slip_m=0.01,
        com_margin_min_m=0.01,
        torso_roll_peak_rad=0.4,
        torso_pitch_peak_rad=0.2,
        post_kick_fall=not success,
        joint_limit_violation=not success,
        torque_limit_violation=False,
        robustness=0.01,
    )


def _window(area: float, peak: float) -> AttitudeWindowMetrics:
    return AttitudeWindowMetrics(
        start_sec=4.6,
        end_sec=13.0,
        integrated_error_rad_sec=area,
        peak_error_rad=peak,
        tail_mean_error_rad=0.2,
    )


def test_causal_gate_requires_effect_isolation_replay_and_attitude_improvement() -> None:
    result = G1CausalFeedbackValidation(
        body_hash="sha256:" + "1" * 64,
        kick_prior_hash="sha256:" + "2" * 64,
        backend_commit="abc",
        scenario_id="causal",
        scenario_commitment="sha256:" + "3" * 64,
        disturbance_n=80.0,
        baseline=_episode_metrics(success=False),
        feedback=_episode_metrics(success=True),
        baseline_window=_window(2.8, 0.65),
        feedback_window=_window(2.4, 0.50),
        baseline_early_window=_window(0.4, 0.3),
        feedback_early_window=_window(0.42, 0.31),
        activation_start_sec=5.6,
        activation_end_sec=7.8,
        active_trace_samples=100,
        max_projected_residual_rad=0.08,
        identical_pre_activation_prefix=True,
        counterfactual_diverged_after_activation=True,
        strict_replay=True,
        deadline_compliance_rate=1.0,
        feedback_receipt={},
    )

    assert result.passed
    assert result.early_transient_regression
    assert result.to_dict()["promotion_assessment"]["eligible"] is False


def test_attitude_error_uses_roll_pitch_not_quaternion_component_norm() -> None:
    half = np.pi / 8.0
    quaternion = np.asarray([[np.cos(half), np.sin(half), 0.0, 0.0]])

    assert _attitude_error(quaternion)[0] == pytest.approx(np.pi / 4.0)


def test_causal_validation_rejects_evidence_inside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        run_g1_causal_feedback_validation(
            asset_root=tmp_path / "missing",
            output_path=tmp_path / "causal.json",
            source_checkout=tmp_path,
        )
