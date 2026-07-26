from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_ilc_validation import (
    G1ILCTrial,
    G1ILCValidation,
    run_g1_ilc_validation,
)


def _trial(index: int) -> G1ILCTrial:
    return G1ILCTrial(
        trial=index,
        selected_learning_scale=0.0,
        update_accepted=False,
        receipt_hash="sha256:" + "1" * 64,
        trajectory_hash="sha256:" + "2" * 64,
        feedforward_hash=None,
        feedforward_peak_rad=0.0,
        tracking_error_rms=0.2,
        energy_proxy=10.0,
        safety_interventions=0,
        status="SUCCESS",
        success=True,
        target_error_m=0.1,
        torso_roll_peak_rad=0.2,
        deadline_miss_count=0,
        raw_error_path="/evidence/trial.npz",
    )


def test_ilc_campaign_gate_requires_ten_safe_trials_and_replay() -> None:
    result = G1ILCValidation(
        body_hash="sha256:" + "3" * 64,
        regime_hash="sha256:" + "4" * 64,
        kick_prior_hash="sha256:" + "5" * 64,
        trials=tuple(_trial(index) for index in range(1, 11)),
        monotonic_error=True,
        error_reduction=0.1,
        safety_not_increased=True,
        energy_within_limit=True,
        strict_replay=True,
        wrong_regime_rejected=True,
        probe_episode_count=45,
        simulation_episode_count=47,
    )

    assert result.passed
    assert result.to_dict()["claims"]["real_hardware"] is False


def test_ilc_campaign_rejects_evidence_inside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        run_g1_ilc_validation(
            asset_root=tmp_path / "missing",
            output_path=tmp_path / "evidence.json",
            source_checkout=tmp_path,
        )
