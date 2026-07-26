from __future__ import annotations

import numpy as np
import pytest

from rosclaw.feedback.ilc import (
    BoundedILC,
    BoundedTrajectoryILC,
    ILCFeedforward,
    ILCTrajectory,
    ILCTrajectoryMemory,
    assess_ilc_convergence,
)


def test_ilc_update_is_bounded_body_bound_and_converges() -> None:
    ilc = BoundedILC(
        body_hash="sha256:" + "1" * 64,
        learning_gain=0.5,
        residual_limit=0.1,
        convergence_tolerance=1e-6,
    )
    error = np.full((8, 2), 1.0)
    first = ilc.update(error, source_receipt_hash="sha256:" + "2" * 64)
    second = ilc.update(error, source_receipt_hash="sha256:" + "3" * 64)

    assert first.residual_peak == pytest.approx(0.1)
    assert second.residual_peak == pytest.approx(0.1)
    assert second.converged
    assert second.snapshot.body_hash == "sha256:" + "1" * 64
    assert second.snapshot.bounded


def test_ilc_rejects_trial_shape_change_and_non_finite_error() -> None:
    ilc = BoundedILC(body_hash="sha256:" + "1" * 64)
    ilc.update(np.zeros((4, 1)), source_receipt_hash="sha256:" + "2" * 64)
    with pytest.raises(ValueError, match="shape"):
        ilc.update(np.zeros((3, 1)), source_receipt_hash="sha256:" + "2" * 64)
    with pytest.raises(ValueError, match="finite"):
        ilc.update(np.asarray([[np.nan]]), source_receipt_hash="sha256:" + "2" * 64)


def _trajectory(
    error: float,
    *,
    body_hash: str = "sha256:" + "1" * 64,
    regime_hash: str = "sha256:" + "2" * 64,
    energy: float = 10.0,
) -> ILCTrajectory:
    return ILCTrajectory(
        receipt_hash="sha256:" + "3" * 64,
        body_hash=body_hash,
        regime_hash=regime_hash,
        tracking_error=np.full((4, 2), error),
        feedforward_residual=np.zeros((4, 2)),
        energy=energy,
        safety_interventions=0,
    )


def test_ilc_memory_rejects_wrong_body_and_regime() -> None:
    memory = ILCTrajectoryMemory(
        body_hash="sha256:" + "1" * 64,
        regime_hash="sha256:" + "2" * 64,
        capacity=2,
    )
    memory.append(_trajectory(1.0))
    with pytest.raises(ValueError, match="wrong-body"):
        memory.append(_trajectory(0.8, body_hash="sha256:" + "4" * 64))
    with pytest.raises(ValueError, match="wrong-regime"):
        memory.append(_trajectory(0.8, regime_hash="sha256:" + "5" * 64))


def test_ilc_convergence_requires_monotonic_error_and_bounded_energy() -> None:
    result = assess_ilc_convergence(
        (_trajectory(1.0), _trajectory(0.8, energy=10.5), _trajectory(0.6, energy=10.8))
    )
    regression = assess_ilc_convergence((_trajectory(1.0), _trajectory(1.1)))

    assert result.passed
    assert result.error_reduction == pytest.approx(0.4)
    assert not regression.passed


def test_ilc_convergence_rejects_a_flat_error_sequence() -> None:
    result = assess_ilc_convergence((_trajectory(1.0), _trajectory(1.0)))

    assert result.monotonic_error
    assert not result.passed


def test_trajectory_ilc_is_smoothed_bounded_and_content_addressed() -> None:
    learner = BoundedTrajectoryILC(
        body_hash="sha256:" + "1" * 64,
        regime_hash="sha256:" + "2" * 64,
        joint_names=("a", "b"),
        learning_gain=0.5,
        residual_limit=0.04,
    )
    error = np.zeros((8, 2))
    error[4] = 1.0
    first = learner.update(
        previous=None,
        tracking_error=error,
        source_receipt_hash="sha256:" + "3" * 64,
    )
    second = learner.update(
        previous=first,
        tracking_error=error * 0.5,
        source_receipt_hash="sha256:" + "4" * 64,
    )

    assert isinstance(first, ILCFeedforward)
    assert first.values.flags.writeable is False
    assert np.max(np.abs(first.values)) <= 0.04
    assert np.count_nonzero(first.values[:, 0]) > 1
    assert first.trajectory_hash.startswith("sha256:")
    assert second.trial == 2
    assert second.trajectory_hash != first.trajectory_hash

    scaled = learner.update(
        previous=first,
        tracking_error=error * 0.5,
        source_receipt_hash="sha256:" + "4" * 64,
        learning_scale=0.5,
    )
    assert not np.array_equal(scaled.values, second.values)


def test_trajectory_ilc_rejects_wrong_regime_reuse() -> None:
    previous = ILCFeedforward(
        body_hash="sha256:" + "1" * 64,
        regime_hash="sha256:" + "9" * 64,
        joint_names=("a",),
        values=np.zeros((3, 1)),
        residual_limit=0.04,
        trial=1,
    )
    learner = BoundedTrajectoryILC(
        body_hash="sha256:" + "1" * 64,
        regime_hash="sha256:" + "2" * 64,
        joint_names=("a",),
    )

    with pytest.raises(ValueError, match="wrong-regime"):
        learner.update(
            previous=previous,
            tracking_error=np.zeros((3, 1)),
            source_receipt_hash="sha256:" + "3" * 64,
        )
