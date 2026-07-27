from __future__ import annotations

import pytest

from rosclaw.feedback.contact_timing import ContactTimingEstimator


def _actual(**changes: float) -> dict[str, float]:
    value = {
        "ball_relative_x_m": 0.20,
        "ball_relative_y_m": 0.0,
        "ball_relative_z_m": 0.0,
        "ball_relative_vx_mps": -0.20,
        "ball_relative_vy_mps": 0.0,
        "ball_relative_vz_mps": 0.0,
        "control_latency_ms": 0.0,
        "sensor_quality": 1.0,
        "contact_detected": 0.0,
    }
    value.update(changes)
    return value


def test_contact_timing_enables_phase_only_for_confident_intercept() -> None:
    belief = ContactTimingEstimator().update(
        timestamp_ns=1,
        policy_phase=0.30,
        actual=_actual(),
    )

    assert belief.source == "relative_motion_intercept"
    assert belief.predicted_time_to_contact_sec == pytest.approx(1.0)
    assert belief.intercept_miss_m == pytest.approx(0.0)
    assert belief.confidence >= 0.70
    assert belief.phase_residual_enabled


def test_contact_timing_disables_static_and_uncertain_intercepts() -> None:
    estimator = ContactTimingEstimator()
    static = estimator.update(
        timestamp_ns=1,
        policy_phase=0.30,
        actual=_actual(ball_relative_vx_mps=0.0),
    )
    uncertain = estimator.update(
        timestamp_ns=2,
        policy_phase=0.31,
        actual=_actual(
            ball_relative_y_m=0.30,
            control_latency_ms=80.0,
            sensor_quality=0.2,
        ),
    )
    observed = estimator.update(
        timestamp_ns=3,
        policy_phase=0.41,
        actual=_actual(contact_detected=1.0),
    )

    assert static.source == "static_phase_prior"
    assert not static.phase_residual_enabled
    assert not uncertain.phase_residual_enabled
    assert observed.contact_observed
    assert not observed.phase_residual_enabled
