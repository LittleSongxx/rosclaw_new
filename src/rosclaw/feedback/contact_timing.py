"""Uncertainty-aware contact timing belief for GoalForge kick control."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ContactTimingConfig:
    expected_static_contact_phase: float = 0.416
    nominal_phase_rate_per_sec: float = 0.0788
    maximum_horizon_sec: float = 2.5
    confidence_threshold: float = 0.70
    maximum_intercept_miss_m: float = 0.24
    history_size: int = 12

    def __post_init__(self) -> None:
        if not 0.0 < self.expected_static_contact_phase < 1.0:
            raise ValueError("expected contact phase must be in (0, 1)")
        if not 0.0 < self.nominal_phase_rate_per_sec <= 1.0:
            raise ValueError("nominal phase rate must be in (0, 1]")
        if not 0.0 < self.maximum_horizon_sec <= 10.0:
            raise ValueError("contact timing horizon must be in (0, 10]")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("contact timing threshold must be in (0, 1)")
        if self.maximum_intercept_miss_m <= 0.0 or self.history_size <= 1:
            raise ValueError("contact timing miss bound and history size must be positive")


@dataclass(frozen=True)
class ContactTimingBelief:
    timestamp_ns: int
    predicted_contact_phase: float
    predicted_time_to_contact_sec: float
    predicted_contact_position_m: tuple[float, float, float]
    intercept_miss_m: float
    uncertainty: float
    confidence: float
    phase_residual_enabled: bool
    contact_observed: bool
    source: str
    schema_version: str = "rosclaw.feedback.contact_timing_belief.v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ContactTimingEstimator:
    """Fuse relative ball motion, phase prior, latency, and recent prediction churn."""

    required_signals = (
        "ball_relative_x_m",
        "ball_relative_y_m",
        "ball_relative_z_m",
        "ball_relative_vx_mps",
        "ball_relative_vy_mps",
        "ball_relative_vz_mps",
        "control_latency_ms",
        "sensor_quality",
        "contact_detected",
    )

    def __init__(self, config: ContactTimingConfig | None = None) -> None:
        self.config = config or ContactTimingConfig()
        self._phase_predictions: deque[float] = deque(maxlen=self.config.history_size)

    def reset(self) -> None:
        self._phase_predictions.clear()

    def update(
        self,
        *,
        timestamp_ns: int,
        policy_phase: float,
        actual: Mapping[str, float],
    ) -> ContactTimingBelief:
        missing = set(self.required_signals).difference(actual)
        if missing:
            raise ValueError(f"contact timing is missing signals: {sorted(missing)}")
        if not 0.0 <= policy_phase <= 1.0:
            raise ValueError("contact timing policy phase must be in [0, 1]")
        position = np.asarray(
            [
                actual["ball_relative_x_m"],
                actual["ball_relative_y_m"],
                actual["ball_relative_z_m"],
            ],
            dtype=np.float64,
        )
        velocity = np.asarray(
            [
                actual["ball_relative_vx_mps"],
                actual["ball_relative_vy_mps"],
                actual["ball_relative_vz_mps"],
            ],
            dtype=np.float64,
        )
        latency_sec = max(0.0, float(actual["control_latency_ms"]) / 1000.0)
        sensor_quality = float(np.clip(actual["sensor_quality"], 0.0, 1.0))
        contact = bool(actual["contact_detected"] >= 0.5)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("contact timing state must be finite")
        speed_squared = float(velocity @ velocity)
        moving = speed_squared >= 0.0025
        if contact:
            time_to_contact = 0.0
            predicted_position = position
            source = "observed_contact"
        elif moving:
            time_to_contact = float(
                np.clip(
                    -(position @ velocity) / speed_squared - latency_sec,
                    0.0,
                    self.config.maximum_horizon_sec,
                )
            )
            predicted_position = position + velocity * (time_to_contact + latency_sec)
            source = "relative_motion_intercept"
        else:
            time_to_contact = float(
                np.clip(
                    (self.config.expected_static_contact_phase - policy_phase)
                    / self.config.nominal_phase_rate_per_sec,
                    0.0,
                    self.config.maximum_horizon_sec,
                )
            )
            predicted_position = position
            source = "static_phase_prior"
        predicted_phase = float(
            np.clip(
                policy_phase
                + self.config.nominal_phase_rate_per_sec * (time_to_contact + latency_sec),
                0.0,
                1.0,
            )
        )
        intercept_miss = float(np.linalg.norm(predicted_position))
        history_churn = (
            float(np.std(tuple(self._phase_predictions)))
            if len(self._phase_predictions) >= 3
            else 0.0
        )
        self._phase_predictions.append(predicted_phase)
        uncertainty = float(
            np.clip(
                0.03
                + 0.55 * min(1.0, intercept_miss / self.config.maximum_intercept_miss_m)
                + 0.20 * min(1.0, latency_sec / 0.08)
                + 0.25 * (1.0 - sensor_quality)
                + 1.5 * history_churn
                + (0.20 if not moving and not contact else 0.0),
                0.0,
                1.0,
            )
        )
        confidence = math.exp(-2.0 * uncertainty)
        enabled = bool(
            not contact
            and moving
            and 0.0 < time_to_contact < self.config.maximum_horizon_sec
            and intercept_miss <= self.config.maximum_intercept_miss_m
            and confidence >= self.config.confidence_threshold
        )
        return ContactTimingBelief(
            timestamp_ns=timestamp_ns,
            predicted_contact_phase=predicted_phase,
            predicted_time_to_contact_sec=time_to_contact,
            predicted_contact_position_m=tuple(map(float, predicted_position)),
            intercept_miss_m=intercept_miss,
            uncertainty=uncertainty,
            confidence=confidence,
            phase_residual_enabled=enabled,
            contact_observed=contact,
            source=source,
        )


__all__ = [
    "ContactTimingBelief",
    "ContactTimingConfig",
    "ContactTimingEstimator",
]
