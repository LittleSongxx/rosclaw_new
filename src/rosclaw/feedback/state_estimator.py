"""Deterministic reference-error state for the synchronous Feedback Plane."""

from __future__ import annotations

import math
from collections.abc import Mapping

from rosclaw.feedback.contracts import ErrorState


class FeedbackStateEstimator:
    """Compute error, derivative and anti-windup-bounded integral state."""

    def __init__(self, signals: tuple[str, ...], *, integral_limit: float = 10.0) -> None:
        if not signals:
            raise ValueError("signals must not be empty")
        if integral_limit <= 0.0:
            raise ValueError("integral_limit must be positive")
        self.signals = signals
        self.integral_limit = float(integral_limit)
        self.reset()

    def reset(self) -> None:
        self._previous_error: dict[str, float] | None = None
        self._integral = dict.fromkeys(self.signals, 0.0)
        self._previous_timestamp_ns: int | None = None

    def update(
        self,
        reference: Mapping[str, float],
        actual: Mapping[str, float],
        timestamp_ns: int,
    ) -> ErrorState:
        missing_reference = set(self.signals).difference(reference)
        missing_actual = set(self.signals).difference(actual)
        if missing_reference or missing_actual:
            raise ValueError(
                "missing feedback signals: "
                f"reference={sorted(missing_reference)},actual={sorted(missing_actual)}"
            )
        error = {
            signal: float(reference[signal]) - float(actual[signal]) for signal in self.signals
        }
        if any(not math.isfinite(value) for value in error.values()):
            raise ValueError("reference and actual values must be finite")
        derivative = dict.fromkeys(self.signals, 0.0)
        if self._previous_timestamp_ns is not None:
            dt = (timestamp_ns - self._previous_timestamp_ns) / 1_000_000_000.0
            if dt <= 0.0:
                raise ValueError("feedback timestamps must increase monotonically")
            assert self._previous_error is not None
            for signal in self.signals:
                derivative[signal] = (error[signal] - self._previous_error[signal]) / dt
                integrated = (
                    self._integral[signal]
                    + 0.5 * (error[signal] + self._previous_error[signal]) * dt
                )
                self._integral[signal] = max(
                    -self.integral_limit,
                    min(self.integral_limit, integrated),
                )
        self._previous_error = error
        self._previous_timestamp_ns = timestamp_ns
        return ErrorState(
            value=error,
            derivative=derivative,
            integral=self._integral,
            timestamp_ns=timestamp_ns,
        )
