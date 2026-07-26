"""Phase-aware error monitoring helpers for feedback controllers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from rosclaw.feedback.contracts import ErrorState


@dataclass(frozen=True)
class ErrorThresholds:
    warning: Mapping[str, float]
    critical: Mapping[str, float]


@dataclass(frozen=True)
class ErrorAssessment:
    warning_signals: tuple[str, ...]
    critical_signals: tuple[str, ...]
    weighted_rms: float


class ErrorMonitor:
    def __init__(
        self,
        thresholds: ErrorThresholds,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.weights = dict(weights or {})

    def assess(self, error: ErrorState) -> ErrorAssessment:
        warning = tuple(
            sorted(
                signal
                for signal, limit in self.thresholds.warning.items()
                if abs(error.value.get(signal, 0.0)) > limit
            )
        )
        critical = tuple(
            sorted(
                signal
                for signal, limit in self.thresholds.critical.items()
                if abs(error.value.get(signal, 0.0)) > limit
            )
        )
        denominator = sum(self.weights.get(signal, 1.0) for signal in error.value)
        weighted_rms = (
            math.sqrt(
                sum(
                    self.weights.get(signal, 1.0) * value * value
                    for signal, value in error.value.items()
                )
                / denominator
            )
            if denominator
            else 0.0
        )
        return ErrorAssessment(warning, critical, weighted_rms)
