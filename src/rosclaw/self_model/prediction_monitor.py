"""Persistent prediction-residual trigger for bounded shadow adaptation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class AdaptationState(StrEnum):
    NORMAL = "NORMAL"
    SUSPECTED_SHIFT = "SUSPECTED_SHIFT"
    CONFIRMED_SHIFT = "CONFIRMED_SHIFT"
    SHADOW_LEARNING = "SHADOW_LEARNING"
    CANDIDATE_READY = "CANDIDATE_READY"
    CONSOLIDATED = "CONSOLIDATED"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True)
class PredictionResiduals:
    body_state: float
    contact_outcome: float
    contact_mode: float
    control_latency: float
    energy: float
    task_performance: float
    timestamp_ns: int
    episode_id: str
    schema_version: str = "rosclaw.self.prediction_residuals.v1"

    def __post_init__(self) -> None:
        values = self.components().values()
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("prediction residuals must be finite non-negative values")
        if self.timestamp_ns < 0 or not self.episode_id.strip():
            raise ValueError("prediction residual timestamp and episode must be valid")

    def components(self) -> dict[str, float]:
        return {
            "body_state": self.body_state,
            "contact_outcome": self.contact_outcome,
            "contact_mode": self.contact_mode,
            "control_latency": self.control_latency,
            "energy": self.energy,
            "task_performance": self.task_performance,
        }


@dataclass(frozen=True)
class AdaptationTriggerConfig:
    suspected_threshold: float = 0.35
    confirmed_threshold: float = 0.55
    suspected_persistence: int = 2
    confirmed_persistence: int = 3
    recovery_persistence: int = 3
    shadow_min_samples: int = 100
    minimum_improvement: float = 0.02
    maximum_anchor_degradation: float = 0.03
    weights: tuple[float, ...] = (0.25, 0.18, 0.12, 0.15, 0.10, 0.20)

    def __post_init__(self) -> None:
        if not 0.0 < self.suspected_threshold < self.confirmed_threshold <= 1.0:
            raise ValueError("adaptation thresholds must satisfy 0 < suspected < confirmed <= 1")
        if (
            min(
                self.suspected_persistence,
                self.confirmed_persistence,
                self.recovery_persistence,
                self.shadow_min_samples,
            )
            <= 0
        ):
            raise ValueError("adaptation persistence and sample requirements must be positive")
        if len(self.weights) != 6 or any(value < 0.0 for value in self.weights):
            raise ValueError("adaptation trigger requires six non-negative weights")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-12):
            raise ValueError("adaptation trigger weights must sum to one")
        if (
            not math.isfinite(self.minimum_improvement)
            or not math.isfinite(self.maximum_anchor_degradation)
            or self.minimum_improvement < 0.0
            or self.maximum_anchor_degradation < 0.0
        ):
            raise ValueError("adaptation improvement and retention limits must be non-negative")


@dataclass(frozen=True)
class AdaptationReceipt:
    previous_state: AdaptationState
    state: AdaptationState
    score: float
    learning_enabled: bool
    reason: str
    observation_count: int
    transition_hash: str
    registry_write_count: int = 0
    dds_opened: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.self.adaptation_receipt.v1"


class AdaptationTrigger:
    """Ignore transient noise and enable learning only after persistent shift."""

    def __init__(self, config: AdaptationTriggerConfig | None = None) -> None:
        self.config = config or AdaptationTriggerConfig()
        self.state = AdaptationState.NORMAL
        self.observation_count = 0
        self._suspected = 0
        self._confirmed = 0
        self._recovered = 0
        self._last_score = 0.0

    def observe(self, residuals: PredictionResiduals) -> AdaptationReceipt:
        previous = self.state
        score = self.score(residuals)
        self._last_score = score
        self.observation_count += 1
        reason = "observation recorded without opening adaptation"
        if self.state is AdaptationState.NORMAL:
            self._suspected = self._suspected + 1 if score >= self.config.suspected_threshold else 0
            if self._suspected >= self.config.suspected_persistence:
                self.state = AdaptationState.SUSPECTED_SHIFT
                self._confirmed = 0
                reason = "prediction residual persisted beyond the suspected threshold"
        elif self.state is AdaptationState.SUSPECTED_SHIFT:
            if score >= self.config.confirmed_threshold:
                self._confirmed += 1
                self._recovered = 0
                if self._confirmed >= self.config.confirmed_persistence:
                    self.state = AdaptationState.CONFIRMED_SHIFT
                    reason = "multi-signal prediction shift persisted and was confirmed"
            elif score < self.config.suspected_threshold:
                self._recovered += 1
                self._confirmed = 0
                if self._recovered >= self.config.recovery_persistence:
                    self.state = AdaptationState.NORMAL
                    self._suspected = 0
                    reason = "suspected shift decayed as transient noise"
            else:
                self._confirmed = 0
                self._recovered = 0
        return self._receipt(previous, reason)

    def begin_shadow_learning(self) -> AdaptationReceipt:
        if self.state is not AdaptationState.CONFIRMED_SHIFT:
            raise RuntimeError("shadow learning requires a confirmed shift")
        previous = self.state
        self.state = AdaptationState.SHADOW_LEARNING
        return self._receipt(previous, "confirmed shift opened SIM-only shadow learning")

    def candidate_update(
        self,
        *,
        sample_count: int,
        target_improvement: float,
        anchor_degradation: float,
        critical_safety_regressions: int,
        converged: bool,
    ) -> AdaptationReceipt:
        if self.state is not AdaptationState.SHADOW_LEARNING:
            raise RuntimeError("candidate evidence is accepted only during shadow learning")
        if sample_count < 0 or critical_safety_regressions < 0:
            raise ValueError("candidate sample and regression counts must be non-negative")
        if any(not math.isfinite(value) for value in (target_improvement, anchor_degradation)):
            raise ValueError("candidate adaptation metrics must be finite")
        if anchor_degradation < 0.0:
            raise ValueError("candidate anchor degradation must be non-negative")
        previous = self.state
        if critical_safety_regressions > 0:
            self.state = AdaptationState.ROLLBACK
            reason = "shadow candidate introduced a critical safety regression"
        elif (
            sample_count >= self.config.shadow_min_samples
            and target_improvement >= self.config.minimum_improvement
            and anchor_degradation <= self.config.maximum_anchor_degradation
            and converged
        ):
            self.state = AdaptationState.CANDIDATE_READY
            reason = (
                "shadow candidate reached sample, improvement, retention, and convergence gates"
            )
        else:
            reason = "shadow learning remains open; candidate evidence is incomplete"
        return self._receipt(previous, reason)

    def consolidate(self, *, matched_gate_passed: bool) -> AdaptationReceipt:
        if self.state is not AdaptationState.CANDIDATE_READY:
            raise RuntimeError("only a ready candidate may be consolidated")
        previous = self.state
        self.state = (
            AdaptationState.CONSOLIDATED if matched_gate_passed else AdaptationState.ROLLBACK
        )
        reason = (
            "matched evaluation consolidated the candidate"
            if matched_gate_passed
            else "matched evaluation rejected the candidate"
        )
        return self._receipt(previous, reason)

    def close_cycle(self, *, completion_evidence_hash: str) -> AdaptationReceipt:
        """Leave a terminal state only after content-addressed completion evidence.

        ROLLBACK and CONSOLIDATED remain fail-closed terminal states until the
        caller supplies the immutable receipt proving that rollback or champion
        consolidation actually completed.  Closing a cycle resets only the
        persistence counters; the lifetime observation count remains auditable.
        """

        if self.state not in {AdaptationState.CONSOLIDATED, AdaptationState.ROLLBACK}:
            raise RuntimeError("only a terminal adaptation cycle may be closed")
        if not _SHA256.fullmatch(completion_evidence_hash):
            raise ValueError("adaptation completion evidence must be a sha256 content hash")
        previous = self.state
        outcome = "consolidation" if previous is AdaptationState.CONSOLIDATED else "rollback"
        self.state = AdaptationState.NORMAL
        receipt = self._receipt(
            previous,
            f"verified {outcome} closed the adaptation cycle: {completion_evidence_hash}",
        )
        self._suspected = 0
        self._confirmed = 0
        self._recovered = 0
        self._last_score = 0.0
        return receipt

    def score(self, residuals: PredictionResiduals) -> float:
        values = residuals.components().values()
        return float(
            sum(
                weight * min(1.0, value)
                for weight, value in zip(self.config.weights, values, strict=True)
            )
        )

    def _receipt(self, previous: AdaptationState, reason: str) -> AdaptationReceipt:
        material: dict[str, Any] = {
            "previous_state": previous.value,
            "state": self.state.value,
            "score": self._last_score,
            "learning_enabled": self.state is AdaptationState.SHADOW_LEARNING,
            "reason": reason,
            "observation_count": self.observation_count,
        }
        return AdaptationReceipt(
            previous_state=previous,
            state=self.state,
            score=self._last_score,
            learning_enabled=self.state is AdaptationState.SHADOW_LEARNING,
            reason=reason,
            observation_count=self.observation_count,
            transition_hash=canonical_hash(material),
        )


__all__ = [
    "AdaptationReceipt",
    "AdaptationState",
    "AdaptationTrigger",
    "AdaptationTriggerConfig",
    "PredictionResiduals",
]
