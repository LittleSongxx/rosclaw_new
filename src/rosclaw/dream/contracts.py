"""Immutable contracts for ROSClaw's bounded Dream Plane.

Dreaming is an asynchronous candidate-generation and consolidation activity.
It cannot write to the synchronous control loop, reveal private holdout rows or
turn model predictions into promotion evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.contracts import EvidenceLevel, EvidenceUsePolicy, TrainingEligibility

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _require_hash(label: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _optional_hash(label: str, value: str | None) -> None:
    if value is not None:
        _require_hash(label, value)


def _hash_tuple(values: tuple[str, ...], *, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    for value in normalized:
        _require_hash(label, value)
    return normalized


def _names(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be non-empty and unique")
    if any(not value.strip() for value in normalized):
        raise ValueError(f"{label} must not contain empty values")
    return normalized


class DreamType(StrEnum):
    REPLAY = "replay"
    COUNTERFACTUAL = "counterfactual"
    PROSPECTIVE = "prospective"
    SELF = "self"
    SOCIAL = "social"
    NIGHTMARE = "nightmare"


@dataclass(frozen=True)
class DreamBudget:
    """Hard resource and drift limits for one campaign."""

    max_gpu_seconds: float
    max_cpu_rollouts: int
    max_candidates: int
    max_wall_seconds: float
    max_policy_change: float
    max_anchor_drift: float
    schema_version: str = "rosclaw.dream.budget.v1"

    def __post_init__(self) -> None:
        for label in ("max_gpu_seconds", "max_wall_seconds", "max_policy_change", "max_anchor_drift"):
            value = getattr(self, label)
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.max_gpu_seconds < 0.0:
            raise ValueError("max_gpu_seconds must be non-negative")
        if self.max_cpu_rollouts < 0:
            raise ValueError("max_cpu_rollouts must be non-negative")
        if self.max_gpu_seconds == 0.0 and self.max_cpu_rollouts == 0:
            raise ValueError("at least one compute budget must be positive")
        if self.max_candidates <= 0 or self.max_wall_seconds <= 0.0:
            raise ValueError("candidate and wall-clock budgets must be positive")
        if not 0.0 <= self.max_policy_change <= 1.0:
            raise ValueError("max_policy_change must be in [0, 1]")
        if not 0.0 <= self.max_anchor_drift <= 1.0:
            raise ValueError("max_anchor_drift must be in [0, 1]")

    @property
    def budget_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "max_gpu_seconds": self.max_gpu_seconds,
            "max_cpu_rollouts": self.max_cpu_rollouts,
            "max_candidates": self.max_candidates,
            "max_wall_seconds": self.max_wall_seconds,
            "max_policy_change": self.max_policy_change,
            "max_anchor_drift": self.max_anchor_drift,
        }


@dataclass(frozen=True)
class DreamCampaign:
    """Content-addressed intent for one bounded offline learning run."""

    skill_growth_spec_hash: str
    body_hash: str
    parent_policy_hash: str
    trigger_kind: str
    trigger_evidence_hashes: tuple[str, ...]
    objectives: tuple[str, ...]
    constraint_hashes: tuple[str, ...]
    practice_snapshot_hashes: tuple[str, ...]
    collective_capsule_hashes: tuple[str, ...]
    historical_anchor_hashes: tuple[str, ...]
    boundary_suite_hashes: tuple[str, ...]
    private_holdout_commitment: str
    dream_types: tuple[DreamType, ...]
    learner_ids: tuple[str, ...]
    budget: DreamBudget
    schema_version: str = "rosclaw.dream.campaign.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.budget, DreamBudget):
            raise ValueError("budget must be a DreamBudget record")
        for label, value in (
            ("skill_growth_spec_hash", self.skill_growth_spec_hash),
            ("body_hash", self.body_hash),
            ("parent_policy_hash", self.parent_policy_hash),
            ("private_holdout_commitment", self.private_holdout_commitment),
        ):
            _require_hash(label, value)
        if not self.trigger_kind.strip():
            raise ValueError("trigger_kind must not be empty")
        object.__setattr__(
            self,
            "trigger_evidence_hashes",
            _hash_tuple(self.trigger_evidence_hashes, label="trigger_evidence_hashes"),
        )
        for label in ("constraint_hashes", "historical_anchor_hashes", "boundary_suite_hashes"):
            object.__setattr__(self, label, _hash_tuple(getattr(self, label), label=label))
        for label in ("practice_snapshot_hashes", "collective_capsule_hashes"):
            object.__setattr__(
                self,
                label,
                _hash_tuple(getattr(self, label), label=label, allow_empty=True),
            )
        object.__setattr__(self, "objectives", _names(self.objectives, label="objectives"))
        object.__setattr__(self, "learner_ids", _names(self.learner_ids, label="learner_ids"))
        dream_types = tuple(self.dream_types)
        if not dream_types or len(dream_types) != len(set(dream_types)):
            raise ValueError("dream_types must be non-empty and unique")
        if any(not isinstance(dream_type, DreamType) for dream_type in dream_types):
            raise ValueError("dream_types must contain recognized DreamType values")
        object.__setattr__(self, "dream_types", dream_types)
        if DreamType.SOCIAL in dream_types and not self.collective_capsule_hashes:
            raise ValueError("social dreaming requires collective experience capsules")
        if DreamType.REPLAY in dream_types and not (
            self.practice_snapshot_hashes or self.historical_anchor_hashes
        ):
            raise ValueError("replay dreaming requires practice or historical evidence")

    @property
    def campaign_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "skill_growth_spec_hash": self.skill_growth_spec_hash,
            "body_hash": self.body_hash,
            "parent_policy_hash": self.parent_policy_hash,
            "trigger_kind": self.trigger_kind,
            "trigger_evidence_hashes": list(self.trigger_evidence_hashes),
            "objectives": list(self.objectives),
            "constraint_hashes": list(self.constraint_hashes),
            "practice_snapshot_hashes": list(self.practice_snapshot_hashes),
            "collective_capsule_hashes": list(self.collective_capsule_hashes),
            "historical_anchor_hashes": list(self.historical_anchor_hashes),
            "boundary_suite_hashes": list(self.boundary_suite_hashes),
            "private_holdout_commitment": self.private_holdout_commitment,
            "dream_types": [dream_type.value for dream_type in self.dream_types],
            "learner_ids": list(self.learner_ids),
            "budget": self.budget.to_dict(),
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class DreamEpisode:
    """One dreamed or replayed episode with an immutable evidence label."""

    campaign_hash: str
    source_episode_hash: str
    body_hash: str
    policy_hash: str
    dream_type: DreamType
    evidence_policy: EvidenceUsePolicy
    uncertainty: float
    self_model_hash: str | None = None
    world_model_hash: str | None = None
    counterfactual_changes_hash: str | None = None
    predicted_outcome_hash: str | None = None
    simulated_outcome_hash: str | None = None
    execution_receipt_hash: str | None = None
    physics_replay_verified: bool = False
    schema_version: str = "rosclaw.dream.episode.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.dream_type, DreamType):
            raise ValueError("dream_type must be a recognized DreamType")
        if not isinstance(self.evidence_policy, EvidenceUsePolicy):
            raise ValueError("evidence_policy must be an EvidenceUsePolicy record")
        for label, value in (
            ("campaign_hash", self.campaign_hash),
            ("source_episode_hash", self.source_episode_hash),
            ("body_hash", self.body_hash),
            ("policy_hash", self.policy_hash),
        ):
            _require_hash(label, value)
        for label in (
            "self_model_hash",
            "world_model_hash",
            "counterfactual_changes_hash",
            "predicted_outcome_hash",
            "simulated_outcome_hash",
            "execution_receipt_hash",
        ):
            _optional_hash(label, getattr(self, label))
        if not math.isfinite(self.uncertainty) or not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be finite and in [0, 1]")
        if self.evidence_policy.level is EvidenceLevel.WORLD_MODEL and (
            self.world_model_hash is None or self.predicted_outcome_hash is None
        ):
            raise ValueError("world-model evidence requires model and prediction hashes")
        if self.evidence_policy.level in {
            EvidenceLevel.PHYSICS_REPLAY,
            EvidenceLevel.ACCELERATED_SIM,
        } and self.simulated_outcome_hash is None:
            raise ValueError("simulation evidence requires a simulated outcome")
        if (
            self.evidence_policy.level is EvidenceLevel.EXECUTION_RECEIPT
            and self.execution_receipt_hash is None
        ):
            raise ValueError("execution evidence requires a signed receipt hash")
        if self.dream_type is DreamType.COUNTERFACTUAL:
            if self.counterfactual_changes_hash is None:
                raise ValueError("counterfactual dreams require a change-set hash")
            if self.predicted_outcome_hash is None and self.simulated_outcome_hash is None:
                raise ValueError("counterfactual dreams require a predicted or simulated outcome")
        if self.dream_type is DreamType.PROSPECTIVE and (
            self.world_model_hash is None or self.predicted_outcome_hash is None
        ):
            raise ValueError("prospective dreams require a world model and prediction")

    @property
    def training_eligible(self) -> bool:
        return self.evidence_policy.training is not TrainingEligibility.DENIED

    @property
    def promotion_truth_allowed(self) -> bool:
        level = self.evidence_policy.level
        if level is EvidenceLevel.EXECUTION_RECEIPT:
            return self.execution_receipt_hash is not None
        if level is EvidenceLevel.PHYSICS_REPLAY:
            return self.physics_replay_verified and self.simulated_outcome_hash is not None
        return False

    @property
    def episode_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign_hash,
            "source_episode_hash": self.source_episode_hash,
            "body_hash": self.body_hash,
            "policy_hash": self.policy_hash,
            "dream_type": self.dream_type.value,
            "evidence_policy": self.evidence_policy.to_dict(),
            "uncertainty": self.uncertainty,
            "self_model_hash": self.self_model_hash,
            "world_model_hash": self.world_model_hash,
            "counterfactual_changes_hash": self.counterfactual_changes_hash,
            "predicted_outcome_hash": self.predicted_outcome_hash,
            "simulated_outcome_hash": self.simulated_outcome_hash,
            "execution_receipt_hash": self.execution_receipt_hash,
            "physics_replay_verified": self.physics_replay_verified,
            "training_eligible": self.training_eligible,
            "promotion_truth_allowed": self.promotion_truth_allowed,
            "hardware_authorized": False,
        }
