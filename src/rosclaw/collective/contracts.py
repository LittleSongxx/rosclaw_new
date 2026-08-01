"""Fail-closed provenance contracts for collective experience.

External data is useful only after its license, body mapping, task semantics
and learning role are explicit.  These records intentionally separate a motion
reference from an RL transition dataset.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.contracts import EvidenceUsePolicy, TrainingEligibility

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _require_hash(label: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _optional_hash(label: str, value: str | None) -> None:
    if value is not None:
        _require_hash(label, value)


def _unit_score(label: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")


class LicenseUse(StrEnum):
    RESEARCH_NONCOMMERCIAL = "research_noncommercial"
    COMMERCIAL = "commercial"
    REDISTRIBUTION = "redistribution"


class LicenseDecision(StrEnum):
    PERMITTED = "permitted"
    PENDING = "pending"
    DENIED = "denied"


@dataclass(frozen=True)
class SourceLicenseEvidence:
    """License conclusion for one concrete intended use.

    ``PENDING`` deliberately permits absent terms so an incompletely documented
    source can be catalogued without becoming trainable.
    """

    requested_use: LicenseUse
    decision: LicenseDecision
    terms_uri: str | None = None
    terms_hash: str | None = None
    attribution: str = ""
    schema_version: str = "rosclaw.collective.source_license.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.requested_use, LicenseUse):
            raise ValueError("requested_use must be a recognized LicenseUse")
        if not isinstance(self.decision, LicenseDecision):
            raise ValueError("decision must be a recognized LicenseDecision")
        _optional_hash("terms_hash", self.terms_hash)
        if self.decision is not LicenseDecision.PENDING and (
            not self.terms_uri or self.terms_hash is None
        ):
            raise ValueError("a final license decision requires URI and hashed terms")
        if self.terms_uri is not None and not self.terms_uri.strip():
            raise ValueError("terms_uri must be non-empty when supplied")

    @property
    def evidence_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_use": self.requested_use.value,
            "decision": self.decision.value,
            "terms_uri": self.terms_uri,
            "terms_hash": self.terms_hash,
            "attribution": self.attribution,
        }


class ApplicabilityDecision(StrEnum):
    ACCEPT = "accept"
    PENDING = "pending"
    REJECT = "reject"


@dataclass(frozen=True)
class ApplicabilityAssessment:
    """Evidence-backed source-to-target morphology and regime assessment."""

    target_body_hash: str
    target_mapping_hash: str
    body_score: float
    task_score: float
    regime_score: float
    confidence: float
    evidence_hashes: tuple[str, ...]
    decision: ApplicabilityDecision
    schema_version: str = "rosclaw.collective.applicability.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ApplicabilityDecision):
            raise ValueError("decision must be a recognized ApplicabilityDecision")
        _require_hash("target_body_hash", self.target_body_hash)
        _require_hash("target_mapping_hash", self.target_mapping_hash)
        for label in ("body_score", "task_score", "regime_score", "confidence"):
            _unit_score(label, getattr(self, label))
        evidence_hashes = tuple(self.evidence_hashes)
        if not evidence_hashes or len(evidence_hashes) != len(set(evidence_hashes)):
            raise ValueError("evidence_hashes must be non-empty and unique")
        for value in evidence_hashes:
            _require_hash("evidence_hashes", value)
        object.__setattr__(self, "evidence_hashes", evidence_hashes)
        if self.decision is ApplicabilityDecision.ACCEPT:
            scores = (self.body_score, self.task_score, self.regime_score, self.confidence)
            if any(score < 0.8 for score in scores):
                raise ValueError("accepted applicability requires all scores and confidence >= 0.8")

    @property
    def assessment_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_body_hash": self.target_body_hash,
            "target_mapping_hash": self.target_mapping_hash,
            "body_score": self.body_score,
            "task_score": self.task_score,
            "regime_score": self.regime_score,
            "confidence": self.confidence,
            "evidence_hashes": list(self.evidence_hashes),
            "decision": self.decision.value,
        }


@dataclass(frozen=True)
class CollectiveSourceIdentity:
    provider: str
    dataset: str
    revision: str
    file_hashes: Mapping[str, str]
    license_evidence: SourceLicenseEvidence
    source_body_id: str
    source_body_hash: str | None = None
    schema_version: str = "rosclaw.collective.source_identity.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.license_evidence, SourceLicenseEvidence):
            raise ValueError("license_evidence must be a SourceLicenseEvidence record")
        for label in ("provider", "dataset", "revision", "source_body_id"):
            if not getattr(self, label).strip():
                raise ValueError(f"{label} must not be empty")
        _optional_hash("source_body_hash", self.source_body_hash)
        file_hashes = {str(path): str(value) for path, value in self.file_hashes.items()}
        if not file_hashes or any(not path.strip() for path in file_hashes):
            raise ValueError("file_hashes must contain at least one named file")
        for value in file_hashes.values():
            _require_hash("file_hashes value", value)
        object.__setattr__(self, "file_hashes", MappingProxyType(file_hashes))

    @property
    def source_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "dataset": self.dataset,
            "revision": self.revision,
            "file_hashes": dict(sorted(self.file_hashes.items())),
            "license_evidence": self.license_evidence.to_dict(),
            "source_body_id": self.source_body_id,
            "source_body_hash": self.source_body_hash,
        }


class CollectiveUse(StrEnum):
    MOTION_REFERENCE = "motion_reference"
    MOTION_TRACKING = "motion_tracking"
    BEHAVIOR_CLONING = "behavior_cloning"
    OFFLINE_RL = "offline_rl"
    WORLD_MODEL = "world_model"


@dataclass(frozen=True)
class ExperienceCapsule:
    """A governed external experience unit with explicit semantic limits."""

    source: CollectiveSourceIdentity
    applicability: ApplicabilityAssessment
    task_semantics_hash: str
    observation_semantics_hash: str
    modalities: tuple[str, ...]
    requested_uses: tuple[CollectiveUse, ...]
    quality_report_hash: str
    evidence_policy: EvidenceUsePolicy
    source_episode_count: int
    action_semantics_hash: str | None = None
    transition_semantics_hash: str | None = None
    reward_semantics_hash: str | None = None
    schema_version: str = "rosclaw.collective.experience_capsule.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.source, CollectiveSourceIdentity):
            raise ValueError("source must be a CollectiveSourceIdentity record")
        if not isinstance(self.applicability, ApplicabilityAssessment):
            raise ValueError("applicability must be an ApplicabilityAssessment record")
        if not isinstance(self.evidence_policy, EvidenceUsePolicy):
            raise ValueError("evidence_policy must be an EvidenceUsePolicy record")
        for label, value in (
            ("task_semantics_hash", self.task_semantics_hash),
            ("observation_semantics_hash", self.observation_semantics_hash),
            ("quality_report_hash", self.quality_report_hash),
        ):
            _require_hash(label, value)
        for label in (
            "action_semantics_hash",
            "transition_semantics_hash",
            "reward_semantics_hash",
        ):
            _optional_hash(label, getattr(self, label))
        modalities = tuple(self.modalities)
        if not modalities or len(modalities) != len(set(modalities)):
            raise ValueError("modalities must be non-empty and unique")
        if any(not modality.strip() for modality in modalities):
            raise ValueError("modalities must not contain empty values")
        object.__setattr__(self, "modalities", modalities)
        requested_uses = tuple(self.requested_uses)
        if not requested_uses or len(requested_uses) != len(set(requested_uses)):
            raise ValueError("requested_uses must be non-empty and unique")
        if any(not isinstance(value, CollectiveUse) for value in requested_uses):
            raise ValueError("requested_uses must contain recognized CollectiveUse values")
        object.__setattr__(self, "requested_uses", requested_uses)
        if self.source_episode_count <= 0:
            raise ValueError("source_episode_count must be positive")
        if CollectiveUse.BEHAVIOR_CLONING in requested_uses and self.action_semantics_hash is None:
            raise ValueError("behavior cloning requires action semantics")
        if CollectiveUse.WORLD_MODEL in requested_uses and self.transition_semantics_hash is None:
            raise ValueError("world-model training requires transition semantics")
        if CollectiveUse.OFFLINE_RL in requested_uses:
            required = (
                self.action_semantics_hash,
                self.transition_semantics_hash,
                self.reward_semantics_hash,
            )
            if any(value is None for value in required):
                raise ValueError("offline RL requires action, transition and reward semantics")

    @property
    def training_eligible(self) -> bool:
        return (
            self.source.license_evidence.decision is LicenseDecision.PERMITTED
            and self.applicability.decision is ApplicabilityDecision.ACCEPT
            and self.evidence_policy.training is not TrainingEligibility.DENIED
        )

    @property
    def promotion_truth_allowed(self) -> bool:
        """Collective data may teach, but cannot independently approve."""

        return False

    @property
    def capsule_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "applicability": self.applicability.to_dict(),
            "task_semantics_hash": self.task_semantics_hash,
            "observation_semantics_hash": self.observation_semantics_hash,
            "action_semantics_hash": self.action_semantics_hash,
            "transition_semantics_hash": self.transition_semantics_hash,
            "reward_semantics_hash": self.reward_semantics_hash,
            "modalities": list(self.modalities),
            "requested_uses": [value.value for value in self.requested_uses],
            "quality_report_hash": self.quality_report_hash,
            "evidence_policy": self.evidence_policy.to_dict(),
            "source_episode_count": self.source_episode_count,
            "training_eligible": self.training_eligible,
            "promotion_truth_allowed": self.promotion_truth_allowed,
        }
