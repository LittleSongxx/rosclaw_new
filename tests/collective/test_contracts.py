from __future__ import annotations

import hashlib

import pytest

from rosclaw.collective import (
    ApplicabilityAssessment,
    ApplicabilityDecision,
    CollectiveSourceIdentity,
    CollectiveUse,
    ExperienceCapsule,
    LicenseDecision,
    LicenseUse,
    SourceLicenseEvidence,
)
from rosclaw.growth import EvidenceLevel, EvidenceUsePolicy


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _license(decision: LicenseDecision) -> SourceLicenseEvidence:
    if decision is LicenseDecision.PENDING:
        return SourceLicenseEvidence(
            requested_use=LicenseUse.RESEARCH_NONCOMMERCIAL,
            decision=decision,
        )
    return SourceLicenseEvidence(
        requested_use=LicenseUse.RESEARCH_NONCOMMERCIAL,
        decision=decision,
        terms_uri="https://example.invalid/dataset-license",
        terms_hash=_hash("license"),
        attribution="Example Dataset Authors",
    )


def _source(decision: LicenseDecision = LicenseDecision.PERMITTED) -> CollectiveSourceIdentity:
    return CollectiveSourceIdentity(
        provider="CMRobot",
        dataset="MotionDecode",
        revision="pinned-revision",
        file_hashes={"motions/example.csv": _hash("csv")},
        license_evidence=_license(decision),
        source_body_id="source-humanoid",
    )


def _applicability(
    decision: ApplicabilityDecision = ApplicabilityDecision.ACCEPT,
) -> ApplicabilityAssessment:
    return ApplicabilityAssessment(
        target_body_hash=_hash("g1-body"),
        target_mapping_hash=_hash("mapping"),
        body_score=0.9,
        task_score=0.9,
        regime_score=0.9,
        confidence=0.9,
        evidence_hashes=(_hash("applicability-report"),),
        decision=decision,
    )


def _capsule(
    *,
    source: CollectiveSourceIdentity | None = None,
    requested_uses: tuple[CollectiveUse, ...] = (CollectiveUse.MOTION_REFERENCE,),
    action: str | None = None,
    transition: str | None = None,
    reward: str | None = None,
) -> ExperienceCapsule:
    return ExperienceCapsule(
        source=source or _source(),
        applicability=_applicability(),
        task_semantics_hash=_hash("task"),
        observation_semantics_hash=_hash("observation"),
        action_semantics_hash=action,
        transition_semantics_hash=transition,
        reward_semantics_hash=reward,
        modalities=("joint_position", "root_pose"),
        requested_uses=requested_uses,
        quality_report_hash=_hash("quality"),
        evidence_policy=EvidenceUsePolicy(EvidenceLevel.EXTERNAL_APPLICABLE),
        source_episode_count=42,
    )


def test_unknown_license_catalogues_but_never_trains() -> None:
    capsule = _capsule(source=_source(LicenseDecision.PENDING))

    assert capsule.source.license_evidence.terms_hash is None
    assert capsule.training_eligible is False


def test_motion_reference_can_train_after_license_and_applicability_gates() -> None:
    capsule = _capsule()

    assert capsule.training_eligible is True
    assert capsule.promotion_truth_allowed is False
    assert capsule.to_dict()["requested_uses"] == ["motion_reference"]


def test_motion_csv_cannot_be_declared_offline_rl_without_transition_semantics() -> None:
    with pytest.raises(ValueError, match="offline RL requires"):
        _capsule(requested_uses=(CollectiveUse.OFFLINE_RL,))

    capsule = _capsule(
        requested_uses=(CollectiveUse.OFFLINE_RL,),
        action=_hash("action"),
        transition=_hash("transition"),
        reward=_hash("reward"),
    )
    assert capsule.training_eligible is True


def test_learning_use_cannot_be_spoofed_to_bypass_semantic_requirements() -> None:
    with pytest.raises(ValueError, match="recognized CollectiveUse"):
        _capsule(requested_uses=("offline_rl",))  # type: ignore[arg-type]


def test_applicability_acceptance_requires_high_confidence_scores() -> None:
    with pytest.raises(ValueError, match=">= 0.8"):
        ApplicabilityAssessment(
            target_body_hash=_hash("g1-body"),
            target_mapping_hash=_hash("mapping"),
            body_score=0.79,
            task_score=0.9,
            regime_score=0.9,
            confidence=0.9,
            evidence_hashes=(_hash("report"),),
            decision=ApplicabilityDecision.ACCEPT,
        )


def test_source_file_manifest_is_immutable() -> None:
    source = _source()

    with pytest.raises(TypeError):
        source.file_hashes["other.csv"] = _hash("other")  # type: ignore[index]
