from __future__ import annotations

import hashlib
import math

import pytest

from rosclaw.dream import DreamBudget, DreamCampaign, DreamEpisode, DreamType
from rosclaw.growth import EvidenceLevel, EvidenceUsePolicy


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _budget() -> DreamBudget:
    return DreamBudget(
        max_gpu_seconds=3600.0,
        max_cpu_rollouts=100,
        max_candidates=8,
        max_wall_seconds=7200.0,
        max_policy_change=0.05,
        max_anchor_drift=0.02,
    )


def _campaign(
    *,
    dream_types: tuple[DreamType, ...] = (DreamType.REPLAY, DreamType.COUNTERFACTUAL),
    collective: tuple[str, ...] = (),
) -> DreamCampaign:
    return DreamCampaign(
        skill_growth_spec_hash=_hash("growth-spec"),
        body_hash=_hash("body"),
        parent_policy_hash=_hash("parent-policy"),
        trigger_kind="post_practice_regression",
        trigger_evidence_hashes=(_hash("trigger"),),
        objectives=("improve_recovery", "retain_kick_success"),
        constraint_hashes=(_hash("constraints"),),
        practice_snapshot_hashes=(_hash("practice"),),
        collective_capsule_hashes=collective,
        historical_anchor_hashes=(_hash("anchor"),),
        boundary_suite_hashes=(_hash("boundary"),),
        private_holdout_commitment=_hash("private-holdout-commitment"),
        dream_types=dream_types,
        learner_ids=("residual.ppo",),
        budget=_budget(),
    )


def _episode(
    level: EvidenceLevel,
    **kwargs: object,
) -> DreamEpisode:
    values: dict[str, object] = {
        "campaign_hash": _hash("campaign"),
        "source_episode_hash": _hash("episode"),
        "body_hash": _hash("body"),
        "policy_hash": _hash("policy"),
        "dream_type": DreamType.REPLAY,
        "evidence_policy": EvidenceUsePolicy(level),
        "uncertainty": 0.1,
    }
    values.update(kwargs)
    return DreamEpisode(**values)  # type: ignore[arg-type]


def test_campaign_exposes_holdout_commitment_but_no_holdout_rows() -> None:
    campaign = _campaign()
    payload = campaign.to_dict()

    assert payload["private_holdout_commitment"] == _hash("private-holdout-commitment")
    assert "holdout_rows" not in str(payload)
    assert campaign.hardware_authorized is False


def test_social_dream_requires_governed_collective_capsules() -> None:
    with pytest.raises(ValueError, match="social dreaming requires"):
        _campaign(dream_types=(DreamType.SOCIAL,))

    campaign = _campaign(
        dream_types=(DreamType.SOCIAL,),
        collective=(_hash("experience-capsule"),),
    )
    assert campaign.collective_capsule_hashes == (_hash("experience-capsule"),)


def test_dream_type_cannot_be_spoofed_to_bypass_social_gate() -> None:
    with pytest.raises(ValueError, match="recognized DreamType"):
        _campaign(dream_types=("social",))  # type: ignore[arg-type]


def test_world_model_dream_can_train_but_never_approve() -> None:
    episode = _episode(
        EvidenceLevel.WORLD_MODEL,
        dream_type=DreamType.PROSPECTIVE,
        world_model_hash=_hash("world-model"),
        predicted_outcome_hash=_hash("prediction"),
    )

    assert episode.training_eligible is True
    assert episode.promotion_truth_allowed is False


def test_physics_replay_requires_verified_outcome_for_promotion_truth() -> None:
    unverified = _episode(
        EvidenceLevel.PHYSICS_REPLAY,
        simulated_outcome_hash=_hash("sim-result"),
    )
    verified = _episode(
        EvidenceLevel.PHYSICS_REPLAY,
        simulated_outcome_hash=_hash("sim-result"),
        physics_replay_verified=True,
    )

    assert unverified.promotion_truth_allowed is False
    assert verified.promotion_truth_allowed is True


def test_execution_truth_requires_receipt_and_external_unverified_cannot_train() -> None:
    with pytest.raises(ValueError, match="signed receipt"):
        _episode(EvidenceLevel.EXECUTION_RECEIPT)

    receipt = _episode(
        EvidenceLevel.EXECUTION_RECEIPT,
        execution_receipt_hash=_hash("execution-receipt"),
    )
    external = _episode(EvidenceLevel.EXTERNAL_UNVERIFIED)

    assert receipt.promotion_truth_allowed is True
    assert external.training_eligible is False
    assert external.promotion_truth_allowed is False


def test_budget_rejects_nonfinite_or_unbounded_drift() -> None:
    with pytest.raises(ValueError, match="finite"):
        DreamBudget(math.inf, 0, 1, 1.0, 0.1, 0.1)
    with pytest.raises(ValueError, match="max_policy_change"):
        DreamBudget(1.0, 0, 1, 1.0, 1.1, 0.1)
