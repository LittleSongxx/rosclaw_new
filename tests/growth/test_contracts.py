from __future__ import annotations

import hashlib

import pytest

from rosclaw.growth import (
    ConsolidationDecision,
    ConsolidationManifest,
    EvidenceLevel,
    EvidenceUsePolicy,
    GateName,
    GateResult,
    GateStatus,
    GrowthMetricSpec,
    MetricDirection,
    SkillGrowthSpec,
    TrainingEligibility,
)


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _metric(*, primary: bool = True) -> GrowthMetricSpec:
    return GrowthMetricSpec(
        metric_id="kick.recovery_stability",
        direction=MetricDirection.MAXIMIZE,
        primary=primary,
    )


def _skill_spec(metrics: tuple[GrowthMetricSpec, ...] | None = None) -> SkillGrowthSpec:
    return SkillGrowthSpec(
        skill_id="g1.kick",
        adapter_id="g1.motion_adapter",
        body_hashes=(_hash("body"),),
        capability_ids=("kick", "recover"),
        observation_contract_hash=_hash("observation"),
        action_contract_hash=_hash("action"),
        reward_contract_hash=_hash("reward"),
        cost_contract_hash=_hash("cost"),
        practice_source_ids=("practice.g1_hat_trick",),
        collective_source_ids=(),
        allowed_dream_types=("replay", "counterfactual"),
        allowed_learner_ids=("residual.ppo",),
        historical_anchor_hashes=(_hash("anchor"),),
        boundary_suite_hash=_hash("boundary"),
        metrics=metrics or (_metric(),),
        promotion_profile_hash=_hash("promotion"),
        rollback_policy_hash=_hash("rollback"),
    )


def _gates(statuses: dict[GateName, GateStatus] | None = None) -> tuple[GateResult, ...]:
    statuses = statuses or {}
    return tuple(
        GateResult(
            name=name,
            status=statuses.get(name, GateStatus.PASS),
            report_hash=(
                None if statuses.get(name) is GateStatus.MISSING else _hash(f"report-{name.value}")
            ),
        )
        for name in GateName
    )


def _manifest(
    *,
    gates: tuple[GateResult, ...] | None = None,
    decision: ConsolidationDecision = ConsolidationDecision.CONSOLIDATE_SIM,
    rollback: str | None = None,
    forgotten: tuple[str, ...] = (),
) -> ConsolidationManifest:
    return ConsolidationManifest(
        skill_growth_spec_hash=_hash("growth-spec"),
        candidate_artifact_hash=_hash("candidate"),
        parent_artifact_hash=_hash("parent"),
        rollback_artifact_hash=rollback or _hash("parent"),
        learned_changes={"residual_policy": _hash("residual-policy")},
        new_capability_ids=("recover_without_step",),
        retained_capability_ids=("kick", "recover"),
        forgotten_capability_ids=forgotten,
        gate_results=gates or _gates(),
        darwin_report_hash=_hash("report-darwin"),
        decision=decision,
    )


@pytest.mark.parametrize(
    ("level", "training", "promotion"),
    [
        (EvidenceLevel.EXECUTION_RECEIPT, TrainingEligibility.ALLOWED, True),
        (EvidenceLevel.PHYSICS_REPLAY, TrainingEligibility.ALLOWED, True),
        (EvidenceLevel.ACCELERATED_SIM, TrainingEligibility.ALLOWED, False),
        (EvidenceLevel.WORLD_MODEL, TrainingEligibility.ALLOWED, False),
        (EvidenceLevel.EXTERNAL_APPLICABLE, TrainingEligibility.CONDITIONAL, False),
        (EvidenceLevel.EXTERNAL_UNVERIFIED, TrainingEligibility.DENIED, False),
    ],
)
def test_evidence_permissions_are_canonical(
    level: EvidenceLevel,
    training: TrainingEligibility,
    promotion: bool,
) -> None:
    policy = EvidenceUsePolicy(level)

    assert policy.training is training
    assert policy.promotion_truth_allowed is promotion
    assert (policy.required_replay_level is None) is promotion


def test_evidence_level_cannot_be_spoofed_with_a_string() -> None:
    with pytest.raises(ValueError, match="recognized EvidenceLevel"):
        EvidenceUsePolicy("e5_external_unverified")  # type: ignore[arg-type]


def test_skill_growth_spec_is_content_addressed_and_task_neutral() -> None:
    first = _skill_spec()
    second = _skill_spec()

    assert first.spec_hash == second.spec_hash
    assert first.to_dict()["skill_id"] == "g1.kick"
    assert "football" not in first.to_dict()


def test_skill_growth_spec_requires_exactly_one_positive_primary_metric() -> None:
    with pytest.raises(ValueError, match="exactly one primary"):
        _skill_spec((_metric(primary=False),))

    zero = GrowthMetricSpec(
        metric_id="kick.success",
        direction=MetricDirection.MAXIMIZE,
        primary=True,
        minimum_relative_improvement=0.0,
    )
    with pytest.raises(ValueError, match="positive relative improvement"):
        _skill_spec((zero,))


def test_complete_passed_gates_only_consolidate_in_sim() -> None:
    manifest = _manifest()

    assert manifest.decision is ConsolidationDecision.CONSOLIDATE_SIM
    assert manifest.evidence_domain == "SIM_ONLY"
    assert manifest.hardware_authorized is False
    assert manifest.to_dict()["rollback_artifact_hash"] == _hash("parent")


def test_missing_or_failed_gate_forces_fail_closed_decision() -> None:
    missing = _gates({GateName.RETENTION: GateStatus.MISSING})
    manifest = _manifest(gates=missing, decision=ConsolidationDecision.NEED_MORE_EVIDENCE)
    assert manifest.decision is ConsolidationDecision.NEED_MORE_EVIDENCE

    failed = _gates({GateName.SAFETY: GateStatus.FAIL})
    rejected = _manifest(gates=failed, decision=ConsolidationDecision.REJECT)
    assert rejected.decision is ConsolidationDecision.REJECT

    with pytest.raises(ValueError, match="decision must be reject"):
        _manifest(gates=failed, decision=ConsolidationDecision.CONSOLIDATE_SIM)


def test_manifest_pins_rollback_and_rejects_forgetting() -> None:
    with pytest.raises(ValueError, match="rollback artifact"):
        _manifest(rollback=_hash("unrelated"))
    with pytest.raises(ValueError, match="cannot forget"):
        _manifest(forgotten=("stand",))
