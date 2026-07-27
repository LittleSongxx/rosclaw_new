from __future__ import annotations

from dataclasses import replace

from rosclaw.continual.stability import (
    ContinualCandidateEvidence,
    ContinualDecision,
    PlasticityEvidence,
    SelfCoreEvidence,
    StabilityPlasticityGate,
    TaskRetention,
)
from tests.continual.helpers import digest


def _passing_evidence() -> ContinualCandidateEvidence:
    return ContinualCandidateEvidence(
        parent_policy_hash=digest("parent"),
        candidate_policy_hash=digest("candidate"),
        body_hash=digest("body"),
        parent_body_hash=digest("body"),
        safety_kernel_hash=digest("safety"),
        parent_safety_kernel_hash=digest("safety"),
        task_retention=(
            TaskRetention("stand", 0.95, 0.94, critical=True),
            TaskRetention("kick", 0.80, 0.82),
        ),
        plasticity=PlasticityEvidence(
            fine_tune_steps_to_threshold=1000,
            candidate_steps_to_threshold=650,
            fresh_network_gap_start=0.20,
            fresh_network_gap_end=0.18,
            dead_unit_ratio_start=0.05,
            dead_unit_ratio_end=0.05,
            effective_rank_start=40.0,
            effective_rank_end=39.0,
            output_churn=0.02,
        ),
        self_core=SelfCoreEvidence(
            shared_reference_hash=digest("reference-bank"),
            continual_seed_count=10,
            single_task_seed_count=8,
            threshold_sweep_count=12,
            persistence_gap=0.16,
            bootstrap_support=0.99,
            freeze_matched_pairs=120,
            freeze_advantage=0.02,
            lesion_matched_pairs=120,
            lesion_disadvantage=0.03,
            body_prediction_improved=True,
            body_change_update_passed=True,
        ),
        replay_recent_count=100,
        replay_anchor_count=50,
        replay_boundary_count=30,
        replay_self_count=20,
        anchor_action_drift_rms=0.01,
        critical_safety_regressions=0,
        stale_action_executions=0,
        old_version_replays=0,
    )


def test_gate_promotes_only_complete_stability_and_plasticity_evidence() -> None:
    report = StabilityPlasticityGate().evaluate(_passing_evidence())

    assert report.decision is ContinualDecision.PROMOTE_SIM
    assert report.activation_allowed
    assert all(check.status.value == "PASS" for check in report.checks)


def test_gate_distinguishes_missing_self_evidence_from_safety_failure() -> None:
    gate = StabilityPlasticityGate()
    missing = gate.evaluate(replace(_passing_evidence(), self_core=None))
    unsafe = gate.evaluate(replace(_passing_evidence(), critical_safety_regressions=1))

    assert missing.decision is ContinualDecision.NEED_MORE_EVIDENCE
    assert not missing.activation_allowed
    assert unsafe.decision is ContinualDecision.REJECT
    assert not unsafe.activation_allowed


def test_gate_rejects_catastrophic_forgetting_even_when_new_task_improves() -> None:
    evidence = replace(
        _passing_evidence(),
        task_retention=(
            TaskRetention("stand", 0.95, 0.80, critical=True),
            TaskRetention("moving_ball", 0.20, 0.90),
        ),
    )

    report = StabilityPlasticityGate().evaluate(evidence)

    assert report.decision is ContinualDecision.REJECT
    assert any(
        check.name == "critical_skill" and check.status.value == "FAIL" for check in report.checks
    )
