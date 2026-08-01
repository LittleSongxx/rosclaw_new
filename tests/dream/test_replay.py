from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw.continual.contracts import ExperiencePartition, PolicyVersion, SkillPhase
from rosclaw.continual.experience import ExperienceRecord
from rosclaw.continual.services.experience import ExperienceService
from rosclaw.continual.services.inference import InferenceService
from rosclaw.continual.services.learner import LearnerProduct, LearnerService
from rosclaw.continual.services.weight_update import WeightUpdateService
from rosclaw.continual.stability import (
    ContinualCandidateEvidence,
    ContinualDecision,
    StabilityPlasticityGate,
)
from rosclaw.dream import (
    BlindEvaluationEvidence,
    BlindGateEvaluator,
    CandidateFreezer,
    DreamBudget,
    DreamBudgetExceededError,
    DreamCampaign,
    DreamCampaignState,
    DreamScheduler,
    DreamType,
    ReplayConsolidationAdapter,
    ReplayConsolidationStatus,
    ReplayDreamService,
    ReplaySnapshotBuilder,
)
from rosclaw.growth import (
    ConsolidationDecision,
    GateName,
    GateResult,
    GateStatus,
    GrowthMetricSpec,
    MetricDirection,
    SkillGrowthSpec,
)
from tests.continual.helpers import digest, policy, trajectory
from tests.continual.test_stability import _passing_evidence


def _source_checkout() -> Path:
    return Path(__file__).parents[2]


def _spec(parent: PolicyVersion) -> SkillGrowthSpec:
    return SkillGrowthSpec(
        skill_id="g1.kick",
        adapter_id="g1.motion_adapter",
        body_hashes=(parent.body_hash,),
        capability_ids=("kick", "recover", "moving_ball_contact"),
        observation_contract_hash=digest("observation"),
        action_contract_hash=digest("action"),
        reward_contract_hash=digest("reward"),
        cost_contract_hash=digest("cost"),
        practice_source_ids=("practice.goalforge",),
        collective_source_ids=(),
        allowed_dream_types=("replay",),
        allowed_learner_ids=("residual.sac",),
        historical_anchor_hashes=(digest("anchor-suite"),),
        boundary_suite_hash=digest("boundary-suite"),
        metrics=(
            GrowthMetricSpec(
                metric_id="kick.recovery_stability",
                direction=MetricDirection.MAXIMIZE,
                primary=True,
            ),
        ),
        promotion_profile_hash=digest("promotion"),
        rollback_policy_hash=digest("rollback-policy"),
    )


def _campaign(
    *,
    spec: SkillGrowthSpec,
    parent: PolicyVersion,
    practice_snapshot_hash: str,
    max_policy_change: float = 0.05,
    max_anchor_drift: float = 0.02,
) -> DreamCampaign:
    return DreamCampaign(
        skill_growth_spec_hash=spec.spec_hash,
        body_hash=parent.body_hash,
        parent_policy_hash=parent.version_hash,
        trigger_kind="post_practice",
        trigger_evidence_hashes=(digest("trigger"),),
        objectives=("learn_moving_ball", "retain_stand_and_kick"),
        constraint_hashes=(digest("constraints"),),
        practice_snapshot_hashes=(practice_snapshot_hash,),
        collective_capsule_hashes=(),
        historical_anchor_hashes=(digest("anchor-suite"),),
        boundary_suite_hashes=(digest("boundary-suite"),),
        private_holdout_commitment=digest("sealed-private-holdout"),
        dream_types=(DreamType.REPLAY,),
        learner_ids=("residual.sac",),
        budget=DreamBudget(
            max_gpu_seconds=100.0,
            max_cpu_rollouts=100,
            max_candidates=2,
            max_wall_seconds=300.0,
            max_policy_change=max_policy_change,
            max_anchor_drift=max_anchor_drift,
        ),
    )


def _append_four_partitions(
    service: ExperienceService,
    parent: PolicyVersion,
) -> None:
    for record in (
        ExperienceRecord(
            trajectory(parent, episode="recent"),
            ExperiencePartition.RECENT,
        ),
        ExperienceRecord(
            trajectory(parent, episode="anchor"),
            ExperiencePartition.ANCHOR,
            anchor_policy_hash=parent.artifact_hash,
        ),
        ExperienceRecord(
            trajectory(parent, episode="boundary", critical=True),
            ExperiencePartition.BOUNDARY,
            boundary_reason="fall counterexample",
        ),
        ExperienceRecord(
            trajectory(parent, episode="self"),
            ExperiencePartition.SELF,
            self_change_hash=digest("body-change"),
        ),
    ):
        service.append(record)


def _candidate_evidence(
    parent: PolicyVersion,
    candidate: PolicyVersion,
    *,
    safety_regressions: int = 0,
    replay_counts: tuple[int, int, int, int] = (10, 5, 3, 2),
    anchor_drift: float = 0.01,
) -> ContinualCandidateEvidence:
    return replace(
        _passing_evidence(),
        parent_policy_hash=parent.artifact_hash,
        candidate_policy_hash=candidate.artifact_hash,
        body_hash=candidate.body_hash,
        parent_body_hash=parent.body_hash,
        safety_kernel_hash=candidate.safety_kernel_hash,
        parent_safety_kernel_hash=parent.safety_kernel_hash,
        replay_recent_count=replay_counts[0],
        replay_anchor_count=replay_counts[1],
        replay_boundary_count=replay_counts[2],
        replay_self_count=replay_counts[3],
        anchor_action_drift_rms=anchor_drift,
        critical_safety_regressions=safety_regressions,
    )


def _external_gate(
    name: GateName,
    status: GateStatus = GateStatus.PASS,
) -> GateResult:
    report_hash = None if status is GateStatus.MISSING else digest(f"{name.value}-{status.value}")
    return GateResult(name=name, status=status, report_hash=report_hash)


def _blind_evidence(
    parent: PolicyVersion,
    candidate: PolicyVersion,
    *,
    safety_regressions: int = 0,
    applicability: GateStatus = GateStatus.PASS,
    darwin: GateStatus = GateStatus.PASS,
    replay_counts: tuple[int, int, int, int] = (10, 5, 3, 2),
    anchor_drift: float = 0.01,
) -> BlindEvaluationEvidence:
    darwin_gate = _external_gate(GateName.DARWIN, darwin)
    darwin_commitment = (
        darwin_gate.report_hash
        if darwin_gate.report_hash is not None
        else digest("sealed-darwin-evaluation-not-complete")
    )
    return BlindEvaluationEvidence(
        continual=_candidate_evidence(
            parent,
            candidate,
            safety_regressions=safety_regressions,
            replay_counts=replay_counts,
            anchor_drift=anchor_drift,
        ),
        applicability_result=_external_gate(GateName.APPLICABILITY, applicability),
        darwin_result=darwin_gate,
        public_suite_hash=digest("boundary-suite"),
        darwin_commitment_hash=darwin_commitment,
        evaluator_id="darwin.private-evaluator.v1",
        evaluator_build_hash=digest("darwin-evaluator-build"),
        evaluation_protocol_hash=digest("promotion"),
        evaluated_case_count=0 if darwin is GateStatus.MISSING else 40,
    )


def _learner_executor(
    candidate: PolicyVersion,
    artifact: bytes,
    *,
    policy_change: float = 0.02,
):  # type: ignore[no-untyped-def]
    def execute(_batch):  # type: ignore[no-untyped-def]
        return LearnerProduct(
            candidate=candidate,
            artifact=artifact,
            checkpoint=b"complete-replay-dream-checkpoint",
            metrics={
                "policy_change": policy_change,
                "anchor_distillation_loss": 0.01,
                "output_churn": 0.02,
            },
        )

    return execute


def _freeze(
    root: Path,
    *,
    campaign: DreamCampaign,
    parent: PolicyVersion,
    candidate: PolicyVersion,
    candidate_artifact: bytes,
    experience: ExperienceService,
):  # type: ignore[no-untyped-def]
    workset = ReplaySnapshotBuilder().capture(
        campaign=campaign,
        parent=parent,
        experience=experience,
        batch_size=20,
        seed=7,
    )
    learner = LearnerService(root, source_checkout=_source_checkout(), parent=parent)
    receipt = learner.execute(
        workset.batch,
        executor=_learner_executor(candidate, candidate_artifact),
    )
    frozen = CandidateFreezer().freeze(
        campaign=campaign,
        snapshot=workset.snapshot,
        parent=parent,
        candidate=candidate,
        artifact=candidate_artifact,
        learner_receipt=receipt,
    )
    return workset, learner, frozen


def test_replay_snapshot_freezes_exact_four_partition_batch_without_holdout_rows(
    tmp_path: Path,
) -> None:
    parent, _ = policy(0)
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as experience:
        _append_four_partitions(experience, parent)
        spec = _spec(parent)
        campaign = _campaign(
            spec=spec,
            parent=parent,
            practice_snapshot_hash=experience.log.last_hash,
        )
        workset = ReplaySnapshotBuilder().capture(
            campaign=campaign,
            parent=parent,
            experience=experience,
            batch_size=20,
            seed=3,
        )

        assert workset.snapshot.requested_counts == {
            ExperiencePartition.RECENT: 10,
            ExperiencePartition.ANCHOR: 5,
            ExperiencePartition.BOUNDARY: 3,
            ExperiencePartition.SELF: 2,
        }
        assert workset.snapshot.batch_hash == workset.batch.batch_hash
        assert workset.snapshot.strict_replay_verified
        assert not workset.snapshot.private_holdout_rows_revealed
        assert "holdout_rows" not in str(workset.snapshot.to_dict())
        with pytest.raises(ValueError, match="batch commitment"):
            replace(workset.snapshot, batch_hash=digest("tampered-batch"))

        experience.append(
            ExperienceRecord(
                trajectory(parent, episode="new-recent"),
                ExperiencePartition.RECENT,
            )
        )
        with pytest.raises(ValueError, match="frozen practice snapshots"):
            ReplaySnapshotBuilder().capture(
                campaign=campaign,
                parent=parent,
                experience=experience,
                batch_size=20,
                seed=3,
            )


def test_candidate_freeze_recovers_exact_learner_product_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    parent, _ = policy(0)
    candidate, artifact = policy(1, parent=parent)
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as experience:
        _append_four_partitions(experience, parent)
        spec = _spec(parent)
        campaign = _campaign(
            spec=spec,
            parent=parent,
            practice_snapshot_hash=experience.log.last_hash,
        )
        workset, learner, frozen = _freeze(
            tmp_path,
            campaign=campaign,
            parent=parent,
            candidate=candidate,
            candidate_artifact=artifact,
            experience=experience,
        )

        recovered = LearnerService(
            tmp_path,
            source_checkout=_source_checkout(),
            parent=parent,
        )
        assert recovered.candidate_for_batch(workset.batch.batch_hash) == candidate
        assert recovered.artifact_bytes(candidate.artifact_hash) == artifact
        assert frozen.candidate.version_hash == candidate.version_hash
        assert frozen.policy_change == 0.02
        assert not frozen.hardware_authorized
        with pytest.raises(ValueError, match="artifact bytes"):
            CandidateFreezer().freeze(
                campaign=campaign,
                snapshot=workset.snapshot,
                parent=parent,
                candidate=candidate,
                artifact=b"tampered",
                learner_receipt=learner.execute(
                    workset.batch,
                    executor=pytest.fail,  # idempotent completed batch; executor is not called
                ),
            )


def test_blind_gate_derives_retention_and_rejects_snapshot_count_spoofing(
    tmp_path: Path,
) -> None:
    parent, _ = policy(0)
    candidate, artifact = policy(1, parent=parent)
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as experience:
        _append_four_partitions(experience, parent)
        spec = _spec(parent)
        campaign = _campaign(
            spec=spec,
            parent=parent,
            practice_snapshot_hash=experience.log.last_hash,
        )
        workset, _, frozen = _freeze(
            tmp_path,
            campaign=campaign,
            parent=parent,
            candidate=candidate,
            candidate_artifact=artifact,
            experience=experience,
        )
        evaluator = BlindGateEvaluator()
        report = evaluator.evaluate(
            campaign=campaign,
            snapshot=workset.snapshot,
            frozen=frozen,
            evidence=_blind_evidence(parent, candidate),
        )

        assert report.decision is ConsolidationDecision.CONSOLIDATE_SIM
        assert report.continual_decision is ContinualDecision.PROMOTE_SIM
        assert not report.private_holdout_rows_revealed
        assert "holdout_rows" not in str(report.to_dict())
        assert {gate.name for gate in report.gate_results} == set(GateName)

        with pytest.raises(ValueError, match="replay counts"):
            evaluator.evaluate(
                campaign=campaign,
                snapshot=workset.snapshot,
                frozen=frozen,
                evidence=_blind_evidence(
                    parent,
                    candidate,
                    replay_counts=(11, 4, 3, 2),
                ),
            )


def test_campaign_drift_budgets_can_reject_a_passing_continual_report(
    tmp_path: Path,
) -> None:
    parent, _ = policy(0)
    candidate, artifact = policy(1, parent=parent)
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as experience:
        _append_four_partitions(experience, parent)
        spec = _spec(parent)
        campaign = _campaign(
            spec=spec,
            parent=parent,
            practice_snapshot_hash=experience.log.last_hash,
            max_policy_change=0.01,
            max_anchor_drift=0.005,
        )
        workset, learner, _ = _freeze(
            tmp_path,
            campaign=campaign,
            parent=parent,
            candidate=candidate,
            candidate_artifact=artifact,
            experience=experience,
        )
        receipt = learner.execute(
            workset.batch,
            executor=pytest.fail,
        )
        frozen = CandidateFreezer().freeze(
            campaign=campaign,
            snapshot=workset.snapshot,
            parent=parent,
            candidate=candidate,
            artifact=artifact,
            learner_receipt=receipt,
        )
        report = BlindGateEvaluator().evaluate(
            campaign=campaign,
            snapshot=workset.snapshot,
            frozen=frozen,
            evidence=_blind_evidence(parent, candidate),
        )

        assert report.continual_decision is ContinualDecision.PROMOTE_SIM
        assert report.decision is ConsolidationDecision.REJECT
        assert (
            next(gate for gate in report.gate_results if gate.name is GateName.LEARNING).status
            is GateStatus.FAIL
        )
        assert (
            next(gate for gate in report.gate_results if gate.name is GateName.RETENTION).status
            is GateStatus.FAIL
        )


def test_replay_dream_end_to_end_stages_passing_sim_candidate_without_activation(
    tmp_path: Path,
) -> None:
    parent, parent_artifact = policy(0)
    candidate, candidate_artifact = policy(1, parent=parent)
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as experience:
        _append_four_partitions(experience, parent)
        spec = _spec(parent)
        campaign = _campaign(
            spec=spec,
            parent=parent,
            practice_snapshot_hash=experience.log.last_hash,
        )
        learner = LearnerService(
            tmp_path,
            source_checkout=_source_checkout(),
            parent=parent,
        )
        inference = InferenceService(
            tmp_path,
            source_checkout=_source_checkout(),
            active=parent,
            active_artifact=parent_artifact,
        )
        updater = WeightUpdateService(
            tmp_path,
            source_checkout=_source_checkout(),
            inference=inference,
        )
        with DreamScheduler(
            tmp_path,
            source_checkout=_source_checkout(),
            token_factory=lambda: "replay-dream-worker-token-0001",
        ) as scheduler:
            scheduler.submit(campaign)
            lease = scheduler.acquire(
                worker_id="offline-learner-0",
                lease_seconds=120.0,
                campaign_hash=campaign.campaign_hash,
            )
            result = ReplayDreamService(
                experience=experience,
                learner=learner,
                scheduler=scheduler,
                weight_updates=updater,
            ).run(
                spec=spec,
                campaign=campaign,
                parent=parent,
                lease_token=lease.lease_token,
                batch_size=20,
                seed=11,
                learner_executor=_learner_executor(candidate, candidate_artifact),
                evaluation_provider=lambda frozen, snapshot: _blind_evidence(
                    frozen.parent,
                    frozen.candidate,
                ),
                learned_changes={"residual_policy": candidate.artifact_hash},
                new_capability_ids=("moving_ball_contact",),
                retained_capability_ids=("kick", "recover"),
                gpu_seconds=2.0,
                cpu_rollouts=40,
            )

            assert result.campaign_status.state is DreamCampaignState.COMPLETED
            assert result.consolidation.manifest.decision is (ConsolidationDecision.CONSOLIDATE_SIM)
            assert result.consolidation.receipt.status is (
                ReplayConsolidationStatus.SIM_CANDIDATE_STAGED
            )
            assert inference.active == parent
            assert inference.candidate == candidate
            assert result.receipt.activation_requested is False
            assert result.receipt.hardware_authorized is False
            operations = [
                event.payload["operation"]
                for event in updater.log.events
                if event.kind == "REQUESTED"
            ]
            assert operations == ["publish", "verify", "stage"]
            assert "activate" not in operations

    recovered = InferenceService(tmp_path, source_checkout=_source_checkout())
    assert recovered.active == parent
    assert recovered.candidate == candidate


def test_rejected_blind_candidate_is_rolled_back_to_exact_parent(
    tmp_path: Path,
) -> None:
    parent, parent_artifact = policy(0)
    candidate, candidate_artifact = policy(1, parent=parent)
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as experience:
        _append_four_partitions(experience, parent)
        spec = _spec(parent)
        campaign = _campaign(
            spec=spec,
            parent=parent,
            practice_snapshot_hash=experience.log.last_hash,
        )
        workset, _, frozen = _freeze(
            tmp_path,
            campaign=campaign,
            parent=parent,
            candidate=candidate,
            candidate_artifact=candidate_artifact,
            experience=experience,
        )
        evidence = _blind_evidence(
            parent,
            candidate,
            applicability=GateStatus.FAIL,
        )
        blind_report = BlindGateEvaluator().evaluate(
            campaign=campaign,
            snapshot=workset.snapshot,
            frozen=frozen,
            evidence=evidence,
        )
        inference = InferenceService(
            tmp_path,
            source_checkout=_source_checkout(),
            active=parent,
            active_artifact=parent_artifact,
        )
        updater = WeightUpdateService(
            tmp_path,
            source_checkout=_source_checkout(),
            inference=inference,
        )
        updater.publish(candidate, artifact=candidate_artifact)
        updater.verify()
        updater.stage()
        passing_continual = StabilityPlasticityGate().evaluate(evidence.continual)
        assert passing_continual.decision is ContinualDecision.PROMOTE_SIM
        updater.activate(
            phase=SkillPhase.COMPLETE,
            gate_report=passing_continual,
        )
        assert inference.active == candidate

        result = ReplayConsolidationAdapter().consolidate(
            spec=spec,
            campaign=campaign,
            snapshot=workset.snapshot,
            frozen=frozen,
            blind_report=blind_report,
            learned_changes={"residual_policy": candidate.artifact_hash},
            new_capability_ids=(),
            retained_capability_ids=(),
            forgotten_capability_ids=("recover",),
            artifact=candidate_artifact,
            weight_updates=updater,
        )

        assert result.manifest.decision is ConsolidationDecision.REJECT
        assert result.receipt.status is ReplayConsolidationStatus.ROLLED_BACK
        assert result.receipt.rollback_performed
        assert inference.active == parent
        assert result.receipt.active_policy_hash_after == parent.version_hash


def test_missing_blind_evidence_discards_staged_candidate_and_preserves_parent(
    tmp_path: Path,
) -> None:
    parent, parent_artifact = policy(0)
    candidate, candidate_artifact = policy(1, parent=parent)
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as experience:
        _append_four_partitions(experience, parent)
        spec = _spec(parent)
        campaign = _campaign(
            spec=spec,
            parent=parent,
            practice_snapshot_hash=experience.log.last_hash,
        )
        workset, _, frozen = _freeze(
            tmp_path,
            campaign=campaign,
            parent=parent,
            candidate=candidate,
            candidate_artifact=candidate_artifact,
            experience=experience,
        )
        blind_report = BlindGateEvaluator().evaluate(
            campaign=campaign,
            snapshot=workset.snapshot,
            frozen=frozen,
            evidence=_blind_evidence(
                parent,
                candidate,
                darwin=GateStatus.MISSING,
            ),
        )
        inference = InferenceService(
            tmp_path,
            source_checkout=_source_checkout(),
            active=parent,
            active_artifact=parent_artifact,
        )
        updater = WeightUpdateService(
            tmp_path,
            source_checkout=_source_checkout(),
            inference=inference,
        )
        updater.publish(candidate, artifact=candidate_artifact)
        updater.verify()
        updater.stage()

        with pytest.raises(ValueError, match="not bound"):
            ReplayConsolidationAdapter().consolidate(
                spec=spec,
                campaign=campaign,
                snapshot=workset.snapshot,
                frozen=frozen,
                blind_report=replace(
                    blind_report,
                    evaluation_protocol_hash=digest("unapproved-protocol"),
                ),
                learned_changes={"residual_policy": candidate.artifact_hash},
                new_capability_ids=(),
                retained_capability_ids=(),
                forgotten_capability_ids=(),
                artifact=candidate_artifact,
                weight_updates=updater,
            )

        result = ReplayConsolidationAdapter().consolidate(
            spec=spec,
            campaign=campaign,
            snapshot=workset.snapshot,
            frozen=frozen,
            blind_report=blind_report,
            learned_changes={"residual_policy": candidate.artifact_hash},
            new_capability_ids=(),
            retained_capability_ids=(),
            forgotten_capability_ids=(),
            artifact=candidate_artifact,
            weight_updates=updater,
        )

        assert result.manifest.decision is ConsolidationDecision.NEED_MORE_EVIDENCE
        assert result.receipt.status is ReplayConsolidationStatus.NEED_MORE_EVIDENCE
        assert inference.active == parent
        assert inference.candidate is None
        assert inference.published is None
        assert any(
            event.payload["operation"] == "discard"
            for event in updater.log.events
            if event.kind == "REQUESTED"
        )


def test_weight_update_discard_recovers_without_resurrecting_candidate(
    tmp_path: Path,
) -> None:
    parent, parent_artifact = policy(0)
    candidate, candidate_artifact = policy(1, parent=parent)
    inference = InferenceService(
        tmp_path,
        source_checkout=_source_checkout(),
        active=parent,
        active_artifact=parent_artifact,
    )
    updater = WeightUpdateService(
        tmp_path,
        source_checkout=_source_checkout(),
        inference=inference,
    )
    updater.publish(candidate, artifact=candidate_artifact)
    updater.verify()
    updater.stage()
    updater.discard(reason="blind evaluator rejected candidate")

    recovered = InferenceService(tmp_path, source_checkout=_source_checkout())
    assert recovered.active == parent
    assert recovered.candidate is None
    assert recovered.published is None


def test_interrupted_discard_is_recovered_as_completed(
    tmp_path: Path,
) -> None:
    parent, parent_artifact = policy(0)
    candidate, candidate_artifact = policy(1, parent=parent)
    inference = InferenceService(
        tmp_path,
        source_checkout=_source_checkout(),
        active=parent,
        active_artifact=parent_artifact,
    )
    updater = WeightUpdateService(
        tmp_path,
        source_checkout=_source_checkout(),
        inference=inference,
    )
    updater.publish(candidate, artifact=candidate_artifact)
    updater.verify()
    updater.stage()
    updater.log.append(
        "REQUESTED",
        {
            "operation_id": digest("interrupted-discard"),
            "operation": "discard",
            "parameters": {"reason": "blind rejection"},
            "inference_event_hash_before": inference.log.last_hash,
        },
    )
    inference._discard_candidate("blind rejection")

    recovered = WeightUpdateService(
        tmp_path,
        source_checkout=_source_checkout(),
        inference=inference,
    )

    assert recovered.recovered_completion_count == 1
    assert recovered.recovered_abort_count == 0
    assert inference.active == parent
    assert inference.candidate is None


def test_budget_rejection_happens_before_candidate_staging(
    tmp_path: Path,
) -> None:
    parent, parent_artifact = policy(0)
    candidate, candidate_artifact = policy(1, parent=parent)
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as experience:
        _append_four_partitions(experience, parent)
        spec = _spec(parent)
        campaign = _campaign(
            spec=spec,
            parent=parent,
            practice_snapshot_hash=experience.log.last_hash,
        )
        campaign = replace(
            campaign,
            budget=replace(campaign.budget, max_cpu_rollouts=5),
        )
        learner = LearnerService(
            tmp_path,
            source_checkout=_source_checkout(),
            parent=parent,
        )
        inference = InferenceService(
            tmp_path,
            source_checkout=_source_checkout(),
            active=parent,
            active_artifact=parent_artifact,
        )
        updater = WeightUpdateService(
            tmp_path,
            source_checkout=_source_checkout(),
            inference=inference,
        )
        with DreamScheduler(
            tmp_path,
            source_checkout=_source_checkout(),
            token_factory=lambda: "budget-rejection-worker-token",
        ) as scheduler:
            scheduler.submit(campaign)
            lease = scheduler.acquire(
                worker_id="offline-learner-0",
                lease_seconds=120.0,
                campaign_hash=campaign.campaign_hash,
            )
            service = ReplayDreamService(
                experience=experience,
                learner=learner,
                scheduler=scheduler,
                weight_updates=updater,
            )

            with pytest.raises(DreamBudgetExceededError):
                service.run(
                    spec=spec,
                    campaign=campaign,
                    parent=parent,
                    lease_token=lease.lease_token,
                    batch_size=20,
                    seed=11,
                    learner_executor=_learner_executor(candidate, candidate_artifact),
                    evaluation_provider=lambda frozen, snapshot: _blind_evidence(
                        frozen.parent,
                        frozen.candidate,
                    ),
                    learned_changes={"residual_policy": candidate.artifact_hash},
                    new_capability_ids=("moving_ball_contact",),
                    retained_capability_ids=("kick", "recover"),
                    cpu_rollouts=40,
                )

            assert scheduler.status(campaign.campaign_hash).state is (
                DreamCampaignState.BUDGET_EXHAUSTED
            )
            assert inference.active == parent
            assert inference.candidate is None
            assert inference.published is None
            assert not any(
                event.payload["operation"] in {"publish", "verify", "stage"}
                for event in updater.log.events
                if event.kind == "REQUESTED"
            )
