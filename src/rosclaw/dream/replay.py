"""Replay Dream orchestration with blind evaluation and fail-closed consolidation.

The Replay Dream Plane is deliberately asynchronous and simulation-only.  It
freezes the exact four-partition training batch, binds a learner product to its
parent and campaign, derives the learning/retention/safety gates from the
existing continual-learning gate, and exposes only commitments for private
holdout evaluation.

No method in this module activates a policy.  A passing candidate may be
published, verified, and staged for SIM review.  A rejected candidate is
discarded or, if some external actor already activated that exact successor,
rolled back to its pinned parent.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.continual.contracts import (
    ExperiencePartition,
    ExperienceUse,
    PolicyVersion,
)
from rosclaw.continual.experience import ExperienceBatch
from rosclaw.continual.services.experience import ExperienceService
from rosclaw.continual.services.learner import (
    LearnerProduct,
    LearnerService,
    LearnerServiceReceipt,
)
from rosclaw.continual.services.weight_update import WeightUpdateService
from rosclaw.continual.stability import (
    CheckStatus,
    ContinualCandidateEvidence,
    ContinualDecision,
    ContinualGateReport,
    GateCheck,
    StabilityPlasticityGate,
)
from rosclaw.dream.contracts import DreamCampaign, DreamType
from rosclaw.dream.control import DreamCampaignState, DreamCampaignStatus, DreamScheduler
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.contracts import (
    ConsolidationDecision,
    ConsolidationManifest,
    GateName,
    GateResult,
    GateStatus,
    SkillGrowthSpec,
)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTINUAL_CHECK_GROUPS: Mapping[GateName, frozenset[str]] = MappingProxyType(
    {
        GateName.LEARNING: frozenset({"plasticity", "self_core"}),
        GateName.RETENTION: frozenset(
            {"historical_mean", "critical_skill", "anchor_drift", "replay_coverage"}
        ),
        GateName.SAFETY: frozenset({"identity", "safety", "version_execution"}),
    }
)


def _require_hash(label: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _hash_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _hashes(
    values: tuple[str, ...],
    *,
    label: str,
    allow_duplicates: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if not allow_duplicates and len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be unique")
    for value in normalized:
        _require_hash(label, value)
    return normalized


def _status_for_checks(checks: tuple[GateCheck, ...]) -> GateStatus:
    if any(check.status is CheckStatus.FAIL for check in checks):
        return GateStatus.FAIL
    if any(check.status is CheckStatus.MISSING for check in checks):
        return GateStatus.MISSING
    return GateStatus.PASS


def _consolidation_decision(gates: tuple[GateResult, ...]) -> ConsolidationDecision:
    if any(gate.status is GateStatus.FAIL for gate in gates):
        return ConsolidationDecision.REJECT
    if any(gate.status is GateStatus.MISSING for gate in gates):
        return ConsolidationDecision.NEED_MORE_EVIDENCE
    return ConsolidationDecision.CONSOLIDATE_SIM


@dataclass(frozen=True)
class ReplaySnapshot:
    """Content commitment for the exact replay batch shown to the learner.

    Repeated record hashes are allowed because the bounded replay store samples
    with replacement.  The ordered record/trajectory/use vectors preserve that
    exact learner input rather than pretending it was a set.
    """

    campaign_hash: str
    parent_policy_hash: str
    body_hash: str
    batch_hash: str
    source_journal_hash: str
    private_holdout_commitment: str
    record_hashes: tuple[str, ...]
    trajectory_hashes: tuple[str, ...]
    partitions: tuple[ExperiencePartition, ...]
    permitted_uses: tuple[ExperienceUse, ...]
    requested_counts: Mapping[ExperiencePartition, int]
    learner_version: int
    schema_version: str = "rosclaw.dream.replay_snapshot.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("campaign_hash", self.campaign_hash),
            ("parent_policy_hash", self.parent_policy_hash),
            ("body_hash", self.body_hash),
            ("batch_hash", self.batch_hash),
            ("source_journal_hash", self.source_journal_hash),
            ("private_holdout_commitment", self.private_holdout_commitment),
        ):
            _require_hash(label, value)
        records = _hashes(
            self.record_hashes,
            label="record_hashes",
            allow_duplicates=True,
        )
        trajectories = _hashes(
            self.trajectory_hashes,
            label="trajectory_hashes",
            allow_duplicates=True,
        )
        partitions = tuple(self.partitions)
        uses = tuple(self.permitted_uses)
        if not (len(records) == len(trajectories) == len(partitions) == len(uses)):
            raise ValueError("replay snapshot vectors must be non-empty and aligned")
        if any(not isinstance(item, ExperiencePartition) for item in partitions):
            raise ValueError("partitions must contain recognized replay partitions")
        if any(not isinstance(item, ExperienceUse) for item in uses):
            raise ValueError("permitted_uses must contain recognized experience uses")
        if ExperienceUse.REJECT in uses:
            raise ValueError("a frozen replay snapshot cannot include rejected experience")
        counts = dict(self.requested_counts)
        if set(counts) != set(ExperiencePartition):
            raise ValueError("requested_counts must contain every replay partition")
        for partition, count in counts.items():
            if (
                not isinstance(partition, ExperiencePartition)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError("requested replay counts must be positive integers")
        observed = Counter(partitions)
        if any(observed[partition] != counts[partition] for partition in ExperiencePartition):
            raise ValueError("requested replay counts do not match the frozen records")
        if (
            isinstance(self.learner_version, bool)
            or not isinstance(self.learner_version, int)
            or self.learner_version < 0
        ):
            raise ValueError("learner_version must be a non-negative integer")
        expected_batch_hash = canonical_hash(
            {
                "schema_version": "rosclaw.continual.experience_batch.v1",
                "record_hashes": list(records),
                "permitted_uses": [item.value for item in uses],
                "requested_counts": {
                    partition.value: counts[partition] for partition in ExperiencePartition
                },
                "learner_version": self.learner_version,
            }
        )
        if self.batch_hash != expected_batch_hash:
            raise ValueError("replay snapshot batch commitment is inconsistent")
        object.__setattr__(self, "record_hashes", records)
        object.__setattr__(self, "trajectory_hashes", trajectories)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "permitted_uses", uses)
        object.__setattr__(self, "requested_counts", MappingProxyType(counts))

    @property
    def strict_replay_verified(self) -> bool:
        return True

    @property
    def private_holdout_rows_revealed(self) -> bool:
        return False

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign_hash,
            "parent_policy_hash": self.parent_policy_hash,
            "body_hash": self.body_hash,
            "batch_hash": self.batch_hash,
            "source_journal_hash": self.source_journal_hash,
            "private_holdout_commitment": self.private_holdout_commitment,
            "record_hashes": list(self.record_hashes),
            "trajectory_hashes": list(self.trajectory_hashes),
            "partitions": [item.value for item in self.partitions],
            "permitted_uses": [item.value for item in self.permitted_uses],
            "requested_counts": {
                partition.value: self.requested_counts[partition]
                for partition in ExperiencePartition
            },
            "learner_version": self.learner_version,
            "strict_replay_verified": self.strict_replay_verified,
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class ReplayWorkset:
    """In-memory pairing of a learner batch and its public commitment."""

    batch: ExperienceBatch
    snapshot: ReplaySnapshot

    def __post_init__(self) -> None:
        if self.batch.batch_hash != self.snapshot.batch_hash:
            raise ValueError("replay workset batch does not match its snapshot")
        if (
            tuple(record.record_hash for record in self.batch.records)
            != self.snapshot.record_hashes
            or tuple(record.trajectory.trajectory_hash for record in self.batch.records)
            != self.snapshot.trajectory_hashes
            or tuple(record.partition for record in self.batch.records) != self.snapshot.partitions
            or self.batch.permitted_uses != self.snapshot.permitted_uses
            or dict(self.batch.requested_counts) != dict(self.snapshot.requested_counts)
            or self.batch.learner_version != self.snapshot.learner_version
        ):
            raise ValueError("replay workset vectors do not match their snapshot")


class ReplaySnapshotBuilder:
    """Capture one exact, strict-replay four-partition training workset."""

    def capture(
        self,
        *,
        campaign: DreamCampaign,
        parent: PolicyVersion,
        experience: ExperienceService,
        batch_size: int,
        seed: int,
    ) -> ReplayWorkset:
        if DreamType.REPLAY not in campaign.dream_types:
            raise ValueError("campaign does not authorize Replay Dream")
        if parent.version_hash != campaign.parent_policy_hash:
            raise ValueError("replay parent does not match the campaign")
        if parent.body_hash != campaign.body_hash:
            raise ValueError("replay parent body does not match the campaign")
        source_journal_hash = experience.log.last_hash
        if source_journal_hash not in campaign.practice_snapshot_hashes:
            raise ValueError(
                "current experience journal is not one of the campaign's frozen practice snapshots"
            )
        batch = experience.sample(
            batch_size=batch_size,
            learner_version=parent.version,
            seed=seed,
        )
        if experience.log.last_hash != source_journal_hash:
            raise RuntimeError("experience journal changed while capturing replay snapshot")
        self._validate_batch(
            parent=parent,
            batch=batch,
            max_policy_lag=experience.store.config.max_policy_lag,
        )
        snapshot = ReplaySnapshot(
            campaign_hash=campaign.campaign_hash,
            parent_policy_hash=parent.version_hash,
            body_hash=parent.body_hash,
            batch_hash=batch.batch_hash,
            source_journal_hash=source_journal_hash,
            private_holdout_commitment=campaign.private_holdout_commitment,
            record_hashes=tuple(record.record_hash for record in batch.records),
            trajectory_hashes=tuple(record.trajectory.trajectory_hash for record in batch.records),
            partitions=tuple(record.partition for record in batch.records),
            permitted_uses=batch.permitted_uses,
            requested_counts=batch.requested_counts,
            learner_version=batch.learner_version,
        )
        return ReplayWorkset(batch=batch, snapshot=snapshot)

    @staticmethod
    def _validate_batch(
        *,
        parent: PolicyVersion,
        batch: ExperienceBatch,
        max_policy_lag: int,
    ) -> None:
        if batch.learner_version != parent.version:
            raise ValueError("replay batch learner version does not match the parent")
        for record, permitted_use in zip(
            batch.records,
            batch.permitted_uses,
            strict=True,
        ):
            trajectory = record.trajectory
            if not trajectory.strict_replay:
                raise ValueError("Replay Dream requires strict-replay trajectory truth")
            policy = trajectory.policy
            for field in (
                "body_hash",
                "safety_kernel_hash",
                "controller_snapshot_hash",
                "observation_names",
                "residual_action_names",
            ):
                if getattr(policy, field) != getattr(parent, field):
                    raise ValueError(f"replay trajectory changes immutable identity: {field}")
            expected_use = trajectory.permitted_use(
                learner_version=parent.version,
                max_policy_lag=max_policy_lag,
            )
            if permitted_use is not expected_use:
                raise ValueError("replay permitted use is inconsistent with policy staleness")


@dataclass(frozen=True)
class FrozenDreamCandidate:
    """Immutable learner output bound to one campaign and replay snapshot."""

    campaign_hash: str
    replay_snapshot_hash: str
    parent: PolicyVersion
    candidate: PolicyVersion
    batch_hash: str
    learner_job_id: str
    learner_event_hash: str
    checkpoint_hash: str
    learner_metrics_hash: str
    policy_change: float
    schema_version: str = "rosclaw.dream.frozen_candidate.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("campaign_hash", self.campaign_hash),
            ("replay_snapshot_hash", self.replay_snapshot_hash),
            ("batch_hash", self.batch_hash),
            ("learner_job_id", self.learner_job_id),
            ("learner_event_hash", self.learner_event_hash),
            ("checkpoint_hash", self.checkpoint_hash),
            ("learner_metrics_hash", self.learner_metrics_hash),
        ):
            _require_hash(label, value)
        if self.candidate.version != self.parent.version + 1:
            raise ValueError("frozen candidate must directly follow its parent version")
        if self.candidate.parent_version_hash != self.parent.version_hash:
            raise ValueError("frozen candidate parent lineage mismatch")
        for field in (
            "body_hash",
            "safety_kernel_hash",
            "controller_snapshot_hash",
            "observation_names",
            "residual_action_names",
        ):
            if getattr(self.candidate, field) != getattr(self.parent, field):
                raise ValueError(f"frozen candidate changes immutable identity: {field}")
        if (
            isinstance(self.policy_change, bool)
            or not math.isfinite(self.policy_change)
            or not 0.0 <= self.policy_change <= 1.0
        ):
            raise ValueError("policy_change must be finite and in [0, 1]")

    @property
    def freeze_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign_hash,
            "replay_snapshot_hash": self.replay_snapshot_hash,
            "parent": self.parent.to_dict(),
            "parent_policy_hash": self.parent.version_hash,
            "candidate": self.candidate.to_dict(),
            "candidate_policy_hash": self.candidate.version_hash,
            "batch_hash": self.batch_hash,
            "learner_job_id": self.learner_job_id,
            "learner_event_hash": self.learner_event_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "learner_metrics_hash": self.learner_metrics_hash,
            "policy_change": self.policy_change,
            "hardware_authorized": self.hardware_authorized,
        }


class CandidateFreezer:
    """Verify persisted learner bytes and freeze their complete lineage."""

    def freeze(
        self,
        *,
        campaign: DreamCampaign,
        snapshot: ReplaySnapshot,
        parent: PolicyVersion,
        candidate: PolicyVersion,
        artifact: bytes,
        learner_receipt: LearnerServiceReceipt,
    ) -> FrozenDreamCandidate:
        if snapshot.campaign_hash != campaign.campaign_hash:
            raise ValueError("replay snapshot does not belong to the campaign")
        if (
            parent.version_hash != campaign.parent_policy_hash
            or parent.body_hash != campaign.body_hash
        ):
            raise ValueError("candidate parent does not match the campaign")
        if snapshot.parent_policy_hash != parent.version_hash:
            raise ValueError("replay snapshot parent mismatch")
        if (
            snapshot.body_hash != campaign.body_hash
            or snapshot.private_holdout_commitment != campaign.private_holdout_commitment
        ):
            raise ValueError("replay snapshot identity does not match the campaign")
        if learner_receipt.batch_hash != snapshot.batch_hash:
            raise ValueError("learner receipt batch does not match the replay snapshot")
        if learner_receipt.parent_policy_hash != parent.version_hash:
            raise ValueError("learner receipt parent mismatch")
        if learner_receipt.candidate_policy_hash != candidate.version_hash:
            raise ValueError("learner receipt candidate mismatch")
        if learner_receipt.artifact_hash != candidate.artifact_hash:
            raise ValueError("learner receipt artifact mismatch")
        if _hash_bytes(artifact) != candidate.artifact_hash:
            raise ValueError("candidate artifact bytes do not match their content hash")
        raw_policy_change = learner_receipt.metrics.get("policy_change")
        if isinstance(raw_policy_change, bool) or not isinstance(raw_policy_change, (int, float)):
            raise ValueError("learner metrics require a numeric policy_change")
        policy_change = float(raw_policy_change)
        metrics_hash = canonical_hash(dict(sorted(learner_receipt.metrics.items())))
        return FrozenDreamCandidate(
            campaign_hash=campaign.campaign_hash,
            replay_snapshot_hash=snapshot.snapshot_hash,
            parent=parent,
            candidate=candidate,
            batch_hash=snapshot.batch_hash,
            learner_job_id=learner_receipt.job_id,
            learner_event_hash=learner_receipt.event_hash,
            checkpoint_hash=learner_receipt.checkpoint_hash,
            learner_metrics_hash=metrics_hash,
            policy_change=policy_change,
        )


@dataclass(frozen=True)
class BlindEvaluationEvidence:
    """Evaluator-owned result surface; private case rows never cross it."""

    continual: ContinualCandidateEvidence
    applicability_result: GateResult
    darwin_result: GateResult
    public_suite_hash: str
    darwin_commitment_hash: str
    evaluator_id: str
    evaluator_build_hash: str
    evaluation_protocol_hash: str
    evaluated_case_count: int
    schema_version: str = "rosclaw.dream.blind_evaluation_evidence.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.continual, ContinualCandidateEvidence):
            raise ValueError("continual must be ContinualCandidateEvidence")
        if self.applicability_result.name is not GateName.APPLICABILITY:
            raise ValueError("applicability_result must be the applicability gate")
        if self.darwin_result.name is not GateName.DARWIN:
            raise ValueError("darwin_result must be the Darwin gate")
        _require_hash("public_suite_hash", self.public_suite_hash)
        _require_hash("darwin_commitment_hash", self.darwin_commitment_hash)
        _require_hash("evaluator_build_hash", self.evaluator_build_hash)
        _require_hash("evaluation_protocol_hash", self.evaluation_protocol_hash)
        if not isinstance(self.evaluator_id, str) or not self.evaluator_id.strip():
            raise ValueError("evaluator_id must not be empty")
        if (
            isinstance(self.evaluated_case_count, bool)
            or not isinstance(self.evaluated_case_count, int)
            or self.evaluated_case_count < 0
        ):
            raise ValueError("evaluated_case_count must be a non-negative integer")
        if self.darwin_result.status is not GateStatus.MISSING:
            if self.evaluated_case_count <= 0:
                raise ValueError("completed Darwin evaluation requires evaluated cases")
            if self.darwin_result.report_hash != self.darwin_commitment_hash:
                raise ValueError("Darwin gate report must match its blind result commitment")

    @property
    def private_holdout_rows_revealed(self) -> bool:
        return False


@dataclass(frozen=True)
class BlindGateReport:
    """Five-gate report with only hashes and decisions from private evaluation."""

    campaign_hash: str
    replay_snapshot_hash: str
    candidate_freeze_hash: str
    parent_policy_hash: str
    candidate_policy_hash: str
    private_holdout_commitment: str
    public_suite_hash: str
    darwin_commitment_hash: str
    evaluator_id: str
    evaluator_build_hash: str
    evaluation_protocol_hash: str
    continual_gate_report_hash: str
    continual_decision: ContinualDecision
    gate_results: tuple[GateResult, ...]
    evaluated_case_count: int
    schema_version: str = "rosclaw.dream.blind_gate_report.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("campaign_hash", self.campaign_hash),
            ("replay_snapshot_hash", self.replay_snapshot_hash),
            ("candidate_freeze_hash", self.candidate_freeze_hash),
            ("parent_policy_hash", self.parent_policy_hash),
            ("candidate_policy_hash", self.candidate_policy_hash),
            ("private_holdout_commitment", self.private_holdout_commitment),
            ("public_suite_hash", self.public_suite_hash),
            ("darwin_commitment_hash", self.darwin_commitment_hash),
            ("evaluator_build_hash", self.evaluator_build_hash),
            ("evaluation_protocol_hash", self.evaluation_protocol_hash),
            ("continual_gate_report_hash", self.continual_gate_report_hash),
        ):
            _require_hash(label, value)
        if not isinstance(self.evaluator_id, str) or not self.evaluator_id.strip():
            raise ValueError("evaluator_id must not be empty")
        if not isinstance(self.continual_decision, ContinualDecision):
            raise ValueError("continual_decision must be recognized")
        gates = tuple(self.gate_results)
        if len(gates) != len(GateName) or {gate.name for gate in gates} != set(GateName):
            raise ValueError("blind report must contain each growth gate exactly once")
        if any(not isinstance(gate, GateResult) for gate in gates):
            raise ValueError("blind report gates must be GateResult records")
        darwin_gate = next(gate for gate in gates if gate.name is GateName.DARWIN)
        if (
            darwin_gate.status is not GateStatus.MISSING
            and darwin_gate.report_hash != self.darwin_commitment_hash
        ):
            raise ValueError("blind report Darwin commitment does not match its gate result")
        if (
            isinstance(self.evaluated_case_count, bool)
            or not isinstance(self.evaluated_case_count, int)
            or self.evaluated_case_count < 0
        ):
            raise ValueError("evaluated_case_count must be non-negative")
        object.__setattr__(self, "gate_results", gates)

    @property
    def decision(self) -> ConsolidationDecision:
        return _consolidation_decision(self.gate_results)

    @property
    def private_holdout_rows_revealed(self) -> bool:
        return False

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign_hash,
            "replay_snapshot_hash": self.replay_snapshot_hash,
            "candidate_freeze_hash": self.candidate_freeze_hash,
            "parent_policy_hash": self.parent_policy_hash,
            "candidate_policy_hash": self.candidate_policy_hash,
            "private_holdout_commitment": self.private_holdout_commitment,
            "public_suite_hash": self.public_suite_hash,
            "darwin_commitment_hash": self.darwin_commitment_hash,
            "evaluator_id": self.evaluator_id,
            "evaluator_build_hash": self.evaluator_build_hash,
            "evaluation_protocol_hash": self.evaluation_protocol_hash,
            "continual_gate_report_hash": self.continual_gate_report_hash,
            "continual_decision": self.continual_decision.value,
            "gate_results": [gate.to_dict() for gate in self.gate_results],
            "evaluated_case_count": self.evaluated_case_count,
            "decision": self.decision.value,
            "activation_authorized": False,
            "hardware_authorized": False,
        }


class BlindGateEvaluator:
    """Run stability/plasticity checks and derive three non-spoofable gates."""

    def __init__(self, gate: StabilityPlasticityGate | None = None) -> None:
        self.gate = gate or StabilityPlasticityGate()

    def evaluate(
        self,
        *,
        campaign: DreamCampaign,
        snapshot: ReplaySnapshot,
        frozen: FrozenDreamCandidate,
        evidence: BlindEvaluationEvidence,
    ) -> BlindGateReport:
        if not isinstance(evidence, BlindEvaluationEvidence):
            raise ValueError("evidence must be BlindEvaluationEvidence")
        self._validate_bindings(
            campaign=campaign,
            snapshot=snapshot,
            frozen=frozen,
            evidence=evidence,
        )
        continual = self.gate.evaluate(evidence.continual)
        checks_by_name = {check.name: check for check in continual.checks}
        expected_names = set().union(*_CONTINUAL_CHECK_GROUPS.values())
        if set(checks_by_name) != expected_names:
            raise ValueError("continual gate check surface changed without a blind-gate mapping")

        learning_checks = tuple(
            checks_by_name[name] for name in sorted(_CONTINUAL_CHECK_GROUPS[GateName.LEARNING])
        ) + (
            GateCheck(
                name="campaign_policy_change",
                status=(
                    CheckStatus.PASS
                    if frozen.policy_change <= campaign.budget.max_policy_change
                    else CheckStatus.FAIL
                ),
                detail=(
                    f"policy change={frozen.policy_change:.6f}; "
                    f"budget={campaign.budget.max_policy_change:.6f}"
                ),
            ),
        )
        retention_checks = tuple(
            checks_by_name[name] for name in sorted(_CONTINUAL_CHECK_GROUPS[GateName.RETENTION])
        ) + (
            GateCheck(
                name="campaign_anchor_drift",
                status=(
                    CheckStatus.PASS
                    if evidence.continual.anchor_action_drift_rms
                    <= campaign.budget.max_anchor_drift
                    else CheckStatus.FAIL
                ),
                detail=(
                    f"anchor drift={evidence.continual.anchor_action_drift_rms:.6f}; "
                    f"budget={campaign.budget.max_anchor_drift:.6f}"
                ),
            ),
        )
        safety_checks = tuple(
            checks_by_name[name] for name in sorted(_CONTINUAL_CHECK_GROUPS[GateName.SAFETY])
        )
        derived = (
            self._derived_gate(
                GateName.LEARNING,
                learning_checks,
                continual=continual,
                frozen=frozen,
            ),
            self._derived_gate(
                GateName.RETENTION,
                retention_checks,
                continual=continual,
                frozen=frozen,
            ),
            self._derived_gate(
                GateName.SAFETY,
                safety_checks,
                continual=continual,
                frozen=frozen,
            ),
            evidence.applicability_result,
            evidence.darwin_result,
        )
        expected_continual = _consolidation_decision(derived[:3])
        mapped = {
            ContinualDecision.PROMOTE_SIM: ConsolidationDecision.CONSOLIDATE_SIM,
            ContinualDecision.NEED_MORE_EVIDENCE: ConsolidationDecision.NEED_MORE_EVIDENCE,
            ContinualDecision.REJECT: ConsolidationDecision.REJECT,
        }[continual.decision]
        severity = {
            ConsolidationDecision.CONSOLIDATE_SIM: 0,
            ConsolidationDecision.NEED_MORE_EVIDENCE: 1,
            ConsolidationDecision.REJECT: 2,
        }
        if severity[expected_continual] < severity[mapped]:
            raise ValueError("blind gate derivation diverges from the continual gate decision")
        return BlindGateReport(
            campaign_hash=campaign.campaign_hash,
            replay_snapshot_hash=snapshot.snapshot_hash,
            candidate_freeze_hash=frozen.freeze_hash,
            parent_policy_hash=frozen.parent.version_hash,
            candidate_policy_hash=frozen.candidate.version_hash,
            private_holdout_commitment=campaign.private_holdout_commitment,
            public_suite_hash=evidence.public_suite_hash,
            darwin_commitment_hash=evidence.darwin_commitment_hash,
            evaluator_id=evidence.evaluator_id,
            evaluator_build_hash=evidence.evaluator_build_hash,
            evaluation_protocol_hash=evidence.evaluation_protocol_hash,
            continual_gate_report_hash=continual.report_hash,
            continual_decision=continual.decision,
            gate_results=derived,
            evaluated_case_count=evidence.evaluated_case_count,
        )

    @staticmethod
    def _derived_gate(
        name: GateName,
        checks: tuple[GateCheck, ...],
        *,
        continual: ContinualGateReport,
        frozen: FrozenDreamCandidate,
    ) -> GateResult:
        status = _status_for_checks(checks)
        report_hash = None
        if status is not GateStatus.MISSING:
            report_hash = canonical_hash(
                {
                    "schema_version": "rosclaw.dream.derived_gate_report.v1",
                    "name": name.value,
                    "continual_gate_report_hash": continual.report_hash,
                    "candidate_freeze_hash": frozen.freeze_hash,
                    "checks": [
                        {
                            "name": check.name,
                            "status": check.status.value,
                            "detail": check.detail,
                        }
                        for check in checks
                    ],
                }
            )
        return GateResult(
            name=name,
            status=status,
            report_hash=report_hash,
            detail="derived from the pinned continual stability/plasticity report",
        )

    @staticmethod
    def _validate_bindings(
        *,
        campaign: DreamCampaign,
        snapshot: ReplaySnapshot,
        frozen: FrozenDreamCandidate,
        evidence: BlindEvaluationEvidence,
    ) -> None:
        if snapshot.campaign_hash != campaign.campaign_hash:
            raise ValueError("blind evaluation snapshot campaign mismatch")
        if snapshot.private_holdout_commitment != campaign.private_holdout_commitment:
            raise ValueError("blind evaluation holdout commitment mismatch")
        if evidence.public_suite_hash not in campaign.boundary_suite_hashes:
            raise ValueError("blind evaluation public suite is outside the campaign")
        if frozen.campaign_hash != campaign.campaign_hash:
            raise ValueError("blind evaluation candidate campaign mismatch")
        if frozen.replay_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("blind evaluation candidate snapshot mismatch")
        continual = evidence.continual
        if (
            continual.parent_policy_hash != frozen.parent.artifact_hash
            or continual.candidate_policy_hash != frozen.candidate.artifact_hash
            or continual.body_hash != frozen.candidate.body_hash
            or continual.parent_body_hash != frozen.parent.body_hash
            or continual.safety_kernel_hash != frozen.candidate.safety_kernel_hash
            or continual.parent_safety_kernel_hash != frozen.parent.safety_kernel_hash
        ):
            raise ValueError("continual evidence identity does not match the frozen candidate")
        observed = Counter(snapshot.partitions)
        evidence_counts = {
            ExperiencePartition.RECENT: continual.replay_recent_count,
            ExperiencePartition.ANCHOR: continual.replay_anchor_count,
            ExperiencePartition.BOUNDARY: continual.replay_boundary_count,
            ExperiencePartition.SELF: continual.replay_self_count,
        }
        if any(evidence_counts[item] != observed[item] for item in ExperiencePartition):
            raise ValueError("continual replay counts do not match the frozen snapshot")


class ReplayConsolidationStatus(StrEnum):
    SIM_CANDIDATE_STAGED = "sim_candidate_staged"
    REJECTED = "rejected"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class ReplayConsolidationReceipt:
    campaign_hash: str
    replay_snapshot_hash: str
    candidate_freeze_hash: str
    blind_gate_report_hash: str
    manifest_hash: str
    status: ReplayConsolidationStatus
    active_policy_hash_before: str
    active_policy_hash_after: str
    weight_update_receipt_hashes: tuple[str, ...]
    candidate_staged: bool
    rollback_performed: bool
    activation_requested: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.dream.replay_consolidation_receipt.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("campaign_hash", self.campaign_hash),
            ("replay_snapshot_hash", self.replay_snapshot_hash),
            ("candidate_freeze_hash", self.candidate_freeze_hash),
            ("blind_gate_report_hash", self.blind_gate_report_hash),
            ("manifest_hash", self.manifest_hash),
            ("active_policy_hash_before", self.active_policy_hash_before),
            ("active_policy_hash_after", self.active_policy_hash_after),
        ):
            _require_hash(label, value)
        if not isinstance(self.status, ReplayConsolidationStatus):
            raise ValueError("status must be a recognized ReplayConsolidationStatus")
        object.__setattr__(
            self,
            "weight_update_receipt_hashes",
            _hashes(
                self.weight_update_receipt_hashes,
                label="weight_update_receipt_hashes",
                allow_duplicates=False,
            )
            if self.weight_update_receipt_hashes
            else (),
        )
        if self.activation_requested or self.hardware_authorized:
            raise ValueError("Replay Dream consolidation cannot authorize activation or hardware")
        if self.candidate_staged != (self.status is ReplayConsolidationStatus.SIM_CANDIDATE_STAGED):
            raise ValueError("candidate_staged is inconsistent with consolidation status")
        if self.rollback_performed != (self.status is ReplayConsolidationStatus.ROLLED_BACK):
            raise ValueError("rollback_performed is inconsistent with consolidation status")

    @property
    def receipt_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign_hash,
            "replay_snapshot_hash": self.replay_snapshot_hash,
            "candidate_freeze_hash": self.candidate_freeze_hash,
            "blind_gate_report_hash": self.blind_gate_report_hash,
            "manifest_hash": self.manifest_hash,
            "status": self.status.value,
            "active_policy_hash_before": self.active_policy_hash_before,
            "active_policy_hash_after": self.active_policy_hash_after,
            "weight_update_receipt_hashes": list(self.weight_update_receipt_hashes),
            "candidate_staged": self.candidate_staged,
            "rollback_performed": self.rollback_performed,
            "activation_requested": self.activation_requested,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class ReplayConsolidationResult:
    manifest: ConsolidationManifest
    receipt: ReplayConsolidationReceipt


class ReplayConsolidationAdapter:
    """Build a SIM-only manifest and safely reconcile the inference slots."""

    def consolidate(
        self,
        *,
        spec: SkillGrowthSpec,
        campaign: DreamCampaign,
        snapshot: ReplaySnapshot,
        frozen: FrozenDreamCandidate,
        blind_report: BlindGateReport,
        learned_changes: Mapping[str, str],
        new_capability_ids: tuple[str, ...],
        retained_capability_ids: tuple[str, ...],
        forgotten_capability_ids: tuple[str, ...],
        artifact: bytes,
        weight_updates: WeightUpdateService,
    ) -> ReplayConsolidationResult:
        self._validate_bindings(
            spec=spec,
            campaign=campaign,
            snapshot=snapshot,
            frozen=frozen,
            blind_report=blind_report,
            artifact=artifact,
        )
        decision = blind_report.decision
        declared = set(spec.capability_ids)
        new = set(new_capability_ids)
        retained = set(retained_capability_ids)
        forgotten = set(forgotten_capability_ids)
        if (
            not (new | retained | forgotten).issubset(declared)
            or new & retained
            or new & forgotten
            or retained & forgotten
        ):
            raise ValueError("capability deltas must be disjoint and declared by the growth spec")
        if decision is ConsolidationDecision.CONSOLIDATE_SIM and new | retained != declared:
            raise ValueError("SIM consolidation must account for every declared capability")
        if frozen.candidate.artifact_hash not in learned_changes.values():
            raise ValueError("learned_changes must bind the frozen candidate artifact")
        manifest = ConsolidationManifest(
            skill_growth_spec_hash=spec.spec_hash,
            candidate_artifact_hash=frozen.candidate.artifact_hash,
            parent_artifact_hash=frozen.parent.artifact_hash,
            rollback_artifact_hash=frozen.parent.artifact_hash,
            learned_changes=learned_changes,
            new_capability_ids=new_capability_ids,
            retained_capability_ids=retained_capability_ids,
            forgotten_capability_ids=forgotten_capability_ids,
            gate_results=blind_report.gate_results,
            darwin_report_hash=blind_report.darwin_commitment_hash,
            decision=decision,
        )

        before = weight_updates.inference.active.version_hash
        operation_hashes: list[str] = []
        rollback_performed = False
        if decision is ConsolidationDecision.CONSOLIDATE_SIM:
            operation_hashes.extend(
                self._stage_candidate(
                    frozen=frozen,
                    artifact=artifact,
                    weight_updates=weight_updates,
                )
            )
            status = ReplayConsolidationStatus.SIM_CANDIDATE_STAGED
            candidate_staged = True
        else:
            rejected_status, rejected_hashes, rollback_performed = self._reject_candidate(
                frozen=frozen,
                decision=decision,
                weight_updates=weight_updates,
            )
            operation_hashes.extend(rejected_hashes)
            status = rejected_status
            candidate_staged = False

        after = weight_updates.inference.active.version_hash
        if after != frozen.parent.version_hash:
            raise RuntimeError(
                "Replay Dream consolidation failed to preserve the parent active slot"
            )
        return ReplayConsolidationResult(
            manifest=manifest,
            receipt=ReplayConsolidationReceipt(
                campaign_hash=campaign.campaign_hash,
                replay_snapshot_hash=snapshot.snapshot_hash,
                candidate_freeze_hash=frozen.freeze_hash,
                blind_gate_report_hash=blind_report.report_hash,
                manifest_hash=manifest.manifest_hash,
                status=status,
                active_policy_hash_before=before,
                active_policy_hash_after=after,
                weight_update_receipt_hashes=tuple(operation_hashes),
                candidate_staged=candidate_staged,
                rollback_performed=rollback_performed,
            ),
        )

    @staticmethod
    def _stage_candidate(
        *,
        frozen: FrozenDreamCandidate,
        artifact: bytes,
        weight_updates: WeightUpdateService,
    ) -> tuple[str, ...]:
        inference = weight_updates.inference
        if inference.active.version_hash != frozen.parent.version_hash:
            raise RuntimeError("candidate staging requires its exact parent to remain active")
        for existing in (inference.candidate, inference.published):
            if existing is not None and existing.version_hash != frozen.candidate.version_hash:
                raise RuntimeError("another inference candidate already occupies the slot")
        hashes: list[str] = []
        if inference.candidate is not None:
            return ()
        if inference.published is None:
            hashes.append(
                canonical_hash(
                    weight_updates.publish(frozen.candidate, artifact=artifact).to_dict()
                )
            )
        hashes.append(canonical_hash(weight_updates.verify().to_dict()))
        hashes.append(canonical_hash(weight_updates.stage().to_dict()))
        if (
            inference.active.version_hash != frozen.parent.version_hash
            or inference.candidate is None
            or inference.candidate.version_hash != frozen.candidate.version_hash
        ):
            raise RuntimeError("candidate staging changed the active slot or lost the candidate")
        return tuple(hashes)

    @staticmethod
    def _reject_candidate(
        *,
        frozen: FrozenDreamCandidate,
        decision: ConsolidationDecision,
        weight_updates: WeightUpdateService,
    ) -> tuple[ReplayConsolidationStatus, tuple[str, ...], bool]:
        inference = weight_updates.inference
        hashes: list[str] = []
        rolled_back = False
        if inference.active.version_hash == frozen.candidate.version_hash:
            if (
                inference.rollback is None
                or inference.rollback.version_hash != frozen.parent.version_hash
            ):
                raise RuntimeError("rejected active candidate has no exact parent rollback target")
            receipt = weight_updates.rollback(
                reason="Replay Dream blind gates rejected the active SIM candidate"
            )
            hashes.append(canonical_hash(receipt.to_dict()))
            rolled_back = True
        elif inference.active.version_hash != frozen.parent.version_hash:
            raise RuntimeError("rejection found an unrelated active inference policy")

        for existing in (inference.candidate, inference.published):
            if existing is not None and existing.version_hash != frozen.candidate.version_hash:
                raise RuntimeError("rejection found an unrelated inference candidate")
        if inference.candidate is not None or inference.published is not None:
            receipt = weight_updates.discard(
                reason="Replay Dream candidate failed or lacks blind-gate evidence"
            )
            hashes.append(canonical_hash(receipt.to_dict()))
        if decision is ConsolidationDecision.NEED_MORE_EVIDENCE:
            status = ReplayConsolidationStatus.NEED_MORE_EVIDENCE
        elif rolled_back:
            status = ReplayConsolidationStatus.ROLLED_BACK
        else:
            status = ReplayConsolidationStatus.REJECTED
        return status, tuple(hashes), rolled_back

    @staticmethod
    def _validate_bindings(
        *,
        spec: SkillGrowthSpec,
        campaign: DreamCampaign,
        snapshot: ReplaySnapshot,
        frozen: FrozenDreamCandidate,
        blind_report: BlindGateReport,
        artifact: bytes,
    ) -> None:
        if campaign.skill_growth_spec_hash != spec.spec_hash:
            raise ValueError("campaign growth specification mismatch")
        if snapshot.campaign_hash != campaign.campaign_hash:
            raise ValueError("consolidation snapshot campaign mismatch")
        if frozen.campaign_hash != campaign.campaign_hash:
            raise ValueError("consolidation candidate campaign mismatch")
        if frozen.replay_snapshot_hash != snapshot.snapshot_hash:
            raise ValueError("consolidation candidate replay snapshot mismatch")
        if (
            blind_report.campaign_hash != campaign.campaign_hash
            or blind_report.replay_snapshot_hash != snapshot.snapshot_hash
            or blind_report.candidate_freeze_hash != frozen.freeze_hash
            or blind_report.private_holdout_commitment != campaign.private_holdout_commitment
            or blind_report.parent_policy_hash != frozen.parent.version_hash
            or blind_report.candidate_policy_hash != frozen.candidate.version_hash
            or blind_report.public_suite_hash not in campaign.boundary_suite_hashes
            or blind_report.evaluation_protocol_hash != spec.promotion_profile_hash
        ):
            raise ValueError("blind gate report is not bound to this consolidation")
        if _hash_bytes(artifact) != frozen.candidate.artifact_hash:
            raise ValueError("consolidation artifact checksum mismatch")


@dataclass(frozen=True)
class ReplayDreamRunReceipt:
    campaign_hash: str
    replay_snapshot_hash: str
    candidate_freeze_hash: str
    blind_gate_report_hash: str
    consolidation_receipt_hash: str
    manifest_hash: str
    scheduler_status_hash: str
    decision: ConsolidationDecision
    activation_requested: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.dream.replay_run_receipt.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("campaign_hash", self.campaign_hash),
            ("replay_snapshot_hash", self.replay_snapshot_hash),
            ("candidate_freeze_hash", self.candidate_freeze_hash),
            ("blind_gate_report_hash", self.blind_gate_report_hash),
            ("consolidation_receipt_hash", self.consolidation_receipt_hash),
            ("manifest_hash", self.manifest_hash),
            ("scheduler_status_hash", self.scheduler_status_hash),
        ):
            _require_hash(label, value)
        if not isinstance(self.decision, ConsolidationDecision):
            raise ValueError("decision must be a recognized ConsolidationDecision")
        if self.activation_requested or self.hardware_authorized:
            raise ValueError("Replay Dream run cannot authorize activation or hardware")

    @property
    def receipt_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign_hash,
            "replay_snapshot_hash": self.replay_snapshot_hash,
            "candidate_freeze_hash": self.candidate_freeze_hash,
            "blind_gate_report_hash": self.blind_gate_report_hash,
            "consolidation_receipt_hash": self.consolidation_receipt_hash,
            "manifest_hash": self.manifest_hash,
            "scheduler_status_hash": self.scheduler_status_hash,
            "decision": self.decision.value,
            "activation_requested": self.activation_requested,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class ReplayDreamRunResult:
    snapshot: ReplaySnapshot
    frozen_candidate: FrozenDreamCandidate
    blind_gate_report: BlindGateReport
    consolidation: ReplayConsolidationResult
    campaign_status: DreamCampaignStatus
    receipt: ReplayDreamRunReceipt


class ReplayDreamService:
    """Execute one recoverable learner-to-consolidation Replay Dream."""

    def __init__(
        self,
        *,
        experience: ExperienceService,
        learner: LearnerService,
        scheduler: DreamScheduler,
        weight_updates: WeightUpdateService,
        stability_gate: StabilityPlasticityGate | None = None,
    ) -> None:
        self.experience = experience
        self.learner = learner
        self.scheduler = scheduler
        self.weight_updates = weight_updates
        self.snapshot_builder = ReplaySnapshotBuilder()
        self.freezer = CandidateFreezer()
        self.blind_evaluator = BlindGateEvaluator(stability_gate)
        self.consolidator = ReplayConsolidationAdapter()

    def run(
        self,
        *,
        spec: SkillGrowthSpec,
        campaign: DreamCampaign,
        parent: PolicyVersion,
        lease_token: str,
        batch_size: int,
        seed: int,
        learner_executor: Callable[[ExperienceBatch], LearnerProduct],
        evaluation_provider: Callable[
            [FrozenDreamCandidate, ReplaySnapshot],
            BlindEvaluationEvidence,
        ],
        learned_changes: Mapping[str, str],
        new_capability_ids: tuple[str, ...],
        retained_capability_ids: tuple[str, ...],
        forgotten_capability_ids: tuple[str, ...] = (),
        gpu_seconds: float = 0.0,
        cpu_rollouts: int = 0,
    ) -> ReplayDreamRunResult:
        status = self.scheduler.status(campaign.campaign_hash)
        if status.state is not DreamCampaignState.RUNNING:
            raise RuntimeError("Replay Dream campaign requires an active worker lease")
        if self.learner.parent.version_hash != parent.version_hash:
            raise ValueError("learner service parent does not match the campaign parent")
        if (
            campaign.skill_growth_spec_hash != spec.spec_hash
            or campaign.parent_policy_hash != parent.version_hash
            or campaign.body_hash != parent.body_hash
        ):
            raise ValueError("Replay Dream service inputs do not match the campaign")
        frozen: FrozenDreamCandidate | None = None
        try:
            workset = self.snapshot_builder.capture(
                campaign=campaign,
                parent=parent,
                experience=self.experience,
                batch_size=batch_size,
                seed=seed,
            )
            learner_receipt = self.learner.execute(
                workset.batch,
                executor=learner_executor,
            )
            candidate = self.learner.candidate_for_batch(workset.batch.batch_hash)
            artifact = self.learner.artifact_bytes(learner_receipt.artifact_hash)
            frozen = self.freezer.freeze(
                campaign=campaign,
                snapshot=workset.snapshot,
                parent=parent,
                candidate=candidate,
                artifact=artifact,
                learner_receipt=learner_receipt,
            )
            evaluation = evaluation_provider(frozen, workset.snapshot)
            blind_report = self.blind_evaluator.evaluate(
                campaign=campaign,
                snapshot=workset.snapshot,
                frozen=frozen,
                evidence=evaluation,
            )
            self._account_usage(
                campaign=campaign,
                lease_token=lease_token,
                gpu_seconds=gpu_seconds,
                cpu_rollouts=cpu_rollouts,
            )
            consolidation = self.consolidator.consolidate(
                spec=spec,
                campaign=campaign,
                snapshot=workset.snapshot,
                frozen=frozen,
                blind_report=blind_report,
                learned_changes=learned_changes,
                new_capability_ids=new_capability_ids,
                retained_capability_ids=retained_capability_ids,
                forgotten_capability_ids=forgotten_capability_ids,
                artifact=artifact,
                weight_updates=self.weight_updates,
            )
            terminal = self.scheduler.complete(
                campaign.campaign_hash,
                lease_token=lease_token,
                result_manifest_hash=consolidation.manifest.manifest_hash,
                candidate_artifact_hashes=(frozen.candidate.artifact_hash,),
            )
        except BaseException:
            if frozen is not None:
                with suppress(Exception):
                    self._cleanup_uncommitted_candidate(frozen)
            current = self.scheduler.status(campaign.campaign_hash)
            if current.state is DreamCampaignState.RUNNING:
                self.scheduler.fail(
                    campaign.campaign_hash,
                    lease_token=lease_token,
                    reason="Replay Dream worker failed closed before durable consolidation",
                )
            raise
        scheduler_status_hash = canonical_hash(terminal.to_dict())
        run_receipt = ReplayDreamRunReceipt(
            campaign_hash=campaign.campaign_hash,
            replay_snapshot_hash=workset.snapshot.snapshot_hash,
            candidate_freeze_hash=frozen.freeze_hash,
            blind_gate_report_hash=blind_report.report_hash,
            consolidation_receipt_hash=consolidation.receipt.receipt_hash,
            manifest_hash=consolidation.manifest.manifest_hash,
            scheduler_status_hash=scheduler_status_hash,
            decision=consolidation.manifest.decision,
        )
        return ReplayDreamRunResult(
            snapshot=workset.snapshot,
            frozen_candidate=frozen,
            blind_gate_report=blind_report,
            consolidation=consolidation,
            campaign_status=terminal,
            receipt=run_receipt,
        )

    def _account_usage(
        self,
        *,
        campaign: DreamCampaign,
        lease_token: str,
        gpu_seconds: float,
        cpu_rollouts: int,
    ) -> None:
        status = self.scheduler.status(campaign.campaign_hash)
        if status.usage.candidates == 0:
            self.scheduler.record_usage(
                campaign.campaign_hash,
                lease_token=lease_token,
                gpu_seconds=gpu_seconds,
                cpu_rollouts=cpu_rollouts,
                candidates=1,
            )
            return
        if (
            status.usage.candidates != 1
            or not math.isclose(status.usage.gpu_seconds, gpu_seconds, abs_tol=1e-12)
            or status.usage.cpu_rollouts != cpu_rollouts
        ):
            raise RuntimeError("recovered Replay Dream usage does not match this worker request")

    def _cleanup_uncommitted_candidate(self, frozen: FrozenDreamCandidate) -> None:
        """Best-effort cleanup; never activates and never touches unrelated slots."""
        inference = self.weight_updates.inference
        if inference.active.version_hash == frozen.candidate.version_hash:
            if (
                inference.rollback is not None
                and inference.rollback.version_hash == frozen.parent.version_hash
            ):
                self.weight_updates.rollback(
                    reason="Replay Dream failed before scheduler accepted its manifest"
                )
            return
        if inference.active.version_hash != frozen.parent.version_hash:
            return
        pending = inference.candidate or inference.published
        if pending is not None and pending.version_hash == frozen.candidate.version_hash:
            self.weight_updates.discard(
                reason="Replay Dream failed before scheduler accepted its manifest"
            )


__all__ = [
    "BlindEvaluationEvidence",
    "BlindGateEvaluator",
    "BlindGateReport",
    "CandidateFreezer",
    "FrozenDreamCandidate",
    "ReplayConsolidationAdapter",
    "ReplayConsolidationReceipt",
    "ReplayConsolidationResult",
    "ReplayConsolidationStatus",
    "ReplayDreamRunReceipt",
    "ReplayDreamRunResult",
    "ReplayDreamService",
    "ReplaySnapshot",
    "ReplaySnapshotBuilder",
    "ReplayWorkset",
]
