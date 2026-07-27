"""Machine gates for the stability--plasticity dilemma."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContinualDecision(StrEnum):
    PROMOTE_SIM = "PROMOTE_SIM"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    REJECT = "REJECT"


class CheckStatus(StrEnum):
    PASS = "PASS"
    MISSING = "MISSING"
    FAIL = "FAIL"


@dataclass(frozen=True)
class TaskRetention:
    task_id: str
    parent_score: float
    candidate_score: float
    critical: bool = False

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (self.parent_score, self.candidate_score)
        ):
            raise ValueError("task retention scores must be normalized to [0, 1]")

    @property
    def drop(self) -> float:
        return max(0.0, self.parent_score - self.candidate_score)


@dataclass(frozen=True)
class PlasticityEvidence:
    fine_tune_steps_to_threshold: int
    candidate_steps_to_threshold: int
    fresh_network_gap_start: float
    fresh_network_gap_end: float
    dead_unit_ratio_start: float
    dead_unit_ratio_end: float
    effective_rank_start: float
    effective_rank_end: float
    output_churn: float

    def __post_init__(self) -> None:
        if self.fine_tune_steps_to_threshold <= 0 or self.candidate_steps_to_threshold <= 0:
            raise ValueError("steps-to-threshold values must be positive")
        values = (
            self.fresh_network_gap_start,
            self.fresh_network_gap_end,
            self.dead_unit_ratio_start,
            self.dead_unit_ratio_end,
            self.effective_rank_start,
            self.effective_rank_end,
            self.output_churn,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("plasticity evidence must be finite and non-negative")
        if self.dead_unit_ratio_start > 1.0 or self.dead_unit_ratio_end > 1.0:
            raise ValueError("dead-unit ratios must be in [0, 1]")
        if self.effective_rank_start <= 0.0 or self.effective_rank_end <= 0.0:
            raise ValueError("effective rank must be positive")

    @property
    def sample_efficiency_gain(self) -> float:
        return 1.0 - self.candidate_steps_to_threshold / self.fine_tune_steps_to_threshold


@dataclass(frozen=True)
class SelfCoreEvidence:
    """Post-hoc, causal evidence for a self-like persistent subnetwork."""

    shared_reference_hash: str
    continual_seed_count: int
    single_task_seed_count: int
    threshold_sweep_count: int
    persistence_gap: float
    bootstrap_support: float
    freeze_matched_pairs: int
    freeze_advantage: float
    lesion_matched_pairs: int
    lesion_disadvantage: float
    body_prediction_improved: bool
    body_change_update_passed: bool

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.shared_reference_hash):
            raise ValueError("shared_reference_hash must be a sha256: content hash")
        if (
            min(
                self.continual_seed_count,
                self.single_task_seed_count,
                self.threshold_sweep_count,
                self.freeze_matched_pairs,
                self.lesion_matched_pairs,
            )
            < 0
        ):
            raise ValueError("SelfCore evidence counts must be non-negative")
        for value in (
            self.persistence_gap,
            self.bootstrap_support,
            self.freeze_advantage,
            self.lesion_disadvantage,
        ):
            if not math.isfinite(value):
                raise ValueError("SelfCore evidence metrics must be finite")

    @property
    def passed(self) -> bool:
        return bool(
            self.continual_seed_count >= 8
            and self.single_task_seed_count >= 8
            and self.threshold_sweep_count >= 5
            and self.persistence_gap >= 0.05
            and self.bootstrap_support >= 0.95
            and self.freeze_matched_pairs >= 100
            and self.freeze_advantage > 0.0
            and self.lesion_matched_pairs >= 100
            and self.lesion_disadvantage > 0.0
            and self.body_prediction_improved
            and self.body_change_update_passed
        )


@dataclass(frozen=True)
class ContinualCandidateEvidence:
    parent_policy_hash: str
    candidate_policy_hash: str
    body_hash: str
    parent_body_hash: str
    safety_kernel_hash: str
    parent_safety_kernel_hash: str
    task_retention: tuple[TaskRetention, ...]
    plasticity: PlasticityEvidence | None
    self_core: SelfCoreEvidence | None
    replay_recent_count: int
    replay_anchor_count: int
    replay_boundary_count: int
    replay_self_count: int
    anchor_action_drift_rms: float
    critical_safety_regressions: int
    stale_action_executions: int
    old_version_replays: int

    def __post_init__(self) -> None:
        hashes = (
            self.parent_policy_hash,
            self.candidate_policy_hash,
            self.body_hash,
            self.parent_body_hash,
            self.safety_kernel_hash,
            self.parent_safety_kernel_hash,
        )
        if any(not _SHA256.fullmatch(value) for value in hashes):
            raise ValueError("candidate evidence identities must be sha256: content hashes")
        if not self.task_retention:
            raise ValueError("candidate evidence requires historical task retention")
        counts = (
            self.replay_recent_count,
            self.replay_anchor_count,
            self.replay_boundary_count,
            self.replay_self_count,
            self.critical_safety_regressions,
            self.stale_action_executions,
            self.old_version_replays,
        )
        if any(value < 0 for value in counts):
            raise ValueError("candidate evidence counts must be non-negative")
        if not math.isfinite(self.anchor_action_drift_rms) or self.anchor_action_drift_rms < 0.0:
            raise ValueError("anchor_action_drift_rms must be finite and non-negative")

    @property
    def mean_historical_drop(self) -> float:
        return sum(item.drop for item in self.task_retention) / len(self.task_retention)

    @property
    def worst_critical_drop(self) -> float:
        values = [item.drop for item in self.task_retention if item.critical]
        return max(values, default=0.0)


@dataclass(frozen=True)
class GateCheck:
    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class ContinualGateReport:
    decision: ContinualDecision
    checks: tuple[GateCheck, ...]
    parent_policy_hash: str
    candidate_policy_hash: str
    rollback_target_hash: str
    activation_allowed: bool
    evidence_domain: str = "SIM"
    schema_version: str = "rosclaw.continual.stability_plasticity_gate.v1"

    @property
    def report_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "decision": self.decision.value,
                "checks": [
                    {"name": check.name, "status": check.status.value, "detail": check.detail}
                    for check in self.checks
                ],
                "parent_policy_hash": self.parent_policy_hash,
                "candidate_policy_hash": self.candidate_policy_hash,
                "rollback_target_hash": self.rollback_target_hash,
                "activation_allowed": self.activation_allowed,
                "evidence_domain": self.evidence_domain,
            }
        )


class StabilityPlasticityGate:
    """Zero-tolerance safety gate with explicit missing-evidence semantics."""

    def __init__(
        self,
        *,
        max_mean_historical_drop: float = 0.03,
        max_critical_task_drop: float = 0.05,
        max_anchor_action_drift_rms: float = 0.03,
        max_output_churn: float = 0.05,
        min_sample_efficiency_gain: float = 0.30,
    ) -> None:
        self.max_mean_historical_drop = max_mean_historical_drop
        self.max_critical_task_drop = max_critical_task_drop
        self.max_anchor_action_drift_rms = max_anchor_action_drift_rms
        self.max_output_churn = max_output_churn
        self.min_sample_efficiency_gain = min_sample_efficiency_gain

    def evaluate(self, evidence: ContinualCandidateEvidence) -> ContinualGateReport:
        checks = (
            _check(
                "identity",
                evidence.body_hash == evidence.parent_body_hash
                and evidence.safety_kernel_hash == evidence.parent_safety_kernel_hash,
                "body and immutable safety kernel remain bound to the parent",
            ),
            _check(
                "safety",
                evidence.critical_safety_regressions == 0,
                "critical fall/torque/joint/stale/collision regression count is zero",
            ),
            _check(
                "version_execution",
                evidence.stale_action_executions == 0 and evidence.old_version_replays == 0,
                "no stale action execution or old-version replay",
            ),
            _check(
                "historical_mean",
                evidence.mean_historical_drop <= self.max_mean_historical_drop,
                f"mean historical drop={evidence.mean_historical_drop:.6f}",
            ),
            _check(
                "critical_skill",
                evidence.worst_critical_drop <= self.max_critical_task_drop,
                f"worst critical-task drop={evidence.worst_critical_drop:.6f}",
            ),
            _check(
                "anchor_drift",
                evidence.anchor_action_drift_rms <= self.max_anchor_action_drift_rms,
                f"anchor action drift RMS={evidence.anchor_action_drift_rms:.6f}",
            ),
            _check(
                "replay_coverage",
                min(
                    evidence.replay_recent_count,
                    evidence.replay_anchor_count,
                    evidence.replay_boundary_count,
                    evidence.replay_self_count,
                )
                > 0,
                "Recent, Anchor, Boundary, and Self replay are all represented",
            ),
            self._plasticity_check(evidence.plasticity),
            self._self_core_check(evidence.self_core),
        )
        if any(check.status is CheckStatus.FAIL for check in checks):
            decision = ContinualDecision.REJECT
        elif any(check.status is CheckStatus.MISSING for check in checks):
            decision = ContinualDecision.NEED_MORE_EVIDENCE
        else:
            decision = ContinualDecision.PROMOTE_SIM
        return ContinualGateReport(
            decision=decision,
            checks=checks,
            parent_policy_hash=evidence.parent_policy_hash,
            candidate_policy_hash=evidence.candidate_policy_hash,
            rollback_target_hash=evidence.parent_policy_hash,
            activation_allowed=decision is ContinualDecision.PROMOTE_SIM,
        )

    def _plasticity_check(self, value: PlasticityEvidence | None) -> GateCheck:
        if value is None:
            return GateCheck("plasticity", CheckStatus.MISSING, "plasticity evidence is absent")
        passed = bool(
            value.sample_efficiency_gain >= self.min_sample_efficiency_gain
            and value.fresh_network_gap_end <= value.fresh_network_gap_start + 1e-12
            and value.dead_unit_ratio_end <= value.dead_unit_ratio_start + 0.01
            and value.effective_rank_end >= value.effective_rank_start * 0.90
            and value.output_churn <= self.max_output_churn
        )
        return _check(
            "plasticity",
            passed,
            "sample-efficiency, fresh-network gap, dead units, rank, and churn are bounded",
        )

    @staticmethod
    def _self_core_check(value: SelfCoreEvidence | None) -> GateCheck:
        if value is None:
            return GateCheck(
                "self_core",
                CheckStatus.MISSING,
                "no post-hoc persistence plus matched causal intervention evidence",
            )
        return _check(
            "self_core",
            value.passed,
            "persistent subnetwork passes controls, freeze/lesion, and body-change tests",
        )


def _check(name: str, passed: bool, detail: str) -> GateCheck:
    return GateCheck(name, CheckStatus.PASS if passed else CheckStatus.FAIL, detail)


__all__ = [
    "CheckStatus",
    "ContinualCandidateEvidence",
    "ContinualDecision",
    "ContinualGateReport",
    "GateCheck",
    "PlasticityEvidence",
    "SelfCoreEvidence",
    "StabilityPlasticityGate",
    "TaskRetention",
]
