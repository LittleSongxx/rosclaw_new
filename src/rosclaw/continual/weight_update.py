"""Double-buffered residual-policy staging with atomic safe-boundary swaps."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from rosclaw.continual.contracts import PolicyVersion, SkillPhase
from rosclaw.continual.stability import ContinualDecision, ContinualGateReport


class WeightSlotState(StrEnum):
    ACTIVE = "ACTIVE"
    CANDIDATE_STAGED = "CANDIDATE_STAGED"
    FROZEN = "FROZEN"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True)
class WeightSlotReceipt:
    state: WeightSlotState
    active_version_hash: str
    candidate_version_hash: str | None
    rollback_version_hash: str | None
    reason: str
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.continual.weight_slot_receipt.v1"


class ResidualWeightSlot:
    """In-memory reference state machine; never writes the active Registry."""

    _SAFE_SWAP_PHASES = {SkillPhase.STAND, SkillPhase.PREPARE, SkillPhase.COMPLETE}

    def __init__(self, active: PolicyVersion, *, active_artifact: bytes) -> None:
        self._verify_artifact(active, active_artifact)
        self.active = active
        self.candidate: PolicyVersion | None = None
        self.rollback_target: PolicyVersion | None = None
        self.state = WeightSlotState.ACTIVE

    def stage(self, candidate: PolicyVersion, *, artifact: bytes) -> WeightSlotReceipt:
        if self.state is WeightSlotState.FROZEN:
            raise RuntimeError("weight slot is frozen")
        if self.candidate is not None:
            raise RuntimeError("a candidate is already staged")
        self._verify_artifact(candidate, artifact)
        if candidate.version != self.active.version + 1:
            raise ValueError("candidate policy version must increment active version by one")
        if candidate.parent_version_hash != self.active.version_hash:
            raise ValueError("candidate parent does not match the active policy")
        if candidate.body_hash != self.active.body_hash:
            raise ValueError("candidate cannot change body identity")
        if candidate.safety_kernel_hash != self.active.safety_kernel_hash:
            raise ValueError("candidate cannot change the immutable safety kernel")
        self.candidate = candidate
        self.state = WeightSlotState.CANDIDATE_STAGED
        return self._receipt("candidate checksum verified and staged")

    def activate(
        self,
        *,
        phase: SkillPhase,
        gate_report: ContinualGateReport,
    ) -> WeightSlotReceipt:
        if self.candidate is None or self.state is not WeightSlotState.CANDIDATE_STAGED:
            raise RuntimeError("no candidate is staged")
        if phase not in self._SAFE_SWAP_PHASES:
            self.state = WeightSlotState.FROZEN
            return self._receipt(f"unsafe mid-motion swap requested during {phase.value}")
        if (
            gate_report.decision is not ContinualDecision.PROMOTE_SIM
            or not gate_report.activation_allowed
            or gate_report.evidence_domain != "SIM"
            or gate_report.parent_policy_hash != self.active.artifact_hash
            or gate_report.candidate_policy_hash != self.candidate.artifact_hash
        ):
            self.state = WeightSlotState.FROZEN
            return self._receipt("promotion evidence is missing, rejected, or identity-mismatched")
        self.rollback_target = self.active
        self.active = self.candidate
        self.candidate = None
        self.state = WeightSlotState.ACTIVE
        return self._receipt("SIM-only candidate atomically activated at a safe boundary")

    def rollback(self, *, reason: str) -> WeightSlotReceipt:
        if not reason.strip():
            raise ValueError("rollback reason must not be empty")
        if self.rollback_target is None:
            raise RuntimeError("no rollback target is available")
        failed = self.active
        self.active = self.rollback_target
        self.rollback_target = failed
        self.candidate = None
        self.state = WeightSlotState.ROLLED_BACK
        return self._receipt(reason)

    def _receipt(self, reason: str) -> WeightSlotReceipt:
        return WeightSlotReceipt(
            state=self.state,
            active_version_hash=self.active.version_hash,
            candidate_version_hash=(self.candidate.version_hash if self.candidate else None),
            rollback_version_hash=(
                self.rollback_target.version_hash if self.rollback_target else None
            ),
            reason=reason,
        )

    @staticmethod
    def _verify_artifact(policy: PolicyVersion, artifact: bytes) -> None:
        if not artifact:
            raise ValueError("policy artifact must not be empty")
        actual = "sha256:" + hashlib.sha256(artifact).hexdigest()
        if actual != policy.artifact_hash:
            raise ValueError("policy artifact checksum does not match its version contract")


__all__ = ["ResidualWeightSlot", "WeightSlotReceipt", "WeightSlotState"]
