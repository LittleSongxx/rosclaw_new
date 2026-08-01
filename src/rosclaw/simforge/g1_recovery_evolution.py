"""Evidence-gated self-evolution for G1 post-kick momentum unloading.

The learned object is a bounded GoalForge parameter expert, not a new torque
controller.  A candidate may replace its parent only inside the exact SIM
regime whose strict-replay evidence promoted it.  Every other regime routes to
the retained parent, which is the stability side of the stability-plasticity
contract and the deterministic rollback target.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw.simforge.g1_recovery_quality import (
    G1MomentumUnloadingComparison,
    G1RecoveryQuality,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters, hash_json

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class G1RecoveryEvolutionDecision(StrEnum):
    SIM_CHAMPION = "SIM_CHAMPION"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class G1RecoveryRouteReceipt:
    candidate_hash: str
    recovery_controller_hash: str
    recovery_config_hash: str
    evaluated_regime_commitment: str
    promoted_regime_commitment: str
    selected_policy_hash: str
    rollback_target_hash: str
    used_candidate: bool
    fallback_reason: str | None
    evidence_domain: str = "SIM"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.recovery_route_receipt.v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1MomentumUnloadingEvolution:
    body_hash: str
    kick_prior_hash: str
    recovery_controller_hash: str
    recovery_config_hash: str
    regime_commitment: str
    parent: ShotParameters
    candidate: ShotParameters
    parent_metrics: G1RecoveryQuality
    candidate_metrics: G1RecoveryQuality
    comparison: G1MomentumUnloadingComparison
    decision: G1RecoveryEvolutionDecision
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.g1_goalforge.momentum_unloading_evolution.v2"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("kick_prior_hash", self.kick_prior_hash),
            ("recovery_controller_hash", self.recovery_controller_hash),
            ("recovery_config_hash", self.recovery_config_hash),
            ("regime_commitment", self.regime_commitment),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        expected = (
            G1RecoveryEvolutionDecision.SIM_CHAMPION
            if self.comparison.passed
            else G1RecoveryEvolutionDecision.REJECTED
        )
        if self.decision is not expected:
            raise ValueError("recovery evolution decision contradicts its promotion evidence")
        if self.parent.policy_hash == self.candidate.policy_hash:
            raise ValueError("recovery evolution candidate must differ from its parent")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("recovery evolution activation ceiling must remain SIM_ONLY")

    @classmethod
    def evaluate(
        cls,
        *,
        body_hash: str,
        kick_prior_hash: str,
        recovery_controller_hash: str,
        recovery_config_hash: str,
        regime_commitment: str,
        parent: ShotParameters,
        candidate: ShotParameters,
        parent_metrics: G1RecoveryQuality,
        candidate_metrics: G1RecoveryQuality,
        comparison: G1MomentumUnloadingComparison,
    ) -> G1MomentumUnloadingEvolution:
        decision = (
            G1RecoveryEvolutionDecision.SIM_CHAMPION
            if comparison.passed
            else G1RecoveryEvolutionDecision.REJECTED
        )
        return cls(
            body_hash=body_hash,
            kick_prior_hash=kick_prior_hash,
            recovery_controller_hash=recovery_controller_hash,
            recovery_config_hash=recovery_config_hash,
            regime_commitment=regime_commitment,
            parent=parent,
            candidate=candidate,
            parent_metrics=parent_metrics,
            candidate_metrics=candidate_metrics,
            comparison=comparison,
            decision=decision,
        )

    @property
    def candidate_hash(self) -> str:
        return hash_json(self._bound_dict())

    @property
    def rollback_target_hash(self) -> str:
        return self.parent.policy_hash

    def route(self, *, regime_commitment: str) -> tuple[ShotParameters, G1RecoveryRouteReceipt]:
        if not _SHA256.fullmatch(regime_commitment):
            raise ValueError("evaluated regime commitment must be a sha256 content hash")
        in_regime = regime_commitment == self.regime_commitment
        promoted = self.decision is G1RecoveryEvolutionDecision.SIM_CHAMPION
        use_candidate = bool(in_regime and promoted)
        selected = self.candidate if use_candidate else self.parent
        fallback_reason = (
            None
            if use_candidate
            else "candidate_rejected"
            if in_regime
            else "out_of_evidence_regime"
        )
        receipt = G1RecoveryRouteReceipt(
            candidate_hash=self.candidate_hash,
            recovery_controller_hash=self.recovery_controller_hash,
            recovery_config_hash=self.recovery_config_hash,
            evaluated_regime_commitment=regime_commitment,
            promoted_regime_commitment=self.regime_commitment,
            selected_policy_hash=selected.policy_hash,
            rollback_target_hash=self.rollback_target_hash,
            used_candidate=use_candidate,
            fallback_reason=fallback_reason,
        )
        return selected, receipt

    def _bound_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "recovery_controller_hash": self.recovery_controller_hash,
            "recovery_config_hash": self.recovery_config_hash,
            "regime_commitment": self.regime_commitment,
            "parent": self.parent.to_dict(),
            "candidate": self.candidate.to_dict(),
            "parent_metrics": self.parent_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "comparison": self.comparison.to_dict(),
            "decision": self.decision.value,
            "activation_ceiling": self.activation_ceiling,
            "rollback_target_hash": self.rollback_target_hash,
            "evidence_domain": "SIM",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._bound_dict(),
            "candidate_hash": self.candidate_hash,
            "claims": {
                "self_evolution_scope": "bounded_post_kick_parameter_expert",
                "stability_plasticity": "exact_regime_candidate_else_parent",
                "real_hardware": False,
                "unrestricted_online_weight_updates": False,
            },
        }


__all__ = [
    "G1MomentumUnloadingEvolution",
    "G1RecoveryEvolutionDecision",
    "G1RecoveryRouteReceipt",
]
