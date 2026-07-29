"""Evidence receipt for deterministic candidate inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.feedback.contracts import canonical_hash


@dataclass(frozen=True)
class CandidateInferenceReceipt:
    policy_version: int
    policy_version_hash: str
    parent_version_hash: str
    artifact_hash: str
    body_hash: str
    observation_contract_hash: str
    action_contract_hash: str
    trace_hash: str
    inference_count: int
    actions_bounded: bool
    action_rms: float
    maximum_action_limit_ratio: float
    contact_timing_enabled_count: int
    contact_timing_mean_confidence: float
    version_switch_count: int = 0
    registry_write_count: int = 0
    dds_opened: bool = False
    evidence_domain: str = "SIM_ONLY"
    schema_version: str = "rosclaw.continual.candidate_inference_receipt.v1"

    @property
    def receipt_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["CandidateInferenceReceipt"]
