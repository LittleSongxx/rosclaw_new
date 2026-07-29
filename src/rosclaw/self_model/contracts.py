"""Operational embodied-self contracts without consciousness claims."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _hash(label: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


@dataclass(frozen=True)
class ScalarBelief:
    mean: float
    standard_deviation: float
    confidence: float
    unit: str

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value)
            for value in (self.mean, self.standard_deviation, self.confidence)
        ):
            raise ValueError("belief values must be finite")
        if self.standard_deviation < 0.0:
            raise ValueError("belief standard deviation must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("belief confidence must be in [0, 1]")
        if not self.unit.strip():
            raise ValueError("belief unit must not be empty")

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "confidence": self.confidence,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class CapabilityBelief:
    success_probability: float
    uncertainty: float
    evidence_count: int
    policy_version: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("capability success probability must be in [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("capability uncertainty must be in [0, 1]")
        if self.evidence_count < 0 or self.policy_version < 0:
            raise ValueError("capability evidence count and policy version must be non-negative")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "success_probability": self.success_probability,
            "uncertainty": self.uncertainty,
            "evidence_count": self.evidence_count,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class SelfIdentity:
    """S0: immutable body/layout identity plus versioned control lineage."""

    body_hash: str
    sensor_layout_hash: str
    actuator_layout_hash: str
    safety_kernel_hash: str
    controller_lineage: tuple[str, ...]
    current_policy_versions: Mapping[str, int]
    discovered_self_core_hash: str | None = None
    self_core_evidence_hash: str | None = None
    schema_version: str = "rosclaw.self.identity.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("sensor_layout_hash", self.sensor_layout_hash),
            ("actuator_layout_hash", self.actuator_layout_hash),
            ("safety_kernel_hash", self.safety_kernel_hash),
        ):
            _hash(label, value)
        if not self.controller_lineage:
            raise ValueError("controller lineage must not be empty")
        for value in self.controller_lineage:
            _hash("controller_lineage", value)
        versions = {str(key): int(value) for key, value in self.current_policy_versions.items()}
        if not versions or any(not key.strip() or value < 0 for key, value in versions.items()):
            raise ValueError("current policy versions must be non-empty and non-negative")
        paired = (self.discovered_self_core_hash, self.self_core_evidence_hash)
        if (paired[0] is None) != (paired[1] is None):
            raise ValueError("a discovered SelfCore requires its causal evidence hash")
        for value in paired:
            if value is not None:
                _hash("SelfCore identity", value)
        object.__setattr__(self, "current_policy_versions", MappingProxyType(versions))

    @property
    def identity_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "sensor_layout_hash": self.sensor_layout_hash,
            "actuator_layout_hash": self.actuator_layout_hash,
            "safety_kernel_hash": self.safety_kernel_hash,
            "controller_lineage": list(self.controller_lineage),
            "current_policy_versions": dict(sorted(self.current_policy_versions.items())),
            "discovered_self_core_hash": self.discovered_self_core_hash,
            "self_core_evidence_hash": self.self_core_evidence_hash,
        }


@dataclass(frozen=True)
class SelfStateSnapshot:
    """S1/S4: current body beliefs and calibrated capability estimates."""

    identity_hash: str
    body_hash: str
    sequence: int
    timestamp_ns: int
    joint_health: Mapping[str, float]
    motor_gain_beliefs: Mapping[str, ScalarBelief]
    joint_zero_bias_beliefs: Mapping[str, ScalarBelief]
    latency_belief: ScalarBelief
    friction_belief: ScalarBelief
    payload_belief: ScalarBelief
    balance_margin: float
    energy_state: float
    sensor_quality: Mapping[str, float]
    capabilities: Mapping[str, CapabilityBelief]
    schema_version: str = "rosclaw.self.state_snapshot.v1"

    def __post_init__(self) -> None:
        _hash("identity_hash", self.identity_hash)
        _hash("body_hash", self.body_hash)
        if self.sequence < 0 or self.timestamp_ns < 0:
            raise ValueError("self-state sequence and timestamp must be non-negative")
        if not math.isfinite(self.balance_margin):
            raise ValueError("balance margin must be finite")
        if not math.isfinite(self.energy_state) or not 0.0 <= self.energy_state <= 1.0:
            raise ValueError("energy state must be finite and normalized to [0, 1]")
        joint_health = _probability_mapping(self.joint_health, label="joint_health")
        sensor_quality = _probability_mapping(self.sensor_quality, label="sensor_quality")
        if set(self.motor_gain_beliefs) != set(joint_health):
            raise ValueError("motor gain beliefs must cover every joint-health entry")
        if set(self.joint_zero_bias_beliefs) != set(joint_health):
            raise ValueError("joint zero-bias beliefs must cover every joint-health entry")
        if not self.capabilities:
            raise ValueError("self state must include capability beliefs")
        object.__setattr__(self, "joint_health", joint_health)
        object.__setattr__(self, "sensor_quality", sensor_quality)
        object.__setattr__(
            self,
            "motor_gain_beliefs",
            MappingProxyType(dict(self.motor_gain_beliefs)),
        )
        object.__setattr__(
            self,
            "joint_zero_bias_beliefs",
            MappingProxyType(dict(self.joint_zero_bias_beliefs)),
        )
        object.__setattr__(self, "capabilities", MappingProxyType(dict(self.capabilities)))

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_hash": self.identity_hash,
            "body_hash": self.body_hash,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "joint_health": dict(self.joint_health),
            "motor_gain_beliefs": {
                key: value.to_dict() for key, value in sorted(self.motor_gain_beliefs.items())
            },
            "joint_zero_bias_beliefs": {
                key: value.to_dict() for key, value in sorted(self.joint_zero_bias_beliefs.items())
            },
            "latency_belief": self.latency_belief.to_dict(),
            "friction_belief": self.friction_belief.to_dict(),
            "payload_belief": self.payload_belief.to_dict(),
            "balance_margin": self.balance_margin,
            "energy_state": self.energy_state,
            "sensor_quality": dict(self.sensor_quality),
            "capabilities": {
                key: value.to_dict() for key, value in sorted(self.capabilities.items())
            },
        }


@dataclass(frozen=True)
class AgencyAssessment:
    """S3: action-conditioned attribution, not a subjective consciousness claim."""

    self_state_hash: str
    action_hash: str
    predicted_outcome_hash: str
    observed_outcome_hash: str
    self_caused_probability: float
    external_disturbance_probability: float
    sensor_fault_probability: float
    schema_version: str = "rosclaw.self.agency_assessment.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("self_state_hash", self.self_state_hash),
            ("action_hash", self.action_hash),
            ("predicted_outcome_hash", self.predicted_outcome_hash),
            ("observed_outcome_hash", self.observed_outcome_hash),
        ):
            _hash(label, value)
        probabilities = (
            self.self_caused_probability,
            self.external_disturbance_probability,
            self.sensor_fault_probability,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("agency probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("agency probabilities must sum to one")

    @property
    def assessment_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "self_state_hash": self.self_state_hash,
                "action_hash": self.action_hash,
                "predicted_outcome_hash": self.predicted_outcome_hash,
                "observed_outcome_hash": self.observed_outcome_hash,
                "self_caused_probability": self.self_caused_probability,
                "external_disturbance_probability": self.external_disturbance_probability,
                "sensor_fault_probability": self.sensor_fault_probability,
            }
        )


def _probability_mapping(value: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    normalized = {str(key): float(item) for key, item in value.items()}
    if not normalized or any(
        not key.strip() or not math.isfinite(item) or not 0.0 <= item <= 1.0
        for key, item in normalized.items()
    ):
        raise ValueError(f"{label} must contain named finite values in [0, 1]")
    return MappingProxyType(normalized)


__all__ = [
    "AgencyAssessment",
    "CapabilityBelief",
    "ScalarBelief",
    "SelfIdentity",
    "SelfStateSnapshot",
]
