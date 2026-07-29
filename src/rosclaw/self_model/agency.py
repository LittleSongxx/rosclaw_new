"""Operational action attribution built on forward-model evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class AgencyClass(StrEnum):
    SELF_CAUSED = "SELF_CAUSED"
    EXTERNAL_DISTURBANCE = "EXTERNAL_DISTURBANCE"
    SENSOR_FAULT = "SENSOR_FAULT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AgencyEvidence:
    action_magnitude: float
    prediction_error: float
    external_force_evidence: float
    sensor_inconsistency: float
    action_hash: str
    predicted_outcome_hash: str
    observed_outcome_hash: str
    timestamp_ns: int
    schema_version: str = "rosclaw.self.agency_evidence.v1"

    def __post_init__(self) -> None:
        values = (
            self.action_magnitude,
            self.prediction_error,
            self.external_force_evidence,
            self.sensor_inconsistency,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("agency evidence must be normalized to [0, 1]")
        for value in (
            self.action_hash,
            self.predicted_outcome_hash,
            self.observed_outcome_hash,
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError("agency evidence references must be sha256 hashes")
        if self.timestamp_ns < 0:
            raise ValueError("agency evidence timestamp must be non-negative")


@dataclass(frozen=True)
class OperationalAgencyEstimate:
    classification: AgencyClass
    probabilities: Mapping[AgencyClass, float]
    evidence_hash: str
    confidence: float
    schema_version: str = "rosclaw.self.operational_agency.v1"

    def __post_init__(self) -> None:
        if set(self.probabilities) != set(AgencyClass):
            raise ValueError("agency estimate must contain all four classes")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.probabilities.values()
        ) or not math.isclose(sum(self.probabilities.values()), 1.0, abs_tol=1e-9):
            raise ValueError("agency probabilities must be normalized")
        if self.classification is not max(self.probabilities, key=self.probabilities.__getitem__):
            raise ValueError("agency classification must match maximum probability")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("agency confidence must be in [0, 1]")
        object.__setattr__(
            self,
            "probabilities",
            MappingProxyType(dict(self.probabilities)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "classification": self.classification.value,
            "probabilities": {key.value: value for key, value in self.probabilities.items()},
            "evidence_hash": self.evidence_hash,
            "confidence": self.confidence,
        }


class AgencyEstimator:
    """Attribute outcomes without claiming subjective awareness or intent."""

    def __init__(self, *, temperature: float = 0.65) -> None:
        if not 0.1 <= temperature <= 2.0:
            raise ValueError("agency temperature must be in [0.1, 2.0]")
        self.temperature = temperature

    def estimate(self, evidence: AgencyEvidence) -> OperationalAgencyEstimate:
        action = evidence.action_magnitude
        error = evidence.prediction_error
        external = evidence.external_force_evidence
        sensor = evidence.sensor_inconsistency
        scores = {
            AgencyClass.SELF_CAUSED: 3.0 * action * (1.0 - error) + 0.5 * (1.0 - sensor),
            AgencyClass.EXTERNAL_DISTURBANCE: (3.0 * external * (1.0 - sensor) + 1.5 * error),
            AgencyClass.SENSOR_FAULT: 4.0 * sensor + 0.5 * error,
            AgencyClass.UNKNOWN: (
                1.5 * (1.0 - max(action, external, sensor))
                + error * (1.0 - external) * (1.0 - sensor)
            ),
        }
        maximum = max(scores.values())
        exponentials = {
            key: math.exp((value - maximum) / self.temperature) for key, value in scores.items()
        }
        total = sum(exponentials.values())
        probabilities = {key: value / total for key, value in exponentials.items()}
        classification = max(probabilities, key=probabilities.__getitem__)
        ordered = sorted(probabilities.values(), reverse=True)
        confidence = max(0.0, min(1.0, ordered[0] - ordered[1]))
        evidence_hash = canonical_hash(
            {
                "schema_version": evidence.schema_version,
                "action_magnitude": action,
                "prediction_error": error,
                "external_force_evidence": external,
                "sensor_inconsistency": sensor,
                "action_hash": evidence.action_hash,
                "predicted_outcome_hash": evidence.predicted_outcome_hash,
                "observed_outcome_hash": evidence.observed_outcome_hash,
                "timestamp_ns": evidence.timestamp_ns,
            }
        )
        return OperationalAgencyEstimate(
            classification=classification,
            probabilities=probabilities,
            evidence_hash=evidence_hash,
            confidence=confidence,
        )


__all__ = [
    "AgencyClass",
    "AgencyEstimator",
    "AgencyEvidence",
    "OperationalAgencyEstimate",
]
