"""Immutable contracts for ROSClaw's synchronous Feedback Plane.

The pre-existing :mod:`rosclaw.feedback` package handles user feedback and
telemetry.  These contracts add a deliberately separate, high-rate control
plane under the same namespace without changing that public API.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical_hash(value: Mapping[str, Any]) -> str:
    """Return a stable content hash for a JSON-compatible mapping."""

    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _deep_immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_immutable(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_immutable(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _immutable_floats(value: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    normalized = {str(key): float(item) for key, item in value.items()}
    if any(not math.isfinite(item) for item in normalized.values()):
        raise ValueError(f"{label} must contain only finite values")
    return MappingProxyType(normalized)


class FallbackMode(StrEnum):
    """Fail-closed behavior used when a residual cannot be trusted."""

    BASE_POLICY_ONLY = "base_policy_only"
    DISABLE_RESIDUAL = "disable_residual"
    FREEZE_AND_STABILIZE = "freeze_and_stabilize"


@dataclass(frozen=True)
class FeedbackLoopSpec:
    """Pinned timing, signal and safety contract for one feedback loop."""

    loop_id: str
    body_hash: str
    controller_hash: str
    reference_signals: tuple[str, ...]
    observation_signals: tuple[str, ...]
    output_limits: Mapping[str, float]
    rate_hz: float = 200.0
    deadline_ms: float = 5.0
    max_observation_age_ms: float = 10.0
    fallback_stale_observation: FallbackMode = FallbackMode.BASE_POLICY_ONLY
    fallback_deadline_miss: FallbackMode = FallbackMode.BASE_POLICY_ONLY
    fallback_unsafe_projection: FallbackMode = FallbackMode.BASE_POLICY_ONLY
    schema_version: str = "rosclaw.feedback.loop_spec.v1"

    def __post_init__(self) -> None:
        if not self.loop_id.strip():
            raise ValueError("loop_id must not be empty")
        for label, value in (
            ("body_hash", self.body_hash),
            ("controller_hash", self.controller_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256: content hash")
        if not 1.0 <= self.rate_hz <= 1000.0:
            raise ValueError("rate_hz must be in [1, 1000]")
        if not 0.0 < self.deadline_ms <= 1000.0 / self.rate_hz:
            raise ValueError("deadline_ms must be positive and no larger than the loop period")
        if self.max_observation_age_ms <= 0.0:
            raise ValueError("max_observation_age_ms must be positive")
        if not self.reference_signals or len(set(self.reference_signals)) != len(
            self.reference_signals
        ):
            raise ValueError("reference_signals must be non-empty and unique")
        if not self.observation_signals or len(set(self.observation_signals)) != len(
            self.observation_signals
        ):
            raise ValueError("observation_signals must be non-empty and unique")
        limits = _immutable_floats(self.output_limits, label="output_limits")
        if not limits or any(limit <= 0.0 for limit in limits.values()):
            raise ValueError("output_limits must be non-empty and strictly positive")
        object.__setattr__(self, "output_limits", limits)

    @property
    def period_ns(self) -> int:
        return int(round(1_000_000_000.0 / self.rate_hz))

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "loop_id": self.loop_id,
            "body_hash": self.body_hash,
            "controller_hash": self.controller_hash,
            "reference_signals": list(self.reference_signals),
            "observation_signals": list(self.observation_signals),
            "output_limits": dict(sorted(self.output_limits.items())),
            "rate_hz": self.rate_hz,
            "deadline_ms": self.deadline_ms,
            "max_observation_age_ms": self.max_observation_age_ms,
            "fallback_stale_observation": self.fallback_stale_observation.value,
            "fallback_deadline_miss": self.fallback_deadline_miss.value,
            "fallback_unsafe_projection": self.fallback_unsafe_projection.value,
        }


@dataclass(frozen=True)
class ErrorState:
    """Reference tracking error and its first two temporal terms."""

    value: Mapping[str, float]
    derivative: Mapping[str, float]
    integral: Mapping[str, float]
    timestamp_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _immutable_floats(self.value, label="error"))
        object.__setattr__(
            self, "derivative", _immutable_floats(self.derivative, label="error derivative")
        )
        object.__setattr__(
            self, "integral", _immutable_floats(self.integral, label="error integral")
        )

    @property
    def rms(self) -> float:
        if not self.value:
            return 0.0
        return math.sqrt(sum(value * value for value in self.value.values()) / len(self.value))


@dataclass(frozen=True)
class FeedbackFrame:
    """One controller input frame; no EventBus or asynchronous dependency."""

    sequence: int
    timestamp_ns: int
    observation_timestamp_ns: int
    phase: float
    reference: Mapping[str, float]
    actual: Mapping[str, float]
    error: ErrorState

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not 0.0 <= self.phase <= 1.0:
            raise ValueError("phase must be in [0, 1]")
        object.__setattr__(self, "reference", _immutable_floats(self.reference, label="reference"))
        object.__setattr__(self, "actual", _immutable_floats(self.actual, label="actual"))


@dataclass(frozen=True)
class ResidualCommand:
    """Raw controller output plus the safety-projected command actually used."""

    sequence: int
    timestamp_ns: int
    valid_until_ns: int
    base_action_hash: str
    raw: Mapping[str, float]
    projected: Mapping[str, float]
    reasons: tuple[str, ...] = ()
    saturation_count: int = 0
    deadline_met: bool = True
    fallback: FallbackMode | None = None
    schema_version: str = "rosclaw.feedback.residual_command.v1"

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.base_action_hash):
            raise ValueError("base_action_hash must be a sha256: content hash")
        object.__setattr__(self, "raw", _immutable_floats(self.raw, label="raw residual"))
        object.__setattr__(
            self, "projected", _immutable_floats(self.projected, label="projected residual")
        )

    @property
    def command_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "sequence": self.sequence,
                "timestamp_ns": self.timestamp_ns,
                "valid_until_ns": self.valid_until_ns,
                "base_action_hash": self.base_action_hash,
                "raw": dict(sorted(self.raw.items())),
                "projected": dict(sorted(self.projected.items())),
                "reasons": list(self.reasons),
                "saturation_count": self.saturation_count,
                "deadline_met": self.deadline_met,
                "fallback": self.fallback.value if self.fallback else None,
            }
        )


@dataclass(frozen=True)
class ControllerSnapshot:
    controller_id: str
    controller_type: str
    body_hash: str
    loop_spec_hash: str
    config: Mapping[str, Any]
    schema_version: str = "rosclaw.feedback.controller_snapshot.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("loop_spec_hash", self.loop_spec_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256: content hash")
        object.__setattr__(self, "config", _deep_immutable(self.config))

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "controller_id": self.controller_id,
                "controller_type": self.controller_type,
                "body_hash": self.body_hash,
                "loop_spec_hash": self.loop_spec_hash,
                "config": _json_value(self.config),
            }
        )


@dataclass(frozen=True)
class AdaptationSnapshot:
    adaptation_id: str
    body_hash: str
    source_receipt_hashes: tuple[str, ...]
    update: Mapping[str, Any]
    bounded: bool
    schema_version: str = "rosclaw.feedback.adaptation_snapshot.v1"

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.body_hash):
            raise ValueError("body_hash must be a sha256: content hash")
        if not self.source_receipt_hashes or any(
            not _SHA256.fullmatch(value) for value in self.source_receipt_hashes
        ):
            raise ValueError("source_receipt_hashes must contain sha256: content hashes")
        object.__setattr__(self, "update", _deep_immutable(self.update))

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "adaptation_id": self.adaptation_id,
                "body_hash": self.body_hash,
                "source_receipt_hashes": list(self.source_receipt_hashes),
                "update": _json_value(self.update),
                "bounded": self.bounded,
            }
        )


@dataclass(frozen=True)
class FeedbackReceipt:
    """Bounded evidence summary emitted asynchronously after a loop run."""

    loop_id: str
    loop_spec_hash: str
    controller_hash: str
    controller_snapshot_hash: str
    body_hash: str
    action_id: str
    reference_hash: str
    observation_hash: str
    trace_hash: str
    samples: int
    initial_error_rms: float | None
    final_error_rms: float | None
    deadline_miss_count: int
    stale_observation_count: int
    saturation_count: int
    latency_p50_ms: float
    latency_p99_ms: float
    jitter_p99_ms: float
    observation_age_max_ms: float
    dropped_frame_count: int
    correction_applied: bool
    safety_projection_applied: bool
    strict_replay: bool
    evidence_domain: str = "SHADOW"
    schema_version: str = "rosclaw.feedback.receipt.v1"

    @property
    def tracking_improved(self) -> bool:
        return bool(
            self.initial_error_rms is not None
            and self.final_error_rms is not None
            and self.final_error_rms < self.initial_error_rms
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "loop_id": self.loop_id,
            "loop_spec_hash": self.loop_spec_hash,
            "controller_hash": self.controller_hash,
            "controller_snapshot_hash": self.controller_snapshot_hash,
            "body_hash": self.body_hash,
            "action_id": self.action_id,
            "reference_hash": self.reference_hash,
            "observation_hash": self.observation_hash,
            "trace_hash": self.trace_hash,
            "samples": self.samples,
            "initial_error_rms": self.initial_error_rms,
            "final_error_rms": self.final_error_rms,
            "tracking_improved": self.tracking_improved,
            "deadline_miss_count": self.deadline_miss_count,
            "stale_observation_count": self.stale_observation_count,
            "saturation_count": self.saturation_count,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "jitter_p99_ms": self.jitter_p99_ms,
            "observation_age_max_ms": self.observation_age_max_ms,
            "dropped_frame_count": self.dropped_frame_count,
            "correction_applied": self.correction_applied,
            "safety_projection_applied": self.safety_projection_applied,
            "strict_replay": self.strict_replay,
            "evidence_domain": self.evidence_domain,
        }

    @property
    def receipt_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class FeedbackInput:
    """Replayable input sample retained outside the synchronous hot path."""

    timestamp_ns: int
    observation_timestamp_ns: int
    phase: float
    reference: Mapping[str, float]
    actual: Mapping[str, float]
    base_action: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference", _immutable_floats(self.reference, label="reference"))
        object.__setattr__(self, "actual", _immutable_floats(self.actual, label="actual"))
        object.__setattr__(
            self, "base_action", _immutable_floats(self.base_action, label="base action")
        )


@dataclass(frozen=True)
class FeedbackTickRecord:
    """Compact per-tick trace used to build evidence after execution."""

    input: FeedbackInput
    error_rms: float
    command_hash: str
    latency_ns: int
    deadline_met: bool
    stale_observation: bool
    saturation_count: int
    residual_norm: float
    correction_applied: bool
    safety_projection_applied: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
