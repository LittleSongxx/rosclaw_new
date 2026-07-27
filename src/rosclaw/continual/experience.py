"""Four-timescale replay with explicit provenance and staleness routing."""

from __future__ import annotations

import math
import random
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rosclaw.continual.contracts import (
    ExperiencePartition,
    ExperienceUse,
    VersionedTrajectory,
)
from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReplayMix:
    recent: float = 0.50
    anchor: float = 0.25
    boundary: float = 0.15
    self_: float = 0.10

    def __post_init__(self) -> None:
        values = self.to_dict()
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("all replay partitions require a finite positive sampling weight")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-12):
            raise ValueError("replay sampling weights must sum to one")

    def to_dict(self) -> dict[ExperiencePartition, float]:
        return {
            ExperiencePartition.RECENT: self.recent,
            ExperiencePartition.ANCHOR: self.anchor,
            ExperiencePartition.BOUNDARY: self.boundary,
            ExperiencePartition.SELF: self.self_,
        }


@dataclass(frozen=True)
class ExperienceBufferConfig:
    capacity_per_partition: int = 4096
    max_policy_lag: int = 1
    mix: ReplayMix = ReplayMix()

    def __post_init__(self) -> None:
        if self.capacity_per_partition <= 0:
            raise ValueError("replay capacity must be positive")
        if self.max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")


@dataclass(frozen=True)
class ExperienceRecord:
    trajectory: VersionedTrajectory
    partition: ExperiencePartition
    anchor_policy_hash: str | None = None
    boundary_reason: str | None = None
    self_change_hash: str | None = None
    near_boundary_score: float = 0.0
    schema_version: str = "rosclaw.continual.experience_record.v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.near_boundary_score <= 1.0:
            raise ValueError("near_boundary_score must be in [0, 1]")
        if self.partition is ExperiencePartition.ANCHOR and not self.anchor_policy_hash:
            raise ValueError("anchor experience requires its champion policy hash")
        if self.partition is ExperiencePartition.BOUNDARY:
            if not self.boundary_reason:
                raise ValueError("boundary experience requires a reason")
            if not self.trajectory.has_critical_cost and self.near_boundary_score < 0.80:
                raise ValueError("boundary experience requires a safety event or near miss")
        if self.partition is ExperiencePartition.SELF and not self.self_change_hash:
            raise ValueError("self experience requires a body-change commitment")
        for label, value in (
            ("anchor_policy_hash", self.anchor_policy_hash),
            ("self_change_hash", self.self_change_hash),
        ):
            if value is not None and not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256: content hash")

    @property
    def record_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_hash": self.trajectory.trajectory_hash,
            "partition": self.partition.value,
            "anchor_policy_hash": self.anchor_policy_hash,
            "boundary_reason": self.boundary_reason,
            "self_change_hash": self.self_change_hash,
            "near_boundary_score": self.near_boundary_score,
        }


@dataclass(frozen=True)
class ExperienceBatch:
    records: tuple[ExperienceRecord, ...]
    permitted_uses: tuple[ExperienceUse, ...]
    requested_counts: Mapping[ExperiencePartition, int]
    learner_version: int
    schema_version: str = "rosclaw.continual.experience_batch.v1"

    def __post_init__(self) -> None:
        if not self.records or len(self.records) != len(self.permitted_uses):
            raise ValueError("experience batch records and uses must be non-empty and aligned")
        object.__setattr__(self, "requested_counts", MappingProxyType(dict(self.requested_counts)))

    @property
    def actor_records(self) -> tuple[ExperienceRecord, ...]:
        return tuple(
            record
            for record, use in zip(self.records, self.permitted_uses, strict=True)
            if use is ExperienceUse.ACTOR_CRITIC_SELF
        )

    @property
    def critic_self_only_records(self) -> tuple[ExperienceRecord, ...]:
        return tuple(
            record
            for record, use in zip(self.records, self.permitted_uses, strict=True)
            if use is ExperienceUse.CRITIC_SELF_ONLY
        )

    @property
    def batch_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "record_hashes": [record.record_hash for record in self.records],
                "permitted_uses": [use.value for use in self.permitted_uses],
                "requested_counts": {
                    key.value: value for key, value in sorted(self.requested_counts.items())
                },
                "learner_version": self.learner_version,
            }
        )


class ContinualExperienceStore:
    """Bounded replay memory; its evidence identity remains content-addressed."""

    def __init__(self, config: ExperienceBufferConfig | None = None) -> None:
        self.config = config or ExperienceBufferConfig()
        self._buffers = {
            partition: deque[ExperienceRecord](maxlen=self.config.capacity_per_partition)
            for partition in ExperiencePartition
        }
        self._seen: set[str] = set()

    def append(self, record: ExperienceRecord) -> None:
        record_hash = record.record_hash
        if record_hash in self._seen:
            raise ValueError("experience record is already present in this lineage")
        self._buffers[record.partition].append(record)
        self._seen.add(record_hash)

    def counts(self) -> Mapping[ExperiencePartition, int]:
        return MappingProxyType(
            {partition: len(records) for partition, records in self._buffers.items()}
        )

    @property
    def ready(self) -> bool:
        return all(self._buffers[partition] for partition in ExperiencePartition)

    def sample(self, *, batch_size: int, learner_version: int, seed: int) -> ExperienceBatch:
        if batch_size < len(ExperiencePartition):
            raise ValueError("batch_size must allow every replay partition to be represented")
        if learner_version < 0:
            raise ValueError("learner_version must be non-negative")
        missing = [partition.value for partition, records in self._buffers.items() if not records]
        if missing:
            raise RuntimeError(f"continual replay is not ready; empty partitions: {missing}")
        counts = _apportion(batch_size, self.config.mix.to_dict())
        rng = random.Random(seed)
        sampled: list[ExperienceRecord] = []
        for partition in ExperiencePartition:
            records = tuple(self._buffers[partition])
            sampled.extend(rng.choice(records) for _ in range(counts[partition]))
        rng.shuffle(sampled)
        uses = tuple(
            record.trajectory.permitted_use(
                learner_version=learner_version,
                max_policy_lag=self.config.max_policy_lag,
            )
            for record in sampled
        )
        if ExperienceUse.REJECT in uses:
            raise RuntimeError("replay contains a future policy version")
        return ExperienceBatch(
            records=tuple(sampled),
            permitted_uses=uses,
            requested_counts=counts,
            learner_version=learner_version,
        )


def _apportion(
    batch_size: int,
    weights: Mapping[ExperiencePartition, float],
) -> Mapping[ExperiencePartition, int]:
    raw = {partition: batch_size * weight for partition, weight in weights.items()}
    counts = {partition: max(1, int(math.floor(value))) for partition, value in raw.items()}
    while sum(counts.values()) > batch_size:
        eligible = [partition for partition, count in counts.items() if count > 1]
        if not eligible:
            raise ValueError("batch_size is too small for the configured replay mix")
        selected = min(eligible, key=lambda partition: raw[partition] - counts[partition])
        counts[selected] -= 1
    while sum(counts.values()) < batch_size:
        selected = max(
            counts,
            key=lambda partition: (
                raw[partition] - counts[partition],
                -list(weights).index(partition),
            ),
        )
        counts[selected] += 1
    return MappingProxyType(counts)


__all__ = [
    "ContinualExperienceStore",
    "ExperienceBatch",
    "ExperienceBufferConfig",
    "ExperienceRecord",
    "ReplayMix",
]
