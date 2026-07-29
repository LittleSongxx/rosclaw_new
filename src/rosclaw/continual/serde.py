"""Lossless, hash-checked serialization for versioned continual experience."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rosclaw.continual.contracts import (
    ControlSegment,
    CostVector,
    ExperiencePartition,
    ExperienceUse,
    PolicyVersion,
    RewardVector,
    SkillPhase,
    VersionedTrajectory,
)
from rosclaw.continual.experience import ExperienceBatch, ExperienceRecord

_ENVELOPE_SCHEMA = "rosclaw.continual.experience_batch_envelope.v1"


def experience_batch_to_dict(batch: ExperienceBatch) -> dict[str, Any]:
    """Return a complete transport envelope, not the abbreviated audit view."""

    trajectories = {
        record.trajectory.trajectory_hash: record.trajectory.to_dict() for record in batch.records
    }
    return {
        "schema_version": _ENVELOPE_SCHEMA,
        "batch_schema_version": batch.schema_version,
        "learner_version": batch.learner_version,
        "requested_counts": {
            partition.value: count
            for partition, count in sorted(
                batch.requested_counts.items(), key=lambda item: item[0].value
            )
        },
        "permitted_uses": [item.value for item in batch.permitted_uses],
        "trajectories": dict(sorted(trajectories.items())),
        "records": [
            {
                "schema_version": record.schema_version,
                "partition": record.partition.value,
                "anchor_policy_hash": record.anchor_policy_hash,
                "boundary_reason": record.boundary_reason,
                "self_change_hash": record.self_change_hash,
                "near_boundary_score": record.near_boundary_score,
                "trajectory_hash": record.trajectory.trajectory_hash,
                "record_hash": record.record_hash,
            }
            for record in batch.records
        ],
        "batch_hash": batch.batch_hash,
    }


def experience_batch_from_dict(value: Mapping[str, Any]) -> ExperienceBatch:
    """Rebuild immutable contracts and reject any identity or payload tampering."""

    _schema(value, "schema_version", _ENVELOPE_SCHEMA)
    records_raw = _list(value, "records")
    uses_raw = _list(value, "permitted_uses")
    counts_raw = _mapping(value, "requested_counts")
    trajectories_raw = _mapping(value, "trajectories")
    trajectories = {
        str(expected_hash): _trajectory(_as_mapping(raw, "trajectory"))
        for expected_hash, raw in trajectories_raw.items()
    }
    if any(
        expected_hash != trajectory.trajectory_hash
        for expected_hash, trajectory in trajectories.items()
    ):
        raise ValueError("versioned trajectory hash mismatch")
    records: list[ExperienceRecord] = []
    for raw in records_raw:
        item = _as_mapping(raw, "record")
        _schema(item, "schema_version", "rosclaw.continual.experience_record.v1")
        trajectory_hash = str(item["trajectory_hash"])
        if trajectory_hash not in trajectories:
            raise ValueError("experience record references an absent trajectory")
        record = ExperienceRecord(
            trajectory=trajectories[trajectory_hash],
            partition=ExperiencePartition(str(item["partition"])),
            anchor_policy_hash=_optional_string(item.get("anchor_policy_hash")),
            boundary_reason=_optional_string(item.get("boundary_reason")),
            self_change_hash=_optional_string(item.get("self_change_hash")),
            near_boundary_score=float(item["near_boundary_score"]),
        )
        if item.get("record_hash") != record.record_hash:
            raise ValueError("experience record hash mismatch")
        records.append(record)
    batch = ExperienceBatch(
        records=tuple(records),
        permitted_uses=tuple(ExperienceUse(str(item)) for item in uses_raw),
        requested_counts={
            ExperiencePartition(str(partition)): int(count)
            for partition, count in counts_raw.items()
        },
        learner_version=int(value["learner_version"]),
    )
    if value.get("batch_schema_version") != batch.schema_version:
        raise ValueError("experience batch schema mismatch")
    if value.get("batch_hash") != batch.batch_hash:
        raise ValueError("experience batch hash mismatch")
    return batch


def _trajectory(value: Mapping[str, Any]) -> VersionedTrajectory:
    _schema(value, "schema_version", "rosclaw.continual.versioned_trajectory.v1")
    return VersionedTrajectory(
        segments=tuple(_segment(_as_mapping(item, "segment")) for item in _list(value, "segments")),
        strict_replay=bool(value["strict_replay"]),
    )


def _segment(value: Mapping[str, Any]) -> ControlSegment:
    _schema(value, "schema_version", "rosclaw.continual.control_segment.v1")
    policy = _policy(_mapping(value, "policy"))
    if value.get("policy_version_hash") != policy.version_hash:
        raise ValueError("control segment policy hash mismatch")
    return ControlSegment(
        segment_id=str(value["segment_id"]),
        episode_id=str(value["episode_id"]),
        task_id=str(value["task_id"]),
        phase=SkillPhase(str(value["phase"])),
        start_step=int(value["start_step"]),
        end_step=int(value["end_step"]),
        policy=policy,
        controller_snapshot_hash=str(value["controller_snapshot_hash"]),
        body_hash=str(value["body_hash"]),
        regime_hash=str(value["regime_hash"]),
        self_state_hash=str(value["self_state_hash"]),
        observation=_ordered_numeric(
            _mapping(value, "observation"),
            policy.observation_names,
            "observation",
        ),
        residual_action=_ordered_numeric(
            _mapping(value, "residual_action"),
            policy.residual_action_names,
            "residual_action",
        ),
        next_observation=_ordered_numeric(
            _mapping(value, "next_observation"),
            policy.observation_names,
            "next_observation",
        ),
        behavior_logprob=float(value["behavior_logprob"]),
        reward=RewardVector(**_numeric_kwargs(_mapping(value, "reward"))),
        cost=CostVector(**_numeric_kwargs(_mapping(value, "cost"))),
        terminal=bool(value["terminal"]),
    )


def _policy(value: Mapping[str, Any]) -> PolicyVersion:
    _schema(value, "schema_version", "rosclaw.continual.policy_version.v1")
    return PolicyVersion(
        version=int(value["version"]),
        artifact_hash=str(value["artifact_hash"]),
        parent_version_hash=_optional_string(value.get("parent_version_hash")),
        controller_snapshot_hash=str(value["controller_snapshot_hash"]),
        body_hash=str(value["body_hash"]),
        safety_kernel_hash=str(value["safety_kernel_hash"]),
        observation_names=tuple(str(item) for item in _list(value, "observation_names")),
        residual_action_names=tuple(str(item) for item in _list(value, "residual_action_names")),
    )


def _schema(value: Mapping[str, Any], key: str, expected: str) -> None:
    if value.get(key) != expected:
        raise ValueError(f"unsupported {key}: {value.get(key)!r}")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _as_mapping(value.get(key), key)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list):
        raise ValueError(f"{key} must be a list")
    return result


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _numeric_kwargs(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in value.items()}


def _ordered_numeric(
    value: Mapping[str, Any],
    names: tuple[str, ...],
    label: str,
) -> dict[str, float]:
    if set(value) != set(names):
        raise ValueError(f"{label} keys do not match the policy contract")
    return {name: float(value[name]) for name in names}


__all__ = ["experience_batch_from_dict", "experience_batch_to_dict"]
