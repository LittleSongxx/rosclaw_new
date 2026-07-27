"""Turn matched-evaluation regressions into explicit re-rollout requests."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.continual.serde import policy_version_from_dict
from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class BoundaryReplayRequest:
    scenario_id: str
    scenario_commitment: str
    replay_partition: str
    parent_policy_hash: str
    candidate_policy_hash: str
    parent_status: str
    candidate_status: str
    critical_signals: tuple[str, ...]
    source_evidence_hash: str
    schema_version: str = "rosclaw.continual.boundary_replay_request.v1"

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.replay_partition.strip():
            raise ValueError("boundary replay scenario and partition must not be empty")
        for label, value in (
            ("scenario_commitment", self.scenario_commitment),
            ("parent_policy_hash", self.parent_policy_hash),
            ("candidate_policy_hash", self.candidate_policy_hash),
            ("source_evidence_hash", self.source_evidence_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256: content hash")
        if not self.critical_signals:
            raise ValueError("boundary replay request requires a critical safety signal")

    @property
    def request_hash(self) -> str:
        return canonical_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "request_hash": self.request_hash}


def extract_boundary_replay_requests(
    report: Mapping[str, Any],
    *,
    source_evidence_hash: str,
) -> tuple[BoundaryReplayRequest, ...]:
    """Extract only new Candidate safety failures; never fabricate trajectories."""

    if report.get("schema_version") != "rosclaw.continual.g1_candidate_matched_evaluation.v1":
        raise ValueError("unsupported matched candidate evaluation schema")
    gate = _mapping(report, "gate")
    if gate.get("candidate_activated") is not False:
        raise ValueError("boundary extraction requires a non-activated candidate report")
    parent = policy_version_from_dict(_mapping(report, "parent_policy"))
    candidate = policy_version_from_dict(_mapping(report, "candidate_policy"))
    rows = _sequence(report, "rows")
    parent_rows = {
        str(row["scenario_commitment"]): row
        for raw in rows
        if (row := _mapping_value(raw, "row")).get("arm") == "active_parent_v2"
    }
    candidate_rows = {
        str(row["scenario_commitment"]): row
        for raw in rows
        if (row := _mapping_value(raw, "row")).get("arm") == "candidate_v3"
    }
    requests = []
    for commitment in sorted(set(parent_rows) & set(candidate_rows)):
        baseline = parent_rows[commitment]
        changed = candidate_rows[commitment]
        baseline_critical = _critical_signals(baseline)
        changed_critical = _critical_signals(changed)
        if changed_critical and not baseline_critical:
            requests.append(
                BoundaryReplayRequest(
                    scenario_id=str(changed["scenario_id"]),
                    scenario_commitment=commitment,
                    replay_partition=str(changed["replay_partition"]),
                    parent_policy_hash=parent.version_hash,
                    candidate_policy_hash=candidate.version_hash,
                    parent_status=str(baseline["status"]),
                    candidate_status=str(changed["status"]),
                    critical_signals=changed_critical,
                    source_evidence_hash=source_evidence_hash,
                )
            )
    expected = int(gate.get("critical_safety_regressions", -1))
    if expected != len(requests):
        raise ValueError("matched report critical-regression count does not match its rows")
    return tuple(requests)


def _critical_signals(row: Mapping[str, Any]) -> tuple[str, ...]:
    names = (
        ("fall", "fall"),
        ("joint_violation", "joint_limit"),
        ("torque_violation", "torque_limit"),
    )
    return tuple(label for key, label in names if bool(row.get(key)))


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _mapping_value(value.get(key), key)


def _mapping_value(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Mapping[str, Any], key: str) -> Sequence[Any]:
    result = value.get(key)
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")
    return result


__all__ = ["BoundaryReplayRequest", "extract_boundary_replay_requests"]
