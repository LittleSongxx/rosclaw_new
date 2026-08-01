"""Recoverable rollout assignment service with motion-version pinning."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw.continual.contracts import PolicyVersion, VersionedTrajectory
from rosclaw.continual.serde import policy_version_from_dict
from rosclaw.continual.services.persistence import (
    DurableEventLog,
    require_external_service_root,
)
from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class RolloutState(StrEnum):
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class RolloutAssignment:
    assignment_id: str
    episode_id: str
    scenario_commitment: str
    policy: PolicyVersion
    body_hash: str
    state: RolloutState = RolloutState.ASSIGNED
    worker_id: str | None = None
    trajectory_hash: str | None = None
    strict_replay: bool = False
    abort_reason: str | None = None
    schema_version: str = "rosclaw.continual.rollout_assignment.v1"

    @property
    def version_switch_count(self) -> int:
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assignment_id": self.assignment_id,
            "episode_id": self.episode_id,
            "scenario_commitment": self.scenario_commitment,
            "policy": self.policy.to_dict(),
            "policy_version_hash": self.policy.version_hash,
            "body_hash": self.body_hash,
            "state": self.state.value,
            "worker_id": self.worker_id,
            "trajectory_hash": self.trajectory_hash,
            "strict_replay": self.strict_replay,
            "abort_reason": self.abort_reason,
            "version_switch_count": 0,
        }


class RolloutService:
    """Assign scenarios and fail closed on crash without replaying old actions."""

    def __init__(self, root: Path, *, source_checkout: Path) -> None:
        service_root = require_external_service_root(root, source_checkout) / "rollout"
        self.log = DurableEventLog(service_root, service="rollout")
        self._assignments: dict[str, RolloutAssignment] = {}
        for event in self.log.events:
            self._apply(event.kind, dict(event.payload))
        running = [
            item.assignment_id
            for item in self._assignments.values()
            if item.state is RolloutState.RUNNING
        ]
        for assignment_id in running:
            event = self.log.append(
                "ABORTED",
                {
                    "assignment_id": assignment_id,
                    "reason": "service recovery aborted an in-flight motion; old actions were not replayed",
                    "recovered": True,
                },
            )
            self._apply(event.kind, dict(event.payload))

    @property
    def assignments(self) -> tuple[RolloutAssignment, ...]:
        return tuple(self._assignments[key] for key in sorted(self._assignments))

    @property
    def recovered_abort_count(self) -> int:
        return sum(
            event.kind == "ABORTED" and bool(event.payload.get("recovered"))
            for event in self.log.events
        )

    def assign(
        self,
        *,
        episode_id: str,
        scenario_commitment: str,
        policy: PolicyVersion,
    ) -> RolloutAssignment:
        if not episode_id.strip():
            raise ValueError("rollout episode id must not be empty")
        if not _SHA256.fullmatch(scenario_commitment):
            raise ValueError("rollout scenario commitment must be a sha256 hash")
        assignment_id = canonical_hash(
            {
                "episode_id": episode_id,
                "scenario_commitment": scenario_commitment,
                "policy_version_hash": policy.version_hash,
                "body_hash": policy.body_hash,
            }
        )
        if assignment_id in self._assignments:
            raise ValueError("rollout assignment already exists")
        item = RolloutAssignment(
            assignment_id=assignment_id,
            episode_id=episode_id,
            scenario_commitment=scenario_commitment,
            policy=policy,
            body_hash=policy.body_hash,
        )
        event = self.log.append("ASSIGNED", {"assignment": item.to_dict()})
        self._apply(event.kind, dict(event.payload))
        return self._assignments[assignment_id]

    def start(self, assignment_id: str, *, worker_id: str) -> RolloutAssignment:
        item = self._require(assignment_id, RolloutState.ASSIGNED)
        if not worker_id.strip():
            raise ValueError("rollout worker id must not be empty")
        event = self.log.append(
            "STARTED",
            {
                "assignment_id": item.assignment_id,
                "worker_id": worker_id,
                "policy_version_hash": item.policy.version_hash,
            },
        )
        self._apply(event.kind, dict(event.payload))
        return self._assignments[assignment_id]

    def complete(
        self,
        assignment_id: str,
        *,
        trajectory: VersionedTrajectory,
    ) -> RolloutAssignment:
        item = self._require(assignment_id, RolloutState.RUNNING)
        if trajectory.segments[0].episode_id != item.episode_id:
            raise ValueError("rollout trajectory episode does not match its assignment")
        if trajectory.policy.version_hash != item.policy.version_hash:
            raise ValueError("rollout changed policy version inside a pinned assignment")
        if trajectory.policy.body_hash != item.body_hash:
            raise ValueError("rollout trajectory Body does not match its assignment")
        if trajectory.segments[0].regime_hash != item.scenario_commitment:
            raise ValueError("rollout trajectory scenario does not match its assignment")
        if not trajectory.strict_replay:
            raise ValueError("rollout service accepts only strict-replay trajectories")
        event = self.log.append(
            "COMPLETED",
            {
                "assignment_id": item.assignment_id,
                "trajectory_hash": trajectory.trajectory_hash,
                "strict_replay": True,
                "version_switch_count": 0,
            },
        )
        self._apply(event.kind, dict(event.payload))
        return self._assignments[assignment_id]

    def abort(self, assignment_id: str, *, reason: str) -> RolloutAssignment:
        item = self._assignments.get(assignment_id)
        if item is None or item.state not in {RolloutState.ASSIGNED, RolloutState.RUNNING}:
            raise RuntimeError("only assigned or running rollout work may be aborted")
        if not reason.strip():
            raise ValueError("rollout abort reason must not be empty")
        event = self.log.append(
            "ABORTED",
            {"assignment_id": assignment_id, "reason": reason, "recovered": False},
        )
        self._apply(event.kind, dict(event.payload))
        return self._assignments[assignment_id]

    def _require(self, assignment_id: str, state: RolloutState) -> RolloutAssignment:
        item = self._assignments.get(assignment_id)
        if item is None or item.state is not state:
            raise RuntimeError(f"rollout assignment must be in {state.value}")
        return item

    def _apply(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "ASSIGNED":
            raw = dict(payload["assignment"])
            policy = policy_version_from_dict(dict(raw["policy"]))
            if raw.get("policy_version_hash") != policy.version_hash:
                raise ValueError("rollout assignment policy hash mismatch")
            item = RolloutAssignment(
                assignment_id=str(raw["assignment_id"]),
                episode_id=str(raw["episode_id"]),
                scenario_commitment=str(raw["scenario_commitment"]),
                policy=policy,
                body_hash=str(raw["body_hash"]),
            )
            if item.body_hash != policy.body_hash:
                raise ValueError("rollout assignment Body hash mismatch")
            if not _SHA256.fullmatch(item.scenario_commitment):
                raise ValueError("rollout scenario commitment hash mismatch")
            expected_id = canonical_hash(
                {
                    "episode_id": item.episode_id,
                    "scenario_commitment": item.scenario_commitment,
                    "policy_version_hash": policy.version_hash,
                    "body_hash": item.body_hash,
                }
            )
            if item.assignment_id != expected_id:
                raise ValueError("rollout assignment identity hash mismatch")
            if item.assignment_id in self._assignments:
                raise ValueError("rollout assignment appears more than once")
            self._assignments[item.assignment_id] = item
            return
        assignment_id = str(payload["assignment_id"])
        item = self._assignments[assignment_id]
        if kind == "STARTED":
            if item.state is not RolloutState.ASSIGNED:
                raise ValueError("rollout start event does not follow assignment")
            if payload.get("policy_version_hash") != item.policy.version_hash:
                raise ValueError("rollout start policy hash mismatch")
            self._assignments[assignment_id] = replace(
                item,
                state=RolloutState.RUNNING,
                worker_id=str(payload["worker_id"]),
            )
        elif kind == "COMPLETED":
            if item.state is not RolloutState.RUNNING:
                raise ValueError("rollout completion does not follow a running motion")
            if not payload.get("strict_replay") or payload.get("version_switch_count") != 0:
                raise ValueError("completed rollout violated strict version replay")
            self._assignments[assignment_id] = replace(
                item,
                state=RolloutState.COMPLETE,
                trajectory_hash=str(payload["trajectory_hash"]),
                strict_replay=bool(payload["strict_replay"]),
            )
        elif kind == "ABORTED":
            if item.state not in {RolloutState.ASSIGNED, RolloutState.RUNNING}:
                raise ValueError("rollout abort does not follow active work")
            self._assignments[assignment_id] = replace(
                item,
                state=RolloutState.ABORTED,
                abort_reason=str(payload["reason"]),
            )
        else:
            raise ValueError(f"unsupported rollout service event: {kind}")


__all__ = ["RolloutAssignment", "RolloutService", "RolloutState"]
