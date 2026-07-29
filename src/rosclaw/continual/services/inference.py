"""Recoverable active/candidate/rollback slots and motion-scoped version leases."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.continual.contracts import PolicyVersion, SkillPhase
from rosclaw.continual.serde import policy_version_from_dict
from rosclaw.continual.services.persistence import (
    DurableEventLog,
    fsync_directory,
    require_external_service_root,
)
from rosclaw.continual.stability import ContinualGateReport
from rosclaw.continual.weight_update import ResidualWeightSlot, WeightSlotState
from rosclaw.feedback.contracts import canonical_hash


@dataclass(frozen=True)
class MotionVersionLease:
    lease_id: str
    episode_id: str
    policy_version: int
    policy_version_hash: str
    artifact_hash: str
    body_hash: str
    phase_at_start: SkillPhase
    schema_version: str = "rosclaw.continual.motion_version_lease.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lease_id": self.lease_id,
            "episode_id": self.episode_id,
            "policy_version": self.policy_version,
            "policy_version_hash": self.policy_version_hash,
            "artifact_hash": self.artifact_hash,
            "body_hash": self.body_hash,
            "phase_at_start": self.phase_at_start.value,
        }


@dataclass(frozen=True)
class InferenceSlotReceipt:
    active_version_hash: str
    candidate_version_hash: str | None
    rollback_version_hash: str | None
    published_version_hash: str | None
    active_motion_count: int
    frozen: bool
    reason: str
    event_hash: str
    recovered_abort_count: int
    hardware_authorized: bool = False
    registry_write_count: int = 0
    dds_opened: bool = False
    schema_version: str = "rosclaw.continual.inference_slot_receipt.v1"


class InferenceService:
    """Inference-plane slot owner; weight mutation is exposed only via its service."""

    def __init__(
        self,
        root: Path,
        *,
        source_checkout: Path,
        active: PolicyVersion | None = None,
        active_artifact: bytes | None = None,
    ) -> None:
        self.root = require_external_service_root(root, source_checkout) / "inference"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.log = DurableEventLog(self.root / "journal", service="inference")
        self._active: PolicyVersion | None = None
        self._candidate: PolicyVersion | None = None
        self._rollback: PolicyVersion | None = None
        self._published: PolicyVersion | None = None
        self._published_verified = False
        self._frozen = False
        self._motions: dict[str, MotionVersionLease] = {}
        for event in self.log.events:
            self._apply(event.kind, dict(event.payload))
        if not self.log.events:
            if active is None or active_artifact is None:
                raise ValueError("new inference service requires active policy and artifact")
            self._store_blob(active, active_artifact)
            event = self.log.append(
                "INITIALIZED",
                {"active": active.to_dict(), "artifact_hash": active.artifact_hash},
            )
            self._apply(event.kind, dict(event.payload))
        elif active is not None and active.version_hash != self.active.version_hash:
            raise ValueError("supplied active policy does not match recovered inference state")
        for policy in (self._active, self._candidate, self._rollback, self._published):
            if policy is not None:
                self._verify_blob(policy)
        inflight = tuple(self._motions)
        for lease_id in inflight:
            event = self.log.append(
                "MOTION_ABORTED",
                {
                    "lease_id": lease_id,
                    "reason": "service recovery aborted the in-flight motion; old actions were not replayed",
                    "recovered": True,
                },
            )
            self._apply(event.kind, dict(event.payload))

    @property
    def active(self) -> PolicyVersion:
        if self._active is None:
            raise RuntimeError("inference service has no active policy")
        return self._active

    @property
    def candidate(self) -> PolicyVersion | None:
        return self._candidate

    @property
    def rollback(self) -> PolicyVersion | None:
        return self._rollback

    @property
    def published(self) -> PolicyVersion | None:
        return self._published

    @property
    def active_motion_count(self) -> int:
        return len(self._motions)

    @property
    def recovered_abort_count(self) -> int:
        return sum(
            event.kind == "MOTION_ABORTED" and bool(event.payload.get("recovered"))
            for event in self.log.events
        )

    def begin_motion(self, *, episode_id: str, phase: SkillPhase) -> MotionVersionLease:
        if self._frozen:
            raise RuntimeError("inference service is frozen")
        if not episode_id.strip():
            raise ValueError("motion episode id must not be empty")
        if any(lease.episode_id == episode_id for lease in self._motions.values()):
            raise ValueError("motion episode already has a version lease")
        policy = self.active
        lease_id = canonical_hash(
            {
                "episode_id": episode_id,
                "policy_version_hash": policy.version_hash,
                "next_event_sequence": len(self.log.events) + 1,
            }
        )
        lease = MotionVersionLease(
            lease_id=lease_id,
            episode_id=episode_id,
            policy_version=policy.version,
            policy_version_hash=policy.version_hash,
            artifact_hash=policy.artifact_hash,
            body_hash=policy.body_hash,
            phase_at_start=phase,
        )
        event = self.log.append("MOTION_BEGAN", {"lease": lease.to_dict()})
        self._apply(event.kind, dict(event.payload))
        return self._motions[lease_id]

    def end_motion(self, lease_id: str, *, aborted: bool = False, reason: str = "") -> None:
        if lease_id not in self._motions:
            raise KeyError("unknown or completed motion lease")
        if aborted and not reason.strip():
            raise ValueError("aborted motion requires a reason")
        event = self.log.append(
            "MOTION_ABORTED" if aborted else "MOTION_ENDED",
            {
                "lease_id": lease_id,
                "reason": reason if aborted else "motion reached COMPLETE",
                "recovered": False,
            },
        )
        self._apply(event.kind, dict(event.payload))

    def receipt(self, reason: str) -> InferenceSlotReceipt:
        return InferenceSlotReceipt(
            active_version_hash=self.active.version_hash,
            candidate_version_hash=(self._candidate.version_hash if self._candidate else None),
            rollback_version_hash=(self._rollback.version_hash if self._rollback else None),
            published_version_hash=(self._published.version_hash if self._published else None),
            active_motion_count=len(self._motions),
            frozen=self._frozen,
            reason=reason,
            event_hash=self.log.last_hash,
            recovered_abort_count=self.recovered_abort_count,
        )

    def _publish(self, policy: PolicyVersion, artifact: bytes) -> InferenceSlotReceipt:
        if self._frozen:
            raise RuntimeError("inference service is frozen")
        if self._published is not None or self._candidate is not None:
            raise RuntimeError("a published or staged candidate already exists")
        self._validate_successor(policy)
        self._store_blob(policy, artifact)
        event = self.log.append(
            "PUBLISHED",
            {"policy": policy.to_dict(), "artifact_hash": policy.artifact_hash},
        )
        self._apply(event.kind, dict(event.payload))
        return self.receipt("candidate artifact published outside the active slot")

    def _verify_published(self) -> InferenceSlotReceipt:
        if self._published is None:
            raise RuntimeError("no published candidate exists")
        self._validate_successor(self._published)
        self._verify_blob(self._published)
        event = self.log.append(
            "VERIFIED",
            {"policy_version_hash": self._published.version_hash},
        )
        self._apply(event.kind, dict(event.payload))
        return self.receipt("published candidate checksum and lineage verified")

    def _stage(self) -> InferenceSlotReceipt:
        if self._published is None or not self._published_verified:
            raise RuntimeError("candidate must be published and verified before staging")
        event = self.log.append(
            "STAGED",
            {"policy_version_hash": self._published.version_hash},
        )
        self._apply(event.kind, dict(event.payload))
        return self.receipt("verified candidate staged; active policy unchanged")

    def _activate(
        self,
        *,
        phase: SkillPhase,
        gate_report: ContinualGateReport,
    ) -> InferenceSlotReceipt:
        if self._candidate is None:
            raise RuntimeError("no staged candidate exists")
        if self._motions:
            self._freeze("activation requested while a motion version lease was active")
            return self.receipt("activation blocked by active motion lease")
        slot = ResidualWeightSlot(self.active, active_artifact=self._read_blob(self.active))
        slot.stage(self._candidate, artifact=self._read_blob(self._candidate))
        result = slot.activate(phase=phase, gate_report=gate_report)
        if result.state is not WeightSlotState.ACTIVE:
            self._freeze(result.reason)
            return self.receipt(result.reason)
        old = self.active
        new = self._candidate
        event = self.log.append(
            "ACTIVATED",
            {
                "active": new.to_dict(),
                "rollback": old.to_dict(),
                "reason": result.reason,
                "hardware_authorized": False,
            },
        )
        self._apply(event.kind, dict(event.payload))
        return self.receipt(result.reason)

    def _freeze(self, reason: str) -> InferenceSlotReceipt:
        if not reason.strip():
            raise ValueError("freeze reason must not be empty")
        event = self.log.append("FROZEN", {"reason": reason})
        self._apply(event.kind, dict(event.payload))
        return self.receipt(reason)

    def _rollback_active(self, reason: str) -> InferenceSlotReceipt:
        if self._motions:
            raise RuntimeError("rollback is forbidden while a motion lease is active")
        if self._rollback is None:
            raise RuntimeError("no rollback policy exists")
        if not reason.strip():
            raise ValueError("rollback reason must not be empty")
        failed = self.active
        target = self._rollback
        event = self.log.append(
            "ROLLED_BACK",
            {"active": target.to_dict(), "rollback": failed.to_dict(), "reason": reason},
        )
        self._apply(event.kind, dict(event.payload))
        return self.receipt(reason)

    def _validate_successor(self, policy: PolicyVersion) -> None:
        active = self.active
        if (
            policy.version != active.version + 1
            or policy.parent_version_hash != active.version_hash
        ):
            raise ValueError("candidate is not the direct successor of active policy")
        for name in (
            "body_hash",
            "safety_kernel_hash",
            "controller_snapshot_hash",
            "observation_names",
            "residual_action_names",
        ):
            if getattr(policy, name) != getattr(active, name):
                raise ValueError(f"candidate changes immutable inference identity: {name}")

    def _blob_path(self, policy: PolicyVersion) -> Path:
        return self.blobs / (policy.artifact_hash.removeprefix("sha256:") + ".bin")

    def _store_blob(self, policy: PolicyVersion, artifact: bytes) -> None:
        _verify_artifact(policy, artifact)
        destination = self._blob_path(policy)
        if destination.exists():
            if destination.read_bytes() != artifact:
                raise ValueError("content-addressed policy blob collision")
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=self.blobs
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(artifact)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            Path(temporary).unlink()
            fsync_directory(self.blobs)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _read_blob(self, policy: PolicyVersion) -> bytes:
        value = self._blob_path(policy).read_bytes()
        _verify_artifact(policy, value)
        return value

    def _verify_blob(self, policy: PolicyVersion) -> None:
        self._read_blob(policy)

    def _apply(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "INITIALIZED":
            if self._active is not None:
                raise ValueError("inference service may be initialized only once")
            active = policy_version_from_dict(dict(payload["active"]))
            if payload.get("artifact_hash") != active.artifact_hash:
                raise ValueError("initialized inference artifact identity mismatch")
            self._active = active
        elif kind == "PUBLISHED":
            if self._published is not None or self._candidate is not None:
                raise ValueError("inference publish event conflicts with an existing candidate")
            published = policy_version_from_dict(dict(payload["policy"]))
            if payload.get("artifact_hash") != published.artifact_hash:
                raise ValueError("published inference artifact identity mismatch")
            self._validate_successor(published)
            self._published = published
            self._published_verified = False
        elif kind == "VERIFIED":
            if (
                self._published is None
                or payload["policy_version_hash"] != self._published.version_hash
            ):
                raise ValueError("verified inference policy does not match published candidate")
            self._published_verified = True
        elif kind == "STAGED":
            if (
                self._published is None
                or payload["policy_version_hash"] != self._published.version_hash
            ):
                raise ValueError("staged inference policy does not match published candidate")
            self._candidate = self._published
            self._published = None
            self._published_verified = False
        elif kind == "ACTIVATED":
            active = policy_version_from_dict(dict(payload["active"]))
            rollback = policy_version_from_dict(dict(payload["rollback"]))
            if (
                self._candidate is None
                or active.version_hash != self._candidate.version_hash
                or rollback.version_hash != self.active.version_hash
            ):
                raise ValueError("inference activation event violates candidate lineage")
            self._active = active
            self._rollback = rollback
            self._candidate = None
            self._frozen = False
        elif kind == "FROZEN":
            self._frozen = True
        elif kind == "ROLLED_BACK":
            active = policy_version_from_dict(dict(payload["active"]))
            rollback = policy_version_from_dict(dict(payload["rollback"]))
            if self._rollback is None or active.version_hash != self._rollback.version_hash:
                raise ValueError("inference rollback event does not target its rollback slot")
            if rollback.version_hash != self.active.version_hash:
                raise ValueError("inference rollback event does not preserve failed policy")
            self._active = active
            self._rollback = rollback
            self._candidate = None
            self._published = None
            self._published_verified = False
            self._frozen = False
        elif kind == "MOTION_BEGAN":
            raw = dict(payload["lease"])
            lease = MotionVersionLease(
                lease_id=str(raw["lease_id"]),
                episode_id=str(raw["episode_id"]),
                policy_version=int(raw["policy_version"]),
                policy_version_hash=str(raw["policy_version_hash"]),
                artifact_hash=str(raw["artifact_hash"]),
                body_hash=str(raw["body_hash"]),
                phase_at_start=SkillPhase(str(raw["phase_at_start"])),
            )
            if lease.policy_version_hash != self.active.version_hash:
                raise ValueError("motion lease does not pin the active inference policy")
            if (
                lease.policy_version != self.active.version
                or lease.artifact_hash != self.active.artifact_hash
                or lease.body_hash != self.active.body_hash
                or any(item.episode_id == lease.episode_id for item in self._motions.values())
            ):
                raise ValueError("motion lease identity does not match active inference")
            self._motions[lease.lease_id] = lease
        elif kind in {"MOTION_ENDED", "MOTION_ABORTED"}:
            lease_id = str(payload["lease_id"])
            if lease_id not in self._motions:
                raise ValueError("motion completion references an unknown active lease")
            self._motions.pop(lease_id)
        else:
            raise ValueError(f"unsupported inference service event: {kind}")


def _verify_artifact(policy: PolicyVersion, artifact: bytes) -> None:
    if not artifact:
        raise ValueError("policy artifact must not be empty")
    actual = "sha256:" + hashlib.sha256(artifact).hexdigest()
    if actual != policy.artifact_hash:
        raise ValueError("policy artifact checksum mismatch")


__all__ = ["InferenceService", "InferenceSlotReceipt", "MotionVersionLease"]
