"""Recoverable learner jobs with durable optimizer checkpoints and idempotency."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.continual.contracts import PolicyVersion
from rosclaw.continual.experience import ExperienceBatch
from rosclaw.continual.learner import ConstrainedResidualSAC
from rosclaw.continual.serde import policy_version_from_dict
from rosclaw.continual.services.persistence import (
    DurableEventLog,
    fsync_directory,
    require_external_service_root,
)
from rosclaw.feedback.contracts import canonical_hash


@dataclass(frozen=True)
class LearnerProduct:
    candidate: PolicyVersion
    artifact: bytes
    checkpoint: bytes
    metrics: Mapping[str, float | int | bool]

    def __post_init__(self) -> None:
        if not self.artifact or not self.checkpoint:
            raise ValueError("learner product requires artifact and full checkpoint bytes")
        _validate_metrics(self.metrics)

    @property
    def checkpoint_hash(self) -> str:
        return "sha256:" + hashlib.sha256(self.checkpoint).hexdigest()


@dataclass(frozen=True)
class LearnerServiceReceipt:
    job_id: str
    batch_hash: str
    parent_policy_hash: str
    candidate_policy_hash: str
    artifact_hash: str
    checkpoint_hash: str
    metrics: Mapping[str, float | int | bool]
    event_hash: str
    registry_write_count: int = 0
    dds_opened: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.continual.learner_service_receipt.v1"


class ResidualSACServiceExecutor:
    """Adapter that emits both inference artifact and full SAC checkpoint."""

    def __init__(
        self,
        learner: ConstrainedResidualSAC,
        *,
        parent: PolicyVersion,
        updates_per_job: int = 1,
    ) -> None:
        if not 1 <= updates_per_job <= 1000:
            raise ValueError("updates_per_job must be in [1, 1000]")
        self.learner = learner
        self.parent = parent
        self.updates_per_job = updates_per_job

    def __call__(self, batch: ExperienceBatch) -> LearnerProduct:
        if batch.learner_version != self.parent.version:
            raise ValueError("learner batch version does not match executor parent")
        updates = [self.learner.update(batch) for _ in range(self.updates_per_job)]
        candidate, artifact = self.learner.candidate_policy(parent=self.parent)
        last = updates[-1]
        metrics: dict[str, float | int | bool] = {
            **{key: value for key, value in asdict(last).items() if key != "schema_version"},
            "job_update_count": len(updates),
        }
        return LearnerProduct(
            candidate=candidate,
            artifact=artifact,
            checkpoint=self.learner.checkpoint_bytes(),
            metrics=metrics,
        )


class LearnerService:
    """Consumes versioned batches but has no inference or hardware authority."""

    def __init__(
        self,
        root: Path,
        *,
        source_checkout: Path,
        parent: PolicyVersion | None = None,
    ) -> None:
        self.root = require_external_service_root(root, source_checkout) / "learner"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.log = DurableEventLog(self.root / "journal", service="learner")
        self._parent: PolicyVersion | None = None
        self._completed: dict[str, LearnerServiceReceipt] = {}
        self._pending: dict[str, str] = {}
        self._quarantined: set[str] = set()
        for event in self.log.events:
            self._apply(event.kind, dict(event.payload), event.event_hash)
        if not self.log.events:
            if parent is None:
                raise ValueError("new learner service requires a parent policy")
            event = self.log.append("INITIALIZED", {"parent": parent.to_dict()})
            self._apply(event.kind, dict(event.payload), event.event_hash)
        elif parent is not None and parent.version_hash != self.parent.version_hash:
            raise ValueError("supplied learner parent does not match recovered state")
        for job_id, batch_hash in tuple(self._pending.items()):
            event = self.log.append(
                "JOB_QUARANTINED",
                {
                    "job_id": job_id,
                    "batch_hash": batch_hash,
                    "reason": "service recovery found an incomplete optimizer update",
                    "recovered": True,
                },
            )
            self._apply(event.kind, dict(event.payload), event.event_hash)

    @property
    def parent(self) -> PolicyVersion:
        if self._parent is None:
            raise RuntimeError("learner service has no parent policy")
        return self._parent

    @property
    def completed_batch_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(self._completed))

    @property
    def quarantined_batch_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(self._quarantined))

    def execute(
        self,
        batch: ExperienceBatch,
        *,
        executor: Callable[[ExperienceBatch], LearnerProduct],
    ) -> LearnerServiceReceipt:
        if batch.batch_hash in self._completed:
            return self._completed[batch.batch_hash]
        if batch.batch_hash in self._quarantined:
            raise RuntimeError("batch is quarantined after an uncertain optimizer update")
        if batch.learner_version != self.parent.version:
            raise ValueError("experience batch learner version does not match service parent")
        job_id = canonical_hash(
            {
                "batch_hash": batch.batch_hash,
                "parent_policy_hash": self.parent.version_hash,
                "next_event_sequence": len(self.log.events) + 1,
            }
        )
        started = self.log.append(
            "JOB_STARTED",
            {
                "job_id": job_id,
                "batch_hash": batch.batch_hash,
                "parent_policy_hash": self.parent.version_hash,
            },
        )
        self._apply(started.kind, dict(started.payload), started.event_hash)
        try:
            product = executor(batch)
            self._validate_product(product)
            self._store_blob(product.candidate.artifact_hash, product.artifact, suffix="artifact")
            self._store_blob(product.checkpoint_hash, product.checkpoint, suffix="checkpoint")
        except BaseException as exc:
            aborted = self.log.append(
                "JOB_QUARANTINED",
                {
                    "job_id": job_id,
                    "batch_hash": batch.batch_hash,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "recovered": False,
                },
            )
            self._apply(aborted.kind, dict(aborted.payload), aborted.event_hash)
            raise
        completed = self.log.append(
            "JOB_COMPLETED",
            {
                "job_id": job_id,
                "batch_hash": batch.batch_hash,
                "parent_policy_hash": self.parent.version_hash,
                "candidate": product.candidate.to_dict(),
                "artifact_hash": product.candidate.artifact_hash,
                "checkpoint_hash": product.checkpoint_hash,
                "metrics": dict(product.metrics),
            },
        )
        self._apply(completed.kind, dict(completed.payload), completed.event_hash)
        return self._completed[batch.batch_hash]

    def checkpoint_bytes(self, checkpoint_hash: str) -> bytes:
        return self._read_blob(checkpoint_hash, suffix="checkpoint")

    def advance_parent(self, policy: PolicyVersion) -> None:
        known = {receipt.candidate_policy_hash for receipt in self._completed.values()}
        if policy.version_hash not in known:
            raise ValueError("new learner parent is not a completed candidate")
        event = self.log.append("PARENT_ADVANCED", {"parent": policy.to_dict()})
        self._apply(event.kind, dict(event.payload), event.event_hash)

    def _validate_product(self, product: LearnerProduct) -> None:
        candidate = product.candidate
        if candidate.version != self.parent.version + 1:
            raise ValueError("learner candidate version does not follow parent")
        if candidate.parent_version_hash != self.parent.version_hash:
            raise ValueError("learner candidate parent hash mismatch")
        for name in (
            "body_hash",
            "safety_kernel_hash",
            "controller_snapshot_hash",
            "observation_names",
            "residual_action_names",
        ):
            if getattr(candidate, name) != getattr(self.parent, name):
                raise ValueError(f"learner candidate changes immutable identity: {name}")
        if "sha256:" + hashlib.sha256(product.artifact).hexdigest() != candidate.artifact_hash:
            raise ValueError("learner candidate artifact checksum mismatch")

    def _blob_path(self, content_hash: str, *, suffix: str) -> Path:
        return self.blobs / f"{content_hash.removeprefix('sha256:')}.{suffix}.bin"

    def _store_blob(self, content_hash: str, payload: bytes, *, suffix: str) -> None:
        if "sha256:" + hashlib.sha256(payload).hexdigest() != content_hash:
            raise ValueError(f"learner {suffix} content hash mismatch")
        destination = self._blob_path(content_hash, suffix=suffix)
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ValueError(f"learner {suffix} content-address collision")
            return
        descriptor, temporary = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".tmp", dir=self.blobs
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
            Path(temporary).unlink()
            fsync_directory(self.blobs)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def _read_blob(self, content_hash: str, *, suffix: str) -> bytes:
        payload = self._blob_path(content_hash, suffix=suffix).read_bytes()
        if "sha256:" + hashlib.sha256(payload).hexdigest() != content_hash:
            raise ValueError(f"learner {suffix} blob checksum mismatch")
        return payload

    def _apply(self, kind: str, payload: dict[str, Any], event_hash: str) -> None:
        if kind == "INITIALIZED":
            if self._parent is not None:
                raise ValueError("learner service may be initialized only once")
            self._parent = policy_version_from_dict(dict(payload["parent"]))
        elif kind == "PARENT_ADVANCED":
            parent = policy_version_from_dict(dict(payload["parent"]))
            known = {receipt.candidate_policy_hash for receipt in self._completed.values()}
            if parent.version_hash not in known:
                raise ValueError("learner parent advance does not reference a completed candidate")
            self._parent = parent
        elif kind == "JOB_STARTED":
            batch_hash = str(payload["batch_hash"])
            if payload["parent_policy_hash"] != self.parent.version_hash:
                raise ValueError("learner job parent hash mismatch")
            job_id = str(payload["job_id"])
            if job_id in self._pending or batch_hash in self._completed:
                raise ValueError("learner job is duplicated in its durable log")
            self._pending[job_id] = batch_hash
        elif kind == "JOB_COMPLETED":
            job_id = str(payload["job_id"])
            batch_hash = str(payload["batch_hash"])
            candidate = policy_version_from_dict(dict(payload["candidate"]))
            artifact_hash = str(payload["artifact_hash"])
            checkpoint_hash = str(payload["checkpoint_hash"])
            if self._pending.get(job_id) != batch_hash:
                raise ValueError("learner completion does not match a pending job")
            if (
                payload["parent_policy_hash"] != self.parent.version_hash
                or candidate.parent_version_hash != self.parent.version_hash
                or candidate.version != self.parent.version + 1
            ):
                raise ValueError("learner completion candidate lineage mismatch")
            for name in (
                "body_hash",
                "safety_kernel_hash",
                "controller_snapshot_hash",
                "observation_names",
                "residual_action_names",
            ):
                if getattr(candidate, name) != getattr(self.parent, name):
                    raise ValueError(f"learner completion changes immutable identity: {name}")
            if candidate.artifact_hash != artifact_hash:
                raise ValueError("learner completion artifact identity mismatch")
            self._read_blob(artifact_hash, suffix="artifact")
            self._read_blob(checkpoint_hash, suffix="checkpoint")
            metrics = dict(payload["metrics"])
            _validate_metrics(metrics)
            receipt = LearnerServiceReceipt(
                job_id=job_id,
                batch_hash=batch_hash,
                parent_policy_hash=str(payload["parent_policy_hash"]),
                candidate_policy_hash=candidate.version_hash,
                artifact_hash=artifact_hash,
                checkpoint_hash=checkpoint_hash,
                metrics=metrics,
                event_hash=event_hash,
            )
            self._completed[batch_hash] = receipt
            self._pending.pop(job_id, None)
        elif kind == "JOB_QUARANTINED":
            job_id = str(payload["job_id"])
            batch_hash = str(payload["batch_hash"])
            if self._pending.get(job_id) != batch_hash:
                raise ValueError("learner quarantine does not match a pending job")
            self._pending.pop(job_id, None)
            self._quarantined.add(batch_hash)
        else:
            raise ValueError(f"unsupported learner service event: {kind}")


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    for name, value in metrics.items():
        if isinstance(value, bool):
            continue
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            raise ValueError(f"learner metric must be finite numeric value: {name}")


__all__ = [
    "LearnerProduct",
    "LearnerService",
    "LearnerServiceReceipt",
    "ResidualSACServiceExecutor",
]
