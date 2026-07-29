"""Persistent four-partition experience service with a SQLite catalog."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rosclaw.continual.boundary_feedback import BoundaryReplayRequest
from rosclaw.continual.contracts import ExperiencePartition
from rosclaw.continual.experience import (
    ContinualExperienceStore,
    ExperienceBatch,
    ExperienceBufferConfig,
    ExperienceRecord,
)
from rosclaw.continual.serde import experience_record_from_dict, experience_record_to_dict
from rosclaw.continual.services.persistence import (
    DurableEventLog,
    require_external_service_root,
)


class ExperienceService:
    """Append-only truth plus recoverable catalog and bounded replay cache."""

    def __init__(
        self,
        root: Path,
        *,
        source_checkout: Path,
        config: ExperienceBufferConfig | None = None,
    ) -> None:
        self.root = require_external_service_root(root, source_checkout) / "experience"
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = DurableEventLog(self.root / "journal", service="experience")
        self.store = ContinualExperienceStore(config)
        self._records: dict[str, ExperienceRecord] = {}
        self._boundary: dict[str, tuple[BoundaryReplayRequest, str | None]] = {}
        self._database = sqlite3.connect(self.root / "catalog.sqlite3")
        self._database.execute("PRAGMA journal_mode=WAL")
        self._database.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        for event in self.log.events:
            self._apply(event.kind, dict(event.payload))
        self._audit_catalog()

    def close(self) -> None:
        self._database.close()

    def __enter__(self) -> ExperienceService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def append(self, record: ExperienceRecord) -> str:
        if not record.trajectory.strict_replay:
            raise ValueError("persistent experience requires strict-replay trajectory truth")
        if record.record_hash in self._records:
            raise ValueError("experience record is already durably cataloged")
        event = self.log.append(
            "EXPERIENCE_APPENDED",
            {"envelope": experience_record_to_dict(record)},
        )
        self._apply(event.kind, dict(event.payload))
        return record.record_hash

    def sample(self, *, batch_size: int, learner_version: int, seed: int) -> ExperienceBatch:
        return self.store.sample(
            batch_size=batch_size,
            learner_version=learner_version,
            seed=seed,
        )

    def enqueue_boundary(self, request: BoundaryReplayRequest) -> str:
        if request.request_hash in self._boundary:
            raise ValueError("boundary replay request is already cataloged")
        event = self.log.append("BOUNDARY_QUEUED", {"request": request.to_dict()})
        self._apply(event.kind, dict(event.payload))
        return request.request_hash

    def complete_boundary(self, request_hash: str, *, record: ExperienceRecord) -> None:
        entry = self._boundary.get(request_hash)
        if entry is None:
            raise KeyError("unknown boundary replay request")
        if entry[1] is not None:
            raise RuntimeError("boundary replay request is already complete")
        if record.partition is not ExperiencePartition.BOUNDARY:
            raise ValueError("boundary completion requires a Boundary experience record")
        if not record.trajectory.strict_replay:
            raise ValueError("boundary completion requires strict replay")
        if record.record_hash not in self._records:
            self.append(record)
        event = self.log.append(
            "BOUNDARY_COMPLETED",
            {"request_hash": request_hash, "record_hash": record.record_hash},
        )
        self._apply(event.kind, dict(event.payload))

    @property
    def pending_boundary_requests(self) -> tuple[BoundaryReplayRequest, ...]:
        return tuple(request for request, completed in self._boundary.values() if completed is None)

    def catalog_counts(self) -> Mapping[ExperiencePartition, int]:
        rows = self._database.execute(
            "SELECT partition, COUNT(*) FROM experience_records GROUP BY partition"
        ).fetchall()
        values = dict.fromkeys(ExperiencePartition, 0)
        for name, count in rows:
            values[ExperiencePartition(str(name))] = int(count)
        return values

    def audit_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.continual.experience_service_receipt.v1",
            "event_count": len(self.log.events),
            "last_event_hash": self.log.last_hash,
            "catalog_counts": {
                partition.value: count for partition, count in self.catalog_counts().items()
            },
            "cache_counts": {
                partition.value: count for partition, count in self.store.counts().items()
            },
            "pending_boundary_count": len(self.pending_boundary_requests),
            "registry_write_count": 0,
            "dds_opened": False,
            "hardware_authorized": False,
        }

    def _create_schema(self) -> None:
        self._database.executescript(
            """
            CREATE TABLE IF NOT EXISTS experience_records (
                record_hash TEXT PRIMARY KEY,
                trajectory_hash TEXT NOT NULL,
                partition TEXT NOT NULL,
                policy_version INTEGER NOT NULL,
                event_sequence INTEGER NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS boundary_requests (
                request_hash TEXT PRIMARY KEY,
                scenario_commitment TEXT NOT NULL,
                replay_partition TEXT NOT NULL,
                completed_record_hash TEXT,
                event_sequence INTEGER NOT NULL UNIQUE
            );
            """
        )
        self._database.commit()

    def _apply(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "EXPERIENCE_APPENDED":
            record = experience_record_from_dict(_mapping(payload, "envelope"))
            if record.record_hash in self._records:
                raise ValueError("append-only experience log contains a duplicate record")
            event_sequence = _event_sequence_for_payload(self.log, payload)
            with self._database:
                self._database.execute(
                    "INSERT OR IGNORE INTO experience_records VALUES (?, ?, ?, ?, ?)",
                    (
                        record.record_hash,
                        record.trajectory.trajectory_hash,
                        record.partition.value,
                        record.trajectory.policy.version,
                        event_sequence,
                    ),
                )
            self.store.append(record)
            self._records[record.record_hash] = record
            return
        if kind == "BOUNDARY_QUEUED":
            request = _boundary_request(_mapping(payload, "request"))
            if request.request_hash in self._boundary:
                raise ValueError("append-only log contains a duplicate boundary request")
            event_sequence = _event_sequence_for_payload(self.log, payload)
            with self._database:
                self._database.execute(
                    "INSERT OR IGNORE INTO boundary_requests VALUES (?, ?, ?, NULL, ?)",
                    (
                        request.request_hash,
                        request.scenario_commitment,
                        request.replay_partition,
                        event_sequence,
                    ),
                )
            self._boundary[request.request_hash] = (request, None)
            return
        if kind == "BOUNDARY_COMPLETED":
            request_hash = str(payload["request_hash"])
            record_hash = str(payload["record_hash"])
            if record_hash not in self._records:
                raise ValueError("boundary completion references missing durable experience")
            request, completed = self._boundary[request_hash]
            if completed is not None:
                raise ValueError("append-only log completes a boundary request more than once")
            with self._database:
                self._database.execute(
                    "UPDATE boundary_requests SET completed_record_hash=? WHERE request_hash=?",
                    (record_hash, request_hash),
                )
            self._boundary[request_hash] = (request, record_hash)
            return
        raise ValueError(f"unsupported experience service event: {kind}")

    def _audit_catalog(self) -> None:
        records = {
            str(row[0]): (str(row[1]), str(row[2]), int(row[3]), int(row[4]))
            for row in self._database.execute(
                "SELECT record_hash, trajectory_hash, partition, policy_version, "
                "event_sequence FROM experience_records"
            )
        }
        expected_records = {
            record.record_hash: (
                record.trajectory.trajectory_hash,
                record.partition.value,
                record.trajectory.policy.version,
                _record_event_sequence(self.log, record.record_hash),
            )
            for record in self._records.values()
        }
        if records != expected_records:
            raise ValueError("experience SQLite catalog diverges from append-only truth")
        requests = {
            str(row[0]): (str(row[1]), str(row[2]), row[3], int(row[4]))
            for row in self._database.execute(
                "SELECT request_hash, scenario_commitment, replay_partition, "
                "completed_record_hash, event_sequence FROM boundary_requests"
            )
        }
        expected_requests = {
            request.request_hash: (
                request.scenario_commitment,
                request.replay_partition,
                completed,
                _boundary_event_sequence(self.log, request.request_hash),
            )
            for request, completed in self._boundary.values()
        }
        if requests != expected_requests:
            raise ValueError("boundary SQLite catalog diverges from append-only truth")


def _event_sequence_for_payload(log: DurableEventLog, payload: dict[str, Any]) -> int:
    for event in reversed(log.events):
        if dict(event.payload) == payload:
            return event.sequence
    raise RuntimeError("service event payload is not present in its durable log")


def _record_event_sequence(log: DurableEventLog, record_hash: str) -> int:
    for event in log.events:
        envelope = event.payload.get("envelope")
        if (
            event.kind == "EXPERIENCE_APPENDED"
            and isinstance(envelope, Mapping)
            and isinstance(envelope.get("record"), Mapping)
            and envelope["record"].get("record_hash") == record_hash
        ):
            return event.sequence
    raise RuntimeError("experience record has no durable append event")


def _boundary_event_sequence(log: DurableEventLog, request_hash: str) -> int:
    for event in log.events:
        request = event.payload.get("request")
        if (
            event.kind == "BOUNDARY_QUEUED"
            and isinstance(request, Mapping)
            and request.get("request_hash") == request_hash
        ):
            return event.sequence
    raise RuntimeError("boundary request has no durable queue event")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return item


def _boundary_request(value: Mapping[str, Any]) -> BoundaryReplayRequest:
    request = BoundaryReplayRequest(
        scenario_id=str(value["scenario_id"]),
        scenario_commitment=str(value["scenario_commitment"]),
        replay_partition=str(value["replay_partition"]),
        parent_policy_hash=str(value["parent_policy_hash"]),
        candidate_policy_hash=str(value["candidate_policy_hash"]),
        parent_status=str(value["parent_status"]),
        candidate_status=str(value["candidate_status"]),
        critical_signals=tuple(str(item) for item in value["critical_signals"]),
        source_evidence_hash=str(value["source_evidence_hash"]),
        schema_version=str(value["schema_version"]),
    )
    if value.get("request_hash") != request.request_hash:
        raise ValueError("boundary replay request hash mismatch")
    return request


__all__ = ["ExperienceService"]
