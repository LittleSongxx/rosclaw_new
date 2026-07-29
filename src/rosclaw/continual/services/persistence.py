"""Crash-safe append-only event storage for continual-learning services."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_GENESIS_HASH = "sha256:" + "0" * 64


def require_external_service_root(root: Path, source_checkout: Path) -> Path:
    """Keep mutable service state and raw experience outside the checkout."""

    resolved = root.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("continual service state root must not be a filesystem root")
    if resolved == checkout or checkout in resolved.parents:
        raise ValueError("continual service state must be outside the source checkout")
    return resolved


@dataclass(frozen=True)
class DurableServiceEvent:
    service: str
    sequence: int
    timestamp_ns: int
    kind: str
    payload: Mapping[str, Any]
    previous_event_hash: str
    event_hash: str
    schema_version: str = "rosclaw.continual.service_event.v1"

    def __post_init__(self) -> None:
        if not self.service.strip() or not self.kind.strip():
            raise ValueError("service event names must not be empty")
        if self.sequence <= 0 or self.timestamp_ns < 0:
            raise ValueError("service event sequence must be positive and timestamp non-negative")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        if self.event_hash != canonical_hash(self.hash_material()):
            raise ValueError("service event hash mismatch")

    def hash_material(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "service": self.service,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "kind": self.kind,
            "payload": dict(self.payload),
            "previous_event_hash": self.previous_event_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.hash_material(), "event_hash": self.event_hash}


class DurableEventLog:
    """Immutable file-per-event log; recovery never truncates or rewrites history."""

    def __init__(
        self,
        root: Path,
        *,
        service: str,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not service.strip():
            raise ValueError("durable service name must not be empty")
        self.root = root.expanduser().resolve()
        self.service = service
        self.clock_ns = clock_ns
        self.events_dir = self.root / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._events = self._read_events()

    @property
    def events(self) -> tuple[DurableServiceEvent, ...]:
        return self._events

    @property
    def last_hash(self) -> str:
        return self._events[-1].event_hash if self._events else _GENESIS_HASH

    def append(self, kind: str, payload: Mapping[str, Any]) -> DurableServiceEvent:
        sequence = len(self._events) + 1
        material = {
            "schema_version": "rosclaw.continual.service_event.v1",
            "service": self.service,
            "sequence": sequence,
            "timestamp_ns": int(self.clock_ns()),
            "kind": kind,
            "payload": dict(payload),
            "previous_event_hash": self.last_hash,
        }
        event = DurableServiceEvent(event_hash=canonical_hash(material), **material)
        destination = self.events_dir / f"{sequence:020d}.json"
        _atomic_create_json(destination, event.to_dict())
        self._events = (*self._events, event)
        return event

    def _read_events(self) -> tuple[DurableServiceEvent, ...]:
        result = []
        previous = _GENESIS_HASH
        for expected, path in enumerate(sorted(self.events_dir.glob("[0-9]*.json")), start=1):
            value = json.loads(path.read_text(encoding="utf-8"))
            event = DurableServiceEvent(
                service=str(value["service"]),
                sequence=int(value["sequence"]),
                timestamp_ns=int(value["timestamp_ns"]),
                kind=str(value["kind"]),
                payload=dict(value["payload"]),
                previous_event_hash=str(value["previous_event_hash"]),
                event_hash=str(value["event_hash"]),
                schema_version=str(value["schema_version"]),
            )
            if event.service != self.service or event.sequence != expected:
                raise ValueError("service event sequence or owner mismatch")
            if event.previous_event_hash != previous:
                raise ValueError("service event hash chain is broken")
            previous = event.event_hash
            result.append(event)
        return tuple(result)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"append-only service event already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        Path(temporary).unlink()
        fsync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def fsync_directory(path: Path) -> None:
    """Persist a directory entry after an atomic link or replacement."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DurableEventLog",
    "DurableServiceEvent",
    "atomic_write_json",
    "fsync_directory",
    "require_external_service_root",
]
