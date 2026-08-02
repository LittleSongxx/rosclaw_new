"""License evidence for one pinned MotionDecode source revision."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.collective.contracts import (
    LicenseDecision,
    LicenseUse,
    SourceLicenseEvidence,
)
from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_TERMS_BYTES = 1_000_000


@dataclass(frozen=True)
class MotionDecodeLicenseSnapshot:
    """Hashed terms plus a human legal decision.

    A terms file is evidence, not a permission oracle.  A non-empty snapshot
    may remain ``PENDING``; ``PERMITTED`` and ``DENIED`` require explicit
    caller input and non-empty terms.  Empty or absent current terms can only
    produce ``PENDING``.
    """

    requested_use: LicenseUse
    decision: LicenseDecision
    source_revision: str
    terms_uri: str | None
    terms_hash: str | None
    terms_size_bytes: int
    attribution: str
    schema_version: str = "rosclaw.collective.motiondecode_license_snapshot.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.requested_use, LicenseUse):
            raise ValueError("requested_use must be a recognized LicenseUse")
        if not isinstance(self.decision, LicenseDecision):
            raise ValueError("decision must be a recognized LicenseDecision")
        if not self.source_revision.strip():
            raise ValueError("source_revision must not be empty")
        if self.terms_size_bytes < 0:
            raise ValueError("terms_size_bytes must be non-negative")
        if self.terms_hash is not None and not _SHA256.fullmatch(self.terms_hash):
            raise ValueError("terms_hash must be a sha256: content hash")
        if (self.terms_hash is None) != (self.terms_size_bytes == 0):
            raise ValueError("terms hash and non-zero size must be supplied together")
        if self.terms_uri is not None and not self.terms_uri.strip():
            raise ValueError("terms_uri must be non-empty when supplied")
        if self.decision is not LicenseDecision.PENDING:
            if self.terms_hash is None or self.terms_uri is None:
                raise ValueError("a final license decision requires non-empty hashed terms")
            if not self.attribution.strip():
                raise ValueError("a final license decision requires attribution")

    @property
    def evidence(self) -> SourceLicenseEvidence:
        return SourceLicenseEvidence(
            requested_use=self.requested_use,
            decision=self.decision,
            terms_uri=self.terms_uri,
            terms_hash=self.terms_hash,
            attribution=self.attribution,
        )

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def training_permitted(self) -> bool:
        return self.decision is LicenseDecision.PERMITTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_use": self.requested_use.value,
            "decision": self.decision.value,
            "source_revision": self.source_revision,
            "terms_uri": self.terms_uri,
            "terms_hash": self.terms_hash,
            "terms_size_bytes": self.terms_size_bytes,
            "attribution": self.attribution,
            "training_permitted": self.training_permitted,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MotionDecodeLicenseSnapshot:
        return cls(
            requested_use=LicenseUse(str(value["requested_use"])),
            decision=LicenseDecision(str(value["decision"])),
            source_revision=str(value["source_revision"]),
            terms_uri=(str(value["terms_uri"]) if value.get("terms_uri") is not None else None),
            terms_hash=(str(value["terms_hash"]) if value.get("terms_hash") is not None else None),
            terms_size_bytes=int(value["terms_size_bytes"]),
            attribution=str(value.get("attribution", "")),
        )


def snapshot_license(
    *,
    source_revision: str,
    requested_use: LicenseUse,
    decision: LicenseDecision = LicenseDecision.PENDING,
    terms_path: Path | None = None,
    terms_uri: str | None = None,
    attribution: str = "ChingMu / CMRobot MotionDecode",
) -> MotionDecodeLicenseSnapshot:
    """Read at most one small terms document and create immutable evidence."""

    terms_hash: str | None = None
    terms_size = 0
    if terms_path is not None:
        resolved = terms_path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("terms_path must name a regular file")
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            payload = handle.read(_MAX_TERMS_BYTES + 1)
            after = os.fstat(handle.fileno())
        if _changed(before, after):
            raise ValueError("license terms changed while evidence was captured")
        if len(payload) > _MAX_TERMS_BYTES:
            raise ValueError("license terms exceed the 1 MB evidence limit")
        terms_size = len(payload)
        if payload:
            terms_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    if decision is not LicenseDecision.PENDING and terms_hash is None:
        raise ValueError("empty or absent terms cannot receive a final license decision")
    return MotionDecodeLicenseSnapshot(
        requested_use=requested_use,
        decision=decision,
        source_revision=source_revision,
        terms_uri=terms_uri,
        terms_hash=terms_hash,
        terms_size_bytes=terms_size,
        attribution=attribution,
    )


def _changed(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    )


__all__ = ["MotionDecodeLicenseSnapshot", "snapshot_license"]
