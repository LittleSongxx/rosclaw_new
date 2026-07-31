"""BimanualActionEnvelope (v4 §7.1) — one atomic dual-body action.

The envelope is the ONLY unit the Bimanual ActionGateway accepts.  It
binds both bodies' actions, the coordination mode, the synchronization
barrier and the safety contract into one content-addressed record, so
"what was attempted" is never reconstructed from logs after the fact.

Design rules:

* ``validate()`` returns violations; it never raises on bad data.
* The permitted contact pair is part of the SAFETY block, not implied
  by the actions — an envelope whose safety permits thumb_thumb while
  pair_id says index_index is invalid on its face.
* Hashes (body snapshot, calibration, contract) are required identity:
  an action without the body state it was planned against is not
  dispatchable (v4 §7.2 step 4-5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash
from rosclaw.twintouch.pairs import is_valid_pair_id

SCHEMA_VERSION = "rosclaw.bimanual_action.v1"

COORDINATION_MODES = ("active_passive", "passive_active", "mutual")


@dataclass(frozen=True)
class BodyActionBlock:
    """One side of the envelope: the action plus the identity of the
    body state it was planned against."""

    body_id: str
    action: dict[str, Any]  # e.g. {"gesture": "approach", "targets": {"index": 620, ...}}
    body_snapshot_hash: str | None
    calibration_hash: str | None

    def validate(self, *, side: str) -> list[str]:
        violations: list[str] = []
        if not self.body_id:
            violations.append(f"{side}: body_id missing")
        if not self.action:
            violations.append(f"{side}: action missing")
        if not self.body_snapshot_hash:
            violations.append(f"{side}: body_snapshot_hash missing (planned against what?)")
        if not self.calibration_hash:
            violations.append(f"{side}: calibration_hash missing")
        return violations

    def to_record(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "action": self.action,
            "body_snapshot_hash": self.body_snapshot_hash,
            "calibration_hash": self.calibration_hash,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> BodyActionBlock:
        return cls(
            body_id=str(record.get("body_id") or ""),
            action=dict(record.get("action") or {}),
            body_snapshot_hash=record.get("body_snapshot_hash"),
            calibration_hash=record.get("calibration_hash"),
        )


@dataclass(frozen=True)
class CoordinationBlock:
    mode: str  # active_passive | passive_active | mutual
    synchronization_barrier: str  # e.g. "start_together" | "active_only"
    maximum_start_skew_ms: float
    timeout_ms: float

    def validate(self) -> list[str]:
        violations: list[str] = []
        if self.mode not in COORDINATION_MODES:
            violations.append(f"coordination mode {self.mode!r} not in {COORDINATION_MODES}")
        if not self.synchronization_barrier:
            violations.append("synchronization_barrier missing")
        if self.maximum_start_skew_ms <= 0:
            violations.append("maximum_start_skew_ms must be positive")
        if self.timeout_ms <= 0:
            violations.append("timeout_ms must be positive")
        return violations

    def to_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "synchronization_barrier": self.synchronization_barrier,
            "maximum_start_skew_ms": self.maximum_start_skew_ms,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CoordinationBlock:
        return cls(
            mode=str(record.get("mode") or ""),
            synchronization_barrier=str(record.get("synchronization_barrier") or ""),
            maximum_start_skew_ms=float(record.get("maximum_start_skew_ms") or 0),
            timeout_ms=float(record.get("timeout_ms") or 0),
        )


@dataclass(frozen=True)
class SafetyBlock:
    """The safety binding of this action: which choreography contract
    authorizes it, which single pair may contact, which pairs are
    forbidden, and the retreat action both hands fall back to."""

    contract_hash: str | None
    permitted_contact_pair: str
    forbidden_contact_pairs: tuple[str, ...]
    retreat_action: dict[str, Any]  # e.g. {"gesture": "safe_open", "speed": 300, "force": 150}

    def validate(self, *, pair_id: str) -> list[str]:
        violations: list[str] = []
        if not self.contract_hash:
            violations.append("safety.contract_hash missing (unauthorized action)")
        if self.permitted_contact_pair != pair_id:
            violations.append(
                f"safety permits {self.permitted_contact_pair!r} but pair_id is {pair_id!r}"
            )
        if pair_id in self.forbidden_contact_pairs:
            violations.append(f"pair_id {pair_id!r} is in the forbidden list")
        if not self.retreat_action:
            violations.append("safety.retreat_action missing (no fallback)")
        return violations

    def to_record(self) -> dict[str, Any]:
        return {
            "contract_hash": self.contract_hash,
            "permitted_contact_pair": self.permitted_contact_pair,
            "forbidden_contact_pairs": list(self.forbidden_contact_pairs),
            "retreat_action": self.retreat_action,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SafetyBlock:
        return cls(
            contract_hash=record.get("contract_hash"),
            permitted_contact_pair=str(record.get("permitted_contact_pair") or ""),
            forbidden_contact_pairs=tuple(record.get("forbidden_contact_pairs") or ()),
            retreat_action=dict(record.get("retreat_action") or {}),
        )


@dataclass(frozen=True)
class BimanualActionEnvelope:
    interaction_id: str
    sequence_id: str
    pair_id: str
    left: BodyActionBlock
    right: BodyActionBlock
    coordination: CoordinationBlock
    safety: SafetyBlock

    def validate(self) -> list[str]:
        violations: list[str] = []
        if not self.interaction_id:
            violations.append("interaction_id missing")
        if not self.sequence_id:
            violations.append("sequence_id missing")
        if not is_valid_pair_id(self.pair_id):
            violations.append(f"pair_id {self.pair_id!r} is not a permitted contact pair")
        violations.extend(self.left.validate(side="left"))
        violations.extend(self.right.validate(side="right"))
        if self.left.body_id and self.left.body_id == self.right.body_id:
            violations.append("left and right body_id must be distinct bodies")
        violations.extend(self.coordination.validate())
        violations.extend(self.safety.validate(pair_id=self.pair_id))
        return violations

    def envelope_hash(self) -> str:
        """Content-addressed identity of exactly this attempt."""
        return canonical_hash(self.to_record(), prefix="bimact")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "interaction_id": self.interaction_id,
            "sequence_id": self.sequence_id,
            "pair_id": self.pair_id,
            "left": self.left.to_record(),
            "right": self.right.to_record(),
            "coordination": self.coordination.to_record(),
            "safety": self.safety.to_record(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> BimanualActionEnvelope:
        return cls(
            interaction_id=str(record.get("interaction_id") or ""),
            sequence_id=str(record.get("sequence_id") or ""),
            pair_id=str(record.get("pair_id") or ""),
            left=BodyActionBlock.from_record(record.get("left") or {}),
            right=BodyActionBlock.from_record(record.get("right") or {}),
            coordination=CoordinationBlock.from_record(record.get("coordination") or {}),
            safety=SafetyBlock.from_record(record.get("safety") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_record(), sort_keys=True, ensure_ascii=False)
