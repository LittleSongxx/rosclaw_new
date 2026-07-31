"""ContactChoreographyContract + one-time SequencePermit (v4 §18).

The operator never approves "let the hands interact"; the operator
approves ONE exact, content-hashed choreography — pair sequence, cycle
count, force level, bound bodies, bound camera pose, bound pair
envelopes.  The permit derived from it is:

* bound to the contract hash (any change to the choreography is a new
  approval),
* bound to the operator's intent hash (deadline-bound intent, same
  pattern as the authorized-blackbox choreography),
* single-use (a PARTIAL_DISPATCH or completion revokes it — v4 §7.3).

No permit, no dispatch.  A stale or mismatched permit is a finding,
not a request for a new one.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash
from rosclaw.twintouch.pairs import is_valid_pair_id

SCHEMA_VERSION = "rosclaw.contact_choreography.v1"

PATTERNS = ("fingertip_marquee", "echo_touch", "mirror_wave", "tap_code", "adaptive_duet")
# Phase-1 acceptance uses ONLY the marquee (v4 §19); the rest are
# declared so the schema does not pretend they do not exist, but a
# contract requesting them is invalid until explicitly enabled.
PHASE1_ENABLED_PATTERNS = ("fingertip_marquee",)

FORCE_LEVELS = ("ultra_light", "light")

# Canonical marquee: thumb → little → back (v4 §1/§18).
CANONICAL_MARQUEE_PAIRS = (
    "thumb_thumb",
    "index_index",
    "middle_middle",
    "ring_ring",
    "little_little",
    "ring_ring",
    "middle_middle",
    "index_index",
    "thumb_thumb",
)


@dataclass(frozen=True)
class ContactChoreographyContract:
    """The exact approved interaction.  ``pair_envelope_hashes`` binds
    the T1-calibrated FingerContactEnvelope per pair: a choreography
    over uncalibrated pairs is only valid at ``ultra_light`` probe
    force with envelopes absent — never at contact force levels."""

    pattern: str
    pairs: tuple[str, ...]
    cycles: int
    force_level: str
    left_body_hash: str | None
    right_body_hash: str | None
    camera_pose_hash: str | None
    pair_envelope_hashes: dict[str, str] = field(default_factory=dict)
    created_at: str | None = None
    phase1_only: bool = True

    def contract_hash(self) -> str:
        return canonical_hash(self.to_record(), prefix="chor")

    def validate(self) -> list[str]:
        violations: list[str] = []
        if self.pattern not in PATTERNS:
            violations.append(f"pattern {self.pattern!r} is not a declared pattern")
        elif self.phase1_only and self.pattern not in PHASE1_ENABLED_PATTERNS:
            violations.append(
                f"pattern {self.pattern!r} is not enabled in phase 1 "
                f"(only {PHASE1_ENABLED_PATTERNS})"
            )
        if not self.pairs:
            violations.append("pairs sequence is empty")
        for pair_id in self.pairs:
            if not is_valid_pair_id(pair_id):
                violations.append(f"pair {pair_id!r} is not a permitted contact pair")
        if self.pattern == "fingertip_marquee" and tuple(self.pairs) != CANONICAL_MARQUEE_PAIRS:
            violations.append(
                "fingertip_marquee requires the canonical thumb→little→thumb sequence; "
                "declare a different pattern for custom sequences"
            )
        if self.cycles < 1:
            violations.append("cycles must be >= 1")
        if self.force_level not in FORCE_LEVELS:
            violations.append(f"force_level {self.force_level!r} not in {FORCE_LEVELS}")
        if not self.left_body_hash or not self.right_body_hash:
            violations.append("both body hashes are required (unbound choreography)")
        if self.left_body_hash and self.left_body_hash == self.right_body_hash:
            violations.append("left and right body hashes must differ")
        if not self.camera_pose_hash:
            violations.append("camera_pose_hash missing (visual contract unbound)")
        # Force-level discipline: contact-force levels need T1 envelopes.
        uncalibrated = [p for p in set(self.pairs) if p not in self.pair_envelope_hashes]
        if self.force_level != "ultra_light" and uncalibrated:
            violations.append(
                f"force_level {self.force_level!r} requires T1 pair envelopes; "
                f"missing for {sorted(uncalibrated)}"
            )
        if not self.created_at:
            violations.append("created_at missing")
        return violations

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "pattern": self.pattern,
            "pairs": list(self.pairs),
            "cycles": self.cycles,
            "force_level": self.force_level,
            "left_body_hash": self.left_body_hash,
            "right_body_hash": self.right_body_hash,
            "camera_pose_hash": self.camera_pose_hash,
            "pair_envelope_hashes": dict(self.pair_envelope_hashes),
            "created_at": self.created_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ContactChoreographyContract:
        return cls(
            pattern=str(record.get("pattern") or ""),
            pairs=tuple(record.get("pairs") or ()),
            cycles=int(record.get("cycles") or 0),
            force_level=str(record.get("force_level") or ""),
            left_body_hash=record.get("left_body_hash"),
            right_body_hash=record.get("right_body_hash"),
            camera_pose_hash=record.get("camera_pose_hash"),
            pair_envelope_hashes={
                str(k): str(v) for k, v in (record.get("pair_envelope_hashes") or {}).items()
            },
            created_at=record.get("created_at"),
        )


# Permit verification verdicts.
PERMIT_OK = "ok"
PERMIT_EXPIRED = "expired"
PERMIT_HASH_MISMATCH = "contract_hash_mismatch"
PERMIT_ALREADY_USED = "already_used"


@dataclass
class SequencePermit:
    """One-time, deadline-bound authority for ONE contract hash."""

    permit_id: str
    contract_hash: str
    intent_hash: str  # operator's deadline-bound intent (confirmation card)
    issued_at_s: float
    expires_at_s: float
    used: bool = False
    revoked_reason: str | None = None

    @classmethod
    def issue(
        cls,
        contract: ContactChoreographyContract,
        *,
        intent_hash: str,
        lifetime_s: float,
        now_s: float | None = None,
    ) -> SequencePermit:
        now = time.time() if now_s is None else now_s
        return cls(
            permit_id=f"permit_{uuid.uuid4().hex[:12]}",
            contract_hash=contract.contract_hash(),
            intent_hash=intent_hash,
            issued_at_s=now,
            expires_at_s=now + lifetime_s,
        )

    def verify(self, contract: ContactChoreographyContract, *, now_s: float | None = None) -> str:
        now = time.time() if now_s is None else now_s
        if self.used:
            return PERMIT_ALREADY_USED
        if self.revoked_reason:
            return PERMIT_ALREADY_USED
        if now > self.expires_at_s:
            return PERMIT_EXPIRED
        if contract.contract_hash() != self.contract_hash:
            return PERMIT_HASH_MISMATCH
        return PERMIT_OK

    def consume(self, *, reason: str = "sequence_started") -> None:
        """Single-use: starting the sequence consumes the permit.  A
        PARTIAL_DISPATCH revokes it mid-sequence (v4 §7.3) — same
        mechanism, different reason."""
        self.used = True
        self.revoked_reason = reason

    def revoke(self, *, reason: str) -> None:
        self.used = True
        self.revoked_reason = reason

    def to_record(self) -> dict[str, Any]:
        return {
            "permit_id": self.permit_id,
            "contract_hash": self.contract_hash,
            "intent_hash": self.intent_hash,
            "issued_at_s": self.issued_at_s,
            "expires_at_s": self.expires_at_s,
            "used": self.used,
            "revoked_reason": self.revoked_reason,
        }
