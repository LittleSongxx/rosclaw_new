"""InteractionReceipt (v4 §7.3/§7.4) — what actually happened.

One receipt per interaction attempt, including the attempts that never
reached contact.  The receipt is the unit the data quality gate, the
trace tree, memory distillation and the evolution gate all consume —
so every field that was not measured stays None instead of being
back-filled with a plausible number.

Outcomes cover the full §5 anomaly space plus §7.3 partial dispatch:
a receipt whose left hand dispatched but whose right hand did not is
``PARTIAL_DISPATCH`` — never a silent half-action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash
from rosclaw.twintouch.pairs import is_valid_pair_id

SCHEMA_VERSION = "rosclaw.interaction_receipt.v1"

# Terminal outcomes (v4 §5 state machine + §7.3 dispatch failures).
OUTCOME_CONTACT_CONFIRMED = "CONTACT_CONFIRMED"
OUTCOME_NO_CONTACT = "NO_CONTACT"
OUTCOME_EARLY_CONTACT = "EARLY_CONTACT"
OUTCOME_WRONG_FINGER_CONTACT = "WRONG_FINGER_CONTACT"
OUTCOME_UNINTENDED_CONTACT = "UNINTENDED_CONTACT"
OUTCOME_ONE_SIDED_FORCE = "ONE_SIDED_FORCE"
OUTCOME_VISUAL_FORCE_CONFLICT = "VISUAL_FORCE_CONFLICT"
OUTCOME_PEER_NOT_READY = "PEER_NOT_READY"
OUTCOME_STALE_OBSERVATION = "STALE_OBSERVATION"
OUTCOME_TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
OUTCOME_RELEASE_FAILED = "RELEASE_FAILED"
OUTCOME_THERMAL_ABORT = "THERMAL_ABORT"
OUTCOME_PARTIAL_DISPATCH = "PARTIAL_DISPATCH"
OUTCOME_ABORTED_BEFORE_DISPATCH = "ABORTED_BEFORE_DISPATCH"


def _all_outcomes() -> frozenset[str]:
    return frozenset(
        v for k, v in list(globals().items()) if k.startswith("OUTCOME_") and isinstance(v, str)
    )


# Outcomes that mean "the hands touched something they should not have".
CONTACT_ANOMALY_OUTCOMES = frozenset(
    {
        OUTCOME_EARLY_CONTACT,
        OUTCOME_WRONG_FINGER_CONTACT,
        OUTCOME_UNINTENDED_CONTACT,
        OUTCOME_ONE_SIDED_FORCE,
    }
)


@dataclass(frozen=True)
class InteractionReceipt:
    interaction_id: str
    pair_id: str
    left_action_receipt: str | None  # receipt ids / hashes of each side's dispatch
    right_action_receipt: str | None

    intended_contact: str  # pair_id the envelope intended
    observed_contact: str | None  # pair_id actually observed, None if none/unknown
    contact_confirmed: bool
    wrong_finger_contact: bool
    unintended_contact: bool

    left_force_peak: float | None
    right_force_peak: float | None
    visual_distance_min_m: float | None
    contact_latency_ms: float | None
    start_skew_ms: float | None
    dwell_ms: float | None

    clearance_verified: bool
    outcome: str
    evidence_refs: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        violations: list[str] = []
        if not self.interaction_id:
            violations.append("interaction_id missing")
        if not is_valid_pair_id(self.pair_id):
            violations.append(f"pair_id {self.pair_id!r} is not a permitted pair")
        if self.intended_contact != self.pair_id:
            violations.append(
                f"intended_contact {self.intended_contact!r} disagrees with pair_id {self.pair_id!r}"
            )
        if self.outcome not in _all_outcomes():
            violations.append(f"outcome {self.outcome!r} is not a terminal outcome")
        # Consistency rules — the receipt must not contradict itself.
        if self.contact_confirmed and self.outcome != OUTCOME_CONTACT_CONFIRMED:
            violations.append("contact_confirmed=true but outcome is not CONTACT_CONFIRMED")
        if self.outcome == OUTCOME_CONTACT_CONFIRMED and not self.contact_confirmed:
            violations.append("outcome CONTACT_CONFIRMED requires contact_confirmed=true")
        if self.wrong_finger_contact and self.outcome != OUTCOME_WRONG_FINGER_CONTACT:
            violations.append("wrong_finger_contact=true but outcome is not WRONG_FINGER_CONTACT")
        if self.unintended_contact and self.outcome != OUTCOME_UNINTENDED_CONTACT:
            violations.append("unintended_contact=true but outcome is not UNINTENDED_CONTACT")
        if self.outcome == OUTCOME_PARTIAL_DISPATCH and (
            self.left_action_receipt and self.right_action_receipt
        ):
            violations.append("PARTIAL_DISPATCH with both sides dispatched is contradictory")
        if self.contact_confirmed and not self.clearance_verified:
            violations.append("contact confirmed but clearance never verified (release untracked)")
        if (
            self.contact_confirmed
            and self.left_force_peak is None
            and self.right_force_peak is None
        ):
            violations.append("contact confirmed without any force peak evidence")
        return violations

    def receipt_hash(self) -> str:
        return canonical_hash(self.to_record(), prefix="ircpt")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "interaction_id": self.interaction_id,
            "pair_id": self.pair_id,
            "left_action_receipt": self.left_action_receipt,
            "right_action_receipt": self.right_action_receipt,
            "intended_contact": self.intended_contact,
            "observed_contact": self.observed_contact,
            "contact_confirmed": self.contact_confirmed,
            "wrong_finger_contact": self.wrong_finger_contact,
            "unintended_contact": self.unintended_contact,
            "left_force_peak": self.left_force_peak,
            "right_force_peak": self.right_force_peak,
            "visual_distance_min_m": self.visual_distance_min_m,
            "contact_latency_ms": self.contact_latency_ms,
            "start_skew_ms": self.start_skew_ms,
            "dwell_ms": self.dwell_ms,
            "clearance_verified": self.clearance_verified,
            "outcome": self.outcome,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> InteractionReceipt:
        return cls(
            interaction_id=str(record.get("interaction_id") or ""),
            pair_id=str(record.get("pair_id") or ""),
            left_action_receipt=record.get("left_action_receipt"),
            right_action_receipt=record.get("right_action_receipt"),
            intended_contact=str(record.get("intended_contact") or ""),
            observed_contact=record.get("observed_contact"),
            contact_confirmed=bool(record.get("contact_confirmed")),
            wrong_finger_contact=bool(record.get("wrong_finger_contact")),
            unintended_contact=bool(record.get("unintended_contact")),
            left_force_peak=record.get("left_force_peak"),
            right_force_peak=record.get("right_force_peak"),
            visual_distance_min_m=record.get("visual_distance_min_m"),
            contact_latency_ms=record.get("contact_latency_ms"),
            start_skew_ms=record.get("start_skew_ms"),
            dwell_ms=record.get("dwell_ms"),
            clearance_verified=bool(record.get("clearance_verified")),
            outcome=str(record.get("outcome") or ""),
            evidence_refs=tuple(record.get("evidence_refs") or ()),
        )
