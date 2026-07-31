"""Finger-pair contact topology for TwinTouch (v4 §5/§11 T0).

Declares WHICH fingertip contacts are meaningful, which are forbidden,
and the measured physical layout they are grounded in.  Everything here
is declarative topology — no motion logic, no thresholds.

Physical basis (measured 2026-07-31, T0):

* Both RH56 hands are mounted on rigid stands, palms facing each other,
  fingertips pointing at the shared D435i (~0.2 m).
* Open-pose inter-hand lateral gap: ~0.045 m.  Mutual full curl closes
  it to CONTACT (observed, incident 2026-07-31) — reachability is
  proven for the all-finger case; PER-PAIR reachability stays honestly
  UNCALIBRATED until T1 measures each pair with micro-steps.

Topology rules (v4 §3.2/§5):

* Exactly ONE active pair at a time.
* Only same-finger pairs (thumb_thumb … little_little) are valid
  contact pairs; every cross-finger combination is a forbidden
  WRONG_FINGER_CONTACT.
* Palm bodies, finger roots, wrists and mounts are never approach
  targets — only the declared fingertip contact zones may enter
  near-contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RH56_JOINTS = ("little", "ring", "middle", "index", "thumb", "thumb_rot")
# Joints that end in a contactable fingertip (thumb_rot orients the thumb).
CONTACT_FINGERS = ("thumb", "index", "middle", "ring", "little")

SCHEMA_LAYOUT = "rosclaw.twintouch_layout.v1"
SCHEMA_REACHABILITY = "rosclaw.twintouch_reachability.v1"


@dataclass(frozen=True)
class FingerPair:
    """One permitted fingertip pair.  pair_id is canonical 'left_right'
    and for v4 phase 1 always same-finger (thumb_thumb …)."""

    pair_id: str
    left_finger: str
    right_finger: str

    def validate(self) -> list[str]:
        violations: list[str] = []
        if self.left_finger not in CONTACT_FINGERS:
            violations.append(f"left finger {self.left_finger!r} is not a contact finger")
        if self.right_finger not in CONTACT_FINGERS:
            violations.append(f"right finger {self.right_finger!r} is not a contact finger")
        if self.pair_id != f"{self.left_finger}_{self.right_finger}":
            violations.append(
                f"pair_id {self.pair_id!r} does not match {self.left_finger}_{self.right_finger}"
            )
        return violations


VALID_PAIRS: tuple[FingerPair, ...] = tuple(
    FingerPair(pair_id=f"{f}_{f}", left_finger=f, right_finger=f) for f in CONTACT_FINGERS
)
VALID_PAIR_IDS = frozenset(p.pair_id for p in VALID_PAIRS)

# Every cross-finger fingertip combination is forbidden (WRONG_FINGER_CONTACT).
FORBIDDEN_FINGERTIP_PAIRS: frozenset[str] = frozenset(
    f"{a}_{b}" for a in CONTACT_FINGERS for b in CONTACT_FINGERS if a != b
)

# Body regions that must never be an approach target (v4 §3.1).
FORBIDDEN_ZONES = ("palm_body", "finger_root", "wrist", "mount", "camera")

MAX_SIMULTANEOUS_ACTIVE_PAIRS = 1


def pair_by_id(pair_id: str) -> FingerPair | None:
    for pair in VALID_PAIRS:
        if pair.pair_id == pair_id:
            return pair
    return None


def is_valid_pair_id(pair_id: str) -> bool:
    return pair_id in VALID_PAIR_IDS


@dataclass(frozen=True)
class ForbiddenCollisionMap:
    """The explicit never-touch list an InteractionReceipt is checked
    against.  Symmetric by construction: forbidden fingertip pairs are
    declared as unordered semantics but canonical 'left_right' ids."""

    forbidden_fingertip_pairs: frozenset[str] = FORBIDDEN_FINGERTIP_PAIRS
    forbidden_zones: tuple[str, ...] = FORBIDDEN_ZONES
    max_active_pairs: int = MAX_SIMULTANEOUS_ACTIVE_PAIRS

    def is_forbidden_pair(self, pair_id: str) -> bool:
        return pair_id in self.forbidden_fingertip_pairs

    def validate_action_pairing(self, pair_id: str, active_pair_count: int = 1) -> list[str]:
        """Violations for attempting ``pair_id`` as the active pair."""
        violations: list[str] = []
        if self.is_forbidden_pair(pair_id):
            violations.append(f"{pair_id} is a forbidden cross-finger pair")
        elif not is_valid_pair_id(pair_id):
            violations.append(f"{pair_id} is not a declared contact pair")
        if active_pair_count > self.max_active_pairs:
            violations.append(f"{active_pair_count} active pairs exceeds one-pair-at-a-time rule")
        return violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "forbidden_fingertip_pairs": sorted(self.forbidden_fingertip_pairs),
            "forbidden_zones": list(self.forbidden_zones),
            "max_active_pairs": self.max_active_pairs,
        }


@dataclass(frozen=True)
class TwinTouchPhysicalLayout:
    """Measured physical arrangement (T0 output).  Any field that is not
    measured is None — an unmeasured layout property is a gap, not a
    guessed constant."""

    left_body_id: str
    right_body_id: str
    camera_id: str
    camera_pose_hash: str | None
    palms_facing_each_other: bool | None
    open_pose_lateral_gap_m: float | None
    mutual_reach_proven: bool | None  # all-finger mutual curl reached contact
    mount_separation_m: float | None
    measured_at: str | None  # ISO-8601 UTC
    evidence_refs: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        violations: list[str] = []
        if self.left_body_id == self.right_body_id:
            violations.append("left and right body ids must be distinct")
        for name in (
            "camera_pose_hash",
            "open_pose_lateral_gap_m",
            "mutual_reach_proven",
            "mount_separation_m",
            "measured_at",
        ):
            if getattr(self, name) in (None, ""):
                violations.append(f"layout property {name} is unmeasured")
        return violations

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_LAYOUT,
            "left_body_id": self.left_body_id,
            "right_body_id": self.right_body_id,
            "camera_id": self.camera_id,
            "camera_pose_hash": self.camera_pose_hash,
            "palms_facing_each_other": self.palms_facing_each_other,
            "open_pose_lateral_gap_m": self.open_pose_lateral_gap_m,
            "mutual_reach_proven": self.mutual_reach_proven,
            "mount_separation_m": self.mount_separation_m,
            "measured_at": self.measured_at,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TwinTouchPhysicalLayout:
        return cls(
            left_body_id=str(record.get("left_body_id")),
            right_body_id=str(record.get("right_body_id")),
            camera_id=str(record.get("camera_id")),
            camera_pose_hash=record.get("camera_pose_hash"),
            palms_facing_each_other=record.get("palms_facing_each_other"),
            open_pose_lateral_gap_m=record.get("open_pose_lateral_gap_m"),
            mutual_reach_proven=record.get("mutual_reach_proven"),
            mount_separation_m=record.get("mount_separation_m"),
            measured_at=record.get("measured_at"),
            evidence_refs=tuple(record.get("evidence_refs") or ()),
        )


# Per-pair reachability states (T0 vs T1 honesty).
REACHABILITY_UNKNOWN = "unknown"  # never measured
REACHABILITY_MUTUAL_CURL_ONLY = "mutual_curl_only"  # T0 all-finger touch; pair unisolated
REACHABILITY_CALIBRATED = "calibrated"  # T1 pair envelope exists with evidence


@dataclass(frozen=True)
class FingerPairReachabilityMatrix:
    """T0 output: per-pair reachability with evidence.  After the
    2026-07-31 T0 probe every pair is at most ``mutual_curl_only`` —
    the all-finger curl touched, but no pair was isolated; claiming
    per-pair reachability from that evidence would be fabrication."""

    states: dict[str, str]  # pair_id -> reachability state
    evidence_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def validate(self) -> list[str]:
        violations: list[str] = []
        for pair_id, state in self.states.items():
            if not is_valid_pair_id(pair_id):
                violations.append(f"{pair_id} is not a valid pair id")
            if state not in (
                REACHABILITY_UNKNOWN,
                REACHABILITY_MUTUAL_CURL_ONLY,
                REACHABILITY_CALIBRATED,
            ):
                violations.append(f"{pair_id} has unknown reachability state {state!r}")
            if state == REACHABILITY_CALIBRATED and not self.evidence_refs.get(pair_id):
                violations.append(f"{pair_id} claims calibrated reachability without evidence")
        missing = VALID_PAIR_IDS - set(self.states)
        if missing:
            violations.append(f"pairs missing from matrix: {sorted(missing)}")
        return violations

    def calibrated_pairs(self) -> list[str]:
        return sorted(p for p, s in self.states.items() if s == REACHABILITY_CALIBRATED)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_REACHABILITY,
            "states": dict(self.states),
            "evidence_refs": {k: list(v) for k, v in self.evidence_refs.items()},
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> FingerPairReachabilityMatrix:
        return cls(
            states={str(k): str(v) for k, v in (record.get("states") or {}).items()},
            evidence_refs={
                str(k): tuple(v) for k, v in (record.get("evidence_refs") or {}).items()
            },
        )

    @classmethod
    def from_t0_measurement(cls, evidence_ref: str) -> FingerPairReachabilityMatrix:
        """The honest T0 result: mutual curl reached contact (all
        fingers together); every pair is mutual_curl_only with the SAME
        shared evidence, calibrated for none."""
        return cls(
            states={p.pair_id: REACHABILITY_MUTUAL_CURL_ONLY for p in VALID_PAIRS},
            evidence_refs={p.pair_id: (evidence_ref,) for p in VALID_PAIRS},
        )
