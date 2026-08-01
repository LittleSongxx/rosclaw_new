"""FingerPairMemory + relationship memory + failure signatures (v4 §16, PR-TT-8).

Memory for TwinTouch is NOT "how the left hand moves" — it is
cross-body RELATIONSHIP memory (v4 §16.2):

    左手食指 + 右手食指
    在 Mutual 模式、Warm Regime 下
    需要左手提前 60 ms

Hard applicability filters (v4 §16.3): a thumb_thumb memory must never
apply to index_index; a left-active memory must never silently apply
to mutual.  The filter is EXACT-MATCH on the hard metadata — an
envelope that doesn't match the query's pair/mode/bodies/camera is
invisible, not downweighted.

Failure signatures (v4 §13.1) are the vocabulary candidates are born
from: every FingerPairMemory carries its observed signatures with
counts, so AUTO generates candidates from real failure residuals, not
blind enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash

SCHEMA_PAIR_MEMORY = "rosclaw.twintouch_pair_memory.v1"

# v4 §13.1 failure signatures
FAILURE_SIGNATURES = frozenset(
    {
        "NO_CONTACT_LEFT_SHORT",
        "NO_CONTACT_RIGHT_SHORT",
        "LEFT_EARLY_CONTACT",
        "RIGHT_EARLY_CONTACT",
        "CONTACT_FORCE_ASYMMETRY",
        "WRONG_FINGER_CONTACT",
        "UNINTENDED_NEIGHBOR_CONTACT",
        "START_SKEW_HIGH",
        "RELEASE_MARGIN_INSUFFICIENT",
        "VISUAL_FORCE_CONFLICT",
    }
)


@dataclass(frozen=True)
class MemoryScope:
    """The hard identity of a pair memory (v4 §16.3).  Every field is a
    hard filter: exact match or the memory does not exist for you."""

    left_body_hash: str
    right_body_hash: str
    pair_id: str
    interaction_mode: str  # active_passive | passive_active | mutual
    camera_pose_hash: str
    calibration_hashes: tuple[str, ...] = ()
    temperature_regime: str | None = None  # cold | warm_stable | hot_but_safe

    def matches(self, query: MemoryScope) -> bool:
        """Exact-match on every hard field.  temperature_regime=None in
        the QUERY means unscoped (matches any regime); in the MEMORY it
        means regime-unknown (matches only an unscoped query)."""
        if self.left_body_hash != query.left_body_hash:
            return False
        if self.right_body_hash != query.right_body_hash:
            return False
        if self.pair_id != query.pair_id:
            return False
        if self.interaction_mode != query.interaction_mode:
            return False
        if self.camera_pose_hash != query.camera_pose_hash:
            return False
        return not (
            query.temperature_regime is not None
            and self.temperature_regime is not None
            and self.temperature_regime != query.temperature_regime
        )


@dataclass(frozen=True)
class SuccessfulEnvelope:
    """The validated contact envelope for a pair×mode (from T1)."""

    precontact: dict[str, int]  # per-side target raw at fine-zone entry
    contact_range: dict[str, tuple[int, int]]  # per-side (min, max) confirm raw
    visual_near_range_m: tuple[float, float]
    force_baseline: dict[str, float]
    contact_force_envelope: dict[str, tuple[float, float]]  # per-side signed (min, max)
    release_margin_raw: int
    evidence_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "precontact": self.precontact,
            "contact_range": {k: list(v) for k, v in self.contact_range.items()},
            "visual_near_range_m": list(self.visual_near_range_m),
            "force_baseline": self.force_baseline,
            "contact_force_envelope": {k: list(v) for k, v in self.contact_force_envelope.items()},
            "release_margin_raw": self.release_margin_raw,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SuccessfulEnvelope:
        return cls(
            precontact=dict(record.get("precontact") or {}),
            contact_range={k: tuple(v) for k, v in (record.get("contact_range") or {}).items()},
            visual_near_range_m=tuple(record.get("visual_near_range_m") or (0.0, 0.0)),
            force_baseline=dict(record.get("force_baseline") or {}),
            contact_force_envelope={
                k: tuple(v) for k, v in (record.get("contact_force_envelope") or {}).items()
            },
            release_margin_raw=int(record.get("release_margin_raw") or 0),
            evidence_count=int(record.get("evidence_count") or 0),
        )


@dataclass(frozen=True)
class FingerPairMemory:
    """v4 §16.1 contact_pair memory: scope + successful envelope +
    failure signatures with counts + recovery hint + evidence refs."""

    scope: MemoryScope
    successful_envelope: SuccessfulEnvelope | None
    failure_signatures: dict[str, int]  # signature -> observed count
    recovery_hint: str | None
    evidence_refs: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        violations: list[str] = []
        unknown = set(self.failure_signatures) - FAILURE_SIGNATURES
        if unknown:
            violations.append(f"unknown failure signatures: {sorted(unknown)}")
        if self.successful_envelope is not None and self.successful_envelope.evidence_count < 2:
            violations.append(
                "successful envelope with < 2 evidence — a single session never validates (v3 gate)"
            )
        return violations

    def memory_hash(self) -> str:
        return canonical_hash(self.to_record(), prefix="fpmem")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_PAIR_MEMORY,
            "memory_type": "contact_pair",
            "scope": {
                "left_body_hash": self.scope.left_body_hash,
                "right_body_hash": self.scope.right_body_hash,
                "pair_id": self.scope.pair_id,
                "interaction_mode": self.scope.interaction_mode,
                "camera_pose_hash": self.scope.camera_pose_hash,
                "calibration_hashes": list(self.scope.calibration_hashes),
                "temperature_regime": self.scope.temperature_regime,
            },
            "successful_envelope": (
                None if self.successful_envelope is None else self.successful_envelope.to_record()
            ),
            "failure_signatures": dict(self.failure_signatures),
            "recovery_hint": self.recovery_hint,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> FingerPairMemory:
        s = record.get("scope") or {}
        return cls(
            scope=MemoryScope(
                left_body_hash=str(s.get("left_body_hash") or ""),
                right_body_hash=str(s.get("right_body_hash") or ""),
                pair_id=str(s.get("pair_id") or ""),
                interaction_mode=str(s.get("interaction_mode") or ""),
                camera_pose_hash=str(s.get("camera_pose_hash") or ""),
                calibration_hashes=tuple(s.get("calibration_hashes") or ()),
                temperature_regime=s.get("temperature_regime"),
            ),
            successful_envelope=(
                None
                if record.get("successful_envelope") is None
                else SuccessfulEnvelope.from_record(record["successful_envelope"])
            ),
            failure_signatures={
                str(k): int(v) for k, v in (record.get("failure_signatures") or {}).items()
            },
            recovery_hint=record.get("recovery_hint"),
            evidence_refs=tuple(record.get("evidence_refs") or ()),
        )


def filter_pair_memories(
    memories: list[FingerPairMemory], query: MemoryScope
) -> list[FingerPairMemory]:
    """The §16.3 hard filter: exact scope match or invisible.  A
    thumb_thumb memory NEVER applies to index_index; a left-active
    memory NEVER silently applies to mutual; a stale camera pose hides
    everything measured under it."""
    return [m for m in memories if m.scope.matches(query)]


# ------------------------------------------------------------ signatures


# Outcome -> failure signature mapping (supervisor outcomes to §13.1
# vocabulary).  Agency-bearing fields (side) come from the episode.
def failure_signature_for(
    *,
    outcome: str,
    active_mode: str,
    one_sided_side: str | None = None,
) -> str | None:
    """Map a supervisor outcome to its §13.1 failure signature.
    Returns None for outcomes that are not failure residuals
    (CONTACT_CONFIRMED, transport/thermal/stale — those are
    environment, not pair residuals)."""
    if outcome == "NO_CONTACT":
        side = one_sided_side or (
            "LEFT"
            if active_mode == "active_passive"
            else "RIGHT"
            if active_mode == "passive_active"
            else None
        )
        if side == "LEFT":
            return "NO_CONTACT_LEFT_SHORT"
        if side == "RIGHT":
            return "NO_CONTACT_RIGHT_SHORT"
        return None
    if outcome == "EARLY_CONTACT":
        side = one_sided_side or (
            "LEFT"
            if active_mode == "active_passive"
            else "RIGHT"
            if active_mode == "passive_active"
            else None
        )
        if side == "LEFT":
            return "LEFT_EARLY_CONTACT"
        if side == "RIGHT":
            return "RIGHT_EARLY_CONTACT"
        return None
    if outcome == "ONE_SIDED_FORCE":
        return "CONTACT_FORCE_ASYMMETRY"
    if outcome == "WRONG_FINGER_CONTACT":
        return "WRONG_FINGER_CONTACT"
    if outcome == "UNINTENDED_CONTACT":
        return "UNINTENDED_NEIGHBOR_CONTACT"
    if outcome == "VISUAL_FORCE_CONFLICT":
        return "VISUAL_FORCE_CONFLICT"
    if outcome == "RELEASE_FAILED":
        return "RELEASE_MARGIN_INSUFFICIENT"
    return None
