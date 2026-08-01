"""Peer state, interaction snapshot and agency attribution (v4 §9/§10, PR-TT-7).

The two hands are NOT merged into one body: each keeps its own
SelfSnapshot, and on top of them sits an interaction layer — a read-only
PeerState per side and an InteractionSelfSnapshot per episode (v4 §9.3).

Agency (v4 §10) answers "who caused this contact?":

    LEFT_SELF_CAUSED   left actively approached, right held, contact
    RIGHT_SELF_CAUSED  mirror
    MUTUAL_CONTACT     both approached per plan, contact in the window
    PEER_DISTURBANCE   the passive side moved without a plan
    SENSOR_FAULT      camera says contact, neither force channel agrees
    UNKNOWN            occlusion / channel anomaly — cannot attribute

Agency decides: whether the episode may be learned from, which body a
failure belongs to, which side a candidate may adjust, and whether a
cross-body memory entry is created.  SENSOR_FAULT episodes NEVER enter
control learning (v4 §22).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash

SCHEMA_PEER = "rosclaw.twintouch_peer.v1"
SCHEMA_INTERACTION_SNAPSHOT = "rosclaw.twintouch_interaction_snapshot.v1"

AGENCY_LEFT = "LEFT_SELF_CAUSED"
AGENCY_RIGHT = "RIGHT_SELF_CAUSED"
AGENCY_MUTUAL = "MUTUAL_CONTACT"
AGENCY_PEER_DISTURBANCE = "PEER_DISTURBANCE"
AGENCY_SENSOR_FAULT = "SENSOR_FAULT"
AGENCY_UNKNOWN = "UNKNOWN"

ALL_AGENCIES = frozenset(
    {
        AGENCY_LEFT,
        AGENCY_RIGHT,
        AGENCY_MUTUAL,
        AGENCY_PEER_DISTURBANCE,
        AGENCY_SENSOR_FAULT,
        AGENCY_UNKNOWN,
    }
)


@dataclass(frozen=True)
class PeerState:
    """One hand's read-only view of the other (v4 §9.2)."""

    peer_body_id: str
    ready: bool
    intended_finger: str | None
    current_finger_state: str  # open | presenting | approaching | contacting | retreating | unknown
    visual_tip_position: tuple[float, float, float] | None
    approaching: bool | None
    contact_force: float | None  # signed delta the peer feels
    prediction_uncertainty: float | None  # None until calibrated (v3 §8)
    health: str  # ok | degraded | fault | unknown

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_PEER,
            "peer_body_id": self.peer_body_id,
            "ready": self.ready,
            "intended_finger": self.intended_finger,
            "current_finger_state": self.current_finger_state,
            "visual_tip_position": self.visual_tip_position,
            "approaching": self.approaching,
            "contact_force": self.contact_force,
            "prediction_uncertainty": self.prediction_uncertainty,
            "health": self.health,
        }


@dataclass(frozen=True)
class InteractionSelfSnapshot:
    """The relationship layer over two independent SelfSnapshots
    (v4 §9.3) — relative geometry + expectations + current contact
    state, not a merged body."""

    left_self_snapshot_hash: str | None
    right_self_snapshot_hash: str | None
    intended_pair: str
    relative_tip_geometry: dict[str, Any]  # {"distance_m": x, "axis": "lateral"|"mixed"|"unknown"}
    expected_time_to_contact_ms: float | None
    expected_contact_force: dict[str, float] | None
    expected_active_side: str
    current_contact_state: str  # supervisor state
    peer_readiness: dict[str, bool]
    collision_risk: str  # none | watch | abort

    def snapshot_hash(self) -> str:
        return canonical_hash(self.to_record(), prefix="isnap")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_INTERACTION_SNAPSHOT,
            "left_self_snapshot_hash": self.left_self_snapshot_hash,
            "right_self_snapshot_hash": self.right_self_snapshot_hash,
            "intended_pair": self.intended_pair,
            "relative_tip_geometry": self.relative_tip_geometry,
            "expected_time_to_contact_ms": self.expected_time_to_contact_ms,
            "expected_contact_force": self.expected_contact_force,
            "expected_active_side": self.expected_active_side,
            "current_contact_state": self.current_contact_state,
            "peer_readiness": self.peer_readiness,
            "collision_risk": self.collision_risk,
        }


@dataclass(frozen=True)
class AgencyEvidence:
    """The facts agency may use — nothing else.  Every field is
    measured; unmeasured is None, never assumed."""

    active_mode: str  # active_passive | passive_active | mutual
    contact_happened: bool
    left_target_delta: float | None
    right_target_delta: float | None
    left_approached: bool | None  # left hand moved toward the peer per plan
    right_approached: bool | None
    passive_side_moved_unplanned: bool | None  # EXTERNAL_ONLY on the passive side
    visual_says_contact: bool | None  # camera claims near/merged
    visual_unknown: bool  # occlusion / stale / no measurement
    contact_threshold: float = 35.0


@dataclass(frozen=True)
class AgencyVerdict:
    agency: str
    rationale: str
    learnable: bool  # SENSOR_FAULT and UNKNOWN never enter control learning
    failure_owner: str | None  # which body owns the failure, if any
    candidate_side: str | None  # which side a candidate may adjust

    def to_record(self) -> dict[str, Any]:
        return {
            "agency": self.agency,
            "rationale": self.rationale,
            "learnable": self.learnable,
            "failure_owner": self.failure_owner,
            "candidate_side": self.candidate_side,
        }


def attribute_agency(evidence: AgencyEvidence) -> AgencyVerdict:
    """Deterministic §10 attribution over measured evidence only."""

    # SENSOR_FAULT: camera claims contact while NEITHER force channel
    # crosses — the camera and the hands disagree, trust neither.
    if (
        evidence.visual_says_contact is True
        and evidence.left_target_delta is not None
        and evidence.right_target_delta is not None
        and abs(evidence.left_target_delta) < evidence.contact_threshold
        and abs(evidence.right_target_delta) < evidence.contact_threshold
    ):
        return AgencyVerdict(
            agency=AGENCY_SENSOR_FAULT,
            rationale=(
                f"camera claims contact at L {evidence.left_target_delta:+.0f} / "
                f"R {evidence.right_target_delta:+.0f} — force channels disagree"
            ),
            learnable=False,
            failure_owner=None,
            candidate_side=None,
        )

    # UNKNOWN: measurement channels broken or visual blind.
    if (
        evidence.visual_unknown
        or evidence.left_target_delta is None
        or evidence.right_target_delta is None
    ):
        return AgencyVerdict(
            agency=AGENCY_UNKNOWN,
            rationale="occlusion / missing force channel — attribution impossible",
            learnable=False,
            failure_owner=None,
            candidate_side=None,
        )

    # PEER_DISTURBANCE: the passive side moved without a plan.
    if evidence.passive_side_moved_unplanned:
        owner = {
            "active_passive": "right",  # right was passive and moved
            "passive_active": "left",
            "mutual": None,
        }.get(evidence.active_mode)
        return AgencyVerdict(
            agency=AGENCY_PEER_DISTURBANCE,
            rationale="passive side moved unplanned (EXTERNAL_ONLY evidence)",
            learnable=True,
            failure_owner=owner,
            candidate_side=owner,
        )

    if not evidence.contact_happened:
        return AgencyVerdict(
            agency=AGENCY_UNKNOWN,
            rationale="no contact to attribute",
            learnable=False,
            failure_owner=None,
            candidate_side=None,
        )

    # MUTUAL: both approached per plan.
    if evidence.active_mode == "mutual" and evidence.left_approached and evidence.right_approached:
        return AgencyVerdict(
            agency=AGENCY_MUTUAL,
            rationale="both approached per plan; contact in the expected window",
            learnable=True,
            failure_owner=None,
            candidate_side=None,  # coordination candidates, not a side
        )

    # Single-side self-caused.
    if evidence.active_mode == "active_passive" and evidence.left_approached:
        return AgencyVerdict(
            agency=AGENCY_LEFT,
            rationale="left actively approached; right held",
            learnable=True,
            failure_owner=None,
            candidate_side="left",
        )
    if evidence.active_mode == "passive_active" and evidence.right_approached:
        return AgencyVerdict(
            agency=AGENCY_RIGHT,
            rationale="right actively approached; left held",
            learnable=True,
            failure_owner=None,
            candidate_side="right",
        )

    return AgencyVerdict(
        agency=AGENCY_UNKNOWN,
        rationale=f"mode {evidence.active_mode} with approach evidence "
        f"L={evidence.left_approached} R={evidence.right_approached} does not resolve",
        learnable=False,
        failure_owner=None,
        candidate_side=None,
    )
