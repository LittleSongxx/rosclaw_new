"""PR-TT-7 tests: PeerState / InteractionSelfSnapshot / agency attribution."""

from __future__ import annotations

from rosclaw.twintouch.peer import (
    AGENCY_LEFT,
    AGENCY_MUTUAL,
    AGENCY_PEER_DISTURBANCE,
    AGENCY_RIGHT,
    AGENCY_SENSOR_FAULT,
    AGENCY_UNKNOWN,
    AgencyEvidence,
    InteractionSelfSnapshot,
    PeerState,
    attribute_agency,
)


def _ev(**overrides) -> AgencyEvidence:
    base = {
        "active_mode": "mutual",
        "contact_happened": True,
        "left_target_delta": 140.0,
        "right_target_delta": -90.0,
        "left_approached": True,
        "right_approached": True,
        "passive_side_moved_unplanned": False,
        "visual_says_contact": None,
        "visual_unknown": False,
    }
    base.update(overrides)
    return AgencyEvidence(**base)


def test_mutual_contact_attribution():
    verdict = attribute_agency(_ev())
    assert verdict.agency == AGENCY_MUTUAL
    assert verdict.learnable is True
    assert verdict.failure_owner is None


def test_single_side_self_caused():
    left = attribute_agency(_ev(active_mode="active_passive", right_approached=False))
    assert left.agency == AGENCY_LEFT
    assert left.candidate_side == "left"
    right = attribute_agency(
        _ev(active_mode="passive_active", left_approached=False, right_approached=True)
    )
    assert right.agency == AGENCY_RIGHT
    assert right.candidate_side == "right"


def test_peer_disturbance_names_the_passive_side():
    verdict = attribute_agency(
        _ev(active_mode="active_passive", right_approached=False, passive_side_moved_unplanned=True)
    )
    assert verdict.agency == AGENCY_PEER_DISTURBANCE
    assert verdict.failure_owner == "right"
    assert verdict.candidate_side == "right"
    assert verdict.learnable is True


def test_sensor_fault_when_camera_and_forces_disagree():
    verdict = attribute_agency(
        _ev(visual_says_contact=True, left_target_delta=5.0, right_target_delta=-8.0)
    )
    assert verdict.agency == AGENCY_SENSOR_FAULT
    assert verdict.learnable is False  # never enters control learning (v4 §22)


def test_unknown_on_missing_channels():
    assert attribute_agency(_ev(visual_unknown=True)).agency == AGENCY_UNKNOWN
    assert attribute_agency(_ev(left_target_delta=None)).agency == AGENCY_UNKNOWN
    no_contact = attribute_agency(_ev(contact_happened=False))
    assert no_contact.agency == AGENCY_UNKNOWN
    assert no_contact.learnable is False


def test_sensor_fault_check_precedes_unknown():
    """A camera-vs-force disagreement must classify as SENSOR_FAULT even
    with tiny force values — not be swallowed by generic UNKNOWN."""
    verdict = attribute_agency(
        _ev(
            visual_says_contact=True,
            left_target_delta=0.0,
            right_target_delta=0.0,
            visual_unknown=False,
        )
    )
    assert verdict.agency == AGENCY_SENSOR_FAULT


def test_peer_state_and_snapshot_records():
    peer = PeerState(
        peer_body_id="rh56_right_01",
        ready=True,
        intended_finger="index",
        current_finger_state="approaching",
        visual_tip_position=(0.01, -0.02, 0.21),
        approaching=True,
        contact_force=-43.0,
        prediction_uncertainty=None,  # PRIOR until calibrated (v3 §8)
        health="ok",
    )
    record = peer.to_record()
    assert record["schema_version"] == "rosclaw.twintouch_peer.v1"
    assert record["prediction_uncertainty"] is None

    snapshot = InteractionSelfSnapshot(
        left_self_snapshot_hash="snap_l",
        right_self_snapshot_hash="snap_r",
        intended_pair="index_index",
        relative_tip_geometry={"distance_m": 0.048, "axis": "lateral"},
        expected_time_to_contact_ms=600.0,
        expected_contact_force={"left": 100.0, "right": -45.0},
        expected_active_side="mutual",
        current_contact_state="FINE_APPROACH",
        peer_readiness={"left": True, "right": True},
        collision_risk="watch",
    )
    assert snapshot.snapshot_hash() == snapshot.snapshot_hash()
    assert snapshot.snapshot_hash().startswith("isnap_")
    record = snapshot.to_record()
    assert record["schema_version"] == "rosclaw.twintouch_interaction_snapshot.v1"
    assert record["relative_tip_geometry"]["distance_m"] == 0.048
