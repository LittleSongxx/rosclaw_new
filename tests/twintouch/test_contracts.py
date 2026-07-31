"""PR-TT-1 contract tests: contact topology, envelope, receipt,
choreography contract + permit, config evolution bounds."""

from __future__ import annotations

import time

import pytest

from rosclaw.twintouch import (
    CANONICAL_MARQUEE_PAIRS,
    FORBIDDEN_FINGERTIP_PAIRS,
    OUTCOME_CONTACT_CONFIRMED,
    OUTCOME_NO_CONTACT,
    OUTCOME_PARTIAL_DISPATCH,
    OUTCOME_UNINTENDED_CONTACT,
    OUTCOME_WRONG_FINGER_CONTACT,
    PERMIT_ALREADY_USED,
    PERMIT_EXPIRED,
    PERMIT_HASH_MISMATCH,
    PERMIT_OK,
    REACHABILITY_CALIBRATED,
    REACHABILITY_MUTUAL_CURL_ONLY,
    VALID_PAIR_IDS,
    BimanualActionEnvelope,
    BodyActionBlock,
    ContactChoreographyContract,
    CoordinationBlock,
    FingerPairReachabilityMatrix,
    ForbiddenCollisionMap,
    InteractionReceipt,
    SafetyBlock,
    SequencePermit,
    TwinTouchConfig,
    TwinTouchPhysicalLayout,
    is_valid_pair_id,
    pair_by_id,
    validate_candidate_changes,
)
from rosclaw.twintouch.receipt import _all_outcomes

# ---------------------------------------------------------------- pairs


def test_five_same_finger_pairs_are_the_only_valid_pairs():
    assert {
        "thumb_thumb",
        "index_index",
        "middle_middle",
        "ring_ring",
        "little_little",
    } == VALID_PAIR_IDS
    for pair_id in VALID_PAIR_IDS:
        assert pair_by_id(pair_id).validate() == []


def test_every_cross_finger_combination_is_forbidden():
    # 5x5 minus 5 same = 20 forbidden cross pairs
    assert len(FORBIDDEN_FINGERTIP_PAIRS) == 20
    assert "thumb_index" in FORBIDDEN_FINGERTIP_PAIRS
    assert "index_thumb" in FORBIDDEN_FINGERTIP_PAIRS
    assert "thumb_rot" not in " ".join(FORBIDDEN_FINGERTIP_PAIRS)  # not a fingertip
    assert not is_valid_pair_id("thumb_index")
    assert not is_valid_pair_id("thumbrot_thumbrot")


def test_collision_map_rejects_cross_and_unknown_pairs():
    cmap = ForbiddenCollisionMap()
    assert cmap.validate_action_pairing("thumb_index") != []
    assert cmap.validate_action_pairing("nonsense") != []
    assert cmap.validate_action_pairing("thumb_thumb") == []
    assert cmap.validate_action_pairing("thumb_thumb", active_pair_count=2) != []


def test_layout_records_t0_measurement_honestly():
    layout = TwinTouchPhysicalLayout(
        left_body_id="rh56_left_01",
        right_body_id="rh56_right_01",
        camera_id="d435i",
        camera_pose_hash="pose_abc",
        palms_facing_each_other=True,
        open_pose_lateral_gap_m=0.0448,
        mutual_reach_proven=True,
        mount_separation_m=None,  # not measured
        measured_at="2026-07-31T08:35:36Z",
        evidence_refs=("artifact://twintouch/t0/20260731T083536Z",),
    )
    # mount_separation unmeasured is a disclosed gap, not a fabricated constant
    assert layout.validate() == ["layout property mount_separation_m is unmeasured"]
    roundtrip = TwinTouchPhysicalLayout.from_record(layout.to_record())
    assert roundtrip.open_pose_lateral_gap_m == pytest.approx(0.0448)


def test_t0_reachability_matrix_claims_mutual_curl_only_not_calibrated():
    matrix = FingerPairReachabilityMatrix.from_t0_measurement(
        "artifact://twintouch/t0/20260731T083536Z"
    )
    assert matrix.validate() == []
    assert matrix.calibrated_pairs() == []
    assert set(matrix.states.values()) == {REACHABILITY_MUTUAL_CURL_ONLY}
    # claiming calibrated without per-pair evidence is a violation
    dishonest = FingerPairReachabilityMatrix(
        states=dict.fromkeys(VALID_PAIR_IDS, REACHABILITY_CALIBRATED)
    )
    assert any("without evidence" in v for v in dishonest.validate())


# ------------------------------------------------------------- envelope


def _body_block(body_id: str) -> BodyActionBlock:
    return BodyActionBlock(
        body_id=body_id,
        action={"gesture": "approach", "targets": {"index": 620}},
        body_snapshot_hash="snap_x",
        calibration_hash="cal_x",
    )


def _envelope(pair_id: str = "index_index") -> BimanualActionEnvelope:
    return BimanualActionEnvelope(
        interaction_id="int_1",
        sequence_id="seq_1",
        pair_id=pair_id,
        left=_body_block("rh56_left_01"),
        right=_body_block("rh56_right_01"),
        coordination=CoordinationBlock(
            mode="mutual",
            synchronization_barrier="start_together",
            maximum_start_skew_ms=250.0,
            timeout_ms=2000.0,
        ),
        safety=SafetyBlock(
            contract_hash="chor_abc",
            permitted_contact_pair=pair_id,
            forbidden_contact_pairs=tuple(sorted(FORBIDDEN_FINGERTIP_PAIRS)),
            retreat_action={"gesture": "safe_open", "speed": 300, "force": 150},
        ),
    )


def test_valid_envelope_passes_and_hashes_stably():
    env = _envelope()
    assert env.validate() == []
    assert env.envelope_hash() == env.envelope_hash()
    assert env.envelope_hash().startswith("bimact_")
    roundtrip = BimanualActionEnvelope.from_record(env.to_record())
    assert roundtrip.envelope_hash() == env.envelope_hash()


def test_envelope_rejects_safety_pair_mismatch():
    env = _envelope("index_index")
    bad = BimanualActionEnvelope(
        **{**env.__dict__, "pair_id": "thumb_thumb"}
    )  # safety still permits index_index
    assert any("permits" in v for v in bad.validate())


def test_envelope_rejects_same_body_both_sides():
    env = BimanualActionEnvelope(
        interaction_id="int_1",
        sequence_id="seq_1",
        pair_id="index_index",
        left=_body_block("rh56_left_01"),
        right=_body_block("rh56_left_01"),
        coordination=_envelope().coordination,
        safety=_envelope().safety,
    )
    assert any("distinct bodies" in v for v in env.validate())


def test_envelope_rejects_missing_snapshot_hash_and_bad_mode():
    env = BimanualActionEnvelope(
        interaction_id="int_1",
        sequence_id="seq_1",
        pair_id="index_index",
        left=BodyActionBlock("rh56_left_01", {"gesture": "hold"}, None, "cal_x"),
        right=_body_block("rh56_right_01"),
        coordination=CoordinationBlock("magic", "", -1.0, 0.0),
        safety=_envelope().safety,
    )
    violations = env.validate()
    assert any("body_snapshot_hash" in v for v in violations)
    assert any("coordination mode" in v for v in violations)
    assert any("synchronization_barrier" in v for v in violations)


# -------------------------------------------------------------- receipt


def _receipt(**overrides) -> InteractionReceipt:
    base = {
        "interaction_id": "int_1",
        "pair_id": "index_index",
        "left_action_receipt": "arec_l",
        "right_action_receipt": "arec_r",
        "intended_contact": "index_index",
        "observed_contact": "index_index",
        "contact_confirmed": True,
        "wrong_finger_contact": False,
        "unintended_contact": False,
        "left_force_peak": 112.0,
        "right_force_peak": 98.0,
        "visual_distance_min_m": 0.004,
        "contact_latency_ms": 1840.0,
        "start_skew_ms": 95.0,
        "dwell_ms": 300.0,
        "clearance_verified": True,
        "outcome": OUTCOME_CONTACT_CONFIRMED,
        "evidence_refs": ("artifact://x",),
    }
    base.update(overrides)
    return InteractionReceipt(**base)


def test_confirmed_receipt_validates_and_roundtrips():
    receipt = _receipt()
    assert receipt.validate() == []
    again = InteractionReceipt.from_record(receipt.to_record())
    assert again.receipt_hash() == receipt.receipt_hash()


def test_receipt_rejects_self_contradictions():
    assert _receipt(contact_confirmed=False).validate() != []
    assert _receipt(outcome=OUTCOME_NO_CONTACT).validate() != []
    assert _receipt(clearance_verified=False).validate() != []
    assert _receipt(left_force_peak=None, right_force_peak=None).validate() != []
    assert _receipt(intended_contact="thumb_thumb").validate() != []
    wrong = _receipt(
        contact_confirmed=False,
        wrong_finger_contact=True,
        observed_contact="index_middle",
        outcome=OUTCOME_WRONG_FINGER_CONTACT,
        left_force_peak=90.0,
        right_force_peak=None,
        clearance_verified=True,
    )
    assert wrong.validate() == []
    assert (
        _receipt(
            contact_confirmed=False,
            wrong_finger_contact=True,
            outcome=OUTCOME_UNINTENDED_CONTACT,
        ).validate()
        != []
    )  # wrong finger but outcome says unintended


def test_partial_dispatch_receipt_rules():
    partial = _receipt(
        right_action_receipt=None,
        contact_confirmed=False,
        observed_contact=None,
        left_force_peak=None,
        right_force_peak=None,
        visual_distance_min_m=None,
        contact_latency_ms=None,
        dwell_ms=None,
        clearance_verified=False,
        outcome=OUTCOME_PARTIAL_DISPATCH,
    )
    assert partial.validate() == []
    # both sides dispatched contradicts PARTIAL_DISPATCH
    both = _receipt(outcome=OUTCOME_PARTIAL_DISPATCH, contact_confirmed=False)
    assert any("PARTIAL_DISPATCH" in v for v in both.validate())


def test_outcome_set_covers_state_machine_anomalies():
    outcomes = _all_outcomes()
    for expected in (
        "CONTACT_CONFIRMED",
        "NO_CONTACT",
        "EARLY_CONTACT",
        "WRONG_FINGER_CONTACT",
        "UNINTENDED_CONTACT",
        "ONE_SIDED_FORCE",
        "VISUAL_FORCE_CONFLICT",
        "PEER_NOT_READY",
        "STALE_OBSERVATION",
        "TRANSPORT_FAILURE",
        "RELEASE_FAILED",
        "THERMAL_ABORT",
        "PARTIAL_DISPATCH",
    ):
        assert expected in outcomes


# --------------------------------------------------------- choreography


def _contract(**overrides) -> ContactChoreographyContract:
    base = {
        "pattern": "fingertip_marquee",
        "pairs": CANONICAL_MARQUEE_PAIRS,
        "cycles": 2,
        "force_level": "ultra_light",
        "left_body_hash": "body_l",
        "right_body_hash": "body_r",
        "camera_pose_hash": "pose_x",
        "created_at": "2026-07-31T09:00:00Z",
    }
    base.update(overrides)
    return ContactChoreographyContract(**base)


def test_canonical_marquee_contract_validates():
    contract = _contract()
    assert contract.validate() == []
    assert contract.contract_hash().startswith("chor_")
    assert (
        ContactChoreographyContract.from_record(contract.to_record()).contract_hash()
        == contract.contract_hash()
    )


def test_marquee_rejects_custom_sequence_and_phase2_patterns():
    assert _contract(pairs=("thumb_thumb",)).validate() != []
    assert _contract(pattern="echo_touch").validate() != []  # declared but not phase-1
    assert _contract(pattern="unknown_pattern").validate() != []


def test_contact_force_levels_require_t1_envelopes():
    assert _contract(force_level="light").validate() != []  # no envelopes
    with_envelopes = _contract(
        force_level="light",
        pair_envelope_hashes={p: f"env_{p}" for p in set(CANONICAL_MARQUEE_PAIRS)},
    )
    assert with_envelopes.validate() == []


def test_permit_lifecycle():
    contract = _contract()
    now = time.time()
    permit = SequencePermit.issue(contract, intent_hash="intent_x", lifetime_s=60.0, now_s=now)
    assert permit.verify(contract, now_s=now + 1) == PERMIT_OK
    assert permit.verify(contract, now_s=now + 61) == PERMIT_EXPIRED
    other = _contract(cycles=3)
    assert permit.verify(other, now_s=now + 1) == PERMIT_HASH_MISMATCH
    permit.consume(reason="sequence_started")
    assert permit.verify(contract, now_s=now + 1) == PERMIT_ALREADY_USED
    # partial dispatch revokes mid-sequence (v4 §7.3)
    permit2 = SequencePermit.issue(contract, intent_hash="intent_x", lifetime_s=60.0, now_s=now)
    permit2.revoke(reason="partial_dispatch")
    assert permit2.verify(contract, now_s=now + 1) == PERMIT_ALREADY_USED
    assert permit2.to_record()["revoked_reason"] == "partial_dispatch"


# --------------------------------------------------------------- config


def test_default_config_loads_and_validates():
    config = TwinTouchConfig.load()
    assert config.validate() == []
    assert config.max_active_pairs == 1
    assert config.coarse_step_raw == 40
    assert config.fine_step_raw == 10


def test_candidate_changes_gate():
    # whitelisted in-bounds change passes
    assert validate_candidate_changes({"finger_precontact_offset_raw": 4}) == []
    assert validate_candidate_changes({"left_right_start_skew_ms": -60}) == []
    assert validate_candidate_changes({"active_side_selection": "left"}) == []
    # hard limits are never touchable
    assert validate_candidate_changes({"hard_force_limit_raw": 500}) != []
    assert validate_candidate_changes({"temperature_abort_c": 55}) != []
    assert validate_candidate_changes({"servo_max_speed_approach": 300}) != []
    assert validate_candidate_changes({"camera_freshness_ms": 2000}) != []
    # unknown keys rejected (not in §12.1)
    assert validate_candidate_changes({"look_ma_no_hands": 1}) != []
    # out-of-bound whitelisted values rejected
    assert validate_candidate_changes({"finger_precontact_offset_raw": 40}) != []
    assert validate_candidate_changes({"left_right_start_skew_ms": 500}) != []
    assert validate_candidate_changes({"retry_count": 9}) != []
