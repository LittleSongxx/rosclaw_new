"""PR-TT-4 Contact Supervisor tests — scripted observation streams
driving the state machine through the happy path and every anomaly
into the recovery spine (STOP_APPROACH → RETREAT_BOTH → VERIFY_CLEAR
→ RECORD_FAILURE).  The other hand never continues after an anomaly."""

from __future__ import annotations

from rosclaw.twintouch.supervisor import (
    CLEARANCE_VERIFIED,
    COARSE_APPROACH,
    CONTACT_CANDIDATE,
    CONTACT_CONFIRMED,
    DECISION_FAIL,
    DECISION_ISSUE_STEP,
    DECISION_NONE,
    DECISION_RETREAT,
    DWELL,
    EPISODE_COMMITTED,
    FINE_APPROACH,
    PAIR_SELECTED,
    PEER_READY,
    RECORD_FAILURE,
    RELEASE,
    RETREAT_BOTH,
    SAFE_RESET,
    STOP_APPROACH,
    VERIFY_CLEAR,
    VISUAL_ALIGN,
    ContactSupervisor,
    ForceBaseline,
    HandObservation,
    SupervisorObservation,
    VisualObservation,
)

JOINTS = ("little", "ring", "middle", "index", "thumb", "thumb_rot")


def _baselines() -> dict[str, ForceBaseline]:
    samples = [dict.fromkeys(JOINTS, 0.0) for _ in range(6)]
    return {
        "left": ForceBaseline.capture("left", samples),
        "right": ForceBaseline.capture("right", samples),
    }


def _hand(
    *,
    force: dict[str, float | None] | None = None,
    angle: dict[str, int | None] | None = None,
    ok: bool = True,
    temp: float = 40.0,
) -> HandObservation:
    return HandObservation(
        ok=ok,
        angle_actual=angle or dict.fromkeys(JOINTS, 1000),
        force_act=force if force is not None else dict.fromkeys(JOINTS, 0.0),
        temperature_max_c=temp,
    )


def _visual(
    dist: float | None = 0.05,
    *,
    age_ms: float = 30.0,
    clusters_ok: bool = True,
    pair_ok: bool | None = True,
) -> VisualObservation:
    return VisualObservation(
        age_ms=age_ms,
        left_cluster_ok=clusters_ok,
        right_cluster_ok=clusters_ok,
        min_distance_m=dist,
        pair_identity_confirmed=pair_ok,
    )


MISSING = object()  # sentinel: a channel that is absent, not defaulted


def _obs(ts: float, left=MISSING, right=MISSING, visual=MISSING) -> SupervisorObservation:
    return SupervisorObservation(
        ts_s=ts,
        left=_hand() if left is MISSING else left,
        right=_hand() if right is MISSING else right,
        visual=_visual() if visual is MISSING else visual,
    )


def _supervisor(mode: str = "mutual", calibrated: bool = True) -> ContactSupervisor:
    return ContactSupervisor(
        interaction_id="int_test",
        pair_id="index_index",
        active_mode=mode,
        baselines=_baselines(),
        reachability_calibrated=calibrated,
    )


def _drive_to_coarse(sup: ContactSupervisor, ts: float = 0.0) -> float:
    assert sup.step(_obs(ts)).kind == DECISION_NONE  # SAFE_RESET -> PAIR_SELECTED
    ts += 0.1
    sup.step(_obs(ts))  # PAIR_SELECTED -> PEER_READY
    ts += 0.1
    sup.step(_obs(ts))  # PEER_READY -> VISUAL_ALIGN
    ts += 0.1
    sup.step(_obs(ts))  # VISUAL_ALIGN -> COARSE_APPROACH
    ts += 0.1
    assert sup.state == COARSE_APPROACH
    return ts


def _drive_recovery(sup: ContactSupervisor, ts: float) -> list[str]:
    """Run the recovery spine to RECORD_FAILURE; return decisions' kinds."""
    kinds: list[str] = []
    for _ in range(6):
        ts += 0.1
        decision = sup.step(_obs(ts))
        kinds.append(decision.kind)
        if sup.state == RECORD_FAILURE:
            return kinds
    raise AssertionError(f"recovery did not terminate: {sup.history}")


# -------------------------------------------------------------- happy path


def test_happy_path_full_cycle_and_receipt():
    sup = _supervisor(mode="mutual")
    ts = _drive_to_coarse(sup)
    assert sup.history == [SAFE_RESET, PAIR_SELECTED, PEER_READY, VISUAL_ALIGN, COARSE_APPROACH]

    # coarse: two steps, then fine zone
    decision = sup.step(_obs(ts, visual=_visual(0.05)))
    assert decision.kind == DECISION_ISSUE_STEP
    assert decision.step["joints"] == {"index": -40}
    ts += 0.1
    decision = sup.step(_obs(ts, visual=_visual(0.015)))
    assert sup.state == FINE_APPROACH
    ts += 0.1
    decision = sup.step(_obs(ts, visual=_visual(0.012)))
    assert decision.kind == DECISION_ISSUE_STEP
    assert decision.step["joints"] == {"index": -10}
    ts += 0.1

    # first force rise on the left -> candidate
    left_rising = _hand(force={**dict.fromkeys(JOINTS, 0.0), "index": 70.0})
    sup.step(_obs(ts, left=left_rising, visual=_visual(0.006)))
    assert sup.state == CONTACT_CANDIDATE
    ts += 0.1
    # bilateral rise + visual near -> confirmed
    both_rising = _hand(force={**dict.fromkeys(JOINTS, 0.0), "index": 72.0})
    decision = sup.step(_obs(ts, left=both_rising, right=both_rising, visual=_visual(0.004)))
    assert sup.state == CONTACT_CONFIRMED
    ts += 0.1
    sup.step(_obs(ts, left=both_rising, right=both_rising, visual=_visual(0.004)))
    assert sup.state == DWELL
    ts += 0.35  # dwell_ms = 300
    sup.step(_obs(ts, left=both_rising, right=both_rising, visual=_visual(0.004)))
    assert sup.state == RELEASE
    ts += 0.1
    # forces back to baseline -> clearance
    sup.step(_obs(ts, visual=_visual(0.004)))
    assert sup.state == CLEARANCE_VERIFIED
    ts += 0.1
    # visual clearance (tips apart) -> committed with receipt
    decision = sup.step(_obs(ts, visual=_visual(0.03)))
    assert sup.state == EPISODE_COMMITTED
    assert decision.kind == "COMMIT"
    receipt = decision.receipt
    assert receipt is not None
    assert receipt.contact_confirmed is True
    assert receipt.outcome == "CONTACT_CONFIRMED"
    assert receipt.clearance_verified is True
    assert receipt.left_force_peak == 72.0
    assert receipt.right_force_peak == 72.0
    assert receipt.visual_distance_min_m == 0.004
    assert receipt.contact_latency_ms is not None
    assert 300.0 <= receipt.dwell_ms < 400.0
    assert receipt.validate() == []


def test_motion_response_can_substitute_visual_near():
    """§6.4: bilateral force + (visual near OR motion response).  When
    the camera loses the pair (depth boundary), bilateral rise +
    position saturation still confirms."""
    sup = _supervisor(mode="active_passive")  # left approaches
    ts = _drive_to_coarse(sup)
    # saturate: angle stops changing while fine steps continue
    left_angle = {"index": 500, **{j: 1000 for j in JOINTS if j != "index"}}
    sup.step(_obs(ts, visual=_visual(0.015)))
    ts += 0.1
    for _ in range(4):  # seed + accumulate saturation frames
        sup.step(_obs(ts, left=_hand(angle=left_angle), visual=_visual(None)))
        ts += 0.1
    assert sup.track.saturated_frames >= 2
    left_rising = _hand(angle=left_angle, force={**dict.fromkeys(JOINTS, 0.0), "index": 70.0})
    sup.step(_obs(ts, left=left_rising, visual=_visual(None)))
    ts += 0.1
    both = {
        "left": left_rising,
        "right": _hand(force={**dict.fromkeys(JOINTS, 0.0), "index": 71.0}),
    }
    sup.step(_obs(ts, left=both["left"], right=both["right"], visual=_visual(None)))
    assert sup.state == CONTACT_CONFIRMED


# ---------------------------------------------------------------- anomalies


def test_uncalibrated_pair_is_never_approached():
    sup = _supervisor(calibrated=False)
    sup.step(_obs(0.0))  # SAFE_RESET -> PAIR_SELECTED
    sup.step(_obs(0.1))  # PAIR_SELECTED -> recovery
    assert sup.state == STOP_APPROACH
    assert sup.track.anomaly == "NO_CONTACT"
    kinds = _drive_recovery(sup, 0.1)
    assert DECISION_ISSUE_STEP not in kinds  # the hands never approached


def test_non_target_finger_force_aborts_as_wrong_finger():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    bad = _hand(force={**dict.fromkeys(JOINTS, 0.0), "middle": 80.0})  # middle is a contact finger
    sup.step(_obs(ts, left=bad))
    assert sup.state == STOP_APPROACH
    assert sup.track.anomaly == "WRONG_FINGER_CONTACT"


def test_thumb_rot_force_aborts_as_unintended():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    bad = _hand(force={**dict.fromkeys(JOINTS, 0.0), "thumb_rot": 80.0})
    sup.step(_obs(ts, right=bad))
    assert sup.state == STOP_APPROACH
    assert sup.track.anomaly == "UNINTENDED_CONTACT"


def test_early_contact_during_coarse():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    rising = _hand(force={**dict.fromkeys(JOINTS, 0.0), "index": 75.0})
    sup.step(_obs(ts, left=rising))
    assert sup.track.anomaly == "EARLY_CONTACT"


def test_no_contact_when_budget_exhausted():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    for _ in range(6):  # max_coarse_steps
        decision = sup.step(_obs(ts, visual=_visual(0.05)))
        assert decision.kind == DECISION_ISSUE_STEP
        ts += 0.1
    sup.step(_obs(ts, visual=_visual(0.05)))
    assert sup.track.anomaly == "NO_CONTACT"


def test_one_sided_force_budget():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    sup.step(_obs(ts, visual=_visual(0.015)))  # -> FINE
    ts += 0.1
    left_rising = _hand(force={**dict.fromkeys(JOINTS, 0.0), "index": 70.0})
    sup.step(_obs(ts, left=left_rising, visual=_visual(0.006)))  # -> CANDIDATE
    ts += 0.1
    for _ in range(5):  # one_sided_frame_budget
        sup.step(_obs(ts, left=left_rising, visual=_visual(0.006)))
        ts += 0.1
    assert sup.track.anomaly == "ONE_SIDED_FORCE"


def test_visual_force_conflict():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    sup.step(_obs(ts, visual=_visual(0.015)))  # -> FINE
    ts += 0.1
    left_rising = _hand(force={**dict.fromkeys(JOINTS, 0.0), "index": 70.0})
    sup.step(_obs(ts, left=left_rising, visual=_visual(0.04)))  # -> CANDIDATE
    ts += 0.1
    # forces rise but camera says 6cm apart — conflict
    both_rising = _hand(force={**dict.fromkeys(JOINTS, 0.0), "index": 71.0})
    sup.step(_obs(ts, left=both_rising, right=both_rising, visual=_visual(0.06)))
    assert sup.track.anomaly == "VISUAL_FORCE_CONFLICT"


def test_stale_camera_blocks_approach():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    sup.step(_obs(ts, visual=_visual(0.05, age_ms=900.0)))
    assert sup.track.anomaly == "STALE_OBSERVATION"


def test_transport_loss_retreats_both():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    sup.step(_obs(ts, right=None))
    assert sup.track.anomaly == "TRANSPORT_FAILURE"
    kinds = _drive_recovery(sup, ts)
    assert DECISION_RETREAT in kinds
    assert DECISION_ISSUE_STEP not in kinds


def test_thermal_abort_mid_approach():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    sup.step(_obs(ts, left=_hand(temp=50.0)))
    assert sup.track.anomaly == "THERMAL_ABORT"


def test_peer_not_ready_when_force_channels_dead():
    sup = _supervisor()
    sup.step(_obs(0.0))
    sup.step(_obs(0.1))
    dead_right = _hand(force=dict.fromkeys(JOINTS))
    sup.step(_obs(0.2, right=dead_right))
    assert sup.track.anomaly == "PEER_NOT_READY"


def test_wrong_pair_identity_from_camera():
    sup = _supervisor()
    sup.step(_obs(0.0))
    sup.step(_obs(0.1))
    sup.step(_obs(0.2))
    sup.step(_obs(0.3, visual=_visual(0.05, pair_ok=False)))
    assert sup.track.anomaly == "WRONG_FINGER_CONTACT"


def test_release_failed_when_forces_stay_high():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    # drive to confirmed quickly
    sup.step(_obs(ts, visual=_visual(0.015)))
    ts += 0.1
    both = _hand(force={**dict.fromkeys(JOINTS, 0.0), "index": 70.0})
    sup.step(_obs(ts, left=both, right=both, visual=_visual(0.005)))
    ts += 0.1
    sup.step(_obs(ts, left=both, right=both, visual=_visual(0.004)))
    assert sup.state == CONTACT_CONFIRMED
    ts += 0.1
    sup.step(_obs(ts, left=both, right=both, visual=_visual(0.004)))
    ts += 0.35
    sup.step(_obs(ts, left=both, right=both, visual=_visual(0.004)))
    assert sup.state == RELEASE
    ts += 0.1
    for _ in range(4):  # max_release_steps: forces stay high
        sup.step(_obs(ts, left=both, right=both, visual=_visual(0.004)))
        ts += 0.1
    sup.step(_obs(ts, left=both, right=both, visual=_visual(0.004)))
    assert sup.track.anomaly == "RELEASE_FAILED"


def test_recovery_spine_shape_and_failure_receipt():
    sup = _supervisor()
    ts = _drive_to_coarse(sup)
    bad = _hand(force={**dict.fromkeys(JOINTS, 0.0), "middle": 90.0})
    sup.step(_obs(ts, left=bad))
    assert sup.history[-1] == STOP_APPROACH
    decision = sup.step(_obs(ts + 0.1))
    assert sup.state == RETREAT_BOTH
    assert decision.kind == DECISION_RETREAT  # both hands retreat
    decision = sup.step(_obs(ts + 0.2))
    assert sup.state == VERIFY_CLEAR
    decision = sup.step(_obs(ts + 0.3))
    assert sup.state == RECORD_FAILURE
    assert decision.kind == DECISION_FAIL
    receipt = decision.receipt
    assert receipt is not None
    assert receipt.outcome == "WRONG_FINGER_CONTACT"
    assert receipt.wrong_finger_contact is True
    assert receipt.contact_confirmed is False
    assert receipt.validate() == []
    # the full spine is in the history, in order
    assert sup.history[-4:] == [STOP_APPROACH, RETREAT_BOTH, VERIFY_CLEAR, RECORD_FAILURE]


def test_safe_reset_retreats_when_hand_not_open():
    sup = _supervisor()
    curled = _hand(angle={"index": 300, **{j: 1000 for j in JOINTS if j != "index"}})
    decision = sup.step(_obs(0.0, left=curled))
    assert decision.kind == DECISION_RETREAT
    assert sup.state == RETREAT_BOTH  # posture correction, not an anomaly
    assert sup.track.anomaly is None
