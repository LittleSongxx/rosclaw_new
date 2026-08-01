"""PR-TT-8 tests: FingerPairMemory scope filters + failure signatures."""

from __future__ import annotations

from rosclaw.twintouch.pair_memory import (
    FAILURE_SIGNATURES,
    FingerPairMemory,
    MemoryScope,
    SuccessfulEnvelope,
    failure_signature_for,
    filter_pair_memories,
)


def _scope(**overrides) -> MemoryScope:
    base = {
        "left_body_hash": "body_l",
        "right_body_hash": "body_r",
        "pair_id": "index_index",
        "interaction_mode": "mutual",
        "camera_pose_hash": "pose_x",
        "temperature_regime": "warm_stable",
    }
    base.update(overrides)
    return MemoryScope(**base)


def _memory(scope=None) -> FingerPairMemory:
    return FingerPairMemory(
        scope=scope or _scope(),
        successful_envelope=SuccessfulEnvelope(
            precontact={"left": 680, "right": 680},
            contact_range={"left": (690, 700), "right": (690, 700)},
            visual_near_range_m=(0.047, 0.050),
            force_baseline={"left_index": -54.0, "right_index": -41.0},
            contact_force_envelope={"left": (107.0, 145.0), "right": (178.0, 183.0)},
            release_margin_raw=60,
            evidence_count=2,
        ),
        failure_signatures={"CONTACT_FORCE_ASYMMETRY": 1},
        recovery_hint="rebaseline_force_before_retry",
        evidence_refs=("artifact://twintouch/t1/20260731T155027Z",),
    )


def test_memory_roundtrip_and_hash():
    memory = _memory()
    assert memory.validate() == []
    again = FingerPairMemory.from_record(memory.to_record())
    assert again.memory_hash() == memory.memory_hash()


def test_single_session_envelope_never_validates():
    memory = _memory()
    thin = FingerPairMemory(
        scope=memory.scope,
        successful_envelope=SuccessfulEnvelope(
            precontact={},
            contact_range={},
            visual_near_range_m=(0.0, 0.0),
            force_baseline={},
            contact_force_envelope={},
            release_margin_raw=0,
            evidence_count=1,
        ),
        failure_signatures={},
        recovery_hint=None,
    )
    assert any("single session" in v for v in thin.validate())


def test_unknown_signature_rejected():
    memory = FingerPairMemory(
        scope=_scope(),
        successful_envelope=None,
        failure_signatures={"MADE_UP_FAILURE": 1},
        recovery_hint=None,
    )
    assert any("unknown failure signatures" in v for v in memory.validate())


def test_hard_filter_pair_and_mode_and_camera():
    index_memory = _memory()
    thumb_memory = _memory(scope=_scope(pair_id="thumb_thumb"))
    left_active_memory = _memory(scope=_scope(interaction_mode="active_passive"))
    other_camera = _memory(scope=_scope(camera_pose_hash="pose_moved"))
    pool = [index_memory, thumb_memory, left_active_memory, other_camera]

    hits = filter_pair_memories(pool, _scope())
    assert hits == [index_memory]  # mutual index on this camera only

    # thumb_thumb memory never applies to index_index
    assert thumb_memory not in filter_pair_memories(pool, _scope())
    # left-active memory never silently applies to mutual
    assert left_active_memory not in filter_pair_memories(pool, _scope())
    # moved camera hides everything
    assert filter_pair_memories(pool, _scope(camera_pose_hash="pose_moved")) == [other_camera]


def test_regime_scoping():
    warm = _memory()
    cold = _memory(scope=_scope(temperature_regime="cold"))
    pool = [warm, cold]
    # unscoped query matches both
    assert sorted(
        id(m) for m in filter_pair_memories(pool, _scope(temperature_regime=None))
    ) == sorted((id(warm), id(cold)))
    # warm query matches only warm
    assert filter_pair_memories(pool, _scope(temperature_regime="warm_stable")) == [warm]
    # cold query matches only cold
    assert filter_pair_memories(pool, _scope(temperature_regime="cold")) == [cold]


def test_failure_signature_mapping():
    assert (
        failure_signature_for(outcome="NO_CONTACT", active_mode="active_passive")
        == "NO_CONTACT_LEFT_SHORT"
    )
    assert (
        failure_signature_for(outcome="NO_CONTACT", active_mode="passive_active")
        == "NO_CONTACT_RIGHT_SHORT"
    )
    assert (
        failure_signature_for(outcome="NO_CONTACT", active_mode="mutual") is None
    )  # agency unresolved
    assert (
        failure_signature_for(outcome="EARLY_CONTACT", active_mode="passive_active")
        == "RIGHT_EARLY_CONTACT"
    )
    assert (
        failure_signature_for(outcome="ONE_SIDED_FORCE", active_mode="mutual")
        == "CONTACT_FORCE_ASYMMETRY"
    )
    assert (
        failure_signature_for(outcome="UNINTENDED_CONTACT", active_mode="mutual")
        == "UNINTENDED_NEIGHBOR_CONTACT"
    )
    assert (
        failure_signature_for(outcome="VISUAL_FORCE_CONFLICT", active_mode="mutual")
        == "VISUAL_FORCE_CONFLICT"
    )
    assert (
        failure_signature_for(outcome="RELEASE_FAILED", active_mode="mutual")
        == "RELEASE_MARGIN_INSUFFICIENT"
    )
    # environment outcomes are not pair residuals
    assert failure_signature_for(outcome="CONTACT_CONFIRMED", active_mode="mutual") is None
    assert failure_signature_for(outcome="TRANSPORT_FAILURE", active_mode="mutual") is None
    assert failure_signature_for(outcome="THERMAL_ABORT", active_mode="mutual") is None
    # every mapped signature is in the §13.1 vocabulary
    for outcome in (
        "NO_CONTACT",
        "EARLY_CONTACT",
        "ONE_SIDED_FORCE",
        "WRONG_FINGER_CONTACT",
        "UNINTENDED_CONTACT",
        "VISUAL_FORCE_CONFLICT",
        "RELEASE_FAILED",
    ):
        for mode in ("active_passive", "passive_active", "mutual"):
            sig = failure_signature_for(outcome=outcome, active_mode=mode)
            assert sig is None or sig in FAILURE_SIGNATURES
