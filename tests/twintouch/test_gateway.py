"""PR-TT-2 gateway tests: atomic lease, barrier, partial-dispatch
retreat, permit lifecycle, E-Stop fan-out — all in-memory."""

from __future__ import annotations

import time

from rosclaw.twintouch import (
    CANONICAL_MARQUEE_PAIRS,
    FORBIDDEN_FINGERTIP_PAIRS,
    OUTCOME_ABORTED_BEFORE_DISPATCH,
    OUTCOME_DISPATCHED,
    OUTCOME_PARTIAL_DISPATCH,
    BimanualActionEnvelope,
    BimanualActionGateway,
    BodyActionBlock,
    ContactChoreographyContract,
    CoordinationBlock,
    LeaseRegistry,
    SafetyBlock,
    SequencePermit,
)
from rosclaw.twintouch.gateway import (
    DISPATCH_SKEW_VIOLATION,
    ENVELOPE_CONTRACT_MISMATCH,
    ENVELOPE_INVALID,
    EXECUTOR_FAILURE,
    LEASE_CONFLICT,
    PERMIT_INVALID,
    PRECONDITION_FAILED,
)


class FakeExecutor:
    """Configurable in-memory executor: failure mode, per-call delay,
    and a full call log for assertions."""

    def __init__(self, name: str, *, fail_on: set[str] | None = None, delay_s: float = 0.0):
        self.name = name
        self.fail_on = fail_on or set()
        self.delay_s = delay_s
        self.calls: list[str] = []

    def dispatch(self, action, *, timeout_ms: float) -> str:
        self.calls.append(f"dispatch:{action.get('gesture')}")
        if self.delay_s:
            time.sleep(self.delay_s)
        if "dispatch" in self.fail_on:
            raise RuntimeError(f"{self.name} transport dead")
        return f"arec_{self.name}_{len(self.calls)}"

    def retreat(self, retreat_action) -> bool:
        self.calls.append(f"retreat:{retreat_action.get('gesture')}")
        return "retreat" not in self.fail_on

    def estop(self) -> bool:
        self.calls.append("estop")
        return "estop" not in self.fail_on


class FakeProbe:
    def __init__(self, *, camera_fresh: bool = True, snapshot_violations: list[str] | None = None):
        self._camera_fresh = camera_fresh
        self._snapshot_violations = snapshot_violations or []

    def camera_fresh(self, *, max_age_ms: float) -> bool:
        return self._camera_fresh

    def snapshots_valid(self, envelope) -> list[str]:
        return list(self._snapshot_violations)


def _envelope(
    pair_id: str = "index_index", contract: ContactChoreographyContract | None = None
) -> BimanualActionEnvelope:
    # The envelope's safety block must name the contract that authorizes
    # it — the gateway refuses a stale/foreign binding before leasing.
    contract = contract or _contract()
    return BimanualActionEnvelope(
        interaction_id="int_1",
        sequence_id="seq_1",
        pair_id=pair_id,
        left=BodyActionBlock("rh56_left_01", {"gesture": "approach"}, "snap_l", "cal_l"),
        right=BodyActionBlock("rh56_right_01", {"gesture": "hold"}, "snap_r", "cal_r"),
        coordination=CoordinationBlock("mutual", "start_together", 250.0, 2000.0),
        safety=SafetyBlock(
            contract_hash=contract.contract_hash(),
            permitted_contact_pair=pair_id,
            forbidden_contact_pairs=tuple(sorted(FORBIDDEN_FINGERTIP_PAIRS)),
            retreat_action={"gesture": "safe_open"},
        ),
    )


def _contract() -> ContactChoreographyContract:
    return ContactChoreographyContract(
        pattern="fingertip_marquee",
        pairs=CANONICAL_MARQUEE_PAIRS,
        cycles=1,
        force_level="ultra_light",
        left_body_hash="body_l",
        right_body_hash="body_r",
        camera_pose_hash="pose_x",
        created_at="2026-07-31T09:00:00Z",
    )


def _gateway(
    left: FakeExecutor | None = None,
    right: FakeExecutor | None = None,
    probe: FakeProbe | None = None,
) -> tuple[BimanualActionGateway, LeaseRegistry, FakeExecutor, FakeExecutor]:
    left = left or FakeExecutor("left")
    right = right or FakeExecutor("right")
    leases = LeaseRegistry({"left": "rh56_left_01", "right": "rh56_right_01"})
    gateway = BimanualActionGateway(
        executors={"left": left, "right": right},
        leases=leases,
        probe=probe or FakeProbe(),
    )
    return gateway, leases, left, right


def _permit(contract: ContactChoreographyContract, lifetime_s: float = 60.0) -> SequencePermit:
    return SequencePermit.issue(contract, intent_hash="intent_x", lifetime_s=lifetime_s)


# ------------------------------------------------------------------ happy


def test_happy_path_dispatches_both_and_consumes_permit():
    gateway, leases, left, right = _gateway()
    contract = _contract()
    report = gateway.dispatch(_envelope(), contract=contract, permit=_permit(contract))
    assert report.violation_kind is None
    assert report.left_dispatched and report.right_dispatched
    assert report.receipt.outcome == OUTCOME_DISPATCHED
    assert report.receipt.validate() == []
    assert report.receipt.start_skew_ms is not None
    assert left.calls == ["dispatch:approach"]
    assert right.calls == ["dispatch:hold"]
    assert leases.holders() == {"left": None, "right": None}  # released after


# ------------------------------------------------------- pre-dispatch gates


def test_invalid_envelope_aborts_with_zero_executor_calls():
    gateway, _, left, right = _gateway()
    contract = _contract()
    bad = BimanualActionEnvelope(
        **{**_envelope().__dict__, "pair_id": "thumb_thumb"}  # safety mismatch
    )
    report = gateway.dispatch(bad, contract=contract, permit=_permit(contract))
    assert report.violation_kind == ENVELOPE_INVALID
    assert left.calls == [] and right.calls == []


def test_expired_permit_aborts_with_zero_executor_calls():
    gateway, _, left, right = _gateway()
    contract = _contract()
    permit = SequencePermit.issue(contract, intent_hash="intent_x", lifetime_s=-1.0)
    report = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert report.violation_kind == PERMIT_INVALID
    assert report.receipt.outcome == OUTCOME_ABORTED_BEFORE_DISPATCH
    assert left.calls == [] and right.calls == []


def test_permit_contract_mismatch_aborts():
    gateway, _, _, _ = _gateway()
    contract = _contract()
    other = ContactChoreographyContract(**{**contract.__dict__, "cycles": 2})
    permit = _permit(other)  # bound to a DIFFERENT contract hash
    report = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert report.violation_kind == PERMIT_INVALID


def test_forbidden_pair_never_reaches_executors():
    gateway, _, left, right = _gateway()
    contract = _contract()
    env = _envelope("thumb_index")
    # envelope-level pair validation fires first (invalid pair), but even a
    # structurally valid envelope over a forbidden pairing is stopped here
    report = gateway.dispatch(env, contract=contract, permit=_permit(contract))
    assert report.violation_kind in (ENVELOPE_INVALID, PRECONDITION_FAILED)
    assert left.calls == [] and right.calls == []


def test_stale_camera_blocks_dispatch():
    gateway, _, left, right = _gateway(probe=FakeProbe(camera_fresh=False))
    contract = _contract()
    report = gateway.dispatch(_envelope(), contract=contract, permit=_permit(contract))
    assert report.violation_kind == PRECONDITION_FAILED
    assert any("camera" in v for v in report.violations)
    assert left.calls == [] and right.calls == []


def test_invalid_snapshots_block_dispatch():
    gateway, _, left, right = _gateway(
        probe=FakeProbe(snapshot_violations=["left body snapshot hash mismatch"])
    )
    contract = _contract()
    report = gateway.dispatch(_envelope(), contract=contract, permit=_permit(contract))
    assert report.violation_kind == PRECONDITION_FAILED
    assert left.calls == [] and right.calls == []


def test_lease_conflict_aborts_and_discloses_side():
    gateway, leases, left, right = _gateway()
    # pre-hold both leases: fixed-order acquisition checks left first,
    # so the disclosed conflict side is left (the first unavailable).
    leases.acquire_all("someone_else")
    contract = _contract()
    report = gateway.dispatch(_envelope(), contract=contract, permit=_permit(contract))
    assert report.violation_kind == LEASE_CONFLICT
    assert any("left" in v for v in report.violations)
    assert left.calls == [] and right.calls == []
    leases.release_all("someone_else")
    # after release the same envelope dispatches cleanly
    report2 = gateway.dispatch(_envelope(), contract=contract, permit=_permit(contract))
    assert report2.violation_kind is None


# ------------------------------------------------------- partial dispatch


def test_partial_dispatch_retreats_moved_side_and_revokes_permit():
    gateway, _, left, right = _gateway(right=FakeExecutor("right", fail_on={"dispatch"}))
    contract = _contract()
    permit = _permit(contract)
    report = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert report.violation_kind == EXECUTOR_FAILURE
    assert report.left_dispatched and not report.right_dispatched
    assert report.receipt.outcome == OUTCOME_PARTIAL_DISPATCH
    assert report.receipt.validate() == []
    assert report.receipt.left_action_receipt is not None
    assert report.receipt.right_action_receipt is None
    assert left.calls == ["dispatch:approach", "retreat:safe_open"]
    assert report.retreat_confirmed == {"left": True}
    assert report.permit_state == "revoked_partial"
    # the revoked permit cannot dispatch again
    again = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert again.violation_kind == PERMIT_INVALID


def test_left_executor_failure_means_nothing_moved():
    gateway, _, left, right = _gateway(left=FakeExecutor("left", fail_on={"dispatch"}))
    contract = _contract()
    report = gateway.dispatch(_envelope(), contract=contract, permit=_permit(contract))
    assert report.violation_kind == EXECUTOR_FAILURE
    assert not report.left_dispatched and not report.right_dispatched
    assert report.receipt.outcome == OUTCOME_ABORTED_BEFORE_DISPATCH
    assert right.calls == []  # right never attempted
    assert "retreat:safe_open" not in left.calls  # nothing to retreat


def test_barrier_skew_violation_retreats_both():
    # right executor takes 0.4s; barrier allows 0.25s
    gateway, _, left, right = _gateway(right=FakeExecutor("right", delay_s=0.4))
    contract = _contract()
    permit = _permit(contract)
    report = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert report.violation_kind == DISPATCH_SKEW_VIOLATION
    assert report.receipt.outcome == OUTCOME_ABORTED_BEFORE_DISPATCH
    assert report.receipt.start_skew_ms >= 400.0
    assert "retreat:safe_open" in left.calls and "retreat:safe_open" in right.calls
    assert report.permit_state == "revoked_skew"


# ------------------------------------------------------------------ estop


def test_estop_fans_out_to_both_and_releases_leases():
    gateway, leases, left, right = _gateway()
    leases.acquire_all("someone_else")
    report = gateway.estop()
    assert report.all_confirmed
    assert left.calls == ["estop"] and right.calls == ["estop"]
    assert leases.holders() == {"left": None, "right": None}


def test_estop_reports_unconfirmed_side_honestly():
    gateway, _, left, right = _gateway(right=FakeExecutor("right", fail_on={"estop"}))
    report = gateway.estop()
    assert not report.all_confirmed
    assert report.left_confirmed and not report.right_confirmed


# ------------------------------------------------- contract binding (fix 2)


def test_envelope_with_foreign_contract_hash_refused_despite_valid_permit():
    """The envelope's safety block must name the authorizing contract:
    a valid permit never dispatches an envelope bound to a stale or
    foreign contract hash."""
    gateway, leases, left, right = _gateway()
    contract = _contract()
    env = _envelope()
    stale = BimanualActionEnvelope(
        **{
            **env.__dict__,
            "safety": SafetyBlock(
                contract_hash="chor_stale_foreign",
                permitted_contact_pair=env.pair_id,
                forbidden_contact_pairs=env.safety.forbidden_contact_pairs,
                retreat_action=env.safety.retreat_action,
            ),
        }
    )
    report = gateway.dispatch(stale, contract=contract, permit=_permit(contract))
    assert report.violation_kind == ENVELOPE_CONTRACT_MISMATCH
    assert report.receipt.outcome == OUTCOME_ABORTED_BEFORE_DISPATCH
    assert left.calls == [] and right.calls == []
    assert leases.holders() == {"left": None, "right": None}


# ------------------------------------------- estop permit revocation (fix 1)


def test_estop_revokes_active_sequence_permit_and_blocks_redispatch():
    """Regression: the first dispatch consumes the sequence permit, so an
    E-Stop that only revokes 'unused' permits never actually revokes.
    After estop the same permit must never authorize again."""
    gateway, _, left, right = _gateway()
    contract = _contract()
    permit = _permit(contract)
    first = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert first.violation_kind is None
    assert permit.used  # consumed by the first dispatch
    report = gateway.estop()
    assert report.permit_revoked is True
    assert permit.revoked_reason == "estop"
    again = gateway.dispatch(_second_envelope(), contract=contract, permit=permit)
    assert again.violation_kind == PERMIT_INVALID
    # the executors were NOT called a second time
    assert left.calls == ["dispatch:approach", "estop"]
    assert right.calls == ["dispatch:hold", "estop"]


# --------------------------------------- estop vs in-flight leases (fix 3)


class BlockingExecutor(FakeExecutor):
    """Left dispatch blocks until released, simulating an in-flight
    hardware call during an E-Stop."""

    def __init__(self, name: str):
        super().__init__(name)
        import threading

        self.entered = threading.Event()
        self.release = threading.Event()

    def dispatch(self, action, *, timeout_ms: float) -> str:
        self.calls.append(f"dispatch:{action.get('gesture')}")
        self.entered.set()
        self.release.wait(timeout=5.0)
        return f"arec_{self.name}_{len(self.calls)}"


def test_estop_does_not_release_leases_held_by_in_flight_dispatch():
    """Releasing a lease under an in-flight dispatch lets a new dispatch
    re-acquire and touch hardware concurrently.  E-Stop must leave those
    leases alone (revoked-and-draining) and report them honestly."""
    import threading

    blocking = BlockingExecutor("left")
    gateway, leases, left, right = _gateway(left=blocking)
    contract = _contract()
    permit = _permit(contract)
    outcome: list[object] = []
    thread = threading.Thread(
        target=lambda: outcome.append(
            gateway.dispatch(_envelope(), contract=contract, permit=permit)
        ),
        daemon=True,
    )
    thread.start()
    assert blocking.entered.wait(timeout=5.0)  # dispatch is in flight
    report = gateway.estop()
    assert report.all_confirmed
    assert report.permit_revoked is True
    # the in-flight leases were NOT released out from under the dispatch
    assert report.leases_released == ()
    assert set(report.leases_draining) == {"left", "right"}
    assert leases.holders() != {"left": None, "right": None}
    # and no new dispatch can acquire while the old one drains
    blocked = gateway.dispatch(_second_envelope(), contract=contract, permit=_permit(contract))
    assert blocked.violation_kind in (LEASE_CONFLICT, PERMIT_INVALID)
    blocking.release.set()
    thread.join(timeout=5.0)
    assert len(outcome) == 1
    # after draining, the dispatch's own cleanup released the leases
    assert leases.holders() == {"left": None, "right": None}
    assert right.calls[-1] == "dispatch:hold" or "estop" in right.calls


def test_estop_report_discloses_released_sides():
    gateway, leases, _, _ = _gateway()
    leases.acquire_all("someone_else")
    report = gateway.estop()
    assert set(report.leases_released) == {"left", "right"}
    assert report.leases_draining == ()
    assert leases.holders() == {"left": None, "right": None}


# ------------------------------------- executor failure revokes (fix 4)


def test_left_executor_failure_revokes_permit_against_retry():
    """A failed executor consumes the attempt: the permit is revoked so
    the same permit cannot retry the failed dispatch."""
    gateway, _, left, right = _gateway(left=FakeExecutor("left", fail_on={"dispatch"}))
    contract = _contract()
    permit = _permit(contract)
    report = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert report.violation_kind == EXECUTOR_FAILURE
    assert report.permit_state == "revoked_dispatch_failed"
    assert permit.revoked_reason == "dispatch_failed"
    again = gateway.dispatch(_second_envelope(), contract=contract, permit=permit)
    assert again.violation_kind == PERMIT_INVALID
    assert right.calls == []


# ------------------------------------------------- sequence permit reuse


def _second_envelope(seq: str = "seq_1") -> BimanualActionEnvelope:
    env = _envelope()
    return BimanualActionEnvelope(**{**env.__dict__, "interaction_id": "int_2"})


def test_one_permit_authorizes_many_envelopes_of_one_sequence():
    """v4 §18: one Sequence Permit per SEQUENCE, many envelopes inside.
    The first dispatch consumes; further envelopes of the same sequence
    dispatch under the gateway's active permit."""
    gateway, _, left, right = _gateway()
    contract = _contract()
    permit = _permit(contract)
    first = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert first.violation_kind is None
    second = gateway.dispatch(_second_envelope(), contract=contract, permit=permit)
    assert second.violation_kind is None
    assert left.calls == ["dispatch:approach", "dispatch:approach"]


def test_revoked_sequence_permit_never_authorizes_again():
    gateway, _, _, _ = _gateway(right=FakeExecutor("right", fail_on={"dispatch"}))
    contract = _contract()
    permit = _permit(contract)
    first = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert first.permit_state == "revoked_partial"
    again = gateway.dispatch(_second_envelope(), contract=contract, permit=permit)
    assert again.violation_kind == PERMIT_INVALID


def test_expired_active_sequence_permit_stops_authorizing():
    gateway, _, _, _ = _gateway()
    contract = _contract()
    permit = SequencePermit.issue(contract, intent_hash="intent_x", lifetime_s=0.05)
    first = gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert first.violation_kind is None
    time.sleep(0.06)
    again = gateway.dispatch(_second_envelope(), contract=contract, permit=permit)
    assert again.violation_kind == PERMIT_INVALID


def test_foreign_used_permit_is_not_a_sequence_permit():
    """A permit consumed by ANOTHER gateway (or attacker) must not be
    adoptable: only the gateway holding it as active may continue."""
    gateway, _, _, _ = _gateway()
    other_gateway, _, _, _ = _gateway()
    contract = _contract()
    permit = _permit(contract)
    first = other_gateway.dispatch(_envelope(), contract=contract, permit=permit)
    assert first.violation_kind is None
    adopt = gateway.dispatch(_second_envelope(), contract=contract, permit=permit)
    assert adopt.violation_kind == PERMIT_INVALID
