"""Bimanual ActionGateway (v4 §7.2/§7.3, PR-TT-2) — atomic dual-body dispatch.

The gateway is the ONLY path from an approved BimanualActionEnvelope to
hardware.  Its invariants:

* Atomic lease: body leases are acquired in ONE FIXED ORDER (left, then
  right); any failure releases both — zero hardware action on any gate
  failure (v4 §7.2).
* Every gate BEFORE dispatch is a receipt, not an exception: invalid
  envelope, stale permit, stale camera, missing snapshots, forbidden
  pair, held lease — each produces an ABORTED_BEFORE_DISPATCH receipt
  naming its violation.
* Partial dispatch is a first-class outcome (v4 §7.3): if the left hand
  dispatched and the right hand failed, the moved side is retreated
  IMMEDIATELY, the permit is revoked, and the receipt says
  PARTIAL_DISPATCH.  Never a silent half-action.
* E-Stop fans out to BOTH bodies and is reported per side — an
  unconfirmed E-Stop on either side is named in the report.

The gateway does NOT judge contact — dispatch success means "the
commands were delivered inside the barrier window"; CONTACT_CONFIRMED
belongs to the Contact Supervisor (PR-TT-4).

Executors are injected protocols so phase-1 tests run fully in memory;
the RH56 binding arrives with the supervisor runner (PR-TT-4/5).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from rosclaw.twintouch.choreography import (
    PERMIT_ALREADY_USED,
    PERMIT_EXPIRED,
    PERMIT_HASH_MISMATCH,
    PERMIT_OK,
    ContactChoreographyContract,
    SequencePermit,
)
from rosclaw.twintouch.envelope import BimanualActionEnvelope
from rosclaw.twintouch.pairs import ForbiddenCollisionMap
from rosclaw.twintouch.receipt import (
    OUTCOME_ABORTED_BEFORE_DISPATCH,
    OUTCOME_DISPATCHED,
    OUTCOME_PARTIAL_DISPATCH,
    InteractionReceipt,
)

# Dispatch verdict for the receipt: commands delivered inside the
# barrier window.  Contact is the supervisor's verdict, not the gateway's.
DISPATCH_SKEW_VIOLATION = "DISPATCH_SKEW_VIOLATION"
LEASE_CONFLICT = "LEASE_CONFLICT"
PRECONDITION_FAILED = "PRECONDITION_FAILED"
ENVELOPE_INVALID = "ENVELOPE_INVALID"
ENVELOPE_CONTRACT_MISMATCH = "ENVELOPE_CONTRACT_MISMATCH"
PERMIT_INVALID = "PERMIT_INVALID"
EXECUTOR_FAILURE = "EXECUTOR_FAILURE"

# Fixed global lease order (v4 §7.2 step 1-2) — deadlock-free by
# construction: everyone who takes two leases takes them left→right.
LEASE_ORDER = ("left", "right")


class BodyExecutor(Protocol):
    """One body's dispatch surface.  Implementations MUST treat
    ``dispatch`` as deliver-only (no contact judgement) and ``retreat``
    as best-effort-always-attempted."""

    def dispatch(self, action: dict[str, Any], *, timeout_ms: float) -> str:
        """Deliver the action; returns a per-side dispatch receipt id.
        Raises on transport/executor failure."""
        ...

    def retreat(self, retreat_action: dict[str, Any]) -> bool:
        """Best-effort retreat to the safe pose; True iff confirmed."""
        ...

    def estop(self) -> bool:
        """Immediate stop; True iff the body confirmed the stop."""
        ...


class PreconditionProbe(Protocol):
    """External freshness/identity probes evaluated before leasing."""

    def camera_fresh(self, *, max_age_ms: float) -> bool: ...

    def snapshots_valid(self, envelope: BimanualActionEnvelope) -> list[str]: ...


@dataclass
class _BodyLease:
    body_id: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    holder: str | None = None

    def acquire(self, owner: str) -> bool:
        if self.lock.acquire(blocking=False):
            self.holder = owner
            return True
        return False

    def release(self, owner: str) -> None:
        if self.holder == owner:
            self.holder = None
            self.lock.release()


class LeaseRegistry:
    """Per-body exclusive leases with fixed-order atomic acquisition."""

    def __init__(self, body_ids: dict[str, str]) -> None:
        # body_ids: {"left": ..., "right": ...}
        self._leases = {side: _BodyLease(body_id) for side, body_id in body_ids.items()}

    def acquire_all(self, owner: str) -> tuple[bool, str | None]:
        """Fixed-order (LEASE_ORDER), all-or-nothing.  Returns
        (acquired, conflicting_side)."""
        acquired: list[str] = []
        for side in LEASE_ORDER:
            lease = self._leases[side]
            if not lease.acquire(owner):
                for done in acquired:
                    self._leases[done].release(owner)
                return False, side
            acquired.append(side)
        return True, None

    def release_all(self, owner: str) -> None:
        for side in reversed(LEASE_ORDER):
            self._leases[side].release(owner)

    def release_any(self, side: str) -> bool:
        """E-Stop path: release a side regardless of who holds it.
        Hardware stop outranks software lease ownership."""
        lease = self._leases[side]
        if lease.holder is not None:
            lease.release(lease.holder)
            return True
        return False

    def holders(self) -> dict[str, str | None]:
        return {side: lease.holder for side, lease in self._leases.items()}


@dataclass(frozen=True)
class DispatchReport:
    receipt: InteractionReceipt
    violation_kind: str | None
    violations: tuple[str, ...]
    left_dispatched: bool
    right_dispatched: bool
    retreat_confirmed: dict[str, bool]
    permit_state: str


@dataclass(frozen=True)
class EstopReport:
    left_confirmed: bool
    right_confirmed: bool
    # Sides whose lease the E-Stop actually freed.  A lease held by an
    # in-flight dispatch is NOT freed (see estop()) — it is reported
    # under leases_draining instead.
    leases_released: tuple[str, ...]
    # Sides still leased to an in-flight dispatch at E-Stop time: the
    # dispatch is draining and releases the lease itself when it
    # unwinds.  Releasing it here would let a new dispatch re-acquire
    # and touch hardware concurrently with the draining dispatch.
    leases_draining: tuple[str, ...]
    permit_revoked: bool

    @property
    def all_confirmed(self) -> bool:
        return self.left_confirmed and self.right_confirmed


class BimanualActionGateway:
    def __init__(
        self,
        *,
        executors: dict[str, BodyExecutor],
        leases: LeaseRegistry,
        probe: PreconditionProbe,
        collision_map: ForbiddenCollisionMap | None = None,
        camera_freshness_ms: float = 500.0,
        clock=time.monotonic,
    ) -> None:
        missing = set(LEASE_ORDER) - set(executors)
        if missing:
            raise ValueError(f"executors missing for sides: {sorted(missing)}")
        self._executors = executors
        self._leases = leases
        self._probe = probe
        self._collision_map = collision_map or ForbiddenCollisionMap()
        self._camera_freshness_ms = camera_freshness_ms
        self._clock = clock
        self._active_permit: SequencePermit | None = None
        # Guards _active_permit and the in-flight lease accounting so an
        # E-Stop interleaving with a dispatch is fail-closed: the permit
        # is re-checked under this lock after lease acquisition, and a
        # lease with in_flight > 0 is never released by estop().
        self._state_lock = threading.Lock()
        self._in_flight: dict[str, int] = dict.fromkeys(LEASE_ORDER, 0)

    # ---------------------------------------------------------- helpers

    def _permit_state_for_dispatch(
        self, permit: SequencePermit, contract: ContactChoreographyContract
    ) -> str:
        """Sequence-permit semantics: ONE permit authorizes ONE sequence,
        which contains MANY envelopes (v4 §18).  The first dispatch
        consumes the permit; further envelopes of the same sequence are
        authorized iff the permit is THIS gateway's active one and still
        inside its deadline — verify() short-circuits on ``used`` before
        checking hash/expiry, so the active-sequence path re-checks both
        explicitly.  A permit revoked mid-sequence (partial dispatch,
        barrier violation, E-Stop) never authorizes again."""
        state = permit.verify(contract)
        if (
            state == PERMIT_ALREADY_USED
            and self._active_permit is permit
            and permit.revoked_reason == "sequence_started"
        ):
            if time.time() > permit.expires_at_s:
                return PERMIT_EXPIRED
            if permit.contract_hash != contract.contract_hash():
                return PERMIT_HASH_MISMATCH
            return PERMIT_OK
        return state

    def _abort_receipt(
        self,
        envelope: BimanualActionEnvelope,
        *,
        violations: list[str],
        left_dispatched: bool,
        right_dispatched: bool,
        left_receipt: str | None = None,
        right_receipt: str | None = None,
        outcome: str = OUTCOME_ABORTED_BEFORE_DISPATCH,
        start_skew_ms: float | None = None,
    ) -> InteractionReceipt:
        return InteractionReceipt(
            interaction_id=envelope.interaction_id,
            pair_id=envelope.pair_id,
            left_action_receipt=left_receipt,
            right_action_receipt=right_receipt,
            intended_contact=envelope.pair_id,
            observed_contact=None,
            contact_confirmed=False,
            wrong_finger_contact=False,
            unintended_contact=False,
            left_force_peak=None,
            right_force_peak=None,
            visual_distance_min_m=None,
            contact_latency_ms=None,
            start_skew_ms=start_skew_ms,
            dwell_ms=None,
            clearance_verified=False,
            outcome=outcome,
            evidence_refs=tuple(violations),
        )

    # ---------------------------------------------------------- dispatch

    def dispatch(
        self,
        envelope: BimanualActionEnvelope,
        *,
        contract: ContactChoreographyContract,
        permit: SequencePermit,
    ) -> DispatchReport:
        owner = f"gw_{uuid.uuid4().hex[:12]}"
        # 1. Envelope validity (contract-level).
        violations = envelope.validate()
        if violations:
            return DispatchReport(
                receipt=self._abort_receipt(
                    envelope, violations=violations, left_dispatched=False, right_dispatched=False
                ),
                violation_kind=ENVELOPE_INVALID,
                violations=tuple(violations),
                left_dispatched=False,
                right_dispatched=False,
                retreat_confirmed={},
                permit_state="unused",
            )
        # 1b. The envelope must name the contract that authorizes it:
        # a valid permit for contract C never dispatches an envelope
        # whose safety block points at a stale or foreign contract.
        contract_hash = contract.contract_hash()
        if envelope.safety.contract_hash != contract_hash:
            violations = [
                f"envelope safety contract {envelope.safety.contract_hash!r} does not match "
                f"authorizing contract {contract_hash!r}"
            ]
            return DispatchReport(
                receipt=self._abort_receipt(
                    envelope, violations=violations, left_dispatched=False, right_dispatched=False
                ),
                violation_kind=ENVELOPE_CONTRACT_MISMATCH,
                violations=tuple(violations),
                left_dispatched=False,
                right_dispatched=False,
                retreat_confirmed={},
                permit_state="unused",
            )
        # 2. Permit (contract hash binding, deadline, single-use).
        permit_state = self._permit_state_for_dispatch(permit, contract)
        if permit_state != PERMIT_OK:
            return DispatchReport(
                receipt=self._abort_receipt(
                    envelope,
                    violations=[f"permit: {permit_state}"],
                    left_dispatched=False,
                    right_dispatched=False,
                ),
                violation_kind=PERMIT_INVALID,
                violations=(f"permit: {permit_state}",),
                left_dispatched=False,
                right_dispatched=False,
                retreat_confirmed={},
                permit_state=permit_state,
            )
        # 3. Collision map (forbidden pairs never reach the lease stage).
        pair_violations = self._collision_map.validate_action_pairing(envelope.pair_id)
        if pair_violations:
            return DispatchReport(
                receipt=self._abort_receipt(
                    envelope,
                    violations=pair_violations,
                    left_dispatched=False,
                    right_dispatched=False,
                ),
                violation_kind=PRECONDITION_FAILED,
                violations=tuple(pair_violations),
                left_dispatched=False,
                right_dispatched=False,
                retreat_confirmed={},
                permit_state="unused",
            )
        # 4. Camera freshness + self snapshots (v4 §7.2 steps 3-4).
        precondition_violations: list[str] = []
        if not self._probe.camera_fresh(max_age_ms=self._camera_freshness_ms):
            precondition_violations.append("camera observation stale — no new dispatch")
        precondition_violations.extend(self._probe.snapshots_valid(envelope))
        if precondition_violations:
            return DispatchReport(
                receipt=self._abort_receipt(
                    envelope,
                    violations=precondition_violations,
                    left_dispatched=False,
                    right_dispatched=False,
                ),
                violation_kind=PRECONDITION_FAILED,
                violations=tuple(precondition_violations),
                left_dispatched=False,
                right_dispatched=False,
                retreat_confirmed={},
                permit_state="unused",
            )
        # 5. Atomic lease (fixed order, all-or-nothing).  Acquisition,
        # the in-flight accounting and a permit re-check all happen
        # under the state lock, so an E-Stop racing this dispatch either
        # revokes the permit BEFORE we re-check (we refuse here) or
        # finds in_flight > 0 and leaves our leases alone (we drain).
        with self._state_lock:
            acquired, conflicting = self._leases.acquire_all(owner)
            permit_recheck: str | None = None
            if acquired:
                permit_recheck = self._permit_state_for_dispatch(permit, contract)
                if permit_recheck != PERMIT_OK:
                    self._leases.release_all(owner)
                    acquired = False
                else:
                    # Consume + activate under the lock: an E-Stop either
                    # landed before this section (the re-check above
                    # refuses) or lands after and revokes THIS permit.
                    permit.consume(reason="sequence_started")
                    self._active_permit = permit
                    for side in LEASE_ORDER:
                        self._in_flight[side] += 1
        if not acquired:
            if permit_recheck is not None:
                violations = [f"permit revoked before dispatch: {permit_recheck}"]
                return DispatchReport(
                    receipt=self._abort_receipt(
                        envelope,
                        violations=violations,
                        left_dispatched=False,
                        right_dispatched=False,
                    ),
                    violation_kind=PERMIT_INVALID,
                    violations=tuple(violations),
                    left_dispatched=False,
                    right_dispatched=False,
                    retreat_confirmed={},
                    permit_state=permit_recheck,
                )
            violations = [f"lease conflict on {conflicting} body"]
            return DispatchReport(
                receipt=self._abort_receipt(
                    envelope, violations=violations, left_dispatched=False, right_dispatched=False
                ),
                violation_kind=LEASE_CONFLICT,
                violations=tuple(violations),
                left_dispatched=False,
                right_dispatched=False,
                retreat_confirmed={},
                permit_state="unused",
            )
        try:
            return self._dispatch_under_lease(envelope, permit, owner)
        finally:
            self._leases.release_all(owner)
            with self._state_lock:
                for side in LEASE_ORDER:
                    self._in_flight[side] -= 1

    def _dispatch_under_lease(
        self,
        envelope: BimanualActionEnvelope,
        permit: SequencePermit,
        owner: str,
    ) -> DispatchReport:
        # The permit was consumed and made active under the state lock
        # in dispatch() — nothing to do here but run the barrier.
        timeout = envelope.coordination.timeout_ms
        max_skew_ms = envelope.coordination.maximum_start_skew_ms

        left_receipt: str | None = None
        right_receipt: str | None = None
        retreat_confirmed: dict[str, bool] = {}

        # Barrier: left dispatches first (fixed order); the start skew
        # is the measured wall time between the two dispatch calls.
        try:
            left_receipt = self._executors["left"].dispatch(
                envelope.left.action, timeout_ms=timeout
            )
        except Exception as exc:  # noqa: BLE001 — executor failure is data
            # Nothing moved, but the attempt consumed the sequence:
            # revoke so a failed executor is never retried under the
            # same permit.
            permit.revoke(reason="dispatch_failed")
            with self._state_lock:
                if self._active_permit is permit:
                    self._active_permit = None
            violations = [f"left executor: {exc!r}"]
            return DispatchReport(
                receipt=self._abort_receipt(
                    envelope, violations=violations, left_dispatched=False, right_dispatched=False
                ),
                violation_kind=EXECUTOR_FAILURE,
                violations=tuple(violations),
                left_dispatched=False,
                right_dispatched=False,
                retreat_confirmed={},
                permit_state="revoked_dispatch_failed",
            )
        t_left = self._clock()
        try:
            right_receipt = self._executors["right"].dispatch(
                envelope.right.action, timeout_ms=timeout
            )
        except Exception as exc:  # noqa: BLE001
            # PARTIAL DISPATCH (v4 §7.3): left moved, right did not —
            # retreat the moved side immediately, revoke the permit.
            violations = [f"right executor: {exc!r}", "partial dispatch: retreating left"]
            retreat_confirmed["left"] = self._executors["left"].retreat(
                envelope.safety.retreat_action
            )
            permit.revoke(reason="partial_dispatch")
            with self._state_lock:
                if self._active_permit is permit:
                    self._active_permit = None
            receipt = self._abort_receipt(
                envelope,
                violations=violations,
                left_dispatched=True,
                right_dispatched=False,
                left_receipt=left_receipt,
                outcome=OUTCOME_PARTIAL_DISPATCH,
            )
            return DispatchReport(
                receipt=receipt,
                violation_kind=EXECUTOR_FAILURE,
                violations=tuple(violations),
                left_dispatched=True,
                right_dispatched=False,
                retreat_confirmed=retreat_confirmed,
                permit_state="revoked_partial",
            )
        t_right = self._clock()

        start_skew_ms = (t_right - t_left) * 1000.0
        if start_skew_ms > max_skew_ms:
            # Barrier violated AFTER both dispatched: retreat both —
            # coordinated start is a safety property, not a nicety.
            violations = [
                f"start skew {start_skew_ms:.1f} ms exceeds barrier {max_skew_ms:.1f} ms",
                "retreating both",
            ]
            for side in LEASE_ORDER:
                retreat_confirmed[side] = self._executors[side].retreat(
                    envelope.safety.retreat_action
                )
            permit.revoke(reason="barrier_skew_violation")
            with self._state_lock:
                if self._active_permit is permit:
                    self._active_permit = None
            receipt = self._abort_receipt(
                envelope,
                violations=violations,
                left_dispatched=True,
                right_dispatched=True,
                left_receipt=left_receipt,
                right_receipt=right_receipt,
                outcome=OUTCOME_ABORTED_BEFORE_DISPATCH,
                start_skew_ms=start_skew_ms,
            )
            return DispatchReport(
                receipt=receipt,
                violation_kind=DISPATCH_SKEW_VIOLATION,
                violations=tuple(violations),
                left_dispatched=True,
                right_dispatched=True,
                retreat_confirmed=retreat_confirmed,
                permit_state="revoked_skew",
            )

        receipt = InteractionReceipt(
            interaction_id=envelope.interaction_id,
            pair_id=envelope.pair_id,
            left_action_receipt=left_receipt,
            right_action_receipt=right_receipt,
            intended_contact=envelope.pair_id,
            observed_contact=None,
            contact_confirmed=False,
            wrong_finger_contact=False,
            unintended_contact=False,
            left_force_peak=None,
            right_force_peak=None,
            visual_distance_min_m=None,
            contact_latency_ms=None,
            start_skew_ms=start_skew_ms,
            dwell_ms=None,
            clearance_verified=False,
            outcome=OUTCOME_DISPATCHED,
            evidence_refs=(f"barrier_skew_ms={start_skew_ms:.1f}",),
        )
        return DispatchReport(
            receipt=receipt,
            violation_kind=None,
            violations=(),
            left_dispatched=True,
            right_dispatched=True,
            retreat_confirmed={},
            permit_state="consumed",
        )

    # ------------------------------------------------------------ estop

    def estop(self) -> EstopReport:
        """Fan the E-Stop out to BOTH bodies (v4 §21 PR-TT-2).  Every
        side's confirmation is reported individually — an unconfirmed
        stop is named, never averaged away.

        Lease semantics: a lease held by an in-flight dispatch is NOT
        released here — that dispatch is still touching hardware and
        must drain (its own finally-block releases the lease).  Releasing
        it would let a new dispatch re-acquire and run concurrently with
        the draining one.  The permit, by contrast, is revoked
        unconditionally: the first dispatch consumed it, so any check of
        ``used`` would make the revocation dead code."""
        results: dict[str, bool] = {}
        for side in LEASE_ORDER:
            try:
                results[side] = bool(self._executors[side].estop())
            except Exception:  # noqa: BLE001 — an estop that raises is unconfirmed
                results[side] = False
        with self._state_lock:
            permit_revoked = False
            if self._active_permit is not None:
                self._active_permit.revoke(reason="estop")
                # Clearing the active reference guarantees
                # _permit_state_for_dispatch can never re-authorize this
                # permit again — its active-sequence path keys on
                # ``self._active_permit is permit``.
                self._active_permit = None
                permit_revoked = True
            released: list[str] = []
            draining: list[str] = []
            for side in LEASE_ORDER:
                if self._in_flight[side] > 0:
                    draining.append(side)
                elif self._leases.release_any(side):
                    # No in-flight dispatch: hardware stop outranks
                    # software lease ownership.
                    released.append(side)
        return EstopReport(
            left_confirmed=results.get("left", False),
            right_confirmed=results.get("right", False),
            leases_released=tuple(released),
            leases_draining=tuple(draining),
            permit_revoked=permit_revoked,
        )
