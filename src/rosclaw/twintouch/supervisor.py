"""Contact Supervisor (v4 §5/§6, PR-TT-4) — the deterministic contact
state machine.  No LLM in the loop.

One supervisor instance runs ONE pair episode:

    SAFE_RESET → PAIR_SELECTED → PEER_READY → VISUAL_ALIGN
    → COARSE_APPROACH → FINE_APPROACH → CONTACT_CANDIDATE
    → CONTACT_CONFIRMED → DWELL → RELEASE → CLEARANCE_VERIFIED
    → EPISODE_COMMITTED

Every anomaly (§5) enters the same recovery spine:

    STOP_APPROACH → RETREAT_BOTH → VERIFY_CLEAR → RECORD_FAILURE

and the other hand NEVER continues its remaining motion (v4 §5).

Architecture: the supervisor is PURE — it consumes
``SupervisorObservation`` and emits ``SupervisorDecision``; it never
touches hardware.  The runner (PR-TT-5/6) collects observations and
routes ISSUE_STEP decisions through the Bimanual ActionGateway (which
enforces lease + permit).  Global guards — thermal, transport,
stale camera, NON-TARGET FORCE — run before every state evaluation,
so an unintended contact during ANY phase aborts the episode.

Contact confirmation is the §6.4 consensus, never a single channel:
    left target force rise
AND right target force rise
AND (visual near-contact OR motion-response saturation)
AND no non-target force rise

The visual channel uses the 3D nearest-cluster distance (min over
pairwise point distances between the two hands' clusters) — the metric
that sees interleaving, unlike the lateral extremes that missed the
2026-07-31 incident.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from rosclaw.twintouch.pairs import is_valid_pair_id, pair_by_id
from rosclaw.twintouch.receipt import (
    OUTCOME_CONTACT_CONFIRMED,
    OUTCOME_EARLY_CONTACT,
    OUTCOME_NO_CONTACT,
    OUTCOME_ONE_SIDED_FORCE,
    OUTCOME_PEER_NOT_READY,
    OUTCOME_RELEASE_FAILED,
    OUTCOME_STALE_OBSERVATION,
    OUTCOME_THERMAL_ABORT,
    OUTCOME_TRANSPORT_FAILURE,
    OUTCOME_UNINTENDED_CONTACT,
    OUTCOME_VISUAL_FORCE_CONFLICT,
    OUTCOME_WRONG_FINGER_CONTACT,
    InteractionReceipt,
)

# ------------------------------------------------------------------ states

SAFE_RESET = "SAFE_RESET"
PAIR_SELECTED = "PAIR_SELECTED"
PEER_READY = "PEER_READY"
VISUAL_ALIGN = "VISUAL_ALIGN"
COARSE_APPROACH = "COARSE_APPROACH"
FINE_APPROACH = "FINE_APPROACH"
CONTACT_CANDIDATE = "CONTACT_CANDIDATE"
CONTACT_CONFIRMED = "CONTACT_CONFIRMED"
DWELL = "DWELL"
RELEASE = "RELEASE"
CLEARANCE_VERIFIED = "CLEARANCE_VERIFIED"
EPISODE_COMMITTED = "EPISODE_COMMITTED"

STOP_APPROACH = "STOP_APPROACH"
RETREAT_BOTH = "RETREAT_BOTH"
VERIFY_CLEAR = "VERIFY_CLEAR"
RECORD_FAILURE = "RECORD_FAILURE"

TERMINAL_STATES = frozenset({EPISODE_COMMITTED, RECORD_FAILURE})
RECOVERY_STATES = frozenset({STOP_APPROACH, RETREAT_BOTH, VERIFY_CLEAR})

ALL_STATES = frozenset(
    {
        SAFE_RESET,
        PAIR_SELECTED,
        PEER_READY,
        VISUAL_ALIGN,
        COARSE_APPROACH,
        FINE_APPROACH,
        CONTACT_CANDIDATE,
        CONTACT_CONFIRMED,
        DWELL,
        RELEASE,
        CLEARANCE_VERIFIED,
        EPISODE_COMMITTED,
        STOP_APPROACH,
        RETREAT_BOTH,
        VERIFY_CLEAR,
        RECORD_FAILURE,
    }
)

# Decision kinds
DECISION_NONE = "NONE"
DECISION_ISSUE_STEP = "ISSUE_STEP"
DECISION_RETREAT = "RETREAT"
DECISION_COMMIT = "COMMIT"
DECISION_FAIL = "FAIL"


@dataclass(frozen=True)
class SupervisorTuning:
    """Step/threshold constants.  Every value must come from
    TwinTouchConfig or be smaller — the runner enforces that binding;
    this dataclass only types them."""

    contact_force_delta_raw: float = 60.0
    non_target_force_abort_raw: float = 60.0
    coarse_step_raw: int = 40
    fine_step_raw: int = 10
    coarse_to_fine_distance_m: float = 0.02
    near_contact_m: float = 0.008
    visual_conflict_distance_m: float = 0.05
    max_coarse_steps: int = 6
    max_fine_steps: int = 8
    one_sided_frame_budget: int = 5
    dwell_ms: float = 300.0
    release_margin_raw: int = 60
    release_force_epsilon_raw: float = 20.0
    max_release_steps: int = 4
    camera_freshness_ms: float = 500.0
    temperature_abort_c: float = 49.0
    saturation_delta_raw: int = 2  # |Δactual| below this while stepping = saturated


# ------------------------------------------------------------- observation


@dataclass(frozen=True)
class HandObservation:
    """One hand's live telemetry slice.  ok=False = transport failure."""

    ok: bool
    angle_actual: dict[str, int | None]
    force_act: dict[str, float | None]
    temperature_max_c: float | None


@dataclass(frozen=True)
class VisualObservation:
    """The shared camera's view of the pair: per-hand cluster validity,
    the 3D NEAREST-cluster distance (not lateral extremes), whether the
    declared pair's identity is visually confirmed, and frame age."""

    age_ms: float
    left_cluster_ok: bool
    right_cluster_ok: bool
    min_distance_m: float | None
    pair_identity_confirmed: bool | None


@dataclass(frozen=True)
class SupervisorObservation:
    ts_s: float
    left: HandObservation | None
    right: HandObservation | None
    visual: VisualObservation | None


@dataclass(frozen=True)
class SupervisorDecision:
    kind: str  # DECISION_*
    note: str
    # ISSUE_STEP payload: side -> {"joints": {joint: delta_raw}, "step_raw": int}
    step: dict[str, Any] | None = None
    receipt: InteractionReceipt | None = None


# ------------------------------------------------------------- baselines


@dataclass(frozen=True)
class ForceBaseline:
    """Session force baseline captured with the hand OPEN and touching
    nothing (RH56 calibration rule, v4 §3).  Frozen for the episode."""

    side: str
    medians: dict[str, float]
    samples: int

    @classmethod
    def capture(
        cls, side: str, samples: list[dict[str, float | None]], *, min_samples: int = 5
    ) -> ForceBaseline:
        fingers = sorted({f for s in samples for f, v in s.items() if v is not None})
        if len(samples) < min_samples:
            raise ValueError(f"baseline needs >= {min_samples} samples, got {len(samples)}")
        if not fingers:
            raise ValueError("baseline captured no force channels")
        medians: dict[str, float] = {}
        for finger in fingers:
            values = [float(s[finger]) for s in samples if s.get(finger) is not None]
            if len(values) < min_samples:
                raise ValueError(f"baseline channel {finger} has {len(values)} < {min_samples}")
            medians[finger] = float(statistics.median(values))
        return cls(side=side, medians=medians, samples=len(samples))

    def delta(self, force_act: dict[str, float | None]) -> dict[str, float]:
        """Signed rise above baseline per finger (missing channel = 0.0
        and the caller's responsibility to notice via coverage checks)."""
        out: dict[str, float] = {}
        for finger, base in self.medians.items():
            current = force_act.get(finger)
            out[finger] = 0.0 if current is None else float(current) - base
        return out


@dataclass(frozen=True)
class BilateralForceEvidence:
    left_target_rise: bool
    right_target_rise: bool
    non_target_violations: tuple[str, ...]
    left_target_delta: float
    right_target_delta: float


def bilateral_force_consensus(
    *,
    target_finger: str,
    left_delta: dict[str, float],
    right_delta: dict[str, float],
    contact_threshold: float,
    abort_threshold: float,
) -> BilateralForceEvidence:
    """§6.1/§6.4 force evidence for one observation cycle.  Every joint
    except the target finger is a non-target — including thumb_rot:
    even for thumb_thumb, fingertip contact force shows on ``thumb``;
    a thumb_rot rise means something is pressing the thumb base, which
    is unintended by definition."""
    violations: list[str] = []
    for side, deltas in (("left", left_delta), ("right", right_delta)):
        for finger, delta in deltas.items():
            if finger == target_finger:
                continue
            if delta >= abort_threshold:
                violations.append(f"{side}.{finger} +{delta:.0f} raw (non-target)")
    left_target = left_delta.get(target_finger, 0.0)
    right_target = right_delta.get(target_finger, 0.0)
    return BilateralForceEvidence(
        left_target_rise=left_target >= contact_threshold,
        right_target_rise=right_target >= contact_threshold,
        non_target_violations=tuple(violations),
        left_target_delta=left_target,
        right_target_delta=right_target,
    )


# -------------------------------------------------------------- supervisor


@dataclass
class _EpisodeTrack:
    """Running receipt fields collected across the episode."""

    left_force_peak: float | None = None
    right_force_peak: float | None = None
    visual_distance_min_m: float | None = None
    contact_candidate_ts: float | None = None
    contact_confirmed_ts: float | None = None
    dwell_start_ts: float | None = None
    coarse_steps: int = 0
    fine_steps: int = 0
    release_steps: int = 0
    one_sided_frames: int = 0
    anomaly: str | None = None
    anomaly_detail: str = ""
    last_angles: dict[str, int] = field(default_factory=dict)
    saturated_frames: int = 0
    dwell_actual_ms: float | None = None


class ContactSupervisor:
    """Deterministic per-episode state machine (one pair, one episode)."""

    def __init__(
        self,
        *,
        interaction_id: str,
        pair_id: str,
        active_mode: str,  # active_passive | passive_active | mutual
        baselines: dict[str, ForceBaseline],
        reachability_calibrated: bool,
        tuning: SupervisorTuning | None = None,
    ) -> None:
        if not is_valid_pair_id(pair_id):
            raise ValueError(f"{pair_id!r} is not a permitted contact pair")
        pair = pair_by_id(pair_id)
        assert pair is not None
        if active_mode not in ("active_passive", "passive_active", "mutual"):
            raise ValueError(f"active_mode {active_mode!r} unknown")
        missing = {"left", "right"} - set(baselines)
        if missing:
            raise ValueError(f"force baselines missing for {sorted(missing)}")
        self.interaction_id = interaction_id
        self.pair_id = pair_id
        self.target_finger = pair.left_finger
        self.active_mode = active_mode
        self.baselines = baselines
        self.reachability_calibrated = reachability_calibrated
        self.tuning = tuning or SupervisorTuning()
        self.state = SAFE_RESET
        self.track = _EpisodeTrack()
        self.history: list[str] = [SAFE_RESET]

    # ------------------------------------------------------------ helpers

    def _transition(self, new_state: str) -> None:
        assert new_state in ALL_STATES, new_state
        self.state = new_state
        self.history.append(new_state)

    def _enter_recovery(self, anomaly: str, detail: str) -> SupervisorDecision:
        self.track.anomaly = anomaly
        self.track.anomaly_detail = detail
        self._transition(STOP_APPROACH)
        return SupervisorDecision(
            kind=DECISION_NONE,
            note=f"{anomaly}: {detail} — STOP_APPROACH",
        )

    def _active_sides(self) -> tuple[str, ...]:
        if self.active_mode == "mutual":
            return ("left", "right")
        if self.active_mode == "active_passive":
            return ("left",)
        return ("right",)

    def _deltas(self, obs: SupervisorObservation) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for side in ("left", "right"):
            hand = obs.left if side == "left" else obs.right
            if hand is not None and hand.ok:
                out[side] = self.baselines[side].delta(hand.force_act)
            else:
                out[side] = {}
        return out

    def _update_peaks(self, deltas: dict[str, dict[str, float]]) -> None:
        for side, key in (("left", "left_force_peak"), ("right", "right_force_peak")):
            value = deltas.get(side, {}).get(self.target_finger)
            if value is None:
                continue
            current = getattr(self.track, key)
            if current is None or value > current:
                setattr(self.track, key, value)

    def _update_visual_min(self, obs: SupervisorObservation) -> None:
        if obs.visual is None or obs.visual.min_distance_m is None:
            return
        current = self.track.visual_distance_min_m
        if current is None or obs.visual.min_distance_m < current:
            self.track.visual_distance_min_m = obs.visual.min_distance_m

    # -------------------------------------------------------- global guards

    def _global_guard(self, obs: SupervisorObservation) -> SupervisorDecision | None:
        """Guards that outrank any state logic.  Returns a recovery
        decision or None when the cycle is clean."""
        # transport: any hand lost -> both retreat (v4 §3.2)
        if obs.left is None or not obs.left.ok or obs.right is None or not obs.right.ok:
            return self._enter_recovery(
                OUTCOME_TRANSPORT_FAILURE,
                "hand telemetry lost — both hands must retreat",
            )
        # thermal
        for side, hand in (("left", obs.left), ("right", obs.right)):
            temp = hand.temperature_max_c
            if temp is not None and temp >= self.tuning.temperature_abort_c:
                return self._enter_recovery(
                    OUTCOME_THERMAL_ABORT,
                    f"{side} at {temp}°C >= abort {self.tuning.temperature_abort_c}°C",
                )
        # stale camera blocks anything that is still APPROACHING
        if self.state in (VISUAL_ALIGN, COARSE_APPROACH, FINE_APPROACH, CONTACT_CANDIDATE):
            if obs.visual is None or obs.visual.age_ms > self.tuning.camera_freshness_ms:
                return self._enter_recovery(
                    OUTCOME_STALE_OBSERVATION,
                    "camera observation stale — no approach allowed",
                )
        # non-target force (unintended contact) — the highest-priority
        # safety check after transport/thermal; valid in every state
        deltas = self._deltas(obs)
        consensus = bilateral_force_consensus(
            target_finger=self.target_finger,
            left_delta=deltas["left"],
            right_delta=deltas["right"],
            contact_threshold=self.tuning.contact_force_delta_raw,
            abort_threshold=self.tuning.non_target_force_abort_raw,
        )
        if consensus.non_target_violations:
            # A rise on another CONTACT finger means a wrong fingertip
            # pair touched (§5 WRONG_FINGER_CONTACT); a rise anywhere
            # else (or mixed) is UNINTENDED_CONTACT.
            from rosclaw.twintouch.pairs import CONTACT_FINGERS

            rising = {v.split(".")[1].split(" ")[0] for v in consensus.non_target_violations}
            outcome = (
                OUTCOME_WRONG_FINGER_CONTACT
                if rising and rising <= set(CONTACT_FINGERS)
                else OUTCOME_UNINTENDED_CONTACT
            )
            return self._enter_recovery(
                outcome,
                "; ".join(consensus.non_target_violations),
            )
        return None

    # ---------------------------------------------------------------- step

    def step(self, obs: SupervisorObservation) -> SupervisorDecision:
        if self.state in TERMINAL_STATES:
            return SupervisorDecision(kind=DECISION_NONE, note=f"terminal {self.state}")

        if self.state in RECOVERY_STATES:
            return self._recovery_step(obs)

        guard = self._global_guard(obs)
        if guard is not None:
            return guard

        deltas = self._deltas(obs)
        self._update_peaks(deltas)
        self._update_visual_min(obs)
        self._update_saturation(obs)
        consensus = bilateral_force_consensus(
            target_finger=self.target_finger,
            left_delta=deltas["left"],
            right_delta=deltas["right"],
            contact_threshold=self.tuning.contact_force_delta_raw,
            abort_threshold=self.tuning.non_target_force_abort_raw,
        )

        handler = {
            SAFE_RESET: self._step_safe_reset,
            PAIR_SELECTED: self._step_pair_selected,
            PEER_READY: self._step_peer_ready,
            VISUAL_ALIGN: self._step_visual_align,
            COARSE_APPROACH: self._step_coarse,
            FINE_APPROACH: self._step_fine,
            CONTACT_CANDIDATE: self._step_candidate,
            CONTACT_CONFIRMED: self._step_confirmed,
            DWELL: self._step_dwell,
            RELEASE: self._step_release,
            CLEARANCE_VERIFIED: self._step_clearance,
        }[self.state]
        return handler(obs, deltas, consensus)

    # --------------------------------------------------- state handlers

    def _step_safe_reset(self, obs, deltas, consensus) -> SupervisorDecision:
        # both hands must be near-open with no force anomalies (the
        # global guard already cleared force; check posture coarsely)
        for side, hand in (("left", obs.left), ("right", obs.right)):
            assert hand is not None
            for joint in ("little", "ring", "middle", "index"):
                angle = hand.angle_actual.get(joint)
                if angle is not None and angle < 700:
                    # not near open — retreat first, not an anomaly
                    self._transition(RETREAT_BOTH)
                    self.track.anomaly = None
                    return SupervisorDecision(
                        kind=DECISION_RETREAT,
                        note=f"{side}.{joint} at {angle} not near safe_open — retreat before pair",
                    )
        self._transition(PAIR_SELECTED)
        return SupervisorDecision(kind=DECISION_NONE, note="both hands at safe_open")

    def _step_pair_selected(self, obs, deltas, consensus) -> SupervisorDecision:
        if not self.reachability_calibrated:
            # §12.2: unvalidated contact pairs are never attempted.
            return self._enter_recovery(
                OUTCOME_NO_CONTACT,
                f"pair {self.pair_id} has no T1-calibrated envelope — never approached",
            )
        self._transition(PEER_READY)
        return SupervisorDecision(kind=DECISION_NONE, note=f"pair {self.pair_id} selected")

    def _step_peer_ready(self, obs, deltas, consensus) -> SupervisorDecision:
        # Both hands must be live AND reporting force channels — a hand
        # whose force channels are all dead cannot feel the peer, so it
        # is NOT ready (v4 §5 PEER_NOT_READY).
        for side, hand in (("left", obs.left), ("right", obs.right)):
            assert hand is not None
            if all(v is None for v in hand.force_act.values()):
                return self._enter_recovery(
                    OUTCOME_PEER_NOT_READY,
                    f"{side} force channels dead — cannot feel the peer",
                )
        self._transition(VISUAL_ALIGN)
        return SupervisorDecision(kind=DECISION_NONE, note="peer states live")

    def _step_visual_align(self, obs, deltas, consensus) -> SupervisorDecision:
        visual = obs.visual
        assert visual is not None
        if not (visual.left_cluster_ok and visual.right_cluster_ok):
            return self._enter_recovery(
                OUTCOME_STALE_OBSERVATION,
                "cannot see both hands — alignment impossible",
            )
        if visual.pair_identity_confirmed is False:
            return self._enter_recovery(
                OUTCOME_WRONG_FINGER_CONTACT,
                f"camera does not confirm pair identity {self.pair_id}",
            )
        if visual.min_distance_m is None:
            return SupervisorDecision(kind=DECISION_NONE, note="waiting for distance estimate")
        self._transition(COARSE_APPROACH)
        return SupervisorDecision(kind=DECISION_NONE, note="visual pair aligned")

    def _approach_step(self, *, fine: bool) -> SupervisorDecision:
        step_raw = self.tuning.fine_step_raw if fine else self.tuning.coarse_step_raw
        sides = self._active_sides()
        if fine:
            self.track.fine_steps += 1
        else:
            self.track.coarse_steps += 1
        return SupervisorDecision(
            kind=DECISION_ISSUE_STEP,
            note=f"{'fine' if fine else 'coarse'} step {step_raw} raw on {sides}",
            step={
                "sides": sides,
                "joints": {self.target_finger: -step_raw},  # raw decreases = curl toward peer
                "step_raw": step_raw,
            },
        )

    def _step_coarse(self, obs, deltas, consensus) -> SupervisorDecision:
        # target force rising already => contact came before the fine zone
        if consensus.left_target_rise or consensus.right_target_rise:
            return self._enter_recovery(
                OUTCOME_EARLY_CONTACT,
                f"target force rise during coarse approach "
                f"(L {consensus.left_target_delta:.0f} R {consensus.right_target_delta:.0f})",
            )
        visual = obs.visual
        assert visual is not None
        if visual.min_distance_m is not None and visual.min_distance_m <= (
            self.tuning.coarse_to_fine_distance_m
        ):
            self._transition(FINE_APPROACH)
            return SupervisorDecision(kind=DECISION_NONE, note="entering fine zone")
        if self.track.coarse_steps >= self.tuning.max_coarse_steps:
            return self._enter_recovery(
                OUTCOME_NO_CONTACT,
                f"coarse budget {self.tuning.max_coarse_steps} exhausted at "
                f"distance {visual.min_distance_m}",
            )
        return self._approach_step(fine=False)

    def _step_fine(self, obs, deltas, consensus) -> SupervisorDecision:
        if consensus.left_target_rise or consensus.right_target_rise:
            self.track.contact_candidate_ts = obs.ts_s
            self._transition(CONTACT_CANDIDATE)
            return SupervisorDecision(kind=DECISION_NONE, note="first target force rise")
        if self.track.fine_steps >= self.tuning.max_fine_steps:
            return self._enter_recovery(
                OUTCOME_NO_CONTACT,
                f"fine budget {self.tuning.max_fine_steps} exhausted at "
                f"distance {obs.visual.min_distance_m if obs.visual else None}",
            )
        return self._approach_step(fine=True)

    def _step_candidate(self, obs, deltas, consensus) -> SupervisorDecision:
        visual = obs.visual
        visual_near = (
            visual is not None
            and visual.min_distance_m is not None
            and visual.min_distance_m <= self.tuning.near_contact_m
        )
        visual_far = (
            visual is not None
            and visual.min_distance_m is not None
            and visual.min_distance_m > self.tuning.visual_conflict_distance_m
        )
        # conflict: forces rise but the camera says the pair is far apart
        if visual_far and (consensus.left_target_rise or consensus.right_target_rise):
            return self._enter_recovery(
                OUTCOME_VISUAL_FORCE_CONFLICT,
                f"force rise at visual distance {visual.min_distance_m:.3f} m",
            )
        both_rise = consensus.left_target_rise and consensus.right_target_rise
        # motion response: commanded steps continue but actual position
        # saturates (target finger of the active side stopped moving)
        motion_response = self._motion_response_saturated(obs)
        if both_rise and (visual_near or motion_response):
            self.track.contact_confirmed_ts = obs.ts_s
            self.track.one_sided_frames = 0
            self._transition(CONTACT_CONFIRMED)
            return SupervisorDecision(
                kind=DECISION_NONE,
                note=f"CONTACT_CONFIRMED {self.pair_id} "
                f"(visual_near={visual_near}, motion_response={motion_response})",
            )
        if not both_rise:
            self.track.one_sided_frames += 1
            if self.track.one_sided_frames >= self.tuning.one_sided_frame_budget:
                return self._enter_recovery(
                    OUTCOME_ONE_SIDED_FORCE,
                    f"one-sided force for {self.track.one_sided_frames} frames "
                    f"(L {consensus.left_target_delta:.0f} R {consensus.right_target_delta:.0f})",
                )
        return SupervisorDecision(kind=DECISION_NONE, note="awaiting bilateral consensus")

    def _motion_response_saturated(self, obs: SupervisorObservation) -> bool:
        """Position-saturation evidence (§6.3): the active side's target
        finger stopped moving while fine steps continued — commands keep
        changing but the external geometry no longer does.  Requires the
        angle history the track accumulates per cycle."""
        if self.track.saturated_frames >= 2:
            return True
        return False

    def _update_saturation(self, obs: SupervisorObservation) -> None:
        """Track per-cycle angle deltas on the active target finger while
        steps are being issued (fine approach / candidate)."""
        if self.state not in (FINE_APPROACH, CONTACT_CANDIDATE):
            self.track.last_angles = {}
            self.track.saturated_frames = 0
            return
        if self.track.coarse_steps + self.track.fine_steps == 0:
            return
        any_saturated = False
        for side in self._active_sides():
            hand = obs.left if side == "left" else obs.right
            if hand is None or not hand.ok:
                continue
            angle = hand.angle_actual.get(self.target_finger)
            last = self.track.last_angles.get(side)
            if angle is not None and last is not None:
                if abs(angle - last) <= self.tuning.saturation_delta_raw:
                    any_saturated = True
            if angle is not None:
                self.track.last_angles[side] = angle
        if any_saturated:
            self.track.saturated_frames += 1
        else:
            self.track.saturated_frames = 0

    def _step_confirmed(self, obs, deltas, consensus) -> SupervisorDecision:
        self.track.dwell_start_ts = obs.ts_s
        self._transition(DWELL)
        return SupervisorDecision(kind=DECISION_NONE, note="dwell begins")

    def _step_dwell(self, obs, deltas, consensus) -> SupervisorDecision:
        assert self.track.dwell_start_ts is not None
        elapsed_ms = (obs.ts_s - self.track.dwell_start_ts) * 1000.0
        if elapsed_ms >= self.tuning.dwell_ms:
            self.track.dwell_actual_ms = elapsed_ms
            self._transition(RELEASE)
            return SupervisorDecision(kind=DECISION_NONE, note="dwell complete — release")
        return SupervisorDecision(kind=DECISION_NONE, note="dwelling")

    def _step_release(self, obs, deltas, consensus) -> SupervisorDecision:
        # released = both target forces back near baseline
        released = (
            abs(consensus.left_target_delta) <= self.tuning.release_force_epsilon_raw
            and abs(consensus.right_target_delta) <= self.tuning.release_force_epsilon_raw
        )
        if released:
            self._transition(CLEARANCE_VERIFIED)
            return SupervisorDecision(kind=DECISION_NONE, note="forces returned to baseline")
        if self.track.release_steps >= self.tuning.max_release_steps:
            return self._enter_recovery(
                OUTCOME_RELEASE_FAILED,
                f"forces still L {consensus.left_target_delta:.0f} "
                f"R {consensus.right_target_delta:.0f} after "
                f"{self.tuning.max_release_steps} release steps",
            )
        self.track.release_steps += 1
        return SupervisorDecision(
            kind=DECISION_ISSUE_STEP,
            note=f"release step +{self.tuning.release_margin_raw} raw on {self._active_sides()}",
            step={
                "sides": self._active_sides(),
                "joints": {self.target_finger: self.tuning.release_margin_raw},
                "step_raw": self.tuning.release_margin_raw,
            },
        )

    def _step_clearance(self, obs, deltas, consensus) -> SupervisorDecision:
        visual = obs.visual
        if (
            visual is not None
            and visual.min_distance_m is not None
            and (visual.min_distance_m <= self.tuning.near_contact_m)
        ):
            return SupervisorDecision(kind=DECISION_NONE, note="awaiting visual clearance")
        self._transition(EPISODE_COMMITTED)
        return SupervisorDecision(
            kind=DECISION_COMMIT,
            note=f"episode committed: {self.pair_id}",
            receipt=self._build_receipt(
                outcome=OUTCOME_CONTACT_CONFIRMED, obs=obs, consensus=consensus
            ),
        )

    # -------------------------------------------------------------- recovery

    def _recovery_step(self, obs: SupervisorObservation) -> SupervisorDecision:
        if self.state == STOP_APPROACH:
            self._transition(RETREAT_BOTH)
            return SupervisorDecision(
                kind=DECISION_RETREAT,
                note=f"retreat both hands ({self.track.anomaly})",
            )
        if self.state == RETREAT_BOTH:
            self._transition(VERIFY_CLEAR)
            return SupervisorDecision(kind=DECISION_NONE, note="verify clearance")
        # VERIFY_CLEAR: hands retreated when target forces near baseline
        # (or unknown — a hand we cannot read is retreated by decree of
        # the transport guard, and we do not wait forever on forces)
        deltas = self._deltas(obs)
        left_delta = deltas["left"].get(self.target_finger, 0.0)
        right_delta = deltas["right"].get(self.target_finger, 0.0)
        forces_clear = (
            abs(left_delta) <= self.tuning.release_force_epsilon_raw
            and abs(right_delta) <= self.tuning.release_force_epsilon_raw
        )
        transport_down = (
            obs.left is None or not obs.left.ok or obs.right is None or not obs.right.ok
        )
        if not (forces_clear or transport_down):
            return SupervisorDecision(kind=DECISION_NONE, note="awaiting force clearance")
        self._transition(RECORD_FAILURE)
        receipt = self._build_receipt(
            outcome=self.track.anomaly or OUTCOME_TRANSPORT_FAILURE, obs=obs, consensus=None
        )
        return SupervisorDecision(
            kind=DECISION_FAIL,
            note=f"RECORD_FAILURE: {self.track.anomaly} ({self.track.anomaly_detail})",
            receipt=receipt,
        )

    # -------------------------------------------------------------- receipt

    def _build_receipt(
        self,
        *,
        outcome: str,
        obs: SupervisorObservation,
        consensus: BilateralForceEvidence | None,
    ) -> InteractionReceipt:
        latency_ms = None
        if (
            self.track.contact_confirmed_ts is not None
            and self.track.contact_candidate_ts is not None
        ):
            latency_ms = (
                self.track.contact_confirmed_ts - self.track.contact_candidate_ts
            ) * 1000.0
        confirmed = outcome == OUTCOME_CONTACT_CONFIRMED
        return InteractionReceipt(
            interaction_id=self.interaction_id,
            pair_id=self.pair_id,
            left_action_receipt=None,  # runner fills gateway receipt ids
            right_action_receipt=None,
            intended_contact=self.pair_id,
            observed_contact=self.pair_id if confirmed else None,
            contact_confirmed=confirmed,
            wrong_finger_contact=outcome == OUTCOME_WRONG_FINGER_CONTACT,
            unintended_contact=outcome == OUTCOME_UNINTENDED_CONTACT,
            left_force_peak=self.track.left_force_peak,
            right_force_peak=self.track.right_force_peak,
            visual_distance_min_m=self.track.visual_distance_min_m,
            contact_latency_ms=latency_ms,
            start_skew_ms=None,  # gateway's field
            dwell_ms=self.track.dwell_actual_ms,
            clearance_verified=confirmed,
            outcome=outcome,
            evidence_refs=tuple(self.history),
        )
