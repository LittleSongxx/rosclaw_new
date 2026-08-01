"""Action Effect Gate (v4 §8, PR-TT-3) — did the command change the RIGHT body?

v3 exposed the hole this gate closes: a hand commanded through the wrong
Modbus slave id never moved, yet the system judged the static hand
"settled" — a static hand is trivially stable.  No downstream stage may
ever again confuse "nothing happened" with "command succeeded".

Per hand, per action, the gate cross-examines TWO independent evidence
channels:

* internal — servo telemetry: did angle_actual on the COMMANDED joints
  follow the command (direction + magnitude)?
* external — camera: did the visual cluster in THIS body's ROI move,
  and did the OTHER body's ROI stay put (attribution)?

Verdicts (v4 §8):

* EFFECT_CONFIRMED  — internal followed AND external moved (move), or
                      internal held AND external held (hold)
* NO_EFFECT         — command issued, neither channel shows an effect
                      (the v3 wrong-slave case)
* WRONG_BODY_EFFECT — this body static, the OTHER body moved
* INTERNAL_ONLY     — servo followed, camera saw nothing (occluded?
                      stale?  flagged, not trusted)
* EXTERNAL_ONLY     — camera saw this body move, servo did not follow
                      (passive body disturbed — PEER_DISTURBANCE
                      precursor)
* UNKNOWN           — a channel's evidence is missing; never guessed

Command TYPE matters: a "hold" command is confirmed by STABILITY, a
"move" command by following.  The passive hand's hold is verified just
as strictly as the active hand's move — a disturbed hold is an effect,
not a confirmation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rosclaw.practice.physical_observation import canonical_hash

SCHEMA_VERSION = "rosclaw.action_effect.v1"

EFFECT_CONFIRMED = "EFFECT_CONFIRMED"
NO_EFFECT = "NO_EFFECT"
WRONG_BODY_EFFECT = "WRONG_BODY_EFFECT"
INTERNAL_ONLY = "INTERNAL_ONLY"
EXTERNAL_ONLY = "EXTERNAL_ONLY"
EFFECT_UNKNOWN = "UNKNOWN"

ALL_EFFECTS = frozenset(
    {EFFECT_CONFIRMED, NO_EFFECT, WRONG_BODY_EFFECT, INTERNAL_ONLY, EXTERNAL_ONLY, EFFECT_UNKNOWN}
)

COMMAND_MOVE = "move"
COMMAND_HOLD = "hold"


@dataclass(frozen=True)
class EffectThresholds:
    """Noise-aware bounds (RH56: settle noise ≤ ~5 raw; PE-3 centroid
    settle threshold 0.01 m)."""

    min_position_delta_raw: int = 8  # move must exceed settle noise
    hold_tolerance_raw: int = 15  # hold may drift within this band
    min_visual_move_m: float = 0.008  # visible motion beyond centroid jitter


@dataclass(frozen=True)
class JointCommand:
    """What one body was told to do, plus its type (move | hold)."""

    action_id: str
    body_id: str
    command_type: str  # COMMAND_MOVE | COMMAND_HOLD
    targets: dict[str, int]  # joint -> raw target
    issued_at_s: float
    window_s: float

    def validate(self) -> list[str]:
        violations: list[str] = []
        if self.command_type not in (COMMAND_MOVE, COMMAND_HOLD):
            violations.append(f"command_type {self.command_type!r} unknown")
        if not self.targets:
            violations.append("targets missing")
        if self.window_s <= 0:
            violations.append("window_s must be positive")
        return violations


@dataclass(frozen=True)
class TelemetryPoint:
    ts_s: float
    angle_actual: dict[str, int | None]


@dataclass(frozen=True)
class VisualSample:
    """One body's visual cluster snapshot (PE-3 semantics)."""

    ok: bool
    centroid_3d: tuple[float, float, float] | None


@dataclass(frozen=True)
class PhysicalActionEffectReceipt:
    action_id: str
    body_id: str
    command_type: str

    commanded_joints: tuple[str, ...]
    internal_delta_max_raw: int | None  # max |delta| over commanded joints
    internal_followed: bool | None  # move: followed; hold: held; None: no data
    external_displacement_m: float | None
    external_moved: bool | None
    other_body_displacement_m: float | None
    other_body_moved: bool | None

    verdict: str
    reasons: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        violations: list[str] = []
        if self.verdict not in ALL_EFFECTS:
            violations.append(f"verdict {self.verdict!r} unknown")
            return violations
        if self.verdict == EFFECT_CONFIRMED:
            if self.internal_followed is not True:
                violations.append("EFFECT_CONFIRMED requires internal_followed=True")
            # external_moved is PURE semantics (displacement beyond
            # jitter): a move is confirmed by motion, a hold by stillness.
            if self.command_type == COMMAND_MOVE and self.external_moved is not True:
                violations.append("move EFFECT_CONFIRMED requires external_moved=True")
            if self.command_type == COMMAND_HOLD and self.external_moved is not False:
                violations.append("hold EFFECT_CONFIRMED requires external_moved=False")
        if self.verdict == NO_EFFECT and self.internal_followed is True:
            violations.append("NO_EFFECT contradicts internal_followed=True")
        if self.verdict == WRONG_BODY_EFFECT and self.other_body_moved is not True:
            violations.append("WRONG_BODY_EFFECT requires other_body_moved=True")
        return violations

    def receipt_hash(self) -> str:
        return canonical_hash(self.to_record(), prefix="efx")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "action_id": self.action_id,
            "body_id": self.body_id,
            "command_type": self.command_type,
            "commanded_joints": list(self.commanded_joints),
            "internal_delta_max_raw": self.internal_delta_max_raw,
            "internal_followed": self.internal_followed,
            "external_displacement_m": self.external_displacement_m,
            "external_moved": self.external_moved,
            "other_body_displacement_m": self.other_body_displacement_m,
            "other_body_moved": self.other_body_moved,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
        }


def _displacement(a: VisualSample, b: VisualSample) -> float | None:
    if not (a.ok and b.ok and a.centroid_3d and b.centroid_3d):
        return None
    return math.dist(a.centroid_3d, b.centroid_3d)


def evaluate_effect(
    command: JointCommand,
    telemetry: list[TelemetryPoint],
    *,
    visual_before: VisualSample,
    visual_after: VisualSample,
    other_before: VisualSample,
    other_after: VisualSample,
    thresholds: EffectThresholds | None = None,
) -> PhysicalActionEffectReceipt:
    """Cross-examine internal vs external evidence for ONE body's action.

    ``telemetry`` must cover the command window (first point ≈ before,
    last ≈ after).  Missing channels degrade the verdict to UNKNOWN —
    they never upgrade it.
    """
    th = thresholds or EffectThresholds()
    reasons: list[str] = []
    joints = tuple(command.targets)

    # ---- internal channel
    internal_delta_max: int | None = None
    internal_followed: bool | None = None
    if len(telemetry) < 2:
        reasons.append("telemetry window missing — internal channel unknown")
    else:
        before, after = telemetry[0], telemetry[-1]
        deltas: dict[str, int] = {}
        for joint in joints:
            b, a = before.angle_actual.get(joint), after.angle_actual.get(joint)
            if b is None or a is None:
                continue
            deltas[joint] = int(a) - int(b)
        if not deltas:
            reasons.append("no commanded joint has before/after telemetry")
        else:
            internal_delta_max = max(abs(d) for d in deltas.values())
            if command.command_type == COMMAND_MOVE:
                # follow = at least one commanded joint moved beyond noise
                # TOWARD its target (sign agreement with target - before)
                followed_joints = []
                for joint, delta in deltas.items():
                    b = before.angle_actual.get(joint)
                    target = command.targets[joint]
                    assert b is not None
                    direction = 1 if target > b else (-1 if target < b else 0)
                    if direction != 0 and delta * direction >= th.min_position_delta_raw:
                        followed_joints.append(joint)
                    elif direction == 0:
                        followed_joints.append(joint)  # already at target
                internal_followed = bool(followed_joints)
                if not internal_followed:
                    reasons.append(
                        f"no commanded joint followed (max |delta| {internal_delta_max} raw)"
                    )
            else:  # hold
                internal_followed = internal_delta_max <= th.hold_tolerance_raw
                if not internal_followed:
                    reasons.append(
                        f"hold drifted {internal_delta_max} raw > tolerance {th.hold_tolerance_raw}"
                    )

    # ---- external channels
    ext_disp = _displacement(visual_before, visual_after)
    other_disp = _displacement(other_before, other_after)
    if ext_disp is None:
        reasons.append("own-body visual unknown — external channel unknown")
    if other_disp is None:
        reasons.append("other-body visual unknown — attribution unknown")
    # PURE semantics, command-independent: did this body visibly move?
    external_moved = None if ext_disp is None else ext_disp >= th.min_visual_move_m
    other_moved = None if other_disp is None else other_disp >= th.min_visual_move_m

    # ---- verdict matrix (per command type)
    if internal_followed is None or external_moved is None:
        verdict = EFFECT_UNKNOWN
        # A WRONG_BODY signal survives missing own-visual: if the servo
        # did NOT follow but the OTHER body demonstrably moved, that is
        # too specific to bury in UNKNOWN.
        if internal_followed is False and other_moved is True:
            verdict = WRONG_BODY_EFFECT
            reasons.append("own visual missing but wrong-body motion is explicit")
    elif not internal_followed and other_moved is True:
        verdict = WRONG_BODY_EFFECT
        reasons.append("this body static while the other body moved")
    elif command.command_type == COMMAND_MOVE:
        if internal_followed and external_moved:
            verdict = EFFECT_CONFIRMED
        elif internal_followed and not external_moved:
            verdict = INTERNAL_ONLY
            reasons.append("servo followed but camera saw no change in this body")
        elif not internal_followed and external_moved:
            verdict = EXTERNAL_ONLY
            reasons.append("camera saw this body move without servo following (disturbed?)")
        else:
            verdict = NO_EFFECT
            reasons.append("command issued; neither channel shows an effect")
    else:  # hold
        if internal_followed and not external_moved:
            verdict = EFFECT_CONFIRMED
        elif internal_followed and external_moved:
            verdict = EXTERNAL_ONLY
            reasons.append("servo held but camera saw this body move (mount shift / external push)")
        elif not internal_followed and external_moved:
            verdict = NO_EFFECT
            reasons.append("hold failed: both channels agree the body moved")
        else:
            verdict = INTERNAL_ONLY
            reasons.append("servo drifted but camera saw this body static")

    return PhysicalActionEffectReceipt(
        action_id=command.action_id,
        body_id=command.body_id,
        command_type=command.command_type,
        commanded_joints=joints,
        internal_delta_max_raw=internal_delta_max,
        internal_followed=internal_followed,
        external_displacement_m=None if ext_disp is None else round(ext_disp, 5),
        external_moved=external_moved,
        other_body_displacement_m=None if other_disp is None else round(other_disp, 5),
        other_body_moved=other_moved,
        verdict=verdict,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class BimanualEffectGateVerdict:
    """The §8 precondition for entering the Contact Supervisor:

    left EFFECT_CONFIRMED AND right EFFECT_CONFIRMED AND visual changes
    attributed to the correct bodies.  Anything else blocks with named
    reasons — a gate that blocks is working, not failing."""

    proceed: bool
    violations: tuple[str, ...]
    left_receipt: PhysicalActionEffectReceipt
    right_receipt: PhysicalActionEffectReceipt

    def to_record(self) -> dict[str, Any]:
        return {
            "proceed": self.proceed,
            "violations": list(self.violations),
            "left_receipt": self.left_receipt.to_record(),
            "right_receipt": self.right_receipt.to_record(),
        }


def bimanual_effect_gate(
    left_receipt: PhysicalActionEffectReceipt,
    right_receipt: PhysicalActionEffectReceipt,
) -> BimanualEffectGateVerdict:
    violations: list[str] = []
    for side, receipt in (("left", left_receipt), ("right", right_receipt)):
        if receipt.verdict == EFFECT_CONFIRMED:
            continue
        violations.append(
            f"{side}: {receipt.verdict} ({'; '.join(receipt.reasons) or 'no reason'})"
        )
        if receipt.verdict == WRONG_BODY_EFFECT:
            violations.append(
                f"{side}: command affected the WRONG body — identity fault, stop the line"
            )
    return BimanualEffectGateVerdict(
        proceed=not violations,
        violations=tuple(violations),
        left_receipt=left_receipt,
        right_receipt=right_receipt,
    )
