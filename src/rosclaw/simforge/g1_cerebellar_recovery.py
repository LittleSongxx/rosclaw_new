"""Contact-gated post-kick recovery motion for the G1 GoalForge sandbox.

The qualified RoboNaldo policy remains responsible for the kick.  This module
only reshapes the late recovery segment after observed ball contact and kick-
foot landing.  It never produces torque commands or opens a robot transport.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    G1_DDS_JOINT_NAMES,
    hash_bytes,
    hash_json,
)

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class G1CerebellarRecoveryConfig:
    """Bounded target-space recovery segment discovered by matched SIM A/B.

    ``start_policy_frame=420`` begins after the original contact and recovery
    step.  A smooth 100-frame blend avoids the unstable discontinuity produced
    by immediately switching the whole-body policy to a standing pose.
    """

    start_policy_frame: int = 420
    blend_frames: int = 100
    standing_pose_blend: float = 0.30
    roll_posture_bias_rad: float = -0.05
    contact_required: bool = True
    kick_foot_landing_required: bool = True
    minimum_calibrated_support_friction: float = 0.95
    maximum_calibrated_control_latency_ms: float = 5.0
    minimum_calibrated_disturbance_n: float = 70.0
    maximum_calibrated_disturbance_n: float = 80.0

    def __post_init__(self) -> None:
        if self.start_policy_frame < 0:
            raise ValueError("start_policy_frame must be non-negative")
        if self.blend_frames <= 0:
            raise ValueError("blend_frames must be positive")
        if not 0.0 <= self.standing_pose_blend <= 0.50:
            raise ValueError("standing_pose_blend must be in [0, 0.50]")
        if not math.isfinite(self.roll_posture_bias_rad) or not (
            -0.08 <= self.roll_posture_bias_rad <= 0.08
        ):
            raise ValueError("roll_posture_bias_rad must be finite and in [-0.08, 0.08]")
        if not 0.0 < self.minimum_calibrated_support_friction <= 2.0:
            raise ValueError("minimum calibrated support friction must be in (0, 2]")
        if not 0.0 <= self.maximum_calibrated_control_latency_ms <= 100.0:
            raise ValueError("maximum calibrated control latency must be in [0, 100]")
        if not (
            0.0 < self.minimum_calibrated_disturbance_n <= self.maximum_calibrated_disturbance_n
        ):
            raise ValueError("calibrated disturbance bounds must be positive and ordered")


@dataclass(frozen=True)
class G1CerebellarRecoveryEffect:
    target: np.ndarray
    active: bool
    blend_fraction: float
    contact_latched: bool
    kick_foot_landing_latched: bool


@dataclass(frozen=True)
class G1CerebellarRecoveryReceipt:
    controller_hash: str
    body_hash: str
    motion_hash: str
    standing_pose_hash: str
    regime_commitment: str
    regime_eligible: bool
    regime_reasons: tuple[str, ...]
    contact_latched: bool
    kick_foot_landing_latched: bool
    activation_policy_frame: int | None
    activation_time_sec: float | None
    peak_blend_fraction: float
    strict_replay: bool
    evidence_domain: str
    config: dict[str, Any]
    schema_version: str = "rosclaw.g1_goalforge.cerebellar_recovery_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["regime_reasons"] = list(self.regime_reasons)
        return value


class G1CerebellarRecoveryController:
    """Blend the late kick target toward a qualified, damped standing pose.

    The controller is stateful only for causal gates and evidence.  It is
    transparent until ball contact and a subsequent right-foot landing are
    observed.  The existing high-rate balance reflex remains the closed-loop
    disturbance layer; this controller supplies the slower recovery segment.
    """

    def __init__(
        self,
        *,
        body_hash: str,
        motion_hash: str,
        regime_commitment: str,
        regime_eligible: bool,
        regime_reasons: tuple[str, ...],
        standing_pose: np.ndarray,
        config: G1CerebellarRecoveryConfig | None = None,
    ) -> None:
        for label, value in (
            ("body_hash", body_hash),
            ("motion_hash", motion_hash),
            ("regime_commitment", regime_commitment),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        pose = np.asarray(standing_pose, dtype=np.float64)
        if pose.shape != (len(G1_DDS_JOINT_NAMES),) or not np.all(np.isfinite(pose)):
            raise ValueError("standing_pose must be a finite 29-joint G1 target")
        self.body_hash = body_hash
        self.motion_hash = motion_hash
        self.regime_commitment = regime_commitment
        self.regime_eligible = bool(regime_eligible)
        self.regime_reasons = tuple(str(reason) for reason in regime_reasons)
        if self.regime_eligible == bool(self.regime_reasons):
            raise ValueError("eligible regimes must have no rejection reasons")
        self.standing_pose = pose.copy()
        self.standing_pose.setflags(write=False)
        self.standing_pose_hash = hash_bytes(np.ascontiguousarray(pose).tobytes())
        self.config = config or G1CerebellarRecoveryConfig()
        self._roll_pattern = _roll_posture_pattern()
        self.reset()

    @property
    def controller_hash(self) -> str:
        return hash_json(
            {
                "controller_type": "g1_cerebellar_post_kick_recovery",
                "version": 1,
                "body_hash": self.body_hash,
                "motion_hash": self.motion_hash,
                "standing_pose_hash": self.standing_pose_hash,
                "regime_commitment": self.regime_commitment,
                "regime_eligible": self.regime_eligible,
                "regime_reasons": list(self.regime_reasons),
                "config": asdict(self.config),
            }
        )

    def require_compatible(
        self,
        *,
        body_hash: str,
        motion_hash: str,
        regime_commitment: str,
    ) -> None:
        if body_hash != self.body_hash:
            raise ValueError("cerebellar recovery Body hash mismatch")
        if motion_hash != self.motion_hash:
            raise ValueError("cerebellar recovery motion hash mismatch")
        if regime_commitment != self.regime_commitment:
            raise ValueError("cerebellar recovery regime commitment mismatch")

    def reset(self) -> None:
        self._contact_latched = False
        self._landing_latched = False
        self._activation_policy_frame: int | None = None
        self._activation_time_sec: float | None = None
        self._peak_blend_fraction = 0.0

    def adapt_target(
        self,
        *,
        target: np.ndarray,
        policy_frame: int,
        timestamp_sec: float,
        ball_contact_detected: bool,
        left_support: bool,
        right_support: bool,
    ) -> G1CerebellarRecoveryEffect:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != self.standing_pose.shape or not np.all(np.isfinite(value)):
            raise ValueError("recovery target must match the finite G1 standing pose")
        if policy_frame < 0 or not math.isfinite(timestamp_sec):
            raise ValueError("recovery phase inputs must be finite and non-negative")

        self._contact_latched = self._contact_latched or bool(ball_contact_detected)
        if self._contact_latched and right_support:
            self._landing_latched = True
        eligible = (
            self.regime_eligible
            and (self._contact_latched or not self.config.contact_required)
            and (self._landing_latched or not self.config.kick_foot_landing_required)
            and policy_frame >= self.config.start_policy_frame
        )
        if not eligible:
            return G1CerebellarRecoveryEffect(
                target=value.copy(),
                active=False,
                blend_fraction=0.0,
                contact_latched=self._contact_latched,
                kick_foot_landing_latched=self._landing_latched,
            )

        linear = min(
            1.0,
            max(
                0.0,
                (policy_frame - self.config.start_policy_frame) / self.config.blend_frames,
            ),
        )
        fraction = linear * linear * (3.0 - 2.0 * linear)
        standing_weight = fraction * self.config.standing_pose_blend
        adapted = (
            (1.0 - standing_weight) * value
            + standing_weight * self.standing_pose
            + fraction * self.config.roll_posture_bias_rad * self._roll_pattern
        )
        active = fraction > 0.0
        if active and self._activation_policy_frame is None:
            self._activation_policy_frame = policy_frame
            self._activation_time_sec = timestamp_sec
        self._peak_blend_fraction = max(self._peak_blend_fraction, fraction)
        return G1CerebellarRecoveryEffect(
            target=adapted,
            active=active,
            blend_fraction=fraction,
            contact_latched=self._contact_latched,
            kick_foot_landing_latched=self._landing_latched,
        )

    def build_receipt(
        self,
        *,
        strict_replay: bool,
        evidence_domain: str = "SIM",
    ) -> G1CerebellarRecoveryReceipt:
        return G1CerebellarRecoveryReceipt(
            controller_hash=self.controller_hash,
            body_hash=self.body_hash,
            motion_hash=self.motion_hash,
            standing_pose_hash=self.standing_pose_hash,
            regime_commitment=self.regime_commitment,
            regime_eligible=self.regime_eligible,
            regime_reasons=self.regime_reasons,
            contact_latched=self._contact_latched,
            kick_foot_landing_latched=self._landing_latched,
            activation_policy_frame=self._activation_policy_frame,
            activation_time_sec=self._activation_time_sec,
            peak_blend_fraction=self._peak_blend_fraction,
            strict_replay=strict_replay,
            evidence_domain=evidence_domain,
            config=asdict(self.config),
        )


def evaluate_g1_cerebellar_recovery_regime(
    *,
    support_friction: float,
    control_latency_ms: float,
    disturbance_n: float,
    config: G1CerebellarRecoveryConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Fail closed outside the SIM regimes covered by matched validation."""

    values = (support_friction, control_latency_ms, disturbance_n)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("recovery regime inputs must be finite")
    reasons = []
    if support_friction < config.minimum_calibrated_support_friction:
        reasons.append("support_friction_below_calibrated_range")
    if control_latency_ms > config.maximum_calibrated_control_latency_ms:
        reasons.append("control_latency_above_calibrated_range")
    magnitude = abs(disturbance_n)
    if 0.0 < magnitude < config.minimum_calibrated_disturbance_n:
        reasons.append("disturbance_below_calibrated_recovery_range")
    if magnitude > config.maximum_calibrated_disturbance_n:
        reasons.append("disturbance_above_calibrated_recovery_range")
    return not reasons, tuple(reasons)


def _roll_posture_pattern() -> np.ndarray:
    index = {name: position for position, name in enumerate(G1_DDS_JOINT_NAMES)}
    pattern: np.ndarray = np.zeros(len(G1_DDS_JOINT_NAMES), dtype=np.float64)
    for name in ("left_hip_roll_joint", "right_hip_roll_joint"):
        pattern[index[name]] = 1.0
    for name in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
        pattern[index[name]] = -0.65
    pattern[index["waist_roll_joint"]] = 0.45
    pattern.setflags(write=False)
    return pattern


__all__ = [
    "G1CerebellarRecoveryConfig",
    "G1CerebellarRecoveryController",
    "G1CerebellarRecoveryEffect",
    "G1CerebellarRecoveryReceipt",
    "evaluate_g1_cerebellar_recovery_regime",
]
