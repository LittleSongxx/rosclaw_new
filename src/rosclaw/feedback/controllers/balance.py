"""Phase-gated G1 balance reflex for kick and recovery motion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from rosclaw.feedback.contracts import FeedbackFrame, canonical_hash
from rosclaw.feedback.controllers.phase import LatchedPhaseGate


@dataclass(frozen=True)
class G1BalanceReflexConfig:
    """Small joint-target corrections; final limits live in the safety projector."""

    roll_kp: float = 0.18
    roll_kd: float = 0.008
    pitch_kp: float = 0.14
    pitch_kd: float = 0.006
    com_y_kp: float = 0.30
    slip_damping: float = 0.18
    trigger_roll_rad: float = 0.40
    trigger_pitch_rad: float = 0.45
    trigger_com_y_m: float = 0.13
    trigger_com_y_rate_mps: float = 0.72
    trigger_phase_end: float = 0.50
    active_phase_start: float = 0.32
    active_phase_end: float = 0.60
    fade_fraction: float = 0.03

    def __post_init__(self) -> None:
        if not 0.0 <= self.active_phase_start < self.active_phase_end <= 1.0:
            raise ValueError("active phase must be an ordered subset of [0, 1]")
        if not self.active_phase_start < self.trigger_phase_end < self.active_phase_end:
            raise ValueError("trigger_phase_end must be inside the active phase")
        if self.fade_fraction < 0.0:
            raise ValueError("fade_fraction must be non-negative")
        if (
            self.trigger_roll_rad <= 0.0
            or self.trigger_pitch_rad <= 0.0
            or self.trigger_com_y_m <= 0.0
            or self.trigger_com_y_rate_mps <= 0.0
        ):
            raise ValueError("reflex trigger thresholds must be positive")


class G1BalanceReflexController:
    """Correct torso/COM error without replacing the qualified kick prior.

    The controller emits target-position residuals for a deliberately small
    set of G1 joints.  It never writes torques, transports or safety policy.
    """

    _WAIST_ROLL = "joint:waist_roll_joint"
    _WAIST_PITCH = "joint:waist_pitch_joint"
    _LEFT_HIP_ROLL = "joint:left_hip_roll_joint"
    _RIGHT_HIP_ROLL = "joint:right_hip_roll_joint"
    _LEFT_ANKLE_ROLL = "joint:left_ankle_roll_joint"
    _RIGHT_ANKLE_ROLL = "joint:right_ankle_roll_joint"
    _LEFT_HIP_PITCH = "joint:left_hip_pitch_joint"
    _RIGHT_HIP_PITCH = "joint:right_hip_pitch_joint"
    _LEFT_ANKLE_PITCH = "joint:left_ankle_pitch_joint"
    _RIGHT_ANKLE_PITCH = "joint:right_ankle_pitch_joint"

    def __init__(self, config: G1BalanceReflexConfig | None = None) -> None:
        self.config = config or G1BalanceReflexConfig()
        self._gate = LatchedPhaseGate(
            active_start=self.config.active_phase_start,
            trigger_end=self.config.trigger_phase_end,
            active_end=self.config.active_phase_end,
            fade_fraction=self.config.fade_fraction,
        )

    @property
    def controller_hash(self) -> str:
        return canonical_hash(self.config_dict())

    def reset(self) -> None:
        self._gate.reset()

    def compute(
        self,
        frame: FeedbackFrame,
        base_action: Mapping[str, float],
    ) -> Mapping[str, float]:
        cfg = self.config
        roll = float(frame.actual.get("torso_roll", 0.0))
        pitch = float(frame.actual.get("torso_pitch", 0.0))
        com_y = float(frame.actual.get("com_y_relative", 0.0))
        com_y_rate = -float(frame.error.derivative.get("com_y_relative", 0.0))
        gain = self._gate.update(
            phase=frame.phase,
            trigger=(
                abs(roll) >= cfg.trigger_roll_rad
                or abs(pitch) >= cfg.trigger_pitch_rad
                or (
                    abs(com_y) >= cfg.trigger_com_y_m
                    and abs(com_y_rate) >= cfg.trigger_com_y_rate_mps
                )
            ),
        )
        if gain <= 0.0:
            return {}
        roll_rate = -float(frame.error.derivative.get("torso_roll", 0.0))
        pitch_rate = -float(frame.error.derivative.get("torso_pitch", 0.0))
        # The sign convention is joint-target correction in the MuJoCo G1
        # actuator order.  Symmetric hip/ankle terms move the support polygon
        # under the measured torso and COM error.
        roll_correction = gain * (
            -cfg.roll_kp * roll - cfg.roll_kd * roll_rate - cfg.com_y_kp * com_y
        )
        pitch_correction = gain * (-cfg.pitch_kp * pitch - cfg.pitch_kd * pitch_rate)
        residual = {
            self._WAIST_ROLL: 0.45 * roll_correction,
            self._LEFT_HIP_ROLL: roll_correction,
            self._RIGHT_HIP_ROLL: roll_correction,
            self._LEFT_ANKLE_ROLL: -0.65 * roll_correction,
            self._RIGHT_ANKLE_ROLL: -0.65 * roll_correction,
            self._WAIST_PITCH: 0.35 * pitch_correction,
            self._LEFT_HIP_PITCH: pitch_correction,
            self._RIGHT_HIP_PITCH: pitch_correction,
            self._LEFT_ANKLE_PITCH: -0.60 * pitch_correction,
            self._RIGHT_ANKLE_PITCH: -0.60 * pitch_correction,
        }
        slip = max(0.0, float(frame.actual.get("support_slip_m", 0.0)))
        if slip:
            damping = gain * min(cfg.slip_damping, cfg.slip_damping * slip / 0.04)
            for output in (
                self._LEFT_HIP_ROLL,
                self._LEFT_ANKLE_ROLL,
                self._LEFT_HIP_PITCH,
                self._LEFT_ANKLE_PITCH,
            ):
                actual_signal = output.removeprefix("joint:")
                if output in base_action and actual_signal in frame.actual:
                    residual[output] += damping * (
                        frame.actual[actual_signal] - base_action[output]
                    )
        return residual

    def config_dict(self) -> dict[str, object]:
        return {
            "controller_type": "g1_balance_reflex",
            "version": 1,
            "config": asdict(self.config),
        }
