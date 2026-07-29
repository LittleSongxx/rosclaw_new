"""GoalForge contact-phase, aim, and recovery skill feedback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from rosclaw.feedback.contracts import FeedbackFrame, canonical_hash


@dataclass(frozen=True)
class G1KickSkillFeedbackConfig:
    expected_contact_phase: float = 0.48
    precontact_start: float = 0.28
    precontact_end: float = 0.52
    recovery_end: float = 0.78
    contact_phase_kp: float = 0.50
    ball_lateral_kp: float = 0.16
    recovery_roll_kp: float = 0.10
    recovery_pitch_kp: float = 0.08

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.precontact_start
            < self.expected_contact_phase
            < self.precontact_end
            < self.recovery_end
            <= 1.0
        ):
            raise ValueError("kick skill phase bounds must be ordered in [0, 1]")
        if any(
            value < 0.0
            for value in (
                self.contact_phase_kp,
                self.ball_lateral_kp,
                self.recovery_roll_kp,
                self.recovery_pitch_kp,
            )
        ):
            raise ValueError("kick skill feedback gains must be non-negative")


class G1KickSkillFeedbackController:
    """Emit bounded L2 directives without producing torque commands directly."""

    def __init__(self, config: G1KickSkillFeedbackConfig | None = None) -> None:
        self.config = config or G1KickSkillFeedbackConfig()
        self.reset()

    @property
    def controller_hash(self) -> str:
        return canonical_hash(self.config_dict())

    def reset(self) -> None:
        self._contact_latched = False

    def compute(
        self,
        frame: FeedbackFrame,
        base_action: Mapping[str, float],
    ) -> Mapping[str, float]:
        del base_action
        cfg = self.config
        contact = float(frame.actual.get("contact_detected", 0.0)) >= 0.5
        self._contact_latched = self._contact_latched or contact
        if frame.phase < cfg.precontact_start or frame.phase > cfg.recovery_end:
            return {}
        if not self._contact_latched and frame.phase <= cfg.precontact_end:
            contact_phase_error = float(frame.error.value.get("contact_phase", 0.0))
            lateral_error = float(frame.actual.get("ball_lateral_error_m", 0.0))
            lateral_correction = -cfg.ball_lateral_kp * lateral_error
            return {
                "skill:kick_phase_rate": cfg.contact_phase_kp * contact_phase_error,
                "joint:right_hip_yaw_joint": lateral_correction,
                "joint:right_ankle_roll_joint": -0.60 * lateral_correction,
            }
        roll = float(frame.actual.get("torso_roll", 0.0))
        pitch = float(frame.actual.get("torso_pitch", 0.0))
        return {
            "joint:waist_roll_joint": -cfg.recovery_roll_kp * roll,
            "joint:left_hip_roll_joint": -cfg.recovery_roll_kp * roll,
            "joint:right_hip_roll_joint": -cfg.recovery_roll_kp * roll,
            "joint:waist_pitch_joint": -cfg.recovery_pitch_kp * pitch,
            "joint:left_hip_pitch_joint": -cfg.recovery_pitch_kp * pitch,
            "joint:right_hip_pitch_joint": -cfg.recovery_pitch_kp * pitch,
        }

    def config_dict(self) -> dict[str, object]:
        return {
            "controller_type": "g1_goalforge_skill_feedback",
            "version": 1,
            "config": asdict(self.config),
        }


__all__ = ["G1KickSkillFeedbackConfig", "G1KickSkillFeedbackController"]
