"""Composite G1 balance and kick-timing residual controller."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from rosclaw.feedback.contracts import FeedbackFrame, canonical_hash
from rosclaw.feedback.controllers.balance import G1BalanceReflexConfig, G1BalanceReflexController
from rosclaw.feedback.controllers.kick_skill import (
    G1KickSkillFeedbackConfig,
    G1KickSkillFeedbackController,
)


@dataclass(frozen=True)
class G1CerebellumConfig:
    balance: G1BalanceReflexConfig = G1BalanceReflexConfig()
    kick_skill: G1KickSkillFeedbackConfig = G1KickSkillFeedbackConfig()
    phase_modulation_enabled: bool = False
    lateral_modulation_enabled: bool = False
    recovery_modulation_enabled: bool = False


class G1CerebellumController:
    """Sum L1 balance and L2 skill corrections before one safety projection."""

    def __init__(self, config: G1CerebellumConfig | None = None) -> None:
        self.config = config or G1CerebellumConfig()
        self.balance = G1BalanceReflexController(self.config.balance)
        self.kick_skill = G1KickSkillFeedbackController(self.config.kick_skill)

    @property
    def controller_hash(self) -> str:
        return canonical_hash(self.config_dict())

    def reset(self) -> None:
        self.balance.reset()
        self.kick_skill.reset()

    def compute(
        self,
        frame: FeedbackFrame,
        base_action: Mapping[str, float],
    ) -> Mapping[str, float]:
        combined: dict[str, float] = {}
        balance_output = self.balance.compute(frame, base_action)
        skill_output = dict(self.kick_skill.compute(frame, base_action))
        if not self.config.phase_modulation_enabled:
            skill_output.pop("skill:kick_phase_rate", None)
        if not self.config.lateral_modulation_enabled:
            skill_output.pop("joint:right_hip_yaw_joint", None)
            skill_output.pop("joint:right_ankle_roll_joint", None)
        if not self.config.recovery_modulation_enabled:
            for name in (
                "joint:waist_roll_joint",
                "joint:left_hip_roll_joint",
                "joint:right_hip_roll_joint",
                "joint:waist_pitch_joint",
                "joint:left_hip_pitch_joint",
                "joint:right_hip_pitch_joint",
            ):
                skill_output.pop(name, None)
        for output in (balance_output, skill_output):
            for name, value in output.items():
                combined[name] = combined.get(name, 0.0) + float(value)
        return combined

    def config_dict(self) -> dict[str, object]:
        return {
            "controller_type": "g1_goalforge_cerebellum",
            "version": 1,
            "config": asdict(self.config),
            "composition": "balance_plus_kick_skill_before_safety_projection",
        }


__all__ = ["G1CerebellumConfig", "G1CerebellumController"]
