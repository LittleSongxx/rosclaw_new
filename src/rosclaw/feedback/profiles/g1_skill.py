"""Pinned L2 GoalForge skill-feedback profile."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

from rosclaw.feedback.contracts import FallbackMode, FeedbackLoopSpec
from rosclaw.feedback.controllers.kick_skill import (
    G1KickSkillFeedbackConfig,
    G1KickSkillFeedbackController,
)
from rosclaw.feedback.runtime import FeedbackRuntime

G1_SKILL_OUTPUT_LIMITS = MappingProxyType(
    {
        "skill:kick_phase_rate": 0.08,
        "joint:right_hip_yaw_joint": 0.035,
        "joint:right_ankle_roll_joint": 0.025,
        "joint:waist_roll_joint": 0.035,
        "joint:left_hip_roll_joint": 0.05,
        "joint:right_hip_roll_joint": 0.05,
        "joint:waist_pitch_joint": 0.035,
        "joint:left_hip_pitch_joint": 0.05,
        "joint:right_hip_pitch_joint": 0.05,
    }
)


def build_g1_kick_skill_runtime(
    *,
    body_hash: str,
    config: G1KickSkillFeedbackConfig | None = None,
    rate_hz: float = 50.0,
    compute_clock_ns: Callable[[], int] | None = None,
) -> FeedbackRuntime:
    controller = G1KickSkillFeedbackController(config)
    spec = FeedbackLoopSpec(
        loop_id="g1/goalforge-skill-feedback/v1",
        body_hash=body_hash,
        controller_hash=controller.controller_hash,
        reference_signals=(
            "contact_phase",
            "ball_lateral_error_m",
            "torso_roll",
            "torso_pitch",
        ),
        observation_signals=(
            "contact_phase",
            "ball_lateral_error_m",
            "contact_detected",
            "torso_roll",
            "torso_pitch",
        ),
        output_limits=G1_SKILL_OUTPUT_LIMITS,
        rate_hz=rate_hz,
        deadline_ms=1000.0 / rate_hz,
        max_observation_age_ms=2.0 * 1000.0 / rate_hz,
        fallback_deadline_miss=FallbackMode.FREEZE_AND_STABILIZE,
    )
    if compute_clock_ns is None:
        return FeedbackRuntime(spec=spec, controller=controller)
    return FeedbackRuntime(
        spec=spec,
        controller=controller,
        compute_clock_ns=compute_clock_ns,
    )


__all__ = ["G1_SKILL_OUTPUT_LIMITS", "build_g1_kick_skill_runtime"]
