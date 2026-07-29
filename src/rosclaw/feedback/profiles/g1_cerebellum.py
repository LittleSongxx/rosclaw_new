"""Pinned Phase 7 G1 residual-cerebellum profile."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

from rosclaw.feedback.contracts import FallbackMode, FeedbackLoopSpec
from rosclaw.feedback.controllers.cerebellum import G1CerebellumConfig, G1CerebellumController
from rosclaw.feedback.profiles.g1 import G1_BALANCE_OUTPUT_LIMITS
from rosclaw.feedback.profiles.g1_skill import G1_SKILL_OUTPUT_LIMITS
from rosclaw.feedback.runtime import FeedbackRuntime

G1_CEREBELLUM_OUTPUT_LIMITS = MappingProxyType(
    {
        name: max(
            G1_BALANCE_OUTPUT_LIMITS.get(name, 0.0),
            G1_SKILL_OUTPUT_LIMITS.get(name, 0.0),
        )
        for name in set(G1_BALANCE_OUTPUT_LIMITS) | set(G1_SKILL_OUTPUT_LIMITS)
    }
)


def build_g1_cerebellum_runtime(
    *,
    body_hash: str,
    config: G1CerebellumConfig | None = None,
    rate_hz: float = 250.0,
    compute_clock_ns: Callable[[], int] | None = None,
) -> FeedbackRuntime:
    resolved = config or G1CerebellumConfig()
    controller = G1CerebellumController(resolved)
    output_limits = dict(G1_BALANCE_OUTPUT_LIMITS)
    reference_signals = ["torso_roll", "torso_pitch", "com_y_relative"]
    observation_signals = [
        "torso_roll",
        "torso_pitch",
        "com_y_relative",
        "support_slip_m",
    ]
    skill_enabled = bool(
        resolved.phase_modulation_enabled
        or resolved.lateral_modulation_enabled
        or resolved.recovery_modulation_enabled
    )
    if skill_enabled:
        observation_signals.extend(("contact_phase", "contact_detected"))
    if resolved.phase_modulation_enabled:
        reference_signals.append("contact_phase")
        output_limits["skill:kick_phase_rate"] = G1_SKILL_OUTPUT_LIMITS["skill:kick_phase_rate"]
    if resolved.lateral_modulation_enabled:
        reference_signals.append("ball_lateral_error_m")
        observation_signals.append("ball_lateral_error_m")
        for name in ("joint:right_hip_yaw_joint", "joint:right_ankle_roll_joint"):
            output_limits[name] = max(
                output_limits.get(name, 0.0),
                G1_SKILL_OUTPUT_LIMITS[name],
            )
    spec = FeedbackLoopSpec(
        loop_id="g1/goalforge-cerebellum/v1",
        body_hash=body_hash,
        controller_hash=controller.controller_hash,
        reference_signals=tuple(reference_signals),
        observation_signals=tuple(observation_signals),
        output_limits=output_limits,
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


__all__ = ["G1_CEREBELLUM_OUTPUT_LIMITS", "build_g1_cerebellum_runtime"]
