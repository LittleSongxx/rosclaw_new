"""G1 kick/recovery Feedback Plane profile."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType

from rosclaw.feedback.contracts import FallbackMode, FeedbackLoopSpec
from rosclaw.feedback.controllers.balance import G1BalanceReflexConfig, G1BalanceReflexController
from rosclaw.feedback.runtime import FeedbackRuntime

G1_BALANCE_OUTPUT_LIMITS = MappingProxyType(
    {
        "joint:waist_roll_joint": 0.04,
        "joint:waist_pitch_joint": 0.04,
        "joint:left_hip_roll_joint": 0.08,
        "joint:right_hip_roll_joint": 0.08,
        "joint:left_ankle_roll_joint": 0.06,
        "joint:right_ankle_roll_joint": 0.06,
        "joint:left_hip_pitch_joint": 0.07,
        "joint:right_hip_pitch_joint": 0.07,
        "joint:left_ankle_pitch_joint": 0.05,
        "joint:right_ankle_pitch_joint": 0.05,
    }
)


def g1_joint_residual_limits(joint_names: tuple[str, ...]) -> tuple[float, ...]:
    """Return final combined residual limits in the supplied G1 joint order."""

    return tuple(
        G1_BALANCE_OUTPUT_LIMITS.get("joint:" + joint_name, 0.04) for joint_name in joint_names
    )


def build_g1_balance_runtime(
    *,
    body_hash: str,
    config: G1BalanceReflexConfig | None = None,
    rate_hz: float = 250.0,
    compute_clock_ns: Callable[[], int] | None = None,
) -> FeedbackRuntime:
    """Create the pinned simulation profile; no hardware transport is opened."""

    controller = G1BalanceReflexController(config)
    spec = FeedbackLoopSpec(
        loop_id="g1/kick-balance-reflex/v1",
        body_hash=body_hash,
        controller_hash=controller.controller_hash,
        reference_signals=("torso_roll", "torso_pitch", "com_y_relative"),
        observation_signals=(
            "torso_roll",
            "torso_pitch",
            "com_y_relative",
            "support_slip_m",
        ),
        output_limits=G1_BALANCE_OUTPUT_LIMITS,
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
