"""External visual critic + visual/telemetry consensus (PR-EVO-HW-2 §7.2).

Success is judged by TWO independent evidences:

    external visual recognition (D435i)  +  RH56 joint/force/state telemetry

When they conflict the round is INVALID with low critic confidence and a
review requirement — the robot's internal joint state may never be the
sole proof that its own gesture was correct (§7.2: 不能只用机器人内部
Joint State 证明自己手势正确).

The recognizer is a pluggable deterministic component (MediaPipe /
keypoints / lightweight classifier per §7.3); VLM may only review, never
control safety actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class GestureRecognizer(Protocol):
    """Deterministic primary recognizer (§7.3 主判定)."""

    def recognize(self, frame: Any) -> VisualObservation: ...


@dataclass(frozen=True)
class VisualObservation:
    gesture: str | None  # rock | paper | scissors | None (nothing seen)
    confidence: float
    source: str  # recognizer identity, e.g. "mediapipe_v1"
    frame_ref: str | None = None  # artifact://... of the judged frame
    stale: bool = False


@dataclass(frozen=True)
class TelemetryObservation:
    gesture: str | None
    verified: bool
    source: str = "rh56_joint_state"


@dataclass(frozen=True)
class ConsensusVerdict:
    outcome: str  # VALID | INVALID | UNCERTAIN
    agreed_gesture: str | None
    critic_confidence: float  # 0..1
    requires_review: bool
    reason: str
    visual: VisualObservation
    telemetry: TelemetryObservation


def judge_consensus(
    visual: VisualObservation,
    telemetry: TelemetryObservation,
    *,
    min_visual_confidence: float = 0.6,
) -> ConsensusVerdict:
    """Two-evidence judgment with honest conflict handling (§7.2)."""
    if visual.stale:
        return ConsensusVerdict(
            outcome="INVALID",
            agreed_gesture=None,
            critic_confidence=0.0,
            requires_review=True,
            reason="visual observation is stale — old frames never prove a round",
            visual=visual,
            telemetry=telemetry,
        )
    if visual.gesture is None:
        return ConsensusVerdict(
            outcome="UNCERTAIN",
            agreed_gesture=None,
            critic_confidence=0.2,
            requires_review=True,
            reason="no external visual observation — telemetry alone cannot self-certify",
            visual=visual,
            telemetry=telemetry,
        )
    if visual.confidence < min_visual_confidence:
        return ConsensusVerdict(
            outcome="UNCERTAIN",
            agreed_gesture=None,
            critic_confidence=round(visual.confidence * 0.5, 3),
            requires_review=True,
            reason=f"visual confidence {visual.confidence:.2f} < {min_visual_confidence}",
            visual=visual,
            telemetry=telemetry,
        )
    if not telemetry.verified:
        return ConsensusVerdict(
            outcome="INVALID",
            agreed_gesture=None,
            critic_confidence=0.3,
            requires_review=False,
            reason="telemetry verification failed even though vision saw a gesture",
            visual=visual,
            telemetry=telemetry,
        )
    if visual.gesture != telemetry.gesture:
        return ConsensusVerdict(
            outcome="INVALID",
            agreed_gesture=None,
            critic_confidence=0.1,
            requires_review=True,
            reason=(
                f"visual/telemetry conflict: vision={visual.gesture} "
                f"vs telemetry={telemetry.gesture}"
            ),
            visual=visual,
            telemetry=telemetry,
        )
    confidence = round(min(0.99, 0.5 + visual.confidence * 0.5), 3)
    return ConsensusVerdict(
        outcome="VALID",
        agreed_gesture=visual.gesture,
        critic_confidence=confidence,
        requires_review=False,
        reason="external vision and telemetry agree",
        visual=visual,
        telemetry=telemetry,
    )
