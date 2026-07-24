"""Visual/telemetry consensus tests (PR-EVO-HW-2 §7.2)."""

from __future__ import annotations

from rosclaw.evolution.hardware.visual_critic import (
    TelemetryObservation,
    VisualObservation,
    judge_consensus,
)


def _visual(gesture="rock", confidence=0.9, stale=False):
    return VisualObservation(
        gesture=gesture, confidence=confidence, source="mediapipe_v1",
        frame_ref="artifact://f1", stale=stale,
    )


def _telemetry(gesture="rock", verified=True):
    return TelemetryObservation(gesture=gesture, verified=verified)


def test_agreement_is_valid() -> None:
    verdict = judge_consensus(_visual(), _telemetry())
    assert verdict.outcome == "VALID"
    assert verdict.agreed_gesture == "rock"
    assert verdict.critic_confidence > 0.9
    assert not verdict.requires_review


def test_conflict_is_invalid_with_review() -> None:
    verdict = judge_consensus(_visual(gesture="paper"), _telemetry(gesture="rock"))
    assert verdict.outcome == "INVALID"
    assert verdict.requires_review
    assert verdict.critic_confidence <= 0.1
    assert "conflict" in verdict.reason


def test_telemetry_alone_cannot_self_certify() -> None:
    verdict = judge_consensus(_visual(gesture=None), _telemetry())
    assert verdict.outcome == "UNCERTAIN"
    assert verdict.requires_review
    assert "telemetry alone cannot self-certify" in verdict.reason


def test_stale_visual_is_invalid() -> None:
    verdict = judge_consensus(_visual(stale=True), _telemetry())
    assert verdict.outcome == "INVALID"
    assert verdict.critic_confidence == 0.0
    assert "stale" in verdict.reason


def test_low_confidence_visual_uncertain() -> None:
    verdict = judge_consensus(_visual(confidence=0.3), _telemetry())
    assert verdict.outcome == "UNCERTAIN"
    assert verdict.requires_review


def test_telemetry_failure_invalid_even_when_vision_agrees() -> None:
    verdict = judge_consensus(_visual(), _telemetry(verified=False))
    assert verdict.outcome == "INVALID"
    assert not verdict.requires_review
