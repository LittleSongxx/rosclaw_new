"""Frame freshness gate tests (PR-EVO-HW-2, 真机自进化v2 §7.2)."""

from __future__ import annotations

from rosclaw.evolution.hardware.freshness import (
    CAMERA_NO_FRESH_FRAME,
    CAMERA_RGB_DEPTH_DESYNC,
    CAMERA_STALE_FRAME,
    FrameFreshnessGate,
)


def test_fresh_frame_passes() -> None:
    gate = FrameFreshnessGate()
    verdict = gate.check(frame_age_ms=120.0, rgb_depth_delta_ms=10.0)
    assert verdict.ok
    assert gate.stale_rate == 0.0


def test_stale_frame_flagged() -> None:
    gate = FrameFreshnessGate(max_frame_age_ms=500.0)
    verdict = gate.check(frame_age_ms=800.0)
    assert not verdict.ok
    assert verdict.failure_type == CAMERA_STALE_FRAME
    assert gate.stale_rate == 1.0


def test_consecutive_missing_blocks_at_threshold() -> None:
    gate = FrameFreshnessGate(max_consecutive_missing=3)
    assert not gate.check(frame_age_ms=None).ok  # 1: below threshold
    assert gate.check(frame_age_ms=None).failure_type is None  # 2
    verdict = gate.check(frame_age_ms=None)  # 3: blocks
    assert verdict.failure_type == CAMERA_NO_FRESH_FRAME
    assert verdict.consecutive_missing == 3


def test_missing_counter_resets_on_fresh_frame() -> None:
    gate = FrameFreshnessGate(max_consecutive_missing=3)
    gate.check(frame_age_ms=None)
    gate.check(frame_age_ms=None)
    assert gate.check(frame_age_ms=100.0).ok
    assert gate.consecutive_missing == 0
    assert gate.check(frame_age_ms=None).failure_type is None


def test_rgb_depth_desync_flagged() -> None:
    gate = FrameFreshnessGate(max_rgb_depth_delta_ms=50.0)
    verdict = gate.check(frame_age_ms=100.0, rgb_depth_delta_ms=120.0)
    assert not verdict.ok
    assert verdict.failure_type == CAMERA_RGB_DEPTH_DESYNC


def test_old_frames_never_reused() -> None:
    """A stale verdict followed by no-new-data must stay blocked — the gate
    never upgrades old evidence into a pass."""
    gate = FrameFreshnessGate(max_consecutive_missing=2)
    gate.check(frame_age_ms=900.0)
    verdict = gate.check(frame_age_ms=None)
    assert not verdict.ok
    assert gate.stale_rate == 1.0  # all history entries are failures
