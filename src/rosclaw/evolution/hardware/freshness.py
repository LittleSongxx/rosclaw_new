"""Frame freshness gate (PR-EVO-HW-2, 真机自进化v2 §7.2/§九).

The freshness gate decides whether perception evidence may drive a round:
stale frames, missing frames, and RGB/Depth desync are first-class faults
that BLOCK new actions and mark the running round INVALID — old frames
must never be silently reused.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

CAMERA_NO_FRESH_FRAME = "camera_no_fresh_frame"
CAMERA_STALE_FRAME = "camera_stale_frame"
CAMERA_RGB_DEPTH_DESYNC = "camera_rgb_depth_desync"


@dataclass(frozen=True)
class FreshnessVerdict:
    ok: bool
    failure_type: str | None = None
    frame_age_ms: float | None = None
    rgb_depth_delta_ms: float | None = None
    consecutive_missing: int = 0
    reason: str = ""


@dataclass
class FrameFreshnessGate:
    max_frame_age_ms: float = 500.0
    max_rgb_depth_delta_ms: float = 50.0
    max_consecutive_missing: int = 3
    stale_window: int = 30
    _missing: int = field(default=0, repr=False)
    _history: deque = field(default_factory=lambda: deque(maxlen=30), repr=False)

    def check(
        self,
        *,
        frame_age_ms: float | None,
        rgb_depth_delta_ms: float | None = None,
    ) -> FreshnessVerdict:
        """Evaluate one candidate frame (or the absence of one)."""
        if frame_age_ms is None:
            self._missing += 1
            self._history.append(False)
            if self._missing >= self.max_consecutive_missing:
                return FreshnessVerdict(
                    ok=False,
                    failure_type=CAMERA_NO_FRESH_FRAME,
                    consecutive_missing=self._missing,
                    reason=f"{self._missing} consecutive reads without a fresh frame",
                )
            return FreshnessVerdict(
                ok=False,
                consecutive_missing=self._missing,
                reason="no frame this read (below blocking threshold)",
            )
        self._missing = 0
        if frame_age_ms > self.max_frame_age_ms:
            self._history.append(False)
            return FreshnessVerdict(
                ok=False,
                failure_type=CAMERA_STALE_FRAME,
                frame_age_ms=frame_age_ms,
                reason=f"frame age {frame_age_ms:.0f}ms > {self.max_frame_age_ms:.0f}ms",
            )
        if (
            rgb_depth_delta_ms is not None
            and rgb_depth_delta_ms > self.max_rgb_depth_delta_ms
        ):
            self._history.append(False)
            return FreshnessVerdict(
                ok=False,
                failure_type=CAMERA_RGB_DEPTH_DESYNC,
                frame_age_ms=frame_age_ms,
                rgb_depth_delta_ms=rgb_depth_delta_ms,
                reason=(
                    f"rgb/depth delta {rgb_depth_delta_ms:.0f}ms "
                    f"> {self.max_rgb_depth_delta_ms:.0f}ms"
                ),
            )
        self._history.append(True)
        return FreshnessVerdict(ok=True, frame_age_ms=frame_age_ms)

    @property
    def stale_rate(self) -> float:
        if not self._history:
            return 0.0
        return 1.0 - (sum(self._history) / len(self._history))

    @property
    def consecutive_missing(self) -> int:
        return self._missing


def frame_age_ms(captured_mono: float, now_mono: float | None = None) -> float:
    now = now_mono if now_mono is not None else time.monotonic()
    return (now - captured_mono) * 1000.0
