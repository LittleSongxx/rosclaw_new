"""Reusable phase and condition gate for transient feedback reflexes."""

from __future__ import annotations


class LatchedPhaseGate:
    """Latch a trigger only in its eligible phase, then fade it out safely."""

    def __init__(
        self,
        *,
        active_start: float,
        trigger_end: float,
        active_end: float,
        fade_fraction: float = 0.0,
    ) -> None:
        if not 0.0 <= active_start < trigger_end < active_end <= 1.0:
            raise ValueError("phase gate bounds must be ordered inside [0, 1]")
        if fade_fraction < 0.0:
            raise ValueError("fade_fraction must be non-negative")
        self.active_start = active_start
        self.trigger_end = trigger_end
        self.active_end = active_end
        self.fade_fraction = fade_fraction
        self.reset()

    def reset(self) -> None:
        self.triggered = False

    def update(self, *, phase: float, trigger: bool) -> float:
        if not 0.0 <= phase <= 1.0:
            raise ValueError("phase must be in [0, 1]")
        if not self.active_start <= phase <= self.active_end:
            return 0.0
        if not self.triggered and phase <= self.trigger_end and trigger:
            self.triggered = True
        if not self.triggered:
            return 0.0
        fade = self.fade_fraction
        if fade == 0.0:
            return 1.0
        if phase < self.active_start + fade:
            return (phase - self.active_start) / fade
        if phase > self.active_end - fade:
            return (self.active_end - phase) / fade
        return 1.0
