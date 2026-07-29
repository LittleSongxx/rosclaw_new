"""Fixed-rate scheduler separated from controller computation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerTick:
    sequence: int
    scheduled_ns: int
    actual_ns: int
    jitter_ns: int
    dropped_periods: int


class FixedRateScheduler:
    """Absolute-deadline scheduler that never accumulates relative sleep drift."""

    def __init__(
        self,
        *,
        rate_hz: float,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1.0 <= rate_hz <= 1000.0:
            raise ValueError("rate_hz must be in [1, 1000]")
        self.period_ns = int(round(1_000_000_000.0 / rate_hz))
        self._clock_ns = clock_ns
        self._sleep = sleep
        self.reset()

    def reset(self, *, start_ns: int | None = None) -> None:
        self._next_ns = start_ns
        self._sequence = 0

    def wait_next(self) -> SchedulerTick:
        now = self._clock_ns()
        if self._next_ns is None:
            self._next_ns = now
        scheduled = self._next_ns
        if now < scheduled:
            self._sleep((scheduled - now) / 1_000_000_000.0)
            now = self._clock_ns()
        lateness = max(0, now - scheduled)
        dropped = int(lateness // self.period_ns)
        self._next_ns = scheduled + (dropped + 1) * self.period_ns
        tick = SchedulerTick(
            sequence=self._sequence,
            scheduled_ns=scheduled,
            actual_ns=now,
            jitter_ns=now - scheduled,
            dropped_periods=dropped,
        )
        self._sequence += 1
        return tick
