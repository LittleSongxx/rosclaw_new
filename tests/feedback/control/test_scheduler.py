from __future__ import annotations

from rosclaw.feedback.scheduler import FixedRateScheduler


def test_fixed_rate_scheduler_uses_absolute_deadlines_and_counts_drops() -> None:
    values = iter((1_000_000, 7_000_000, 20_000_000))
    scheduler = FixedRateScheduler(rate_hz=200.0, clock_ns=lambda: next(values))

    first = scheduler.wait_next()
    second = scheduler.wait_next()
    third = scheduler.wait_next()

    assert first.scheduled_ns == 1_000_000
    assert second.scheduled_ns == 6_000_000
    assert second.jitter_ns == 1_000_000
    assert third.dropped_periods == 1
    assert third.scheduled_ns == 11_000_000


def test_fixed_rate_scheduler_sleeps_only_until_absolute_deadline() -> None:
    now = [0]
    sleeps: list[float] = []

    def clock() -> int:
        return now[0]

    def sleep(duration: float) -> None:
        sleeps.append(duration)
        now[0] += int(duration * 1_000_000_000)

    scheduler = FixedRateScheduler(rate_hz=100.0, clock_ns=clock, sleep=sleep)
    scheduler.wait_next()
    scheduler.wait_next()

    assert sleeps == [0.01]
