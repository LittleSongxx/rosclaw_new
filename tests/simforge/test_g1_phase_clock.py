from __future__ import annotations

import pytest

from rosclaw.simforge.backends.unitree_mujoco_backend import _apply_feedback_phase_rate


def test_positive_phase_rate_advances_policy_clock_only_after_accumulation() -> None:
    repeat = 1
    accumulator = 0.0
    repeats = []
    for _ in range(13):
        value, accumulator = _apply_feedback_phase_rate(
            repeat=repeat,
            phase_rate=0.08,
            accumulator=accumulator,
        )
        repeats.append(value)

    assert repeats.count(2) == 1
    assert all(value in {1, 2} for value in repeats)


def test_negative_phase_rate_holds_policy_clock_without_reversing_it() -> None:
    accumulator = 0.0
    repeats = []
    for _ in range(13):
        value, accumulator = _apply_feedback_phase_rate(
            repeat=1,
            phase_rate=-0.08,
            accumulator=accumulator,
        )
        repeats.append(value)

    assert repeats.count(0) == 1
    assert all(value in {0, 1} for value in repeats)


def test_phase_rate_rejects_unbounded_directive() -> None:
    with pytest.raises(ValueError, match=r"in \[-1, 1\]"):
        _apply_feedback_phase_rate(repeat=1, phase_rate=1.1, accumulator=0.0)
