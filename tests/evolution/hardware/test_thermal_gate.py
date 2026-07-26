"""Thermal window gate tests (§Phase 6 matched start temperature)."""

from __future__ import annotations

from rosclaw.evolution.hardware.thermal import wait_for_thermal_window


def test_in_window_passes_immediately() -> None:
    result = wait_for_thermal_window(
        probe=lambda: {"right": 42.0, "left": 43.0},
        start_max_temp_c=46.0,
        max_wait_s=10.0,
        poll_s=0.01,
    )
    assert result.ok
    assert result.waited_s < 1.0


def test_hot_hands_wait_then_pass() -> None:
    readings = iter(
        [
            {"right": 50.0, "left": 49.0},
            {"right": 47.0, "left": 46.5},
            {"right": 44.0, "left": 43.0},
        ]
    )
    result = wait_for_thermal_window(
        probe=lambda: next(readings),
        start_max_temp_c=46.0,
        max_wait_s=10.0,
        poll_s=0.01,
    )
    assert result.ok
    assert result.temps["right"] == 44.0


def test_timeout_blocks_honestly() -> None:
    result = wait_for_thermal_window(
        probe=lambda: {"right": 52.0, "left": 51.0},
        start_max_temp_c=46.0,
        max_wait_s=0.05,
        poll_s=0.01,
    )
    assert not result.ok
    assert "not reached" in result.reason
    assert result.temps["right"] == 52.0


def test_missing_reading_never_counts_as_cold() -> None:
    result = wait_for_thermal_window(
        probe=lambda: {"right": None, "left": None, "error": "probe failed"},
        start_max_temp_c=46.0,
        max_wait_s=0.05,
        poll_s=0.01,
    )
    assert not result.ok
