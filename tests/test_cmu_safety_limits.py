"""Tests for the CMU ARE safety envelope resolved from the embodiment card."""

from __future__ import annotations

import textwrap

import pytest

from rosclaw.apps.cmu_safety import (
    CMU_ARE_ROBOT_ID,
    FALLBACK_MAX_ABSOLUTE_COORDINATE_M,
    FALLBACK_MAX_CIRCLE_RADIUS_M,
    FALLBACK_MAX_RELATIVE_MOVE_M,
    FALLBACK_MAX_SEQUENCE_STEPS,
    FALLBACK_MAX_SPEED,
    FALLBACK_TOLERANCE_M,
    CmuGeofenceError,
    CmuSafetyLimits,
    check_within_workspace,
    resolve_cmu_safety_limits,
    resolve_workspace_ranges,
)

CARD_BODY = textwrap.dedent(
    """
    schema_version: rosclaw.embodiment_card.v1
    robot_id: cmu_are
    body_type: differential_drive_mobile_base
    ros_version: 1
    operational_limits:
      max_absolute_coordinate_m: 40.0
      max_relative_move_m: 5.0
      max_circle_radius_m: 2.0
      max_sequence_steps: 3
      max_speed: 1.25
      default_tolerance_m: 0.75
    workspace_boundaries:
      x: [-40.0, 40.0]
      y: [-40.0, 40.0]
      z: [-5.0, 5.0]
    """
).strip()


@pytest.fixture()
def specs_dir(tmp_path):
    (tmp_path / "cmu_are.yaml").write_text(CARD_BODY, encoding="utf-8")
    return tmp_path


# ----------------------------------------------------------------------
# Resolution order: explicit kwargs > embodiment card > module fallbacks
# ----------------------------------------------------------------------
def test_resolve_falls_back_when_no_card_exists(tmp_path):
    limits = resolve_cmu_safety_limits(specs_dir=tmp_path)

    assert limits.source == "fallback"
    assert limits.robot_id == CMU_ARE_ROBOT_ID
    assert limits.max_absolute_coordinate_m == FALLBACK_MAX_ABSOLUTE_COORDINATE_M
    assert limits.max_relative_move_m == FALLBACK_MAX_RELATIVE_MOVE_M
    assert limits.max_circle_radius_m == FALLBACK_MAX_CIRCLE_RADIUS_M
    assert limits.max_sequence_steps == FALLBACK_MAX_SEQUENCE_STEPS
    assert limits.max_speed == FALLBACK_MAX_SPEED
    assert limits.default_tolerance_m == FALLBACK_TOLERANCE_M
    assert limits.workspace_boundaries is None


def test_card_supplies_the_envelope(specs_dir):
    limits = resolve_cmu_safety_limits(specs_dir=specs_dir)

    assert limits.source == "embodiment_card"
    assert limits.max_absolute_coordinate_m == 40.0
    assert limits.max_relative_move_m == 5.0
    assert limits.max_circle_radius_m == 2.0
    assert limits.max_sequence_steps == 3
    assert limits.max_speed == 1.25
    assert limits.default_tolerance_m == 0.75
    assert limits.workspace_boundaries == {
        "x": [-40.0, 40.0],
        "y": [-40.0, 40.0],
        "z": [-5.0, 5.0],
    }


def test_explicit_kwargs_win_over_the_card(specs_dir):
    limits = resolve_cmu_safety_limits(
        specs_dir=specs_dir,
        max_relative_move_m=2.0,
        max_speed=0.5,
    )

    assert limits.max_relative_move_m == 2.0
    assert limits.max_speed == 0.5
    # Unspecified values still come from the card.
    assert limits.max_absolute_coordinate_m == 40.0
    assert limits.source == "embodiment_card+override"


def test_override_source_is_recorded_without_a_card(tmp_path):
    limits = resolve_cmu_safety_limits(specs_dir=tmp_path, max_speed=0.25)

    assert limits.source == "override"
    assert limits.max_speed == 0.25


def test_malformed_card_is_treated_as_absent(tmp_path):
    """A config typo must not fail navigation closed; fallbacks apply instead."""

    (tmp_path / "cmu_are.yaml").write_text("schema_version: nope\nrobot_id: cmu_are\n", encoding="utf-8")

    limits = resolve_cmu_safety_limits(specs_dir=tmp_path)

    assert limits.source == "fallback"
    assert limits.max_absolute_coordinate_m == FALLBACK_MAX_ABSOLUTE_COORDINATE_M


def test_non_positive_card_limits_fall_back(tmp_path):
    body = CARD_BODY.replace("max_relative_move_m: 5.0", "max_relative_move_m: -1.0").replace(
        "max_sequence_steps: 3", "max_sequence_steps: 0"
    )
    (tmp_path / "cmu_are.yaml").write_text(body, encoding="utf-8")

    limits = resolve_cmu_safety_limits(specs_dir=tmp_path)

    assert limits.max_relative_move_m == FALLBACK_MAX_RELATIVE_MOVE_M
    assert limits.max_sequence_steps == FALLBACK_MAX_SEQUENCE_STEPS
    # Valid siblings are still honoured.
    assert limits.max_absolute_coordinate_m == 40.0


def test_bundled_cmu_are_card_is_the_default_source():
    """With no specs_dir override, the shipped card drives the envelope."""

    limits = resolve_cmu_safety_limits()

    assert limits.source == "embodiment_card"
    assert limits.robot_id == "cmu_are"
    assert limits.workspace_boundaries is not None


# ----------------------------------------------------------------------
# CmuSafetyLimits behaviour
# ----------------------------------------------------------------------
def test_clamp_speed_bounds_both_ends():
    limits = CmuSafetyLimits(max_speed=2.0)

    assert limits.clamp_speed(5.0) == 2.0
    assert limits.clamp_speed(-3.0) == 0.0
    assert limits.clamp_speed(1.0) == 1.0


def test_with_overrides_ignores_none():
    limits = CmuSafetyLimits(max_speed=2.0, max_relative_move_m=20.0)

    updated = limits.with_overrides(max_speed=None, max_relative_move_m=4.0)

    assert updated.max_speed == 2.0
    assert updated.max_relative_move_m == 4.0
    # Frozen dataclass: the original is untouched.
    assert limits.max_relative_move_m == 20.0


def test_with_overrides_returns_self_when_nothing_applies():
    limits = CmuSafetyLimits()

    assert limits.with_overrides(max_speed=None) is limits


def test_to_dict_round_trips_the_envelope():
    limits = CmuSafetyLimits(workspace_boundaries={"x": [-1.0, 1.0]}, source="embodiment_card")

    as_dict = limits.to_dict()

    assert as_dict["source"] == "embodiment_card"
    assert as_dict["workspace_boundaries"] == {"x": [-1.0, 1.0]}
    assert as_dict["max_speed"] == FALLBACK_MAX_SPEED


# ----------------------------------------------------------------------
# Geofence normalisation
# ----------------------------------------------------------------------
def test_resolve_workspace_ranges_returns_none_without_a_geofence():
    assert resolve_workspace_ranges(None) is None
    assert resolve_workspace_ranges({}) is None


def test_resolve_workspace_ranges_reads_explicit_ranges():
    ranges = resolve_workspace_ranges({"x": [-2.0, 3.0], "y": [-4.0, 5.0], "z": [0.0, 1.0]})

    assert ranges == ([-2.0, 3.0], [-4.0, 5.0], [0.0, 1.0])


def test_missing_axis_is_unbounded_not_zero():
    x_range, y_range, z_range = resolve_workspace_ranges({"x": [-2.0, 3.0]})

    assert x_range == [-2.0, 3.0]
    assert y_range == [-float("inf"), float("inf")]
    assert z_range == [-float("inf"), float("inf")]


def test_resolve_workspace_ranges_reads_bounding_box():
    ranges = resolve_workspace_ranges(
        {"type": "bounding_box", "center": [1.0, 2.0, 0.0], "dimensions": [4.0, 6.0, 2.0]}
    )

    assert ranges == ([-1.0, 3.0], [-1.0, 5.0], [-1.0, 1.0])


def test_bounding_box_requires_three_elements():
    with pytest.raises(CmuGeofenceError, match="3-element"):
        resolve_workspace_ranges(
            {"type": "bounding_box", "center": [1.0, 2.0], "dimensions": [4.0, 6.0, 2.0]}
        )


def test_short_explicit_range_is_rejected():
    with pytest.raises(CmuGeofenceError, match="two elements"):
        resolve_workspace_ranges({"x": [0.0]})


# ----------------------------------------------------------------------
# Geofence enforcement
# ----------------------------------------------------------------------
def test_check_within_workspace_allows_in_bounds_goals():
    check_within_workspace(x=1.0, y=1.0, z=0.0, workspace_boundaries={"x": [-2.0, 2.0], "y": [-2.0, 2.0]})


def test_check_within_workspace_permits_everything_without_a_geofence():
    check_within_workspace(x=1e6, y=-1e6, z=0.0, workspace_boundaries=None)


def test_check_within_workspace_rejects_out_of_bounds_goal():
    with pytest.raises(CmuGeofenceError, match="violates workspace boundaries"):
        check_within_workspace(
            x=9.0, y=0.0, z=0.0, workspace_boundaries={"x": [-2.0, 2.0], "y": [-2.0, 2.0]}
        )


def test_geofence_error_names_every_violated_axis():
    with pytest.raises(CmuGeofenceError) as excinfo:
        check_within_workspace(
            x=9.0,
            y=-9.0,
            z=5.0,
            workspace_boundaries={"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [-1.0, 1.0]},
        )

    message = str(excinfo.value)
    assert "x=9.00" in message
    assert "y=-9.00" in message
    assert "z=5.00" in message


def test_boundary_values_are_inclusive():
    check_within_workspace(x=2.0, y=-2.0, z=0.0, workspace_boundaries={"x": [-2.0, 2.0], "y": [-2.0, 2.0]})


def test_bounding_box_geofence_is_enforced():
    fence = {"type": "bounding_box", "center": [0.0, 0.0, 0.0], "dimensions": [4.0, 4.0, 2.0]}

    check_within_workspace(x=1.9, y=-1.9, z=0.9, workspace_boundaries=fence)
    with pytest.raises(CmuGeofenceError):
        check_within_workspace(x=2.1, y=0.0, z=0.0, workspace_boundaries=fence)
