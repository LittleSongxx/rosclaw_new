"""Tests for the safety patches added to cmu_are_bridge.py."""

import pytest

from rosclaw.apps.cmu_are_bridge import (
    CmuAreBridge,
    CmuAreParseError,
    _validate_llm_intent,
    _validate_llm_task,
    parse_cmu_instruction,
    DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE,
)


def test_absolute_intent_enforces_coordinate_bounds():
    """Absolute coordinates must be within ±DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE."""
    data = {"type": "absolute", "x": 1000.0, "y": 500.0}

    with pytest.raises(CmuAreParseError, match="absolute x coordinate.*exceeds safety limit"):
        _validate_llm_intent(
            data,
            instruction="go to 1000, 500",
            places={},
            current_pose=None,
            max_relative_m=20.0,
        )


def test_absolute_intent_enforces_y_coordinate_bounds():
    """Y coordinate must also be within bounds."""
    data = {"type": "absolute", "x": 50.0, "y": -150.0}

    with pytest.raises(CmuAreParseError, match="absolute y coordinate.*exceeds safety limit"):
        _validate_llm_intent(
            data,
            instruction="go to 50, -150",
            places={},
            current_pose=None,
            max_relative_m=20.0,
        )


def test_absolute_intent_enforces_z_coordinate_bounds():
    """Z coordinate must also be within bounds."""
    data = {"type": "absolute", "x": 10.0, "y": 10.0, "z": 200.0}

    with pytest.raises(CmuAreParseError, match="absolute z coordinate.*exceeds safety limit"):
        _validate_llm_intent(
            data,
            instruction="go to 10, 10, 200",
            places={},
            current_pose=None,
            max_relative_m=20.0,
        )


def test_absolute_intent_accepts_valid_coordinates():
    """Valid coordinates within bounds should be accepted."""
    data = {"type": "absolute", "x": 50.0, "y": -30.0, "z": 1.0}

    intent = _validate_llm_intent(
        data,
        instruction="go to 50, -30",
        places={},
        current_pose=None,
        max_relative_m=20.0,
    )

    assert intent.type == "absolute"
    assert intent.x == 50.0
    assert intent.y == -30.0
    assert intent.z == 1.0
    assert intent.frame_id == "map"


def test_absolute_intent_missing_x_raises_parse_error():
    """Missing required x field should raise CmuAreParseError, not TypeError."""
    data = {"type": "absolute", "y": 10.0}

    with pytest.raises(CmuAreParseError, match="absolute intent missing required field 'x'"):
        _validate_llm_intent(
            data,
            instruction="go to y=10",
            places={},
            current_pose=None,
            max_relative_m=20.0,
        )


def test_absolute_intent_missing_y_raises_parse_error():
    """Missing required y field should raise CmuAreParseError, not TypeError."""
    data = {"type": "absolute", "x": 10.0}

    with pytest.raises(CmuAreParseError, match="absolute intent missing required field 'y'"):
        _validate_llm_intent(
            data,
            instruction="go to x=10",
            places={},
            current_pose=None,
            max_relative_m=20.0,
        )


def test_absolute_intent_invalid_coordinate_type_raises_parse_error():
    """Invalid coordinate values should raise CmuAreParseError, not TypeError."""
    data = {"type": "absolute", "x": "invalid", "y": 10.0}

    with pytest.raises(CmuAreParseError, match="absolute intent has invalid coordinate values"):
        _validate_llm_intent(
            data,
            instruction="go to invalid, 10",
            places={},
            current_pose=None,
            max_relative_m=20.0,
        )


def test_absolute_intent_at_boundary_is_accepted():
    """Coordinates exactly at the boundary should be accepted."""
    max_coord = DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE
    data = {"type": "absolute", "x": max_coord, "y": -max_coord}

    intent = _validate_llm_intent(
        data,
        instruction=f"go to {max_coord}, {-max_coord}",
        places={},
        current_pose=None,
        max_relative_m=20.0,
    )

    assert intent.type == "absolute"
    assert intent.x == max_coord
    assert intent.y == -max_coord


def test_absolute_intent_beyond_boundary_is_rejected():
    """Coordinates just beyond the boundary should be rejected."""
    max_coord = DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE
    data = {"type": "absolute", "x": max_coord + 0.1, "y": 0.0}

    with pytest.raises(CmuAreParseError, match="absolute x coordinate.*exceeds safety limit"):
        _validate_llm_intent(
            data,
            instruction=f"go to {max_coord + 0.1}, 0",
            places={},
            current_pose=None,
            max_relative_m=20.0,
        )


# ----------------------------------------------------------------------
# The deterministic (regex) parse path must be bounded too.
#
# The original patch only bounded the LLM-parsed path, so `x=1000000` went
# straight through the `_COORD_RE` branch unchecked. Both paths now share
# `_absolute_intent`.
# ----------------------------------------------------------------------
def test_deterministic_path_enforces_coordinate_bounds():
    with pytest.raises(CmuAreParseError, match="absolute x coordinate.*exceeds safety limit"):
        parse_cmu_instruction("x=1000000, y=-500000")


def test_deterministic_path_enforces_y_coordinate_bounds():
    with pytest.raises(CmuAreParseError, match="absolute y coordinate.*exceeds safety limit"):
        parse_cmu_instruction("x=1.0, y=-500000")


def test_deterministic_path_accepts_in_bounds_coordinates():
    intent = parse_cmu_instruction("x=12.5, y=-3.0")

    assert intent.type == "absolute"
    assert intent.x == 12.5
    assert intent.y == -3.0
    assert intent.source == "deterministic"


def test_deterministic_path_honours_a_tighter_injected_limit():
    """The embodiment card can tighten the cap below the module default."""

    with pytest.raises(CmuAreParseError, match=r"exceeds safety limit ±10\.000m"):
        parse_cmu_instruction("x=50.0, y=0.0", max_absolute_coordinate_m=10.0)

    intent = parse_cmu_instruction("x=9.0, y=0.0", max_absolute_coordinate_m=10.0)
    assert intent.x == 9.0


def test_deterministic_path_boundary_is_inclusive():
    max_coord = DEFAULT_CMU_MAX_ABSOLUTE_COORDINATE

    intent = parse_cmu_instruction(f"x={max_coord}, y={-max_coord}")

    assert intent.x == max_coord
    assert intent.y == -max_coord


def test_llm_intent_honours_a_tighter_injected_limit():
    data = {"type": "absolute", "x": 50.0, "y": 0.0}

    with pytest.raises(CmuAreParseError, match=r"exceeds safety limit ±10\.000m"):
        _validate_llm_intent(
            data,
            instruction="go to 50, 0",
            places={},
            current_pose=None,
            max_relative_m=20.0,
            max_absolute_coordinate_m=10.0,
        )


# ----------------------------------------------------------------------
# The cap must reach every intent inside a multi-step task, not just the
# top-level one.
# ----------------------------------------------------------------------
def test_task_sequence_steps_are_bounded():
    data = {
        "status": "action",
        "type": "sequence",
        "steps": [
            {"type": "absolute", "x": 1.0, "y": 1.0},
            {"type": "absolute", "x": 999999.0, "y": 0.0},
        ],
    }

    with pytest.raises(CmuAreParseError, match="absolute x coordinate.*exceeds safety limit"):
        _validate_llm_task(
            data,
            instruction="go to 1,1 then 999999,0",
            places={},
            current_pose=None,
            max_relative_m=20.0,
            max_sequence_steps=8,
            circle_segments=8,
            max_circle_radius_m=6.0,
        )


def test_bridge_geofence_check_reports_as_a_parse_error():
    """The bridge translates CmuGeofenceError into its own error type.

    Callers of `navigate_to_intent` only handle `CmuAreParseError`, so a leaked
    `CmuGeofenceError` would escape as an unhandled exception.
    """

    bridge = CmuAreBridge.__new__(CmuAreBridge)  # no ROS connection needed
    fence = {"x": [-2.0, 2.0], "y": [-2.0, 2.0]}

    bridge._check_workspace_boundaries(x=1.0, y=1.0, z=0.0, workspace_boundaries=fence)

    with pytest.raises(CmuAreParseError, match="violates workspace boundaries"):
        bridge._check_workspace_boundaries(x=9.0, y=0.0, z=0.0, workspace_boundaries=fence)


def test_task_single_intent_is_bounded():
    data = {"status": "action", "type": "absolute", "x": 999999.0, "y": 0.0}

    with pytest.raises(CmuAreParseError, match="absolute x coordinate.*exceeds safety limit"):
        _validate_llm_task(
            data,
            instruction="go to 999999, 0",
            places={},
            current_pose=None,
            max_relative_m=20.0,
            max_sequence_steps=8,
            circle_segments=8,
            max_circle_radius_m=6.0,
        )
