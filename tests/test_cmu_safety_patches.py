"""Tests for the safety patches added to cmu_are_bridge.py."""

import pytest

from rosclaw.apps.cmu_are_bridge import (
    CmuAreParseError,
    _validate_llm_intent,
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
