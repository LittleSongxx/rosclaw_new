"""Tests for the embodiment card loader."""

from __future__ import annotations

import textwrap

import pytest
import yaml

from rosclaw.connectors.ros.embodiment import (
    EMBODIMENT_CARD_SCHEMA,
    EmbodimentCard,
    EmbodimentCardError,
    find_embodiment_card,
    list_embodiment_cards,
    load_embodiment_card,
    load_embodiment_card_file,
    parse_embodiment_card,
)

CARD_YAML = textwrap.dedent(
    """
    schema_version: rosclaw.embodiment_card.v1
    robot_id: test_bot
    aliases: [tb, testbot]
    body_type: differential_drive_mobile_base
    ros_version: 1
    ros_distro: noetic
    preferred_interfaces:
      - capability_id: test_bot.navigate_to_waypoint
        ros_kind: topic
        ros_name: /way_point
        ros_type: geometry_msgs/PointStamped
      - capability_id: test_bot.stop
        ros_kind: topic
        ros_name: /stop
        ros_type: std_msgs/Int8
    discouraged_interfaces:
      - ros_kind: topic
        ros_name: /cmd_vel
        reason: "The local planner owns cmd_vel."
    safety_defaults:
      max_linear_velocity: 2.0
    operational_limits:
      max_relative_move_m: 20.0
      max_sequence_steps: 8
    workspace_boundaries:
      x: [-100.0, 100.0]
      y: [-100.0, 100.0]
    """
).strip()


def _write_card(tmp_path, name="test_bot.yaml", body=CARD_YAML):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _parse(body=CARD_YAML):
    return parse_embodiment_card(yaml.safe_load(body))


def test_parse_embodiment_card_exposes_typed_fields():
    card = _parse()

    assert isinstance(card, EmbodimentCard)
    assert card.robot_id == "test_bot"
    assert card.schema_version == EMBODIMENT_CARD_SCHEMA
    assert card.body_type == "differential_drive_mobile_base"
    assert card.ros_version == 1
    assert card.ros_distro == "noetic"
    assert card.aliases == ("tb", "testbot")


def test_limit_reads_operational_limits_with_default():
    card = _parse()

    assert card.limit("max_relative_move_m", 1.0) == 20.0
    assert card.limit("max_sequence_steps", 1) == 8
    assert card.limit("not_a_real_limit", 7.5) == 7.5


def test_interface_for_matches_capability_id():
    card = _parse()

    interface = card.interface_for("test_bot.navigate_to_waypoint")
    assert interface is not None
    assert interface["ros_name"] == "/way_point"
    assert interface["ros_type"] == "geometry_msgs/PointStamped"

    assert card.interface_for("test_bot.does_not_exist") is None


def test_is_discouraged_flags_declared_topics():
    card = _parse()

    assert card.is_discouraged("/cmd_vel") is True
    assert card.is_discouraged("/way_point") is False


def test_workspace_boundaries_and_safety_defaults_round_trip():
    card = _parse()

    assert card.workspace_boundaries == {"x": [-100.0, 100.0], "y": [-100.0, 100.0]}
    assert card.safety_defaults["max_linear_velocity"] == 2.0

    as_dict = card.to_dict()
    assert as_dict["robot_id"] == "test_bot"
    assert as_dict["operational_limits"]["max_sequence_steps"] == 8


def test_parse_rejects_unknown_schema_version():
    body = CARD_YAML.replace("rosclaw.embodiment_card.v1", "rosclaw.embodiment_card.v99")

    with pytest.raises(EmbodimentCardError, match="schema_version"):
        _parse(body)


def test_parse_defaults_missing_schema_version_to_v1():
    """Hand-written cards may omit schema_version; that is treated as v1."""

    body = "\n".join(line for line in CARD_YAML.splitlines() if not line.startswith("schema_version:"))

    assert _parse(body).schema_version == EMBODIMENT_CARD_SCHEMA


def test_parse_rejects_missing_robot_id():
    body = "\n".join(line for line in CARD_YAML.splitlines() if not line.startswith("robot_id:"))

    with pytest.raises(EmbodimentCardError, match="robot_id"):
        _parse(body)


def test_parse_rejects_non_mapping_document():
    with pytest.raises(EmbodimentCardError, match="mapping"):
        parse_embodiment_card(["just", "a", "list"])


def test_load_embodiment_card_file_reads_from_disk(tmp_path):
    path = _write_card(tmp_path)

    card = load_embodiment_card_file(path)

    assert card.robot_id == "test_bot"
    assert card.source_path == path


def test_load_embodiment_card_file_raises_on_missing_path(tmp_path):
    with pytest.raises(EmbodimentCardError):
        load_embodiment_card_file(tmp_path / "nope.yaml")


def test_find_and_load_resolve_by_robot_id_and_alias(tmp_path):
    _write_card(tmp_path)

    assert find_embodiment_card("test_bot", specs_dir=tmp_path) is not None
    assert load_embodiment_card("test_bot", specs_dir=tmp_path).robot_id == "test_bot"
    # Aliases resolve by scanning the directory, not by filename.
    assert load_embodiment_card("testbot", specs_dir=tmp_path).robot_id == "test_bot"


def test_load_embodiment_card_returns_none_for_unknown_robot(tmp_path):
    _write_card(tmp_path)

    assert find_embodiment_card("unknown_bot", specs_dir=tmp_path) is None
    assert load_embodiment_card("unknown_bot", specs_dir=tmp_path) is None


def test_list_embodiment_cards_is_sorted(tmp_path):
    _write_card(tmp_path, name="b_bot.yaml", body=CARD_YAML.replace("test_bot", "b_bot"))
    _write_card(tmp_path, name="a_bot.yaml", body=CARD_YAML.replace("test_bot", "a_bot"))

    assert list_embodiment_cards(specs_dir=tmp_path) == ["a_bot", "b_bot"]


def test_bundled_cards_all_parse():
    """Every card shipped in connectors/ros/specs/ must be loadable."""

    robot_ids = list_embodiment_cards()
    assert "cmu_are" in robot_ids

    for robot_id in robot_ids:
        card = load_embodiment_card(robot_id)
        assert card is not None, robot_id
        assert card.robot_id == robot_id


def test_cmu_are_card_declares_the_safety_envelope():
    card = load_embodiment_card("cmu_are")

    assert card is not None
    assert card.ros_version == 1
    assert card.limit("max_absolute_coordinate_m", 0.0) == 100.0
    assert card.limit("max_relative_move_m", 0.0) == 20.0
    assert card.limit("max_circle_radius_m", 0.0) == 6.0
    assert card.limit("max_sequence_steps", 0) == 8
    assert card.workspace_boundaries is not None
    # The CMU local planner owns cmd_vel; publishing to it would fight the stack.
    assert card.is_discouraged("/cmd_vel") is True
    assert card.interface_for("cmu_are.navigate_to_waypoint") is not None
