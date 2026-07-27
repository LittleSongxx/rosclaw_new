import json
from pathlib import Path

from rosclaw.apps.behavior_tree import parse_patrol_instruction, run_patrol_behavior_tree
from rosclaw.apps.mobility import parse_move_instruction, run_language_move
from rosclaw.apps.nav2_bridge import (
    PlacePose,
    check_nav2_available,
    load_places_yaml,
    resolve_place_query,
    write_nav2_artifact,
)


def test_parse_move_instruction_chinese_forward_backward_and_absolute():
    forward = parse_move_instruction("前进 1 米")
    backward = parse_move_instruction("后退 0.5 米")
    absolute = parse_move_instruction("向前移动到 x=2.0")

    assert forward.target_x == 1.0
    assert forward.direction == "forward"
    assert backward.target_x == -0.5
    assert backward.direction == "backward"
    assert absolute.target_x == 2.0
    assert absolute.mode == "absolute"


def test_run_language_move_writes_artifacts(tmp_path: Path):
    result = run_language_move("前进 1 米", output_dir=tmp_path)

    assert result.ok
    assert result.final_error <= result.tolerance_m
    assert "agent.command" in result.event_topics
    assert "skill.execution.start" in result.event_topics
    assert "skill.execution.complete" in result.event_topics

    artifact_dir = Path(result.artifact_dir)
    assert (artifact_dir / "summary.json").exists()
    assert (artifact_dir / "trajectory.json").exists()
    assert (artifact_dir / "trajectory.csv").exists()
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "success"


def test_parse_patrol_instruction_exports_btcpp_xml():
    plan = parse_patrol_instruction("去 A 点巡检，再去 B 点，最后返回起点")
    xml = plan.to_btcpp_xml()

    assert [w.name for w in plan.waypoints] == ["A", "B"]
    assert plan.return_home is True
    assert 'BTCPP_format="4"' in xml
    assert "<Navigate" in xml
    assert "<Inspect" in xml


def test_run_patrol_behavior_tree_writes_bt_artifacts(tmp_path: Path):
    result = run_patrol_behavior_tree("去 A 点巡检，再去 B 点，最后返回起点", output_dir=tmp_path)

    assert result.ok
    assert result.final_pose["x"] < 0.1
    assert result.final_pose["y"] < 0.1
    assert any(entry["status"] == "RUNNING" for entry in result.timeline)
    assert any(entry["status"] == "SUCCESS" for entry in result.timeline)

    artifact_dir = Path(result.artifact_dir)
    assert (artifact_dir / "bt.xml").exists()
    assert (artifact_dir / "bt.json").exists()
    assert (artifact_dir / "timeline.jsonl").exists()
    assert (artifact_dir / "trajectory.json").exists()


def test_nav2_bridge_import_safe_and_places_loader(tmp_path: Path):
    ok, reason = check_nav2_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)

    places_file = tmp_path / "places.yaml"
    places_file.write_text(
        """
places:
  kitchen:
    aliases: [厨房, kitchen]
    x: 1.0
    y: 2.0
    theta: 0.5
""",
        encoding="utf-8",
    )
    places = load_places_yaml(places_file)
    assert places["kitchen"].x == 1.0
    assert places["kitchen"].frame_id == "map"
    assert resolve_place_query("去厨房", places).name == "kitchen"


def test_nav2_artifact_writer_records_goal_feedback_and_topics(tmp_path: Path):
    place = PlacePose("inspection_a", 1.0, 0.5, aliases=("巡检点a",))
    result = {
        "status": "success",
        "feedback": [{"distance_remaining": 0.2}],
        "pose_trace": [{"x": 0.9, "y": 0.4, "theta": 0.0}],
    }

    artifact_dir = write_nav2_artifact(
        place=place,
        result=result,
        output_dir=tmp_path,
        instruction="去 inspection_a",
        ros2_topics=["/navigate_to_pose", "/amcl_pose"],
    )

    assert (artifact_dir / "summary.json").exists()
    assert (artifact_dir / "goal.json").exists()
    assert (artifact_dir / "feedback.jsonl").exists()
    assert (artifact_dir / "pose_trace.jsonl").exists()
    assert (artifact_dir / "ros2_topics.txt").exists()
