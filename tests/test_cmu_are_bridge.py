import json
import time
from pathlib import Path

import pytest

from rosclaw.apps.cmu_are_bridge import (
    CmuAreBridge,
    CmuChatTaskManager,
    CmuIntent,
    CmuRunResult,
    build_cmu_result_message,
    check_cmu_are_available,
    describe_unsupported_cmu_command,
    load_cmu_places,
    parse_cmu_chat_task,
    parse_cmu_instruction,
    _parse_with_llm_chat,
    _validate_llm_intent,
)


def test_cmu_places_and_parser_place_relative_and_explore(tmp_path: Path):
    places_file = tmp_path / "places.yaml"
    places_file.write_text(
        """
places:
  inspection_a:
    aliases: [巡检点A, A点]
    x: 1.0
    y: 2.0
""",
        encoding="utf-8",
    )
    places = load_cmu_places(places_file)

    place = parse_cmu_instruction("去巡检点A", places=places)
    relative = parse_cmu_instruction("向上走 3 米", current_pose={"x": 1.0, "y": 2.0})
    explore = parse_cmu_instruction("暂停待命")

    assert place.type == "place"
    assert place.x == 1.0
    assert place.y == 2.0
    assert relative.type == "relative"
    assert relative.x == 4.0
    assert relative.y == 2.0
    assert relative.dx == 3.0
    assert explore.type == "explore_control"
    assert explore.command == "pause"


def test_cmu_relative_screen_direction_mapping():
    pose = {"x": 1.0, "y": 2.0}

    up = parse_cmu_instruction("向上走 3 米", current_pose=pose)
    down = parse_cmu_instruction("向下走 3 米", current_pose=pose)
    left = parse_cmu_instruction("向左走 3 米", current_pose=pose)
    right = parse_cmu_instruction("向右走 3 米", current_pose=pose)

    assert (up.x, up.y, up.dx, up.dy) == (4.0, 2.0, 3.0, 0.0)
    assert (down.x, down.y, down.dx, down.dy) == (-2.0, 2.0, -3.0, 0.0)
    assert (left.x, left.y, left.dx, left.dy) == (1.0, 5.0, 0.0, 3.0)
    assert (right.x, right.y, right.dx, right.dy) == (1.0, -1.0, 0.0, -3.0)


def test_cmu_parser_allows_twenty_meter_relative_move():
    intent = parse_cmu_instruction("向右走 20 米", max_relative_m=20.0)

    assert intent.type == "relative"
    assert intent.dy == -20.0


def test_cmu_parser_rejects_over_configured_relative_move():
    with pytest.raises(ValueError):
        parse_cmu_instruction("向右走 30 米", max_relative_m=20.0)


def test_cmu_parser_supports_diagonal_and_robot_frame_relative_moves():
    diagonal = parse_cmu_instruction("向右上走 10 米", max_relative_m=20.0)
    forward = parse_cmu_instruction(
        "前进 4 米",
        current_pose={"x": 0.0, "y": 0.0, "theta": 1.57079632679},
        max_relative_m=20.0,
    )

    assert diagonal.dx == pytest.approx(7.071, abs=0.01)
    assert diagonal.dy == pytest.approx(-7.071, abs=0.01)
    assert forward.dx == pytest.approx(0.0, abs=0.01)
    assert forward.dy == pytest.approx(4.0, abs=0.01)


def test_llm_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError):
        _parse_with_llm_chat(
            "向上走3米",
            places={},
            current_pose={"x": 0.0, "y": 0.0},
            max_relative_m=20.0,
        )


def test_llm_direction_schema_uses_screen_mapping():
    intent = _validate_llm_intent(
        {"status": "action", "type": "relative", "direction": "up", "distance": 3.0},
        instruction="向上走3米",
        places={},
        current_pose={"x": 1.0, "y": 2.0},
        max_relative_m=20.0,
    )

    assert intent.x == 4.0
    assert intent.y == 2.0
    assert intent.dx == 3.0
    assert intent.dy == 0.0
    assert intent.source == "llm"


def test_llm_rejects_unknown_place():
    with pytest.raises(ValueError):
        _validate_llm_intent(
            {"status": "action", "type": "place", "place": "unknown"},
            instruction="去那里",
            places={},
            current_pose=None,
            max_relative_m=20.0,
        )


def test_exploration_control_restores_speed_on_resume():
    class FakeMsg:
        def __init__(self):
            self.data = None

    class FakePublisher:
        def __init__(self):
            self.values = []

        def publish(self, msg):
            self.values.append(msg.data)

        def get_num_connections(self):
            return 1

    class FakeRospy:
        @staticmethod
        def is_shutdown():
            return False

    bridge = object.__new__(CmuAreBridge)
    bridge.String = FakeMsg
    bridge.Float32 = FakeMsg
    bridge.Int8 = FakeMsg
    bridge.rospy = FakeRospy()
    bridge.explore_pub = FakePublisher()
    bridge.speed_pub = FakePublisher()
    bridge.stop_pub = FakePublisher()
    bridge._wait_for_publisher_connections = lambda publishers, timeout_sec: None

    bridge.publish_exploration_control("pause", speed=2.0)
    bridge.publish_exploration_control("resume", speed=2.0)

    assert 0.0 in bridge.speed_pub.values
    assert 2.0 in bridge.speed_pub.values
    assert 1 in bridge.stop_pub.values
    assert 0 in bridge.stop_pub.values


def test_cmu_bridge_import_safe_without_ros1():
    ok, reason = check_cmu_are_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


def test_cmu_result_to_dict():
    result = CmuRunResult(
        episode_id="episode",
        instruction="去 inspection_a",
        intent=CmuIntent(type="place", instruction="去 inspection_a", x=1.0, y=2.0),
        status="success",
        artifact_dir="/tmp/episode",
        duration_sec=1.0,
        final_pose={"x": 1.0, "y": 2.0, "theta": 0.0},
        distance_to_goal=0.1,
        odom_trace_count=3,
        cmd_vel_count=2,
        path_count=1,
        waypoint_count=1,
    )

    data = result.to_dict()
    assert data["status"] == "success"
    assert data["intent"]["type"] == "place"
    assert json.dumps(data, ensure_ascii=False)


def test_cmu_chat_result_message_for_explore_commands():
    for command, expected in [
        ("start", "已成功开始探索"),
        ("pause", "已成功暂停探索"),
        ("resume", "已成功继续探索"),
        ("stop", "已成功停止探索"),
    ]:
        result = CmuRunResult(
            episode_id="episode",
            instruction=command,
            intent=CmuIntent(type="explore_control", instruction=command, command=command),
            status="success",
            artifact_dir="/tmp/episode",
            duration_sec=0.0,
            final_pose={"x": 0.0, "y": 0.0, "theta": 0.0},
            distance_to_goal=None,
            odom_trace_count=1,
            cmd_vel_count=0,
            path_count=0,
            waypoint_count=0,
        )

        assert expected in build_cmu_result_message(result, use_llm=False)


def test_cmu_chat_result_message_for_relative_move():
    result = CmuRunResult(
        episode_id="episode",
        instruction="向右走5米",
        intent=CmuIntent(
            type="relative",
            instruction="向右走5米",
            x=1.0,
            y=-5.0,
            dx=0.0,
            dy=-5.0,
        ),
        status="success",
        artifact_dir="/tmp/episode",
        duration_sec=2.0,
        final_pose={"x": 1.0, "y": -4.2, "theta": 0.0},
        distance_to_goal=0.8,
        odom_trace_count=3,
        cmd_vel_count=2,
        path_count=1,
        waypoint_count=1,
    )

    message = build_cmu_result_message(result, use_llm=False)

    assert "已成功向右移动 5 米" in message
    assert "最终误差 0.8 米" in message


def test_circle_task_generates_waypoint_sequence(monkeypatch):
    monkeypatch.setattr(
        "rosclaw.apps.cmu_are_bridge._parse_with_llm_chat",
        lambda *_args, **_kwargs: {
            "status": "action",
            "type": "circle",
            "radius": 2.0,
            "say": "开始执行圆形轨迹。",
        },
    )

    task = parse_cmu_chat_task(
        "以半径为2米原地转圈",
        places={},
        current_pose={"x": 1.0, "y": 2.0, "theta": 0.0},
        circle_segments=8,
    )

    assert task.kind == "sequence"
    assert len(task.intents) == 8
    assert task.metadata["mode"] == "circle"


def test_sequence_task_parses_multiple_relative_steps(monkeypatch):
    monkeypatch.setattr(
        "rosclaw.apps.cmu_are_bridge._parse_with_llm_chat",
        lambda *_args, **_kwargs: {
            "status": "action",
            "type": "sequence",
            "steps": [
                {"type": "relative", "direction": "up", "distance": 5.0},
                {"type": "relative", "direction": "right", "distance": 3.0},
            ],
            "say": "开始执行多步移动任务。",
        },
    )

    task = parse_cmu_chat_task(
        "先向上走5米，再向右走3米",
        places={},
        current_pose={"x": 0.0, "y": 0.0, "theta": 0.0},
        max_relative_m=20.0,
    )

    assert task.kind == "sequence"
    assert len(task.intents) == 2
    assert task.intents[0].x == 5.0
    assert task.intents[0].y == 0.0
    assert task.intents[1].x == 5.0
    assert task.intents[1].y == -3.0


def test_unsupported_raw_velocity_command_gets_capability_boundary():
    message = describe_unsupported_cmu_command("直接发布 cmd_vel 速度曲线")

    assert message is not None
    assert "不会直接接管 /cmd_vel" in message


def test_cmu_result_message_falls_back_when_llm_polish_fails(monkeypatch):
    def fail_polish(*_args, **_kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr("rosclaw.apps.cmu_are_bridge._polish_cmu_result_message", fail_polish)
    result = CmuRunResult(
        episode_id="episode",
        instruction="开始探索",
        intent=CmuIntent(type="explore_control", instruction="开始探索", command="start"),
        status="success",
        artifact_dir="/tmp/episode",
        duration_sec=0.0,
        final_pose={"x": 0.0, "y": 0.0, "theta": 0.0},
        distance_to_goal=None,
        odom_trace_count=1,
        cmd_vel_count=0,
        path_count=0,
        waypoint_count=0,
    )

    assert "已成功开始探索" in build_cmu_result_message(result, use_llm=True)


def test_task_manager_pauses_exploration_before_manual_move(monkeypatch, tmp_path: Path):
    class FakeBridge:
        current_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
        odom_trace = []
        cmd_trace = []
        path_trace = []
        waypoint_trace = []

        def __init__(self):
            self.controls = []

        def wait_for_pose(self, timeout_sec=0.0):
            return self.current_pose

        def publish_exploration_control(self, command, *, speed=2.0):
            self.controls.append(command)

        def publish_stop(self, stopped):
            pass

        def navigate_to_intent(self, intent, **kwargs):
            return {
                "status": "success",
                "duration_sec": 0.01,
                "final_pose": {"x": intent.x, "y": intent.y, "theta": 0.0},
                "distance_to_goal": 0.0,
            }

    bridge = FakeBridge()
    manager = CmuChatTaskManager(
        bridge=bridge,
        places={},
        ros_topics=[],
        output_dir=tmp_path,
        progress_interval_sec=0.01,
    )
    manager._exploration_active = True
    monkeypatch.setattr(
        "rosclaw.apps.cmu_are_bridge._parse_with_llm_chat",
        lambda *_args, **_kwargs: {
            "status": "action",
            "type": "relative",
            "direction": "up",
            "distance": 2.0,
            "say": "开始向上移动。",
        },
    )

    events = manager.submit("向上走2米")
    deadline = time.time() + 2.0
    drained = []
    while time.time() < deadline:
        drained.extend(manager.drain_events())
        if any(event.phase == "end" for event in drained):
            break
        time.sleep(0.02)

    assert "pause" in bridge.controls
    assert any("开始" in event.message for event in events)
    assert any(event.status == "success" for event in drained)


def test_task_manager_can_cancel_current_task(monkeypatch, tmp_path: Path):
    class FakeCancel:
        def __init__(self):
            self.stopped = []
            self.current_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
            self.odom_trace = []
            self.cmd_trace = []
            self.path_trace = []
            self.waypoint_trace = []

        def wait_for_pose(self, timeout_sec=0.0):
            return self.current_pose

        def publish_stop(self, stopped):
            self.stopped.append(stopped)

        def publish_exploration_control(self, command, *, speed=2.0):
            pass

        def navigate_to_intent(self, intent, **kwargs):
            cancel_event = kwargs.get("cancel_event")
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    return {
                        "status": "cancelled",
                        "duration_sec": 0.01,
                        "final_pose": self.current_pose,
                        "distance_to_goal": 10.0,
                    }
                time.sleep(0.01)
            return {
                "status": "timeout",
                "duration_sec": 2.0,
                "final_pose": self.current_pose,
                "distance_to_goal": 10.0,
            }

    bridge = FakeCancel()
    manager = CmuChatTaskManager(
        bridge=bridge,
        places={},
        ros_topics=[],
        output_dir=tmp_path,
        progress_interval_sec=0.01,
    )
    monkeypatch.setattr(
        "rosclaw.apps.cmu_are_bridge._parse_with_llm_chat",
        lambda *_args, **_kwargs: {
            "status": "action",
            "type": "relative",
            "direction": "up",
            "distance": 2.0,
        },
    )

    manager.submit("向上走2米")
    time.sleep(0.05)
    events = manager.cancel_current(reason="测试取消。")

    assert any(event.status == "cancelled" for event in events)
    assert True in bridge.stopped
