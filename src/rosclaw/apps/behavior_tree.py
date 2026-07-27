"""Template behavior-tree patrol app for ROSClaw mobile robots."""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.apps.mobility import _publish
from rosclaw.control.pid_controller import PIDController, PIDGains
from rosclaw.core import EventPriority, Runtime, RuntimeConfig


@dataclass(frozen=True)
class Waypoint:
    """Named 2D navigation target used by the mock patrol app."""

    name: str
    x: float
    y: float
    theta: float = 0.0


@dataclass
class BTAction:
    """A simple action leaf that can be exported to BehaviorTree.CPP XML."""

    node_id: str
    action: str
    target: str


@dataclass
class PatrolPlan:
    """A deterministic patrol behavior tree plan."""

    instruction: str
    waypoints: list[Waypoint]
    actions: list[BTAction]
    return_home: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "return_home": self.return_home,
            "waypoints": [asdict(w) for w in self.waypoints],
            "actions": [asdict(a) for a in self.actions],
        }

    def to_btcpp_xml(self) -> str:
        root = ET.Element("root", {"BTCPP_format": "4"})
        tree = ET.SubElement(root, "BehaviorTree", {"ID": "ROSClawPatrol"})
        sequence = ET.SubElement(tree, "Sequence", {"name": "patrol_sequence"})
        waypoint_map = {w.name: w for w in self.waypoints}
        waypoint_map["home"] = DEFAULT_WAYPOINTS["home"]

        for action in self.actions:
            if action.action == "navigate":
                wp = waypoint_map[action.target]
                ET.SubElement(sequence, "Navigate", {
                    "name": action.node_id,
                    "target": wp.name,
                    "x": f"{wp.x:.3f}",
                    "y": f"{wp.y:.3f}",
                    "theta": f"{wp.theta:.3f}",
                })
            elif action.action == "inspect":
                ET.SubElement(sequence, "Inspect", {
                    "name": action.node_id,
                    "target": action.target,
                })
            elif action.action == "return_home":
                wp = DEFAULT_WAYPOINTS["home"]
                ET.SubElement(sequence, "Navigate", {
                    "name": action.node_id,
                    "target": "home",
                    "x": f"{wp.x:.3f}",
                    "y": f"{wp.y:.3f}",
                    "theta": f"{wp.theta:.3f}",
                })
            elif action.action == "stop":
                ET.SubElement(sequence, "EmergencyStop", {"name": action.node_id})

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")


@dataclass
class PatrolRunResult:
    """Result of executing a template patrol behavior tree."""

    episode_id: str
    robot_id: str
    instruction: str
    status: str
    final_pose: dict[str, float]
    artifact_dir: str
    practice_artifact_dir: str
    plan: PatrolPlan
    bt_xml: str
    timeline: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self, include_traces: bool = True) -> dict[str, Any]:
        data = {
            "episode_id": self.episode_id,
            "robot_id": self.robot_id,
            "instruction": self.instruction,
            "status": self.status,
            "final_pose": self.final_pose,
            "artifact_dir": self.artifact_dir,
            "practice_artifact_dir": self.practice_artifact_dir,
            "plan": self.plan.to_dict(),
        }
        if include_traces:
            data["bt_xml"] = self.bt_xml
            data["timeline"] = self.timeline
            data["trajectory"] = self.trajectory
        return data


DEFAULT_WAYPOINTS: dict[str, Waypoint] = {
    "A": Waypoint("A", 1.0, 0.0, 0.0),
    "B": Waypoint("B", 2.0, 0.8, 0.0),
    "C": Waypoint("C", 3.0, 0.0, 0.0),
    "home": Waypoint("home", 0.0, 0.0, 0.0),
    "kitchen": Waypoint("kitchen", 1.2, -0.6, 0.0),
    "door": Waypoint("door", 2.6, 0.3, 0.0),
}


def parse_patrol_instruction(
    instruction: str,
    *,
    waypoint_catalog: dict[str, Waypoint] | None = None,
) -> PatrolPlan:
    """Create a small Sequence BT from a patrol or inspection instruction."""

    catalog = waypoint_catalog or DEFAULT_WAYPOINTS
    text = instruction.strip()
    if not text:
        raise ValueError("empty patrol instruction")

    labels = _extract_waypoint_labels(text, catalog)
    if not labels and _looks_like_patrol(text):
        labels = ["A", "B", "C"]
    if not labels:
        raise ValueError("no known waypoint found; try A/B/C, kitchen, or door")

    waypoints = [catalog[label] for label in labels]
    actions: list[BTAction] = []
    for index, waypoint in enumerate(waypoints, start=1):
        actions.append(BTAction(f"nav_{index}_{waypoint.name}", "navigate", waypoint.name))
        actions.append(BTAction(f"inspect_{index}_{waypoint.name}", "inspect", waypoint.name))

    return_home = any(word in text.lower() for word in ("return", "home", "back")) or (
        "返回" in text or "回到" in text or "起点" in text
    )
    if return_home:
        actions.append(BTAction("return_home", "return_home", "home"))

    return PatrolPlan(
        instruction=instruction,
        waypoints=waypoints,
        actions=actions,
        return_home=return_home,
    )


def run_patrol_behavior_tree(
    instruction: str,
    *,
    robot_id: str = "mock_mobile_base",
    output_dir: str | Path = "practice_data/app_runs",
    waypoint_catalog: dict[str, Waypoint] | None = None,
    tolerance_m: float = 0.05,
    max_steps_per_nav: int = 500,
) -> PatrolRunResult:
    """Generate and execute a template behavior tree patrol in mock simulation."""

    plan = parse_patrol_instruction(instruction, waypoint_catalog=waypoint_catalog)
    episode_id = f"app_bt_{int(time.time() * 1000)}"
    trace_id = f"trace_{episode_id}"
    artifact_dir = Path(output_dir).expanduser() / episode_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    bt_xml = plan.to_btcpp_xml()

    runtime = Runtime(
        RuntimeConfig(
            robot_id=robot_id,
            default_eurdf_robot=robot_id,
            enable_firewall=True,
            enable_memory=True,
            enable_practice=True,
            enable_how=False,
            enable_auto=False,
            enable_provider=True,
            timeline_output_dir=str(artifact_dir / "timeline"),
            seekdb_backend="memory",
        )
    )

    timeline: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
    status = "success"
    action_map = {w.name: w for w in plan.waypoints}
    action_map["home"] = DEFAULT_WAYPOINTS["home"]

    try:
        runtime.initialize()
        runtime.start()
        _publish(runtime, "agent.command", {
            "episode_id": episode_id,
            "instruction": instruction,
            "action": "patrol_behavior_tree",
            "agent_request": {"instruction": instruction},
        }, trace_id=trace_id, priority=EventPriority.HIGH)
        _publish(runtime, "rosclaw.bt.started", {
            "episode_id": episode_id,
            "tree_id": "ROSClawPatrol",
            "bt_format": "BehaviorTree.CPP XML",
        }, trace_id=trace_id)

        _record_node(timeline, "root_sequence", "Sequence", "RUNNING")
        for action in plan.actions:
            target = action_map[action.target]
            _record_node(timeline, action.node_id, action.action, "RUNNING", target=action.target)
            _publish(runtime, "rosclaw.bt.node.started", {
                "episode_id": episode_id,
                "node_id": action.node_id,
                "node_type": action.action,
                "target": action.target,
                "status": "RUNNING",
            }, trace_id=trace_id)
            _publish(runtime, "skill.execution.start", {
                "episode_id": episode_id,
                "skill_name": action.action,
                "parameters": {"target": asdict(target), "node_id": action.node_id},
                "initial_state": pose.copy(),
            }, trace_id=trace_id, priority=EventPriority.HIGH)

            if action.action in ("navigate", "return_home"):
                pose, nav_status, nav_traj = _simulate_nav_2d(
                    pose,
                    target,
                    tolerance_m=tolerance_m,
                    max_steps=max_steps_per_nav,
                )
                for point in nav_traj:
                    point["node_id"] = action.node_id
                    point["target"] = action.target
                trajectory.extend(nav_traj)
                node_status = "SUCCESS" if nav_status == "success" else "FAILURE"
                if nav_status != "success":
                    status = "failure"
            elif action.action == "inspect":
                observation = {
                    "target": action.target,
                    "summary": f"mock inspection complete at {action.target}",
                }
                trajectory.append({
                    "node_id": action.node_id,
                    "step": 0,
                    "x": pose["x"],
                    "y": pose["y"],
                    "theta": pose["theta"],
                    "observation": observation,
                })
                node_status = "SUCCESS"
            elif action.action == "stop":
                node_status = "FAILURE"
                status = "failure"
            else:
                node_status = "FAILURE"
                status = "failure"

            _record_node(timeline, action.node_id, action.action, node_status, target=action.target)
            _publish(runtime, "skill.execution.complete", {
                "episode_id": episode_id,
                "skill_name": action.action,
                "result": {
                    "status": "success" if node_status == "SUCCESS" else "failure",
                    "node_id": action.node_id,
                    "target": action.target,
                },
                "final_state": pose.copy(),
            }, trace_id=trace_id, priority=EventPriority.HIGH)
            _publish(runtime, "rosclaw.bt.node.completed", {
                "episode_id": episode_id,
                "node_id": action.node_id,
                "node_type": action.action,
                "target": action.target,
                "status": node_status,
            }, trace_id=trace_id)

            if node_status != "SUCCESS":
                break

        _record_node(timeline, "root_sequence", "Sequence", "SUCCESS" if status == "success" else "FAILURE")
        _publish(runtime, "rosclaw.bt.completed", {
            "episode_id": episode_id,
            "tree_id": "ROSClawPatrol",
            "status": "SUCCESS" if status == "success" else "FAILURE",
        }, trace_id=trace_id)
        _publish(runtime, "rosclaw.sandbox.episode.finished", {
            "episode_id": episode_id,
            "instruction": instruction,
            "status": status,
            "final_state": pose.copy(),
            "outcome": {"status": status, "final_pose": pose},
        }, trace_id=trace_id)

        result = PatrolRunResult(
            episode_id=episode_id,
            robot_id=robot_id,
            instruction=instruction,
            status=status,
            final_pose=pose,
            artifact_dir=str(artifact_dir),
            practice_artifact_dir=str(Path.home() / ".rosclaw" / "artifacts" / "episodes" / episode_id),
            plan=plan,
            bt_xml=bt_xml,
            timeline=timeline,
            trajectory=trajectory,
        )
        _write_patrol_artifacts(result)
        return result
    finally:
        runtime.stop()


def _extract_waypoint_labels(text: str, catalog: dict[str, Waypoint]) -> list[str]:
    lowered = text.lower()
    labels: list[str] = []

    for match in re.finditer(r"\b([abcABC])\b|([abcABC])\s*点", text):
        label = (match.group(1) or match.group(2)).upper()
        if label in catalog and label not in labels:
            labels.append(label)

    aliases = {
        "kitchen": "kitchen",
        "厨房": "kitchen",
        "door": "door",
        "doorway": "door",
        "门口": "door",
        "门": "door",
    }
    for word, label in aliases.items():
        if word in lowered or word in text:
            if label in catalog and label not in labels:
                labels.append(label)
    return labels


def _looks_like_patrol(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("patrol", "inspect", "巡检", "检查", "巡视"))


def _simulate_nav_2d(
    pose: dict[str, float],
    target: Waypoint,
    *,
    tolerance_m: float,
    max_steps: int,
) -> tuple[dict[str, float], str, list[dict[str, Any]]]:
    pid_x = PIDController(PIDGains(kp=1.8, ki=0.05, kd=0.35))
    pid_y = PIDController(PIDGains(kp=1.8, ki=0.05, kd=0.35))
    pid_x.set_output_limit(-1.0, 1.0)
    pid_y.set_output_limit(-1.0, 1.0)
    pid_x.set_integral_limit(1.5)
    pid_y.set_integral_limit(1.5)

    x = pose["x"]
    y = pose["y"]
    dt = 0.05
    trajectory: list[dict[str, Any]] = []
    status = "timeout"

    for step in range(max_steps):
        x, cmd_x = pid_x.simulate_step(x, target.x, dt, plant_gain=0.8)
        y, cmd_y = pid_y.simulate_step(y, target.y, dt, plant_gain=0.8)
        error = ((target.x - x) ** 2 + (target.y - y) ** 2) ** 0.5
        trajectory.append({
            "step": step,
            "t": round(step * dt, 6),
            "x": x,
            "y": y,
            "theta": target.theta,
            "target_x": target.x,
            "target_y": target.y,
            "cmd_x": cmd_x,
            "cmd_y": cmd_y,
            "error": error,
        })
        if error <= tolerance_m and step > 5:
            status = "success"
            break

    return {"x": x, "y": y, "theta": target.theta}, status, trajectory


def _record_node(
    timeline: list[dict[str, Any]],
    node_id: str,
    node_type: str,
    status: str,
    *,
    target: str | None = None,
) -> None:
    timeline.append({
        "timestamp": time.time(),
        "node_id": node_id,
        "node_type": node_type,
        "target": target,
        "status": status,
    })


def _write_patrol_artifacts(result: PatrolRunResult) -> None:
    artifact_dir = Path(result.artifact_dir)
    (artifact_dir / "bt.xml").write_text(result.bt_xml, encoding="utf-8")
    (artifact_dir / "bt.json").write_text(
        json.dumps(result.plan.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "summary.json").write_text(
        json.dumps(result.to_dict(include_traces=False), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "trajectory.json").write_text(
        json.dumps(result.trajectory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with open(artifact_dir / "timeline.jsonl", "w", encoding="utf-8") as f:
        for entry in result.timeline:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
