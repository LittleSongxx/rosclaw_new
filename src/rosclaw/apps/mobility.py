"""Natural-language mobile-base application for ROSClaw.

This module is intentionally deterministic.  The first useful ROSClaw
application should not depend on an LLM being available; it should prove that
the runtime can turn a user instruction into a safe, recorded robot motion.
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.control.pid_controller import PIDController, PIDGains
from rosclaw.core import Event, EventPriority, Runtime, RuntimeConfig


class InstructionParseError(ValueError):
    """Raised when a natural-language mobility command cannot be parsed."""


@dataclass
class MoveIntent:
    """Parsed one-dimensional mobile-base intent."""

    instruction: str
    target_x: float
    mode: str
    distance_m: float
    direction: str
    confidence: float = 1.0


@dataclass
class MoveRunResult:
    """Result returned by the natural-language movement app."""

    episode_id: str
    robot_id: str
    instruction: str
    intent: MoveIntent
    status: str
    steps: int
    duration_sec: float
    final_x: float
    final_error: float
    tolerance_m: float
    artifact_dir: str
    practice_artifact_dir: str
    trajectory: list[dict[str, Any]]
    event_topics: list[str]

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_dict(self, include_trajectory: bool = True) -> dict[str, Any]:
        data = {
            "episode_id": self.episode_id,
            "robot_id": self.robot_id,
            "instruction": self.instruction,
            "intent": asdict(self.intent),
            "status": self.status,
            "steps": self.steps,
            "duration_sec": self.duration_sec,
            "final_x": self.final_x,
            "final_error": self.final_error,
            "tolerance_m": self.tolerance_m,
            "artifact_dir": self.artifact_dir,
            "practice_artifact_dir": self.practice_artifact_dir,
            "event_topics": self.event_topics,
        }
        if include_trajectory:
            data["trajectory"] = self.trajectory
        return data


_NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres|米)?")
_ABS_X_RE = re.compile(r"(?:x|X)\s*[=:：]?\s*([-+]?\d+(?:\.\d+)?)")

_FORWARD_WORDS = (
    "forward",
    "ahead",
    "front",
    "前进",
    "向前",
    "往前",
    "前行",
)
_BACKWARD_WORDS = (
    "backward",
    "back",
    "reverse",
    "后退",
    "向后",
    "往后",
)
_ABSOLUTE_WORDS = (
    "move to",
    "go to",
    "to x",
    "x=",
    "x:",
    "到 x",
    "移动到",
    "到达",
)


def parse_move_instruction(instruction: str, *, max_abs_target_m: float = 5.0) -> MoveIntent:
    """Parse a small, explicit language command into a PID target."""

    text = instruction.strip()
    if not text:
        raise InstructionParseError("empty movement instruction")

    compact = text.lower().replace("，", ",").replace("。", ".")
    abs_match = _ABS_X_RE.search(text)
    has_absolute_hint = abs_match is not None or any(word in compact for word in _ABSOLUTE_WORDS)

    if abs_match is not None:
        value = float(abs_match.group(1))
        intent = MoveIntent(
            instruction=instruction,
            target_x=value,
            mode="absolute",
            distance_m=abs(value),
            direction="absolute",
        )
        _validate_move_intent(intent, max_abs_target_m)
        return intent

    number_match = _NUMBER_RE.search(text)
    if number_match is None:
        raise InstructionParseError(
            "could not find a distance or x target; try 'forward 1m' or 'move to x=1.0'"
        )

    value = float(number_match.group(1))
    is_backward = any(word in compact for word in _BACKWARD_WORDS)
    is_forward = any(word in compact for word in _FORWARD_WORDS)

    if has_absolute_hint and not is_forward and not is_backward:
        target_x = value
        direction = "absolute"
        mode = "absolute"
        distance = abs(value)
    elif is_backward:
        target_x = -abs(value)
        direction = "backward"
        mode = "relative"
        distance = abs(value)
    else:
        target_x = abs(value) if is_forward else value
        direction = "forward" if target_x >= 0 else "backward"
        mode = "relative"
        distance = abs(value)

    intent = MoveIntent(
        instruction=instruction,
        target_x=target_x,
        mode=mode,
        distance_m=distance,
        direction=direction,
    )
    _validate_move_intent(intent, max_abs_target_m)
    return intent


def _validate_move_intent(intent: MoveIntent, max_abs_target_m: float) -> None:
    if abs(intent.target_x) > max_abs_target_m:
        raise InstructionParseError(
            f"target_x={intent.target_x:.3f}m exceeds mock safety limit {max_abs_target_m:.3f}m"
        )


def run_language_move(
    instruction: str,
    *,
    robot_id: str = "mock_mobile_base",
    output_dir: str | Path = "practice_data/app_runs",
    kp: float = 2.0,
    ki: float = 0.1,
    kd: float = 0.5,
    tolerance_m: float = 0.05,
    dt: float = 0.05,
    max_steps: int = 500,
) -> MoveRunResult:
    """Run a deterministic NL -> PID movement episode on the mock mobile base."""

    intent = parse_move_instruction(instruction)
    episode_id = f"app_move_{int(time.time() * 1000)}"
    trace_id = f"trace_{episode_id}"
    app_artifact_dir = Path(output_dir).expanduser() / episode_id
    app_artifact_dir.mkdir(parents=True, exist_ok=True)

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
            timeline_output_dir=str(app_artifact_dir / "timeline"),
            seekdb_backend="memory",
        )
    )

    trajectory: list[dict[str, Any]] = []
    status = "timeout"
    final_x = 0.0
    start_time = time.time()

    try:
        runtime.initialize()
        runtime.start()
        _publish(runtime, "agent.command", {
            "episode_id": episode_id,
            "instruction": instruction,
            "action": "pid_move",
            "target_x": intent.target_x,
            "agent_request": {"instruction": instruction},
        }, trace_id=trace_id, priority=EventPriority.HIGH)
        _publish(runtime, "agent.response", {
            "episode_id": episode_id,
            "status": "ok",
            "provider": "deterministic_move_parser",
            "intent": asdict(intent),
        }, trace_id=trace_id)
        _publish(runtime, "skill.execution.start", {
            "episode_id": episode_id,
            "skill_name": "pid_move",
            "parameters": {"target_x": intent.target_x, "kp": kp, "ki": ki, "kd": kd},
            "initial_state": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "agent_request": {"instruction": instruction, "intent": asdict(intent)},
        }, trace_id=trace_id, priority=EventPriority.HIGH)

        final_x, status, trajectory = _simulate_pid_1d(
            target_x=intent.target_x,
            kp=kp,
            ki=ki,
            kd=kd,
            dt=dt,
            max_steps=max_steps,
            tolerance_m=tolerance_m,
        )
        duration_sec = time.time() - start_time
        final_error = abs(intent.target_x - final_x)

        for point in trajectory[:: max(1, len(trajectory) // 10 or 1)]:
            _publish(runtime, "robot.telemetry", {
                "episode_id": episode_id,
                "x": point["x"],
                "target_x": intent.target_x,
                "cmd": point["cmd"],
                "error": point["error"],
            }, trace_id=trace_id, priority=EventPriority.LOW)

        result_payload = {
            "status": status,
            "reward": 1.0 if status == "success" else 0.0,
            "final_error": final_error,
            "steps": len(trajectory),
            "artifact_dir": str(app_artifact_dir),
        }
        _publish(runtime, "skill.execution.complete", {
            "episode_id": episode_id,
            "skill_name": "pid_move",
            "result": result_payload,
            "duration_sec": duration_sec,
            "final_state": {"x": final_x, "y": 0.0, "theta": 0.0},
        }, trace_id=trace_id, priority=EventPriority.HIGH)

        _publish(runtime, "rosclaw.sandbox.episode.finished", {
            "episode_id": episode_id,
            "instruction": instruction,
            "status": status,
            "final_state": {"x": final_x, "y": 0.0, "theta": 0.0},
            "outcome": result_payload,
        }, trace_id=trace_id)

        practice_dir = str(Path.home() / ".rosclaw" / "artifacts" / "episodes" / episode_id)
        event_topics = [
            e.topic
            for e in runtime.event_bus.get_history(limit=1000)
            if isinstance(e.payload, dict) and e.payload.get("episode_id") == episode_id
        ]
        run_result = MoveRunResult(
            episode_id=episode_id,
            robot_id=robot_id,
            instruction=instruction,
            intent=intent,
            status=status,
            steps=len(trajectory),
            duration_sec=duration_sec,
            final_x=final_x,
            final_error=final_error,
            tolerance_m=tolerance_m,
            artifact_dir=str(app_artifact_dir),
            practice_artifact_dir=practice_dir,
            trajectory=trajectory,
            event_topics=event_topics,
        )
        _write_move_artifacts(run_result, runtime.event_bus.get_history(limit=1000))
        return run_result
    finally:
        runtime.stop()


def _simulate_pid_1d(
    *,
    target_x: float,
    kp: float,
    ki: float,
    kd: float,
    dt: float,
    max_steps: int,
    tolerance_m: float,
) -> tuple[float, str, list[dict[str, Any]]]:
    pid = PIDController(PIDGains(kp=kp, ki=ki, kd=kd))
    pid.set_output_limit(-1.0, 1.0)
    pid.set_integral_limit(2.0)

    current_x = 0.0
    trajectory: list[dict[str, Any]] = []
    status = "timeout"

    for step in range(max_steps):
        current_x, cmd = pid.simulate_step(current_x, target_x, dt, plant_gain=0.8)
        error = target_x - current_x
        trajectory.append({
            "step": step,
            "t": round(step * dt, 6),
            "x": current_x,
            "y": 0.0,
            "theta": 0.0,
            "target_x": target_x,
            "cmd": cmd,
            "error": error,
        })
        if abs(error) <= tolerance_m and step > 5:
            status = "success"
            break

    return current_x, status, trajectory


def _write_move_artifacts(result: MoveRunResult, events: list[Event]) -> None:
    artifact_dir = Path(result.artifact_dir)
    summary_path = artifact_dir / "summary.json"
    trajectory_json = artifact_dir / "trajectory.json"
    trajectory_csv = artifact_dir / "trajectory.csv"
    events_jsonl = artifact_dir / "events.jsonl"

    summary_path.write_text(
        json.dumps(result.to_dict(include_trajectory=False), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    trajectory_json.write_text(
        json.dumps(result.trajectory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with open(trajectory_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["step", "t", "x", "y", "theta", "target_x", "cmd", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.trajectory:
            writer.writerow({k: row.get(k) for k in fieldnames})

    with open(events_jsonl, "w", encoding="utf-8") as f:
        for event in events:
            payload = event.payload if isinstance(event.payload, dict) else {}
            if payload.get("episode_id") != result.episode_id:
                continue
            f.write(json.dumps({
                "timestamp": event.timestamp,
                "topic": event.topic,
                "source": event.source,
                "trace_id": event.trace_id,
                "payload": payload,
            }, ensure_ascii=False, default=str) + "\n")


def _publish(
    runtime: Runtime,
    topic: str,
    payload: dict[str, Any],
    *,
    trace_id: str,
    priority: EventPriority = EventPriority.NORMAL,
) -> None:
    runtime.event_bus.publish(Event(
        topic=topic,
        payload=payload,
        source="rosclaw.app.mobility",
        trace_id=trace_id,
        priority=priority,
    ))
