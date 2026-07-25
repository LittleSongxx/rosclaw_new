"""Camera recovery procedures + procedural memory (PR-EVO-HW-2 §五B/§九).

Three bounded candidate procedures for ``camera_no_fresh_frame``:

* A — immediate node restart (fast, often unstable afterwards);
* B — stop → wait 1s → hardware_reset → re-enumerate → start →
  10 consecutive fresh frames → new camera session id → resume;
* C — stop → 2s backoff → re-enumerate → start → RGB/Depth sync
  check → resume.

Each run is a timed state machine; the outcome is distilled into a
PROCEDURAL memory (failure_signature, device identity, pre-fault frame
age, ordered steps with per-step duration, recovered?, TTR, manual
replug needed?, recurrence) — the artifact 闭环 B needs to prove that the
SECOND fault is recovered faster by retrieved memory than the first.
Recovery restores perception only: it never unlocks a REAL robot permit;
a fresh camera session + fresh observation are required before resume.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .freshness import CAMERA_NO_FRESH_FRAME

RECOVERY_PROCEDURES: dict[str, list[str]] = {
    "A_immediate_restart": ["stop_node", "start_node"],
    "B_reset_fresh_frames": [
        "stop_node",
        "wait_1s",
        "hardware_reset",
        "wait_reenumeration",
        "start_node",
        "require_10_fresh_frames",
        "new_camera_session",
        "resume_task",
    ],
    "C_backoff_sync_check": [
        "stop_node",
        "wait_2s",
        "wait_reenumeration",
        "start_node",
        "rgb_depth_sync_check",
        "resume_task",
    ],
}


@dataclass
class StepResult:
    name: str
    ok: bool
    duration_s: float
    detail: str = ""


@dataclass
class RecoveryResult:
    procedure: str
    recovered: bool
    ttr_s: float
    steps: list[StepResult] = field(default_factory=list)
    manual_replug_required: bool = False
    camera_session_id: str | None = None
    note: str = ""

    def to_procedural_memory(
        self,
        *,
        body_id: str,
        device: dict[str, Any],
        pre_fault_frame_age_ms: float | None,
        robot_id: str,
        evidence_refs: list[str],
    ) -> dict[str, Any]:
        """The §Camera Experiment B procedural memory record."""
        return {
            "memory_type": "procedural",
            "failure_type": CAMERA_NO_FRESH_FRAME,
            "robot_id": robot_id,
            "body_id": body_id,
            "title": f"相机无新帧恢复程序 {self.procedure}"
            + ("（成功）" if self.recovered else "（失败）"),
            "document": (
                f"camera_no_fresh_frame on {device.get('name')} "
                f"(serial {device.get('serial')}, fw {device.get('firmware')}); "
                f"pre-fault frame age {pre_fault_frame_age_ms}ms; "
                f"procedure {self.procedure}: "
                + " → ".join(s.name for s in self.steps)
                + f"; recovered={self.recovered}; TTR {self.ttr_s:.1f}s; "
                f"manual_replug={self.manual_replug_required}"
            ),
            "outcome": "success" if self.recovered else "failure",
            "metadata": {
                "recovery_hint": " → ".join(RECOVERY_PROCEDURES[self.procedure]),
                "procedure": self.procedure,
                "device": device,
                "pre_fault_frame_age_ms": pre_fault_frame_age_ms,
                "steps": [
                    {"name": s.name, "ok": s.ok, "duration_s": round(s.duration_s, 3)}
                    for s in self.steps
                ],
                "recovered": self.recovered,
                "ttr_s": round(self.ttr_s, 3),
                "manual_replug_required": self.manual_replug_required,
                "camera_session_id": self.camera_session_id,
            },
            "evidence_refs": evidence_refs,
        }


# Step handlers get a context dict so tests can drive the machine with
# fakes while production drives the real capture lifecycle.
StepHandler = Callable[[dict[str, Any]], tuple[bool, str]]


def _default_handlers() -> dict[str, StepHandler]:
    def ok(detail: str = "") -> tuple[bool, str]:
        return True, detail

    return {
        "stop_node": lambda ctx: ctx["stop"](),
        "start_node": lambda ctx: ctx["start"](),
        "wait_1s": lambda ctx: (time.sleep(1.0), ok())[1],
        "wait_2s": lambda ctx: (time.sleep(2.0), ok())[1],
        "hardware_reset": lambda ctx: ctx["hardware_reset"](),
        "wait_reenumeration": lambda ctx: ctx["wait_reenumeration"](),
        "require_10_fresh_frames": lambda ctx: ctx["require_fresh_frames"](10),
        "rgb_depth_sync_check": lambda ctx: ctx["rgb_depth_sync_check"](),
        "new_camera_session": lambda ctx: ctx["new_camera_session"](),
        "resume_task": lambda ctx: ok("perception restored; REAL permit untouched"),
    }


def run_recovery(
    procedure: str,
    *,
    handlers: dict[str, StepHandler] | None = None,
) -> RecoveryResult:
    """Execute one procedure as a timed state machine (fail-fast)."""
    if procedure not in RECOVERY_PROCEDURES:
        raise ValueError(f"unknown recovery procedure {procedure!r}")
    table = dict(_default_handlers())
    if handlers:
        table.update(handlers)
    result = RecoveryResult(procedure=procedure, recovered=False, ttr_s=0.0)
    started = time.monotonic()
    context: dict[str, Any] = {}
    for name in RECOVERY_PROCEDURES[procedure]:
        handler = table.get(name)
        step_started = time.monotonic()
        if handler is None:
            ok, detail = False, f"no handler for step {name}"
        else:
            try:
                ok, detail = handler(context)
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"{type(exc).__name__}: {exc}"
        result.steps.append(
            StepResult(name=name, ok=bool(ok), duration_s=time.monotonic() - step_started, detail=detail)
        )
        if not ok:
            if name in ("hardware_reset", "wait_reenumeration"):
                result.manual_replug_required = True
            result.ttr_s = time.monotonic() - started
            result.note = f"step {name} failed: {detail}"
            return result
        if name == "new_camera_session":
            result.camera_session_id = context.get("camera_session_id")
    result.recovered = True
    result.ttr_s = time.monotonic() - started
    result.note = "perception restored; resume requires fresh observation, never a REAL permit"
    return result
