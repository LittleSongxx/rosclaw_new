#!/usr/bin/env python3
"""Evo-RPS acceptance: safe camera fault injection (真机自进化v2 §九/§十).

Injects CAMERA PERCEPTION faults only, at the software level:

* ``kill-node``   — terminate the camera capture process;
* ``stale``       — hold frame delivery (simulated stale feed for the
  freshness gate, driven through the harness, not the device).

NEVER injects faults while the robot is moving, and NEVER touches USB
(unbind/bind/replug) — physical replug during motion is forbidden (§五B).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

MOTION_CHECK_TIMEOUT_S = 30.0


def robot_moving() -> bool:
    """Conservative motion check: any active RPS/gesture executor process.

    Injection is refused while anything that could command the hands is
    alive — the caller passes a quieter system if needed, never the other
    way around.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-af", "evo3_regime_rps|stress_closed_loop|gesture_executor"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return True  # unknown → treat as moving (fail-safe)
    lines = [ln for ln in out.stdout.splitlines() if ln.strip() and "pgrep" not in ln]
    return bool(lines)


def inject_kill_node(pattern: str) -> dict:
    if robot_moving():
        return {
            "ok": False,
            "injected": False,
            "reason": "robot motion detected or undeterminable — camera fault "
            "injection refused (§五B: 禁止在机器人正在运动时注入)",
        }
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10)
    pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
    if not pids:
        return {"ok": False, "injected": False, "reason": f"no camera process matches {pattern!r}"}
    for pid in pids:
        # SIGTERM to the CAMERA NODE ONLY — the node under test is expected
        # to die as the injected fault; our own capture lifecycle always
        # stops gracefully instead (never SIGTERM a streaming pipeline).
        subprocess.run(["kill", "-TERM", str(pid)], check=False, timeout=10)
    return {"ok": True, "injected": True, "killed_pids": pids, "fault": "camera_node_kill"}


def inject_stale(seconds: float) -> dict:
    if robot_moving():
        return {
            "ok": False,
            "injected": False,
            "reason": "robot motion detected or undeterminable — injection refused",
        }
    time.sleep(seconds)  # placeholder marker: the harness drives staleness
    return {
        "ok": True,
        "injected": True,
        "fault": "stale_feed_marker",
        "seconds": seconds,
        "note": "staleness is driven through the harness freshness gate, "
        "never by pausing the physical device",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["kill-node", "stale"])
    parser.add_argument("--pattern", default="realsense|camera_node|d435i_capture")
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.mode == "kill-node":
        result = inject_kill_node(args.pattern)
    else:
        result = inject_stale(args.seconds)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
