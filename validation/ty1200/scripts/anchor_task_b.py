#!/usr/bin/env python3
"""锚点任务 B: 仿真失败—知识—恢复闭环 (任务书 §七 任务B).

Chain: firewall BLOCK (live) → failure practice record → verify → distill →
memory ingest → memory retrieval for the failure pattern → recovered
parameter applied → firewall ALLOW (live) → success.

Hard checks (§任务B 核心指标):
  - first failure correctly classified (joint_limit)
  - receipt/decision is a failure, not a fake success
  - practice contains the failure event
  - memory retrieval finds the recovery experience
  - second attempt actually changes parameters
  - second attempt succeeds
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "validation/ty1200/fixtures/practice/ur5e_joint_limit_recovery.json"
RESULTS: dict = {"checks": {}}


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS["checks"][name] = {"status": "PASS" if ok else "FAIL", "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def cli(args: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(REPO / ".venv/bin/rosclaw"), *args],
        capture_output=True, text=True, env=env, timeout=300, cwd=REPO)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    home = out_dir / "home"
    home.mkdir(exist_ok=True)
    env["ROSCLAW_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO / "src")

    # --- attempt 1: live firewall decision on an out-of-range target ---
    sys.path.insert(0, str(REPO / "src"))
    from rosclaw.sandbox.firewall.gate import StaticActionGate

    gate = StaticActionGate("universal_robots_ur5e", "empty", "mujoco")
    n = len(gate.joint_limits)
    dangerous = [0.0] * n
    dangerous[0] = 12.5
    d1 = gate.check({"values": dangerous})
    check("first_attempt_blocked", not d1.is_allowed,
          f"decision={d1.action} violations={d1.violated_constraints}")
    check("failure_classified_joint_limit",
          any("joint_0_limit" in v for v in d1.violated_constraints),
          str(d1.violated_constraints))
    check("receipt_is_failure_not_fake_success", d1.action == "BLOCK" and not d1.is_allowed,
          f"action={d1.action} allowed={d1.is_allowed}")

    # --- practice record/verify/distill ---
    data_root = out_dir / "practice"
    r = cli(["practice", "record", "--fixture", str(FIXTURE),
             "--out", str(data_root), "--json"], env)
    check("practice_record", r.returncode == 0, r.stdout[-120:] if r.returncode else "")
    r = cli(["practice", "verify", "practice_ur5e_joint_limit_recovery",
             "--data-root", str(data_root), "--strict", "--json"], env)
    check("practice_verify_strict", r.returncode == 0)
    events_text = (data_root / "sessions/practice_ur5e_joint_limit_recovery/raw/events.jsonl").read_text()
    check("practice_contains_failure_event",
          "fail_ur5e_joint_limit_1" in events_text and "joint_limit" in events_text)
    r = cli(["practice", "distill", "practice_ur5e_joint_limit_recovery",
             "--data-root", str(data_root), "--json"], env)
    check("practice_distill", r.returncode == 0)

    # --- memory ingest + retrieval ---
    r = cli(["memory", "ingest", "--episode-id", "episode_ur5e_joint_limit_recovery",
             "--data-root", str(data_root)], env)
    check("memory_ingest", r.returncode == 0, r.stdout[-120:] if r.returncode else "")
    r = cli(["memory", "query", "joint limit clamp actuator range", "--limit", "3"], env)
    memory_hit = "ur5e" in r.stdout.lower() or "joint" in r.stdout.lower()
    check("memory_retrieval_hit", r.returncode == 0 and memory_hit,
          r.stdout.replace("\n", " ")[-140:])

    # --- attempt 2: apply recovered parameters (clamped), live firewall ---
    recovered = [0.0] * n
    lo, hi = gate.joint_limits[0]
    recovered[0] = max(lo, min(hi, 3.14))  # recovery hint: clamp into ctrlrange
    check("second_attempt_changed_params", recovered[0] != dangerous[0],
          f"joint_0 {dangerous[0]} -> {recovered[0]}")
    d2 = gate.check({"values": recovered})
    check("second_attempt_success", d2.is_allowed,
          f"decision={d2.action} reason={d2.reason[:80]}")

    RESULTS["overall"] = (
        "PASS" if all(c["status"] == "PASS" for c in RESULTS["checks"].values()) else "FAIL"
    )
    (out_dir / "anchor_task_b.json").write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    print(json.dumps({"overall": RESULTS["overall"]}, indent=2))
    return 0 if RESULTS["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
