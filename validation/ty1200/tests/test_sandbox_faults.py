"""Sandbox firewall fault-injection tests (任务书 §八 M08 + §二十一 MuJoCo).

Fault cases: joint target out of range, NaN, Inf, overspeed (joint delta),
excessive force, dimension mismatch, and a legal action as the positive
control. Every dangerous case must be BLOCKed or MODIFYed — never ALLOWed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rosclaw.sandbox.firewall.gate import StaticActionGate

ROBOT = "universal_robots_ur5e"
RESULTS: dict[str, dict] = {}


@pytest.fixture(scope="module")
def gate() -> StaticActionGate:
    return StaticActionGate(ROBOT, "empty", "mujoco")


def _record(name: str, decision) -> None:
    RESULTS[name] = {
        "action": decision.action,
        "is_allowed": decision.is_allowed,
        "risk_score": decision.risk_score,
        "violations": decision.violated_constraints,
        "reason": decision.reason,
    }


def test_joint_target_out_of_range_blocked(gate):
    values = [0.0] * len(gate.joint_limits)
    lo, hi = gate.joint_limits[0]
    values[0] = hi + (hi - lo) * 2 + 10.0  # far beyond actuator range
    d = gate.check({"values": values})
    _record("joint_out_of_range", d)
    assert not d.is_allowed
    assert any("joint_0_limit" in v for v in d.violated_constraints)


def test_nan_blocked(gate):
    values = [0.0] * len(gate.joint_limits)
    values[2] = float("nan")
    d = gate.check({"values": values})
    _record("nan", d)
    assert d.action == "BLOCK"
    assert "invalid_joint_values" in d.violated_constraints


def test_inf_blocked(gate):
    values = [0.0] * len(gate.joint_limits)
    values[1] = float("inf")
    d = gate.check({"values": values})
    _record("inf", d)
    assert d.action == "BLOCK"


def test_dimension_mismatch_blocked(gate):
    d = gate.check({"values": [0.0, 0.0]})
    _record("dimension_mismatch", d)
    assert d.action == "BLOCK"
    assert "action_dimension_mismatch" in d.violated_constraints


def test_overspeed_flagged(gate):
    n = len(gate.joint_limits)
    current = [0.0] * n
    values = [0.0] * n
    values[0] = gate.joint_limits[0][1] * 0.5  # legal target
    current[0] = -values[0]  # enormous delta -> implied overspeed
    d = gate.check({"values": values, "current": current})
    _record("overspeed", d)
    assert not d.is_allowed
    assert any("velocity" in v for v in d.violated_constraints)


def test_excessive_force_flagged(gate):
    values = [0.0] * len(gate.joint_limits)
    d = gate.check({"values": values, "force": gate.MAX_TCP_FORCE * 10})
    _record("excessive_force", d)
    assert not d.is_allowed
    assert "pfl_force" in d.violated_constraints


def test_legal_action_allowed(gate):
    n = len(gate.joint_limits)
    mid = [(lo + hi) / 2 for lo, hi in gate.joint_limits]
    d = gate.check({"values": mid})
    _record("legal_action", d)
    assert d.is_allowed, d.reason


def test_zz_write_results():
    out = {
        "cases": RESULTS,
        "blocked_or_modified": sum(
            1 for k, v in RESULTS.items()
            if k != "legal_action" and not v["is_allowed"]
        ),
        "dangerous_cases": len(RESULTS) - 1,
        "overall": "PASS" if all(
            not v["is_allowed"] for k, v in RESULTS.items() if k != "legal_action"
        ) else "FAIL",
    }
    report = os.environ.get("TY1200_VALIDATION_REPORT_DIR")
    if report:
        Path(report).mkdir(parents=True, exist_ok=True)
        (Path(report) / "sandbox_faults.json").write_text(json.dumps(out, indent=2))
    assert out["overall"] == "PASS"
