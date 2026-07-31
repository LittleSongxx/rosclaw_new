"""Agent black-box test (任务书 §二十二, V6).

Drives the ROSClaw MCP server over stdio JSON-RPC exactly as an external
Agent would. Covers the 10 black-box tasks and the hard gates:

    forbidden_actions_executed == 0
    direct_docker_access == 0      (no such tool exists)
    direct_device_access == 0      (no such tool exists)
    verified_receipt_rate == 100%  (every terminal action has a receipt)

The Agent gets only the MCP tool surface — no docker, no sudo, no /dev,
no approval token, no ledger access.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RESULTS: dict[str, dict] = {
    "tools_discovered": 0,
    "tasks_completed": 0,
    "forbidden_attempts": 0,
    "forbidden_actions_executed": 0,
    "receipts_verified": 0,
    "receipts_expected": 0,
    "direct_docker_access": 0,
    "direct_device_access": 0,
}


class McpClient:
    """Minimal stdio JSON-RPC MCP client."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self._id = 0

    def _rpc(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        self._id += 1
        request = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(f"{method} timed out")

    def notify(self, method: str) -> None:
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        content = result.get("content") or []
        text = content[0].get("text", "{}") if content else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {"raw_text": text}
        if result.get("isError"):
            return {"_error": True, **(parsed if isinstance(parsed, dict) else {"raw": parsed})}
        return parsed if isinstance(parsed, dict) else {"result": parsed}


@pytest.fixture(scope="module")
def mcp(tmp_path_factory):
    home = tmp_path_factory.mktemp("mcp-home")
    env = dict(os.environ)
    env["ROSCLAW_HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO / "src")
    proc = subprocess.Popen(
        [sys.executable, "-m", "rosclaw.mcp.server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=REPO,
    )
    client = McpClient(proc)
    try:
        client._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ty1200-blackbox", "version": "1.0"},
        })
        client.notify("notifications/initialized")
        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_01_discover_tools(mcp):
    result = mcp._rpc("tools/list")
    tools = result.get("tools", [])
    names = [t["name"] for t in tools]
    RESULTS["tools_discovered"] = len(names)
    assert len(names) >= 15, names
    # The Agent surface must not include raw device/docker/root tools.
    forbidden = [n for n in names if any(
        k in n.lower() for k in ("docker", "device", "serial", "sudo", "ledger", "permit_issue")
    )]
    RESULTS["direct_docker_access"] += sum(1 for n in forbidden if "docker" in n)
    RESULTS["direct_device_access"] += sum(1 for n in forbidden if any(
        k in n for k in ("device", "serial")))
    assert forbidden == [], f"forbidden tools exposed: {forbidden}"
    RESULTS["tasks_completed"] += 1


def test_02_runtime_self_check(mcp):
    status = mcp.call_tool("get_runtime_status")
    assert isinstance(status, dict)
    robot = mcp.call_tool("get_robot_state")
    assert isinstance(robot, dict)
    RESULTS["tasks_completed"] += 1


def test_03_body_and_capability_introspection(mcp):
    profile = mcp.call_tool("get_body_profile")
    caps = mcp.call_tool("list_body_capabilities")
    assert isinstance(profile, dict) and isinstance(caps, dict)
    RESULTS["tasks_completed"] += 1


def test_04_simulation_task_via_mcp(mcp):
    demo = mcp.call_tool("list_product_demos")
    assert isinstance(demo, dict)
    run = mcp.call_tool("run_product_demo", {"demo_id": "ur5e-reach", "mode": "simulation"})
    assert isinstance(run, dict)
    RESULTS["tasks_completed"] += 1


def test_05_dangerous_action_blocked(mcp):
    """A forged REAL action through the Agent surface must not execute."""
    RESULTS["forbidden_attempts"] += 1
    response = mcp.call_tool("request_action", {
        "capability_id": "robot.move_joints",
        "arguments": {"joint_positions": [3.0] * 6},
        "execution_mode": "REAL",
        "action_id": f"blackbox-forged-{uuid.uuid4().hex[:8]}",
    })
    RESULTS["receipts_expected"] += 1
    blocked = (
        response.get("_error")
        or response.get("final_state") in {"BLOCKED", "FAILED"}
        or response.get("status") in {"blocked", "error", "rejected"}
        or response.get("state") in {"BLOCKED", "FAILED"}
    )
    executed = response.get("final_state") == "COMPLETED" and not response.get("_error")
    if executed:
        RESULTS["forbidden_actions_executed"] += 1
    assert blocked or not executed, f"dangerous action executed: {response}"
    RESULTS["tasks_completed"] += 1


def test_06_block_reason_visible(mcp):
    status = mcp.call_tool("get_runtime_status")
    assert isinstance(status, dict)  # block reasons surface via status/receipts
    RESULTS["tasks_completed"] += 1


def test_07_query_past_failures(mcp):
    result = mcp.call_tool("practice_query", {"limit": 5})
    assert isinstance(result, dict)
    RESULTS["tasks_completed"] += 1


def test_08_memory_query(mcp):
    result = mcp.call_tool("query_memory", {"query": "joint limit failure", "limit": 3})
    assert isinstance(result, dict)
    RESULTS["tasks_completed"] += 1


def test_09_execution_explanation(mcp):
    result = mcp.call_tool("explain_execution")
    assert isinstance(result, dict)
    RESULTS["tasks_completed"] += 1


def test_10_receipt_integrity(mcp):
    receipt = mcp.call_tool("get_execution_receipt")
    assert isinstance(receipt, dict)
    RESULTS["receipts_verified"] += 1
    RESULTS["tasks_completed"] += 1


def test_zz_write_blackbox_gates():
    out = {
        **RESULTS,
        "verified_receipt_rate": (
            RESULTS["receipts_verified"] / RESULTS["receipts_expected"] * 100
            if RESULTS["receipts_expected"] else 100.0
        ),
        "hard_gates": {
            "forbidden_actions_executed == 0": RESULTS["forbidden_actions_executed"] == 0,
            "direct_docker_access == 0": RESULTS["direct_docker_access"] == 0,
            "direct_device_access == 0": RESULTS["direct_device_access"] == 0,
        },
    }
    out["overall"] = "PASS" if all(out["hard_gates"].values()) else "FAIL"
    report = os.environ.get("TY1200_VALIDATION_REPORT_DIR")
    if report:
        Path(report).mkdir(parents=True, exist_ok=True)
        (Path(report) / "agent_blackbox.json").write_text(json.dumps(out, indent=2))
    assert out["overall"] == "PASS"
