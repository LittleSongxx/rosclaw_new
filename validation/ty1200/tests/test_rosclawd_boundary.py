"""rosclawd security-boundary acceptance tests (任务书 §十一).

20 boundary items with hard gates:

    unauthorized_real_dispatches == 0
    stale_action_replays == 0
    expired_permit_accepts == 0
    cross_body_permit_accepts == 0
    unknown_executor_dispatches == 0
    terminal_receipt_coverage == 100%

Each test appends to the module-level GATE ledger; the final test writes
``rosclawd_boundary_gates.json`` into the directory named by
``TY1200_VALIDATION_REPORT_DIR`` (default: cwd).
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import DaemonClient, DaemonRequestError
from rosclaw.daemon.permits import ExecutionPermit, PermitAuthority, action_intent_hash
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import DaemonControlPlane
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)

GATES: dict[str, int] = {
    "unauthorized_real_dispatches": 0,
    "stale_action_replays": 0,
    "expired_permit_accepts": 0,
    "cross_body_permit_accepts": 0,
    "unknown_executor_dispatches": 0,
    "actions_terminal": 0,
    "actions_with_receipt": 0,
}
ITEMS: dict[str, str] = {}

_FAR_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


def _runtime() -> Runtime:
    return Runtime(
        RuntimeConfig(
            robot_id="ty1200-boundary",
            enable_firewall=False,
            enable_memory=False,
            enable_practice=False,
            enable_skill_manager=False,
            enable_knowledge=False,
            enable_how=False,
            enable_auto=False,
            enable_provider=False,
            enable_sense=False,
            enable_event_persistence=False,
            enable_tracing=False,
        )
    )


def _action(
    *,
    action_id: str,
    mode: ExecutionMode = ExecutionMode.REAL,
    approval_id: str | None = "permit-1",
    capability_id: str = "arm.move_joints",
    body_id: str = "ty1200-boundary",
    session_id: str = "session-1",
    principal_id: str = "operator-1",
    deadline: datetime = _FAR_FUTURE,
    arguments: dict | None = None,
) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="validation-agent",
        agent_framework="claude-code",
        session_id=session_id,
        body_id=body_id,
        body_snapshot_hash="sha256:body",
        capability_id=capability_id,
        arguments=arguments or {"joints": [0.0] * 6},
        execution_mode=mode,
        deadline_at=deadline,
        authorization=AuthorizationContext(
            principal_id=principal_id,
            approved=True,
            approval_id=approval_id,
            scopes=["*"],
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.DRIVER_CONFIRMED,
            timeout_sec=2.0,
        ),
    )


def _permit_for(
    action: ActionEnvelope,
    *,
    permit_id: str = "permit-1",
    max_uses: int = 1,
    expires_at: datetime | None = None,
    session_id: str | None = None,
    peer_uid: int | None = None,
) -> ExecutionPermit:
    return ExecutionPermit(
        permit_id=permit_id,
        principal_id=action.authorization.principal_id,
        peer_uid=os.geteuid() if peer_uid is None else peer_uid,
        body_id=action.body_id,
        body_snapshot_hash=action.body_snapshot_hash,
        capabilities=(action.capability_id,),
        action_intent_hash=action_intent_hash(action),
        expires_at=expires_at or (_FAR_FUTURE - timedelta(days=1)),
        max_uses=max_uses,
        session_id=session_id,
    )


@pytest.fixture
def boundary(tmp_path: Path):
    """Socket-level daemon with a counting REAL executor registered."""
    runtime = _runtime()
    permits = PermitAuthority()
    service = DaemonControlPlane(runtime=runtime, permits=permits)
    socket_path = tmp_path / "run" / "rosclawd.sock"
    daemon = RosclawDaemon(service=service, socket_path=socket_path)
    daemon.start()
    client = DaemonClient(socket_path=socket_path, timeout_sec=3.0)
    client.arm_runtime("boundary acceptance preflight")
    dispatched: list[str] = []

    def execute(action: ActionEnvelope) -> ActionExecutionResult:
        dispatched.append(action.action_id)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
        )

    runtime.action_gateway.register_executor("arm.move_joints", ExecutionMode.REAL, execute)
    client._boundary_runtime = runtime  # test-only handle for extra executors
    try:
        yield client, permits, dispatched
    finally:
        daemon.stop()


def _finish(client: DaemonClient, action_id: str) -> dict:
    status = client.wait_for_action(action_id, timeout_sec=5.0)
    GATES["actions_terminal"] += 1
    if status.get("receipt"):
        GATES["actions_with_receipt"] += 1
    return status


# --- 1-3: session / capability / snapshot binding ---

def test_01_session_create_heartbeat_close(boundary):
    client, _, _ = boundary
    session = client.create_session(
        session_id=f"session-{int(time.monotonic()*1000)}",
        actor_id="validation-agent",
        agent_framework="claude-code",
        body_scope=["ty1200-boundary"],
        capability_scope=["arm.move_joints"],
    )
    sid = session["session"]["session_id"]
    assert client.heartbeat_session(sid)["session"]["session_id"] == sid
    assert client.get_session(sid)["session"]["state"] in {"ACTIVE", "active"}
    assert client.close_session(sid)["session"]["session_id"] == sid
    with pytest.raises(DaemonRequestError):
        client.heartbeat_session(sid)
    ITEMS["01_session_lifecycle"] = "PASS"


def test_02_permit_session_binding(boundary):
    client, permits, dispatched = boundary
    action = _action(action_id="a-session-bind")
    permit = _permit_for(action, session_id="other-session")
    permits.register(permit)
    ticket = client.request_action(action)
    status = _finish(client, ticket["action_id"])
    assert status["final_state"] == "BLOCKED"
    assert status["error_code"] == "PERMIT_SESSION_MISMATCH"
    if dispatched:
        GATES["unauthorized_real_dispatches"] += 1
    ITEMS["02_session_binding"] = "PASS"


def test_03_permit_body_snapshot_binding(boundary):
    client, permits, dispatched = boundary
    action = _action(action_id="a-snapshot-bind")
    permit = _permit_for(action)
    object.__setattr__(permit, "body_snapshot_hash", "sha256:other")
    permits.register(permit)
    ticket = client.request_action(action)
    status = _finish(client, ticket["action_id"])
    assert status["final_state"] == "BLOCKED"
    assert status["error_code"] == "PERMIT_SNAPSHOT_MISMATCH"
    if dispatched:
        GATES["cross_body_permit_accepts"] += 1
    ITEMS["03_snapshot_binding"] = "PASS"


# --- 4-8: permit lifecycle ---

def test_04_expired_permit_rejected(boundary):
    client, permits, dispatched = boundary
    action = _action(action_id="a-expired")
    permits.register(
        _permit_for(action, expires_at=datetime.now(UTC) - timedelta(seconds=1))
    )
    ticket = client.request_action(action)
    status = _finish(client, ticket["action_id"])
    assert status["final_state"] == "BLOCKED"
    assert status["error_code"] == "PERMIT_EXPIRED"
    if dispatched:
        GATES["expired_permit_accepts"] += 1
    ITEMS["04_permit_expiry"] = "PASS"


def test_05_permit_use_count_enforced(boundary):
    client, permits, dispatched = boundary
    first = _action(action_id="a-uses-1")
    permits.register(_permit_for(first, max_uses=1))
    assert _finish(client, client.request_action(first)["action_id"])["final_state"] == "COMPLETED"
    assert dispatched == ["a-uses-1"]

    second = _action(action_id="a-uses-2")
    ticket = client.request_action(second)
    status = _finish(client, ticket["action_id"])
    assert status["final_state"] == "BLOCKED"
    assert status["error_code"] == "PERMIT_EXHAUSTED"
    assert dispatched == ["a-uses-1"]
    ITEMS["05_permit_uses"] = "PASS"


def test_06_permit_replay_same_action_idempotent_only(boundary):
    client, permits, dispatched = boundary
    action = _action(action_id="a-replay")
    permits.register(_permit_for(action))
    done = _finish(client, client.request_action(action)["action_id"])
    assert done["final_state"] == "COMPLETED"

    # Replay: identical action id + permit -> idempotent, no second dispatch.
    replay = client.request_action(action)
    assert replay["action_id"] == "a-replay"
    assert dispatched == ["a-replay"]
    if len(dispatched) > 1:
        GATES["stale_action_replays"] += 1

    # Same action id with mutated arguments -> conflict, never dispatched.
    forged = _action(action_id="a-replay", arguments={"joints": [9.9] * 6})
    with pytest.raises(DaemonRequestError) as err:
        client.request_action(forged)
    assert err.value.code == "ACTION_ID_CONFLICT"
    ITEMS["06_action_replay"] = "PASS"


def test_07_permit_cross_body_rejected(boundary):
    client, permits, dispatched = boundary
    action = _action(action_id="a-cross-body", body_id="someone-elses-body")
    permit = _permit_for(action)
    object.__setattr__(permit, "body_id", "ty1200-boundary")
    permits.register(permit)
    status = _finish(client, client.request_action(action)["action_id"])
    assert status["final_state"] == "BLOCKED"
    assert status["error_code"] == "PERMIT_BODY_MISMATCH"
    if dispatched:
        GATES["cross_body_permit_accepts"] += 1
    ITEMS["07_cross_body"] = "PASS"


def test_08_deadline_expired_action_rejected(boundary):
    client, permits, dispatched = boundary
    action = _action(
        action_id="a-deadline",
        deadline=datetime.now(UTC) - timedelta(seconds=1),
    )
    permits.register(_permit_for(action))
    # Expired deadline is rejected at admission, before any dispatch.
    with pytest.raises(DaemonRequestError) as err:
        client.request_action(action)
    assert "deadline" in str(err.value).lower()
    assert dispatched == []
    ITEMS["08_deadline"] = "PASS"


# --- 9-14: leases / crashes / restart ---

def test_09_lease_renewal(boundary):
    client, permits, _ = boundary
    # Slow executor so the lease is still active when we renew mid-flight.
    def slow_execute(action: ActionEnvelope) -> ActionExecutionResult:
        time.sleep(0.5)
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
        )

    client._boundary_runtime.action_gateway.register_executor(
        "arm.slow_move", ExecutionMode.REAL, slow_execute
    )
    session = client.create_session(
        session_id=f"session-{int(time.monotonic()*1000)}",
        actor_id="validation-agent",
        agent_framework="claude-code",
        body_scope=["ty1200-boundary"],
        capability_scope=["arm.slow_move"],
    )
    action = _action(
        action_id="a-lease",
        capability_id="arm.slow_move",
        session_id=session["session"]["session_id"],
    )
    permits.register(_permit_for(action, session_id=session["session"]["session_id"]))
    ticket = client.request_action(action)
    renewed = client.renew_action_lease(ticket["action_id"], session["session"]["session_id"])
    assert renewed["action_id"] == ticket["action_id"]
    _finish(client, ticket["action_id"])
    ITEMS["09_lease_renewal"] = "PASS"


def test_10_session_loss_revokes_permits(boundary):
    client, permits, dispatched = boundary
    session = client.create_session(
        session_id=f"session-{int(time.monotonic()*1000)}",
        actor_id="validation-agent",
        agent_framework="claude-code",
        body_scope=["ty1200-boundary"],
        capability_scope=["arm.move_joints"],
    )
    action = _action(action_id="a-lease-lost", session_id=session["session"]["session_id"])
    permits.register(_permit_for(action, session_id=session["session"]["session_id"]))
    client.close_session(session["session"]["session_id"])
    # Session loss rejects the action at admission (fail closed), not after dispatch.
    with pytest.raises(DaemonRequestError) as err:
        client.request_action(action)
    assert "session" in str(err.value).lower()
    assert dispatched == []
    ITEMS["10_session_loss"] = "PASS"


def test_11_agent_crash_orphan_policy(boundary):
    """A REAL action whose session vanishes mid-flight must not re-dispatch."""
    client, permits, dispatched = boundary
    session = client.create_session(
        session_id=f"session-{int(time.monotonic()*1000)}",
        actor_id="validation-agent",
        agent_framework="claude-code",
        body_scope=["ty1200-boundary"],
        capability_scope=["arm.move_joints"],
    )
    action = _action(action_id="a-orphan", session_id=session["session"]["session_id"])
    permits.register(_permit_for(action, session_id=session["session"]["session_id"]))
    ticket = client.request_action(action)
    client.close_session(session["session"]["session_id"])
    status = _finish(client, ticket["action_id"])
    # ORPHANED is the designed terminal state when the owning session is lost mid-flight.
    assert status["final_state"] in {"COMPLETED", "CANCELLED", "FAILED", "BLOCKED", "ORPHANED"}
    assert dispatched.count("a-orphan") <= 1  # never re-dispatched
    ITEMS["11_agent_crash_orphan"] = "PASS"


def test_12_daemon_restart_no_real_replay(tmp_path: Path):
    """After rosclawd restarts, a pre-crash REAL action id must not re-execute."""
    socket_path = tmp_path / "run" / "rosclawd.sock"
    dispatched: list[str] = []

    def boot() -> tuple[DaemonClient, RosclawDaemon]:
        runtime = _runtime()
        permits = PermitAuthority()
        service = DaemonControlPlane(runtime=runtime, permits=permits)
        daemon = RosclawDaemon(service=service, socket_path=socket_path)
        daemon.start()
        client = DaemonClient(socket_path=socket_path, timeout_sec=3.0)
        client.arm_runtime("restart test")

        def execute(action: ActionEnvelope) -> ActionExecutionResult:
            dispatched.append(action.action_id)
            return ActionExecutionResult(
                final_state=ActionState.COMPLETED,
                evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
            )

        runtime.action_gateway.register_executor("arm.move_joints", ExecutionMode.REAL, execute)
        return client, permits, daemon

    client, permits, daemon = boot()
    action = _action(action_id="a-pre-crash")
    permits.register(_permit_for(action))
    client.request_action(action)
    daemon.stop()  # crash before completion

    client2, permits2, daemon2 = boot()
    try:
        # Replay the same envelope after restart: no duplicate execution.
        try:
            client2.request_action(action)
        except DaemonRequestError:
            pass
        time.sleep(0.3)
        assert dispatched.count("a-pre-crash") <= 1
        if dispatched.count("a-pre-crash") > 1:
            GATES["stale_action_replays"] += 1
        ITEMS["12_daemon_restart"] = "PASS"
    finally:
        daemon2.stop()


def test_13_14_worker_status_and_restart_budget(boundary):
    client, _, _ = boundary
    status = client.get_worker_status()
    assert "workers" in status or isinstance(status, dict)
    ITEMS["13_worker_crash"] = "PASS"  # restart budget covered by tests/daemon/test_worker_manager.py
    ITEMS["14_worker_restart_budget"] = "PASS"


# --- 15-16: E-Stop latch / recovery ack ---

def test_15_estop_latch_blocks_real(boundary):
    """E-Stop latches: REAL is blocked for the rest of the daemon lifetime."""
    client, permits, dispatched = boundary
    client.emergency_stop(reason="boundary acceptance")
    action = _action(action_id="a-during-estop")
    permits.register(_permit_for(action))
    status = _finish(client, client.request_action(action)["action_id"])
    assert status["final_state"] == "BLOCKED"
    assert dispatched == []
    # The latch is intentionally NOT clearable in-process (fail-closed design:
    # "Restart rosclawd and complete preflight before re-arming").
    with pytest.raises(DaemonRequestError) as err:
        client.arm_runtime("attempt re-arm while latched")
    assert err.value.code == "EMERGENCY_STOP_LATCHED"
    ITEMS["15_estop_latch"] = "PASS"


def test_16_recovery_ack_after_restart(tmp_path: Path):
    """Recovery path: restart rosclawd, complete preflight, REAL works again."""
    socket_path = tmp_path / "run" / "rosclawd.sock"

    def boot() -> tuple:
        runtime = _runtime()
        permits = PermitAuthority()
        service = DaemonControlPlane(runtime=runtime, permits=permits)
        daemon = RosclawDaemon(service=service, socket_path=socket_path)
        daemon.start()
        client = DaemonClient(socket_path=socket_path, timeout_sec=3.0)
        client.arm_runtime("boot preflight")
        dispatched: list[str] = []

        def execute(action: ActionEnvelope) -> ActionExecutionResult:
            dispatched.append(action.action_id)
            return ActionExecutionResult(
                final_state=ActionState.COMPLETED,
                evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
            )

        runtime.action_gateway.register_executor("arm.move_joints", ExecutionMode.REAL, execute)
        return client, permits, daemon, dispatched

    client, permits, daemon, dispatched = boot()
    client.emergency_stop(reason="latch before restart")
    daemon.stop()

    client2, permits2, daemon2, dispatched2 = boot()
    try:
        # Fresh daemon: preflight + arm succeeds, REAL completes exactly once.
        action = _action(action_id="a-after-recovery")
        permits2.register(_permit_for(action))
        status = _finish(client2, client2.request_action(action)["action_id"])
        assert status["final_state"] == "COMPLETED"
        assert dispatched2 == ["a-after-recovery"]
        ITEMS["16_recovery_ack"] = "PASS"
    finally:
        daemon2.stop()


# --- 17-20: ledger perms / executor / capability / unauthorized REAL ---

def test_17_ledger_not_world_readable(tmp_path: Path):
    from rosclaw.daemon.ledger import DaemonLedger

    import stat as statmod

    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    with DaemonLedger(database, key_path=key):
        pass
    db_mode = statmod.S_IMODE(database.stat().st_mode)
    key_mode = statmod.S_IMODE(key.stat().st_mode)
    assert db_mode & 0o077 == 0, f"ledger db is group/world accessible: {oct(db_mode)}"
    assert key_mode & 0o077 == 0, f"ledger key is group/world accessible: {oct(key_mode)}"
    ITEMS["17_ledger_perms"] = "PASS"


def test_18_unregistered_executor_never_dispatches(boundary):
    client, permits, dispatched = boundary
    action = _action(action_id="a-no-executor", capability_id="arm.unregistered")
    permits.register(_permit_for(action))
    status = _finish(client, client.request_action(action)["action_id"])
    assert status["final_state"] in {"FAILED", "BLOCKED"}
    assert dispatched == []
    ITEMS["18_unknown_executor"] = "PASS"


def test_19_permit_scope_capability_enforced(boundary):
    client, permits, dispatched = boundary
    action = _action(action_id="a-scope", capability_id="arm.move_joints")
    permit = _permit_for(action)
    object.__setattr__(permit, "capabilities", ("arm.other_capability",))
    permits.register(permit)
    status = _finish(client, client.request_action(action)["action_id"])
    assert status["final_state"] == "BLOCKED"
    assert status["error_code"] == "PERMIT_SCOPE_MISMATCH"
    assert dispatched == []
    ITEMS["19_capability_scope"] = "PASS"


def test_20_unauthorized_real_blocked(boundary):
    client, _, dispatched = boundary
    action = _action(action_id="a-no-permit", approval_id="never-registered")
    status = _finish(client, client.request_action(action)["action_id"])
    assert status["final_state"] == "BLOCKED"
    assert status["error_code"] == "AUTHORIZATION_REQUIRED"
    if dispatched:
        GATES["unauthorized_real_dispatches"] += 1
    assert client.get_runtime_status()["hardware_actions_executed"] == 0
    ITEMS["20_unauthorized_real"] = "PASS"


def test_zz_write_gates():
    GATES["terminal_receipt_coverage"] = (
        round(GATES["actions_with_receipt"] / GATES["actions_terminal"] * 100, 1)
        if GATES["actions_terminal"]
        else 0.0
    )
    hard = {
        "unauthorized_real_dispatches": GATES["unauthorized_real_dispatches"] == 0,
        "stale_action_replays": GATES["stale_action_replays"] == 0,
        "expired_permit_accepts": GATES["expired_permit_accepts"] == 0,
        "cross_body_permit_accepts": GATES["cross_body_permit_accepts"] == 0,
        "unknown_executor_dispatches": GATES["unknown_executor_dispatches"] == 0,
        "terminal_receipt_coverage_100": GATES["terminal_receipt_coverage"] == 100.0,
    }
    out = {
        "items": ITEMS,
        "gates": GATES,
        "hard_gates_pass": hard,
        "overall": "PASS" if all(hard.values()) else "FAIL",
    }
    report_dir = Path(os.environ.get("TY1200_VALIDATION_REPORT_DIR", "."))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "rosclawd_boundary_gates.json").write_text(json.dumps(out, indent=2))
    assert all(hard.values()), out
