"""Daemon and socket tests for trusted pending proposal decisions."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.daemon.client import DaemonClient, DaemonRequestError
from rosclaw.daemon.ledger import DaemonLedger, LedgerError
from rosclaw.daemon.protocol import PeerCredentials
from rosclaw.daemon.server import RosclawDaemon
from rosclaw.daemon.service import ControlPlaneError, DaemonControlPlane
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)


def _runtime() -> Runtime:
    return Runtime(
        RuntimeConfig(
            robot_id="rh56-test",
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
    action_id: str,
    *,
    lease_ttl_ms: int = 10_000,
    renew_interval_ms: int = 3_000,
) -> ActionEnvelope:
    return ActionEnvelope(
        action_id=action_id,
        actor_id="codex-agent",
        agent_framework="codex",
        session_id=f"session-{action_id}",
        body_id="rh56-test",
        body_snapshot_hash="sha256:body",
        capability_id="rh56.finger.move",
        arguments={"finger": "index", "delta_raw": 20},
        execution_mode=ExecutionMode.REAL,
        deadline_at=datetime.now(UTC) + timedelta(minutes=2),
        authorization=AuthorizationContext(
            principal_id="forged-agent",
            approved=True,
            approval_id="forged-permit",
            scopes=["*"],
        ),
        verification_policy=VerificationPolicy(
            required_evidence=EvidenceLevel.DRIVER_CONFIRMED,
            timeout_sec=2.0,
        ),
        lease_ttl_ms=lease_ttl_ms,
        renew_interval_ms=renew_interval_ms,
    )


def _executor(action: ActionEnvelope) -> ActionExecutionResult:
    return ActionExecutionResult(
        final_state=ActionState.COMPLETED,
        evidence_level=EvidenceLevel.DRIVER_CONFIRMED,
        authorization_decision={"authorized": action.authorization.approved},
        dispatch_result={"accepted": True},
        driver_ack={"acknowledged": True},
    )


def test_broker_accepts_exact_proposal_without_exposing_challenge_or_permit(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    runtime.action_gateway.register_executor("rh56.finger.move", ExecutionMode.REAL, _executor)
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    with DaemonLedger(database, key_path=key) as ledger:
        daemon = RosclawDaemon(
            service=DaemonControlPlane(runtime=runtime, ledger=ledger),
            socket_path=tmp_path / "run" / "rosclawd.sock",
        )
        daemon.start()
        client = DaemonClient(socket_path=daemon.socket_path)
        try:
            proposed_action = _action("action-broker-accepted")
            created = client.create_operator_proposal(
                proposed_action,
                display={"title": "Move one finger", "risk_tier": "HIGH"},
                ttl_sec=60.0,
            )
            public = created["proposal"]
            assert "challenge_nonce" not in public
            assert "permit_id" not in str(created).lower()
            assert "authorized_action" not in created
            pending = client.list_pending_operator_proposals()["proposals"]
            trusted = pending[0]
            decided = client.decide_operator_proposal(
                public["request_id"],
                decision="ACCEPT",
                principal_id="operator-shift-a",
                challenge_nonce=trusted["challenge_nonce"],
                action_intent_hash=trusted["action_intent_hash"],
                channel="operator_cli",
                reason="Reviewed exact bounded action",
            )
            terminal = client.wait_for_action(public["action_id"], timeout_sec=2.0)
            proposal_status = client.get_operator_proposal(public["request_id"])
            duplicate = client.create_operator_proposal(
                proposed_action,
                display={"title": "Move one finger", "risk_tier": "HIGH"},
                ttl_sec=60.0,
            )
            events = ledger.events(entity_kind="OPERATOR_PROPOSAL", entity_id=public["request_id"])
        finally:
            daemon.stop()

    assert decided["permit_exposed"] is False
    assert decided["command_dispatched"] is True
    assert terminal["receipt"]["final_state"] == "COMPLETED", terminal
    provenance = terminal["receipt"]["authorization_decision"]["provenance"]
    assert provenance["proposal_request_id"] == public["request_id"]
    assert provenance["operator_principal"] == "operator-shift-a"
    assert provenance["decision_channel"] == "operator_cli"
    assert proposal_status["proposal"]["state"] == "TERMINAL"
    assert duplicate["proposal"]["request_id"] == public["request_id"]
    assert duplicate["decision"] == "TERMINAL"
    assert [event.event_type for event in events] == [
        "OPERATOR_PROPOSAL_CREATED",
        "OPERATOR_PROPOSAL_PRESENTED",
        "OPERATOR_PROPOSAL_ACCEPTED",
        "OPERATOR_PROPOSAL_PERMIT_ISSUED",
        "OPERATOR_PROPOSAL_SUBMITTED",
        "OPERATOR_PROPOSAL_TERMINAL",
    ]


def test_decline_and_wrong_challenge_never_issue_permit_or_dispatch(tmp_path: Path) -> None:
    runtime = _runtime()
    runtime.action_gateway.register_executor("rh56.finger.move", ExecutionMode.REAL, _executor)
    with DaemonLedger(
        tmp_path / "state" / "ledger.sqlite3",
        key_path=tmp_path / "state" / "ledger.key",
    ) as ledger:
        daemon = RosclawDaemon(
            service=DaemonControlPlane(runtime=runtime, ledger=ledger),
            socket_path=tmp_path / "run" / "rosclawd.sock",
        )
        daemon.start()
        client = DaemonClient(socket_path=daemon.socket_path)
        try:
            created = client.create_operator_proposal(
                _action("action-broker-declined"),
                display={"title": "Move one finger"},
                ttl_sec=60.0,
            )["proposal"]
            trusted = client.list_pending_operator_proposals()["proposals"][0]
            with pytest.raises(DaemonRequestError) as wrong:
                client.decide_operator_proposal(
                    created["request_id"],
                    decision="ACCEPT",
                    principal_id="operator-shift-a",
                    challenge_nonce="wrong-challenge",
                    action_intent_hash=trusted["action_intent_hash"],
                    channel="operator_cli",
                    reason="Injected decision",
                )
            declined = client.decide_operator_proposal(
                created["request_id"],
                decision="DECLINE",
                principal_id="operator-shift-a",
                challenge_nonce=trusted["challenge_nonce"],
                action_intent_hash=trusted["action_intent_hash"],
                channel="operator_cli",
                reason="Workspace is not clear",
            )
            status = client.get_runtime_status()
        finally:
            daemon.stop()

    assert wrong.value.code == "PROPOSAL_CHALLENGE_MISMATCH"
    assert declined["proposal"]["state"] == "DECLINED"
    assert declined["command_dispatched"] is False
    assert status["permits"]["registered"] == 0
    assert status["queue"]["FINISHED"] == 0
    assert status["hardware_actions_executed"] == 0


def test_agent_uid_cannot_list_or_decide_operator_proposals(tmp_path: Path) -> None:
    with DaemonLedger(
        tmp_path / "state" / "ledger.sqlite3",
        key_path=tmp_path / "state" / "ledger.key",
    ) as ledger:
        service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        service.start()
        agent = PeerCredentials(pid=111, uid=os.geteuid() + 1, gid=os.getegid())
        proposal = service.create_operator_proposal(
            _action("action-untrusted-decision"),
            display={"title": "Move one finger"},
            ttl_sec=60.0,
            peer=agent,
        )["proposal"]
        with pytest.raises(ControlPlaneError) as denied_list:
            service.list_pending_operator_proposals(agent)
        with pytest.raises(ControlPlaneError) as denied_decision:
            service.decide_operator_proposal(
                proposal["request_id"],
                decision="ACCEPT",
                principal_id="agent-self",
                challenge_nonce="unknown-to-agent",
                action_intent_hash_value=proposal["action_intent_hash"],
                channel="agent",
                reason="Agent attempted self-approval",
                peer=agent,
            )
        service.close()

    assert denied_list.value.code == "PERMISSION_DENIED"
    assert denied_decision.value.code == "PERMISSION_DENIED"


def test_agent_can_cancel_only_its_own_pending_proposal(tmp_path: Path) -> None:
    with DaemonLedger(
        tmp_path / "state" / "ledger.sqlite3",
        key_path=tmp_path / "state" / "ledger.key",
    ) as ledger:
        service = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        service.start()
        owner = PeerCredentials(pid=111, uid=os.geteuid() + 1, gid=os.getegid())
        stranger = PeerCredentials(pid=222, uid=os.geteuid() + 2, gid=os.getegid())
        proposal = service.create_operator_proposal(
            _action("action-agent-cancelled"),
            display={"title": "Move one finger"},
            ttl_sec=60.0,
            peer=owner,
        )["proposal"]
        with pytest.raises(ControlPlaneError) as denied:
            service.cancel_operator_proposal(proposal["request_id"], stranger)
        cancelled = service.cancel_operator_proposal(proposal["request_id"], owner)
        repeated = service.cancel_operator_proposal(proposal["request_id"], owner)
        events = ledger.events(entity_kind="OPERATOR_PROPOSAL", entity_id=proposal["request_id"])
        service.close()

    assert denied.value.code == "PROPOSAL_OWNERSHIP_MISMATCH"
    assert cancelled["proposal"]["state"] == "CANCELLED"
    assert cancelled["command_dispatched"] is False
    assert repeated["proposal"]["state"] == "CANCELLED"
    assert [event.event_type for event in events] == [
        "OPERATOR_PROPOSAL_CREATED",
        "OPERATOR_PROPOSAL_CANCELLED",
    ]


def test_daemon_restart_durably_invalidates_pending_proposal(tmp_path: Path) -> None:
    database = tmp_path / "state" / "ledger.sqlite3"
    key = tmp_path / "state" / "ledger.key"
    request_id = ""
    with DaemonLedger(database, key_path=key) as ledger:
        first = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        first.start()
        request_id = first.create_operator_proposal(
            _action("action-pending-before-restart"),
            display={"title": "Move one finger"},
            ttl_sec=60.0,
            peer=PeerCredentials(pid=100, uid=os.geteuid(), gid=os.getegid()),
        )["proposal"]["request_id"]
        first.close()

    with DaemonLedger(database, key_path=key) as ledger:
        second = DaemonControlPlane(runtime=_runtime(), ledger=ledger)
        events = ledger.events(entity_kind="OPERATOR_PROPOSAL", entity_id=request_id)
        second.close()

    assert events[-1].event_type == "OPERATOR_PROPOSAL_INVALIDATED"
    assert events[-1].payload["failure_code"] == "PROPOSAL_DAEMON_RESTARTED"


def test_operator_broker_supervises_origin_owned_action_lease(tmp_path: Path) -> None:
    runtime = _runtime()

    def slow_executor(action: ActionEnvelope) -> ActionExecutionResult:
        time.sleep(1.3)
        return _executor(action)

    runtime.action_gateway.register_executor("rh56.finger.move", ExecutionMode.REAL, slow_executor)
    with DaemonLedger(
        tmp_path / "state" / "ledger.sqlite3",
        key_path=tmp_path / "state" / "ledger.key",
    ) as ledger:
        daemon = RosclawDaemon(
            service=DaemonControlPlane(runtime=runtime, ledger=ledger),
            socket_path=tmp_path / "run" / "rosclawd.sock",
        )
        daemon.start()
        client = DaemonClient(socket_path=daemon.socket_path)
        try:
            public = client.create_operator_proposal(
                _action(
                    "action-broker-lease",
                    lease_ttl_ms=1_000,
                    renew_interval_ms=200,
                ),
                display={"title": "Slow bounded action"},
                ttl_sec=60.0,
            )["proposal"]
            trusted = client.list_pending_operator_proposals()["proposals"][0]
            client.decide_operator_proposal(
                public["request_id"],
                decision="ACCEPT",
                principal_id="operator-shift-a",
                challenge_nonce=trusted["challenge_nonce"],
                action_intent_hash=trusted["action_intent_hash"],
                channel="operator_cli",
                reason="Supervise until terminal receipt",
            )
            terminal = client.wait_for_action(public["action_id"], timeout_sec=3.0)
        finally:
            daemon.stop()

    assert terminal["receipt"]["final_state"] == "COMPLETED", terminal
    assert terminal["receipt"]["errors"] == []


def test_proposal_audit_failure_revokes_permit_and_prevents_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    runtime.action_gateway.register_executor("rh56.finger.move", ExecutionMode.REAL, _executor)
    with DaemonLedger(
        tmp_path / "state" / "ledger.sqlite3",
        key_path=tmp_path / "state" / "ledger.key",
    ) as ledger:
        service = DaemonControlPlane(runtime=runtime, ledger=ledger)
        service.start()
        daemon_peer = PeerCredentials(pid=100, uid=os.geteuid(), gid=os.getegid())
        created = service.create_operator_proposal(
            _action("action-proposal-ledger-failure"),
            display={"title": "Move one finger"},
            ttl_sec=60.0,
            peer=daemon_peer,
        )["proposal"]
        trusted = service.list_pending_operator_proposals(daemon_peer)["proposals"][0]
        original_append = ledger.append

        def fail_permit_transition(event_type: str, **kwargs: Any):
            if event_type == "OPERATOR_PROPOSAL_PERMIT_ISSUED":
                raise LedgerError("injected proposal audit failure")
            return original_append(event_type, **kwargs)

        monkeypatch.setattr(ledger, "append", fail_permit_transition)
        with pytest.raises(ControlPlaneError) as failed:
            service.decide_operator_proposal(
                created["request_id"],
                decision="ACCEPT",
                principal_id="operator-shift-a",
                challenge_nonce=trusted["challenge_nonce"],
                action_intent_hash_value=trusted["action_intent_hash"],
                channel="operator_cli",
                reason="Reviewed exact action",
                peer=daemon_peer,
            )
        status = service.get_runtime_status(daemon_peer)
        service.close()

    assert failed.value.code == "LEDGER_UNAVAILABLE"
    assert status["permits"]["registered"] == 1
    assert status["permits"]["revoked"] == 1
    assert status["queue"]["FINISHED"] == 0
    assert status["hardware_actions_executed"] == 0
    assert status["ledger"]["write_failed"] is True
    assert status["supervision_state"] == "ESTOPPED"
