"""Daemon action channel (K3 完整版, ADR-0001).

The agentd's ONLY physical channel: the sanctioned northbound
``DaemonClient`` over the authenticated Unix socket. agentd builds an
ActionEnvelope (SIMULATION in P0), submits it as a *request*, waits for a
terminal scheduler state, and reads back the execution receipt. A receipt
is verified against the requested action — a submitted command is never
reported as a completed task (总纲 §12.3).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.daemon.client import DaemonClient, DaemonClientError
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    AuthorizationContext,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)


class ActionChannelError(ValidationError):
    """Dispatch or receipt verification failed (fail closed)."""


@dataclass(frozen=True)
class ActionOutcome:
    action_id: str
    state: str
    receipt: dict[str, Any]
    trust_level: str
    verified: bool


class DaemonActionChannel:
    def __init__(
        self,
        client: DaemonClient,
        *,
        actor_id: str,
        body_id: str,
        body_hash: str,
    ) -> None:
        self._client = client
        self._actor_id = actor_id
        self._body_id = body_id
        self._body_hash = body_hash
        self._session_id: str | None = None

    async def _ensure_session(self, capability_id: str) -> str:
        if self._session_id is not None:
            return self._session_id
        self._session_id = new_id("sess")
        await asyncio.to_thread(
            self._client.create_session,
            session_id=self._session_id,
            actor_id=self._actor_id,
            agent_framework="rosclaw-native",
            body_scope=[self._body_id],
            # daemon 拒绝空 scope 与 `*`：显式能力清单。
            capability_scope=[capability_id],
            ttl_ms=30_000,
        )
        return self._session_id

    async def request_sim_action(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        grant_id: str,
        timeout_sec: float = 30.0,
    ) -> ActionOutcome:
        """Submit a SIMULATION action and verify its receipt."""
        try:
            session_id = await self._ensure_session(capability_id)
        except DaemonClientError as exc:
            raise ActionChannelError(
                f"daemon session failed (daemon offline?): {exc.code}: {exc}"
            ) from exc
        envelope = ActionEnvelope(
            action_id=new_id("act"),
            actor_id=self._actor_id,
            agent_framework="rosclaw-native",
            session_id=session_id,
            body_id=self._body_id,
            body_snapshot_hash=self._body_hash,
            capability_id=capability_id,
            arguments=arguments,
            execution_mode=ExecutionMode.SIMULATION,
            deadline_at=datetime.now(UTC) + timedelta(seconds=timeout_sec),
            authorization=AuthorizationContext(
                principal_id="user:local:1000",
                approved=True,
                approval_id=grant_id,
                scopes=["simulation"],
            ),
            verification_policy=VerificationPolicy(
                required_evidence=EvidenceLevel.TASK_VERIFIED,
                timeout_sec=timeout_sec,
            ),
        )
        try:
            submitted = await asyncio.to_thread(self._client.request_action, envelope)
        except DaemonClientError as exc:
            raise ActionChannelError(f"daemon rejected action request: {exc.code}: {exc}") from exc
        action_id = submitted.get("action_id", envelope.action_id)
        try:
            status = await asyncio.to_thread(
                self._client.wait_for_action, action_id, timeout_sec=timeout_sec
            )
        except DaemonClientError as exc:
            raise ActionChannelError(f"action did not finish: {exc.code}: {exc}") from exc
        receipt = await asyncio.to_thread(self._client.get_execution_receipt, action_id)
        return self._verify_outcome(action_id, status, receipt, envelope)

    def _verify_outcome(
        self,
        action_id: str,
        status: dict[str, Any],
        receipt: dict[str, Any],
        envelope: ActionEnvelope,
    ) -> ActionOutcome:
        """A submitted command is not a completed task — check the receipt."""
        state = str(status.get("state", "UNKNOWN"))
        # daemon 返回 {"action_id":..., "receipt": {...}} 信封。
        inner = receipt.get("receipt") if isinstance(receipt.get("receipt"), dict) else receipt
        receipt_action_id = inner.get("action_id")
        if receipt_action_id not in (None, action_id):
            raise ActionChannelError(
                f"receipt action {receipt_action_id!r} "
                f"!= requested {action_id!r} — not reporting as our action"
            )
        trust = str(inner.get("trust_level", "UNKNOWN"))
        if trust == "SYNTHETIC":
            raise ActionChannelError("receipt is FIXTURE/SYNTHETIC — never usable as real evidence")
        verified = state in ("FINISHED",) and trust == "SIMULATED"
        return ActionOutcome(
            action_id=action_id,
            state=state,
            receipt=receipt,
            trust_level=trust,
            verified=verified,
        )
