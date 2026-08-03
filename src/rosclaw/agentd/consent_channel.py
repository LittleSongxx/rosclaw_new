"""Daemon consent channel (ADR-0007, K5 完整版).

agentd 的认知层授权（ApprovalRequestV2/MissionGrant）与 daemon 的物理层
consent plane（上游 #185 operator proposals）的桥：

- ``create_proposal``：Agent 侧提交 proposal（**永不**附带 nonce/permit）；
- ``decide``：operator 侧裁决（只有 daemon 服务 UID 能列出 nonce 与裁决；
  same-UID 开发环境验证协议，生产必须 UID 分离）；
- ACCEPT 后 daemon 内部签发 permit、以原 Agent UID 提交动作并监督到
  终态 Receipt——agentd 只读回 public 状态与 receipt provenance。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from rosclaw.contracts.common import ValidationError, new_id
from rosclaw.daemon.client import DaemonClient, DaemonClientError
from rosclaw.kernel.contracts import (
    ActionEnvelope,
    EvidenceLevel,
    ExecutionMode,
    VerificationPolicy,
)


class ConsentChannelError(ValidationError):
    """Proposal creation/decision/receipt failed (fail closed)."""


class DaemonConsentChannel:
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

    async def create_proposal(
        self,
        *,
        capability_id: str,
        arguments: dict[str, Any],
        display: dict[str, Any],
        execution_mode: str = "SIMULATION",
        risk_class: str = "low",
        ttl_sec: float = 60.0,
    ) -> dict[str, Any]:
        """Agent-side proposal. Returns the public view (no nonce, no permit)."""
        envelope = ActionEnvelope(
            action_id=new_id("act"),
            actor_id=self._actor_id,
            agent_framework="rosclaw-native",
            # 与上游一致：proposal 不引用预建 session（daemon 会因会话
            # 绑定语义拒绝 SESSION_ID_CONFLICT）；动作会话由 daemon 在
            # 接受后自行建立与监督。
            session_id=new_id("sess"),
            body_id=self._body_id,
            body_snapshot_hash=self._body_hash,
            capability_id=capability_id,
            arguments=arguments,
            execution_mode=ExecutionMode(execution_mode),
            risk_class=risk_class,
            deadline_at=datetime.now(UTC) + timedelta(seconds=ttl_sec),
            verification_policy=VerificationPolicy(
                required_evidence=EvidenceLevel.DRIVER_CONFIRMED,
                timeout_sec=ttl_sec,
            ),
        )
        try:
            created = await asyncio.to_thread(
                self._client.create_operator_proposal,
                envelope,
                display=display,
                ttl_sec=ttl_sec,
            )
        except DaemonClientError as exc:
            raise ConsentChannelError(
                f"operator.proposal.create failed: {exc.code}: {exc}"
            ) from exc
        proposal = created.get("proposal") or {}
        if "challenge_nonce" in proposal:
            raise ConsentChannelError(
                "daemon leaked decision challenge to the agent view — refusing"
            )
        return proposal

    async def decide(
        self,
        request_id: str,
        *,
        principal_id: str,
        accept: bool,
        channel: str = "rosclaw_console",
        reason: str = "",
        supervise_timeout_sec: float = 60.0,
    ) -> dict[str, Any]:
        """Operator-side decision. On ACCEPT the daemon issues the permit
        internally, submits as the originating UID and supervises to a
        terminal receipt."""
        try:
            pending = await asyncio.to_thread(self._client.list_pending_operator_proposals)
        except DaemonClientError as exc:
            raise ConsentChannelError(
                f"operator pending list failed (operator UID required): {exc.code}: {exc}"
            ) from exc
        trusted = next(
            (p for p in pending.get("proposals", []) if p.get("request_id") == request_id),
            None,
        )
        if trusted is None:
            raise ConsentChannelError(
                f"no pending proposal {request_id!r} (decided, expired, or "
                "invalidated by daemon restart)"
            )
        try:
            decided = await asyncio.to_thread(
                self._client.decide_operator_proposal,
                request_id,
                decision="ACCEPT" if accept else "DECLINE",
                principal_id=principal_id,
                challenge_nonce=trusted["challenge_nonce"],
                action_intent_hash=trusted["action_intent_hash"],
                channel=channel,
                reason=reason or ("reviewed bounded action" if accept else "declined by operator"),
            )
        except DaemonClientError as exc:
            raise ConsentChannelError(
                f"operator.proposal.decide failed: {exc.code}: {exc}"
            ) from exc
        if not accept:
            return decided
        if decided.get("permit_exposed"):
            raise ConsentChannelError(
                "daemon exposed permit material in the decision result — refusing"
            )
        # Supervise to terminal (the daemon renews the lease; SIM finishes fast).
        action_id = trusted.get("action_id")
        if action_id:
            try:
                await asyncio.to_thread(
                    self._client.wait_for_action,
                    action_id,
                    timeout_sec=supervise_timeout_sec,
                )
            except DaemonClientError as exc:
                raise ConsentChannelError(
                    f"accepted action did not terminate: {exc.code}: {exc}"
                ) from exc
        return decided

    async def proposal(self, request_id: str) -> dict[str, Any]:
        try:
            result = await asyncio.to_thread(self._client.get_operator_proposal, request_id)
        except DaemonClientError as exc:
            raise ConsentChannelError(
                f"operator.proposal.status failed: {exc.code}: {exc}"
            ) from exc
        return result.get("proposal") or {}

    async def action_receipt(self, action_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._client.get_execution_receipt, action_id)
        except DaemonClientError as exc:
            raise ConsentChannelError(f"action.receipt failed: {exc.code}: {exc}") from exc
