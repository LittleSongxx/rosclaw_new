"""Recoverable publish/stage/activate/freeze/rollback coordinator."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.continual.contracts import PolicyVersion, SkillPhase
from rosclaw.continual.services.inference import InferenceService, InferenceSlotReceipt
from rosclaw.continual.services.persistence import (
    DurableEventLog,
    require_external_service_root,
)
from rosclaw.continual.stability import ContinualGateReport
from rosclaw.feedback.contracts import canonical_hash


@dataclass(frozen=True)
class WeightUpdateServiceReceipt:
    operation_id: str
    operation: str
    status: str
    inference: InferenceSlotReceipt
    event_hash: str
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.continual.weight_update_service_receipt.v1"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "inference": asdict(self.inference)}


class WeightUpdateService:
    """The sole service API allowed to mutate inference policy slots."""

    def __init__(
        self,
        root: Path,
        *,
        source_checkout: Path,
        inference: InferenceService,
    ) -> None:
        service_root = require_external_service_root(root, source_checkout) / "weight-update"
        self.log = DurableEventLog(service_root, service="weight-update")
        self.inference = inference
        pending: dict[str, tuple[str, str | None]] = {}
        for event in self.log.events:
            operation_id = str(event.payload["operation_id"])
            if event.kind == "REQUESTED":
                before = event.payload.get("inference_event_hash_before")
                pending[operation_id] = (
                    str(event.payload["operation"]),
                    str(before) if before is not None else None,
                )
            elif event.kind in {"COMPLETED", "ABORTED"}:
                pending.pop(operation_id, None)
            else:
                raise ValueError(f"unsupported weight update service event: {event.kind}")
        for operation_id, (operation, before) in pending.items():
            self._recover_pending(operation_id, operation, before)

    def publish(self, policy: PolicyVersion, *, artifact: bytes) -> WeightUpdateServiceReceipt:
        return self._run(
            "publish",
            {"policy_version_hash": policy.version_hash},
            lambda: self.inference._publish(policy, artifact),
        )

    def verify(self) -> WeightUpdateServiceReceipt:
        return self._run("verify", {}, self.inference._verify_published)

    def stage(self) -> WeightUpdateServiceReceipt:
        return self._run("stage", {}, self.inference._stage)

    def activate(
        self,
        *,
        phase: SkillPhase,
        gate_report: ContinualGateReport,
    ) -> WeightUpdateServiceReceipt:
        return self._run(
            "activate",
            {"phase": phase.value, "gate_report_hash": gate_report.report_hash},
            lambda: self.inference._activate(phase=phase, gate_report=gate_report),
        )

    def freeze(self, *, reason: str) -> WeightUpdateServiceReceipt:
        return self._run(
            "freeze",
            {"reason": reason},
            lambda: self.inference._freeze(reason),
        )

    def unfreeze(self, *, reason: str) -> WeightUpdateServiceReceipt:
        return self._run(
            "unfreeze",
            {"reason": reason},
            lambda: self.inference._unfreeze(reason),
        )

    def rollback(self, *, reason: str) -> WeightUpdateServiceReceipt:
        return self._run(
            "rollback",
            {"reason": reason},
            lambda: self.inference._rollback_active(reason),
        )

    @property
    def recovered_abort_count(self) -> int:
        return sum(
            event.kind == "ABORTED" and bool(event.payload.get("recovered"))
            for event in self.log.events
        )

    @property
    def recovered_completion_count(self) -> int:
        return sum(
            event.kind == "COMPLETED" and bool(event.payload.get("recovered"))
            for event in self.log.events
        )

    def _run(
        self,
        operation: str,
        parameters: Mapping[str, Any],
        callback: Callable[[], InferenceSlotReceipt],
    ) -> WeightUpdateServiceReceipt:
        operation_id = canonical_hash(
            {
                "operation": operation,
                "parameters": dict(parameters),
                "next_event_sequence": len(self.log.events) + 1,
                "inference_event_hash": self.inference.log.last_hash,
            }
        )
        self.log.append(
            "REQUESTED",
            {
                "operation_id": operation_id,
                "operation": operation,
                "parameters": dict(parameters),
                "inference_event_hash_before": self.inference.log.last_hash,
            },
        )
        try:
            inference_receipt = callback()
        except BaseException as exc:
            self.log.append(
                "ABORTED",
                {
                    "operation_id": operation_id,
                    "operation": operation,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "recovered": False,
                },
            )
            raise
        event = self.log.append(
            "COMPLETED",
            {
                "operation_id": operation_id,
                "operation": operation,
                "inference_event_hash": inference_receipt.event_hash,
                "recovered": False,
            },
        )
        return WeightUpdateServiceReceipt(
            operation_id=operation_id,
            operation=operation,
            status="COMPLETED",
            inference=inference_receipt,
            event_hash=event.event_hash,
        )

    def _recover_pending(
        self,
        operation_id: str,
        operation: str,
        inference_event_hash_before: str | None,
    ) -> None:
        expected = {
            "publish": {"PUBLISHED"},
            "verify": {"VERIFIED"},
            "stage": {"STAGED"},
            "activate": {"ACTIVATED", "FROZEN"},
            "freeze": {"FROZEN"},
            "unfreeze": {"UNFROZEN"},
            "rollback": {"ROLLED_BACK"},
        }
        if operation not in expected or inference_event_hash_before is None:
            self._recovery_abort(
                operation_id,
                operation,
                "legacy or unsupported incomplete weight operation was quarantined",
            )
            return
        events = self.inference.log.events
        before_indices = [
            index
            for index, event in enumerate(events)
            if event.event_hash == inference_event_hash_before
        ]
        if len(before_indices) != 1:
            self._recovery_abort(
                operation_id,
                operation,
                "weight operation references an unknown inference checkpoint",
            )
            return
        after = events[before_indices[0] + 1 :]
        if len(after) == 1 and after[0].kind in expected[operation]:
            self.log.append(
                "COMPLETED",
                {
                    "operation_id": operation_id,
                    "operation": operation,
                    "inference_event_hash": after[0].event_hash,
                    "recovered": True,
                },
            )
            return
        if after:
            self.inference._freeze(
                "ambiguous inference mutations followed an interrupted weight operation"
            )
            reason = "ambiguous incomplete weight operation froze inference"
        else:
            reason = "incomplete weight request made no inference mutation"
        self._recovery_abort(operation_id, operation, reason)

    def _recovery_abort(self, operation_id: str, operation: str, reason: str) -> None:
        self.log.append(
            "ABORTED",
            {
                "operation_id": operation_id,
                "operation": operation,
                "reason": reason,
                "recovered": True,
            },
        )


__all__ = ["WeightUpdateService", "WeightUpdateServiceReceipt"]
