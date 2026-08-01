"""Crash-recoverable control plane for bounded Dream campaigns.

The scheduler owns intent, leases, accounting and terminal receipts.  It does
not import the inference or weight-update services and therefore has no policy
activation path.  Worker leases are capability tokens; only their hashes are
written to the append-only journal.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import math
import os
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw.continual.services.persistence import (
    DurableEventLog,
    DurableServiceEvent,
    require_external_service_root,
)
from rosclaw.dream.contracts import DreamBudget, DreamCampaign, DreamType
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.contracts import SkillGrowthSpec

_SHA256_PREFIX = "sha256:"
_TERMINAL_STATES: frozenset[DreamCampaignState]


def _require_hash(label: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith(_SHA256_PREFIX):
        raise ValueError(f"{label} must be a sha256: content hash")
    try:
        bytes.fromhex(value.removeprefix(_SHA256_PREFIX))
    except ValueError as exc:
        raise ValueError(f"{label} must be a sha256: content hash") from exc


def _token_hash(token: str) -> str:
    return _SHA256_PREFIX + hashlib.sha256(token.encode("utf-8")).hexdigest()


def _non_negative_number(label: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return normalized


def _non_negative_integer(label: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


class DreamCampaignState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


_TERMINAL_STATES = frozenset(
    {
        DreamCampaignState.COMPLETED,
        DreamCampaignState.CANCELLED,
        DreamCampaignState.FAILED,
        DreamCampaignState.BUDGET_EXHAUSTED,
    }
)


class DreamBudgetExceededError(RuntimeError):
    """Raised after the scheduler durably rejects an over-budget request."""


@dataclass(frozen=True)
class DreamPlanRequest:
    """Caller intent consumed by the pure, non-activating planner."""

    body_hash: str
    parent_policy_hash: str
    trigger_kind: str
    trigger_evidence_hashes: tuple[str, ...]
    objectives: tuple[str, ...]
    constraint_hashes: tuple[str, ...]
    practice_snapshot_hashes: tuple[str, ...]
    collective_capsule_hashes: tuple[str, ...]
    historical_anchor_hashes: tuple[str, ...]
    boundary_suite_hashes: tuple[str, ...]
    private_holdout_commitment: str
    dream_types: tuple[DreamType, ...]
    learner_ids: tuple[str, ...]
    budget: DreamBudget
    schema_version: str = "rosclaw.dream.plan_request.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("parent_policy_hash", self.parent_policy_hash),
            ("private_holdout_commitment", self.private_holdout_commitment),
        ):
            _require_hash(label, value)
        for label in (
            "trigger_evidence_hashes",
            "constraint_hashes",
            "historical_anchor_hashes",
            "boundary_suite_hashes",
        ):
            values = tuple(getattr(self, label))
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{label} must be non-empty and unique")
            for value in values:
                _require_hash(label, value)
            object.__setattr__(self, label, values)
        for label in ("practice_snapshot_hashes", "collective_capsule_hashes"):
            values = tuple(getattr(self, label))
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
            for value in values:
                _require_hash(label, value)
            object.__setattr__(self, label, values)
        for label in ("objectives", "learner_ids"):
            values = tuple(getattr(self, label))
            if (
                not values
                or len(values) != len(set(values))
                or any(not isinstance(item, str) or not item.strip() for item in values)
            ):
                raise ValueError(f"{label} must contain unique non-empty values")
            object.__setattr__(self, label, values)
        if not isinstance(self.trigger_kind, str) or not self.trigger_kind.strip():
            raise ValueError("trigger_kind must not be empty")
        dream_types = tuple(self.dream_types)
        if (
            not dream_types
            or len(dream_types) != len(set(dream_types))
            or any(not isinstance(item, DreamType) for item in dream_types)
        ):
            raise ValueError("dream_types must contain unique recognized DreamType values")
        object.__setattr__(self, "dream_types", dream_types)
        if not isinstance(self.budget, DreamBudget):
            raise ValueError("budget must be a DreamBudget")

    @property
    def request_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "parent_policy_hash": self.parent_policy_hash,
            "trigger_kind": self.trigger_kind,
            "trigger_evidence_hashes": list(self.trigger_evidence_hashes),
            "objectives": list(self.objectives),
            "constraint_hashes": list(self.constraint_hashes),
            "practice_snapshot_hashes": list(self.practice_snapshot_hashes),
            "collective_capsule_hashes": list(self.collective_capsule_hashes),
            "historical_anchor_hashes": list(self.historical_anchor_hashes),
            "boundary_suite_hashes": list(self.boundary_suite_hashes),
            "private_holdout_commitment": self.private_holdout_commitment,
            "dream_types": [item.value for item in self.dream_types],
            "learner_ids": list(self.learner_ids),
            "budget": self.budget.to_dict(),
        }


@dataclass(frozen=True)
class DreamPlanReceipt:
    campaign: DreamCampaign
    growth_spec_hash: str
    planner_id: str
    activation_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.dream.plan_receipt.v1"

    @property
    def receipt_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign.campaign_hash,
            "campaign": self.campaign.to_dict(),
            "growth_spec_hash": self.growth_spec_hash,
            "planner_id": self.planner_id,
            "activation_authorized": self.activation_authorized,
            "hardware_authorized": self.hardware_authorized,
        }


class DreamPlanner:
    """Validate growth intent and construct one immutable campaign."""

    def __init__(self, planner_id: str = "dreamforge.planner.v1") -> None:
        if not planner_id.strip():
            raise ValueError("planner_id must not be empty")
        self.planner_id = planner_id

    def plan(self, spec: SkillGrowthSpec, request: DreamPlanRequest) -> DreamPlanReceipt:
        if not isinstance(spec, SkillGrowthSpec):
            raise ValueError("spec must be a SkillGrowthSpec")
        if not isinstance(request, DreamPlanRequest):
            raise ValueError("request must be a DreamPlanRequest")
        if request.body_hash not in spec.body_hashes:
            raise ValueError("requested body is outside the growth specification")
        dream_type_names = {dream_type.value for dream_type in request.dream_types}
        if not dream_type_names.issubset(set(spec.allowed_dream_types)):
            raise ValueError(
                "request contains a dream type not allowed by the growth specification"
            )
        if not set(request.learner_ids).issubset(set(spec.allowed_learner_ids)):
            raise ValueError("request contains a learner not allowed by the growth specification")
        if not set(spec.historical_anchor_hashes).issubset(request.historical_anchor_hashes):
            raise ValueError("request omits a required historical anchor")
        if spec.boundary_suite_hash not in request.boundary_suite_hashes:
            raise ValueError("request omits the pinned boundary suite")
        if DreamType.SOCIAL in request.dream_types and not spec.collective_source_ids:
            raise ValueError("social dreaming is not enabled by the growth specification")
        campaign = DreamCampaign(
            skill_growth_spec_hash=spec.spec_hash,
            body_hash=request.body_hash,
            parent_policy_hash=request.parent_policy_hash,
            trigger_kind=request.trigger_kind,
            trigger_evidence_hashes=request.trigger_evidence_hashes,
            objectives=request.objectives,
            constraint_hashes=request.constraint_hashes,
            practice_snapshot_hashes=request.practice_snapshot_hashes,
            collective_capsule_hashes=request.collective_capsule_hashes,
            historical_anchor_hashes=request.historical_anchor_hashes,
            boundary_suite_hashes=request.boundary_suite_hashes,
            private_holdout_commitment=request.private_holdout_commitment,
            dream_types=request.dream_types,
            learner_ids=request.learner_ids,
            budget=request.budget,
        )
        return DreamPlanReceipt(
            campaign=campaign,
            growth_spec_hash=spec.spec_hash,
            planner_id=self.planner_id,
        )


@dataclass(frozen=True)
class DreamLease:
    campaign_hash: str
    lease_id: str
    worker_id: str
    expires_at_ns: int
    lease_token: str = field(repr=False)
    activation_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.dream.worker_lease.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign_hash,
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "expires_at_ns": self.expires_at_ns,
            "activation_authorized": self.activation_authorized,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class DreamBudgetUsage:
    gpu_seconds: float = 0.0
    cpu_rollouts: int = 0
    candidates: int = 0

    def __post_init__(self) -> None:
        _non_negative_number("gpu_seconds", self.gpu_seconds)
        _non_negative_integer("cpu_rollouts", self.cpu_rollouts)
        _non_negative_integer("candidates", self.candidates)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "gpu_seconds": self.gpu_seconds,
            "cpu_rollouts": self.cpu_rollouts,
            "candidates": self.candidates,
        }


@dataclass(frozen=True)
class DreamCampaignStatus:
    campaign_hash: str
    state: DreamCampaignState
    submitted_at_ns: int
    elapsed_wall_seconds: float
    usage: DreamBudgetUsage
    budget: DreamBudget
    worker_id: str | None
    lease_id: str | None
    lease_expires_at_ns: int | None
    result_manifest_hash: str | None
    candidate_artifact_hashes: tuple[str, ...]
    reason: str
    journal_event_hash: str
    activation_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.dream.campaign_status.v1"

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_hash": self.campaign_hash,
            "state": self.state.value,
            "terminal": self.terminal,
            "submitted_at_ns": self.submitted_at_ns,
            "elapsed_wall_seconds": self.elapsed_wall_seconds,
            "usage": self.usage.to_dict(),
            "budget": self.budget.to_dict(),
            "worker_id": self.worker_id,
            "lease_id": self.lease_id,
            "lease_expires_at_ns": self.lease_expires_at_ns,
            "result_manifest_hash": self.result_manifest_hash,
            "candidate_artifact_hashes": list(self.candidate_artifact_hashes),
            "reason": self.reason,
            "journal_event_hash": self.journal_event_hash,
            "activation_authorized": self.activation_authorized,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass
class _CampaignRuntime:
    campaign: DreamCampaign
    state: DreamCampaignState
    submitted_at_ns: int
    gpu_seconds: float = 0.0
    cpu_rollouts: int = 0
    candidates: int = 0
    worker_id: str | None = None
    lease_id: str | None = None
    lease_token_hash: str | None = None
    lease_expires_at_ns: int | None = None
    result_manifest_hash: str | None = None
    candidate_artifact_hashes: tuple[str, ...] = ()
    reason: str = ""
    terminal_at_ns: int | None = None


class DreamScheduler:
    """Single-writer, crash-recoverable campaign scheduler."""

    def __init__(
        self,
        root: Path,
        *,
        source_checkout: Path,
        clock_ns: Callable[[], int] = time.time_ns,
        max_lease_seconds: float = 300.0,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.clock_ns = clock_ns
        if isinstance(max_lease_seconds, bool):
            raise ValueError("max_lease_seconds must be finite and positive")
        self.max_lease_seconds = float(max_lease_seconds)
        if not math.isfinite(self.max_lease_seconds) or self.max_lease_seconds <= 0.0:
            raise ValueError("max_lease_seconds must be finite and positive")
        self.token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self.root = require_external_service_root(root, source_checkout) / "dream"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_descriptor = os.open(self.root / "scheduler.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._lock_descriptor)
            self._lock_descriptor = -1
            raise RuntimeError("another Dream scheduler owns the state root") from exc
        self.log = DurableEventLog(
            self.root / "journal", service="dream-scheduler", clock_ns=clock_ns
        )
        self._campaigns: dict[str, _CampaignRuntime] = {}
        self._issued_token_hashes: set[str] = set()
        try:
            for event in self.log.events:
                self._apply(event)
            self._reap_limits()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_lock_descriptor", -1) >= 0:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = -1

    def __enter__(self) -> DreamScheduler:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def submit(self, campaign: DreamCampaign) -> DreamCampaignStatus:
        if not isinstance(campaign, DreamCampaign):
            raise ValueError("campaign must be a DreamCampaign")
        self._reap_limits()
        if campaign.campaign_hash in self._campaigns:
            raise ValueError("campaign is already submitted")
        event = self.log.append("CAMPAIGN_SUBMITTED", {"campaign": campaign.to_dict()})
        self._apply(event)
        return self.status(campaign.campaign_hash)

    def acquire(
        self,
        *,
        worker_id: str,
        lease_seconds: float,
        campaign_hash: str | None = None,
    ) -> DreamLease:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if isinstance(lease_seconds, bool):
            raise ValueError("lease_seconds exceeds the scheduler lease bound")
        lease_seconds = float(lease_seconds)
        if not math.isfinite(lease_seconds) or not 0.0 < lease_seconds <= self.max_lease_seconds:
            raise ValueError("lease_seconds exceeds the scheduler lease bound")
        self._reap_limits()
        runtime = self._select_queued(campaign_hash)
        now = int(self.clock_ns())
        wall_deadline = runtime.submitted_at_ns + int(
            runtime.campaign.budget.max_wall_seconds * 1e9
        )
        expires_at_ns = min(now + int(lease_seconds * 1e9), wall_deadline)
        if expires_at_ns <= now:
            self._exhaust(runtime, "campaign wall-clock budget exhausted before lease acquisition")
            raise DreamBudgetExceededError("campaign wall-clock budget exhausted")
        token = self.token_factory()
        if not token or len(token) < 16:
            raise RuntimeError("token factory returned an unsafe lease token")
        token_hash = _token_hash(token)
        if token_hash in self._issued_token_hashes:
            raise RuntimeError("token factory reused a Dream worker capability")
        lease_id = canonical_hash(
            {
                "campaign_hash": runtime.campaign.campaign_hash,
                "worker_id": worker_id,
                "lease_token_hash": token_hash,
                "next_event_sequence": len(self.log.events) + 1,
            }
        )
        event = self.log.append(
            "LEASE_ACQUIRED",
            {
                "campaign_hash": runtime.campaign.campaign_hash,
                "worker_id": worker_id,
                "lease_id": lease_id,
                "lease_token_hash": token_hash,
                "expires_at_ns": expires_at_ns,
            },
        )
        self._apply(event)
        return DreamLease(
            campaign_hash=runtime.campaign.campaign_hash,
            lease_id=lease_id,
            worker_id=worker_id,
            expires_at_ns=expires_at_ns,
            lease_token=token,
        )

    def heartbeat(
        self,
        campaign_hash: str,
        *,
        lease_token: str,
        extend_seconds: float,
    ) -> DreamCampaignStatus:
        if isinstance(extend_seconds, bool):
            raise ValueError("extend_seconds exceeds the scheduler lease bound")
        extend_seconds = float(extend_seconds)
        if not math.isfinite(extend_seconds) or not 0.0 < extend_seconds <= self.max_lease_seconds:
            raise ValueError("extend_seconds exceeds the scheduler lease bound")
        self._reap_limits()
        runtime = self._require_running(campaign_hash, lease_token)
        now = int(self.clock_ns())
        wall_deadline = runtime.submitted_at_ns + int(
            runtime.campaign.budget.max_wall_seconds * 1e9
        )
        current_expiry = runtime.lease_expires_at_ns or now
        expires_at_ns = min(max(current_expiry, now + int(extend_seconds * 1e9)), wall_deadline)
        if expires_at_ns <= now:
            self._exhaust(runtime, "campaign wall-clock budget exhausted at heartbeat")
            raise DreamBudgetExceededError("campaign wall-clock budget exhausted")
        event = self.log.append(
            "LEASE_RENEWED",
            {
                "campaign_hash": campaign_hash,
                "lease_id": runtime.lease_id,
                "expires_at_ns": expires_at_ns,
            },
        )
        self._apply(event)
        return self.status(campaign_hash)

    def record_usage(
        self,
        campaign_hash: str,
        *,
        lease_token: str,
        gpu_seconds: float = 0.0,
        cpu_rollouts: int = 0,
        candidates: int = 0,
    ) -> DreamCampaignStatus:
        gpu_seconds = _non_negative_number("gpu_seconds", gpu_seconds)
        cpu_rollouts = _non_negative_integer("cpu_rollouts", cpu_rollouts)
        candidates = _non_negative_integer("candidates", candidates)
        if gpu_seconds == 0.0 and cpu_rollouts == 0 and candidates == 0:
            raise ValueError("usage delta must consume at least one resource")
        self._reap_limits()
        runtime = self._require_running(campaign_hash, lease_token)
        projected_gpu = runtime.gpu_seconds + gpu_seconds
        projected_rollouts = runtime.cpu_rollouts + cpu_rollouts
        projected_candidates = runtime.candidates + candidates
        violations = []
        if projected_gpu > runtime.campaign.budget.max_gpu_seconds:
            violations.append("gpu_seconds")
        if projected_rollouts > runtime.campaign.budget.max_cpu_rollouts:
            violations.append("cpu_rollouts")
        if projected_candidates > runtime.campaign.budget.max_candidates:
            violations.append("candidates")
        if violations:
            self._exhaust(runtime, "budget request rejected: " + ",".join(violations))
            raise DreamBudgetExceededError("dream campaign resource budget exhausted")
        event = self.log.append(
            "USAGE_RECORDED",
            {
                "campaign_hash": campaign_hash,
                "lease_id": runtime.lease_id,
                "delta": {
                    "gpu_seconds": gpu_seconds,
                    "cpu_rollouts": cpu_rollouts,
                    "candidates": candidates,
                },
                "cumulative": {
                    "gpu_seconds": projected_gpu,
                    "cpu_rollouts": projected_rollouts,
                    "candidates": projected_candidates,
                },
            },
        )
        self._apply(event)
        return self.status(campaign_hash)

    def pause(self, campaign_hash: str, *, lease_token: str, reason: str) -> DreamCampaignStatus:
        if not reason.strip():
            raise ValueError("pause requires a reason")
        self._reap_limits()
        runtime = self._require_running(campaign_hash, lease_token)
        event = self.log.append(
            "CAMPAIGN_PAUSED",
            {
                "campaign_hash": campaign_hash,
                "lease_id": runtime.lease_id,
                "reason": reason,
                "automatic": False,
            },
        )
        self._apply(event)
        return self.status(campaign_hash)

    def resume(self, campaign_hash: str, *, reason: str) -> DreamCampaignStatus:
        if not reason.strip():
            raise ValueError("resume requires a reason")
        self._reap_limits()
        runtime = self._runtime(campaign_hash)
        if runtime.state is not DreamCampaignState.PAUSED:
            raise RuntimeError("only a paused campaign can be resumed")
        event = self.log.append(
            "CAMPAIGN_RESUMED",
            {"campaign_hash": campaign_hash, "reason": reason},
        )
        self._apply(event)
        return self.status(campaign_hash)

    def cancel(self, campaign_hash: str, *, reason: str) -> DreamCampaignStatus:
        if not reason.strip():
            raise ValueError("cancel requires a reason")
        self._reap_limits()
        runtime = self._runtime(campaign_hash)
        if runtime.state in _TERMINAL_STATES:
            raise RuntimeError("terminal campaign cannot be cancelled")
        event = self.log.append(
            "CAMPAIGN_CANCELLED",
            {"campaign_hash": campaign_hash, "reason": reason},
        )
        self._apply(event)
        return self.status(campaign_hash)

    def fail(self, campaign_hash: str, *, lease_token: str, reason: str) -> DreamCampaignStatus:
        if not reason.strip():
            raise ValueError("failure requires a reason")
        self._reap_limits()
        runtime = self._require_running(campaign_hash, lease_token)
        event = self.log.append(
            "CAMPAIGN_FAILED",
            {
                "campaign_hash": campaign_hash,
                "lease_id": runtime.lease_id,
                "reason": reason,
            },
        )
        self._apply(event)
        return self.status(campaign_hash)

    def complete(
        self,
        campaign_hash: str,
        *,
        lease_token: str,
        result_manifest_hash: str,
        candidate_artifact_hashes: tuple[str, ...] = (),
    ) -> DreamCampaignStatus:
        _require_hash("result_manifest_hash", result_manifest_hash)
        candidates = tuple(candidate_artifact_hashes)
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate_artifact_hashes must be unique")
        for value in candidates:
            _require_hash("candidate_artifact_hashes", value)
        self._reap_limits()
        runtime = self._require_running(campaign_hash, lease_token)
        if len(candidates) > runtime.candidates:
            raise ValueError("candidate artifacts exceed the durably accounted candidate budget")
        event = self.log.append(
            "CAMPAIGN_COMPLETED",
            {
                "campaign_hash": campaign_hash,
                "lease_id": runtime.lease_id,
                "result_manifest_hash": result_manifest_hash,
                "candidate_artifact_hashes": list(candidates),
            },
        )
        self._apply(event)
        return self.status(campaign_hash)

    def status(self, campaign_hash: str) -> DreamCampaignStatus:
        self._reap_limits()
        return self._status(self._runtime(campaign_hash))

    def list_statuses(self) -> tuple[DreamCampaignStatus, ...]:
        self._reap_limits()
        return tuple(self._status(runtime) for runtime in self._campaigns.values())

    def _status(self, runtime: _CampaignRuntime) -> DreamCampaignStatus:
        end_ns = runtime.terminal_at_ns or int(self.clock_ns())
        elapsed = max(0.0, (end_ns - runtime.submitted_at_ns) / 1e9)
        return DreamCampaignStatus(
            campaign_hash=runtime.campaign.campaign_hash,
            state=runtime.state,
            submitted_at_ns=runtime.submitted_at_ns,
            elapsed_wall_seconds=elapsed,
            usage=DreamBudgetUsage(
                gpu_seconds=runtime.gpu_seconds,
                cpu_rollouts=runtime.cpu_rollouts,
                candidates=runtime.candidates,
            ),
            budget=runtime.campaign.budget,
            worker_id=runtime.worker_id,
            lease_id=runtime.lease_id,
            lease_expires_at_ns=runtime.lease_expires_at_ns,
            result_manifest_hash=runtime.result_manifest_hash,
            candidate_artifact_hashes=runtime.candidate_artifact_hashes,
            reason=runtime.reason,
            journal_event_hash=self.log.last_hash,
        )

    def _runtime(self, campaign_hash: str) -> _CampaignRuntime:
        try:
            return self._campaigns[campaign_hash]
        except KeyError as exc:
            raise KeyError("unknown dream campaign") from exc

    def _select_queued(self, campaign_hash: str | None) -> _CampaignRuntime:
        if campaign_hash is not None:
            runtime = self._runtime(campaign_hash)
            if runtime.state is not DreamCampaignState.QUEUED:
                raise RuntimeError("requested campaign is not queued")
            return runtime
        for runtime in self._campaigns.values():
            if runtime.state is DreamCampaignState.QUEUED:
                return runtime
        raise RuntimeError("no queued dream campaign is available")

    def _require_running(self, campaign_hash: str, lease_token: str) -> _CampaignRuntime:
        runtime = self._runtime(campaign_hash)
        if runtime.state is not DreamCampaignState.RUNNING:
            raise RuntimeError("campaign has no active worker lease")
        if runtime.lease_token_hash is None or not hmac.compare_digest(
            runtime.lease_token_hash,
            _token_hash(lease_token),
        ):
            raise PermissionError("dream worker lease token does not match")
        if (
            runtime.lease_expires_at_ns is None
            or int(self.clock_ns()) >= runtime.lease_expires_at_ns
        ):
            self._pause_expired(runtime)
            raise PermissionError("dream worker lease has expired")
        return runtime

    def _reap_limits(self) -> None:
        self._ensure_open()
        now = int(self.clock_ns())
        for runtime in tuple(self._campaigns.values()):
            if runtime.state in _TERMINAL_STATES:
                continue
            deadline = runtime.submitted_at_ns + int(runtime.campaign.budget.max_wall_seconds * 1e9)
            if now >= deadline:
                self._exhaust(runtime, "campaign wall-clock budget exhausted")
                continue
            if (
                runtime.state is DreamCampaignState.RUNNING
                and runtime.lease_expires_at_ns is not None
                and now >= runtime.lease_expires_at_ns
            ):
                self._pause_expired(runtime)

    def _pause_expired(self, runtime: _CampaignRuntime) -> None:
        event = self.log.append(
            "CAMPAIGN_PAUSED",
            {
                "campaign_hash": runtime.campaign.campaign_hash,
                "lease_id": runtime.lease_id,
                "reason": "worker lease expired; stale work was not accepted",
                "automatic": True,
            },
        )
        self._apply(event)

    def _exhaust(self, runtime: _CampaignRuntime, reason: str) -> None:
        event = self.log.append(
            "BUDGET_EXHAUSTED",
            {
                "campaign_hash": runtime.campaign.campaign_hash,
                "lease_id": runtime.lease_id,
                "reason": reason,
            },
        )
        self._apply(event)

    def _apply(self, event: DurableServiceEvent) -> None:
        payload = dict(event.payload)
        kind = event.kind
        if kind == "CAMPAIGN_SUBMITTED":
            from rosclaw.dream.serde import dream_campaign_from_dict

            campaign_value = payload.get("campaign")
            if not isinstance(campaign_value, Mapping):
                raise ValueError("campaign journal event is malformed")
            campaign = dream_campaign_from_dict(campaign_value)
            if campaign.campaign_hash in self._campaigns:
                raise ValueError("journal contains a duplicate dream campaign")
            self._campaigns[campaign.campaign_hash] = _CampaignRuntime(
                campaign=campaign,
                state=DreamCampaignState.QUEUED,
                submitted_at_ns=event.timestamp_ns,
                reason="campaign accepted into bounded SIM-only queue",
            )
            return
        campaign_hash = str(payload.get("campaign_hash", ""))
        _require_hash("campaign_hash", campaign_hash)
        runtime = self._runtime(campaign_hash)
        if kind == "LEASE_ACQUIRED":
            if runtime.state is not DreamCampaignState.QUEUED:
                raise ValueError("journal acquires a lease for a non-queued campaign")
            worker_id = str(payload["worker_id"])
            lease_id = str(payload["lease_id"])
            token_hash = str(payload["lease_token_hash"])
            expires_at_ns = _non_negative_integer("expires_at_ns", payload["expires_at_ns"])
            _require_hash("lease_id", lease_id)
            _require_hash("lease_token_hash", token_hash)
            if not worker_id.strip() or expires_at_ns <= event.timestamp_ns:
                raise ValueError("journal contains an invalid worker lease")
            wall_deadline = runtime.submitted_at_ns + int(
                runtime.campaign.budget.max_wall_seconds * 1e9
            )
            if event.timestamp_ns >= wall_deadline or expires_at_ns > wall_deadline:
                raise ValueError("journal worker lease exceeds the campaign wall-clock budget")
            if token_hash in self._issued_token_hashes:
                raise ValueError("journal reuses a worker lease capability token")
            self._issued_token_hashes.add(token_hash)
            runtime.state = DreamCampaignState.RUNNING
            runtime.worker_id = worker_id
            runtime.lease_id = lease_id
            runtime.lease_token_hash = token_hash
            runtime.lease_expires_at_ns = expires_at_ns
            runtime.reason = "worker lease acquired"
            return
        if kind == "LEASE_RENEWED":
            self._require_journal_lease(runtime, payload)
            expires_at_ns = _non_negative_integer("expires_at_ns", payload["expires_at_ns"])
            if expires_at_ns <= event.timestamp_ns:
                raise ValueError("journal contains an already expired lease renewal")
            wall_deadline = runtime.submitted_at_ns + int(
                runtime.campaign.budget.max_wall_seconds * 1e9
            )
            if expires_at_ns > wall_deadline:
                raise ValueError("journal lease renewal exceeds the campaign wall-clock budget")
            if (
                runtime.lease_expires_at_ns is not None
                and expires_at_ns < runtime.lease_expires_at_ns
            ):
                raise ValueError("journal lease renewal shortens an existing lease")
            runtime.lease_expires_at_ns = expires_at_ns
            runtime.reason = "worker lease renewed"
            return
        if kind == "USAGE_RECORDED":
            self._require_journal_lease(runtime, payload)
            delta = payload.get("delta")
            cumulative = payload.get("cumulative")
            if not isinstance(delta, Mapping) or not isinstance(cumulative, Mapping):
                raise ValueError("journal usage event is malformed")
            delta_gpu = _non_negative_number("delta gpu_seconds", delta["gpu_seconds"])
            delta_rollouts = _non_negative_integer("delta cpu_rollouts", delta["cpu_rollouts"])
            delta_candidates = _non_negative_integer("delta candidates", delta["candidates"])
            expected_gpu = runtime.gpu_seconds + delta_gpu
            expected_rollouts = runtime.cpu_rollouts + delta_rollouts
            expected_candidates = runtime.candidates + delta_candidates
            cumulative_gpu = _non_negative_number(
                "cumulative gpu_seconds", cumulative["gpu_seconds"]
            )
            cumulative_rollouts = _non_negative_integer(
                "cumulative cpu_rollouts", cumulative["cpu_rollouts"]
            )
            cumulative_candidates = _non_negative_integer(
                "cumulative candidates", cumulative["candidates"]
            )
            if not math.isclose(cumulative_gpu, expected_gpu):
                raise ValueError("journal cumulative GPU usage diverges")
            if cumulative_rollouts != expected_rollouts:
                raise ValueError("journal cumulative rollout usage diverges")
            if cumulative_candidates != expected_candidates:
                raise ValueError("journal cumulative candidate usage diverges")
            if (
                expected_gpu > runtime.campaign.budget.max_gpu_seconds
                or expected_rollouts > runtime.campaign.budget.max_cpu_rollouts
                or expected_candidates > runtime.campaign.budget.max_candidates
            ):
                raise ValueError("journal usage exceeds the immutable campaign budget")
            runtime.gpu_seconds = expected_gpu
            runtime.cpu_rollouts = expected_rollouts
            runtime.candidates = expected_candidates
            runtime.reason = "worker usage recorded"
            return
        if kind == "CAMPAIGN_PAUSED":
            if runtime.state is not DreamCampaignState.RUNNING:
                raise ValueError("journal pauses a campaign without an active lease")
            self._require_journal_lease(runtime, payload)
            runtime.state = DreamCampaignState.PAUSED
            runtime.reason = str(payload["reason"])
            self._clear_lease(runtime)
            return
        if kind == "CAMPAIGN_RESUMED":
            if runtime.state is not DreamCampaignState.PAUSED:
                raise ValueError("journal resumes a campaign that is not paused")
            runtime.state = DreamCampaignState.QUEUED
            runtime.reason = str(payload["reason"])
            return
        if kind == "CAMPAIGN_CANCELLED":
            if runtime.state in _TERMINAL_STATES:
                raise ValueError("journal cancels a terminal campaign")
            runtime.state = DreamCampaignState.CANCELLED
            runtime.reason = str(payload["reason"])
            runtime.terminal_at_ns = event.timestamp_ns
            self._clear_lease(runtime)
            return
        if kind == "CAMPAIGN_FAILED":
            if runtime.state is not DreamCampaignState.RUNNING:
                raise ValueError("journal fails a campaign without an active lease")
            self._require_journal_lease(runtime, payload)
            runtime.state = DreamCampaignState.FAILED
            runtime.reason = str(payload["reason"])
            runtime.terminal_at_ns = event.timestamp_ns
            self._clear_lease(runtime)
            return
        if kind == "CAMPAIGN_COMPLETED":
            if runtime.state is not DreamCampaignState.RUNNING:
                raise ValueError("journal completes a campaign without an active lease")
            self._require_journal_lease(runtime, payload)
            result_manifest_hash = str(payload["result_manifest_hash"])
            _require_hash("result_manifest_hash", result_manifest_hash)
            candidates = tuple(str(value) for value in payload["candidate_artifact_hashes"])
            if len(candidates) != len(set(candidates)):
                raise ValueError("journal candidate hashes are duplicated")
            for value in candidates:
                _require_hash("candidate_artifact_hashes", value)
            if len(candidates) > runtime.candidates:
                raise ValueError("journal contains unaccounted candidate artifacts")
            runtime.state = DreamCampaignState.COMPLETED
            runtime.result_manifest_hash = result_manifest_hash
            runtime.candidate_artifact_hashes = candidates
            runtime.reason = "campaign completed; candidates remain inactive"
            runtime.terminal_at_ns = event.timestamp_ns
            self._clear_lease(runtime)
            return
        if kind == "BUDGET_EXHAUSTED":
            if runtime.state in _TERMINAL_STATES:
                raise ValueError("journal exhausts an already terminal campaign")
            runtime.state = DreamCampaignState.BUDGET_EXHAUSTED
            runtime.reason = str(payload["reason"])
            runtime.terminal_at_ns = event.timestamp_ns
            self._clear_lease(runtime)
            return
        raise ValueError(f"unsupported dream scheduler event: {kind}")

    @staticmethod
    def _require_journal_lease(runtime: _CampaignRuntime, payload: Mapping[str, Any]) -> None:
        if runtime.state is not DreamCampaignState.RUNNING:
            raise ValueError("journal lease event targets a non-running campaign")
        if runtime.lease_id is None or str(payload.get("lease_id")) != runtime.lease_id:
            raise ValueError("journal event references the wrong worker lease")

    @staticmethod
    def _clear_lease(runtime: _CampaignRuntime) -> None:
        runtime.worker_id = None
        runtime.lease_id = None
        runtime.lease_token_hash = None
        runtime.lease_expires_at_ns = None

    def _ensure_open(self) -> None:
        if self._lock_descriptor < 0:
            raise RuntimeError("Dream scheduler is closed")


def inspect_dream_journal(
    root: Path,
    *,
    source_checkout: Path,
    clock_ns: Callable[[], int] = time.time_ns,
) -> dict[str, Any]:
    """Read and validate an existing journal without appending or recovering."""

    service_root = require_external_service_root(root, source_checkout) / "dream"
    events_dir = service_root / "journal" / "events"
    if not events_dir.exists():
        return {
            "schema_version": "rosclaw.dream.journal_inspection.v1",
            "state": "EMPTY",
            "event_count": 0,
            "campaigns": [],
            "journal_integrity_verified": True,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    log = DurableEventLog(service_root / "journal", service="dream-scheduler", clock_ns=clock_ns)
    view = object.__new__(DreamScheduler)
    view.clock_ns = clock_ns
    view.log = log
    view._campaigns = {}
    view._issued_token_hashes = set()
    for event in log.events:
        view._apply(event)
    statuses = [view._status(runtime).to_dict() for runtime in view._campaigns.values()]
    return {
        "schema_version": "rosclaw.dream.journal_inspection.v1",
        "state": "READY",
        "event_count": len(log.events),
        "last_event_hash": log.last_hash,
        "campaigns": statuses,
        "journal_integrity_verified": True,
        "activation_authorized": False,
        "hardware_authorized": False,
    }


def dream_doctor(root: Path, *, source_checkout: Path) -> dict[str, Any]:
    """Return a read-only readiness report for an external Dream state root."""

    try:
        resolved = require_external_service_root(root, source_checkout)
    except ValueError as exc:
        return {
            "schema_version": "rosclaw.dream.doctor.v1",
            "ready": False,
            "state_root": str(root.expanduser().resolve()),
            "checks": {"external_state_root": False, "journal_integrity": False},
            "errors": [str(exc)],
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    errors: list[str] = []
    journal_integrity = True
    event_count = 0
    if resolved.exists():
        state_root_usable = resolved.is_dir()
    else:
        ancestor = resolved.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        state_root_usable = ancestor.is_dir() and os.access(ancestor, os.W_OK | os.X_OK)
    if not state_root_usable:
        errors.append("state root is not an initializable directory")
    try:
        report = inspect_dream_journal(resolved, source_checkout=source_checkout)
        event_count = int(report["event_count"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        journal_integrity = False
        errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "schema_version": "rosclaw.dream.doctor.v1",
        "ready": journal_integrity and state_root_usable,
        "state_root": str(resolved),
        "checks": {
            "external_state_root": True,
            "state_root_initializable": state_root_usable,
            "journal_integrity": journal_integrity,
            "planner_cannot_activate": not hasattr(DreamPlanner, "activate"),
        },
        "event_count": event_count,
        "errors": errors,
        "activation_authorized": False,
        "hardware_authorized": False,
    }


__all__ = [
    "DreamBudgetExceededError",
    "DreamBudgetUsage",
    "DreamCampaignState",
    "DreamCampaignStatus",
    "DreamLease",
    "DreamPlanReceipt",
    "DreamPlanRequest",
    "DreamPlanner",
    "DreamScheduler",
    "dream_doctor",
    "inspect_dream_journal",
]
