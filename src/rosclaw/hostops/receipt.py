"""Execution receipts for HostOps jobs (doc §25).

Receipts are JSON-native and carry the plan hash so an auditor can
recompute what exactly was approved and executed.
"""

from __future__ import annotations

import time
import uuid

from rosclaw.hostops.planner import plan_hash


def new_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


def build_receipt(
    *,
    job_id: str,
    plan: dict,
    environment: dict,
    intent: str = "",
    operations_result: dict | None = None,
    verification: dict | None = None,
    result: str = "PLANNED",
) -> dict:
    """Assemble a uniform execution receipt (doc §25)."""
    return {
        "job_id": job_id,
        "skill": plan.get("skill"),
        "domain": plan.get("domain"),
        "intent": intent,
        "environment": dict(environment),
        "plan_hash": plan_hash(plan),
        "operations": (operations_result or {}).get("results", []),
        "verification": verification or {},
        "result": result,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
