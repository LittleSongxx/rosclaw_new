"""Skill job store — the unified skill job state machine (doc §24).

Every skill execution, host-domain or otherwise, gets a job record with a
stable lifecycle:

    RESOLVED → ACQUIRED → VALIDATED → PLANNED → AWAITING_APPROVAL
    → AUTHORIZED → EXECUTING → VERIFYING → SUCCEEDED

plus FAILED / DEGRADED / CANCELLED / ROLLING_BACK / ROLLED_BACK.

Records persist under ``$ROSCLAW_HOME/skills/jobs/<job_id>.json`` so
``get_skill_job`` / ``cancel_skill_job`` have stable semantics across
CLI, MCP and runtime restarts.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.hostops.receipt import new_job_id

JOB_STATES: frozenset[str] = frozenset(
    {
        "RESOLVED",
        "ACQUIRED",
        "VALIDATED",
        "PLANNED",
        "AWAITING_APPROVAL",
        "AUTHENTICATION_REQUIRED",
        "AUTHORIZED",
        "EXECUTING",
        "VERIFYING",
        "SUCCEEDED",
        "FAILED",
        "DEGRADED",
        "CANCELLED",
        "ROLLING_BACK",
        "ROLLED_BACK",
    }
)

TERMINAL_STATES: frozenset[str] = frozenset(
    {"SUCCEEDED", "FAILED", "CANCELLED", "ROLLED_BACK"}
)

_JOB_ID_SAFE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9-]*$")


class SkillJobStore:
    """Filesystem-backed job records (one JSON file per job)."""

    def __init__(self, home: Path | None = None) -> None:
        self._home = Path(home) if home is not None else get_rosclaw_home()

    @property
    def jobs_dir(self) -> Path:
        return self._home / "skills" / "jobs"

    def create(
        self,
        *,
        skill: str,
        capability: str | None,
        status: str,
        plan_hash: str = "",
        plan: dict | None = None,
    ) -> dict:
        if status not in JOB_STATES:
            raise ValueError(f"unknown job status {status!r}")
        job = {
            "job_id": new_job_id(),
            "skill": skill,
            "capability": capability,
            "status": status,
            "plan_hash": plan_hash,
            "plan": plan,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self._write(job)
        return job

    def get(self, job_id: str) -> dict | None:
        path = self._path_for(job_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def update(self, job_id: str, *, status: str | None = None, **fields) -> dict:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"job {job_id!r} not found")
        if status is not None:
            if status not in JOB_STATES:
                raise ValueError(f"unknown job status {status!r}")
            job["status"] = status
        job.update(fields)
        job["updated_at"] = _now()
        self._write(job)
        return job

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"job {job_id!r} not found")
        if job["status"] in TERMINAL_STATES:
            return job  # already terminal; cancel is a no-op
        return self.update(job_id, status="CANCELLED")

    def _path_for(self, job_id: str) -> Path:
        if not _JOB_ID_SAFE.match(job_id):
            raise ValueError(f"invalid job id {job_id!r}")
        return self.jobs_dir / f"{job_id}.json"

    def _write(self, job: dict) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.jobs_dir, prefix=".job-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(job, fh, indent=2, ensure_ascii=False)
        # mkstemp is 0600; job records must stay readable when the root
        # execution phase updates them and the operator reads them back.
        os.chmod(tmp, 0o644)
        os.replace(tmp, self.jobs_dir / f"{job['job_id']}.json")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
