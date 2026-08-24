"""Privileged job execution flow (doc §21/§22/§23/§24/§25).

The operator-facing half of the HostOps plane:

- ``authorize_job`` runs on the operator's TTY: ``sudo -v`` authenticates
  *locally* (the password never enters any process of ours), the job moves
  to AUTHORIZED, and the execution phase is re-launched as root via
  ``sudo -n env … rosclaw host execute <JOB_ID>``.
- ``execute_authorized_job`` (internal, runs as root) performs the typed
  plan through the HostOps executor, then runs the skill's own verifier
  and records a uniform receipt (doc §25) through the job state machine
  (doc §24).

The agent side only ever sees job states — never credentials, never the
execution channel.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # noqa: S404 — fixed argv, no shell
import sys
import tempfile
import time
from pathlib import Path

from rosclaw.firstboot.workspace import get_rosclaw_home
from rosclaw.hostops.executor import HostOpsExecutor
from rosclaw.hostops.policy import HostOpsPolicy
from rosclaw.hostops.receipt import build_receipt
from rosclaw.skill.jobs import SkillJobStore
from rosclaw.skill.resolver import detect_host_context
from rosclaw.skill.service import SkillPlanError, load_skill_callable

logger = logging.getLogger("rosclaw.hostops.runner")

_SUDO_TIMEOUT = 300.0


class AuthorizationError(Exception):
    """The job cannot be authorized or executed in its current state."""


def authorize_job(
    job_id: str,
    *,
    home: Path | None = None,
    sudo_runner: object | None = None,
) -> dict:
    """Authenticate locally and launch the root execution phase.

    ``sudo_runner`` is injectable for tests; it receives an argv list and
    must return a subprocess returncode. The default runs real sudo on the
    controlling TTY (inheriting stdio so the password prompt is sudo's own).
    """
    home = Path(home) if home is not None else get_rosclaw_home()
    store = SkillJobStore(home)
    job = store.get(job_id)
    if job is None:
        raise AuthorizationError(f"job {job_id!r} not found")
    if job["status"] not in {"AWAITING_APPROVAL", "AUTHENTICATION_REQUIRED"}:
        raise AuthorizationError(
            f"job {job_id} is {job['status']}; only jobs awaiting approval or "
            f"authentication can be authorized"
        )
    if not job.get("plan"):
        raise AuthorizationError(f"job {job_id} has no persisted plan to execute")

    runner = sudo_runner or _run_on_tty
    # 1. Local authentication: sudo's own TTY prompt, credentials never
    #    pass through ROSClaw (doc §23).
    rc = runner(["sudo", "-v"], home)
    if rc != 0:
        store.update(job_id, status="FAILED", failure="sudo authentication failed")
        raise AuthorizationError("sudo authentication failed or was cancelled")

    store.update(job_id, status="AUTHORIZED", authorized_at=_now())

    # 2. Re-launch the execution phase as root. ``sudo -n`` rides the
    #    cached credentials from step 1; ROSCLAW_HOME is passed explicitly.
    #    Python/venv env of the *caller's* toolchain must NOT leak into the
    #    root phase: a foreign PYTHONPATH/LD_LIBRARY_PATH breaks system
    #    Python applications (e.g. the ros2 CLI's importlib.metadata).
    entrypoint = [
        sys.executable,
        "-m",
        "rosclaw.entrypoint",
        "host",
        "execute",
        job_id,
    ]
    rc = runner(
        [
            "sudo",
            "-n",
            "env",
            "-u",
            "PYTHONPATH",
            "-u",
            "PYTHONHOME",
            "-u",
            "VIRTUAL_ENV",
            "-u",
            "LD_LIBRARY_PATH",
            f"ROSCLAW_HOME={home}",
            *entrypoint,
        ],
        home,
    )
    if rc != 0:
        fresh = store.get(job_id)
        state = fresh["status"] if fresh else f"unreadable (check {home}/skills/jobs permissions)"
        raise AuthorizationError(
            f"execution phase exited with code {rc}; job {job_id} is "
            f"{state} (see `rosclaw skill job {job_id}`)"
        )
    return store.get(job_id)


def execute_authorized_job(
    job_id: str,
    *,
    home: Path | None = None,
    executor: HostOpsExecutor | None = None,
) -> dict:
    """Internal: run the plan + verifier as root. Returns the receipt."""
    home = Path(home) if home is not None else get_rosclaw_home()
    store = SkillJobStore(home)
    job = store.get(job_id)
    if job is None:
        raise AuthorizationError(f"job {job_id!r} not found")
    if job["status"] != "AUTHORIZED":
        raise AuthorizationError(
            f"job {job_id} is {job['status']}; refusing to execute without "
            f"local authorization (fail closed, doc §23)"
        )
    plan = job.get("plan")
    if not plan:
        raise AuthorizationError(f"job {job_id} has no persisted plan")

    store.update(job_id, status="EXECUTING")
    executor = executor or HostOpsExecutor(dry_run=False)
    approval = HostOpsPolicy().approve(job["plan_hash"])
    execution = executor.execute(plan, approval)

    receipt_extra: dict = {}
    final_status = "FAILED"
    verification: dict = {}
    if execution["status"] == "OK":
        store.update(job_id, status="VERIFYING")
        verification = _run_verifier(home, job, execution)
        if verification.get("result") == "VERIFIED":
            final_status = "SUCCEEDED"
        else:
            receipt_extra["recovery"] = _run_recovery(home, job, execution, verification)
    else:
        receipt_extra["recovery"] = _run_recovery(home, job, execution, verification)

    receipt = build_receipt(
        job_id=job_id,
        plan=plan,
        environment=detect_host_context(),
        intent=job.get("capability") or "",
        operations_result=execution,
        verification=verification,
        result="VERIFIED" if final_status == "SUCCEEDED" else "FAILED",
    )
    receipt.update(receipt_extra)
    _write_receipt(home, job_id, receipt)
    store.update(
        job_id,
        status=final_status,
        verification=verification,
        receipt_path=str(_receipt_path(home, job_id)),
    )
    return receipt


# ---------------------------------------------------------------------------
# Verifier / recovery
# ---------------------------------------------------------------------------


def _run_verifier(home: Path, job: dict, execution: dict) -> dict:
    """Run the skill's own verifier (doc §38/§39); a missing or crashing
    verifier is an honest FAILED, never a silent pass."""
    ref = job.get("skill", "")
    try:
        verify_fn, _manifest, _version = load_skill_callable(home, ref, "verifier")
        result = verify_fn(detect_host_context(), execution)
        if not isinstance(result, dict):
            return {"result": "FAILED", "reason": "verifier returned non-dict"}
        return result
    except SkillPlanError:
        return {"result": "FAILED", "reason": f"skill {ref} has no verifier"}
    except Exception as exc:  # noqa: BLE001 — verifier isolation
        logger.warning("verifier for %s crashed: %s", ref, exc)
        return {"result": "FAILED", "reason": f"verifier crashed: {exc}"}


def _run_recovery(home: Path, job: dict, execution: dict, verification: dict) -> dict:
    """Ask the skill for a recovery plan (doc §37); optional."""
    ref = job.get("skill", "")
    try:
        recover_fn, _manifest, _version = load_skill_callable(home, ref, "recover")
    except SkillPlanError:
        return {}
    failure = {
        "execution": {k: v for k, v in execution.items() if k != "results"},
        "failed_op": next(
            (r for r in execution.get("results", []) if r.get("status") == "FAILED"),
            None,
        ),
        "verification": verification,
    }
    try:
        result = recover_fn(detect_host_context(), failure)
        return result if isinstance(result, dict) else {}
    except Exception as exc:  # noqa: BLE001 — recovery must not crash the job
        logger.warning("recover for %s crashed: %s", ref, exc)
        return {}


# ---------------------------------------------------------------------------
# Receipt persistence
# ---------------------------------------------------------------------------


def _receipt_path(home: Path, job_id: str) -> Path:
    return home / "skills" / "jobs" / f"{job_id}.receipt.json"


def _write_receipt(home: Path, job_id: str, receipt: dict) -> None:
    path = _receipt_path(home, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".rcpt-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, ensure_ascii=False)
    # Written as root during execution; must remain operator-readable.
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _run_on_tty(argv: list[str], home: Path) -> int:
    completed = subprocess.run(argv, check=False, timeout=_SUDO_TIMEOUT)  # noqa: S603
    return completed.returncode


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
