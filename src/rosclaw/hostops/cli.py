"""CLI handlers for ``rosclaw host authorize|execute`` (doc §23).

``authorize`` runs on the operator's TTY; ``execute`` is the internal
root phase re-launched via ``sudo -n``. Neither ever touches credential
material — the password prompt belongs to sudo alone.
"""

from __future__ import annotations

import argparse
import json

from rosclaw.hostops.runner import (
    AuthorizationError,
    authorize_job,
    execute_authorized_job,
)


def cmd_host_authorize(args: argparse.Namespace) -> int:
    try:
        job = authorize_job(args.job_id)
    except AuthorizationError as exc:
        print(f"[ROSClaw] Authorization failed: {exc}")
        return 1
    print(f"[ROSClaw] Job {job['job_id']}: {job['status']}")
    if job.get("receipt_path"):
        print(f"  receipt: {job['receipt_path']}")
    return 0 if job["status"] in {"SUCCEEDED", "AUTHORIZED"} else 1


def cmd_host_execute(args: argparse.Namespace) -> int:
    """Internal entry point; invoked as root by `host authorize`."""
    try:
        receipt = execute_authorized_job(args.job_id)
    except AuthorizationError as exc:
        print(f"[ROSClaw] Execution refused: {exc}")
        return 1
    if args.json:
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] Job {receipt['job_id']}: {receipt['result']}")
        for check, verdict in (receipt.get("verification") or {}).items():
            print(f"  {check}: {verdict}")
    return 0 if receipt["result"] == "VERIFIED" else 1
