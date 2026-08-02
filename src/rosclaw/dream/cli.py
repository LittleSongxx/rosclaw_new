"""Command-line control surface for bounded Dream campaigns."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rosclaw.continual.services.persistence import atomic_write_json
from rosclaw.dream.control import (
    DreamPlanner,
    DreamScheduler,
    dream_doctor,
    inspect_dream_journal,
)
from rosclaw.dream.serde import (
    dream_campaign_from_dict,
    dream_plan_request_from_dict,
    skill_growth_spec_from_dict,
)


def dispatch_dream_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "dream":
        return None
    parser = _parser()
    args = parser.parse_args(argv[1:])
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, KeyError, PermissionError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "rosclaw.dream.cli_error.v1",
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "activation_authorized": False,
                    "hardware_authorized": False,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosclaw dream",
        description="Plan and schedule bounded SIM-only DreamForge campaigns.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect state-root and journal readiness")
    _state_arguments(doctor)
    doctor.set_defaults(handler=_doctor)

    plan = commands.add_parser("plan", help="build an immutable campaign without submitting it")
    plan.add_argument("--spec", type=Path, required=True)
    plan.add_argument("--request", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--force", action="store_true", help="replace an existing campaign output")
    plan.add_argument("--source-checkout", type=Path, default=Path.cwd())
    plan.set_defaults(handler=_plan)

    submit = commands.add_parser("submit", help="submit a planned campaign")
    _state_arguments(submit)
    submit.add_argument("--campaign", type=Path, required=True)
    submit.set_defaults(handler=_submit)

    inspect = commands.add_parser("inspect", help="read the journal without mutating it")
    _state_arguments(inspect)
    inspect.add_argument("--campaign-hash")
    inspect.set_defaults(handler=_inspect)

    acquire = commands.add_parser("acquire", help="lease one queued campaign to a worker")
    _state_arguments(acquire)
    acquire.add_argument("--worker-id", required=True)
    acquire.add_argument("--lease-seconds", type=float, default=60.0)
    acquire.add_argument("--campaign-hash")
    acquire.add_argument("--lease-token-file", type=Path, required=True)
    acquire.set_defaults(handler=_acquire)

    heartbeat = commands.add_parser("heartbeat", help="renew an active worker lease")
    _leased_arguments(heartbeat)
    heartbeat.add_argument("--extend-seconds", type=float, default=60.0)
    heartbeat.set_defaults(handler=_heartbeat)

    usage = commands.add_parser("usage", help="record bounded worker resource consumption")
    _leased_arguments(usage)
    usage.add_argument("--gpu-seconds", type=float, default=0.0)
    usage.add_argument("--cpu-rollouts", type=int, default=0)
    usage.add_argument("--candidates", type=int, default=0)
    usage.set_defaults(handler=_usage)

    pause = commands.add_parser("pause", help="release a worker lease and pause the campaign")
    _leased_arguments(pause)
    pause.add_argument("--reason", required=True)
    pause.set_defaults(handler=_pause)

    resume = commands.add_parser("resume", help="return a paused campaign to the queue")
    _campaign_arguments(resume)
    resume.add_argument("--reason", required=True)
    resume.set_defaults(handler=_resume)

    cancel = commands.add_parser("cancel", help="terminally cancel a non-terminal campaign")
    _campaign_arguments(cancel)
    cancel.add_argument("--reason", required=True)
    cancel.set_defaults(handler=_cancel)

    fail = commands.add_parser("fail", help="terminally fail an active worker campaign")
    _leased_arguments(fail)
    fail.add_argument("--reason", required=True)
    fail.set_defaults(handler=_fail)

    complete = commands.add_parser("complete", help="record inactive candidate result hashes")
    _leased_arguments(complete)
    complete.add_argument("--result-manifest-hash", required=True)
    complete.add_argument("--candidate-artifact-hash", action="append", default=[])
    complete.set_defaults(handler=_complete)
    return parser


def _state_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, default=Path.cwd())


def _campaign_arguments(parser: argparse.ArgumentParser) -> None:
    _state_arguments(parser)
    parser.add_argument("--campaign-hash", required=True)


def _leased_arguments(parser: argparse.ArgumentParser) -> None:
    _campaign_arguments(parser)
    parser.add_argument("--lease-token-file", type=Path, required=True)


def _doctor(args: argparse.Namespace) -> int:
    report = dream_doctor(args.state_root, source_checkout=args.source_checkout)
    _print(report)
    return 0 if report["ready"] else 1


def _plan(args: argparse.Namespace) -> int:
    _require_external_path(args.output, args.source_checkout, label="campaign output")
    if args.output.expanduser().exists() and not args.force:
        raise FileExistsError("campaign output already exists; pass --force to replace it")
    spec = skill_growth_spec_from_dict(_read_object(args.spec, label="growth spec"))
    request = dream_plan_request_from_dict(_read_object(args.request, label="plan request"))
    receipt = DreamPlanner().plan(spec, request)
    atomic_write_json(args.output.expanduser().resolve(), receipt.to_dict())
    _print(
        {
            "schema_version": "rosclaw.dream.plan_cli.v1",
            "ok": True,
            "output": str(args.output.expanduser().resolve()),
            "campaign_hash": receipt.campaign.campaign_hash,
            "receipt_hash": receipt.receipt_hash,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0


def _submit(args: argparse.Namespace) -> int:
    value = _read_object(args.campaign, label="campaign")
    if "campaign" in value:
        value = _as_mapping(value["campaign"], label="campaign receipt payload")
    campaign = dream_campaign_from_dict(value)
    with _scheduler(args) as scheduler:
        status = scheduler.submit(campaign)
    _print(status.to_dict())
    return 0


def _inspect(args: argparse.Namespace) -> int:
    report = inspect_dream_journal(
        args.state_root,
        source_checkout=args.source_checkout,
    )
    if args.campaign_hash is not None:
        campaigns = [
            item
            for item in report["campaigns"]
            if isinstance(item, Mapping) and item.get("campaign_hash") == args.campaign_hash
        ]
        if not campaigns:
            raise KeyError("unknown dream campaign")
        report = {**report, "campaigns": campaigns}
    _print(report)
    return 0


def _acquire(args: argparse.Namespace) -> int:
    _require_external_path(args.lease_token_file, args.source_checkout, label="lease token file")
    if args.lease_token_file.expanduser().exists():
        raise FileExistsError("lease token file already exists")
    with _scheduler(args) as scheduler:
        lease = scheduler.acquire(
            worker_id=args.worker_id,
            lease_seconds=args.lease_seconds,
            campaign_hash=args.campaign_hash,
        )
        try:
            _write_secret(args.lease_token_file, lease.lease_token)
        except BaseException:
            scheduler.pause(
                lease.campaign_hash,
                lease_token=lease.lease_token,
                reason="lease token delivery failed",
            )
            raise
    _print(
        {
            **lease.to_dict(),
            "lease_token_file": str(args.lease_token_file.expanduser().resolve()),
        }
    )
    return 0


def _heartbeat(args: argparse.Namespace) -> int:
    with _scheduler(args) as scheduler:
        status = scheduler.heartbeat(
            args.campaign_hash,
            lease_token=_read_secret(args.lease_token_file, args.source_checkout),
            extend_seconds=args.extend_seconds,
        )
    _print(status.to_dict())
    return 0


def _usage(args: argparse.Namespace) -> int:
    with _scheduler(args) as scheduler:
        status = scheduler.record_usage(
            args.campaign_hash,
            lease_token=_read_secret(args.lease_token_file, args.source_checkout),
            gpu_seconds=args.gpu_seconds,
            cpu_rollouts=args.cpu_rollouts,
            candidates=args.candidates,
        )
    _print(status.to_dict())
    return 0


def _pause(args: argparse.Namespace) -> int:
    with _scheduler(args) as scheduler:
        status = scheduler.pause(
            args.campaign_hash,
            lease_token=_read_secret(args.lease_token_file, args.source_checkout),
            reason=args.reason,
        )
    _print(status.to_dict())
    return 0


def _resume(args: argparse.Namespace) -> int:
    with _scheduler(args) as scheduler:
        status = scheduler.resume(args.campaign_hash, reason=args.reason)
    _print(status.to_dict())
    return 0


def _cancel(args: argparse.Namespace) -> int:
    with _scheduler(args) as scheduler:
        status = scheduler.cancel(args.campaign_hash, reason=args.reason)
    _print(status.to_dict())
    return 0


def _fail(args: argparse.Namespace) -> int:
    with _scheduler(args) as scheduler:
        status = scheduler.fail(
            args.campaign_hash,
            lease_token=_read_secret(args.lease_token_file, args.source_checkout),
            reason=args.reason,
        )
    _print(status.to_dict())
    return 0


def _complete(args: argparse.Namespace) -> int:
    with _scheduler(args) as scheduler:
        status = scheduler.complete(
            args.campaign_hash,
            lease_token=_read_secret(args.lease_token_file, args.source_checkout),
            result_manifest_hash=args.result_manifest_hash,
            candidate_artifact_hashes=tuple(args.candidate_artifact_hash),
        )
    _print(status.to_dict())
    return 0


def _scheduler(args: argparse.Namespace) -> DreamScheduler:
    return DreamScheduler(args.state_root, source_checkout=args.source_checkout)


def _read_object(path: Path, *, label: str) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    return _as_mapping(value, label=label)


def _as_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_external_path(path: Path, checkout: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    source = checkout.expanduser().resolve()
    if resolved == source or source in resolved.parents:
        raise ValueError(f"{label} must be outside the source checkout")
    return resolved


def _write_secret(path: Path, token: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise


def _read_secret(path: Path, checkout: Path) -> str:
    resolved = _require_external_path(path, checkout, label="lease token file")
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise PermissionError("lease token file must not be group/world accessible")
    token = resolved.read_text(encoding="utf-8").strip()
    if len(token) < 16:
        raise ValueError("lease token file is empty or malformed")
    return token


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False))


__all__ = ["dispatch_dream_argv"]
