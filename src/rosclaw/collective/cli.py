"""CLI for governed external-experience registration and ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rosclaw.collective.contracts import LicenseDecision, LicenseUse
from rosclaw.continual.services.persistence import atomic_write_json

if TYPE_CHECKING:
    from rosclaw.collective.sources.motiondecode.manifest import MotionDecodeRegistration
    from rosclaw.collective.sources.motiondecode.taxonomy import MotionFamily


def dispatch_collective_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "collective":
        return None
    args = _parser().parse_args(argv[1:])
    try:
        return int(args.handler(args))
    except (OSError, ValueError, KeyError, PermissionError) as exc:
        _print(
            {
                "schema_version": "rosclaw.collective.cli_error.v1",
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "training_eligible": False,
                "activation_authorized": False,
                "hardware_authorized": False,
            },
            stream=sys.stderr,
        )
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosclaw collective",
        description="Register and audit external experience without authorizing hardware.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("source", help="manage content-addressed source evidence")
    source_commands = source.add_subparsers(dest="source_command", required=True)

    add = source_commands.add_parser("add", help="register an operator-managed local snapshot")
    add.add_argument("adapter", choices=["motiondecode"])
    add.add_argument("--dataset-root", type=Path, required=True)
    add.add_argument("--revision", required=True)
    add.add_argument(
        "--usage",
        choices=[item.value for item in LicenseUse],
        default=LicenseUse.RESEARCH_NONCOMMERCIAL.value,
    )
    add.add_argument(
        "--license-decision",
        choices=[item.value for item in LicenseDecision],
        default=LicenseDecision.PENDING.value,
    )
    add.add_argument("--terms-file", type=Path)
    add.add_argument("--terms-uri")
    add.add_argument("--attribution", default="ChingMu / CMRobot MotionDecode")
    add.add_argument(
        "--families",
        default="",
        help="comma-separated football,balance,gait,transition_recovery,other",
    )
    add.add_argument("--limit", type=int, default=400)
    add.add_argument("--output", type=Path, required=True)
    add.add_argument("--source-checkout", type=Path, default=Path.cwd())
    add.add_argument("--force", action="store_true")
    add.set_defaults(handler=_source_add)

    inspect = source_commands.add_parser(
        "inspect", help="replay and summarize a registration artifact"
    )
    inspect.add_argument("adapter", choices=["motiondecode"])
    inspect.add_argument("--registration", type=Path, required=True)
    inspect.set_defaults(handler=_source_inspect)

    ingest = commands.add_parser(
        "ingest",
        help="rehash and kinematically audit a registered local snapshot",
    )
    ingest.add_argument("adapter", choices=["motiondecode"])
    ingest.add_argument("--registration", type=Path, required=True)
    ingest.add_argument("--dataset-root", type=Path, required=True)
    ingest.add_argument("--target-model", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--source-checkout", type=Path, default=Path.cwd())
    ingest.add_argument("--force", action="store_true")
    ingest.set_defaults(handler=_ingest)
    return parser


def _source_add(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.manifest import (
        register_motiondecode_source,
    )

    output = _output_path(args.output, args.source_checkout, force=args.force)
    registration = register_motiondecode_source(
        args.dataset_root,
        revision=args.revision,
        requested_use=LicenseUse(args.usage),
        license_decision=LicenseDecision(args.license_decision),
        terms_path=args.terms_file,
        terms_uri=args.terms_uri,
        attribution_text=args.attribution,
        families=_families(args.families),
        limit=args.limit,
    )
    artifact = _registration_artifact(registration)
    atomic_write_json(output, artifact)
    _print(
        {
            "schema_version": "rosclaw.collective.source_add_receipt.v1",
            "ok": registration.source_registered,
            "output": str(output),
            "registration_hash": registration.registration_hash,
            "source_manifest_hash": registration.manifest.manifest_hash,
            "catalog_schema_valid": registration.catalog_audit.schema_valid,
            "selected_sample_count": registration.manifest.selected_sample_count,
            "license_decision": registration.manifest.license_snapshot.decision.value,
            "training_eligible": registration.training_eligible,
            "training_blockers": registration.to_dict()["training_blockers"],
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if registration.source_registered else 1


def _source_inspect(args: argparse.Namespace) -> int:
    registration = _read_registration(args.registration)
    _print(
        {
            "schema_version": "rosclaw.collective.source_inspection.v1",
            "ok": registration.source_registered,
            "registration_hash": registration.registration_hash,
            "source_manifest_hash": registration.manifest.manifest_hash,
            "source_identity_hash": registration.manifest.source_identity.source_hash,
            "revision": registration.manifest.revision,
            "catalog_audit": registration.catalog_audit.to_dict(),
            "selected_sample_count": registration.manifest.selected_sample_count,
            "inventory_scope": "operator_managed_local_snapshot",
            "upstream_inventory_verified": False,
            "local_discovered_sample_count": (registration.manifest.local_discovered_sample_count),
            "local_selection_complete": registration.manifest.local_selection_complete,
            "requested_families": [
                family.value for family in registration.manifest.requested_families
            ],
            "license_snapshot": registration.manifest.license_snapshot.to_dict(),
            "attribution": registration.manifest.attribution.to_dict(),
            "training_eligible": registration.training_eligible,
            "training_blockers": registration.to_dict()["training_blockers"],
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if registration.source_registered else 1


def _ingest(args: argparse.Namespace) -> int:
    from rosclaw.collective.sources.motiondecode.audit import (
        audit_motiondecode_snapshot,
    )

    output = _output_path(args.output, args.source_checkout, force=args.force)
    registration = _read_registration(args.registration)
    report = audit_motiondecode_snapshot(
        registration,
        args.dataset_root,
        target_model_path=args.target_model,
    )
    artifact = {
        "schema_version": "rosclaw.collective.motiondecode_ingest_artifact.v1",
        "report": report.to_dict(),
        "report_hash": report.report_hash,
    }
    atomic_write_json(output, artifact)
    _print(
        {
            "schema_version": "rosclaw.collective.ingest_receipt.v1",
            "ok": report.kinematic_valid_count > 0,
            "output": str(output),
            "report_hash": report.report_hash,
            "source_manifest_hash": report.source_manifest_hash,
            "clip_count": len(report.clips),
            "kinematic_valid_count": report.kinematic_valid_count,
            "qualification_counts": report.qualification_counts,
            "issue_clip_counts": report.issue_clip_counts,
            "segmentation_repair_candidate_count": (report.segmentation_repair_candidate_count),
            "experience_capsule_hash": (
                report.experience_capsule.capsule_hash
                if report.experience_capsule is not None
                else None
            ),
            "training_eligible": report.training_eligible,
            "training_blockers": report.training_blockers,
            "activation_authorized": False,
            "hardware_authorized": False,
        }
    )
    return 0 if report.kinematic_valid_count > 0 else 1


def _families(value: str) -> tuple[MotionFamily, ...]:
    from rosclaw.collective.sources.motiondecode.taxonomy import MotionFamily

    if not value.strip():
        return ()
    names = tuple(item.strip() for item in value.split(","))
    if any(not item for item in names) or len(names) != len(set(names)):
        raise ValueError("families must contain unique non-empty names")
    try:
        return tuple(MotionFamily(item) for item in names)
    except ValueError as exc:
        raise ValueError("families contains an unknown MotionDecode family") from exc


def _registration_artifact(registration: MotionDecodeRegistration) -> dict[str, Any]:
    return {
        "schema_version": "rosclaw.collective.motiondecode_registration_artifact.v1",
        "registration": registration.to_dict(),
        "registration_hash": registration.registration_hash,
    }


def _read_registration(path: Path) -> MotionDecodeRegistration:
    from rosclaw.collective.sources.motiondecode.manifest import (
        MotionDecodeRegistration,
    )

    value = _read_object(path)
    registration_value = value.get("registration")
    if not isinstance(registration_value, dict):
        raise ValueError("registration artifact lacks a registration object")
    registration = MotionDecodeRegistration.from_dict(registration_value)
    if value.get("registration_hash") != registration.registration_hash:
        raise ValueError("registration_hash does not replay")
    return registration


def _read_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON artifact must be an object")
    return value


def _output_path(path: Path, checkout: Path, *, force: bool) -> Path:
    resolved = path.expanduser().resolve()
    source = checkout.expanduser().resolve()
    if resolved == source or source in resolved.parents:
        raise ValueError("collective evidence output must be outside the source checkout")
    if resolved.exists() and not force:
        raise FileExistsError("output already exists; pass --force to replace it")
    return resolved


def _print(value: Mapping[str, Any], *, stream: Any | None = None) -> None:
    print(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False),
        file=stream if stream is not None else sys.stdout,
    )


__all__ = ["dispatch_collective_argv"]
