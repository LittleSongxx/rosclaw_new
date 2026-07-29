"""CLI for real-MuJoCo continual service validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dispatch_continual_service_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw continual services")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate",
        help="run the five recoverable services against real G1 MuJoCo evidence",
    )
    validate.add_argument("--asset-root", type=Path, required=True)
    validate.add_argument("--candidate", type=Path, required=True)
    validate.add_argument("--matched-report", type=Path, required=True)
    validate.add_argument("--state-root", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--source-checkout", type=Path, default=Path.cwd())
    validate.add_argument("--learner-device", default="cpu")
    validate.add_argument("--learner-updates", type=int, default=1)
    args = parser.parse_args(argv)

    from rosclaw.continual.service_validation import run_g1_service_validation

    result = run_g1_service_validation(
        asset_root=args.asset_root,
        candidate_artifact_path=args.candidate,
        matched_report_path=args.matched_report,
        state_root=args.state_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        learner_device=args.learner_device,
        learner_updates=args.learner_updates,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "state_root": str(args.state_root.expanduser().resolve()),
                "passed": result["passed"],
                "checks": result["checks"],
                "report_hash": result["report_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


__all__ = ["dispatch_continual_service_argv"]
