"""Product CLI for GoalForge Hat Trick evidence and visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dispatch_hat_trick_argv(argv: list[str]) -> int | None:
    if len(argv) < 3 or argv[:2] != ["goalforge", "hat-trick"]:
        return None
    parser = argparse.ArgumentParser(prog="rosclaw goalforge hat-trick")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute three strictly replayed MuJoCo shots")
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser("export", help="render a passing Hat Trick report")
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_hat_trick import run_goalforge_hat_trick

        result = run_goalforge_hat_trick(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    from rosclaw.simforge.g1_hat_trick_video import render_goalforge_hat_trick_video

    result = render_goalforge_hat_trick_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


__all__ = ["dispatch_hat_trick_argv"]
