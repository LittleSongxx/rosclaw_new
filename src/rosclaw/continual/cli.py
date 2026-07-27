"""CLI for inspecting and evaluating immutable continual candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rosclaw.continual.contracts import PolicyVersion


def dispatch_continual_argv(argv: list[str]) -> int | None:
    if len(argv) < 2 or argv[0] != "continual":
        return None
    if argv[1] == "services":
        from rosclaw.continual.service_cli import dispatch_continual_service_argv

        return dispatch_continual_service_argv(argv[2:])
    if argv[1] != "candidate":
        return None
    parser = argparse.ArgumentParser(prog="rosclaw continual candidate")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="verify a candidate without activating it")
    _candidate_arguments(inspect)
    evaluate = commands.add_parser("evaluate", help="run matched G1 MuJoCo evaluation")
    _candidate_arguments(evaluate)
    evaluate.add_argument("--backend", choices=("mujoco",), default="mujoco")
    evaluate.add_argument(
        "--suite", choices=("g1-goalforge-eval-v1",), default="g1-goalforge-eval-v1"
    )
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--source-checkout", type=Path, default=Path.cwd())
    evaluate.add_argument("--recent", type=int, default=50)
    evaluate.add_argument("--anchor", type=int, default=50)
    evaluate.add_argument("--boundary", type=int, default=100)
    evaluate.add_argument("--self-count", type=int, default=50)
    evaluate.add_argument("--training-seed-count", type=int, default=1)
    evaluate.add_argument("--suite-shard", default="")
    merge = commands.add_parser("merge", help="merge disjoint matched-evaluation shards")
    merge.add_argument("shards", type=Path, nargs="+")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--source-checkout", type=Path, default=Path.cwd())
    args = parser.parse_args(argv[2:])
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "merge":
        return _merge(args)
    return _evaluate(args)


def _candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)


def _inspect(args: argparse.Namespace) -> int:
    from rosclaw.continual.g1_goalforge import build_g1_policy_lineage
    from rosclaw.continual.inference import load_residual_candidate
    from rosclaw.simforge.backends.unitree_mujoco_backend import G1MuJoCoBackend

    candidate = _candidate_policy(args.candidate, args.metadata)
    backend = G1MuJoCoBackend(asset_root=args.asset_root, trace_stride=5)
    qualification = backend.qualification
    lineage = build_g1_policy_lineage(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        motion_hash=qualification.motion_hash,
        backend_commit=qualification.backend_commit,
        torque_guard_scale=backend.torque_guard_scale,
        through_version=2,
    )
    loaded = load_residual_candidate(
        args.candidate,
        policy=candidate,
        parent=lineage.policy(2),
        expected_body_hash=qualification.body_hash,
    )
    print(
        json.dumps(
            {
                "schema_version": "rosclaw.continual.candidate_inspection.v1",
                "verified": True,
                "artifact_hash": loaded.artifact_hash,
                "policy_version": loaded.policy.version,
                "policy_version_hash": loaded.policy.version_hash,
                "parent_version_hash": loaded.parent.version_hash,
                "body_hash": loaded.policy.body_hash,
                "observation_names": list(loaded.observation_names),
                "action_names": list(loaded.action_names),
                "action_limits": list(loaded.action_limits),
                "hidden_dims": list(loaded.hidden_dims),
                "learner_update_index": loaded.update_index,
                "read_only": True,
                "registry_mutated": False,
                "dds_opened": False,
                "candidate_activated": False,
                "hardware_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from rosclaw.simforge.g1_candidate_evaluation import (
        CandidateEvaluationCounts,
        run_g1_candidate_matched_evaluation,
    )

    candidate = _candidate_policy(args.candidate, args.metadata)
    counts = CandidateEvaluationCounts(
        recent=args.recent,
        anchor=args.anchor,
        boundary=args.boundary,
        self_partition=args.self_count,
    )

    def progress(completed: int, total: int) -> None:
        if completed == 1 or completed == total or completed % max(1, total // 20) == 0:
            print(f"candidate-eval {completed}/{total}", file=sys.stderr, flush=True)

    result = run_g1_candidate_matched_evaluation(
        asset_root=args.asset_root,
        candidate_artifact_path=args.candidate,
        candidate_policy=candidate,
        output_path=args.output,
        source_checkout=args.source_checkout,
        counts=counts,
        training_seed_count=args.training_seed_count,
        suite_shard=args.suite_shard,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "scenario_count": counts.total,
                "candidate_policy_hash": candidate.version_hash,
                "paired_statistics": result.paired_statistics,
                "gate": result.gate,
                "passed": result.passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _merge(args: argparse.Namespace) -> int:
    from rosclaw.simforge.g1_candidate_evaluation import merge_g1_candidate_evaluations

    result = merge_g1_candidate_evaluations(
        shard_paths=args.shards,
        output_path=args.output,
        source_checkout=args.source_checkout,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "scenario_count": result.counts["total"],
                "paired_statistics": result.paired_statistics,
                "gate": result.gate,
                "passed": result.passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _candidate_policy(candidate_path: Path, metadata_path: Path | None) -> PolicyVersion:
    metadata = metadata_path or _infer_metadata_path(candidate_path)
    if not metadata.is_file():
        raise FileNotFoundError(
            "candidate learner metadata is missing; provide --metadata: " + str(metadata)
        )
    value: Any = json.loads(metadata.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "candidate_policy" in value:
        value = value["candidate_policy"]
    if not isinstance(value, dict):
        raise ValueError("candidate metadata must contain a candidate_policy object")
    return PolicyVersion(
        version=int(value["version"]),
        artifact_hash=str(value["artifact_hash"]),
        parent_version_hash=value.get("parent_version_hash"),
        controller_snapshot_hash=str(value["controller_snapshot_hash"]),
        body_hash=str(value["body_hash"]),
        safety_kernel_hash=str(value["safety_kernel_hash"]),
        observation_names=tuple(value["observation_names"]),
        residual_action_names=tuple(value["residual_action_names"]),
    )


def _infer_metadata_path(candidate_path: Path) -> Path:
    name = candidate_path.name
    if name.endswith("-candidate.bin"):
        return candidate_path.with_name(name.removesuffix("-candidate.bin") + "-learner.json")
    return candidate_path.with_suffix(".json")


__all__ = ["dispatch_continual_argv"]
