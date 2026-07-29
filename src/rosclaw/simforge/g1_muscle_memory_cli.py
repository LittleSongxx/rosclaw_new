"""CLI for training and inspecting SIM-only G1 muscle memory artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dispatch_muscle_memory_argv(argv: list[str]) -> int | None:
    if len(argv) < 3 or argv[:2] != ["goalforge", "muscle-memory"]:
        return None
    parser = argparse.ArgumentParser(prog="rosclaw goalforge muscle-memory")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train",
        help="train and qualify a bounded proprioceptive recovery residual",
    )
    train.add_argument("--asset-root", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--source-checkout", type=Path, default=Path.cwd())
    train.add_argument("--population", type=int, default=16)
    train.add_argument("--generations", type=int, default=5)
    train.add_argument("--seed", type=int, default=20260730)
    inspect = commands.add_parser(
        "inspect",
        help="validate a safe JSON artifact and print its commitments",
    )
    inspect.add_argument("artifact", type=Path)
    args = parser.parse_args(argv[2:])
    if args.command == "inspect":
        from rosclaw.simforge.g1_muscle_memory import (
            load_g1_muscle_memory_artifact,
        )

        artifact = load_g1_muscle_memory_artifact(args.artifact)
        print(
            json.dumps(
                {
                    "artifact_hash": artifact.artifact_hash,
                    "body_hash": artifact.body_hash,
                    "motion_hash": artifact.motion_hash,
                    "parent_recovery_config_hash": artifact.parent_recovery_config_hash,
                    "training_dataset_hash": artifact.training_dataset_hash,
                    "training_episode_count": artifact.training_episode_count,
                    "activation_duration_sec": artifact.activation_duration_sec,
                    "sagittal_minimum_impulse_ns": (artifact.sagittal_minimum_impulse_ns),
                    "activation_ceiling": artifact.activation_ceiling,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from rosclaw.simforge.g1_muscle_memory_training import (
        G1MuscleMemoryTrainer,
        G1MuscleMemoryTrainingConfig,
        write_g1_muscle_memory_report,
    )

    trainer = G1MuscleMemoryTrainer(
        asset_root=args.asset_root,
        config=G1MuscleMemoryTrainingConfig(
            population_size=args.population,
            generations=args.generations,
            seed=args.seed,
        ),
    )
    report = trainer.train()
    artifact_path, report_path = write_g1_muscle_memory_report(
        report,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
    )
    value = report.to_dict()
    value["artifact_path"] = str(artifact_path)
    value["report_path"] = str(report_path)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if report.qualified else 2


__all__ = ["dispatch_muscle_memory_argv"]
