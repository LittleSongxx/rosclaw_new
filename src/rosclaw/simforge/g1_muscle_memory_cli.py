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
    temporal_train = commands.add_parser(
        "temporal-train",
        help="train and qualify a recurrent proprioceptive recovery residual",
    )
    temporal_train.add_argument("--asset-root", type=Path, required=True)
    temporal_train.add_argument("--output-dir", type=Path, required=True)
    temporal_train.add_argument("--source-checkout", type=Path, default=Path.cwd())
    temporal_train.add_argument("--population", type=int, default=10)
    temporal_train.add_argument("--generations", type=int, default=3)
    temporal_train.add_argument("--seed", type=int, default=20260731)
    temporal_train.add_argument(
        "--cuda-devices",
        type=int,
        nargs="*",
        default=(0, 1, 2, 3),
        help="CUDA devices used only for exported-policy parity, never physics authority",
    )
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
                    "policy_architecture": artifact.policy_architecture,
                    "temporal_basis_count": len(artifact.temporal_basis_centers_sec),
                    "expert_impact_prototypes_ns": list(artifact.expert_impact_prototypes_ns),
                    "expert_impact_max_distance_ns": (artifact.expert_impact_max_distance_ns),
                    "expert_regime_feature_names": list(artifact.expert_regime_feature_names),
                    "expert_regime_prototype_count": len(artifact.expert_regime_prototypes),
                    "expert_regime_max_distance": artifact.expert_regime_max_distance,
                    "structured_recovery_parameters": list(artifact.structured_recovery_parameters),
                    "activation_duration_sec": artifact.activation_duration_sec,
                    "sagittal_minimum_impulse_ns": (artifact.sagittal_minimum_impulse_ns),
                    "activation_ceiling": artifact.activation_ceiling,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "temporal-train":
        from rosclaw.simforge.g1_temporal_muscle_memory_training import (
            G1TemporalMuscleMemoryTrainer,
            G1TemporalMuscleMemoryTrainingConfig,
            write_g1_temporal_muscle_memory_report,
        )

        temporal_trainer = G1TemporalMuscleMemoryTrainer(
            asset_root=args.asset_root,
            config=G1TemporalMuscleMemoryTrainingConfig(
                population_size=args.population,
                generations=args.generations,
                seed=args.seed,
                cuda_devices=tuple(args.cuda_devices),
            ),
        )
        temporal_report = temporal_trainer.train()
        artifact_path, report_path = write_g1_temporal_muscle_memory_report(
            temporal_report,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        value = temporal_report.to_dict()
        value["artifact_path"] = str(artifact_path)
        value["report_path"] = str(report_path)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if temporal_report.qualified else 2

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
