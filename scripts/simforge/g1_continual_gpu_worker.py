#!/usr/bin/env python3
"""One isolated CUDA worker for the Phase 7 residual-SAC systems smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rosclaw.continual.contracts import (
    ControlSegment,
    CostVector,
    ExperiencePartition,
    PolicyVersion,
    RewardVector,
    SkillPhase,
    VersionedTrajectory,
)
from rosclaw.continual.experience import ContinualExperienceStore, ExperienceRecord
from rosclaw.continual.g1_goalforge import (
    G1_CONTINUAL_ACTION_LIMITS,
    G1_CONTINUAL_ACTIONS,
    G1_CONTINUAL_OBSERVATIONS,
)
from rosclaw.continual.learner import ConstrainedResidualSAC, ResidualSACConfig
from rosclaw.continual.serde import experience_batch_from_dict

OBSERVATIONS = G1_CONTINUAL_OBSERVATIONS
ACTIONS = G1_CONTINUAL_ACTIONS
ACTION_LIMITS = G1_CONTINUAL_ACTION_LIMITS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experience-batch", type=Path)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument(
        "--input-domain",
        choices=("synthetic_contract_fixture", "mujoco_goalforge"),
        default="synthetic_contract_fixture",
    )
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--learner-seed", type=int)
    args = parser.parse_args()
    if args.physical_gpu not in range(4) or not 1 <= args.updates <= 20:
        raise SystemExit("invalid continual CUDA worker request")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise SystemExit(f"CUDA identity mismatch: expected {args.physical_gpu}, visible={visible}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("continual CUDA worker requires exactly one visible GPU")

    if args.input_domain == "mujoco_goalforge" and args.experience_batch is None:
        raise SystemExit("MuJoCo worker input requires --experience-batch")
    if args.experience_batch is not None:
        batch = experience_batch_from_dict(
            json.loads(args.experience_batch.read_text(encoding="utf-8"))
        )
        parent = _learner_parent(batch)
    else:
        parent, batch = _screening_batch(args.physical_gpu)
    gpu_uuid, pci_bus_id = _gpu_identity(args.physical_gpu)
    learner_seed = args.learner_seed if args.learner_seed is not None else 7001 + args.physical_gpu
    if learner_seed < 0:
        raise SystemExit("learner seed must be non-negative")
    learner = ConstrainedResidualSAC(
        ResidualSACConfig(
            observation_names=OBSERVATIONS,
            action_names=ACTIONS,
            action_limits=ACTION_LIMITS,
            hidden_dims=(64, 64),
            batch_size=32,
            device="cuda:0",
            seed=learner_seed,
        )
    )
    started = time.perf_counter()
    updates = [learner.update(batch) for _ in range(args.updates)]
    action = learner.action(dict(batch.actor_records[0].trajectory.segments[0].observation))
    reference = _reference_observations(batch)
    hidden = learner.hidden_activations(reference)
    candidate, artifact = learner.candidate_policy(parent=parent)
    if args.artifact_output is not None:
        args.artifact_output.write_bytes(artifact)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    bounded = all(
        abs(action[name]) <= limit + 1e-7
        for name, limit in zip(ACTIONS, ACTION_LIMITS, strict=True)
    )
    value = {
        "schema_version": "rosclaw.continual.cuda_worker.v1",
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": visible,
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_hash": batch.batch_hash,
        "parent_policy_hash": parent.version_hash,
        "candidate_policy_hash": candidate.version_hash,
        "candidate_artifact_hash": candidate.artifact_hash,
        "candidate_policy": candidate.to_dict(),
        "candidate_version": candidate.version,
        "artifact_bytes": len(artifact),
        "update_count": len(updates),
        "learner_seed": learner_seed,
        "updates_finite": all(update.finite for update in updates),
        "critic_transition_count": updates[-1].critic_transition_count,
        "actor_transition_count": updates[-1].actor_transition_count,
        "stale_actor_transition_count": updates[-1].stale_actor_transition_count,
        "final_losses": {
            "critic": updates[-1].critic_loss,
            "fall_critic": updates[-1].fall_critic_loss,
            "constraint_critic": updates[-1].constraint_critic_loss,
            "actor": updates[-1].actor_loss,
            "anchor": updates[-1].anchor_loss,
            "churn": updates[-1].churn_loss,
            "step_churn": updates[-1].step_churn,
        },
        "action_bounded": bounded,
        "hidden_activations_finite": all(np.isfinite(item).all() for item in hidden),
        "hidden_activation_shapes": [list(item.shape) for item in hidden],
        "elapsed_ms": elapsed_ms,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "claims": {
            "input_domain": args.input_domain,
            "synthetic_versioned_transition_screening": (
                args.input_domain == "synthetic_contract_fixture"
            ),
            "mujoco_physics_transitions": args.input_domain == "mujoco_goalforge",
            "g1_motion_effect_proven": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
        },
    }
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _learner_parent(batch):
    matches = {
        record.trajectory.policy.version_hash: record.trajectory.policy
        for record in batch.records
        if record.trajectory.policy.version == batch.learner_version
    }
    if len(matches) != 1:
        raise ValueError("physical replay must identify exactly one current learner parent")
    return next(iter(matches.values()))


def _reference_observations(batch) -> np.ndarray:
    rows = [
        [segment.observation[name] for name in OBSERVATIONS]
        for record in batch.records
        for segment in record.trajectory.segments
    ]
    if not rows:
        raise ValueError("experience batch does not contain reference observations")
    value = np.asarray(rows, dtype=np.float32)
    if len(value) >= 64:
        return value[:64]
    return np.resize(value, (64, len(OBSERVATIONS)))


def _screening_batch(gpu: int):
    v0 = _policy(0, artifact=f"g1-parent-v0-gpu{gpu}".encode())
    v1 = _policy(1, artifact=f"g1-parent-v1-gpu{gpu}".encode(), parent=v0)
    v2 = _policy(2, artifact=f"g1-parent-v2-gpu{gpu}".encode(), parent=v1)
    store = ContinualExperienceStore()
    store.append(
        ExperienceRecord(
            _trajectory(v2, episode=f"gpu{gpu}-recent", delta=0.01 * gpu),
            ExperiencePartition.RECENT,
        )
    )
    store.append(
        ExperienceRecord(
            _trajectory(v0, episode=f"gpu{gpu}-anchor", delta=-0.01),
            ExperiencePartition.ANCHOR,
            anchor_policy_hash=v0.artifact_hash,
        )
    )
    store.append(
        ExperienceRecord(
            _trajectory(
                v2,
                episode=f"gpu{gpu}-boundary",
                delta=0.03,
                critical=True,
            ),
            ExperiencePartition.BOUNDARY,
            boundary_reason="synthetic-fall-boundary-screen",
        )
    )
    store.append(
        ExperienceRecord(
            _trajectory(v2, episode=f"gpu{gpu}-self", delta=0.02),
            ExperiencePartition.SELF,
            self_change_hash=_digest(f"gpu{gpu}-payload-change".encode()),
        )
    )
    return v2, store.sample(batch_size=64, learner_version=v2.version, seed=9001 + gpu)


def _policy(
    version: int,
    *,
    artifact: bytes,
    parent: PolicyVersion | None = None,
) -> PolicyVersion:
    return PolicyVersion(
        version=version,
        artifact_hash=_digest(artifact),
        parent_version_hash=parent.version_hash if parent else None,
        controller_snapshot_hash=_digest(b"g1-cerebellum-controller-v1"),
        body_hash=_digest(b"qualified-g1-screening-body"),
        safety_kernel_hash=_digest(b"immutable-g1-safety-kernel"),
        observation_names=OBSERVATIONS,
        residual_action_names=ACTIONS,
    )


def _trajectory(
    policy: PolicyVersion,
    *,
    episode: str,
    delta: float,
    critical: bool = False,
) -> VersionedTrajectory:
    segments = (
        _segment(policy, episode=episode, start=0, phase=SkillPhase.PREPARE, delta=delta),
        _segment(
            policy,
            episode=episode,
            start=1,
            phase=SkillPhase.RECOVERY,
            delta=delta + 0.01,
            critical=critical,
            terminal=True,
        ),
    )
    return VersionedTrajectory(segments=segments, strict_replay=True)


def _segment(
    policy: PolicyVersion,
    *,
    episode: str,
    start: int,
    phase: SkillPhase,
    delta: float,
    critical: bool = False,
    terminal: bool = False,
) -> ControlSegment:
    observation = {
        "torso_roll": 0.10 + delta,
        "torso_pitch": -0.08 + delta,
        "com_y_relative": 0.02 + delta,
        "support_slip_m": max(0.0, 0.01 + delta),
        "ball_lateral_error_m": 0.04 - delta,
        "contact_phase": 0.35 + 0.1 * start,
        "energy_margin": 0.8 - delta,
        "sensor_quality": 0.95,
    }
    next_observation = dict(observation)
    next_observation["torso_roll"] *= 0.9
    next_observation["torso_pitch"] *= 0.9
    action = {
        "waist_roll_residual": -0.01,
        "right_hip_roll_residual": -0.02,
        "right_hip_yaw_residual": 0.005,
        "kick_phase_rate": 0.01,
    }
    return ControlSegment(
        segment_id=f"{episode}:{start}",
        episode_id=episode,
        task_id="g1_penalty_kick",
        phase=phase,
        start_step=start,
        end_step=start + 1,
        policy=policy,
        controller_snapshot_hash=policy.controller_snapshot_hash,
        body_hash=policy.body_hash,
        regime_hash=_digest(b"cuda-screen-regime"),
        self_state_hash=_digest(f"{episode}-self-state".encode()),
        observation=observation,
        residual_action=action,
        next_observation=next_observation,
        behavior_logprob=-0.5,
        reward=RewardVector(task=1.0, tracking=0.2, balance=0.3, learning=0.1),
        cost=CostVector(
            fall=1.0 if critical else 0.0,
            joint_limit=0.25 if critical else 0.0,
            energy=0.1,
        ),
        terminal=terminal,
    )


def _gpu_identity(index: int) -> tuple[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid, pci_bus_id = (item.strip() for item in output.split(",", maxsplit=1))
    return uuid, pci_bus_id


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
