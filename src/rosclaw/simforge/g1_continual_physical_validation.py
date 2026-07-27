"""Real-MuJoCo to four-CUDA continual learner foundation validation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.continual.contracts import ExperiencePartition, PolicyVersion, SkillPhase
from rosclaw.continual.experience import ContinualExperienceStore, ExperienceRecord
from rosclaw.continual.g1_goalforge import (
    adapt_goalforge_episode,
    build_g1_policy_lineage,
)
from rosclaw.continual.serde import experience_batch_from_dict, experience_batch_to_dict
from rosclaw.continual.stability import (
    ContinualCandidateEvidence,
    ContinualDecision,
    StabilityPlasticityGate,
)
from rosclaw.continual.weight_update import ResidualWeightSlot, WeightSlotState
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import generate_goalforge_scenarios

_FOUNDATION_SECRET = b"rosclaw-phase7-physical-continual-foundation-v1"


@dataclass(frozen=True)
class PhysicalRolloutEvidence:
    replay_partition: str
    scenario_id: str
    scenario_commitment: str
    policy_version: int
    policy_version_hash: str
    source_trajectory_hash: str
    versioned_trajectory_hash: str
    self_identity_hash: str
    first_self_state_hash: str
    last_self_state_hash: str
    trace_samples: int
    physics_steps: int
    status: str
    critical_cost: bool
    strict_replay: bool
    artifact_path: str


@dataclass(frozen=True)
class PhysicalLearnerShard:
    physical_gpu: int
    gpu_uuid: str
    candidate_policy_hash: str
    candidate_artifact_hash: str
    batch_hash: str
    updates_finite: bool
    actor_transition_count: int
    critic_transition_count: int
    stale_actor_transition_count: int
    action_bounded: bool
    hidden_activations_finite: bool
    max_memory_allocated_bytes: int
    evidence_path: str
    artifact_path: str


@dataclass(frozen=True)
class G1ContinualPhysicalFoundation:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    rollouts: tuple[PhysicalRolloutEvidence, ...]
    shards: tuple[PhysicalLearnerShard, ...]
    failures: tuple[str, ...]
    gate_report: dict[str, Any]
    stage_receipt: dict[str, Any]
    activation_receipt: dict[str, Any]
    active_policy_unchanged: bool
    schema_version: str = "rosclaw.continual.g1_physical_foundation.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.rollouts) == 4
            and len(self.shards) == 4
            and not self.failures
            and all(item.physics_steps > 0 and item.strict_replay for item in self.rollouts)
            and any(
                item.replay_partition == ExperiencePartition.BOUNDARY.value and item.critical_cost
                for item in self.rollouts
            )
            and len({item.gpu_uuid for item in self.shards}) == 4
            and all(
                item.updates_finite
                and item.actor_transition_count > 0
                and item.critic_transition_count > item.actor_transition_count
                and item.stale_actor_transition_count > 0
                and item.action_bounded
                and item.hidden_activations_finite
                and item.max_memory_allocated_bytes > 0
                for item in self.shards
            )
            and self.gate_report["decision"] == ContinualDecision.NEED_MORE_EVIDENCE.value
            and not self.gate_report["activation_allowed"]
            and self.activation_receipt["state"] == WeightSlotState.FROZEN.value
            and self.active_policy_unchanged
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "backend_commit": self.backend_commit,
            "rollouts": [asdict(item) for item in self.rollouts],
            "shards": [asdict(item) for item in self.shards],
            "failures": list(self.failures),
            "unique_gpu_uuids": len({item.gpu_uuid for item in self.shards}),
            "gate_report": self.gate_report,
            "stage_receipt": self.stage_receipt,
            "activation_receipt": self.activation_receipt,
            "active_policy_unchanged": self.active_policy_unchanged,
            "safe_activation_refusal": bool(
                self.gate_report["decision"] == ContinualDecision.NEED_MORE_EVIDENCE.value
                and self.activation_receipt.get("state") == WeightSlotState.FROZEN.value
                and self.active_policy_unchanged
            ),
            "passed": self.passed,
            "promotion_assessment": {
                "status": ContinualDecision.NEED_MORE_EVIDENCE.value,
                "eligible": False,
                "blockers": [
                    "the learned candidate has not run matched multi-seed G1 evaluation",
                    "historical retention and critical-skill regression evidence is absent",
                    "plasticity and causal SelfCore evidence is absent",
                    "SIM evidence cannot authorize real hardware",
                ],
            },
            "claims": {
                "evidence_domain": "SIM_TRAINING_FOUNDATION",
                "mujoco_physics_transitions": True,
                "four_physical_gpus_exercised": len(self.shards) == 4,
                "candidate_staged": bool(self.stage_receipt),
                "candidate_activated": False,
                "g1_motion_effect_proven": False,
                "consciousness_claimed": False,
                "hardware_authorized": False,
            },
        }


def run_g1_continual_physical_foundation(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    updates_per_gpu: int = 3,
) -> G1ContinualPhysicalFoundation:
    """Collect physics, ingest on four GPUs, then prove fail-closed activation."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("physical continual evidence must be outside the source checkout")
    if not 1 <= updates_per_gpu <= 20:
        raise ValueError("updates_per_gpu must be in [1, 20]")
    root.mkdir(parents=True, exist_ok=False)
    rollout_root = root / "rollouts"
    batch_root = root / "batches"
    learner_root = root / "learners"
    rollout_root.mkdir()
    batch_root.mkdir()
    learner_root.mkdir()

    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=5)
    qualification = backend.qualification
    lineage = build_g1_policy_lineage(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        motion_hash=qualification.motion_hash,
        backend_commit=qualification.backend_commit,
        torque_guard_scale=backend.torque_guard_scale,
        through_version=2,
    )
    scenarios = _foundation_scenarios()
    policy_for = {
        ExperiencePartition.ANCHOR: lineage.policy(0),
        ExperiencePartition.RECENT: lineage.policy(2),
        ExperiencePartition.BOUNDARY: lineage.policy(2),
        ExperiencePartition.SELF: lineage.policy(2),
    }
    store = ContinualExperienceStore()
    rollouts: list[PhysicalRolloutEvidence] = []
    failures: list[str] = []
    parameters = ShotParameters()
    for partition in ExperiencePartition:
        scenario = scenarios[partition]
        episode = backend.run(scenario, parameters)
        replay = backend.run(scenario, parameters)
        strict_replay = bool(
            replay.result.summary_dict() == episode.result.summary_dict()
            and trajectory_digest(replay.trajectory) == trajectory_digest(episode.trajectory)
        )
        policy = policy_for[partition]
        adaptation = adapt_goalforge_episode(
            episode,
            policy=policy,
            strict_replay=strict_replay,
        )
        episode_path = rollout_root / f"{partition.value}.npz"
        np.savez_compressed(episode_path, **episode.trajectory)
        _atomic_json(
            rollout_root / f"{partition.value}.json",
            {
                "scenario": scenario.to_private_dict(),
                "result": episode.result.summary_dict(),
                "source_trajectory_hash": adaptation.source_trajectory_hash,
                "versioned_trajectory_hash": adaptation.trajectory.trajectory_hash,
                "strict_replay": strict_replay,
            },
        )
        record = _record(partition, adaptation.trajectory, scenario.scenario_commitment)
        store.append(record)
        rollout = PhysicalRolloutEvidence(
            replay_partition=partition.value,
            scenario_id=scenario.scenario_id,
            scenario_commitment=scenario.scenario_commitment,
            policy_version=policy.version,
            policy_version_hash=policy.version_hash,
            source_trajectory_hash=adaptation.source_trajectory_hash,
            versioned_trajectory_hash=adaptation.trajectory.trajectory_hash,
            self_identity_hash=adaptation.self_identity_hash,
            first_self_state_hash=adaptation.self_state_hashes[0],
            last_self_state_hash=adaptation.self_state_hashes[-1],
            trace_samples=len(episode.trajectory["time"]),
            physics_steps=episode.result.physics_steps,
            status=episode.result.status.value,
            critical_cost=adaptation.trajectory.has_critical_cost,
            strict_replay=strict_replay,
            artifact_path=str(episode_path),
        )
        rollouts.append(rollout)
        if not strict_replay:
            failures.append(f"{partition.value}:strict replay failed")
    boundary = next(
        item for item in rollouts if item.replay_partition == ExperiencePartition.BOUNDARY.value
    )
    if not boundary.critical_cost:
        failures.append("boundary:80N rollout did not produce a critical safety cost")

    batch_paths: list[Path] = []
    for gpu in range(4):
        batch = store.sample(batch_size=64, learner_version=2, seed=17001 + gpu)
        envelope = experience_batch_to_dict(batch)
        roundtrip = experience_batch_from_dict(envelope)
        if roundtrip.batch_hash != batch.batch_hash:
            failures.append(f"gpu{gpu}:experience codec roundtrip mismatch")
        path = batch_root / f"gpu-{gpu}-experience.json"
        _atomic_json(path, envelope)
        batch_paths.append(path)

    shards, worker_failures = _run_workers(
        checkout=checkout,
        learner_root=learner_root,
        batch_paths=batch_paths,
        updates_per_gpu=updates_per_gpu,
    )
    failures.extend(worker_failures)
    gate_report: dict[str, Any] = {
        "decision": ContinualDecision.NEED_MORE_EVIDENCE.value,
        "activation_allowed": False,
        "checks": [],
    }
    stage_receipt: dict[str, Any] = {}
    activation_receipt: dict[str, Any] = {}
    active_unchanged = False
    if shards:
        selected = next((item for item in shards if item.physical_gpu == 2), shards[0])
        candidate_value = json.loads(Path(selected.evidence_path).read_text(encoding="utf-8"))
        candidate = _policy_from_dict(candidate_value["candidate_policy"])
        artifact = Path(selected.artifact_path).read_bytes()
        parent = lineage.policy(2)
        slot = ResidualWeightSlot(parent, active_artifact=lineage.artifact(2))
        stage = slot.stage(candidate, artifact=artifact)
        stage_receipt = _weight_receipt(stage)
        counts = experience_batch_from_dict(
            json.loads(batch_paths[selected.physical_gpu].read_text(encoding="utf-8"))
        ).requested_counts
        evidence = ContinualCandidateEvidence(
            parent_policy_hash=parent.artifact_hash,
            candidate_policy_hash=candidate.artifact_hash,
            body_hash=candidate.body_hash,
            parent_body_hash=parent.body_hash,
            safety_kernel_hash=candidate.safety_kernel_hash,
            parent_safety_kernel_hash=parent.safety_kernel_hash,
            task_retention=(),
            plasticity=None,
            self_core=None,
            replay_recent_count=counts[ExperiencePartition.RECENT],
            replay_anchor_count=counts[ExperiencePartition.ANCHOR],
            replay_boundary_count=counts[ExperiencePartition.BOUNDARY],
            replay_self_count=counts[ExperiencePartition.SELF],
            anchor_action_drift_rms=0.0,
            critical_safety_regressions=0,
            stale_action_executions=0,
            old_version_replays=0,
            candidate_evaluation_complete=False,
        )
        report = StabilityPlasticityGate().evaluate(evidence)
        gate_report = _gate_report(report)
        activation = slot.activate(phase=SkillPhase.PREPARE, gate_report=report)
        activation_receipt = _weight_receipt(activation)
        active_unchanged = slot.active.version_hash == parent.version_hash
    else:
        failures.append("no CUDA learner candidate was available for staging")

    result = G1ContinualPhysicalFoundation(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        backend_commit=qualification.backend_commit,
        rollouts=tuple(rollouts),
        shards=tuple(sorted(shards, key=lambda item: item.physical_gpu)),
        failures=tuple(failures),
        gate_report=gate_report,
        stage_receipt=stage_receipt,
        activation_receipt=activation_receipt,
        active_policy_unchanged=active_unchanged,
    )
    _atomic_json(root / "physical-continual-summary.json", result.to_dict())
    return result


def _foundation_scenarios():
    ledger = SeedLedger(task_id="g1_penalty_kick", secret=_FOUNDATION_SECRET)
    generated = {
        ExperiencePartition.ANCHOR: generate_goalforge_scenarios(
            ledger=ledger,
            partition=Partition.VALIDATION,
            count=1,
            generation=0,
        )[0],
        ExperiencePartition.RECENT: generate_goalforge_scenarios(
            ledger=ledger,
            partition=Partition.DEVELOPMENT,
            count=1,
            generation=0,
        )[0],
        ExperiencePartition.BOUNDARY: generate_goalforge_scenarios(
            ledger=ledger,
            partition=Partition.COUNTEREXAMPLE_REGRESSION,
            count=1,
            generation=0,
        )[0],
        ExperiencePartition.SELF: generate_goalforge_scenarios(
            ledger=ledger,
            partition=Partition.STRESS,
            count=1,
            generation=0,
        )[0],
    }
    return {
        ExperiencePartition.ANCHOR: replace(
            generated[ExperiencePartition.ANCHOR],
            scenario_id="g1-continual-anchor-nominal",
        ),
        ExperiencePartition.RECENT: replace(
            generated[ExperiencePartition.RECENT],
            scenario_id="g1-continual-recent-35n",
            disturbance_n=35.0,
        ),
        ExperiencePartition.BOUNDARY: replace(
            generated[ExperiencePartition.BOUNDARY],
            scenario_id="g1-continual-boundary-80n",
            disturbance_n=80.0,
        ),
        ExperiencePartition.SELF: replace(
            generated[ExperiencePartition.SELF],
            scenario_id="g1-continual-self-zero-bias",
            joint_zero_bias_rad=0.02,
        ),
    }


def _record(partition, trajectory, scenario_commitment):
    if partition is ExperiencePartition.ANCHOR:
        return ExperienceRecord(
            trajectory,
            partition,
            anchor_policy_hash=trajectory.policy.artifact_hash,
        )
    if partition is ExperiencePartition.BOUNDARY:
        return ExperienceRecord(
            trajectory,
            partition,
            boundary_reason="80N lateral-push critical counterexample",
            near_boundary_score=1.0 if not trajectory.has_critical_cost else 0.0,
        )
    if partition is ExperiencePartition.SELF:
        return ExperienceRecord(
            trajectory,
            partition,
            self_change_hash=scenario_commitment,
        )
    return ExperienceRecord(trajectory, partition)


def _run_workers(*, checkout, learner_root, batch_paths, updates_per_gpu):
    worker = checkout / "scripts/simforge/g1_continual_gpu_worker.py"
    processes = []
    for gpu, batch_path in enumerate(batch_paths):
        output = learner_root / f"gpu-{gpu}-learner.json"
        artifact = learner_root / f"gpu-{gpu}-candidate.bin"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        process = subprocess.Popen(
            [
                sys.executable,
                str(worker),
                "--physical-gpu",
                str(gpu),
                "--updates",
                str(updates_per_gpu),
                "--experience-batch",
                str(batch_path),
                "--input-domain",
                "mujoco_goalforge",
                "--artifact-output",
                str(artifact),
                "--output",
                str(output),
            ],
            cwd=checkout,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((gpu, output, artifact, process))
    shards = []
    failures = []
    for gpu, output, artifact, process in processes:
        try:
            stdout, stderr = process.communicate(timeout=180.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            failures.append(f"gpu{gpu}:timeout:{stdout[-200:]}:{stderr[-600:]}")
            continue
        if process.returncode != 0 or not output.is_file() or not artifact.is_file():
            (learner_root / f"gpu-{gpu}-failure.log").write_text(
                f"exit={process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}\n",
                encoding="utf-8",
            )
            failures.append(f"gpu{gpu}:exit={process.returncode}:{stdout[-200:]}:{stderr[-800:]}")
            continue
        value = json.loads(output.read_text(encoding="utf-8"))
        if not value["claims"]["mujoco_physics_transitions"]:
            failures.append(f"gpu{gpu}:worker did not bind MuJoCo input domain")
            continue
        shards.append(
            PhysicalLearnerShard(
                physical_gpu=gpu,
                gpu_uuid=str(value["gpu_uuid"]),
                candidate_policy_hash=str(value["candidate_policy_hash"]),
                candidate_artifact_hash=str(value["candidate_artifact_hash"]),
                batch_hash=str(value["batch_hash"]),
                updates_finite=bool(value["updates_finite"]),
                actor_transition_count=int(value["actor_transition_count"]),
                critic_transition_count=int(value["critic_transition_count"]),
                stale_actor_transition_count=int(value["stale_actor_transition_count"]),
                action_bounded=bool(value["action_bounded"]),
                hidden_activations_finite=bool(value["hidden_activations_finite"]),
                max_memory_allocated_bytes=int(value["max_memory_allocated_bytes"]),
                evidence_path=str(output),
                artifact_path=str(artifact),
            )
        )
    return shards, failures


def _policy_from_dict(value: dict[str, Any]) -> PolicyVersion:
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


def _gate_report(report):
    return {
        "schema_version": report.schema_version,
        "decision": report.decision.value,
        "checks": [
            {"name": item.name, "status": item.status.value, "detail": item.detail}
            for item in report.checks
        ],
        "parent_policy_hash": report.parent_policy_hash,
        "candidate_policy_hash": report.candidate_policy_hash,
        "rollback_target_hash": report.rollback_target_hash,
        "activation_allowed": report.activation_allowed,
        "evidence_domain": report.evidence_domain,
        "report_hash": report.report_hash,
    }


def _weight_receipt(receipt):
    return {
        "schema_version": receipt.schema_version,
        "state": receipt.state.value,
        "active_version_hash": receipt.active_version_hash,
        "candidate_version_hash": receipt.candidate_version_hash,
        "rollback_version_hash": receipt.rollback_version_hash,
        "reason": receipt.reason,
        "hardware_authorized": receipt.hardware_authorized,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "G1ContinualPhysicalFoundation",
    "PhysicalLearnerShard",
    "PhysicalRolloutEvidence",
    "run_g1_continual_physical_foundation",
]
