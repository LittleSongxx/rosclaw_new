"""Ten-trial, same-regime G1 iterative feed-forward validation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.ilc import (
    BoundedTrajectoryILC,
    ILCFeedforward,
    ILCTrajectory,
    ILCTrajectoryMemory,
    assess_ilc_convergence,
)
from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.feedback.replay import RecordedLatencyClock
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES, ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import generate_goalforge_scenarios

_ILC_SECRET = b"rosclaw-phase6-ilc-validation"
_DETERMINISTIC_LATENCY_NS = 100_000
_CLOCK_CAPACITY = 4_000


@dataclass(frozen=True)
class G1ILCProbe:
    learning_scale: float
    tracking_error_rms: float
    energy_proxy: float
    safety_interventions: int
    status: str
    target_error_m: float
    torso_roll_peak_rad: float
    feedforward_hash: str | None
    eligible: bool


@dataclass(frozen=True)
class G1ILCTrial:
    trial: int
    selected_learning_scale: float
    update_accepted: bool
    receipt_hash: str
    trajectory_hash: str
    feedforward_hash: str | None
    feedforward_peak_rad: float
    tracking_error_rms: float
    energy_proxy: float
    safety_interventions: int
    status: str
    success: bool
    target_error_m: float
    torso_roll_peak_rad: float
    deadline_miss_count: int
    raw_error_path: str
    probes: tuple[G1ILCProbe, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["probes"] = [asdict(probe) for probe in self.probes]
        return value


@dataclass(frozen=True)
class G1ILCValidation:
    body_hash: str
    regime_hash: str
    kick_prior_hash: str
    trials: tuple[G1ILCTrial, ...]
    monotonic_error: bool
    error_reduction: float
    safety_not_increased: bool
    energy_within_limit: bool
    strict_replay: bool
    wrong_regime_rejected: bool
    probe_episode_count: int
    simulation_episode_count: int
    schema_version: str = "rosclaw.g1_ilc.validation.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.trials) >= 10
            and self.monotonic_error
            and self.error_reduction >= 0.01
            and self.safety_not_increased
            and self.energy_within_limit
            and self.strict_replay
            and self.wrong_regime_rejected
            and all(trial.success for trial in self.trials)
            and sum(trial.deadline_miss_count for trial in self.trials) == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "regime_hash": self.regime_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "trials": [trial.to_dict() for trial in self.trials],
            "monotonic_error": self.monotonic_error,
            "error_reduction": self.error_reduction,
            "safety_not_increased": self.safety_not_increased,
            "energy_within_limit": self.energy_within_limit,
            "strict_replay": self.strict_replay,
            "wrong_regime_rejected": self.wrong_regime_rejected,
            "probe_episode_count": self.probe_episode_count,
            "simulation_episode_count": self.simulation_episode_count,
            "passed": self.passed,
            "claims": {
                "evidence_domain": "SIM",
                "real_hardware": False,
                "method": "bounded trial-to-trial joint feedforward with rollback",
                "tracking_error": "RMS(policy joint reference - MuJoCo joint position)",
                "energy_proxy": "sum(joint_torque^2) * 0.02 second control frame",
            },
        }


def run_g1_ilc_validation(
    *,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    trial_count: int = 10,
    learning_scales: tuple[float, ...] = (0.0, 0.25, 1.0, 2.5, 5.0),
) -> G1ILCValidation:
    """Run a transactional ILC campaign and persist selected error trajectories."""

    destination = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination == checkout or checkout in destination.parents:
        raise ValueError("ILC validation evidence must be outside the source checkout")
    if trial_count < 10:
        raise ValueError("ILC validation requires at least ten selected trials")
    if (
        not learning_scales
        or 0.0 not in learning_scales
        or any(value < 0.0 or value > 10.0 for value in learning_scales)
    ):
        raise ValueError("learning scales must include rollback scale 0 and stay in [0, 10]")

    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    scenario = generate_goalforge_scenarios(
        ledger=SeedLedger(task_id="g1_penalty_kick", secret=_ILC_SECRET),
        partition=Partition.VALIDATION,
        count=1,
        generation=0,
    )[0]
    parameters = ShotParameters()
    learner = BoundedTrajectoryILC(
        body_hash=backend.qualification.body_hash,
        regime_hash=scenario.scenario_commitment,
        joint_names=G1_DDS_JOINT_NAMES,
        learning_gain=0.001,
        retention=1.0,
        residual_limit=0.008,
        smoothing_passes=5,
    )
    memory = ILCTrajectoryMemory(
        body_hash=backend.qualification.body_hash,
        regime_hash=scenario.scenario_commitment,
        capacity=trial_count,
    )
    trial_root = destination.parent / (destination.stem + "-trials")
    trial_root.mkdir(parents=True, exist_ok=True)

    feedforward: ILCFeedforward | None = None
    episode = _run_episode(backend, scenario, parameters, feedforward)
    baseline_energy = _energy_proxy(episode)
    trials: list[G1ILCTrial] = []
    probe_count = 0

    for trial_index in range(1, trial_count + 1):
        probes: tuple[G1ILCProbe, ...] = ()
        selected_scale = 0.0
        update_accepted = False
        if trial_index > 1:
            prior_error = _tracking_error(episode)
            prior_receipt_hash = _receipt_hash(episode)
            choices: list[tuple[G1ILCProbe, GoalForgeEpisode, ILCFeedforward | None]] = []
            for scale in learning_scales:
                candidate = feedforward
                if scale > 0.0:
                    candidate = learner.update(
                        previous=feedforward,
                        tracking_error=prior_error,
                        source_receipt_hash=prior_receipt_hash,
                        learning_scale=scale,
                    )
                candidate_episode = _run_episode(backend, scenario, parameters, candidate)
                probe = _probe(
                    candidate_episode,
                    learning_scale=scale,
                    baseline_energy=baseline_energy,
                )
                choices.append((probe, candidate_episode, candidate))
            probe_count += len(choices)
            eligible = [choice for choice in choices if choice[0].eligible]
            selected_probe, episode, selected_feedforward = min(
                eligible,
                key=lambda choice: choice[0].tracking_error_rms,
            )
            update_accepted = bool(
                selected_probe.learning_scale > 0.0
                and selected_probe.tracking_error_rms <= _error_rms(prior_error) - 1e-12
            )
            if update_accepted:
                feedforward = selected_feedforward
            else:
                rollback = next(choice for choice in choices if choice[0].learning_scale == 0.0)
                selected_probe, episode, feedforward = rollback
            selected_scale = selected_probe.learning_scale
            probes = tuple(choice[0] for choice in choices)

        error = _tracking_error(episode)
        receipt_hash = _receipt_hash(episode)
        applied_residual = (
            np.zeros_like(error) if feedforward is None else np.asarray(feedforward.values)
        )
        safety_interventions = _safety_interventions(episode)
        energy = _energy_proxy(episode)
        memory.append(
            ILCTrajectory(
                receipt_hash=receipt_hash,
                body_hash=backend.qualification.body_hash,
                regime_hash=scenario.scenario_commitment,
                tracking_error=error,
                feedforward_residual=applied_residual,
                energy=energy,
                safety_interventions=safety_interventions,
            )
        )
        raw_path = trial_root / f"trial-{trial_index:02d}-error.npz"
        _atomic_npz(
            raw_path,
            tracking_error=error,
            feedforward_residual=applied_residual,
            policy_action=episode.trajectory["policy_action"],
            joint_position=episode.trajectory["joint_position"],
            joint_torque=episode.trajectory["joint_torque"],
        )
        receipt = episode.feedback_receipt
        assert receipt is not None
        trials.append(
            G1ILCTrial(
                trial=trial_index,
                selected_learning_scale=selected_scale,
                update_accepted=update_accepted,
                receipt_hash=receipt_hash,
                trajectory_hash=trajectory_digest(episode.trajectory),
                feedforward_hash=episode.feedforward_hash,
                feedforward_peak_rad=(
                    0.0 if feedforward is None else float(np.max(np.abs(feedforward.values)))
                ),
                tracking_error_rms=_error_rms(error),
                energy_proxy=energy,
                safety_interventions=safety_interventions,
                status=episode.result.status.value,
                success=episode.result.success,
                target_error_m=episode.result.target_error_m,
                torso_roll_peak_rad=episode.result.torso_roll_peak_rad,
                deadline_miss_count=receipt.deadline_miss_count,
                raw_error_path=str(raw_path),
                probes=probes,
            )
        )

    convergence = assess_ilc_convergence(memory.items)
    replay = _run_episode(backend, scenario, parameters, feedforward)
    strict_replay = bool(
        replay.result.summary_dict() == episode.result.summary_dict()
        and trajectory_digest(replay.trajectory) == trajectory_digest(episode.trajectory)
    )
    wrong_regime_rejected = False
    if feedforward is not None:
        wrong_regime = replace(scenario, scenario_id=scenario.scenario_id + "-wrong-regime")
        try:
            backend.run(wrong_regime, parameters, feedforward=feedforward)
        except ValueError as error:
            wrong_regime_rejected = "wrong-regime" in str(error)

    result = G1ILCValidation(
        body_hash=backend.qualification.body_hash,
        regime_hash=scenario.scenario_commitment,
        kick_prior_hash=backend.qualification.kick_prior_hash,
        trials=tuple(trials),
        monotonic_error=convergence.monotonic_error,
        error_reduction=convergence.error_reduction,
        safety_not_increased=convergence.safety_not_increased,
        energy_within_limit=convergence.energy_within_limit,
        strict_replay=strict_replay,
        wrong_regime_rejected=wrong_regime_rejected,
        probe_episode_count=probe_count,
        simulation_episode_count=1 + probe_count + 1,
    )
    _atomic_json(destination, result.to_dict())
    return result


def _run_episode(
    backend: G1MuJoCoBackend,
    scenario: Any,
    parameters: ShotParameters,
    feedforward: ILCFeedforward | None,
) -> GoalForgeEpisode:
    runtime = build_g1_balance_runtime(
        body_hash=backend.qualification.body_hash,
        compute_clock_ns=RecordedLatencyClock((_DETERMINISTIC_LATENCY_NS,) * _CLOCK_CAPACITY),
    )
    return backend.run(
        scenario,
        parameters,
        feedback_runtime=runtime,
        feedforward=feedforward,
    )


def _tracking_error(episode: GoalForgeEpisode) -> np.ndarray:
    return np.asarray(
        episode.trajectory["policy_action"] - episode.trajectory["joint_position"],
        dtype=np.float64,
    )


def _error_rms(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def _energy_proxy(episode: GoalForgeEpisode) -> float:
    torque = np.asarray(episode.trajectory["joint_torque"], dtype=np.float64)
    return float(np.sum(np.square(torque)) * 0.02)


def _safety_interventions(episode: GoalForgeEpisode) -> int:
    result = episode.result
    combined_saturation = episode.trajectory.get("combined_residual_saturation")
    return sum(
        (
            result.post_kick_fall,
            result.joint_limit_violation,
            result.torque_limit_violation,
            result.actuator_saturation,
            bool(combined_saturation is not None and combined_saturation[-1] > 0),
        )
    )


def _receipt_hash(episode: GoalForgeEpisode) -> str:
    if episode.feedback_receipt is None:
        raise ValueError("ILC episode is missing its FeedbackReceipt")
    return episode.feedback_receipt.receipt_hash


def _probe(
    episode: GoalForgeEpisode,
    *,
    learning_scale: float,
    baseline_energy: float,
) -> G1ILCProbe:
    error_rms = _error_rms(_tracking_error(episode))
    energy = _energy_proxy(episode)
    safety = _safety_interventions(episode)
    eligible = bool(episode.result.success and safety == 0 and energy <= baseline_energy * 1.10)
    return G1ILCProbe(
        learning_scale=learning_scale,
        tracking_error_rms=error_rms,
        energy_proxy=energy,
        safety_interventions=safety,
        status=episode.result.status.value,
        target_error_m=episode.result.target_error_m,
        torso_roll_peak_rad=episode.result.torso_roll_peak_rad,
        feedforward_hash=episode.feedforward_hash,
        eligible=eligible,
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


__all__ = [
    "G1ILCProbe",
    "G1ILCTrial",
    "G1ILCValidation",
    "run_g1_ilc_validation",
]
