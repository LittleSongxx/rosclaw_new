"""GoalForge Hat Trick: target, moving-ball, and disturbance-rescue evidence."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.profiles.g1 import build_g1_balance_runtime
from rosclaw.feedback.replay import RecordedLatencyClock
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    GoalForgeEpisode,
    trajectory_digest,
)
from rosclaw.simforge.g1_moving_ball import MovingBallInterceptAdapter
from rosclaw.simforge.models import Partition
from rosclaw.simforge.seed_ledger import SeedLedger
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters
from rosclaw.simforge.tasks.g1_goalforge.scenario import (
    GoalForgeScenario,
    generate_goalforge_scenarios,
)

_SECRET = b"rosclaw-phase7.1-goalforge-hat-trick-v1"
_CLOCK_CAPACITY = 5000
_LATENCY_NS = 100_000


@dataclass(frozen=True)
class HatTrickShot:
    name: str
    title: str
    capability: str
    scenario: dict[str, Any]
    parameters: dict[str, Any]
    result: dict[str, Any]
    trajectory_path: str
    trajectory_hash: str
    strict_replay: bool
    feedback_receipt: dict[str, Any] | None = None
    comparison_result: dict[str, Any] | None = None
    comparison_trajectory_path: str | None = None
    comparison_trajectory_hash: str | None = None
    planner_receipt: dict[str, Any] | None = None


@dataclass(frozen=True)
class GoalForgeHatTrick:
    body_hash: str
    kick_prior_hash: str
    backend_commit: str
    shots: tuple[HatTrickShot, ...]
    schema_version: str = "rosclaw.g1_goalforge.hat_trick.v1"

    @property
    def passed(self) -> bool:
        if len(self.shots) != 3:
            return False
        target, moving, rescue = self.shots
        return bool(
            all(shot.result["success"] and shot.strict_replay for shot in self.shots)
            and target.result["target_zone_hit"]
            and target.result["ball_speed_mps"] >= 6.0
            and moving.scenario["ball_launch_delay_sec"] > 0.0
            and moving.result["kick_foot_contacted"]
            and moving.result["ball_contact_time_sec"] is not None
            and rescue.scenario["disturbance_n"] >= 80.0
            and rescue.comparison_result is not None
            and not rescue.comparison_result["success"]
            and not any(
                shot.result[name]
                for shot in self.shots
                for name in (
                    "post_kick_fall",
                    "joint_limit_violation",
                    "torque_limit_violation",
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "body_hash": self.body_hash,
            "kick_prior_hash": self.kick_prior_hash,
            "backend_commit": self.backend_commit,
            "shots": [asdict(shot) for shot in self.shots],
            "passed": self.passed,
            "visualization_may_consume_evidence": self.passed,
            "claims": {
                "evidence_domain": "SIM",
                "static_nine_grid_shot": self.shots[0].result["success"],
                "moving_ball_first_time_shot": self.shots[1].result["success"],
                "disturbance_feedback_rescue": self.shots[2].result["success"],
                "candidate_v3_promoted": False,
                "magnus_curve_claimed": False,
                "real_hardware": False,
            },
        }


def run_goalforge_hat_trick(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> GoalForgeHatTrick:
    """Execute and strictly replay all three SIM-only flagship shots."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("Hat Trick evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    qualification = backend.qualification
    base = _base_scenario()
    selection_seed = 202607289
    target_y, target_z = random.Random(selection_seed).choice(
        [(y, z) for y in (-0.75, 0.0, 0.75) for z in (0.20, 0.55, 0.90)]
    )
    target_scenario = replace(
        base,
        scenario_id="goalforge-hat-trick-nine-grid",
        generation=4,
        ball_y_m=0.10,
        target_y_m=target_y,
        target_z_m=target_z,
    )
    target_parameters = ShotParameters(
        stance_offset_y=0.12,
        pelvis_yaw_offset=-0.20,
        foot_yaw_offset=-0.0595,
        policy_type="parameter",
    )
    target = _run_strict_pair(
        backend=backend,
        scenario=target_scenario,
        parameters=target_parameters,
    )
    target_path = _save_trajectory(root / "shot-1-nine-grid.npz", target[0])

    moving_scenario = replace(
        base,
        scenario_id="goalforge-hat-trick-moving-ball",
        ball_x_m=1.12,
        ball_y_m=0.0,
        ball_velocity_x_mps=-0.08,
        ball_velocity_y_mps=0.0,
        ball_launch_delay_sec=4.0,
        ball_ground_friction=0.03,
        target_y_m=0.0,
        target_z_m=0.20,
    )
    moving_plan = MovingBallInterceptAdapter().plan(moving_scenario)
    if not moving_plan.eligible:
        raise RuntimeError("Hat Trick moving-ball scenario is outside the adapter envelope")
    moving_parameters = moving_plan.parameters
    moving = _run_strict_pair(
        backend=backend,
        scenario=moving_scenario,
        parameters=moving_parameters,
    )
    moving_path = _save_trajectory(root / "shot-2-moving-ball.npz", moving[0])

    rescue_scenario = replace(
        base,
        scenario_id="goalforge-hat-trick-80n-rescue",
        disturbance_n=80.0,
    )
    rescue_parameters = ShotParameters()
    rescue_baseline = backend.run(rescue_scenario, rescue_parameters)
    runtime = build_g1_balance_runtime(
        body_hash=qualification.body_hash,
        compute_clock_ns=RecordedLatencyClock((_LATENCY_NS,) * _CLOCK_CAPACITY),
    )
    rescue_episode = backend.run(
        rescue_scenario,
        rescue_parameters,
        feedback_runtime=runtime,
    )
    replay_runtime = build_g1_balance_runtime(
        body_hash=qualification.body_hash,
        compute_clock_ns=RecordedLatencyClock((_LATENCY_NS,) * _CLOCK_CAPACITY),
    )
    rescue_replay = backend.run(
        rescue_scenario,
        rescue_parameters,
        feedback_runtime=replay_runtime,
    )
    rescue_strict = bool(
        rescue_replay.result.summary_dict() == rescue_episode.result.summary_dict()
        and trajectory_digest(rescue_replay.trajectory)
        == trajectory_digest(rescue_episode.trajectory)
    )
    feedback_receipt = runtime.build_receipt(
        action_id="goalforge-hat-trick-80n-rescue",
        strict_replay=rescue_strict,
        evidence_domain="SIM",
    )
    rescue_path = _save_trajectory(root / "shot-3-feedback-on.npz", rescue_episode)
    rescue_off_path = _save_trajectory(root / "shot-3-feedback-off.npz", rescue_baseline)

    shots = (
        _shot(
            name="nine_grid_power",
            title="SHOT 1 · 3×3 TARGET POWER",
            capability="precision_and_power",
            scenario=target_scenario,
            parameters=target_parameters,
            episode=target[0],
            path=target_path,
            strict=target[1],
            planner_receipt={
                "schema_version": "rosclaw.g1_goalforge.nine_grid_selection.v1",
                "selection_seed": selection_seed,
                "selection_seed_commitment": "sha256:"
                + hashlib.sha256(str(selection_seed).encode()).hexdigest(),
                "selected_target_y_m": target_y,
                "selected_target_z_m": target_z,
                "candidate_zones": 9,
            },
        ),
        _shot(
            name="moving_ball_first_time",
            title="SHOT 2 · MOVING BALL FIRST-TIME",
            capability="moving_ball_intercept_adapter",
            scenario=moving_scenario,
            parameters=moving_parameters,
            episode=moving[0],
            path=moving_path,
            strict=moving[1],
            planner_receipt=moving_plan.to_dict(),
        ),
        _shot(
            name="disturbance_feedback_rescue",
            title="SHOT 3 · 80 N FEEDBACK RESCUE",
            capability="balance_and_disturbance_recovery",
            scenario=rescue_scenario,
            parameters=rescue_parameters,
            episode=rescue_episode,
            path=rescue_path,
            strict=rescue_strict,
            feedback_receipt=feedback_receipt.to_dict(),
            comparison_result=rescue_baseline.result.summary_dict(),
            comparison_path=rescue_off_path,
        ),
    )
    result = GoalForgeHatTrick(
        body_hash=qualification.body_hash,
        kick_prior_hash=qualification.kick_prior_hash,
        backend_commit=qualification.backend_commit,
        shots=shots,
    )
    _atomic_json(root / "goalforge-hat-trick.json", result.to_dict())
    return result


def _base_scenario() -> GoalForgeScenario:
    return generate_goalforge_scenarios(
        ledger=SeedLedger(task_id="g1_penalty_kick", secret=_SECRET),
        partition=Partition.VALIDATION,
        count=1,
        generation=0,
    )[0]


def _run_strict_pair(
    *,
    backend: G1MuJoCoBackend,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
) -> tuple[GoalForgeEpisode, bool]:
    episode = backend.run(scenario, parameters)
    replay = backend.run(scenario, parameters)
    return episode, bool(
        replay.result.summary_dict() == episode.result.summary_dict()
        and trajectory_digest(replay.trajectory) == trajectory_digest(episode.trajectory)
    )


def _save_trajectory(path: Path, episode: GoalForgeEpisode) -> Path:
    np.savez_compressed(path, **episode.trajectory)
    return path


def _shot(
    *,
    name: str,
    title: str,
    capability: str,
    scenario: GoalForgeScenario,
    parameters: ShotParameters,
    episode: GoalForgeEpisode,
    path: Path,
    strict: bool,
    feedback_receipt: dict[str, Any] | None = None,
    comparison_result: dict[str, Any] | None = None,
    comparison_path: Path | None = None,
    planner_receipt: dict[str, Any] | None = None,
) -> HatTrickShot:
    return HatTrickShot(
        name=name,
        title=title,
        capability=capability,
        scenario=scenario.to_private_dict(),
        parameters=parameters.to_dict(),
        result=episode.result.summary_dict(),
        trajectory_path=str(path),
        trajectory_hash=_file_hash(path),
        strict_replay=strict,
        feedback_receipt=feedback_receipt,
        comparison_result=comparison_result,
        comparison_trajectory_path=str(comparison_path) if comparison_path else None,
        comparison_trajectory_hash=_file_hash(comparison_path) if comparison_path else None,
        planner_receipt=planner_receipt,
    )


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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


__all__ = ["GoalForgeHatTrick", "HatTrickShot", "run_goalforge_hat_trick"]
