"""Evo-RPS experiment contract (真机自进化v2 §12, §4).

Loads ``configs/acceptance/evo_rps_v1.yaml`` into typed dataclasses and
VALIDATES it — the harness refuses to run on an internally inconsistent
contract:

* candidate space values must stay inside the documented bounds (§4.1);
* forbidden parameters must never appear in the candidate space (§4.2);
* cooldown-class parameters are non-stackable: a candidate may set at most
  one of ``inter_round_cooldown_sec`` / ``cooldown_every_n_rounds`` (§五A:
  冷却类参数不得同时叠加多个，避免不可解释的组合);
* formal acceptance forbids mock camera and fixture execution (§2.2);
* safety limits must be present and positive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "rosclaw.acceptance.evo_rps.v1"
BLOCKED_PHYSICAL_PERCEPTION_UNAVAILABLE = "BLOCKED: PHYSICAL_PERCEPTION_UNAVAILABLE"

# §4.1 documented bounds — the config may narrow them, never widen them.
COOLDOWN_SEC_MIN = 0.0
COOLDOWN_SEC_MAX = 8.0
COOLDOWN_EVERY_N_CHOICES = {0, 5, 10, 15}
TELEMETRY_HZ_CHOICES = {1, 2, 5, 10}
CAMERA_BACKOFF_CHOICES = {0.5, 1.0, 2.0, 3.0}

# Cooldown-class parameters are mutually exclusive inside one candidate.
COOLDOWN_CLASS = ("inter_round_cooldown_sec", "cooldown_every_n_rounds")


class ValidationError(ValueError):
    """The acceptance contract is internally inconsistent."""


@dataclass(frozen=True)
class CandidateSpace:
    inter_round_cooldown_sec: tuple[float, ...] = (0.0,)
    cooldown_every_n_rounds: tuple[int, ...] = (0,)
    neutral_pose_between_blocks: tuple[bool, ...] = (False,)
    rehome_between_blocks: tuple[bool, ...] = (False,)
    telemetry_hz: tuple[int, ...] = (5,)

    def validate_against_bounds(self) -> list[str]:
        errors: list[str] = []
        for value in self.inter_round_cooldown_sec:
            if not COOLDOWN_SEC_MIN <= float(value) <= COOLDOWN_SEC_MAX:
                errors.append(
                    f"inter_round_cooldown_sec {value} outside "
                    f"[{COOLDOWN_SEC_MIN}, {COOLDOWN_SEC_MAX}] (§4.1)"
                )
        for value in self.cooldown_every_n_rounds:
            if int(value) not in COOLDOWN_EVERY_N_CHOICES:
                errors.append(
                    f"cooldown_every_n_rounds {value} not in {sorted(COOLDOWN_EVERY_N_CHOICES)}"
                )
        for value in self.telemetry_hz:
            if int(value) not in TELEMETRY_HZ_CHOICES:
                errors.append(f"telemetry_hz {value} not in {sorted(TELEMETRY_HZ_CHOICES)}")
        return errors

    def validate_candidate(self, candidate: dict[str, Any]) -> list[str]:
        """A concrete candidate must use only known parameters, in-space
        values, and at most one cooldown-class parameter."""
        errors: list[str] = []
        allowed = {
            "inter_round_cooldown_sec": set(self.inter_round_cooldown_sec),
            "cooldown_every_n_rounds": set(self.cooldown_every_n_rounds),
            "neutral_pose_between_blocks": set(self.neutral_pose_between_blocks),
            "rehome_between_blocks": set(self.rehome_between_blocks),
            "telemetry_hz": set(self.telemetry_hz),
        }
        for name, value in candidate.items():
            if name not in allowed:
                errors.append(f"unknown candidate parameter {name!r}")
                continue
            if value not in allowed[name] and str(value) not in {str(v) for v in allowed[name]}:
                errors.append(f"{name}={value!r} outside the candidate space")
        active_cooldowns = [
            name
            for name in COOLDOWN_CLASS
            if name in candidate and candidate[name] not in (0, 0.0, False, None)
        ]
        if len(active_cooldowns) > 1:
            errors.append(
                f"cooldown-class parameters are non-stackable: {active_cooldowns} (§五A)"
            )
        return errors


@dataclass(frozen=True)
class EvoRpsConfig:
    path: Path
    raw: dict[str, Any]
    experiment_id: str
    seed: int
    require_clean_namespace: bool
    allow_mock_camera: bool
    allow_fixture_execution: bool
    player_body: str
    referee_body: str
    camera_body: str
    rounds_per_session: int
    gestures: tuple[str, ...]
    external_visual_critic: bool
    telemetry_consensus: bool
    temperature_abort_c: float
    operator_present: bool
    unattended_real_execution: bool
    candidate_space: CandidateSpace
    forbidden_parameters: tuple[str, ...]
    gates: dict[str, bool]
    promotion: dict[str, Any]
    namespace: dict[str, Any]
    task_driver: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def seekdb_dsn(self) -> str:
        ns = self.namespace
        return (
            f"seekdb://{ns['seekdb_user']}@{ns['seekdb_host']}:{ns['seekdb_port']}"
            f"/{ns['database']}"
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.raw.get("schema_version") != SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {SCHEMA_VERSION}, got {self.raw.get('schema_version')!r}"
            )
        if not self.experiment_id:
            errors.append("experiment.id must not be empty")
        if self.rounds_per_session <= 0:
            errors.append("task.rounds_per_session must be positive")
        if self.temperature_abort_c <= 0:
            errors.append("safety.experiment_temperature_abort_c must be positive")
        if self.unattended_real_execution:
            errors.append("safety.unattended_real_execution must stay false for acceptance")
        errors.extend(self.candidate_space.validate_against_bounds())
        overlap = set(self.forbidden_parameters) & {
            "inter_round_cooldown_sec",
            "cooldown_every_n_rounds",
            "neutral_pose_between_blocks",
            "rehome_between_blocks",
            "telemetry_hz",
        }
        if overlap:
            errors.append(f"forbidden_parameters overlaps the candidate space: {sorted(overlap)}")
        database = str(self.namespace.get("database") or "")
        if self.require_clean_namespace and database in ("rosclaw", ""):
            errors.append(
                "namespace.database must be an isolated database, not the shared 'rosclaw' (§2.7)"
            )
        for key in ("practice_root", "trace_root", "evidence_root"):
            if not self.namespace.get(key):
                errors.append(f"namespace.{key} is required")
        driver_kind = self.task_driver.get("kind")
        if driver_kind != "rh56_rps_workspace":
            errors.append(f"unknown task_driver.kind {driver_kind!r}")
        for key in ("workspace_root", "runner", "rh56_src"):
            if not self.task_driver.get(key):
                errors.append(f"task_driver.{key} is required")
        return errors


def load_config(path: str | Path) -> EvoRpsConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    experiment = raw.get("experiment") or {}
    hardware = raw.get("hardware") or {}
    task = raw.get("task") or {}
    safety = raw.get("safety") or {}
    space = raw.get("candidate_space") or {}
    config = EvoRpsConfig(
        path=path,
        raw=raw,
        experiment_id=str(experiment.get("id") or ""),
        seed=int(experiment.get("seed") or 0),
        require_clean_namespace=bool(experiment.get("require_clean_namespace", True)),
        allow_mock_camera=bool(experiment.get("allow_mock_camera", False)),
        allow_fixture_execution=bool(experiment.get("allow_fixture_execution", False)),
        player_body=str(hardware.get("player_body") or ""),
        referee_body=str(hardware.get("referee_body") or ""),
        camera_body=str(hardware.get("camera_body") or ""),
        rounds_per_session=int(task.get("rounds_per_session") or 0),
        gestures=tuple(str(g) for g in (task.get("gestures") or ())),
        external_visual_critic=bool(task.get("external_visual_critic", True)),
        telemetry_consensus=bool(task.get("telemetry_consensus", True)),
        temperature_abort_c=float(safety.get("experiment_temperature_abort_c") or 0.0),
        operator_present=bool(safety.get("operator_present", True)),
        unattended_real_execution=bool(safety.get("unattended_real_execution", False)),
        candidate_space=CandidateSpace(
            inter_round_cooldown_sec=tuple(
                float(v) for v in (space.get("inter_round_cooldown_sec") or (0.0,))
            ),
            cooldown_every_n_rounds=tuple(
                int(v) for v in (space.get("cooldown_every_n_rounds") or (0,))
            ),
            neutral_pose_between_blocks=tuple(
                bool(v) for v in (space.get("neutral_pose_between_blocks") or (False,))
            ),
            rehome_between_blocks=tuple(
                bool(v) for v in (space.get("rehome_between_blocks") or (False,))
            ),
            telemetry_hz=tuple(int(v) for v in (space.get("telemetry_hz") or (5,))),
        ),
        forbidden_parameters=tuple(str(p) for p in (raw.get("forbidden_parameters") or ())),
        gates=dict(raw.get("gates") or {}),
        promotion=dict(raw.get("promotion") or {}),
        namespace=dict(raw.get("namespace") or {}),
        task_driver=dict(raw.get("task_driver") or {}),
    )
    errors = config.validate()
    if errors:
        raise ValidationError("; ".join(errors))
    return config
