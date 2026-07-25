"""Session driver adapter (PR-EVO-HW-1 §11 task_driver).

The acceptance machinery lives in this repo; the RH56 RPS task engine
lives in the workspace demo project.  The driver binds the two with a
pinned path + content hash so evidence always names the exact code that
produced it (§2.8).  Each session runs with the experiment's own practice
root via the ``RPS_PRACTICE_DATA_ROOT`` override — the shared workspace
config is never mutated.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import EvoRpsConfig
from .evidence import file_sha256

VENV_PY = "/home/nvidia/workspace/rosclaw/rosclaw_test/.venv/bin/python"


class DriverError(RuntimeError):
    pass


@dataclass
class DriverResult:
    practice_id: str | None
    rounds: int
    summary: dict[str, Any]
    log_path: Path


class Rh56RpsWorkspaceDriver:
    """Runs one RPS session through the workspace stress harness."""

    def __init__(self, config: EvoRpsConfig, practice_root: Path) -> None:
        self._config = config
        self._practice_root = practice_root
        driver = config.task_driver
        self._workspace = Path(str(driver["workspace_root"]))
        self._runner = self._workspace / str(driver["runner"])
        self._rh56_src = str(driver["rh56_src"])
        if not self._runner.is_file():
            raise DriverError(f"task runner not found: {self._runner}")

    def code_hash(self) -> dict[str, str]:
        return {
            "runner": file_sha256(self._runner),
            "stress_closed_loop": file_sha256(
                self._workspace / "scripts" / "stress_closed_loop.py"
            ),
        }

    def run_session(
        self,
        *,
        group: str,
        seed: int,
        rounds: int,
        camera_source: str,
        out_dir: Path,
        timeout_s: int | None = None,
    ) -> DriverResult:
        if camera_source == "mock" and not self._config.allow_mock_camera:
            raise DriverError(
                "mock camera requested but the contract forbids it "
                "(§2.2: BLOCKED: PHYSICAL_PERCEPTION_UNAVAILABLE)"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"{group}_{seed}.json"
        log_path = out_dir / f"{group}_{seed}.log"
        env = {
            **os.environ,
            "RPS_RH56_SRC": self._rh56_src,
            "RPS_PRACTICE_DATA_ROOT": str(self._practice_root),
            "PYTHONPATH": (
                f"{self._workspace}/scripts:{self._workspace}/scripts/experiments:"
                f"{self._workspace}/src:"
                f"/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src:"
                f"{self._rh56_src}"
            ),
        }
        cmd = [
            VENV_PY,
            str(self._runner),
            "--group", group,
            "--seed", str(seed),
            "--rounds", str(rounds),
            "--camera-source", camera_source,
            "--out", str(out_json),
        ]
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(self._workspace),
                timeout=timeout_s or (60 * (rounds + 12)),
            )
        summary: dict[str, Any] = {}
        if out_json.is_file():
            summary = json.loads(out_json.read_text())
        if proc.returncode != 0:
            raise DriverError(
                f"session {group}/{seed} exited rc={proc.returncode}; see {log_path}"
            )
        practice_id = self._latest_practice_id()
        return DriverResult(
            practice_id=practice_id,
            rounds=int(summary.get("rounds", 0)),
            summary=summary,
            log_path=log_path,
        )

    def run_canary(
        self,
        *,
        candidate_id: str,
        candidate_params: dict[str, Any],
        seed: int,
        rounds: int,
        out_dir: Path,
        timeout_s: int | None = None,
    ) -> DriverResult:
        """Arm-C canary: the candidate applied mechanically on REAL hardware
        (§Phase 6/7 operator-approved canary).  Real camera, real hands."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"canary_{candidate_id}_{seed}.json"
        log_path = out_dir / f"canary_{candidate_id}_{seed}.log"
        env = {
            **os.environ,
            "RPS_RH56_SRC": self._rh56_src,
            "RPS_PRACTICE_DATA_ROOT": str(self._practice_root),
            "PYTHONPATH": (
                f"{self._workspace}/scripts:{self._workspace}/scripts/experiments:"
                f"{self._workspace}/src:"
                f"/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src:"
                f"{self._rh56_src}"
            ),
        }
        cmd = [
            VENV_PY,
            str(self._runner),
            "--group", "candidate_canary",
            "--seed", str(seed),
            "--rounds", str(rounds),
            "--camera-source", "realsense",
            "--candidate-params", json.dumps(candidate_params),
            "--out", str(out_json),
        ]
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(self._workspace),
                timeout=timeout_s or (60 * (rounds + 15)),
            )
        summary: dict[str, Any] = {}
        if out_json.is_file():
            summary = json.loads(out_json.read_text())
        if proc.returncode != 0:
            raise DriverError(
                f"canary session seed={seed} exited rc={proc.returncode}; see {log_path}"
            )
        return DriverResult(
            practice_id=self._latest_practice_id(),
            rounds=int(summary.get("rounds", 0)),
            summary=summary,
            log_path=log_path,
        )

    def run_shadow(
        self,
        *,
        candidate_id: str,
        candidate_params: dict[str, Any],
        seed: int,
        rounds: int,
        out_dir: Path,
        timeout_s: int | None = None,
    ) -> dict[str, Any]:
        """L2 shadow: the candidate runs the REAL task code path with mock
        executors — zero hardware actions by construction (§7.14).

        Shadow uses mock hands + mock camera BY DESIGN: it validates the
        candidate parameter lifecycle and timing, and makes no perception
        or hardware claims.  Formal acceptance runs never use mocks.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"shadow_{candidate_id}.json"
        log_path = out_dir / f"shadow_{candidate_id}.log"
        env = {
            **os.environ,
            "RPS_RH56_SRC": self._rh56_src,
            "RPS_PRACTICE_DATA_ROOT": str(self._practice_root),
            "PYTHONPATH": (
                f"{self._workspace}/scripts:{self._workspace}/scripts/experiments:"
                f"{self._workspace}/src:"
                f"/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src:"
                f"{self._rh56_src}"
            ),
        }
        cmd = [
            VENV_PY,
            str(self._runner),
            "--group", "candidate_shadow",
            "--seed", str(seed),
            "--rounds", str(rounds),
            "--camera-source", "mock",
            "--mock-hands",
            "--candidate-params", json.dumps(candidate_params),
            "--out", str(out_json),
        ]
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(self._workspace),
                timeout=timeout_s or (60 * (rounds + 12)),
            )
        summary: dict[str, Any] = {}
        if out_json.is_file():
            summary = json.loads(out_json.read_text())
        if proc.returncode != 0:
            return {
                "hardware_actions_executed": None,
                "rounds_completed": 0,
                "error": f"shadow exited rc={proc.returncode}; see {log_path}",
                "log": str(log_path),
            }
        return {
            "hardware_actions_executed": summary.get("hardware_actions_executed"),
            "rounds_completed": summary.get("rounds", 0),
            "candidate_lifecycle": summary.get("candidate_lifecycle") or {},
            "runtime_s": summary.get("runtime_s"),
            "executor": summary.get("executor"),
            "disclosure": (
                "L2 shadow: mock hands + mock camera BY DESIGN (no "
                "perception/hardware claims; lifecycle + timing only)"
            ),
            "log": str(log_path),
        }

    def _latest_practice_id(self) -> str | None:
        sessions = self._practice_root / "sessions"
        if not sessions.is_dir():
            return None
        candidates = [p for p in sessions.iterdir() if p.is_dir()]
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        # Only accept sessions created in the last 10 minutes — an older
        # directory belongs to a previous run, not to this session.
        if time.time() - latest.stat().st_mtime > 600:
            return None
        return latest.name
