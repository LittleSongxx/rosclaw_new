"""Acceptance orchestrator (PR-EVO-HW-1 §八 Phase 0/1 + CLI backend).

Implements the ``rosclaw acceptance evo-rps`` phases that belong to HW-1:

* ``prepare``  — contract validation, namespace provisioning, preflight
  gates (no mock camera in formal mode), evidence manifest init, task
  driver hash binding, ``rosclaw memory active`` + ``db doctor`` records.
* ``baseline`` — N sessions × M rounds through the pinned task driver,
  each followed by ``practice verify --strict`` + ``db reconcile``; every
  step lands in the evidence manifest.
* ``report``   — manifest → machine-readable + human summary.

Later phases (distill/propose/validate/canary/promote/recurrence) belong
to PR-EVO-HW-3/4/5 and fail loudly here instead of silently pretending.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .contracts import EvoRpsConfig, load_config
from .evidence import EvidenceManifest
from .namespace import ExperimentNamespace
from .preflight import run_preflight
from .session_driver import Rh56RpsWorkspaceDriver

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "acceptance" / "evo_rps_v1.yaml"
VENV_PY = sys.executable

NOT_IMPLEMENTED = {
    "distill": "PR-EVO-HW-3",
    "propose": "PR-EVO-HW-3",
    "validate": "PR-EVO-HW-3",
    "canary": "PR-EVO-HW-4",
    "promote": "PR-EVO-HW-4",
    "recurrence": "PR-EVO-HW-5",
}


class OrchestratorError(RuntimeError):
    pass


class EvoRpsOrchestrator:
    def __init__(self, config: EvoRpsConfig) -> None:
        self.config = config
        self.namespace = ExperimentNamespace.from_config(config)

    # ------------------------------------------------------------------

    def prepare(self, *, dev_allow_mock: bool = False) -> dict[str, Any]:
        preflight = run_preflight(self.config, dev_allow_mock=dev_allow_mock)
        manifest = EvidenceManifest.open(
            self.namespace.evidence_root,
            self.config.experiment_id,
            self.config.config_hash,
        )
        if not preflight.ok:
            # Never provision on a blocked gate — and never crash on it
            # either: the block itself is the evidence (§2.2).
            manifest.record("prepare_blocked", preflight=preflight.to_dict())
            return {
                "ok": False,
                "blocked": preflight.blocked,
                "dev_mode": preflight.dev_mode,
                "evidence": manifest.summary(),
            }
        provision = self.namespace.provision()
        driver = Rh56RpsWorkspaceDriver(self.config, self.namespace.practice_root)
        manifest.record(
            "prepare",
            preflight=preflight.to_dict(),
            namespace=provision,
            task_driver={
                "kind": self.config.task_driver.get("kind"),
                "runner": str(self.config.task_driver.get("runner")),
                **driver.code_hash(),
            },
            dev_mode=preflight.dev_mode,
        )
        memory_active = self._cli(
            ["memory", "active", "--backend", "seekdb_server", "--seekdb-url", self.namespace.dsn]
        )
        db_doctor = self._cli(
            ["db", "doctor", "--backend", "seekdb_server", "--url", self.namespace.dsn, "--json"]
        )
        manifest.record(
            "storage_gate",
            memory_active_rc=memory_active["rc"],
            db_doctor_rc=db_doctor["rc"],
            db_doctor=db_doctor.get("json") or db_doctor.get("text", "")[:400],
        )
        return {
            "ok": preflight.ok,
            "blocked": preflight.blocked,
            "dev_mode": preflight.dev_mode,
            "namespace": provision,
            "evidence": manifest.summary(),
        }

    # ------------------------------------------------------------------

    def baseline(self, *, sessions: int, rounds: int, seed_start: int = 0) -> dict[str, Any]:
        manifest = self._open_manifest()
        preflight = run_preflight(self.config)
        if not preflight.ok:
            manifest.record("baseline_blocked", blocked=preflight.blocked)
            raise OrchestratorError("; ".join(preflight.blocked))
        driver = Rh56RpsWorkspaceDriver(self.config, self.namespace.practice_root)
        results: list[dict[str, Any]] = []
        for index in range(sessions):
            seed = self.config.seed + seed_start + index
            out_dir = self.namespace.evidence_root / "sessions" / f"baseline_{index:02d}"
            started = time.time()
            result = driver.run_session(
                group="no_memory",
                seed=seed,
                rounds=rounds,
                camera_source="realsense",
                out_dir=out_dir,
            )
            verify = self._verify(result.practice_id)
            reconcile = self._cli(["db", "reconcile", "--data-root", str(self.namespace.practice_root)])
            entry = manifest.record(
                "baseline_session",
                index=index,
                seed=seed,
                practice_id=result.practice_id,
                rounds=result.rounds,
                invalid=result.summary.get("invalid_rounds"),
                invalid_rate=result.summary.get("invalid_rate"),
                verified_rate=result.summary.get("verified_rate"),
                peak_temperature=result.summary.get("peak_temperature"),
                runtime_s=round(time.time() - started, 1),
                verify=verify,
                reconcile_rc=reconcile["rc"],
                log=str(result.log_path),
            )
            results.append(entry)
        return {"ok": all(r["verify"].get("rc") == 0 for r in results), "sessions": results}

    # ------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        manifest = self._open_manifest()
        baseline = manifest.by_kind("baseline_session")
        return {
            "experiment_id": self.config.experiment_id,
            "config_hash": self.config.config_hash,
            "manifest": manifest.summary(),
            "baseline_sessions": len(baseline),
            "baseline_ok": all(s.get("verify", {}).get("rc") == 0 for s in baseline),
            "invalid_rates": [s.get("invalid_rate") for s in baseline],
            "peak_temperatures": [s.get("peak_temperature") for s in baseline],
            "blocked": manifest.by_kind("baseline_blocked"),
        }

    # ------------------------------------------------------------------

    def _open_manifest(self) -> EvidenceManifest:
        return EvidenceManifest.open(
            self.namespace.evidence_root,
            self.config.experiment_id,
            self.config.config_hash,
        )

    def _verify(self, practice_id: str | None) -> dict[str, Any]:
        if not practice_id:
            return {"rc": 2, "error": "no practice id"}
        return self._cli(
            [
                "practice", "verify", practice_id, "--strict",
                "--data-root", str(self.namespace.practice_root),
            ]
        )

    @staticmethod
    def _cli(args: list[str], timeout: int = 420) -> dict[str, Any]:
        proc = subprocess.run(
            [VENV_PY, "-m", "rosclaw.cli", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        out: dict[str, Any] = {"rc": proc.returncode}
        text = proc.stdout.strip()
        try:
            out["json"] = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            out["text"] = text[-800:]
        if proc.returncode != 0:
            out["stderr"] = proc.stderr[-400:]
        return out


def orchestrator_for(config_path: str | Path | None = None) -> EvoRpsOrchestrator:
    return EvoRpsOrchestrator(load_config(config_path or DEFAULT_CONFIG))


def phase_not_implemented(phase: str) -> dict[str, Any]:
    pr = NOT_IMPLEMENTED.get(phase)
    return {
        "ok": False,
        "error": f"phase {phase!r} is implemented in {pr or 'a later PR'} — "
        "the harness never pretends a phase ran when it did not",
        "planned_in": pr,
    }
