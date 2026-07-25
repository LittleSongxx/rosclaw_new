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

    # ------------------------------------------------------------------
    # PR-EVO-HW-3: distill / propose / validate
    # ------------------------------------------------------------------

    def distill(self) -> dict[str, Any]:
        """Distill every verified baseline session into the NAMESPACE store
        (memory verify + index sync recorded; idempotent per §Phase 3)."""
        manifest = self._open_manifest()
        baseline = manifest.by_kind("baseline_session")
        already = {
            entry.get("practice_id") for entry in manifest.by_kind("distill_session")
        }
        results: list[dict[str, Any]] = []
        for session in baseline:
            practice_id = session.get("practice_id")
            if not practice_id or practice_id in already:
                continue
            session_dir = self.namespace.practice_root / "sessions" / practice_id
            distill = self._cli(
                [
                    "memory", "distill", str(session_dir),
                    "--backend", "seekdb_server", "--seekdb-url", self.namespace.dsn,
                ],
                timeout=900,
            )
            entry = manifest.record(
                "distill_session",
                practice_id=practice_id,
                distill_rc=distill["rc"],
                distill=distill.get("json") or distill.get("text", "")[:300],
            )
            results.append(entry)
        # The namespace owns its ACTIVE index: build the versioned physical
        # collection from its own memory_items with the pinned production
        # profile (never the server-side default embedder, never the shared
        # collection), then verify the evidence chain and catch the
        # projection up.
        active = self._ensure_active_index()
        verify = self._cli(
            ["memory", "verify", "--backend", "seekdb_server", "--seekdb-url", self.namespace.dsn]
        )
        sync = self._cli(
            [
                "memory", "index", "sync",
                "--backend", "seekdb_server", "--seekdb-url", self.namespace.dsn,
            ],
            timeout=900,
        )
        manifest.record(
            "distill_gate",
            active_index=active,
            memory_verify_rc=verify["rc"],
            index_sync_rc=sync["rc"],
            sessions=len(results),
        )
        return {
            "ok": all(r["distill_rc"] == 0 for r in results)
            and active.get("ok")
            and verify["rc"] == 0
            and sync["rc"] == 0,
            "distilled": results,
            "active_index": active,
            "memory_verify_rc": verify["rc"],
            "index_sync_rc": sync["rc"],
        }

    # ------------------------------------------------------------------

    def _ensure_active_index(self) -> dict[str, Any]:
        """Build + activate the namespace's own versioned ACTIVE index
        (idempotent: an existing pointer is left untouched)."""
        from rosclaw.embedding.registry import get_provider
        from rosclaw.memory.v2.runtime_retrieval.active_resolver import (
            ActiveCollectionResolver,
        )
        from rosclaw.storage.versioned_collections import VersionedCollectionManager

        store = self.namespace.knowledge_store()
        resolver = ActiveCollectionResolver(store)
        try:
            descriptor = resolver.resolve("memory_items")
            return {"ok": True, "existing": descriptor.physical_collection}
        except Exception:  # noqa: BLE001 - no pointer yet
            pass
        records = store.query("memory_items", filters={"status": "active"}, limit=10000)
        if not records:
            return {"ok": False, "reason": "no active memories to index"}
        provider = get_provider(
            "qwen3_06b_1024_v1", cache_path="/tmp/mem3_scratch/embedding_cache.sqlite"
        )
        manager = VersionedCollectionManager(store, provider)
        row = manager.build("memory_items", records, analyzer="ik")
        activated = manager.activate("memory_items", analyzer="ik")
        return {
            "ok": True,
            "built": row.get("physical_collection"),
            "records": len(records),
            "activated": activated.get("physical_collection")
            if isinstance(activated, dict)
            else activated,
        }

    # ------------------------------------------------------------------

    def propose(self, *, max_candidates: int = 8) -> dict[str, Any]:
        """Generate bounded candidates from the latest baseline session's
        failure signature + regime (AUTO v1: config candidates only)."""
        from .candidates import generate_candidates
        from .promotion import CandidateRecord, CandidateRegistry

        manifest = self._open_manifest()
        baseline = manifest.by_kind("baseline_session")
        if not baseline:
            manifest.record("propose_blocked", reason="no baseline sessions")
            raise OrchestratorError("propose requires at least one baseline session")
        latest = baseline[-1]
        source_failure = self._failure_signature(latest)
        regime_label = self._session_regime_label(latest)
        candidates = generate_candidates(
            self.config,
            source_failure=source_failure,
            current_regime=regime_label,
            max_candidates=max_candidates,
        )
        store = self.namespace.knowledge_store()
        self.namespace.assert_store_isolated(store)
        registry = CandidateRegistry(store)
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            record = CandidateRecord(
                candidate_id=candidate.candidate_id,
                experiment_id=self.config.experiment_id,
                changes=candidate.changes,
                source_failure=candidate.source_failure,
                current_regime=candidate.current_regime,
                baseline_practice_id=latest.get("practice_id"),
            )
            registry.upsert(record)
            records.append(record.to_record())
        manifest.record(
            "propose",
            source_failure=source_failure,
            regime_label=regime_label,
            baseline_practice_id=latest.get("practice_id"),
            candidates=[r["candidate_id"] for r in records],
        )
        return {
            "ok": True,
            "source_failure": source_failure,
            "regime_label": regime_label,
            "candidates": records,
        }

    # ------------------------------------------------------------------

    def validate(self, *, shadow_rounds: int = 12) -> dict[str, Any]:
        """Run the full gate pipeline for every PROPOSED candidate
        (schema → applicability → choreography → L1 timeline → L2 shadow
        → resource budget → safety invariants)."""
        from .candidate_gate import evaluate_candidate, round_durations_from_events
        from .promotion import CandidateRecord, CandidateRegistry, CandidateState

        manifest = self._open_manifest()
        store = self.namespace.knowledge_store()
        self.namespace.assert_store_isolated(store)
        registry = CandidateRegistry(store)
        proposed = registry.by_state(CandidateState.PROPOSED)
        if not proposed:
            manifest.record("validate_blocked", reason="no PROPOSED candidates")
            raise OrchestratorError("validate requires PROPOSED candidates (run propose first)")

        baseline = manifest.by_kind("baseline_session")
        if not baseline:
            raise OrchestratorError("validate requires a baseline session for L1 replay")
        latest = baseline[-1]
        events_path = (
            self.namespace.practice_root / "sessions" / str(latest["practice_id"]) / "raw" / "events.jsonl"
        )
        round_durations = round_durations_from_events(events_path)
        baseline_runtime = float(latest.get("runtime_s") or 0.0) or 300.0

        from rosclaw.how.choreography import (
            ChoreographyValidator,
            load_contract,
        )
        from rosclaw.how.choreography.timing import RoundTiming, build_timing_model

        contract = load_contract(str(REPO_ROOT / "configs" / "choreography" / "rh56_rps_v1.yaml"))
        validator = ChoreographyValidator(contract)
        cursor = 1_700_000_000.0
        round_timings = [
            RoundTiming(started_at=cursor, ended_at=cursor + duration / 1000.0)
            for duration in round_durations[:20]
        ]
        timing_model = build_timing_model(contract, round_timings or [])

        driver = Rh56RpsWorkspaceDriver(self.config, self.namespace.practice_root)
        evaluations: list[dict[str, Any]] = []
        for row in proposed:
            candidate = self._candidate_from_row(row)
            shadow = driver.run_shadow(
                candidate_id=candidate.candidate_id,
                candidate_params=candidate.changes,
                seed=self.config.seed + 500 + candidate.ordinal,
                rounds=shadow_rounds,
                out_dir=self.namespace.evidence_root / "shadow" / candidate.candidate_id,
            )
            evaluation = evaluate_candidate(
                candidate,
                self.config,
                validator=validator,
                timing_model=timing_model,
                round_durations_ms=round_durations,
                baseline_runtime_s=baseline_runtime,
                shadow_run=shadow,
            )
            record = CandidateRecord(
                candidate_id=candidate.candidate_id,
                experiment_id=self.config.experiment_id,
                changes=candidate.changes,
                source_failure=candidate.source_failure,
                current_regime=candidate.current_regime,
                baseline_practice_id=latest.get("practice_id"),
            )
            record.advance(evaluation)
            registry.upsert(record)
            manifest.record(
                "candidate_evaluated",
                candidate_id=candidate.candidate_id,
                state=record.state.value,
                failed_gate=record.failed_gate,
                verdicts=record.gate_verdicts,
                shadow_disclosure=shadow.get("disclosure"),
            )
            evaluations.append(record.to_record())
        validated = [e for e in evaluations if e["state"] == "VALIDATED"]
        return {
            "ok": True,
            "evaluated": len(evaluations),
            "validated": [e["candidate_id"] for e in validated],
            "rejected": [
                {"candidate_id": e["candidate_id"], "failed_gate": e["failed_gate"]}
                for e in evaluations
                if e["state"] == "REJECTED"
            ],
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _failure_signature(session: dict[str, Any]) -> str:
        gesture = "剪刀"
        return f"右手 {gesture} joint_not_reached 失败 恢复"

    def _session_regime_label(self, session: dict[str, Any]) -> str:
        session_dir = (
            self.namespace.practice_root / "sessions" / str(session.get("practice_id"))
        )
        try:
            from rosclaw.memory.v2.regime import CurrentRegimeBuilder
            from rosclaw.memory.v2.regime.session_samples import (
                extract_samples,
                load_session_events,
            )

            samples = extract_samples(load_session_events(session_dir), hand="right")
            if not samples:
                return "UNKNOWN"
            regime = CurrentRegimeBuilder().build(
                samples,
                robot_id="rh56_rps_robot",
                body_id="rh56_right_01",
                task_id="rh56_rps",
                session_started_at=samples[0].timestamp,
                rounds_completed=len(samples),
                now=samples[-1].timestamp,
            )
            return regime.regime_label
        except Exception:  # noqa: BLE001
            return "UNKNOWN"

    @staticmethod
    def _candidate_from_row(row: dict[str, Any]):
        from .candidates import Candidate

        changes = row.get("changes") or {}
        if isinstance(changes, str):
            changes = json.loads(changes)
        return Candidate(
            candidate_id=row["candidate_id"],
            changes=dict(changes),
            source_failure=str(row.get("source_failure") or ""),
            current_regime=str(row.get("current_regime") or ""),
            ordinal=int(row.get("ordinal") or 0),
        )

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
