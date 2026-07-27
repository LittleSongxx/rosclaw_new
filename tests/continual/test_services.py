from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw.continual.boundary_feedback import BoundaryReplayRequest
from rosclaw.continual.contracts import ExperiencePartition, SkillPhase
from rosclaw.continual.experience import (
    ContinualExperienceStore,
    ExperienceRecord,
)
from rosclaw.continual.services.experience import ExperienceService
from rosclaw.continual.services.inference import InferenceService
from rosclaw.continual.services.learner import LearnerProduct, LearnerService
from rosclaw.continual.services.persistence import DurableEventLog, require_external_service_root
from rosclaw.continual.services.rollout import RolloutService, RolloutState
from rosclaw.continual.services.weight_update import WeightUpdateService
from rosclaw.continual.stability import StabilityPlasticityGate
from tests.continual.helpers import digest, policy, trajectory
from tests.continual.test_stability import _passing_evidence


def _source_checkout() -> Path:
    return Path(__file__).parents[2]


def _four_partition_store():
    parent, _ = policy(0)
    store = ContinualExperienceStore()
    records = (
        ExperienceRecord(trajectory(parent, episode="recent"), ExperiencePartition.RECENT),
        ExperienceRecord(
            trajectory(parent, episode="anchor"),
            ExperiencePartition.ANCHOR,
            anchor_policy_hash=parent.artifact_hash,
        ),
        ExperienceRecord(
            trajectory(parent, episode="boundary", critical=True),
            ExperiencePartition.BOUNDARY,
            boundary_reason="fall counterexample",
        ),
        ExperienceRecord(
            trajectory(parent, episode="self"),
            ExperiencePartition.SELF,
            self_change_hash=digest("motor-gain-shift"),
        ),
    )
    for record in records:
        store.append(record)
    return parent, records, store


def test_durable_event_log_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "journal"
    log = DurableEventLog(root, service="test", clock_ns=lambda: 1)
    log.append("RECORDED", {"value": 1})
    event_path = next((root / "events").glob("*.json"))
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["payload"]["value"] = 2
    event_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        DurableEventLog(root, service="test")


def test_service_state_rejects_filesystem_root() -> None:
    with pytest.raises(ValueError, match="filesystem root"):
        require_external_service_root(Path("/"), _source_checkout())


def test_rollout_recovery_aborts_motion_and_never_replays_actions(tmp_path: Path) -> None:
    parent, _ = policy(0)
    service = RolloutService(tmp_path, source_checkout=_source_checkout())
    assigned = service.assign(
        episode_id="kick-1",
        scenario_commitment=digest("scenario-1"),
        policy=parent,
    )
    running = service.start(assigned.assignment_id, worker_id="sim-worker-0")
    assert running.state is RolloutState.RUNNING

    recovered = RolloutService(tmp_path, source_checkout=_source_checkout())

    assert recovered.assignments[0].state is RolloutState.ABORTED
    assert recovered.assignments[0].version_switch_count == 0
    assert recovered.recovered_abort_count == 1
    assert "not replayed" in recovered.assignments[0].abort_reason


def test_rollout_completes_only_matching_strict_versioned_trajectory(tmp_path: Path) -> None:
    parent, _ = policy(0)
    service = RolloutService(tmp_path, source_checkout=_source_checkout())
    assignment = service.assign(
        episode_id="kick-2",
        scenario_commitment=digest("scenario-2"),
        policy=parent,
    )
    service.start(assignment.assignment_id, worker_id="sim-worker-0")

    completed = service.complete(
        assignment.assignment_id,
        trajectory=trajectory(parent, episode="kick-2"),
    )

    assert completed.state is RolloutState.COMPLETE
    assert completed.strict_replay
    assert completed.version_switch_count == 0


def test_experience_service_recovers_catalog_and_boundary_queue(tmp_path: Path) -> None:
    parent, records, _ = _four_partition_store()
    request = BoundaryReplayRequest(
        scenario_id="unsafe-seed-7",
        scenario_commitment=digest("scenario-7"),
        replay_partition="boundary",
        parent_policy_hash=parent.version_hash,
        candidate_policy_hash=digest("candidate-version"),
        parent_status="PASS",
        candidate_status="SAFETY_ABORT",
        critical_signals=("fall",),
        source_evidence_hash=digest("matched-evidence"),
    )
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as service:
        for record in records:
            service.append(record)
        service.enqueue_boundary(request)
        first = service.audit_receipt()

    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as recovered:
        second = recovered.audit_receipt()
        batch = recovered.sample(batch_size=20, learner_version=0, seed=4)
        recovered.complete_boundary(
            request.request_hash,
            record=ExperienceRecord(
                trajectory(parent, episode="boundary-rerollout", critical=True),
                ExperiencePartition.BOUNDARY,
                boundary_reason="reproduced matched-evaluation fall",
            ),
        )

        assert len(batch.records) == 20
        assert second["catalog_counts"] == first["catalog_counts"]
        assert len(recovered.pending_boundary_requests) == 0
        assert recovered.audit_receipt()["hardware_authorized"] is False


def test_experience_recovery_rejects_corrupted_catalog_row(tmp_path: Path) -> None:
    _, records, _ = _four_partition_store()
    with ExperienceService(tmp_path, source_checkout=_source_checkout()) as service:
        service.append(records[0])
    database = sqlite3.connect(tmp_path / "experience" / "catalog.sqlite3")
    with database:
        database.execute(
            "UPDATE experience_records SET partition='anchor' WHERE record_hash=?",
            (records[0].record_hash,),
        )
    database.close()

    with pytest.raises(ValueError, match="catalog diverges"):
        ExperienceService(tmp_path, source_checkout=_source_checkout())


def _matching_gate(parent, candidate):
    evidence = replace(
        _passing_evidence(),
        parent_policy_hash=parent.artifact_hash,
        candidate_policy_hash=candidate.artifact_hash,
    )
    return StabilityPlasticityGate().evaluate(evidence)


def test_inference_slot_blocks_mid_motion_switch_then_activates_safely(tmp_path: Path) -> None:
    parent, parent_artifact = policy(0)
    candidate, candidate_artifact = policy(1, parent=parent)
    inference = InferenceService(
        tmp_path,
        source_checkout=_source_checkout(),
        active=parent,
        active_artifact=parent_artifact,
    )
    updater = WeightUpdateService(
        tmp_path,
        source_checkout=_source_checkout(),
        inference=inference,
    )
    updater.publish(candidate, artifact=candidate_artifact)
    updater.verify()
    updater.stage()
    lease = inference.begin_motion(episode_id="kick-live", phase=SkillPhase.SWING)

    blocked = updater.activate(
        phase=SkillPhase.COMPLETE,
        gate_report=_matching_gate(parent, candidate),
    )
    inference.end_motion(lease.lease_id, aborted=True, reason="candidate swap was blocked")
    activated = updater.activate(
        phase=SkillPhase.COMPLETE,
        gate_report=_matching_gate(parent, candidate),
    )

    assert blocked.inference.active_version_hash == parent.version_hash
    assert blocked.inference.frozen
    assert activated.inference.active_version_hash == candidate.version_hash
    assert not activated.inference.frozen
    assert inference.rollback == parent


def test_inference_recovery_aborts_inflight_version_lease(tmp_path: Path) -> None:
    parent, artifact = policy(0)
    service = InferenceService(
        tmp_path,
        source_checkout=_source_checkout(),
        active=parent,
        active_artifact=artifact,
    )
    service.begin_motion(episode_id="interrupted", phase=SkillPhase.CONTACT)

    recovered = InferenceService(tmp_path, source_checkout=_source_checkout())

    assert recovered.active_motion_count == 0
    assert recovered.active.version_hash == parent.version_hash
    assert recovered.recovered_abort_count == 1


def test_weight_update_recovery_commits_completed_inference_mutation(tmp_path: Path) -> None:
    parent, parent_artifact = policy(0)
    candidate, candidate_artifact = policy(1, parent=parent)
    inference = InferenceService(
        tmp_path,
        source_checkout=_source_checkout(),
        active=parent,
        active_artifact=parent_artifact,
    )
    updater = WeightUpdateService(
        tmp_path,
        source_checkout=_source_checkout(),
        inference=inference,
    )
    operation_id = digest("interrupted-publish")
    updater.log.append(
        "REQUESTED",
        {
            "operation_id": operation_id,
            "operation": "publish",
            "parameters": {"policy_version_hash": candidate.version_hash},
            "inference_event_hash_before": inference.log.last_hash,
        },
    )
    inference._publish(candidate, candidate_artifact)

    recovered = WeightUpdateService(
        tmp_path,
        source_checkout=_source_checkout(),
        inference=inference,
    )

    assert recovered.recovered_completion_count == 1
    assert recovered.recovered_abort_count == 0
    assert inference.published == candidate


def test_weight_update_recovery_aborts_request_without_mutation(tmp_path: Path) -> None:
    parent, artifact = policy(0)
    inference = InferenceService(
        tmp_path,
        source_checkout=_source_checkout(),
        active=parent,
        active_artifact=artifact,
    )
    updater = WeightUpdateService(
        tmp_path,
        source_checkout=_source_checkout(),
        inference=inference,
    )
    updater.log.append(
        "REQUESTED",
        {
            "operation_id": digest("interrupted-verify"),
            "operation": "verify",
            "parameters": {},
            "inference_event_hash_before": inference.log.last_hash,
        },
    )

    recovered = WeightUpdateService(
        tmp_path,
        source_checkout=_source_checkout(),
        inference=inference,
    )

    assert recovered.recovered_abort_count == 1
    assert recovered.recovered_completion_count == 0


def test_learner_service_is_idempotent_and_recovers_artifacts(tmp_path: Path) -> None:
    parent, _, store = _four_partition_store()
    batch = store.sample(batch_size=20, learner_version=0, seed=3)
    candidate, artifact = policy(1, parent=parent)
    calls = 0

    def executor(_batch):
        nonlocal calls
        calls += 1
        return LearnerProduct(
            candidate=candidate,
            artifact=artifact,
            checkpoint=b"trusted-optimizer-checkpoint",
            metrics={"loss": 0.1, "converged": True},
        )

    service = LearnerService(tmp_path, source_checkout=_source_checkout(), parent=parent)
    first = service.execute(batch, executor=executor)
    repeated = service.execute(batch, executor=executor)
    recovered = LearnerService(tmp_path, source_checkout=_source_checkout(), parent=parent)

    assert first == repeated
    assert calls == 1
    assert recovered.completed_batch_hashes == (batch.batch_hash,)
    assert recovered.checkpoint_bytes(first.checkpoint_hash) == b"trusted-optimizer-checkpoint"
    assert first.hardware_authorized is False


def test_learner_recovery_quarantines_uncertain_optimizer_job(tmp_path: Path) -> None:
    parent, _, store = _four_partition_store()
    batch = store.sample(batch_size=20, learner_version=0, seed=8)
    service = LearnerService(tmp_path, source_checkout=_source_checkout(), parent=parent)
    job_id = digest("unfinished-job")
    service.log.append(
        "JOB_STARTED",
        {
            "job_id": job_id,
            "batch_hash": batch.batch_hash,
            "parent_policy_hash": parent.version_hash,
        },
    )

    recovered = LearnerService(tmp_path, source_checkout=_source_checkout(), parent=parent)

    assert recovered.quarantined_batch_hashes == (batch.batch_hash,)
    with pytest.raises(RuntimeError, match="quarantined"):
        recovered.execute(batch, executor=lambda _: pytest.fail("must not replay update"))
