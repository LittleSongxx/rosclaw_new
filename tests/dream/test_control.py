from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rosclaw.continual.services.persistence import DurableEventLog
from rosclaw.dream import (
    DreamBudget,
    DreamBudgetExceededError,
    DreamCampaignState,
    DreamPlanner,
    DreamPlanRequest,
    DreamScheduler,
    DreamType,
    dream_doctor,
    inspect_dream_journal,
)
from rosclaw.growth import GrowthMetricSpec, MetricDirection, SkillGrowthSpec


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def __call__(self) -> int:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += int(seconds * 1e9)


def _budget(**overrides: float | int) -> DreamBudget:
    values: dict[str, float | int] = {
        "max_gpu_seconds": 100.0,
        "max_cpu_rollouts": 20,
        "max_candidates": 4,
        "max_wall_seconds": 300.0,
        "max_policy_change": 0.05,
        "max_anchor_drift": 0.02,
    }
    values.update(overrides)
    return DreamBudget(**values)  # type: ignore[arg-type]


def _spec(*, collective: bool = False) -> SkillGrowthSpec:
    return SkillGrowthSpec(
        skill_id="g1.kick",
        adapter_id="g1.motion_adapter",
        body_hashes=(_hash("body"),),
        capability_ids=("kick", "recover"),
        observation_contract_hash=_hash("observation"),
        action_contract_hash=_hash("action"),
        reward_contract_hash=_hash("reward"),
        cost_contract_hash=_hash("cost"),
        practice_source_ids=("practice.goalforge",),
        collective_source_ids=(("collective.motiondecode",) if collective else ()),
        allowed_dream_types=("replay", "counterfactual", "social"),
        allowed_learner_ids=("residual.sac",),
        historical_anchor_hashes=(_hash("anchor"),),
        boundary_suite_hash=_hash("boundary"),
        metrics=(
            GrowthMetricSpec(
                metric_id="kick.recovery_stability",
                direction=MetricDirection.MAXIMIZE,
                primary=True,
            ),
        ),
        promotion_profile_hash=_hash("promotion"),
        rollback_policy_hash=_hash("rollback"),
    )


def _request(
    *,
    budget: DreamBudget | None = None,
    dream_types: tuple[DreamType, ...] = (DreamType.REPLAY,),
    collective: tuple[str, ...] = (),
) -> DreamPlanRequest:
    return DreamPlanRequest(
        body_hash=_hash("body"),
        parent_policy_hash=_hash("parent"),
        trigger_kind="post_practice",
        trigger_evidence_hashes=(_hash("trigger"),),
        objectives=("improve_recovery",),
        constraint_hashes=(_hash("constraint"),),
        practice_snapshot_hashes=(_hash("practice"),),
        collective_capsule_hashes=collective,
        historical_anchor_hashes=(_hash("anchor"),),
        boundary_suite_hashes=(_hash("boundary"),),
        private_holdout_commitment=_hash("holdout"),
        dream_types=dream_types,
        learner_ids=("residual.sac",),
        budget=budget or _budget(),
    )


def _campaign(*, budget: DreamBudget | None = None):  # type: ignore[no-untyped-def]
    return DreamPlanner().plan(_spec(), _request(budget=budget)).campaign


def test_planner_is_pure_and_has_no_activation_authority() -> None:
    planner = DreamPlanner()
    receipt = planner.plan(_spec(), _request())

    assert not hasattr(planner, "activate")
    assert receipt.activation_authorized is False
    assert receipt.hardware_authorized is False
    assert receipt.campaign.parent_policy_hash == _hash("parent")
    assert receipt.to_dict()["campaign_hash"] == receipt.campaign.campaign_hash


def test_plan_request_rejects_runtime_type_spoofing() -> None:
    base = _request()
    with pytest.raises(ValueError, match="recognized DreamType"):
        DreamPlanRequest(**{**base.__dict__, "dream_types": ("replay",)})  # type: ignore[arg-type]


def test_planner_enforces_growth_spec_body_anchors_boundary_and_learners() -> None:
    base = _request()
    with pytest.raises(ValueError, match="body"):
        DreamPlanner().plan(
            _spec(), DreamPlanRequest(**{**base.__dict__, "body_hash": _hash("other")})
        )
    with pytest.raises(ValueError, match="historical anchor"):
        DreamPlanner().plan(
            _spec(),
            DreamPlanRequest(**{**base.__dict__, "historical_anchor_hashes": (_hash("wrong"),)}),
        )
    with pytest.raises(ValueError, match="boundary suite"):
        DreamPlanner().plan(
            _spec(),
            DreamPlanRequest(**{**base.__dict__, "boundary_suite_hashes": (_hash("wrong"),)}),
        )
    with pytest.raises(ValueError, match="learner"):
        DreamPlanner().plan(
            _spec(), DreamPlanRequest(**{**base.__dict__, "learner_ids": ("unbounded.learner",)})
        )


def test_social_dream_requires_spec_permission_and_capsule() -> None:
    with pytest.raises(ValueError, match="not enabled"):
        DreamPlanner().plan(
            _spec(),
            _request(
                dream_types=(DreamType.SOCIAL,),
                collective=(_hash("capsule"),),
            ),
        )
    receipt = DreamPlanner().plan(
        _spec(collective=True),
        _request(
            dream_types=(DreamType.SOCIAL,),
            collective=(_hash("capsule"),),
        ),
    )
    assert receipt.campaign.dream_types == (DreamType.SOCIAL,)


def test_scheduler_submit_lease_usage_and_complete_are_recoverable(tmp_path: Path) -> None:
    clock = _Clock()
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(
        state,
        source_checkout=checkout,
        clock_ns=clock,
        token_factory=lambda: "a" * 32,
    ) as scheduler:
        submitted = scheduler.submit(campaign)
        assert submitted.state is DreamCampaignState.QUEUED
        lease = scheduler.acquire(worker_id="gpu-worker-0", lease_seconds=30.0)
        assert "lease_token" not in lease.to_dict()
        assert lease.activation_authorized is False
        status = scheduler.record_usage(
            campaign.campaign_hash,
            lease_token=lease.lease_token,
            gpu_seconds=2.5,
            cpu_rollouts=3,
            candidates=1,
        )
        assert status.usage.gpu_seconds == 2.5
        completed = scheduler.complete(
            campaign.campaign_hash,
            lease_token=lease.lease_token,
            result_manifest_hash=_hash("result"),
            candidate_artifact_hashes=(_hash("candidate"),),
        )
        assert completed.state is DreamCampaignState.COMPLETED
        assert completed.activation_authorized is False
        assert completed.hardware_authorized is False
        completed_elapsed = completed.elapsed_wall_seconds
        clock.advance(30.0)
        assert scheduler.status(campaign.campaign_hash).elapsed_wall_seconds == completed_elapsed

    with DreamScheduler(state, source_checkout=checkout, clock_ns=clock) as recovered:
        status = recovered.status(campaign.campaign_hash)
        assert status.state is DreamCampaignState.COMPLETED
        assert status.result_manifest_hash == _hash("result")
        assert status.candidate_artifact_hashes == (_hash("candidate"),)


def test_journal_stores_token_commitment_not_worker_secret(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    secret = "worker-secret-token-0123456789"
    campaign = _campaign()
    with DreamScheduler(
        state,
        source_checkout=checkout,
        token_factory=lambda: secret,
    ) as scheduler:
        scheduler.submit(campaign)
        scheduler.acquire(worker_id="worker", lease_seconds=30.0)

    journal_text = "".join(
        path.read_text(encoding="utf-8")
        for path in (state / "dream" / "journal" / "events").glob("*.json")
    )
    assert secret not in journal_text
    assert "lease_token_hash" in journal_text


def test_wrong_or_expired_worker_token_cannot_mutate_campaign(tmp_path: Path) -> None:
    clock = _Clock()
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(
        state,
        source_checkout=checkout,
        clock_ns=clock,
        token_factory=lambda: "b" * 32,
    ) as scheduler:
        scheduler.submit(campaign)
        lease = scheduler.acquire(worker_id="worker", lease_seconds=5.0)
        with pytest.raises(PermissionError, match="does not match"):
            scheduler.record_usage(campaign.campaign_hash, lease_token="wrong" * 8, candidates=1)
        clock.advance(6.0)
        status = scheduler.status(campaign.campaign_hash)
        assert status.state is DreamCampaignState.PAUSED
        assert status.worker_id is None
        with pytest.raises(RuntimeError, match="no active worker lease"):
            scheduler.record_usage(
                campaign.campaign_hash,
                lease_token=lease.lease_token,
                candidates=1,
            )


def test_scheduler_rejects_boolean_numeric_spoofing(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(
        state,
        source_checkout=checkout,
        token_factory=lambda: "numeric-spoof-token" * 2,
    ) as scheduler:
        scheduler.submit(campaign)
        with pytest.raises(ValueError, match="lease_seconds"):
            scheduler.acquire(worker_id="worker", lease_seconds=True)  # type: ignore[arg-type]
        lease = scheduler.acquire(worker_id="worker", lease_seconds=30.0)
        with pytest.raises(ValueError, match="gpu_seconds must be a number"):
            scheduler.record_usage(
                campaign.campaign_hash,
                lease_token=lease.lease_token,
                gpu_seconds=True,  # type: ignore[arg-type]
            )


def test_replay_revalidates_usage_instead_of_trusting_writer(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(
        state,
        source_checkout=checkout,
        token_factory=lambda: "replay-validation-token" * 2,
    ) as scheduler:
        scheduler.submit(campaign)
        lease = scheduler.acquire(worker_id="worker", lease_seconds=30.0)

    foreign_writer = DurableEventLog(
        state / "dream" / "journal",
        service="dream-scheduler",
    )
    foreign_writer.append(
        "USAGE_RECORDED",
        {
            "campaign_hash": campaign.campaign_hash,
            "lease_id": lease.lease_id,
            "delta": {"gpu_seconds": True, "cpu_rollouts": 0, "candidates": 0},
            "cumulative": {"gpu_seconds": 1.0, "cpu_rollouts": 0, "candidates": 0},
        },
    )
    with pytest.raises(ValueError, match="delta gpu_seconds must be a number"):
        DreamScheduler(state, source_checkout=checkout)


def test_paused_campaign_can_resume_with_a_fresh_lease(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    tokens = iter(("c" * 32, "d" * 32))
    with DreamScheduler(
        state,
        source_checkout=checkout,
        token_factory=lambda: next(tokens),
    ) as scheduler:
        scheduler.submit(campaign)
        first = scheduler.acquire(worker_id="worker-1", lease_seconds=30.0)
        paused = scheduler.pause(
            campaign.campaign_hash,
            lease_token=first.lease_token,
            reason="checkpoint complete",
        )
        assert paused.state is DreamCampaignState.PAUSED
        assert (
            scheduler.resume(campaign.campaign_hash, reason="new worker").state
            is DreamCampaignState.QUEUED
        )
        second = scheduler.acquire(worker_id="worker-2", lease_seconds=30.0)
        assert second.lease_token != first.lease_token


def test_active_lease_survives_restart_until_expiry_then_pauses(tmp_path: Path) -> None:
    clock = _Clock()
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(
        state,
        source_checkout=checkout,
        clock_ns=clock,
        token_factory=lambda: "restart-safe-token" * 2,
    ) as scheduler:
        scheduler.submit(campaign)
        lease = scheduler.acquire(worker_id="worker", lease_seconds=10.0)

    clock.advance(2.0)
    with DreamScheduler(state, source_checkout=checkout, clock_ns=clock) as recovered:
        status = recovered.record_usage(
            campaign.campaign_hash,
            lease_token=lease.lease_token,
            candidates=1,
        )
        assert status.state is DreamCampaignState.RUNNING

    clock.advance(9.0)
    with DreamScheduler(state, source_checkout=checkout, clock_ns=clock) as expired:
        assert expired.status(campaign.campaign_hash).state is DreamCampaignState.PAUSED


def test_heartbeat_never_shortens_existing_lease(tmp_path: Path) -> None:
    clock = _Clock()
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(
        state,
        source_checkout=checkout,
        clock_ns=clock,
        token_factory=lambda: "f" * 32,
    ) as scheduler:
        scheduler.submit(campaign)
        lease = scheduler.acquire(worker_id="worker", lease_seconds=30.0)
        clock.advance(1.0)
        renewed = scheduler.heartbeat(
            campaign.campaign_hash,
            lease_token=lease.lease_token,
            extend_seconds=1.0,
        )
        assert renewed.lease_expires_at_ns == lease.expires_at_ns


def test_reused_capability_token_is_rejected_before_journal_append(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    first = _campaign()
    second_request = DreamPlanRequest(
        **{**_request().__dict__, "parent_policy_hash": _hash("parent-2")}
    )
    second = DreamPlanner().plan(_spec(), second_request).campaign
    with DreamScheduler(
        state,
        source_checkout=checkout,
        token_factory=lambda: "g" * 32,
    ) as scheduler:
        scheduler.submit(first)
        scheduler.submit(second)
        lease = scheduler.acquire(
            worker_id="worker-1", lease_seconds=30.0, campaign_hash=first.campaign_hash
        )
        scheduler.pause(first.campaign_hash, lease_token=lease.lease_token, reason="rotate")
        event_count = len(scheduler.log.events)
        with pytest.raises(RuntimeError, match="reused"):
            scheduler.acquire(
                worker_id="worker-2",
                lease_seconds=30.0,
                campaign_hash=second.campaign_hash,
            )
        assert len(scheduler.log.events) == event_count
        assert scheduler.status(second.campaign_hash).state is DreamCampaignState.QUEUED


def test_resource_overrun_is_durably_terminal_and_cannot_resume(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign(budget=_budget(max_candidates=1))
    with DreamScheduler(
        state,
        source_checkout=checkout,
        token_factory=lambda: "e" * 32,
    ) as scheduler:
        scheduler.submit(campaign)
        lease = scheduler.acquire(worker_id="worker", lease_seconds=30.0)
        scheduler.record_usage(campaign.campaign_hash, lease_token=lease.lease_token, candidates=1)
        with pytest.raises(DreamBudgetExceededError):
            scheduler.record_usage(
                campaign.campaign_hash,
                lease_token=lease.lease_token,
                candidates=1,
            )
        assert scheduler.status(campaign.campaign_hash).state is DreamCampaignState.BUDGET_EXHAUSTED
        with pytest.raises(RuntimeError, match="only a paused"):
            scheduler.resume(campaign.campaign_hash, reason="try again")


def test_completion_cannot_smuggle_unaccounted_candidates(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(
        state,
        source_checkout=checkout,
        token_factory=lambda: "h" * 32,
    ) as scheduler:
        scheduler.submit(campaign)
        lease = scheduler.acquire(worker_id="worker", lease_seconds=30.0)
        with pytest.raises(ValueError, match="durably accounted"):
            scheduler.complete(
                campaign.campaign_hash,
                lease_token=lease.lease_token,
                result_manifest_hash=_hash("result"),
                candidate_artifact_hashes=(_hash("unaccounted"),),
            )
        assert scheduler.status(campaign.campaign_hash).state is DreamCampaignState.RUNNING


def test_cancel_and_fail_are_terminal_without_activation(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    cancelled_campaign = _campaign()
    failed_request = DreamPlanRequest(
        **{**_request().__dict__, "parent_policy_hash": _hash("failed-parent")}
    )
    failed_campaign = DreamPlanner().plan(_spec(), failed_request).campaign
    with DreamScheduler(
        state,
        source_checkout=checkout,
        token_factory=lambda: "i" * 32,
    ) as scheduler:
        scheduler.submit(cancelled_campaign)
        cancelled = scheduler.cancel(cancelled_campaign.campaign_hash, reason="superseded")
        assert cancelled.state is DreamCampaignState.CANCELLED
        assert cancelled.activation_authorized is False
        with pytest.raises(RuntimeError, match="terminal"):
            scheduler.cancel(cancelled_campaign.campaign_hash, reason="again")

        scheduler.submit(failed_campaign)
        lease = scheduler.acquire(worker_id="worker", lease_seconds=30.0)
        failed = scheduler.fail(
            failed_campaign.campaign_hash,
            lease_token=lease.lease_token,
            reason="non-finite learner output",
        )
        assert failed.state is DreamCampaignState.FAILED
        assert failed.hardware_authorized is False


def test_wall_clock_budget_exhausts_queued_campaign(tmp_path: Path) -> None:
    clock = _Clock()
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign(budget=_budget(max_wall_seconds=2.0))
    with DreamScheduler(state, source_checkout=checkout, clock_ns=clock) as scheduler:
        scheduler.submit(campaign)
        clock.advance(3.0)
        assert scheduler.status(campaign.campaign_hash).state is DreamCampaignState.BUDGET_EXHAUSTED
        with pytest.raises(RuntimeError, match="no queued"):
            scheduler.acquire(worker_id="worker", lease_seconds=1.0)


def test_single_writer_lock_and_external_state_boundary(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    with DreamScheduler(state, source_checkout=checkout) as scheduler:
        with pytest.raises(RuntimeError, match="another Dream scheduler"):
            DreamScheduler(state, source_checkout=checkout)
        assert scheduler.list_statuses() == ()
    with pytest.raises(ValueError, match="outside"):
        DreamScheduler(checkout / "state", source_checkout=checkout)

    closed = DreamScheduler(state, source_checkout=checkout)
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.list_statuses()


def test_read_only_inspection_does_not_append_recovery_events(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(state, source_checkout=checkout) as scheduler:
        scheduler.submit(campaign)
    events = state / "dream" / "journal" / "events"
    before = tuple(events.iterdir())
    report = inspect_dream_journal(state, source_checkout=checkout)
    after = tuple(events.iterdir())

    assert report["journal_integrity_verified"] is True
    assert report["hardware_authorized"] is False
    assert before == after


def test_doctor_detects_corrupt_journal_and_checkout_contamination(tmp_path: Path) -> None:
    state = tmp_path / "state"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    campaign = _campaign()
    with DreamScheduler(state, source_checkout=checkout) as scheduler:
        scheduler.submit(campaign)
    event = next((state / "dream" / "journal" / "events").glob("*.json"))
    value = json.loads(event.read_text(encoding="utf-8"))
    value["kind"] = "TAMPERED"
    event.write_text(json.dumps(value), encoding="utf-8")

    report = dream_doctor(state, source_checkout=checkout)
    assert report["ready"] is False
    assert report["checks"]["journal_integrity"] is False
    contaminated = dream_doctor(checkout / "dream-state", source_checkout=checkout)
    assert contaminated["ready"] is False
    assert contaminated["checks"]["external_state_root"] is False

    state_file = tmp_path / "not-a-directory"
    state_file.write_text("occupied", encoding="utf-8")
    unusable = dream_doctor(state_file, source_checkout=checkout)
    assert unusable["ready"] is False
    assert unusable["checks"]["state_root_initializable"] is False


def test_budget_rejects_boolean_integer_spoofing() -> None:
    with pytest.raises(ValueError, match="max_candidates must be an integer"):
        _budget(max_candidates=True)
    with pytest.raises(ValueError, match="max_cpu_rollouts must be an integer"):
        _budget(max_cpu_rollouts=False)
