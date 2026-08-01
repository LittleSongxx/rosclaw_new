from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from rosclaw.dream import DreamBudget, DreamPlanner, DreamPlanRequest, DreamType
from rosclaw.dream.cli import dispatch_dream_argv
from rosclaw.growth import GrowthMetricSpec, MetricDirection, SkillGrowthSpec


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _spec() -> SkillGrowthSpec:
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
        collective_source_ids=(),
        allowed_dream_types=("replay",),
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


def _request() -> DreamPlanRequest:
    return DreamPlanRequest(
        body_hash=_hash("body"),
        parent_policy_hash=_hash("parent"),
        trigger_kind="post_practice",
        trigger_evidence_hashes=(_hash("trigger"),),
        objectives=("improve_recovery",),
        constraint_hashes=(_hash("constraint"),),
        practice_snapshot_hashes=(_hash("practice"),),
        collective_capsule_hashes=(),
        historical_anchor_hashes=(_hash("anchor"),),
        boundary_suite_hashes=(_hash("boundary"),),
        private_holdout_commitment=_hash("holdout"),
        dream_types=(DreamType.REPLAY,),
        learner_ids=("residual.sac",),
        budget=DreamBudget(
            max_gpu_seconds=100.0,
            max_cpu_rollouts=20,
            max_candidates=4,
            max_wall_seconds=300.0,
            max_policy_change=0.05,
            max_anchor_drift=0.02,
        ),
    )


def _request_dict(request: DreamPlanRequest) -> dict[str, Any]:
    return {
        "body_hash": request.body_hash,
        "parent_policy_hash": request.parent_policy_hash,
        "trigger_kind": request.trigger_kind,
        "trigger_evidence_hashes": list(request.trigger_evidence_hashes),
        "objectives": list(request.objectives),
        "constraint_hashes": list(request.constraint_hashes),
        "practice_snapshot_hashes": list(request.practice_snapshot_hashes),
        "collective_capsule_hashes": list(request.collective_capsule_hashes),
        "historical_anchor_hashes": list(request.historical_anchor_hashes),
        "boundary_suite_hashes": list(request.boundary_suite_hashes),
        "private_holdout_commitment": request.private_holdout_commitment,
        "dream_types": [item.value for item in request.dream_types],
        "learner_ids": list(request.learner_ids),
        "budget": request.budget.to_dict(),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cli_plan_submit_acquire_usage_complete_and_inspect(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    spec_path = tmp_path / "spec.json"
    request_path = tmp_path / "request.json"
    campaign_path = tmp_path / "campaign.json"
    state_root = tmp_path / "state"
    token_path = tmp_path / "worker.token"
    _write_json(spec_path, _spec().to_dict())
    _write_json(request_path, _request_dict(_request()))

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "plan",
                "--spec",
                str(spec_path),
                "--request",
                str(request_path),
                "--output",
                str(campaign_path),
                "--source-checkout",
                str(checkout),
            ]
        )
        == 0
    )
    plan_output = json.loads(capsys.readouterr().out)
    campaign_hash = str(plan_output["campaign_hash"])
    assert plan_output["activation_authorized"] is False

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "submit",
                "--state-root",
                str(state_root),
                "--source-checkout",
                str(checkout),
                "--campaign",
                str(campaign_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["state"] == "queued"

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "acquire",
                "--state-root",
                str(state_root),
                "--source-checkout",
                str(checkout),
                "--worker-id",
                "gpu-worker-0",
                "--campaign-hash",
                campaign_hash,
                "--lease-token-file",
                str(token_path),
            ]
        )
        == 0
    )
    lease_output = json.loads(capsys.readouterr().out)
    assert "lease_token" not in lease_output
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "usage",
                "--state-root",
                str(state_root),
                "--source-checkout",
                str(checkout),
                "--campaign-hash",
                campaign_hash,
                "--lease-token-file",
                str(token_path),
                "--gpu-seconds",
                "2.5",
                "--candidates",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["usage"]["candidates"] == 1

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "complete",
                "--state-root",
                str(state_root),
                "--source-checkout",
                str(checkout),
                "--campaign-hash",
                campaign_hash,
                "--lease-token-file",
                str(token_path),
                "--result-manifest-hash",
                _hash("result"),
                "--candidate-artifact-hash",
                _hash("candidate"),
            ]
        )
        == 0
    )
    complete_output = json.loads(capsys.readouterr().out)
    assert complete_output["state"] == "completed"
    assert complete_output["hardware_authorized"] is False

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "inspect",
                "--state-root",
                str(state_root),
                "--source-checkout",
                str(checkout),
                "--campaign-hash",
                campaign_hash,
            ]
        )
        == 0
    )
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["event_count"] == 4
    assert inspection["campaigns"][0]["state"] == "completed"
    assert "lease_token" not in json.dumps(inspection)


def test_cli_doctor_is_read_only_for_new_external_root(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    state_root = tmp_path / "not-created"

    result = dispatch_dream_argv(
        [
            "dream",
            "doctor",
            "--state-root",
            str(state_root),
            "--source-checkout",
            str(checkout),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert result == 0
    assert report["ready"] is True
    assert report["checks"]["planner_cannot_activate"] is True
    assert not state_root.exists()


def test_cli_rejects_unknown_plan_fields(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    spec_path = tmp_path / "spec.json"
    request_path = tmp_path / "request.json"
    output = tmp_path / "campaign.json"
    _write_json(spec_path, _spec().to_dict())
    request = _request_dict(_request())
    request["activation_allowed"] = True
    _write_json(request_path, request)

    result = dispatch_dream_argv(
        [
            "dream",
            "plan",
            "--spec",
            str(spec_path),
            "--request",
            str(request_path),
            "--output",
            str(output),
            "--source-checkout",
            str(checkout),
        ]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert "unknown fields" in captured.err
    assert not output.exists()


def test_cli_plan_refuses_unapproved_output_overwrite(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    spec_path = tmp_path / "spec.json"
    request_path = tmp_path / "request.json"
    output = tmp_path / "campaign.json"
    _write_json(spec_path, _spec().to_dict())
    _write_json(request_path, _request_dict(_request()))
    output.write_text("preserve-me", encoding="utf-8")

    args = [
        "dream",
        "plan",
        "--spec",
        str(spec_path),
        "--request",
        str(request_path),
        "--output",
        str(output),
        "--source-checkout",
        str(checkout),
    ]
    assert dispatch_dream_argv(args) == 2
    assert "pass --force" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "preserve-me"

    assert dispatch_dream_argv([*args, "--force"]) == 0
    capsys.readouterr()
    assert json.loads(output.read_text(encoding="utf-8"))["campaign_hash"]


def test_cli_rejects_unknown_schema_and_boolean_budget_spoofing(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    spec_path = tmp_path / "spec.json"
    request_path = tmp_path / "request.json"
    output = tmp_path / "campaign.json"
    spec = _spec().to_dict()
    spec["schema_version"] = "rosclaw.growth.skill_spec.v999"
    _write_json(spec_path, spec)
    request = _request_dict(_request())
    request["budget"]["max_candidates"] = True
    _write_json(request_path, request)

    result = dispatch_dream_argv(
        [
            "dream",
            "plan",
            "--spec",
            str(spec_path),
            "--request",
            str(request_path),
            "--output",
            str(output),
            "--source-checkout",
            str(checkout),
        ]
    )
    assert result == 2
    assert "schema_version" in capsys.readouterr().err

    spec["schema_version"] = "rosclaw.growth.skill_spec.v1"
    _write_json(spec_path, spec)
    result = dispatch_dream_argv(
        [
            "dream",
            "plan",
            "--spec",
            str(spec_path),
            "--request",
            str(request_path),
            "--output",
            str(output),
            "--source-checkout",
            str(checkout),
        ]
    )
    assert result == 2
    assert "max_candidates must be an integer" in capsys.readouterr().err


def test_cli_rejects_world_readable_worker_token(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    state = tmp_path / "state"
    token = tmp_path / "token"
    campaign = DreamPlanner().plan(_spec(), _request()).campaign
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign.to_dict())
    assert (
        dispatch_dream_argv(
            [
                "dream",
                "submit",
                "--state-root",
                str(state),
                "--source-checkout",
                str(checkout),
                "--campaign",
                str(campaign_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "acquire",
                "--state-root",
                str(state),
                "--source-checkout",
                str(checkout),
                "--worker-id",
                "worker",
                "--lease-token-file",
                str(token),
            ]
        )
        == 0
    )
    lease_output = json.loads(capsys.readouterr().out)
    token.chmod(0o644)

    result = dispatch_dream_argv(
        [
            "dream",
            "usage",
            "--state-root",
            str(state),
            "--source-checkout",
            str(checkout),
            "--campaign-hash",
            str(lease_output["campaign_hash"]),
            "--lease-token-file",
            str(token),
            "--candidates",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert "group/world accessible" in captured.err


def test_cli_does_not_acquire_when_token_destination_exists(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    state = tmp_path / "state"
    token = tmp_path / "token"
    token.write_text("do-not-overwrite", encoding="utf-8")
    campaign = DreamPlanner().plan(_spec(), _request()).campaign
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign.to_dict())
    assert (
        dispatch_dream_argv(
            [
                "dream",
                "submit",
                "--state-root",
                str(state),
                "--source-checkout",
                str(checkout),
                "--campaign",
                str(campaign_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "acquire",
                "--state-root",
                str(state),
                "--source-checkout",
                str(checkout),
                "--worker-id",
                "worker",
                "--lease-token-file",
                str(token),
            ]
        )
        == 2
    )
    capsys.readouterr()
    assert token.read_text(encoding="utf-8") == "do-not-overwrite"

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "inspect",
                "--state-root",
                str(state),
                "--source-checkout",
                str(checkout),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["campaigns"][0]["state"] == "queued"


def test_cli_pauses_lease_when_secret_delivery_fails(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    state = tmp_path / "state"
    campaign = DreamPlanner().plan(_spec(), _request()).campaign
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign.to_dict())
    assert (
        dispatch_dream_argv(
            [
                "dream",
                "submit",
                "--state-root",
                str(state),
                "--source-checkout",
                str(checkout),
                "--campaign",
                str(campaign_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    def fail_delivery(_path: Path, _token: str) -> None:
        raise OSError("simulated secret-store failure")

    monkeypatch.setattr("rosclaw.dream.cli._write_secret", fail_delivery)
    assert (
        dispatch_dream_argv(
            [
                "dream",
                "acquire",
                "--state-root",
                str(state),
                "--source-checkout",
                str(checkout),
                "--worker-id",
                "worker",
                "--lease-token-file",
                str(tmp_path / "token"),
            ]
        )
        == 2
    )
    assert "secret-store failure" in capsys.readouterr().err

    assert (
        dispatch_dream_argv(
            [
                "dream",
                "inspect",
                "--state-root",
                str(state),
                "--source-checkout",
                str(checkout),
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)["campaigns"][0]
    assert status["state"] == "paused"
    assert status["reason"] == "lease token delivery failed"
