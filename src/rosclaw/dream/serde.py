"""Strict JSON decoding for DreamForge public contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rosclaw.dream.contracts import DreamBudget, DreamCampaign, DreamType
from rosclaw.dream.control import DreamPlanRequest
from rosclaw.growth.contracts import (
    GrowthMetricSpec,
    MetricDirection,
    SkillGrowthSpec,
)


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    present = set(value)
    missing = required - present
    unknown = present - required - optional
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _schema(value: Mapping[str, Any], *, expected: str, label: str) -> None:
    supplied = value.get("schema_version", expected)
    if supplied != expected:
        raise ValueError(f"{label} schema_version must be {expected}")


def dream_budget_from_dict(value: Mapping[str, Any]) -> DreamBudget:
    required = {
        "max_gpu_seconds",
        "max_cpu_rollouts",
        "max_candidates",
        "max_wall_seconds",
        "max_policy_change",
        "max_anchor_drift",
    }
    _strict_keys(value, required=required, optional={"schema_version"}, label="dream budget")
    _schema(value, expected="rosclaw.dream.budget.v1", label="dream budget")
    return DreamBudget(
        max_gpu_seconds=_number(value["max_gpu_seconds"], label="max_gpu_seconds"),
        max_cpu_rollouts=_integer(value["max_cpu_rollouts"], label="max_cpu_rollouts"),
        max_candidates=_integer(value["max_candidates"], label="max_candidates"),
        max_wall_seconds=_number(value["max_wall_seconds"], label="max_wall_seconds"),
        max_policy_change=_number(value["max_policy_change"], label="max_policy_change"),
        max_anchor_drift=_number(value["max_anchor_drift"], label="max_anchor_drift"),
    )


def dream_campaign_from_dict(value: Mapping[str, Any]) -> DreamCampaign:
    required = {
        "skill_growth_spec_hash",
        "body_hash",
        "parent_policy_hash",
        "trigger_kind",
        "trigger_evidence_hashes",
        "objectives",
        "constraint_hashes",
        "practice_snapshot_hashes",
        "collective_capsule_hashes",
        "historical_anchor_hashes",
        "boundary_suite_hashes",
        "private_holdout_commitment",
        "dream_types",
        "learner_ids",
        "budget",
    }
    _strict_keys(
        value,
        required=required,
        optional={"schema_version", "hardware_authorized"},
        label="dream campaign",
    )
    _schema(value, expected="rosclaw.dream.campaign.v1", label="dream campaign")
    if value.get("hardware_authorized", False) is not False:
        raise ValueError("dream campaign cannot authorize hardware")
    budget = dream_budget_from_dict(_mapping(value["budget"], label="budget"))
    dream_type_values = _string_tuple(value["dream_types"], label="dream_types")
    try:
        dream_types = tuple(DreamType(item) for item in dream_type_values)
    except ValueError as exc:
        raise ValueError("dream_types contains an unknown value") from exc
    return DreamCampaign(
        skill_growth_spec_hash=_string(
            value["skill_growth_spec_hash"], label="skill_growth_spec_hash"
        ),
        body_hash=_string(value["body_hash"], label="body_hash"),
        parent_policy_hash=_string(value["parent_policy_hash"], label="parent_policy_hash"),
        trigger_kind=_string(value["trigger_kind"], label="trigger_kind"),
        trigger_evidence_hashes=_string_tuple(
            value["trigger_evidence_hashes"], label="trigger_evidence_hashes"
        ),
        objectives=_string_tuple(value["objectives"], label="objectives"),
        constraint_hashes=_string_tuple(value["constraint_hashes"], label="constraint_hashes"),
        practice_snapshot_hashes=_string_tuple(
            value["practice_snapshot_hashes"], label="practice_snapshot_hashes"
        ),
        collective_capsule_hashes=_string_tuple(
            value["collective_capsule_hashes"], label="collective_capsule_hashes"
        ),
        historical_anchor_hashes=_string_tuple(
            value["historical_anchor_hashes"], label="historical_anchor_hashes"
        ),
        boundary_suite_hashes=_string_tuple(
            value["boundary_suite_hashes"], label="boundary_suite_hashes"
        ),
        private_holdout_commitment=_string(
            value["private_holdout_commitment"], label="private_holdout_commitment"
        ),
        dream_types=dream_types,
        learner_ids=_string_tuple(value["learner_ids"], label="learner_ids"),
        budget=budget,
    )


def growth_metric_spec_from_dict(value: Mapping[str, Any]) -> GrowthMetricSpec:
    required = {
        "metric_id",
        "direction",
        "primary",
        "minimum_relative_improvement",
        "confidence_level",
        "require_ci_lower_bound_positive",
    }
    _strict_keys(value, required=required, optional={"schema_version"}, label="growth metric")
    _schema(value, expected="rosclaw.growth.metric_spec.v1", label="growth metric")
    try:
        direction = MetricDirection(_string(value["direction"], label="direction"))
    except ValueError as exc:
        raise ValueError("growth metric direction is unknown") from exc
    if not isinstance(value["primary"], bool) or not isinstance(
        value["require_ci_lower_bound_positive"], bool
    ):
        raise ValueError("growth metric boolean fields must be booleans")
    return GrowthMetricSpec(
        metric_id=_string(value["metric_id"], label="metric_id"),
        direction=direction,
        primary=value["primary"],
        minimum_relative_improvement=_number(
            value["minimum_relative_improvement"], label="minimum_relative_improvement"
        ),
        confidence_level=_number(value["confidence_level"], label="confidence_level"),
        require_ci_lower_bound_positive=value["require_ci_lower_bound_positive"],
    )


def skill_growth_spec_from_dict(value: Mapping[str, Any]) -> SkillGrowthSpec:
    required = {
        "skill_id",
        "adapter_id",
        "body_hashes",
        "capability_ids",
        "observation_contract_hash",
        "action_contract_hash",
        "reward_contract_hash",
        "cost_contract_hash",
        "practice_source_ids",
        "collective_source_ids",
        "allowed_dream_types",
        "allowed_learner_ids",
        "historical_anchor_hashes",
        "boundary_suite_hash",
        "metrics",
        "promotion_profile_hash",
        "rollback_policy_hash",
    }
    _strict_keys(value, required=required, optional={"schema_version"}, label="growth spec")
    _schema(value, expected="rosclaw.growth.skill_spec.v1", label="growth spec")
    metrics_value = value["metrics"]
    if not isinstance(metrics_value, list):
        raise ValueError("metrics must be an array")
    metrics = tuple(
        growth_metric_spec_from_dict(_mapping(metric, label="metric")) for metric in metrics_value
    )
    return SkillGrowthSpec(
        skill_id=_string(value["skill_id"], label="skill_id"),
        adapter_id=_string(value["adapter_id"], label="adapter_id"),
        body_hashes=_string_tuple(value["body_hashes"], label="body_hashes"),
        capability_ids=_string_tuple(value["capability_ids"], label="capability_ids"),
        observation_contract_hash=_string(
            value["observation_contract_hash"], label="observation_contract_hash"
        ),
        action_contract_hash=_string(value["action_contract_hash"], label="action_contract_hash"),
        reward_contract_hash=_string(value["reward_contract_hash"], label="reward_contract_hash"),
        cost_contract_hash=_string(value["cost_contract_hash"], label="cost_contract_hash"),
        practice_source_ids=_string_tuple(
            value["practice_source_ids"], label="practice_source_ids"
        ),
        collective_source_ids=_string_tuple(
            value["collective_source_ids"], label="collective_source_ids"
        ),
        allowed_dream_types=_string_tuple(
            value["allowed_dream_types"], label="allowed_dream_types"
        ),
        allowed_learner_ids=_string_tuple(
            value["allowed_learner_ids"], label="allowed_learner_ids"
        ),
        historical_anchor_hashes=_string_tuple(
            value["historical_anchor_hashes"], label="historical_anchor_hashes"
        ),
        boundary_suite_hash=_string(value["boundary_suite_hash"], label="boundary_suite_hash"),
        metrics=metrics,
        promotion_profile_hash=_string(
            value["promotion_profile_hash"], label="promotion_profile_hash"
        ),
        rollback_policy_hash=_string(value["rollback_policy_hash"], label="rollback_policy_hash"),
    )


def dream_plan_request_from_dict(value: Mapping[str, Any]) -> DreamPlanRequest:
    required = {
        "body_hash",
        "parent_policy_hash",
        "trigger_kind",
        "trigger_evidence_hashes",
        "objectives",
        "constraint_hashes",
        "practice_snapshot_hashes",
        "collective_capsule_hashes",
        "historical_anchor_hashes",
        "boundary_suite_hashes",
        "private_holdout_commitment",
        "dream_types",
        "learner_ids",
        "budget",
    }
    _strict_keys(value, required=required, optional={"schema_version"}, label="dream plan request")
    _schema(value, expected="rosclaw.dream.plan_request.v1", label="dream plan request")
    dream_type_values = _string_tuple(value["dream_types"], label="dream_types")
    try:
        dream_types = tuple(DreamType(item) for item in dream_type_values)
    except ValueError as exc:
        raise ValueError("dream_types contains an unknown value") from exc
    return DreamPlanRequest(
        body_hash=_string(value["body_hash"], label="body_hash"),
        parent_policy_hash=_string(value["parent_policy_hash"], label="parent_policy_hash"),
        trigger_kind=_string(value["trigger_kind"], label="trigger_kind"),
        trigger_evidence_hashes=_string_tuple(
            value["trigger_evidence_hashes"], label="trigger_evidence_hashes"
        ),
        objectives=_string_tuple(value["objectives"], label="objectives"),
        constraint_hashes=_string_tuple(value["constraint_hashes"], label="constraint_hashes"),
        practice_snapshot_hashes=_string_tuple(
            value["practice_snapshot_hashes"], label="practice_snapshot_hashes"
        ),
        collective_capsule_hashes=_string_tuple(
            value["collective_capsule_hashes"], label="collective_capsule_hashes"
        ),
        historical_anchor_hashes=_string_tuple(
            value["historical_anchor_hashes"], label="historical_anchor_hashes"
        ),
        boundary_suite_hashes=_string_tuple(
            value["boundary_suite_hashes"], label="boundary_suite_hashes"
        ),
        private_holdout_commitment=_string(
            value["private_holdout_commitment"], label="private_holdout_commitment"
        ),
        dream_types=dream_types,
        learner_ids=_string_tuple(value["learner_ids"], label="learner_ids"),
        budget=dream_budget_from_dict(_mapping(value["budget"], label="budget")),
    )


__all__ = [
    "dream_budget_from_dict",
    "dream_campaign_from_dict",
    "dream_plan_request_from_dict",
    "growth_metric_spec_from_dict",
    "skill_growth_spec_from_dict",
]
