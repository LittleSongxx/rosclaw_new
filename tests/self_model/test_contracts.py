from __future__ import annotations

import pytest

from rosclaw.self_model import (
    AgencyAssessment,
    CapabilityBelief,
    ScalarBelief,
    SelfIdentity,
    SelfStateSnapshot,
)
from tests.continual.helpers import digest


def _belief(mean: float, unit: str) -> ScalarBelief:
    return ScalarBelief(mean, 0.01, 0.9, unit)


def test_self_state_binds_body_health_uncertainty_and_capabilities() -> None:
    identity = SelfIdentity(
        body_hash=digest("body"),
        sensor_layout_hash=digest("sensors"),
        actuator_layout_hash=digest("actuators"),
        safety_kernel_hash=digest("safety"),
        controller_lineage=(digest("controller-0"),),
        current_policy_versions={"g1_target_kick": 0},
    )
    state = SelfStateSnapshot(
        identity_hash=identity.identity_hash,
        body_hash=identity.body_hash,
        sequence=1,
        timestamp_ns=10,
        joint_health={"right_hip": 0.8},
        motor_gain_beliefs={"right_hip": _belief(0.8, "ratio")},
        joint_zero_bias_beliefs={"right_hip": _belief(0.01, "rad")},
        latency_belief=_belief(20.0, "ms"),
        friction_belief=_belief(0.75, "coefficient"),
        payload_belief=_belief(0.0, "kg"),
        balance_margin=0.04,
        energy_state=0.8,
        sensor_quality={"imu": 0.95},
        capabilities={"target_kick": CapabilityBelief(0.7, 0.2, 20, 0)},
    )

    assert state.body_hash == identity.body_hash
    assert len(state.snapshot_hash) == 71
    assert state.to_dict()["capabilities"]["target_kick"]["success_probability"] == 0.7


def test_discovered_self_core_cannot_exist_without_causal_evidence() -> None:
    with pytest.raises(ValueError, match="causal evidence"):
        SelfIdentity(
            body_hash=digest("body"),
            sensor_layout_hash=digest("sensors"),
            actuator_layout_hash=digest("actuators"),
            safety_kernel_hash=digest("safety"),
            controller_lineage=(digest("controller"),),
            current_policy_versions={"stand": 0},
            discovered_self_core_hash=digest("cluster"),
        )


def test_agency_is_a_calibrated_attribution_contract() -> None:
    assessment = AgencyAssessment(
        self_state_hash=digest("self"),
        action_hash=digest("action"),
        predicted_outcome_hash=digest("prediction"),
        observed_outcome_hash=digest("observation"),
        self_caused_probability=0.1,
        external_disturbance_probability=0.8,
        sensor_fault_probability=0.1,
    )

    assert len(assessment.assessment_hash) == 71
