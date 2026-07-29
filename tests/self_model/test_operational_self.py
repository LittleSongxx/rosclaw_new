from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw.self_model.agency import (
    AgencyClass,
    AgencyEstimator,
    AgencyEvidence,
)
from rosclaw.self_model.forward_model import (
    ForwardAction,
    ForwardModelInput,
    ForwardState,
    HybridForwardSelfModel,
)
from rosclaw.self_model.prediction_monitor import (
    AdaptationState,
    AdaptationTrigger,
    AdaptationTriggerConfig,
    PredictionResiduals,
)
from rosclaw.self_model.regime import (
    RegimeEncoder,
    RegimeMemory,
    RegimeObservation,
)
from tests.continual.helpers import digest


def _forward_state(*, joint_offset: float = 0.0) -> ForwardState:
    return ForwardState(
        joint_position={"hip": 0.1 + joint_offset, "knee": -0.2},
        joint_velocity={"hip": 0.05, "knee": -0.02},
        pelvis_position=(0.0, 0.0, 0.78),
        pelvis_velocity=(0.1, 0.0, 0.0),
        com_position=(0.0, 0.01, 0.72),
        foot_contact=(1.0, 0.2),
        ball_position=(0.3, 0.0, 0.05),
        ball_velocity=(0.0, 0.0, 0.0),
        energy_state=0.9,
        balance_margin=0.04,
    )


def _forward_input() -> ForwardModelInput:
    return ForwardModelInput(
        state=_forward_state(),
        action=ForwardAction(
            joint_acceleration={"hip": 0.2, "knee": -0.1},
            ball_impulse=(0.4, 0.0, 0.05),
        ),
        dt_seconds=0.02,
        phase_progress=0.5,
        contact_mode=(1.0, 0.0),
    )


def test_forward_model_starts_analytical_and_learns_only_behind_shadow_gate() -> None:
    model = HybridForwardSelfModel(("hip", "knee"), learning_rate=0.1)
    model_input = _forward_input()
    analytical = model.predict(model_input).analytical_state
    actual = replace(
        analytical,
        joint_position={
            "hip": analytical.joint_position["hip"] + 0.03,
            "knee": analytical.joint_position["knee"],
        },
    )
    original_hash = model.model_hash

    blocked = model.learn_transition(model_input, actual, shadow_learning=False)
    updates = [model.learn_transition(model_input, actual, shadow_learning=True) for _ in range(80)]
    prediction = model.predict(model_input)

    assert not blocked.trained
    assert blocked.error_before == blocked.error_after
    assert updates[-1].error_after < updates[0].error_before
    assert model.model_hash != original_hash
    assert prediction.neural_residual_norm <= model.residual_limit * (model._output_size**0.5)
    assert 0.0 <= prediction.fall_risk <= 1.0


def test_forward_model_checkpoint_is_exact_and_configuration_bound() -> None:
    source = HybridForwardSelfModel(("hip", "knee"))
    source.learn_transition(
        _forward_input(),
        replace(source.predict(_forward_input()).next_state, balance_margin=0.02),
        shadow_learning=True,
    )
    checkpoint = source.checkpoint()
    recovered = HybridForwardSelfModel(("hip", "knee"))

    recovered.restore_checkpoint(checkpoint)

    assert recovered.model_hash == source.model_hash
    assert (
        recovered.predict(_forward_input()).next_state.to_dict()
        == source.predict(_forward_input()).next_state.to_dict()
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        HybridForwardSelfModel(("ankle",)).restore_checkpoint(checkpoint)


def _residual(value: float, *, episode: str = "episode") -> PredictionResiduals:
    return PredictionResiduals(
        body_state=value,
        contact_outcome=value,
        contact_mode=value,
        control_latency=value,
        energy=value,
        task_performance=value,
        timestamp_ns=1,
        episode_id=episode,
    )


def test_adaptation_trigger_ignores_noise_and_enforces_shadow_safety_gate() -> None:
    config = AdaptationTriggerConfig(
        suspected_persistence=2,
        confirmed_persistence=2,
        recovery_persistence=2,
        shadow_min_samples=10,
    )
    transient = AdaptationTrigger(config)
    transient.observe(_residual(0.4))
    receipt = transient.observe(_residual(0.1))
    assert receipt.state is AdaptationState.NORMAL

    trigger = AdaptationTrigger(config)
    trigger.observe(_residual(0.6))
    trigger.observe(_residual(0.6))
    trigger.observe(_residual(0.8))
    confirmed = trigger.observe(_residual(0.8))
    assert confirmed.state is AdaptationState.CONFIRMED_SHIFT
    shadow = trigger.begin_shadow_learning()
    rejected = trigger.candidate_update(
        sample_count=100,
        target_improvement=0.2,
        anchor_degradation=0.0,
        critical_safety_regressions=1,
        converged=True,
    )

    assert shadow.learning_enabled
    assert rejected.state is AdaptationState.ROLLBACK
    assert not rejected.learning_enabled


def test_adaptation_trigger_requires_retention_before_consolidation() -> None:
    trigger = AdaptationTrigger(
        AdaptationTriggerConfig(
            suspected_persistence=1,
            confirmed_persistence=1,
            shadow_min_samples=10,
        )
    )
    trigger.observe(_residual(0.6))
    trigger.observe(_residual(0.8))
    trigger.begin_shadow_learning()
    incomplete = trigger.candidate_update(
        sample_count=100,
        target_improvement=0.2,
        anchor_degradation=0.2,
        critical_safety_regressions=0,
        converged=True,
    )
    ready = trigger.candidate_update(
        sample_count=100,
        target_improvement=0.2,
        anchor_degradation=0.01,
        critical_safety_regressions=0,
        converged=True,
    )
    consolidated = trigger.consolidate(matched_gate_passed=True)

    assert incomplete.state is AdaptationState.SHADOW_LEARNING
    assert ready.state is AdaptationState.CANDIDATE_READY
    assert consolidated.state is AdaptationState.CONSOLIDATED


def _regime_observation(
    episode_id: str,
    support_friction: float,
    *,
    timestamp_ns: int,
) -> RegimeObservation:
    return RegimeObservation(
        episode_id=episode_id,
        timestamp_ns=timestamp_ns,
        support_friction=support_friction,
        ball_friction=0.4,
        control_latency_ms=8.0,
        joint_zero_bias={"hip": 0.01, "knee": -0.02},
        motor_gain={"hip": 0.98, "knee": 1.02},
        payload_kg=0.2,
        disturbance_magnitude=0.1,
        sensor_confidence=0.95,
    )


def test_regime_encoder_separates_timescales_and_reuses_known_regime() -> None:
    encoder = RegimeEncoder(("hip", "knee"))
    for index in range(5):
        encoder.observe(_regime_observation("episode-a", 0.7 + index * 0.01, timestamp_ns=index))
    first = encoder.end_episode("episode-a")
    assert first.persistent.observation_count == 1
    assert first.episode.observation_count == 0
    assert first.fast.observation_count == 5

    memory = RegimeMemory()
    created = memory.assign(first.persistent)
    reused = memory.assign(first.persistent)

    assert not created.reused
    assert reused.reused
    assert reused.expert_id == created.expert_id


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.95, 0.05, 0.0, 0.0), AgencyClass.SELF_CAUSED),
        ((0.2, 0.8, 0.95, 0.0), AgencyClass.EXTERNAL_DISTURBANCE),
        ((0.2, 0.8, 0.0, 0.95), AgencyClass.SENSOR_FAULT),
        ((0.0, 0.1, 0.0, 0.0), AgencyClass.UNKNOWN),
    ],
)
def test_agency_estimator_distinguishes_operational_causes(
    values: tuple[float, float, float, float], expected: AgencyClass
) -> None:
    action, error, external, sensor = values
    evidence = AgencyEvidence(
        action_magnitude=action,
        prediction_error=error,
        external_force_evidence=external,
        sensor_inconsistency=sensor,
        action_hash=digest("action"),
        predicted_outcome_hash=digest("predicted"),
        observed_outcome_hash=digest("observed"),
        timestamp_ns=1,
    )

    estimate = AgencyEstimator().estimate(evidence)

    assert estimate.classification is expected
    assert sum(estimate.probabilities.values()) == pytest.approx(1.0)
