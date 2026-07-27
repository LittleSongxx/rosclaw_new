"""Multi-timescale physical-regime inference and reusable regime memory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.self_model.contracts import ScalarBelief


def _finite_mapping(value: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    normalized = {str(key): float(item) for key, item in value.items()}
    if not normalized or any(
        not key.strip() or not math.isfinite(item) for key, item in normalized.items()
    ):
        raise ValueError(f"{label} must be a non-empty finite mapping")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class RegimeObservation:
    episode_id: str
    timestamp_ns: int
    support_friction: float
    ball_friction: float
    control_latency_ms: float
    joint_zero_bias: Mapping[str, float]
    motor_gain: Mapping[str, float]
    payload_kg: float
    disturbance_magnitude: float
    sensor_confidence: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.support_friction,
            self.ball_friction,
            self.control_latency_ms,
            self.payload_kg,
            self.disturbance_magnitude,
            self.sensor_confidence,
        )
        if not self.episode_id.strip() or self.timestamp_ns < 0:
            raise ValueError("regime observation requires an episode and timestamp")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("regime observation values must be finite")
        if min(values[:-1]) < 0.0 or not 0.0 <= self.sensor_confidence <= 1.0:
            raise ValueError("regime magnitudes must be non-negative and confidence normalized")
        zero = _finite_mapping(self.joint_zero_bias, label="joint zero bias")
        gain = _finite_mapping(self.motor_gain, label="motor gain")
        if tuple(zero) != tuple(gain):
            raise ValueError("joint zero-bias and motor-gain order must match")
        if any(value <= 0.0 for value in gain.values()):
            raise ValueError("motor gains must be positive")
        object.__setattr__(self, "joint_zero_bias", zero)
        object.__setattr__(self, "motor_gain", gain)


@dataclass(frozen=True)
class RegimeBelief:
    timescale: str
    observation_count: int
    support_friction: ScalarBelief
    ball_friction: ScalarBelief
    control_latency: ScalarBelief
    joint_zero_bias: Mapping[str, ScalarBelief]
    motor_gain: Mapping[str, ScalarBelief]
    payload: ScalarBelief
    disturbance: ScalarBelief
    uncertainty: float
    schema_version: str = "rosclaw.self.regime_belief.v1"

    def __post_init__(self) -> None:
        if self.timescale not in {"fast", "episode", "persistent"}:
            raise ValueError("unsupported regime timescale")
        if self.observation_count < 0 or not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("invalid regime observation count or uncertainty")
        zero = MappingProxyType(dict(self.joint_zero_bias))
        gain = MappingProxyType(dict(self.motor_gain))
        if not zero or tuple(zero) != tuple(gain):
            raise ValueError("regime joint beliefs must have matching non-empty order")
        object.__setattr__(self, "joint_zero_bias", zero)
        object.__setattr__(self, "motor_gain", gain)

    @property
    def regime_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timescale": self.timescale,
            "observation_count": self.observation_count,
            "support_friction": self.support_friction.to_dict(),
            "ball_friction": self.ball_friction.to_dict(),
            "control_latency": self.control_latency.to_dict(),
            "joint_zero_bias": {
                key: value.to_dict() for key, value in self.joint_zero_bias.items()
            },
            "motor_gain": {key: value.to_dict() for key, value in self.motor_gain.items()},
            "payload": self.payload.to_dict(),
            "disturbance": self.disturbance.to_dict(),
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class RegimeEstimate:
    fast: RegimeBelief
    episode: RegimeBelief
    persistent: RegimeBelief
    estimate_hash: str
    schema_version: str = "rosclaw.self.regime_estimate.v1"


class _EWMA:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.mean = 0.0
        self.variance = 0.0
        self.count = 0
        self.confidence = 0.0

    def update(self, value: float, confidence: float) -> None:
        if self.count == 0:
            self.mean = value
        else:
            delta = value - self.mean
            self.mean += self.alpha * delta
            self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * delta * delta)
        self.confidence = (1.0 - self.alpha) * self.confidence + self.alpha * confidence
        self.count += 1

    def belief(self, unit: str) -> ScalarBelief:
        return ScalarBelief(
            mean=self.mean,
            standard_deviation=math.sqrt(max(0.0, self.variance)),
            confidence=min(1.0, self.confidence * (1.0 - math.exp(-self.count / 4.0))),
            unit=unit,
        )


class _TimescaleState:
    def __init__(self, joint_names: tuple[str, ...], alpha: float) -> None:
        self.support = _EWMA(alpha)
        self.ball = _EWMA(alpha)
        self.latency = _EWMA(alpha)
        self.payload = _EWMA(alpha)
        self.disturbance = _EWMA(alpha)
        self.zero = {name: _EWMA(alpha) for name in joint_names}
        self.gain = {name: _EWMA(alpha) for name in joint_names}

    @property
    def count(self) -> int:
        return self.support.count

    def update(self, observation: RegimeObservation) -> None:
        confidence = observation.sensor_confidence
        self.support.update(observation.support_friction, confidence)
        self.ball.update(observation.ball_friction, confidence)
        self.latency.update(observation.control_latency_ms, confidence)
        self.payload.update(observation.payload_kg, confidence)
        self.disturbance.update(observation.disturbance_magnitude, confidence)
        for name in self.zero:
            self.zero[name].update(observation.joint_zero_bias[name], confidence)
            self.gain[name].update(observation.motor_gain[name], confidence)

    def belief(self, timescale: str) -> RegimeBelief:
        confidence_values = [
            self.support.belief("coefficient").confidence,
            self.ball.belief("coefficient").confidence,
            self.latency.belief("ms").confidence,
            self.payload.belief("kg").confidence,
            self.disturbance.belief("normalized").confidence,
        ]
        uncertainty = 1.0 - sum(confidence_values) / len(confidence_values)
        return RegimeBelief(
            timescale=timescale,
            observation_count=self.count,
            support_friction=self.support.belief("coefficient"),
            ball_friction=self.ball.belief("coefficient"),
            control_latency=self.latency.belief("ms"),
            joint_zero_bias={name: value.belief("rad") for name, value in self.zero.items()},
            motor_gain={name: value.belief("ratio") for name, value in self.gain.items()},
            payload=self.payload.belief("kg"),
            disturbance=self.disturbance.belief("normalized"),
            uncertainty=max(0.0, min(1.0, uncertainty)),
        )


class RegimeEncoder:
    """Separate quick response, episode context, and persistent identity drift."""

    def __init__(self, joint_names: tuple[str, ...]) -> None:
        if not joint_names or len(joint_names) != len(set(joint_names)):
            raise ValueError("regime encoder requires unique joint names")
        self.joint_names = tuple(joint_names)
        self._fast = _TimescaleState(self.joint_names, 0.45)
        self._episode = _TimescaleState(self.joint_names, 0.12)
        self._persistent = _TimescaleState(self.joint_names, 0.03)
        self._active_episode: str | None = None
        self._episode_observations: list[RegimeObservation] = []

    def observe(self, observation: RegimeObservation) -> RegimeEstimate:
        if tuple(observation.joint_zero_bias) != self.joint_names:
            raise ValueError("regime observation does not match encoder joints")
        if self._active_episode not in {None, observation.episode_id}:
            raise RuntimeError("end the active regime episode before starting another")
        self._active_episode = observation.episode_id
        self._episode_observations.append(observation)
        self._fast.update(observation)
        self._episode.update(observation)
        return self.estimate()

    def end_episode(self, episode_id: str) -> RegimeEstimate:
        if self._active_episode != episode_id or not self._episode_observations:
            raise RuntimeError("cannot end an inactive regime episode")
        averaged = self._average_episode(self._episode_observations)
        self._persistent.update(averaged)
        self._active_episode = None
        self._episode_observations = []
        self._episode = _TimescaleState(self.joint_names, 0.12)
        return self.estimate()

    def estimate(self) -> RegimeEstimate:
        fast = self._fast.belief("fast")
        episode = self._episode.belief("episode")
        persistent = self._persistent.belief("persistent")
        material = {
            "fast_hash": fast.regime_hash,
            "episode_hash": episode.regime_hash,
            "persistent_hash": persistent.regime_hash,
            "active_episode": self._active_episode,
        }
        return RegimeEstimate(
            fast=fast,
            episode=episode,
            persistent=persistent,
            estimate_hash=canonical_hash(material),
        )

    def _average_episode(self, observations: list[RegimeObservation]) -> RegimeObservation:
        count = len(observations)

        def mean(values: list[float]) -> float:
            return sum(values) / count

        return RegimeObservation(
            episode_id=observations[0].episode_id,
            timestamp_ns=observations[-1].timestamp_ns,
            support_friction=mean([item.support_friction for item in observations]),
            ball_friction=mean([item.ball_friction for item in observations]),
            control_latency_ms=mean([item.control_latency_ms for item in observations]),
            joint_zero_bias={
                name: mean([item.joint_zero_bias[name] for item in observations])
                for name in self.joint_names
            },
            motor_gain={
                name: mean([item.motor_gain[name] for item in observations])
                for name in self.joint_names
            },
            payload_kg=mean([item.payload_kg for item in observations]),
            disturbance_magnitude=mean([item.disturbance_magnitude for item in observations]),
            sensor_confidence=mean([item.sensor_confidence for item in observations]),
        )


@dataclass(frozen=True)
class RegimeExpertAssignment:
    expert_id: str
    reused: bool
    distance: float
    regime_hash: str
    reason: str
    schema_version: str = "rosclaw.self.regime_expert_assignment.v1"


class RegimeMemory:
    """Content-addressed regime prototypes that avoid relearning known dynamics."""

    def __init__(self, *, reuse_threshold: float = 0.18) -> None:
        if not 0.0 < reuse_threshold <= 1.0:
            raise ValueError("regime reuse threshold must be in (0, 1]")
        self.reuse_threshold = reuse_threshold
        self._prototypes: dict[str, RegimeBelief] = {}

    def assign(self, belief: RegimeBelief) -> RegimeExpertAssignment:
        if belief.timescale != "persistent":
            raise ValueError("only persistent regimes may select durable experts")
        distances = {
            expert_id: self._distance(belief, prototype)
            for expert_id, prototype in self._prototypes.items()
        }
        if distances:
            expert_id = min(distances, key=distances.__getitem__)
            distance = distances[expert_id]
            if distance <= self.reuse_threshold:
                return RegimeExpertAssignment(
                    expert_id=expert_id,
                    reused=True,
                    distance=distance,
                    regime_hash=belief.regime_hash,
                    reason="known physical regime reused its existing expert",
                )
        expert_id = f"regime-{belief.regime_hash.removeprefix('sha256:')[:16]}"
        self._prototypes[expert_id] = belief
        return RegimeExpertAssignment(
            expert_id=expert_id,
            reused=False,
            distance=min(distances.values(), default=1.0),
            regime_hash=belief.regime_hash,
            reason="novel persistent regime requires a shadow-trained expert",
        )

    def _distance(self, left: RegimeBelief, right: RegimeBelief) -> float:
        if tuple(left.joint_zero_bias) != tuple(right.joint_zero_bias):
            raise ValueError("regime prototypes must use the same joint order")
        pairs = (
            (left.support_friction.mean, right.support_friction.mean, 1.5),
            (left.ball_friction.mean, right.ball_friction.mean, 1.5),
            (left.control_latency.mean, right.control_latency.mean, 50.0),
            (left.payload.mean, right.payload.mean, 10.0),
            (left.disturbance.mean, right.disturbance.mean, 1.0),
        )
        scalar = [min(1.0, abs(a - b) / scale) for a, b, scale in pairs]
        joints = [
            min(1.0, abs(left.joint_zero_bias[name].mean - value.mean) / 0.2)
            for name, value in right.joint_zero_bias.items()
        ] + [
            min(1.0, abs(left.motor_gain[name].mean - value.mean) / 0.5)
            for name, value in right.motor_gain.items()
        ]
        return sum([*scalar, *joints]) / len([*scalar, *joints])


__all__ = [
    "RegimeBelief",
    "RegimeEncoder",
    "RegimeEstimate",
    "RegimeExpertAssignment",
    "RegimeMemory",
    "RegimeObservation",
]
