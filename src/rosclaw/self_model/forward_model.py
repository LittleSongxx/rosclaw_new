"""Hybrid analytical and bounded neural forward model for embodied prediction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash


def _vector3(value: tuple[float, float, float], *, label: str) -> tuple[float, float, float]:
    normalized = tuple(float(item) for item in value)
    if len(normalized) != 3 or any(not math.isfinite(item) for item in normalized):
        raise ValueError(f"{label} must contain three finite values")
    return normalized


def _finite_mapping(value: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    normalized = {str(key): float(item) for key, item in value.items()}
    if not normalized or any(
        not key.strip() or not math.isfinite(item) for key, item in normalized.items()
    ):
        raise ValueError(f"{label} must be a non-empty finite mapping")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class ForwardState:
    joint_position: Mapping[str, float]
    joint_velocity: Mapping[str, float]
    pelvis_position: tuple[float, float, float]
    pelvis_velocity: tuple[float, float, float]
    com_position: tuple[float, float, float]
    foot_contact: tuple[float, float]
    ball_position: tuple[float, float, float]
    ball_velocity: tuple[float, float, float]
    energy_state: float
    balance_margin: float

    def __post_init__(self) -> None:
        position = _finite_mapping(self.joint_position, label="joint position")
        velocity = _finite_mapping(self.joint_velocity, label="joint velocity")
        if tuple(position) != tuple(velocity):
            raise ValueError("joint position and velocity order must match")
        contact = tuple(float(item) for item in self.foot_contact)
        if len(contact) != 2 or any(
            not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in contact
        ):
            raise ValueError("foot contact must contain two probabilities")
        if not math.isfinite(self.energy_state) or not 0.0 <= self.energy_state <= 1.0:
            raise ValueError("energy state must be in [0, 1]")
        if not math.isfinite(self.balance_margin):
            raise ValueError("balance margin must be finite")
        object.__setattr__(self, "joint_position", position)
        object.__setattr__(self, "joint_velocity", velocity)
        object.__setattr__(
            self, "pelvis_position", _vector3(self.pelvis_position, label="pelvis position")
        )
        object.__setattr__(
            self, "pelvis_velocity", _vector3(self.pelvis_velocity, label="pelvis velocity")
        )
        object.__setattr__(self, "com_position", _vector3(self.com_position, label="COM position"))
        object.__setattr__(self, "foot_contact", contact)
        object.__setattr__(
            self, "ball_position", _vector3(self.ball_position, label="ball position")
        )
        object.__setattr__(
            self, "ball_velocity", _vector3(self.ball_velocity, label="ball velocity")
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.joint_position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_position": dict(self.joint_position),
            "joint_velocity": dict(self.joint_velocity),
            "pelvis_position": list(self.pelvis_position),
            "pelvis_velocity": list(self.pelvis_velocity),
            "com_position": list(self.com_position),
            "foot_contact": list(self.foot_contact),
            "ball_position": list(self.ball_position),
            "ball_velocity": list(self.ball_velocity),
            "energy_state": self.energy_state,
            "balance_margin": self.balance_margin,
        }


@dataclass(frozen=True)
class ForwardAction:
    joint_acceleration: Mapping[str, float]
    pelvis_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ball_impulse: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_acceleration",
            _finite_mapping(self.joint_acceleration, label="joint acceleration"),
        )
        object.__setattr__(
            self,
            "pelvis_acceleration",
            _vector3(self.pelvis_acceleration, label="pelvis acceleration"),
        )
        object.__setattr__(
            self,
            "ball_impulse",
            _vector3(self.ball_impulse, label="ball impulse"),
        )


@dataclass(frozen=True)
class ForwardModelInput:
    state: ForwardState
    action: ForwardAction
    dt_seconds: float
    phase_progress: float
    contact_mode: tuple[float, float]

    def __post_init__(self) -> None:
        if tuple(self.action.joint_acceleration) != self.state.joint_names:
            raise ValueError("action joint order must match state joint order")
        if not math.isfinite(self.dt_seconds) or not 0.0 < self.dt_seconds <= 0.1:
            raise ValueError("forward-model dt must be in (0, 0.1] seconds")
        if not math.isfinite(self.phase_progress) or not 0.0 <= self.phase_progress <= 1.0:
            raise ValueError("phase progress must be in [0, 1]")
        mode = tuple(float(item) for item in self.contact_mode)
        if len(mode) != 2 or any(
            not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in mode
        ):
            raise ValueError("contact mode must contain two probabilities")
        object.__setattr__(self, "contact_mode", mode)


@dataclass(frozen=True)
class ForwardPrediction:
    next_state: ForwardState
    fall_risk: float
    analytical_state: ForwardState
    neural_residual_norm: float
    model_hash: str
    schema_version: str = "rosclaw.self.forward_prediction.v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.fall_risk <= 1.0:
            raise ValueError("fall risk must be in [0, 1]")
        if not math.isfinite(self.neural_residual_norm) or self.neural_residual_norm < 0.0:
            raise ValueError("neural residual norm must be finite and non-negative")


@dataclass(frozen=True)
class ForwardLearningReceipt:
    trained: bool
    error_before: float
    error_after: float
    model_hash: str
    reason: str
    schema_version: str = "rosclaw.self.forward_learning_receipt.v1"


class HybridForwardSelfModel:
    """Physics prior plus a small bounded residual network.

    The hidden layer is deterministic and fixed; online adaptation updates only
    the zero-initialized output layer.  This preserves a stable analytical
    fallback and makes the plastic part bounded and easy to roll back.
    """

    def __init__(
        self,
        joint_names: tuple[str, ...],
        *,
        hidden_size: int = 24,
        residual_limit: float = 0.05,
        learning_rate: float = 0.02,
        damping: float = 0.08,
        seed: int = 7,
    ) -> None:
        if not joint_names or len(joint_names) != len(set(joint_names)):
            raise ValueError("forward model requires unique joint names")
        if hidden_size <= 0 or not 0.0 < residual_limit <= 0.25:
            raise ValueError("invalid forward-model hidden size or residual limit")
        if not 0.0 < learning_rate <= 0.2 or not 0.0 <= damping <= 1.0:
            raise ValueError("invalid forward-model learning rate or damping")
        self.joint_names = tuple(joint_names)
        self.hidden_size = hidden_size
        self.residual_limit = residual_limit
        self.learning_rate = learning_rate
        self.damping = damping
        self._output_size = 2 * len(joint_names) + 19
        input_size = 3 * len(joint_names) + 29
        rng = np.random.default_rng(seed)
        self._hidden_weight = rng.normal(0.0, 0.12, size=(hidden_size, input_size))
        self._hidden_bias = rng.normal(0.0, 0.02, size=hidden_size)
        self._output_weight = np.zeros((self._output_size, hidden_size), dtype=np.float64)
        self._output_bias = np.zeros(self._output_size, dtype=np.float64)

    @property
    def model_hash(self) -> str:
        return canonical_hash(
            {
                "joint_names": list(self.joint_names),
                "hidden_size": self.hidden_size,
                "residual_limit": self.residual_limit,
                "learning_rate": self.learning_rate,
                "damping": self.damping,
                "hidden_weight_hash": canonical_hash(self._hidden_weight.tolist()),
                "hidden_bias_hash": canonical_hash(self._hidden_bias.tolist()),
                "output_weight_hash": canonical_hash(self._output_weight.tolist()),
                "output_bias_hash": canonical_hash(self._output_bias.tolist()),
            }
        )

    def predict(self, model_input: ForwardModelInput) -> ForwardPrediction:
        self._validate_input(model_input)
        analytical = self._analytical(model_input)
        residual, _ = self._residual(model_input)
        predicted, risk = self._decode(self._encode_state(analytical), residual)
        return ForwardPrediction(
            next_state=predicted,
            fall_risk=risk,
            analytical_state=analytical,
            neural_residual_norm=float(np.linalg.norm(residual)),
            model_hash=self.model_hash,
        )

    def learn_transition(
        self,
        model_input: ForwardModelInput,
        actual_next_state: ForwardState,
        *,
        shadow_learning: bool,
    ) -> ForwardLearningReceipt:
        self._validate_input(model_input)
        self._validate_state(actual_next_state)
        analytical = self._analytical(model_input)
        base = self._encode_state(analytical)
        target = self._encode_state(actual_next_state)
        residual, hidden = self._residual(model_input)
        before = float(np.mean(np.square(base + residual - target)))
        if not shadow_learning:
            return ForwardLearningReceipt(
                trained=False,
                error_before=before,
                error_after=before,
                model_hash=self.model_hash,
                reason="learning gate closed; analytical prediction remained active",
            )
        raw = self._output_weight @ hidden + self._output_bias
        derivative = self.residual_limit * (1.0 - np.square(np.tanh(raw)))
        gradient = 2.0 * (base + residual - target) / self._output_size
        raw_gradient = np.clip(gradient * derivative, -1.0, 1.0)
        self._output_weight -= self.learning_rate * np.outer(raw_gradient, hidden)
        self._output_bias -= self.learning_rate * raw_gradient
        after_residual, _ = self._residual(model_input)
        after = float(np.mean(np.square(base + after_residual - target)))
        return ForwardLearningReceipt(
            trained=True,
            error_before=before,
            error_after=after,
            model_hash=self.model_hash,
            reason="bounded residual output layer updated in shadow mode",
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.self.forward_model_checkpoint.v1",
            "joint_names": list(self.joint_names),
            "hidden_size": self.hidden_size,
            "residual_limit": self.residual_limit,
            "learning_rate": self.learning_rate,
            "damping": self.damping,
            "hidden_weight": self._hidden_weight.tolist(),
            "hidden_bias": self._hidden_bias.tolist(),
            "output_weight": self._output_weight.tolist(),
            "output_bias": self._output_bias.tolist(),
        }

    def restore_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("schema_version") != "rosclaw.self.forward_model_checkpoint.v1":
            raise ValueError("unsupported forward-model checkpoint")
        immutable = {
            "joint_names": list(self.joint_names),
            "hidden_size": self.hidden_size,
            "residual_limit": self.residual_limit,
            "learning_rate": self.learning_rate,
            "damping": self.damping,
        }
        if any(checkpoint.get(key) != value for key, value in immutable.items()):
            raise ValueError("forward-model checkpoint configuration mismatch")
        arrays = (
            ("hidden_weight", self._hidden_weight.shape),
            ("hidden_bias", self._hidden_bias.shape),
            ("output_weight", self._output_weight.shape),
            ("output_bias", self._output_bias.shape),
        )
        for name, shape in arrays:
            value = np.asarray(checkpoint[name], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid forward-model checkpoint tensor: {name}")
            setattr(self, f"_{name}", value.copy())

    def _validate_input(self, model_input: ForwardModelInput) -> None:
        self._validate_state(model_input.state)
        if tuple(model_input.action.joint_acceleration) != self.joint_names:
            raise ValueError("forward action does not match model joints")

    def _validate_state(self, state: ForwardState) -> None:
        if state.joint_names != self.joint_names:
            raise ValueError("forward state does not match model joints")

    def _analytical(self, model_input: ForwardModelInput) -> ForwardState:
        state = model_input.state
        dt = model_input.dt_seconds
        acceleration = np.array(list(model_input.action.joint_acceleration.values()))
        velocity = np.array(list(state.joint_velocity.values()))
        position = np.array(list(state.joint_position.values()))
        next_velocity = velocity + (acceleration - self.damping * velocity) * dt
        next_position = position + next_velocity * dt
        pelvis_acceleration = np.asarray(model_input.action.pelvis_acceleration)
        pelvis_velocity = np.asarray(state.pelvis_velocity) + pelvis_acceleration * dt
        pelvis_position = np.asarray(state.pelvis_position) + pelvis_velocity * dt
        com_position = np.asarray(state.com_position) + pelvis_velocity * dt
        ball_velocity = np.asarray(state.ball_velocity) + np.asarray(
            model_input.action.ball_impulse
        )
        ball_velocity *= max(0.0, 1.0 - 0.15 * dt)
        ball_position = np.asarray(state.ball_position) + ball_velocity * dt
        work = float(np.mean(np.abs(acceleration * next_velocity)))
        energy = float(np.clip(state.energy_state - 0.002 * work * dt, 0.0, 1.0))
        contact = tuple(
            float(np.clip(0.7 * old + 0.3 * mode, 0.0, 1.0))
            for old, mode in zip(state.foot_contact, model_input.contact_mode, strict=True)
        )
        lateral_com = abs(float(com_position[1] - pelvis_position[1]))
        support = 0.02 + 0.08 * max(contact)
        balance_margin = support - lateral_com
        return ForwardState(
            joint_position=dict(zip(self.joint_names, next_position.tolist(), strict=True)),
            joint_velocity=dict(zip(self.joint_names, next_velocity.tolist(), strict=True)),
            pelvis_position=tuple(pelvis_position.tolist()),
            pelvis_velocity=tuple(pelvis_velocity.tolist()),
            com_position=tuple(com_position.tolist()),
            foot_contact=contact,
            ball_position=tuple(ball_position.tolist()),
            ball_velocity=tuple(ball_velocity.tolist()),
            energy_state=energy,
            balance_margin=balance_margin,
        )

    def _features(self, model_input: ForwardModelInput) -> np.ndarray:
        state = model_input.state
        return np.asarray(
            [
                *state.joint_position.values(),
                *state.joint_velocity.values(),
                *model_input.action.joint_acceleration.values(),
                *state.pelvis_position,
                *state.pelvis_velocity,
                *state.com_position,
                *state.foot_contact,
                *state.ball_position,
                *state.ball_velocity,
                state.energy_state,
                state.balance_margin,
                *model_input.action.pelvis_acceleration,
                *model_input.action.ball_impulse,
                model_input.dt_seconds,
                model_input.phase_progress,
                *model_input.contact_mode,
            ],
            dtype=np.float64,
        )

    def _residual(self, model_input: ForwardModelInput) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(self._hidden_weight @ self._features(model_input) + self._hidden_bias)
        raw = self._output_weight @ hidden + self._output_bias
        return self.residual_limit * np.tanh(raw), hidden

    def _encode_state(self, state: ForwardState) -> np.ndarray:
        return np.asarray(
            [
                *state.joint_position.values(),
                *state.joint_velocity.values(),
                *state.pelvis_position,
                *state.pelvis_velocity,
                *state.com_position,
                *state.foot_contact,
                *state.ball_position,
                *state.ball_velocity,
                state.energy_state,
                state.balance_margin,
            ],
            dtype=np.float64,
        )

    def _decode(self, encoded: np.ndarray, residual: np.ndarray) -> tuple[ForwardState, float]:
        value = encoded + residual
        joint_count = len(self.joint_names)
        cursor = 0

        def take(count: int) -> np.ndarray:
            nonlocal cursor
            result = value[cursor : cursor + count]
            cursor += count
            return result

        position = take(joint_count)
        velocity = take(joint_count)
        pelvis_position = take(3)
        pelvis_velocity = take(3)
        com_position = take(3)
        contact = np.clip(take(2), 0.0, 1.0)
        ball_position = take(3)
        ball_velocity = take(3)
        energy = float(np.clip(take(1)[0], 0.0, 1.0))
        balance_margin = float(take(1)[0])
        risk = float(np.clip(1.0 / (1.0 + math.exp(25.0 * balance_margin)), 0.0, 1.0))
        return (
            ForwardState(
                joint_position=dict(zip(self.joint_names, position.tolist(), strict=True)),
                joint_velocity=dict(zip(self.joint_names, velocity.tolist(), strict=True)),
                pelvis_position=tuple(pelvis_position.tolist()),
                pelvis_velocity=tuple(pelvis_velocity.tolist()),
                com_position=tuple(com_position.tolist()),
                foot_contact=tuple(contact.tolist()),
                ball_position=tuple(ball_position.tolist()),
                ball_velocity=tuple(ball_velocity.tolist()),
                energy_state=energy,
                balance_margin=balance_margin,
            ),
            risk,
        )


__all__ = [
    "ForwardAction",
    "ForwardLearningReceipt",
    "ForwardModelInput",
    "ForwardPrediction",
    "ForwardState",
    "HybridForwardSelfModel",
]
