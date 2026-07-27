"""Optional PyTorch Constrained Residual SAC learner.

The learner consumes immutable versioned experience and emits a candidate
artifact.  It never mutates the Feedback Plane, active policy slot, Registry,
or hardware transport.  Import this module only in an environment installed
with ``rosclaw[rl]``.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]

from rosclaw.continual.contracts import ExperiencePartition, PolicyVersion
from rosclaw.continual.experience import ExperienceBatch, ExperienceRecord
from rosclaw.feedback.contracts import canonical_hash


@dataclass(frozen=True)
class ResidualSACConfig:
    observation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    action_limits: tuple[float, ...]
    hidden_dims: tuple[int, int] = (128, 128)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    anchor_weight: float = 1.0
    churn_weight: float = 0.25
    fall_cost_limit: float = 0.0
    constraint_cost_limit: float = 0.0
    lagrange_lr: float = 1e-3
    max_lagrange: float = 100.0
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    reward_weights: tuple[float, ...] = (1.0, 0.2, 0.3, 0.2, 0.1, 0.1)
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.observation_names or len(set(self.observation_names)) != len(
            self.observation_names
        ):
            raise ValueError("SAC observation names must be non-empty and unique")
        if not self.action_names or len(set(self.action_names)) != len(self.action_names):
            raise ValueError("SAC action names must be non-empty and unique")
        if len(self.action_limits) != len(self.action_names) or any(
            not math.isfinite(value) or value <= 0.0 for value in self.action_limits
        ):
            raise ValueError("SAC action limits must match action names and be positive")
        if len(self.hidden_dims) != 2 or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("SAC reference actor requires two positive hidden dimensions")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("SAC gamma and tau must be in (0, 1]")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr, self.lagrange_lr) <= 0.0:
            raise ValueError("SAC learning rates must be positive")
        if self.batch_size <= 0:
            raise ValueError("SAC batch size must be positive")
        if (
            min(
                self.anchor_weight,
                self.churn_weight,
                self.fall_cost_limit,
                self.constraint_cost_limit,
                self.max_lagrange,
            )
            < 0.0
        ):
            raise ValueError("SAC regularization and cost limits must be non-negative")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("SAC log standard-deviation bounds are invalid")
        if len(self.reward_weights) != 6 or any(
            not math.isfinite(value) for value in self.reward_weights
        ):
            raise ValueError("SAC reward weights must contain six finite values")


@dataclass(frozen=True)
class ResidualSACUpdate:
    update_index: int
    critic_loss: float
    fall_critic_loss: float
    constraint_critic_loss: float
    actor_loss: float
    alpha: float
    fall_lagrange: float
    constraint_lagrange: float
    anchor_loss: float
    churn_loss: float
    step_churn: float
    actor_transition_count: int
    critic_transition_count: int
    stale_actor_transition_count: int
    finite: bool
    schema_version: str = "rosclaw.continual.residual_sac_update.v1"


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        self.hidden0 = nn.Linear(input_dim, hidden[0])
        self.hidden1 = nn.Linear(hidden[0], hidden[1])
        self.output = nn.Linear(hidden[1], output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.hidden0(value))
        value = torch.relu(self.hidden1(value))
        return self.output(value)

    def activations(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = torch.relu(self.hidden0(value))
        second = torch.relu(self.hidden1(first))
        return first, second


class _GaussianActor(nn.Module):
    def __init__(self, config: ResidualSACConfig) -> None:
        super().__init__()
        observation_dim = len(config.observation_names)
        action_dim = len(config.action_names)
        self.backbone = _MLP(observation_dim, 2 * action_dim, config.hidden_dims)
        self.register_buffer(
            "action_limits",
            torch.as_tensor(config.action_limits, dtype=torch.float32),
        )
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.backbone(observation).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        latent = distribution.rsample()
        unit_action = torch.tanh(latent)
        action = unit_action * self.action_limits
        jacobian = self.action_limits * (1.0 - unit_action.square())
        log_probability = (distribution.log_prob(latent) - torch.log(jacobian.clamp_min(1e-6))).sum(
            dim=-1, keepdim=True
        )
        return action, log_probability

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.backbone(observation).chunk(2, dim=-1)
        return torch.tanh(mean) * self.action_limits


class _TwinCritic(nn.Module):
    def __init__(self, config: ResidualSACConfig) -> None:
        super().__init__()
        input_dim = len(config.observation_names) + len(config.action_names)
        self.q1 = _MLP(input_dim, 1, config.hidden_dims)
        self.q2 = _MLP(input_dim, 1, config.hidden_dims)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat((observation, action), dim=-1)
        return self.q1(value), self.q2(value)


class ConstrainedResidualSAC:
    """Low-dimensional off-policy learner with two independent cost critics."""

    def __init__(self, config: ResidualSACConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA residual SAC requested but torch reports no CUDA device")
        self.device = torch.device(config.device)
        self.actor = _GaussianActor(config).to(self.device)
        # This frozen reference represents the parent policy for out-of-batch
        # output stability.  Comparing the actor with itself immediately
        # before an optimizer step would produce a zero-gradient penalty.
        self.churn_reference = copy.deepcopy(self.actor).eval()
        for parameter in self.churn_reference.parameters():
            parameter.requires_grad_(False)
        self.reward_critic = _TwinCritic(config).to(self.device)
        self.fall_critic = _TwinCritic(config).to(self.device)
        self.constraint_critic = _TwinCritic(config).to(self.device)
        self.reward_target = copy.deepcopy(self.reward_critic).eval()
        self.fall_target = copy.deepcopy(self.fall_critic).eval()
        self.constraint_target = copy.deepcopy(self.constraint_critic).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        critic_parameters = (
            list(self.reward_critic.parameters())
            + list(self.fall_critic.parameters())
            + list(self.constraint_critic.parameters())
        )
        self.critic_optimizer = torch.optim.Adam(critic_parameters, lr=config.critic_lr)
        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=config.alpha_lr)
        self.target_entropy = -float(len(config.action_names))
        self.fall_lagrange = 0.0
        self.constraint_lagrange = 0.0
        self.update_index = 0

    def update(self, batch: ExperienceBatch) -> ResidualSACUpdate:
        critic_rows = _rows(batch.records, self.config)
        actor_rows = _rows(batch.actor_records, self.config)
        if len(critic_rows[0]) < self.config.batch_size:
            raise ValueError("critic replay batch has fewer transitions than SAC batch_size")
        if len(actor_rows[0]) < self.config.batch_size:
            raise ValueError("fresh actor replay batch has fewer transitions than SAC batch_size")
        critic_tensors = self._sample_rows(critic_rows, self.config.batch_size, self.update_index)
        actor_tensors = self._sample_rows(actor_rows, self.config.batch_size, self.update_index + 1)
        (
            observations,
            actions,
            next_observations,
            rewards,
            fall_costs,
            constraint_costs,
            terminals,
        ) = critic_tensors
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_actions, next_log_probability = self.actor.sample(next_observations)
            reward_target = rewards + self.config.gamma * (1.0 - terminals) * (
                torch.minimum(*self.reward_target(next_observations, next_actions))
                - alpha * next_log_probability
            )
            fall_target = fall_costs + self.config.gamma * (1.0 - terminals) * torch.maximum(
                *self.fall_target(next_observations, next_actions)
            )
            constraint_target = constraint_costs + self.config.gamma * (
                1.0 - terminals
            ) * torch.maximum(*self.constraint_target(next_observations, next_actions))
        reward_q = self.reward_critic(observations, actions)
        fall_q = self.fall_critic(observations, actions)
        constraint_q = self.constraint_critic(observations, actions)
        reward_loss = sum(torch.nn.functional.mse_loss(value, reward_target) for value in reward_q)
        fall_loss = sum(torch.nn.functional.mse_loss(value, fall_target) for value in fall_q)
        constraint_loss = sum(
            torch.nn.functional.mse_loss(value, constraint_target) for value in constraint_q
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        (reward_loss + fall_loss + constraint_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.reward_critic.parameters())
            + list(self.fall_critic.parameters())
            + list(self.constraint_critic.parameters()),
            max_norm=10.0,
        )
        self.critic_optimizer.step()

        actor_observations = actor_tensors[0]
        reference_observations, reference_actions = self._reference_rows(batch)
        with torch.no_grad():
            pre_update_reference = self.actor.deterministic(reference_observations)
            parent_reference = self.churn_reference.deterministic(reference_observations)
        sampled_actions, log_probability = self.actor.sample(actor_observations)
        reward_actor_q = torch.minimum(*self.reward_critic(actor_observations, sampled_actions))
        fall_actor_q = torch.maximum(*self.fall_critic(actor_observations, sampled_actions))
        constraint_actor_q = torch.maximum(
            *self.constraint_critic(actor_observations, sampled_actions)
        )
        anchor_prediction = self.actor.deterministic(reference_observations)
        anchor_loss = torch.nn.functional.mse_loss(anchor_prediction, reference_actions)
        churn_loss = torch.nn.functional.mse_loss(anchor_prediction, parent_reference)
        actor_loss = (
            (
                self.log_alpha.exp().detach() * log_probability
                - reward_actor_q
                + self.fall_lagrange * fall_actor_q
                + self.constraint_lagrange * constraint_actor_q
            ).mean()
            + self.config.anchor_weight * anchor_loss
            + self.config.churn_weight * churn_loss
        )
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probability.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.fall_lagrange = _lagrange_update(
            self.fall_lagrange,
            float(fall_costs.mean().item()) - self.config.fall_cost_limit,
            self.config,
        )
        self.constraint_lagrange = _lagrange_update(
            self.constraint_lagrange,
            float(constraint_costs.mean().item()) - self.config.constraint_cost_limit,
            self.config,
        )
        _soft_update(self.reward_target, self.reward_critic, self.config.tau)
        _soft_update(self.fall_target, self.fall_critic, self.config.tau)
        _soft_update(self.constraint_target, self.constraint_critic, self.config.tau)
        with torch.no_grad():
            post_update_reference = self.actor.deterministic(reference_observations)
            measured_churn = torch.nn.functional.mse_loss(
                post_update_reference, pre_update_reference
            )
            cumulative_churn = torch.nn.functional.mse_loss(post_update_reference, parent_reference)
        values = (
            reward_loss,
            fall_loss,
            constraint_loss,
            actor_loss,
            self.log_alpha.exp(),
            anchor_loss,
            measured_churn,
        )
        finite = all(bool(torch.isfinite(value).all().item()) for value in values)
        update = ResidualSACUpdate(
            update_index=self.update_index,
            critic_loss=float(reward_loss.detach().item()),
            fall_critic_loss=float(fall_loss.detach().item()),
            constraint_critic_loss=float(constraint_loss.detach().item()),
            actor_loss=float(actor_loss.detach().item()),
            alpha=float(self.log_alpha.exp().detach().item()),
            fall_lagrange=self.fall_lagrange,
            constraint_lagrange=self.constraint_lagrange,
            anchor_loss=float(anchor_loss.detach().item()),
            churn_loss=float(cumulative_churn.detach().item()),
            step_churn=float(measured_churn.detach().item()),
            actor_transition_count=len(actor_rows[0]),
            critic_transition_count=len(critic_rows[0]),
            stale_actor_transition_count=len(critic_rows[0]) - len(actor_rows[0]),
            finite=finite,
        )
        self.update_index += 1
        return update

    def action(self, observation: Mapping[str, float]) -> Mapping[str, float]:
        if tuple(observation) != self.config.observation_names:
            raise ValueError("inference observation order does not match SAC config")
        tensor = torch.as_tensor(
            [[observation[name] for name in self.config.observation_names]],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            value = self.actor.deterministic(tensor)[0].cpu().numpy()
        return dict(zip(self.config.action_names, map(float, value), strict=True))

    def hidden_activations(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(observations, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(self.config.observation_names):
            raise ValueError("activation observations have the wrong shape")
        tensor = torch.as_tensor(value, device=self.device)
        with torch.no_grad():
            first, second = self.actor.backbone.activations(tensor)
        return first.cpu().numpy(), second.cpu().numpy()

    def artifact_bytes(self) -> bytes:
        """Return a deterministic tensor serialization for content addressing."""

        metadata = {
            "schema_version": "rosclaw.continual.residual_sac_artifact.v1",
            "observation_names": list(self.config.observation_names),
            "action_names": list(self.config.action_names),
            "action_limits": list(self.config.action_limits),
            "hidden_dims": list(self.config.hidden_dims),
            "update_index": self.update_index,
        }
        chunks = [json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()]
        for name, tensor in sorted(self.actor.state_dict().items()):
            array = tensor.detach().cpu().numpy()
            name_bytes = name.encode()
            dtype_bytes = str(array.dtype).encode()
            chunks.extend(
                (
                    struct.pack("!I", len(name_bytes)),
                    name_bytes,
                    struct.pack("!I", len(dtype_bytes)),
                    dtype_bytes,
                    struct.pack("!I", array.ndim),
                    struct.pack("!" + "Q" * array.ndim, *array.shape),
                    np.ascontiguousarray(array).tobytes(),
                )
            )
        return b"".join(chunks)

    def checkpoint_bytes(self) -> bytes:
        """Persist complete learner/optimizer state for crash-safe service recovery."""

        payload = {
            "schema_version": "rosclaw.continual.residual_sac_checkpoint.v1",
            "config_hash": canonical_hash(asdict(self.config)),
            "actor": self.actor.state_dict(),
            "churn_reference": self.churn_reference.state_dict(),
            "reward_critic": self.reward_critic.state_dict(),
            "fall_critic": self.fall_critic.state_dict(),
            "constraint_critic": self.constraint_critic.state_dict(),
            "reward_target": self.reward_target.state_dict(),
            "fall_target": self.fall_target.state_dict(),
            "constraint_target": self.constraint_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "fall_lagrange": self.fall_lagrange,
            "constraint_lagrange": self.constraint_lagrange,
            "update_index": self.update_index,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        return buffer.getvalue()

    def restore_checkpoint(self, checkpoint: bytes) -> None:
        """Restore a trusted service-owned checkpoint into the same SAC config."""

        if not checkpoint:
            raise ValueError("SAC checkpoint must not be empty")
        payload = torch.load(io.BytesIO(checkpoint), map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("SAC checkpoint payload must be a mapping")
        if payload.get("schema_version") != "rosclaw.continual.residual_sac_checkpoint.v1":
            raise ValueError("unsupported SAC checkpoint schema")
        if payload.get("config_hash") != canonical_hash(asdict(self.config)):
            raise ValueError("SAC checkpoint config does not match this learner")
        for name in (
            "actor",
            "churn_reference",
            "reward_critic",
            "fall_critic",
            "constraint_critic",
            "reward_target",
            "fall_target",
            "constraint_target",
        ):
            getattr(self, name).load_state_dict(payload[name])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"])
        self.fall_lagrange = float(payload["fall_lagrange"])
        self.constraint_lagrange = float(payload["constraint_lagrange"])
        self.update_index = int(payload["update_index"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if self.device.type == "cuda":
            cuda_rng_state = payload.get("cuda_rng_state")
            if cuda_rng_state is None:
                raise ValueError("CUDA SAC checkpoint is missing device RNG state")
            torch.cuda.set_rng_state_all(cuda_rng_state)

    def candidate_policy(self, *, parent: PolicyVersion) -> tuple[PolicyVersion, bytes]:
        if parent.observation_names != self.config.observation_names:
            raise ValueError("parent observation contract does not match the SAC learner")
        if parent.residual_action_names != self.config.action_names:
            raise ValueError("parent action contract does not match the SAC learner")
        artifact = self.artifact_bytes()
        candidate = PolicyVersion(
            version=parent.version + 1,
            artifact_hash="sha256:" + hashlib.sha256(artifact).hexdigest(),
            parent_version_hash=parent.version_hash,
            controller_snapshot_hash=parent.controller_snapshot_hash,
            body_hash=parent.body_hash,
            safety_kernel_hash=parent.safety_kernel_hash,
            observation_names=parent.observation_names,
            residual_action_names=parent.residual_action_names,
        )
        return candidate, artifact

    def _sample_rows(
        self,
        rows: tuple[np.ndarray, ...],
        count: int,
        seed_offset: int,
    ) -> tuple[torch.Tensor, ...]:
        rng = np.random.default_rng(self.config.seed + seed_offset)
        indices = rng.choice(len(rows[0]), size=count, replace=len(rows[0]) < count)
        return tuple(
            torch.as_tensor(value[indices], dtype=torch.float32, device=self.device)
            for value in rows
        )

    def _reference_rows(self, batch: ExperienceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        reference = tuple(
            record
            for record in batch.records
            if record.partition in {ExperiencePartition.ANCHOR, ExperiencePartition.SELF}
        )
        if not reference:
            raise ValueError("SAC update requires Anchor or Self reference transitions")
        rows = _rows(reference, self.config)
        observations = torch.as_tensor(rows[0], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rows[1], dtype=torch.float32, device=self.device)
        return observations, actions


def _rows(
    records: tuple[ExperienceRecord, ...],
    config: ResidualSACConfig,
) -> tuple[np.ndarray, ...]:
    observations: list[list[float]] = []
    actions: list[list[float]] = []
    next_observations: list[list[float]] = []
    rewards: list[list[float]] = []
    fall_costs: list[list[float]] = []
    constraint_costs: list[list[float]] = []
    terminals: list[list[float]] = []
    for record in records:
        for segment in record.trajectory.segments:
            if segment.policy.observation_names != config.observation_names:
                raise ValueError("experience observation contract does not match SAC config")
            if segment.policy.residual_action_names != config.action_names:
                raise ValueError("experience action contract does not match SAC config")
            observations.append([segment.observation[name] for name in config.observation_names])
            actions.append([segment.residual_action[name] for name in config.action_names])
            next_observations.append(
                [segment.next_observation[name] for name in config.observation_names]
            )
            reward_vector = tuple(segment.reward.to_dict().values())
            rewards.append(
                [
                    sum(
                        weight * value
                        for weight, value in zip(config.reward_weights, reward_vector, strict=True)
                    )
                ]
            )
            fall_costs.append([segment.cost.fall])
            constraint_costs.append(
                [
                    segment.cost.joint_limit
                    + segment.cost.torque
                    + segment.cost.slip
                    + segment.cost.stale
                    + segment.cost.collision
                    + segment.cost.feedback_saturation
                ]
            )
            terminals.append([float(segment.terminal)])
    if not observations:
        raise ValueError("SAC experience rows must not be empty")
    return tuple(
        np.asarray(value, dtype=np.float32)
        for value in (
            observations,
            actions,
            next_observations,
            rewards,
            fall_costs,
            constraint_costs,
            terminals,
        )
    )


def _lagrange_update(value: float, violation: float, config: ResidualSACConfig) -> float:
    return min(config.max_lagrange, max(0.0, value + config.lagrange_lr * violation))


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_value, source_value in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_value.mul_(1.0 - tau).add_(source_value, alpha=tau)


__all__ = ["ConstrainedResidualSAC", "ResidualSACConfig", "ResidualSACUpdate"]
