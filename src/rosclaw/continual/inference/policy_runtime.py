"""NumPy inference runtime binding a residual candidate to the Feedback Plane."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict

import numpy as np

from rosclaw.continual.inference.loader import ResidualCandidateArtifact
from rosclaw.continual.inference.receipt import CandidateInferenceReceipt
from rosclaw.continual.inference.version_lock import CandidateVersionLock
from rosclaw.feedback.contact_timing import ContactTimingBelief, ContactTimingEstimator
from rosclaw.feedback.contracts import (
    FallbackMode,
    FeedbackFrame,
    FeedbackLoopSpec,
    canonical_hash,
)
from rosclaw.feedback.runtime import FeedbackRuntime

_G1_ACTION_OUTPUTS = {
    "waist_roll_residual": "joint:waist_roll_joint",
    "right_hip_roll_residual": "joint:right_hip_roll_joint",
    "right_hip_yaw_residual": "joint:right_hip_yaw_joint",
    "kick_phase_rate": "skill:kick_phase_rate",
}


class ResidualCandidatePolicy:
    """Deterministic, allocation-light inference over immutable NumPy tensors."""

    def __init__(self, artifact: ResidualCandidateArtifact) -> None:
        self.artifact = artifact
        self.version_lock = CandidateVersionLock.pin(artifact.policy)
        self._observations: list[dict[str, float]] = []
        self._actions: list[dict[str, float]] = []
        self.contact_timing = ContactTimingEstimator()
        self._timing_beliefs: list[ContactTimingBelief] = []

    def reset(self) -> None:
        self.version_lock.verify(self.artifact.policy)
        self._observations.clear()
        self._actions.clear()
        self.contact_timing.reset()
        self._timing_beliefs.clear()

    def infer(
        self,
        observation: Mapping[str, float],
        *,
        enforce_contact_timing: bool = False,
        timestamp_ns: int = 0,
        policy_phase: float | None = None,
    ) -> Mapping[str, float]:
        self.version_lock.verify(self.artifact.policy)
        missing = set(self.artifact.observation_names).difference(observation)
        if missing:
            raise ValueError(f"candidate inference is missing observations: {sorted(missing)}")
        ordered = np.asarray(
            [float(observation[name]) for name in self.artifact.observation_names],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(ordered)):
            raise ValueError("candidate inference observations must be finite")
        tensors = self.artifact.tensors
        hidden0 = np.maximum(
            tensors["backbone.hidden0.weight"] @ ordered + tensors["backbone.hidden0.bias"],
            0.0,
        )
        hidden1 = np.maximum(
            tensors["backbone.hidden1.weight"] @ hidden0 + tensors["backbone.hidden1.bias"],
            0.0,
        )
        actor_output = tensors["backbone.output.weight"] @ hidden1 + tensors["backbone.output.bias"]
        mean = actor_output[: len(self.artifact.action_names)]
        values = np.tanh(mean) * tensors["action_limits"]
        if not np.all(np.isfinite(values)):
            raise ValueError("candidate inference produced a non-finite action")
        result = dict(zip(self.artifact.action_names, map(float, values), strict=True))
        if enforce_contact_timing:
            belief = self.contact_timing.update(
                timestamp_ns=timestamp_ns,
                policy_phase=(
                    float(observation["contact_phase"]) if policy_phase is None else policy_phase
                ),
                actual=observation,
            )
            self._timing_beliefs.append(belief)
            if "kick_phase_rate" in result and not belief.phase_residual_enabled:
                result["kick_phase_rate"] = 0.0
        self._observations.append(
            dict(zip(self.artifact.observation_names, map(float, ordered), strict=True))
        )
        self._actions.append(result)
        return result

    def build_receipt(self) -> CandidateInferenceReceipt:
        if not self._actions:
            raise ValueError("cannot build a candidate inference receipt without samples")
        bounded = all(
            abs(action[name]) <= limit + 1e-8
            for action in self._actions
            for name, limit in zip(
                self.artifact.action_names,
                self.artifact.action_limits,
                strict=True,
            )
        )
        normalized = [
            action[name] / limit
            for action in self._actions
            for name, limit in zip(
                self.artifact.action_names,
                self.artifact.action_limits,
                strict=True,
            )
        ]
        mean_timing_confidence = (
            sum(item.confidence for item in self._timing_beliefs) / len(self._timing_beliefs)
            if self._timing_beliefs
            else 0.0
        )
        return CandidateInferenceReceipt(
            policy_version=self.artifact.policy.version,
            policy_version_hash=self.artifact.policy.version_hash,
            parent_version_hash=self.artifact.parent.version_hash,
            artifact_hash=self.artifact.artifact_hash,
            body_hash=self.artifact.policy.body_hash,
            observation_contract_hash=canonical_hash(
                {"observation_names": list(self.artifact.observation_names)}
            ),
            action_contract_hash=canonical_hash(
                {
                    "action_names": list(self.artifact.action_names),
                    "action_limits": list(self.artifact.action_limits),
                }
            ),
            trace_hash=canonical_hash(
                {"observations": self._observations, "actions": self._actions}
            ),
            inference_count=len(self._actions),
            actions_bounded=bounded,
            action_rms=float(np.sqrt(np.mean(np.square(normalized)))),
            maximum_action_limit_ratio=max(map(abs, normalized)),
            contact_timing_enabled_count=sum(
                item.phase_residual_enabled for item in self._timing_beliefs
            ),
            contact_timing_mean_confidence=mean_timing_confidence,
        )


class ResidualCandidateController:
    """Feedback controller adapter with an explicit ROSClaw action mapping."""

    def __init__(self, policy: ResidualCandidatePolicy) -> None:
        unknown = set(policy.artifact.action_names).difference(_G1_ACTION_OUTPUTS)
        if unknown:
            raise ValueError(
                f"candidate contains unsupported G1 residual actions: {sorted(unknown)}"
            )
        self.policy = policy

    @property
    def controller_hash(self) -> str:
        return canonical_hash(self.config_dict())

    def reset(self) -> None:
        self.policy.reset()

    def compute(
        self,
        frame: FeedbackFrame,
        base_action: Mapping[str, float],
    ) -> Mapping[str, float]:
        del base_action
        action = self.policy.infer(
            frame.actual,
            enforce_contact_timing=True,
            timestamp_ns=frame.timestamp_ns,
            policy_phase=frame.phase,
        )
        return {_G1_ACTION_OUTPUTS[name]: value for name, value in action.items()}

    def config_dict(self) -> dict[str, object]:
        return {
            "controller_type": "residual_candidate_numpy",
            "version": 1,
            "policy_version": self.policy.artifact.policy.version,
            "policy_version_hash": self.policy.artifact.policy.version_hash,
            "artifact_hash": self.policy.artifact.artifact_hash,
            "action_mapping": {
                name: _G1_ACTION_OUTPUTS[name] for name in self.policy.artifact.action_names
            },
            "contact_timing": asdict(self.policy.contact_timing.config),
        }


def build_g1_candidate_runtime(
    artifact: ResidualCandidateArtifact,
    *,
    rate_hz: float = 100.0,
    compute_clock_ns: Callable[[], int] | None = None,
) -> tuple[FeedbackRuntime, ResidualCandidatePolicy]:
    """Build a SIM-only runtime; it has no Registry, network, DDS, or slot dependency."""

    if not math.isfinite(rate_hz) or not 1.0 <= rate_hz <= 500.0:
        raise ValueError("candidate feedback rate must be in [1, 500] Hz")
    policy = ResidualCandidatePolicy(artifact)
    controller = ResidualCandidateController(policy)
    output_limits = {
        _G1_ACTION_OUTPUTS[name]: limit
        for name, limit in zip(artifact.action_names, artifact.action_limits, strict=True)
    }
    preferred_references = tuple(
        name
        for name in (
            "torso_roll",
            "torso_pitch",
            "com_y_relative",
            "ball_lateral_error_m",
        )
        if name in artifact.observation_names
    )
    references = preferred_references or artifact.observation_names[:1]
    spec = FeedbackLoopSpec(
        loop_id=f"g1/goalforge-candidate/v{artifact.policy.version}",
        body_hash=artifact.policy.body_hash,
        controller_hash=controller.controller_hash,
        reference_signals=references,
        observation_signals=tuple(
            dict.fromkeys(artifact.observation_names + ContactTimingEstimator.required_signals)
        ),
        output_limits=output_limits,
        rate_hz=rate_hz,
        deadline_ms=1000.0 / rate_hz,
        max_observation_age_ms=2.0 * 1000.0 / rate_hz,
        fallback_deadline_miss=FallbackMode.BASE_POLICY_ONLY,
        fallback_unsafe_projection=FallbackMode.BASE_POLICY_ONLY,
    )
    if compute_clock_ns is None:
        return FeedbackRuntime(spec=spec, controller=controller), policy
    return (
        FeedbackRuntime(
            spec=spec,
            controller=controller,
            compute_clock_ns=compute_clock_ns,
        ),
        policy,
    )


__all__ = [
    "ResidualCandidateController",
    "ResidualCandidatePolicy",
    "build_g1_candidate_runtime",
]
