from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")

from rosclaw.continual.contracts import ExperiencePartition
from rosclaw.continual.experience import ContinualExperienceStore, ExperienceRecord
from rosclaw.continual.learner import ConstrainedResidualSAC, ResidualSACConfig
from tests.continual.helpers import digest, policy, trajectory


def _batch():
    v0, _ = policy(0)
    v1, _ = policy(1, parent=v0)
    v2, _ = policy(2, parent=v1)
    store = ContinualExperienceStore()
    for index in range(8):
        store.append(
            ExperienceRecord(trajectory(v2, episode=f"recent-{index}"), ExperiencePartition.RECENT)
        )
    for index in range(4):
        store.append(
            ExperienceRecord(
                trajectory(v0, episode=f"anchor-{index}"),
                ExperiencePartition.ANCHOR,
                anchor_policy_hash=v0.artifact_hash,
            )
        )
    for index in range(3):
        store.append(
            ExperienceRecord(
                trajectory(v2, episode=f"boundary-{index}", critical=True),
                ExperiencePartition.BOUNDARY,
                boundary_reason="fall",
            )
        )
    for index in range(2):
        store.append(
            ExperienceRecord(
                trajectory(v2, episode=f"self-{index}"),
                ExperiencePartition.SELF,
                self_change_hash=digest(f"body-change-{index}"),
            )
        )
    return v2, store.sample(batch_size=20, learner_version=2, seed=5)


def test_constrained_residual_sac_updates_without_using_stale_anchor_for_actor() -> None:
    parent, batch = _batch()
    learner = ConstrainedResidualSAC(
        ResidualSACConfig(
            observation_names=parent.observation_names,
            action_names=parent.residual_action_names,
            action_limits=(0.04,),
            hidden_dims=(16, 16),
            batch_size=16,
            seed=3,
        )
    )

    first_update = learner.update(batch)
    update = learner.update(batch)
    candidate, artifact = learner.candidate_policy(parent=parent)
    action = learner.action({"roll": 0.2, "pitch": -0.1})

    assert update.finite
    assert first_update.step_churn > 0.0
    assert update.churn_loss > 0.0
    assert update.stale_actor_transition_count > 0
    assert all(math.isfinite(value) for value in (update.actor_loss, update.critic_loss))
    assert abs(action["pelvis_roll"]) <= 0.04
    assert candidate.version == parent.version + 1
    assert candidate.parent_version_hash == parent.version_hash
    assert artifact


def test_actor_exposes_hidden_activations_for_post_hoc_selfcore_analysis() -> None:
    parent, _ = _batch()
    learner = ConstrainedResidualSAC(
        ResidualSACConfig(
            observation_names=parent.observation_names,
            action_names=parent.residual_action_names,
            action_limits=(0.04,),
            hidden_dims=(8, 6),
        )
    )

    first, second = learner.hidden_activations(
        __import__("numpy").asarray([[0.0, 0.0], [0.1, -0.1]], dtype=float)
    )

    assert first.shape == (2, 8)
    assert second.shape == (2, 6)
