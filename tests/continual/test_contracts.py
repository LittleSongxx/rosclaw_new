from __future__ import annotations

import pytest

from rosclaw.continual.contracts import ExperienceUse, SkillPhase, VersionedTrajectory
from tests.continual.helpers import policy, segment, trajectory


def test_high_dynamic_episode_is_bound_to_one_policy_version() -> None:
    v0, _ = policy(0)
    v1, _ = policy(1, parent=v0)
    segments = (
        segment(v0, episode="mixed", start=0, end=1, phase=SkillPhase.PREPARE),
        segment(
            v1,
            episode="mixed",
            start=1,
            end=2,
            phase=SkillPhase.SWING,
            terminal=True,
        ),
    )

    with pytest.raises(ValueError, match="cannot cross policy versions"):
        VersionedTrajectory(segments, strict_replay=True)


def test_policy_lag_downgrades_actor_data_without_discarding_self_data() -> None:
    v0, _ = policy(0)
    value = trajectory(v0)

    assert value.permitted_use(learner_version=1) is ExperienceUse.ACTOR_CRITIC_SELF
    assert value.permitted_use(learner_version=2) is ExperienceUse.CRITIC_SELF_ONLY
    assert value.permitted_use(learner_version=-1) is ExperienceUse.REJECT


def test_observation_order_is_part_of_policy_contract() -> None:
    v0, _ = policy(0)
    with pytest.raises(ValueError, match="observation order"):
        type(segment(v0, episode="order", start=0, end=1, phase=SkillPhase.STAND))(
            segment_id="bad-order",
            episode_id="order",
            task_id="stand",
            phase=SkillPhase.STAND,
            start_step=0,
            end_step=1,
            policy=v0,
            controller_snapshot_hash=v0.controller_snapshot_hash,
            body_hash=v0.body_hash,
            regime_hash="sha256:" + "1" * 64,
            self_state_hash="sha256:" + "2" * 64,
            observation={"pitch": 0.0, "roll": 0.0},
            residual_action={"pelvis_roll": 0.0},
            next_observation={"roll": 0.0, "pitch": 0.0},
            behavior_logprob=0.0,
            reward=segment(v0, episode="source", start=0, end=1, phase=SkillPhase.STAND).reward,
            cost=segment(v0, episode="source", start=0, end=1, phase=SkillPhase.STAND).cost,
            terminal=True,
        )
