from __future__ import annotations

import pytest

from rosclaw.continual.contracts import ExperiencePartition
from rosclaw.continual.experience import (
    ContinualExperienceStore,
    ExperienceRecord,
)
from tests.continual.helpers import digest, policy, trajectory


def test_four_buffer_mix_is_exact_and_stale_actor_data_is_downgraded() -> None:
    v0, _ = policy(0)
    v1, _ = policy(1, parent=v0)
    v2, _ = policy(2, parent=v1)
    store = ContinualExperienceStore()
    store.append(ExperienceRecord(trajectory(v2, episode="recent"), ExperiencePartition.RECENT))
    store.append(
        ExperienceRecord(
            trajectory(v0, episode="anchor"),
            ExperiencePartition.ANCHOR,
            anchor_policy_hash=v0.artifact_hash,
        )
    )
    store.append(
        ExperienceRecord(
            trajectory(v2, episode="boundary", critical=True),
            ExperiencePartition.BOUNDARY,
            boundary_reason="fall counterexample",
        )
    )
    store.append(
        ExperienceRecord(
            trajectory(v2, episode="self"),
            ExperiencePartition.SELF,
            self_change_hash=digest("right-hip-gain-drop"),
        )
    )

    batch = store.sample(batch_size=20, learner_version=2, seed=7)

    assert batch.requested_counts == {
        ExperiencePartition.RECENT: 10,
        ExperiencePartition.ANCHOR: 5,
        ExperiencePartition.BOUNDARY: 3,
        ExperiencePartition.SELF: 2,
    }
    assert len(batch.actor_records) == 15
    assert len(batch.critic_self_only_records) == 5
    assert len(batch.batch_hash) == 71


def test_boundary_buffer_rejects_ordinary_non_boundary_rollout() -> None:
    v0, _ = policy(0)
    with pytest.raises(ValueError, match="safety event or near miss"):
        ExperienceRecord(
            trajectory(v0),
            ExperiencePartition.BOUNDARY,
            boundary_reason="not actually close",
        )


def test_non_boundary_record_rejects_boundary_request_commitment() -> None:
    v0, _ = policy(0)
    with pytest.raises(ValueError, match="only boundary"):
        ExperienceRecord(
            trajectory(v0),
            ExperiencePartition.RECENT,
            boundary_request_hash=digest("boundary-request"),
        )


def test_replay_fails_closed_until_all_partitions_exist() -> None:
    v0, _ = policy(0)
    store = ContinualExperienceStore()
    store.append(ExperienceRecord(trajectory(v0), ExperiencePartition.RECENT))

    with pytest.raises(RuntimeError, match="empty partitions"):
        store.sample(batch_size=20, learner_version=0, seed=1)
