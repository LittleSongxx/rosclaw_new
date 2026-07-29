from __future__ import annotations

from copy import deepcopy
import json

import pytest

from rosclaw.continual.contracts import ExperiencePartition
from rosclaw.continual.experience import ContinualExperienceStore, ExperienceRecord
from rosclaw.continual.serde import experience_batch_from_dict, experience_batch_to_dict
from tests.continual.helpers import digest, policy, trajectory


def _batch():
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
            boundary_reason="fall",
        )
    )
    store.append(
        ExperienceRecord(
            trajectory(v2, episode="self"),
            ExperiencePartition.SELF,
            self_change_hash=digest("body-change"),
        )
    )
    return store.sample(batch_size=20, learner_version=2, seed=11)


def test_experience_batch_codec_is_lossless_and_deduplicates_trajectories() -> None:
    batch = _batch()
    envelope = experience_batch_to_dict(batch)

    restored = experience_batch_from_dict(json.loads(json.dumps(envelope, sort_keys=True)))

    assert restored.batch_hash == batch.batch_hash
    assert len(envelope["records"]) == 20
    assert len(envelope["trajectories"]) == 4
    assert restored.records[0].trajectory.trajectory_hash == (
        batch.records[0].trajectory.trajectory_hash
    )


def test_experience_batch_codec_rejects_tampered_transition() -> None:
    envelope = deepcopy(experience_batch_to_dict(_batch()))
    trajectory = next(iter(envelope["trajectories"].values()))
    trajectory["segments"][0]["observation"]["roll"] += 0.01

    with pytest.raises(ValueError, match="trajectory hash mismatch"):
        experience_batch_from_dict(envelope)
