from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rosclaw.simforge.g1_hat_trick_video import (
    _load_trajectory,
    _sample_trajectory,
    _slerp_wxyz,
)


def test_hat_trick_video_interpolates_slow_motion_between_trace_frames() -> None:
    trajectory = {
        "time": np.asarray([0.0, 1.0]),
        "pelvis_pose": np.asarray(
            [[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]]
        ),
        "joint_position": np.asarray([np.zeros(29), np.ones(29)]),
        "ball_pose": np.asarray(
            [[1.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]
        ),
    }

    index, pelvis, joints, ball = _sample_trajectory(trajectory, 0.5)

    assert index == 1
    np.testing.assert_allclose(pelvis[:3], [1.0, 0.0, 0.8])
    np.testing.assert_allclose(joints, 0.5)
    np.testing.assert_allclose(ball[:3], [2.0, 0.0, 0.1])
    np.testing.assert_allclose(pelvis[3:], np.sqrt(0.5) * np.asarray([1.0, 0.0, 0.0, 1.0]))
    assert np.linalg.norm(pelvis[3:]) == pytest.approx(1.0)


def test_hat_trick_video_quaternion_interpolation_uses_shortest_arc() -> None:
    interpolated = _slerp_wxyz(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([-1.0, 0.0, 0.0, 0.0]),
        0.5,
    )

    np.testing.assert_allclose(interpolated, [1.0, 0.0, 0.0, 0.0])


def test_hat_trick_video_rejects_non_monotonic_evidence_time(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.npz"
    pose = np.asarray([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]])
    np.savez_compressed(
        path,
        time=np.asarray([0.0, 0.0]),
        pelvis_pose=pose,
        joint_position=np.asarray([np.zeros(29), np.zeros(29)]),
        ball_pose=pose,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        _load_trajectory(path)
