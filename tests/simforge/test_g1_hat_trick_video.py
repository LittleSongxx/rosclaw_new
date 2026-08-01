from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rosclaw.simforge.g1_hat_trick_video import (
    _escape_filtergraph_option,
    _ffmpeg_command,
    _load_trajectory,
    _sample_trajectory,
    _slerp_wxyz,
    _Source,
    _write_metric_text_files,
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


def test_filtergraph_option_escaping_matches_ffmpeg_two_level_rules() -> None:
    assert (
        _escape_filtergraph_option("this is a 'string': may contain one, or more")
        == r"this is a \\\'string\\\'\\: may contain one\, or more"
    )


def test_hat_trick_video_uses_textfile_for_untrusted_titles(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is unavailable")
    dangerous_title = "A:B,C;D[E]'F\\G%{n}\nSECOND LINE"
    source = _Source(
        name="escaping",
        title=dangerous_title,
        result={"ball_speed_mps": 6.2, "target_error_m": 0.1},
        scenario={"target_z_m": 0.6},
        trajectory_hash="sha256:" + "1" * 64,
        trajectory={},
        comparison=None,
        recovery_metrics={},
        recovery_comparison={},
        momentum_comparison={},
        naturalness_comparison={},
        comparison_kind=None,
    )
    metric_files = _write_metric_text_files(tmp_path / "drawtext", (source,))
    output = tmp_path / "escaped-title.mp4"
    command = _ffmpeg_command(
        ffmpeg=ffmpeg,
        output=output,
        fps=10,
        sources=(source,),
        durations=(0.1,),
        metric_files=metric_files,
    )
    filtergraph = command[command.index("-vf") + 1]

    assert dangerous_title not in filtergraph
    assert "textfile=" in filtergraph
    assert "expansion=none" in filtergraph
    assert dangerous_title in metric_files[0].read_text(encoding="utf-8")

    completed = subprocess.run(
        command,
        input=np.zeros((720, 1280, 3), dtype=np.uint8).tobytes(),
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert output.is_file() and output.stat().st_size > 0
