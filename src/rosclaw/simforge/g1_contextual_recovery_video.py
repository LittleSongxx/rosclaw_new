"""Evidence-bound DEVELOPMENT video for contextual G1 recovery learning.

The renderer is deliberately downstream of MuJoCo rollouts.  It never creates
promotion evidence and it refuses to label a validation-rejected candidate as
qualified.  The split-screen clip is useful for inspecting visible motion
while preserving the rejection state in both the frame labels and manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np

from rosclaw.simforge.backends.unitree_mujoco_backend import trajectory_digest
from rosclaw.simforge.g1_contextual_recovery import (
    load_g1_contextual_recovery_artifact,
)
from rosclaw.simforge.g1_contextual_recovery_training import (
    G1ContextualRecoveryTrainer,
    _composite,
    _config_from_primitive,
    _contextual_route,
    _terminal_damping_reductions,
)
from rosclaw.simforge.g1_hat_trick_video import (
    _escape_filtergraph_option,
    _render_pose,
)
from rosclaw.simforge.g1_muscle_memory_training import _case_score
from rosclaw.simforge.g1_recovery_quality import measure_g1_recovery_quality
from rosclaw.simforge.g1_temporal_muscle_memory_training import _moving_reductions

_SCENE_REL = Path("g1_description/scene_with_ball.xml")


@dataclass(frozen=True)
class G1ContextualRecoveryVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    artifact_hash: str
    training_report_hash: str
    case_name: str
    fixed_trajectory_hash: str
    learned_trajectory_hash: str
    strict_replay: bool
    backward_reduction: float
    tail_wobble_reduction: float
    leg_jerk_reduction: float
    composite_reduction: float
    width: int
    height: int
    fps: int
    frame_count: int
    duration_sec: float
    promotion_status: str = "REJECTED"
    validation_qualified: bool = False
    development_only: bool = True
    visualization_only: bool = True
    generates_task_evidence: bool = False
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.contextual_recovery_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class G1ContextualRecoverySuiteClip:
    case_name: str
    challenge_label: str
    selected_primitive_index: int | None
    exact_fallback_replay: bool
    strict_replay: bool
    baseline_trajectory_hash: str
    candidate_trajectory_hash: str
    backward_reduction: float
    tail_wobble_reduction: float
    tail_joint_jerk_reduction: float
    baseline_backward_reversal_m: float
    candidate_backward_reversal_m: float
    baseline_tail_wobble_index: float
    candidate_tail_wobble_index: float
    baseline_tail_joint_jerk_rms_rad_s3: float
    candidate_tail_joint_jerk_rms_rad_s3: float
    duration_sec: float
    frame_count: int
    safe: bool
    goal_preserved: bool
    naturalness_preserved: bool
    schema_version: str = "rosclaw.g1_goalforge.contextual_recovery_suite_clip.v1"


@dataclass(frozen=True)
class G1ContextualRecoverySuiteVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    artifact_hash: str
    training_report_hash: str
    clips: tuple[G1ContextualRecoverySuiteClip, ...]
    width: int
    height: int
    fps: int
    frame_count: int
    duration_sec: float
    strict_replay_count: int
    exact_fallback_replay_count: int
    promotion_status: str = "REJECTED"
    validation_qualified: bool = False
    development_only: bool = True
    visualization_only: bool = True
    generates_task_evidence: bool = False
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.g1_goalforge.contextual_recovery_suite_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_g1_contextual_recovery_video(
    *,
    artifact_path: Path,
    training_report_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    case_name: str = "moving_ball_nominal_velocity_070",
    fps: int = 30,
) -> G1ContextualRecoveryVideoResult:
    """Render fixed-vs-learned DEVELOPMENT motion with rejection labels."""

    artifact_file = artifact_path.expanduser().resolve()
    report_file = training_report_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("contextual recovery video must remain outside the checkout")
    if any(path == checkout or checkout in path.parents for path in (artifact_file, report_file)):
        raise ValueError("contextual recovery video evidence must remain outside the checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("contextual recovery video output must use .mp4")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("contextual recovery video or manifest already exists")
    if not 10 <= fps <= 60:
        raise ValueError("contextual recovery video fps must be in [10, 60]")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for contextual recovery video")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("contextual recovery training report must be an object")
    if report.get("qualified") is not False:
        raise ValueError("development video requires an explicitly rejected candidate")
    rejection_reasons = report.get("rejection_reasons")
    if rejection_reasons != ["sealed_validation_failed"]:
        raise ValueError("development video only permits the sealed-validation rejection state")
    artifact = load_g1_contextual_recovery_artifact(artifact_file)
    report_artifact = report.get("artifact")
    if (
        not isinstance(report_artifact, dict)
        or report_artifact.get("artifact_hash") != artifact.artifact_hash
    ):
        raise ValueError("contextual recovery artifact/report hash mismatch")
    if float(report.get("learned_component_composite_reduction", 0.0)) < 0.05:
        raise ValueError("development video requires at least 5% measured learning contribution")

    trainer = G1ContextualRecoveryTrainer(asset_root=asset_root)
    matches = [case for case in trainer.cases if case.name == case_name]
    if len(matches) != 1:
        raise ValueError("contextual recovery video case is not a unique DEVELOPMENT case")
    case = matches[0]
    parent = trainer._run_config(case, trainer.retained_config)
    fixed = trainer._run_config(case, trainer.fixed_config)
    learned = trainer._run_contextual(case, artifact)
    replay = trainer._run_contextual(case, artifact)
    strict = bool(
        learned.result.summary_dict() == replay.result.summary_dict()
        and trajectory_digest(learned.trajectory) == trajectory_digest(replay.trajectory)
    )
    parent_quality = measure_g1_recovery_quality(parent.trajectory)
    fixed_quality = measure_g1_recovery_quality(fixed.trajectory)
    learned_quality = measure_g1_recovery_quality(learned.trajectory)
    _, safe, goal, natural = _case_score(
        parent=parent,
        parent_quality=parent_quality,
        candidate=learned,
        candidate_quality=learned_quality,
    )
    if not (strict and safe and goal and natural):
        raise ValueError("contextual recovery video requires safe strict DEVELOPMENT replay")
    reductions = _moving_reductions(fixed_quality, learned_quality)
    composite = _composite(reductions)
    if composite <= 0.05:
        raise ValueError("selected video case does not show a material learned contribution")

    evidence_root = output.parent / (output.stem + "-evidence")
    evidence_root.mkdir(parents=True, exist_ok=False)
    fixed_path = evidence_root / "fixed-structured.npz"
    learned_path = evidence_root / "learned-contextual.npz"
    np.savez_compressed(fixed_path, **fixed.trajectory)  # type: ignore[arg-type]
    np.savez_compressed(learned_path, **learned.trajectory)  # type: ignore[arg-type]
    output.parent.mkdir(parents=True, exist_ok=True)

    contact = float(learned.result.ball_contact_time_sec or 5.25)
    start = max(float(learned.trajectory["time"][0]), contact - 2.5)
    end = min(float(learned.trajectory["time"][-1]), contact + 7.0)
    frame_count = max(1, int(np.ceil((end - start) * fps)))
    timeline = np.linspace(start, end, frame_count, endpoint=False)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        scene = asset_root.expanduser().resolve() / _SCENE_REL
        model = mujoco.MjModel.from_xml_path(str(scene))
        left_data = mujoco.MjData(model)
        right_data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=360, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        process = subprocess.Popen(
            _ffmpeg_command(
                ffmpeg=ffmpeg,
                output=output,
                fps=fps,
                reductions=reductions,
                composite=composite,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("contextual recovery ffmpeg pipe is unavailable")
        try:
            _write_frames(
                mujoco=mujoco,
                model=model,
                left_data=left_data,
                right_data=right_data,
                renderer=renderer,
                camera=camera,
                fixed=fixed.trajectory,
                learned=learned.trajectory,
                scenario=case.scenario.to_private_dict(),
                contact_time=contact,
                timeline=timeline,
                stream=cast(BinaryIO, process.stdin),
            )
        except BaseException:
            process.stdin.close()
            process.kill()
            process.wait()
            raise
        finally:
            renderer.close()
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        code = process.wait()
        if code:
            raise RuntimeError(f"contextual recovery ffmpeg failed ({code}): {stderr[-2000:]}")
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    manifest = output.with_suffix(".json")
    result = G1ContextualRecoveryVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_hash_file(output),
        artifact_hash=artifact.artifact_hash,
        training_report_hash=_hash_file(report_file),
        case_name=case_name,
        fixed_trajectory_hash=_hash_file(fixed_path),
        learned_trajectory_hash=_hash_file(learned_path),
        strict_replay=strict,
        backward_reduction=reductions[0],
        tail_wobble_reduction=reductions[1],
        leg_jerk_reduction=reductions[2],
        composite_reduction=composite,
        width=1280,
        height=720,
        fps=fps,
        frame_count=frame_count,
        duration_sec=frame_count / fps,
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def render_g1_contextual_recovery_suite_video(
    *,
    artifact_path: Path,
    training_report_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    case_names: tuple[str, ...] = (
        "static_high",
        "moving_ball_nominal",
        "moving_ball_lateral_005",
        "moving_ball_nominal_velocity_070",
        "moving_ball_light_400g",
    ),
    fps: int = 30,
) -> G1ContextualRecoverySuiteVideoResult:
    """Render a long multi-challenge DEVELOPMENT suite with a damping ablation."""

    artifact_file = artifact_path.expanduser().resolve()
    report_file = training_report_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("contextual recovery suite video must remain outside the checkout")
    if any(path == checkout or checkout in path.parents for path in (artifact_file, report_file)):
        raise ValueError("contextual recovery suite evidence must remain outside the checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("contextual recovery suite video output must use .mp4")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("contextual recovery suite video or manifest already exists")
    if not 10 <= fps <= 60:
        raise ValueError("contextual recovery suite video fps must be in [10, 60]")
    if not 2 <= len(case_names) <= 8 or len(set(case_names)) != len(case_names):
        raise ValueError("contextual recovery suite requires 2 to 8 unique cases")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for contextual recovery suite video")

    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("qualified") is not False:
        raise ValueError("suite video requires an explicitly rejected training report")
    if report.get("rejection_reasons") != ["sealed_validation_failed"]:
        raise ValueError("suite video only permits the sealed-validation rejection state")
    terminal_report = report.get("terminal_damping")
    if not isinstance(terminal_report, dict) or terminal_report.get("qualified") is not True:
        raise ValueError("suite video requires a qualified DEVELOPMENT damping ablation")
    artifact = load_g1_contextual_recovery_artifact(artifact_file)
    report_artifact = report.get("artifact")
    if (
        not isinstance(report_artifact, dict)
        or report_artifact.get("artifact_hash") != artifact.artifact_hash
    ):
        raise ValueError("contextual recovery suite artifact/report hash mismatch")

    trainer = G1ContextualRecoveryTrainer(asset_root=asset_root)
    available = {case.name: case for case in trainer.cases}
    missing = set(case_names).difference(available)
    if missing:
        raise ValueError(f"unknown contextual recovery suite cases: {sorted(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_root = output.parent / (output.stem + "-evidence")
    evidence_root.mkdir(parents=True, exist_ok=False)
    challenge_labels = {
        "static_high": "HIGH TARGET 0.55m + LATERAL BALL OFFSET 0.10m",
        "moving_ball_nominal": "MOVING BALL · -0.08 m/s",
        "moving_ball_lateral_005": "MOVING BALL + LATERAL OFFSET 0.005m",
        "moving_ball_nominal_velocity_070": "MOVING BALL SPEED SHIFT · -0.07 m/s",
        "moving_ball_light_400g": "MOVING BALL + LIGHTER 0.400kg BALL",
    }

    clips: list[G1ContextualRecoverySuiteClip] = []
    temporary_root = Path(tempfile.mkdtemp(prefix="rosclaw-g1-suite-"))
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco

        scene = asset_root.expanduser().resolve() / _SCENE_REL
        model = mujoco.MjModel.from_xml_path(str(scene))
        left_data = mujoco.MjData(model)
        right_data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=440, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        temporary_clips: list[Path] = []
        try:
            for clip_index, case_name in enumerate(case_names, start=1):
                case = available[case_name]
                candidate = trainer._run_contextual(case, artifact)
                replay = trainer._run_contextual(case, artifact)
                strict = bool(
                    candidate.result.summary_dict() == replay.result.summary_dict()
                    and trajectory_digest(candidate.trajectory)
                    == trajectory_digest(replay.trajectory)
                )
                route = _contextual_route(candidate)
                if route is None:
                    baseline = trainer._run_config(case, trainer.retained_config)
                    exact_fallback = bool(
                        baseline.result.summary_dict() == candidate.result.summary_dict()
                        and trajectory_digest(baseline.trajectory)
                        == trajectory_digest(candidate.trajectory)
                    )
                else:
                    damped_config = _config_from_primitive(
                        trainer.fixed_config,
                        artifact.primitives[route],
                    )
                    baseline = trainer._run_config(
                        case,
                        replace(
                            damped_config,
                            terminal_damping_start_policy_frame=None,
                            terminal_kp_scale=1.0,
                            terminal_kd_scale=1.0,
                        ),
                    )
                    exact_fallback = False
                baseline_quality = measure_g1_recovery_quality(baseline.trajectory)
                candidate_quality = measure_g1_recovery_quality(candidate.trajectory)
                _, safe, goal, natural = _case_score(
                    parent=baseline,
                    parent_quality=baseline_quality,
                    candidate=candidate,
                    candidate_quality=candidate_quality,
                )
                reductions = (
                    (0.0, 0.0, 0.0)
                    if route is None
                    else _terminal_damping_reductions(baseline_quality, candidate_quality)
                )
                if not (strict and safe and goal and natural):
                    raise ValueError(f"suite case failed replay/safety gates: {case_name}")
                if route is None and not exact_fallback:
                    raise ValueError(f"suite fallback is not exact: {case_name}")
                if route is not None and (
                    reductions[0] < 0.0 or reductions[1] < -0.05 or reductions[2] < 0.10
                ):
                    raise ValueError(f"suite damping ablation failed: {case_name}")

                baseline_path = evidence_root / f"{clip_index:02d}-{case_name}-ablation.npz"
                candidate_path = evidence_root / f"{clip_index:02d}-{case_name}-candidate.npz"
                np.savez_compressed(baseline_path, **baseline.trajectory)  # type: ignore[arg-type]
                np.savez_compressed(candidate_path, **candidate.trajectory)  # type: ignore[arg-type]
                contact = float(candidate.result.ball_contact_time_sec or 5.25)
                start = max(float(candidate.trajectory["time"][0]), contact - 2.2)
                end = float(candidate.trajectory["time"][-1])
                frame_count = max(1, int(np.ceil((end - start) * fps)))
                timeline = np.linspace(start, end, frame_count, endpoint=False)
                temporary_clip = temporary_root / f"clip-{clip_index:02d}.mp4"
                process = subprocess.Popen(
                    _suite_ffmpeg_command(
                        ffmpeg=ffmpeg,
                        output=temporary_clip,
                        fps=fps,
                        clip_index=clip_index,
                        clip_count=len(case_names),
                        challenge_label=challenge_labels.get(case_name, case_name.upper()),
                        reductions=reductions,
                        exact_fallback=exact_fallback,
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("contextual recovery suite ffmpeg pipe is unavailable")
                try:
                    _write_frames(
                        mujoco=mujoco,
                        model=model,
                        left_data=left_data,
                        right_data=right_data,
                        renderer=renderer,
                        camera=camera,
                        fixed=baseline.trajectory,
                        learned=candidate.trajectory,
                        scenario=case.scenario.to_private_dict(),
                        contact_time=contact,
                        timeline=timeline,
                        stream=cast(BinaryIO, process.stdin),
                        panel_top=140,
                    )
                except BaseException:
                    process.stdin.close()
                    process.kill()
                    process.wait()
                    raise
                process.stdin.close()
                stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
                code = process.wait()
                if code:
                    raise RuntimeError(
                        f"contextual recovery suite ffmpeg failed ({code}): {stderr[-2000:]}"
                    )
                temporary_clips.append(temporary_clip)
                clips.append(
                    G1ContextualRecoverySuiteClip(
                        case_name=case_name,
                        challenge_label=challenge_labels.get(case_name, case_name.upper()),
                        selected_primitive_index=route,
                        exact_fallback_replay=exact_fallback,
                        strict_replay=strict,
                        baseline_trajectory_hash=_hash_file(baseline_path),
                        candidate_trajectory_hash=_hash_file(candidate_path),
                        backward_reduction=reductions[0],
                        tail_wobble_reduction=reductions[1],
                        tail_joint_jerk_reduction=reductions[2],
                        baseline_backward_reversal_m=(
                            baseline_quality.post_contact_backward_reversal_m
                        ),
                        candidate_backward_reversal_m=(
                            candidate_quality.post_contact_backward_reversal_m
                        ),
                        baseline_tail_wobble_index=baseline_quality.tail_wobble_index,
                        candidate_tail_wobble_index=candidate_quality.tail_wobble_index,
                        baseline_tail_joint_jerk_rms_rad_s3=(
                            baseline_quality.tail_joint_jerk_rms_rad_s3
                        ),
                        candidate_tail_joint_jerk_rms_rad_s3=(
                            candidate_quality.tail_joint_jerk_rms_rad_s3
                        ),
                        duration_sec=frame_count / fps,
                        frame_count=frame_count,
                        safe=safe,
                        goal_preserved=goal,
                        naturalness_preserved=natural,
                    )
                )
        finally:
            renderer.close()
        concat_file = temporary_root / "clips.txt"
        concat_file.write_text(
            "".join(f"file '{path}'\n" for path in temporary_clips),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                "contextual recovery suite concatenation failed: "
                + completed.stderr.decode(errors="replace")[-2000:]
            )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    total_frames = sum(clip.frame_count for clip in clips)
    manifest = output.with_suffix(".json")
    result = G1ContextualRecoverySuiteVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_hash_file(output),
        artifact_hash=artifact.artifact_hash,
        training_report_hash=_hash_file(report_file),
        clips=tuple(clips),
        width=1280,
        height=720,
        fps=fps,
        frame_count=total_frames,
        duration_sec=total_frames / fps,
        strict_replay_count=sum(clip.strict_replay for clip in clips),
        exact_fallback_replay_count=sum(clip.exact_fallback_replay for clip in clips),
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    left_data: Any,
    right_data: Any,
    renderer: Any,
    camera: Any,
    fixed: dict[str, np.ndarray],
    learned: dict[str, np.ndarray],
    scenario: dict[str, Any],
    contact_time: float,
    timeline: np.ndarray,
    stream: BinaryIO,
    panel_top: int = 180,
) -> None:
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    for simulation_time in timeline:
        left = _render_pose(
            mujoco=mujoco,
            model=model,
            data=left_data,
            renderer=renderer,
            camera=camera,
            trajectory=fixed,
            simulation_time=float(simulation_time),
            ball_qpos=ball_qpos,
            scenario=scenario,
            contact_time=contact_time,
            show_grid=False,
            show_push=False,
        )
        right = _render_pose(
            mujoco=mujoco,
            model=model,
            data=right_data,
            renderer=renderer,
            camera=camera,
            trajectory=learned,
            simulation_time=float(simulation_time),
            ball_qpos=ball_qpos,
            scenario=scenario,
            contact_time=contact_time,
            show_grid=False,
            show_push=False,
        )
        canvas: np.ndarray = np.zeros((720, 1280, 3), dtype=np.uint8)
        panel_bottom = panel_top + int(left.shape[0])
        canvas[panel_top:panel_bottom, :640] = left
        canvas[panel_top:panel_bottom, 640:] = right
        stream.write(np.ascontiguousarray(canvas).tobytes())


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    reductions: tuple[float, float, float],
    composite: float,
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    labels = (
        ("ROSClaw G1 Contextual Recovery · DEVELOPMENT REPLAY", 34, 20, 34, "white"),
        ("FIXED STRUCTURED · ZERO LEARNING", 95, 145, 20, "0xFFB45F"),
        ("LEARNED ROUTER · DEVELOPMENT ONLY", 720, 145, 20, "0x65F59A"),
        (
            f"backstep -{100 * reductions[0]:.1f}%  ·  wobble -{100 * reductions[1]:.1f}%  "
            f"·  leg jerk -{100 * reductions[2]:.1f}%  ·  composite +{100 * composite:.1f}%",
            34,
            80,
            22,
            "0x65F59A",
        ),
        (
            "SIM VISUALIZATION ONLY · SEALED VALIDATION FAILED · NOT PROMOTED",
            34,
            674,
            21,
            "0xFF7070",
        ),
    )
    filters = [
        "drawbox=x=0:y=0:w=iw:h=120:color=0x050A12@0.82:t=fill",
        "drawbox=x=0:y=h-72:w=iw:h=72:color=0x050A12@0.82:t=fill",
    ]
    filters.extend(
        f"drawtext={font_option}text={_escape_filtergraph_option(text)}:"
        f"expansion=none:x={x}:y={y}:fontsize={size}:fontcolor={color}"
        for text, x, y, size, color in labels
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "1280x720",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _suite_ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    clip_index: int,
    clip_count: int,
    challenge_label: str,
    reductions: tuple[float, float, float],
    exact_fallback: bool,
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""
    if exact_fallback:
        left_label = "RETAINED PARENT"
        right_label = "SAFE OOD FALLBACK · EXACT REPLAY"
        metric_label = "HIGH-TARGET CHALLENGE · UNSEEN STATE ROUTED FAIL-CLOSED"
        metric_color = "0x74C7FF"
    else:
        left_label = "CONTEXTUAL PRIMITIVE · DAMPING OFF"
        right_label = "CONTEXTUAL + TERMINAL LEG DAMPING"
        metric_label = (
            f"backstep improvement {100 * reductions[0]:+.1f}%  ·  "
            f"wobble improvement {100 * reductions[1]:+.1f}%  ·  "
            f"tail jerk improvement {100 * reductions[2]:+.1f}%"
        )
        metric_color = "0x65F59A"
    labels = (
        (
            f"ROSClaw G1 Recovery Suite · EXP {clip_index}/{clip_count}",
            28,
            14,
            29,
            "white",
        ),
        (challenge_label, 28, 57, 22, "0x74C7FF"),
        (metric_label, 28, 92, 19, metric_color),
        (left_label, 72, 118, 17, "0xFFB45F"),
        (right_label, 690, 118, 17, "0x65F59A"),
        (
            "SIM DEVELOPMENT VISUALIZATION · SEALED VALIDATION FAILED · NOT PROMOTED",
            28,
            674,
            18,
            "0xFF7070",
        ),
    )
    filters = [
        "drawbox=x=0:y=0:w=iw:h=140:color=0x050A12@0.86:t=fill",
        "drawbox=x=0:y=h-72:w=iw:h=72:color=0x050A12@0.86:t=fill",
        "drawbox=x=638:y=140:w=4:h=440:color=0xD9E3F0@0.45:t=fill",
    ]
    filters.extend(
        f"drawtext={font_option}text={_escape_filtergraph_option(text)}:"
        f"expansion=none:x={x}:y={y}:fontsize={size}:fontcolor={color}"
        for text, x, y, size, color in labels
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "1280x720",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-vf",
        ",".join(filters),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "G1ContextualRecoverySuiteClip",
    "G1ContextualRecoverySuiteVideoResult",
    "G1ContextualRecoveryVideoResult",
    "render_g1_contextual_recovery_suite_video",
    "render_g1_contextual_recovery_video",
]
