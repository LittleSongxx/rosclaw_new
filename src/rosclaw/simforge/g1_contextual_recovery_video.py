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
from dataclasses import asdict, dataclass
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
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[180:540, :640] = left
        canvas[180:540, 640:] = right
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


def _hash_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "G1ContextualRecoveryVideoResult",
    "render_g1_contextual_recovery_video",
]
