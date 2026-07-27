"""Cinematic, evidence-downstream video for the GoalForge Hat Trick."""

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

_SCENE_REL = Path("g1_description/scene_with_ball.xml")
_RENDER_WIDTH = 640
_RENDER_HEIGHT = 360


@dataclass(frozen=True)
class HatTrickVideoClip:
    name: str
    title: str
    source_trajectory_hash: str
    playback_start_sec: float
    playback_duration_sec: float
    frame_count: int
    success: bool
    target_error_m: float
    ball_speed_mps: float
    tail_wobble_reduction: float


@dataclass(frozen=True)
class HatTrickVideoResult:
    output_path: str
    manifest_path: str
    video_hash: str
    evidence_report_hash: str
    width: int
    height: int
    fps: int
    frame_count: int
    duration_sec: float
    clips: tuple[HatTrickVideoClip, ...]
    visualization_only: bool = True
    schema_version: str = "rosclaw.g1_goalforge.hat_trick_video.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "clips": [asdict(clip) for clip in self.clips],
            "label_source": "verified_hat_trick_evidence",
            "generates_task_evidence": False,
        }


@dataclass(frozen=True)
class _Source:
    name: str
    title: str
    result: dict[str, Any]
    scenario: dict[str, Any]
    trajectory_hash: str
    trajectory: dict[str, np.ndarray]
    comparison: dict[str, np.ndarray] | None
    recovery_metrics: dict[str, Any]
    recovery_comparison: dict[str, Any]


def render_goalforge_hat_trick_video(
    *,
    evidence_path: Path,
    asset_root: Path,
    output_path: Path,
    source_checkout: Path,
    fps: int = 30,
) -> HatTrickVideoResult:
    """Render three verified clips; labels never flow back into evaluation."""

    evidence_file = evidence_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("Hat Trick video output must be outside the source checkout")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Hat Trick video output must use .mp4")
    if not 10 <= fps <= 60:
        raise ValueError("Hat Trick video fps must be in [10, 60]")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Hat Trick video or manifest already exists")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for Hat Trick video export")
    report = json.loads(evidence_file.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("Hat Trick video requires a passing evidence report")
    sources = _load_sources(report, checkout)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_gl = os.environ.get("MUJOCO_GL")
    os.environ.setdefault("MUJOCO_GL", "egl")
    timelines = tuple(_timeline(source, fps=fps) for source in sources)
    durations = tuple(len(timeline) / fps for timeline in timelines)
    try:
        import mujoco

        from rosclaw.simforge.backends.unitree_mujoco_backend import qualify_g1_assets

        qualification = qualify_g1_assets(asset_root)
        qualification.require_eligible()
        if qualification.body_hash != report["body_hash"]:
            raise ValueError("Hat Trick video Body hash does not match evidence")
        scene = asset_root.expanduser().resolve() / _SCENE_REL
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        comparison_data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=_RENDER_HEIGHT, width=_RENDER_WIDTH)
        try:
            process = subprocess.Popen(
                _ffmpeg_command(
                    ffmpeg=ffmpeg,
                    output=output,
                    fps=fps,
                    sources=sources,
                    durations=durations,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None:
                raise RuntimeError("ffmpeg raw-video pipe is unavailable")
            try:
                _write_frames(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    comparison_data=comparison_data,
                    renderer=renderer,
                    sources=sources,
                    timelines=timelines,
                    stream=cast(BinaryIO, process.stdin),
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
                raise RuntimeError(f"Hat Trick ffmpeg failed ({code}): {stderr[-2000:]}")
        finally:
            renderer.close()
    finally:
        if previous_gl is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = previous_gl

    clips = []
    offset = 0.0
    for source, timeline, duration in zip(sources, timelines, durations, strict=True):
        clips.append(
            HatTrickVideoClip(
                name=source.name,
                title=source.title,
                source_trajectory_hash=source.trajectory_hash,
                playback_start_sec=offset,
                playback_duration_sec=duration,
                frame_count=len(timeline),
                success=bool(source.result["success"]),
                target_error_m=float(source.result["target_error_m"]),
                ball_speed_mps=float(source.result["ball_speed_mps"]),
                tail_wobble_reduction=float(
                    source.recovery_comparison.get("tail_wobble_reduction", 0.0)
                ),
            )
        )
        offset += duration
    manifest = output.with_suffix(".json")
    result = HatTrickVideoResult(
        output_path=str(output),
        manifest_path=str(manifest),
        video_hash=_hash_file(output),
        evidence_report_hash=_hash_file(evidence_file),
        width=1280,
        height=720,
        fps=fps,
        frame_count=sum(len(item) for item in timelines),
        duration_sec=offset,
        clips=tuple(clips),
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _load_sources(report: dict[str, Any], checkout: Path) -> tuple[_Source, ...]:
    shots = report.get("shots")
    if not isinstance(shots, list) or len(shots) != 3:
        raise ValueError("Hat Trick evidence must contain exactly three shots")
    result = []
    for shot in shots:
        path = Path(str(shot["trajectory_path"])).expanduser().resolve()
        if checkout == path or checkout in path.parents:
            raise ValueError("Hat Trick source trajectory must be outside the checkout")
        if _hash_file(path) != shot["trajectory_hash"]:
            raise ValueError("Hat Trick trajectory hash mismatch")
        trajectory = _load_trajectory(path)
        comparison = None
        comparison_path = shot.get("comparison_trajectory_path")
        if comparison_path:
            resolved = Path(str(comparison_path)).expanduser().resolve()
            if checkout == resolved or checkout in resolved.parents:
                raise ValueError("Hat Trick comparison trajectory must be outside the checkout")
            if _hash_file(resolved) != shot["comparison_trajectory_hash"]:
                raise ValueError("Hat Trick comparison trajectory hash mismatch")
            comparison = _load_trajectory(resolved)
        result.append(
            _Source(
                name=str(shot["name"]),
                title=str(shot["title"]),
                result=dict(shot["result"]),
                scenario=dict(shot["scenario"]),
                trajectory_hash=str(shot["trajectory_hash"]),
                trajectory=trajectory,
                comparison=comparison,
                recovery_metrics=dict(shot.get("recovery_metrics") or {}),
                recovery_comparison=dict(shot.get("recovery_comparison") or {}),
            )
        )
    return tuple(result)


def _load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        value = {name: archive[name] for name in archive.files}
    for name, shape in {
        "time": (),
        "pelvis_pose": (7,),
        "joint_position": (29,),
        "ball_pose": (7,),
    }.items():
        array = np.asarray(value.get(name))
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
            raise ValueError(f"Hat Trick trajectory {name} has invalid shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Hat Trick trajectory {name} is non-finite")
    return value


def _timeline(source: _Source, *, fps: int) -> tuple[float, ...]:
    contact = float(source.result["ball_contact_time_sec"])
    start = max(float(source.trajectory["time"][0]), contact - 2.7)
    end = min(float(source.trajectory["time"][-1]), contact + 7.5)
    segments = (
        (start, contact - 0.45, 1.35),
        (contact - 0.45, contact + 0.75, 0.45),
        (contact + 0.75, min(end, contact + 3.2), 1.45),
        (min(end, contact + 3.2), end, 2.10),
    )
    values: list[float] = []
    for segment_start, segment_end, speed in segments:
        if segment_end <= segment_start:
            continue
        count = max(1, int(np.ceil((segment_end - segment_start) / speed * fps)))
        values.extend(
            min(segment_end, segment_start + frame / fps * speed) for frame in range(count)
        )
    return tuple(values)


def _write_frames(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    comparison_data: Any,
    renderer: Any,
    sources: tuple[_Source, ...],
    timelines: tuple[tuple[float, ...], ...],
    stream: BinaryIO,
) -> None:
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (1.4, 0.0, 0.72)
    camera.distance = 3.6
    camera.azimuth = 92.0
    camera.elevation = -8.0
    for source, timeline in zip(sources, timelines, strict=True):
        for simulation_time in timeline:
            if source.comparison is None:
                frame = _render_pose(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    renderer=renderer,
                    camera=camera,
                    trajectory=source.trajectory,
                    simulation_time=simulation_time,
                    ball_qpos=ball_qpos,
                    scenario=source.scenario,
                    contact_time=float(source.result["ball_contact_time_sec"]),
                    show_grid=source.name == "nine_grid_power",
                    show_push=source.name == "disturbance_feedback_rescue",
                )
                canvas = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
            else:
                off = _render_pose(
                    mujoco=mujoco,
                    model=model,
                    data=comparison_data,
                    renderer=renderer,
                    camera=camera,
                    trajectory=source.comparison,
                    simulation_time=simulation_time,
                    ball_qpos=ball_qpos,
                    scenario=source.scenario,
                    contact_time=float(source.result["ball_contact_time_sec"]),
                    show_grid=False,
                    show_push=True,
                )
                on = _render_pose(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    renderer=renderer,
                    camera=camera,
                    trajectory=source.trajectory,
                    simulation_time=simulation_time,
                    ball_qpos=ball_qpos,
                    scenario=source.scenario,
                    contact_time=float(source.result["ball_contact_time_sec"]),
                    show_grid=False,
                    show_push=True,
                )
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                canvas[180:540, :640] = off
                canvas[180:540, 640:] = on
            stream.write(np.ascontiguousarray(canvas).tobytes())


def _render_pose(
    *,
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    camera: Any,
    trajectory: dict[str, np.ndarray],
    simulation_time: float,
    ball_qpos: int,
    scenario: dict[str, Any],
    contact_time: float,
    show_grid: bool,
    show_push: bool,
) -> np.ndarray:
    times = trajectory["time"]
    index = min(int(np.searchsorted(times, simulation_time, side="left")), len(times) - 1)
    data.qpos[:] = model.qpos0
    data.qpos[:7] = trajectory["pelvis_pose"][index]
    data.qpos[7:36] = trajectory["joint_position"][index]
    data.qpos[ball_qpos : ball_qpos + 7] = trajectory["ball_pose"][index]
    mujoco.mj_forward(model, data)
    if contact_time + 0.12 <= simulation_time < contact_time + 2.2:
        camera.lookat[:] = (3.0, 0.0, 0.65)
        camera.distance = 6.1
        camera.azimuth = 90.0
        camera.elevation = -7.0
    elif simulation_time >= contact_time + 2.2:
        camera.lookat[:] = np.asarray(trajectory["pelvis_pose"][index, :3], dtype=np.float64)
        camera.lookat[2] = 0.72
        camera.distance = 3.2
        camera.azimuth = 92.0
        camera.elevation = -8.0
    else:
        camera.lookat[:] = (1.4, 0.0, 0.72)
        camera.distance = 3.6
        camera.azimuth = 92.0
        camera.elevation = -8.0
    renderer.update_scene(data, camera=camera)
    _add_targets_and_trail(
        mujoco=mujoco,
        scene=renderer.scene,
        scenario=scenario,
        trajectory=trajectory,
        index=index,
        show_grid=show_grid,
    )
    if show_push and 4.35 <= simulation_time <= 5.05:
        pelvis = np.asarray(trajectory["pelvis_pose"][index, :3], dtype=np.float64)
        for offset in np.linspace(-0.6, -0.15, 6):
            _append_sphere(
                mujoco,
                renderer.scene,
                pelvis + np.asarray((0.0, offset, 0.10)),
                0.025 + 0.02 * (offset + 0.6),
                (1.0, 0.15, 0.12, 0.85),
            )
    return renderer.render().copy()


def _add_targets_and_trail(
    *,
    mujoco: Any,
    scene: Any,
    scenario: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    index: int,
    show_grid: bool,
) -> None:
    target_y = float(scenario["target_y_m"])
    target_z = float(scenario["target_z_m"])
    targets = (
        [(y, z) for y in (-0.75, 0.0, 0.75) for z in (0.20, 0.55, 0.90)]
        if show_grid
        else [(target_y, target_z)]
    )
    for y, z in targets:
        active = abs(y - target_y) < 1e-6 and abs(z - target_z) < 1e-6
        _append_sphere(
            mujoco,
            scene,
            np.asarray((5.02, y, z)),
            0.17 if active else 0.08,
            (0.15, 1.0, 0.35, 0.92) if active else (0.15, 0.45, 0.65, 0.25),
        )
    start = max(0, index - 70)
    indices = np.linspace(start, index, min(14, index - start + 1), dtype=int)
    for trail_index, alpha in zip(indices, np.linspace(0.05, 0.55, len(indices)), strict=True):
        _append_sphere(
            mujoco,
            scene,
            np.asarray(trajectory["ball_pose"][trail_index, :3]),
            0.032,
            (0.25, 0.78, 1.0, float(alpha)),
        )


def _append_sphere(
    mujoco: Any,
    scene: Any,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray((radius,) * 3, dtype=np.float64),
        position,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _ffmpeg_command(
    *,
    ffmpeg: str,
    output: Path,
    fps: int,
    sources: tuple[_Source, ...],
    durations: tuple[float, ...],
) -> list[str]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font_option = f"fontfile={font}:" if font.is_file() else ""
    filters = [
        "drawbox=x=0:y=0:w=iw:h=112:color=0x050A12@0.78:t=fill",
        "drawbox=x=0:y=h-70:w=iw:h=70:color=0x050A12@0.78:t=fill",
        f"drawtext={font_option}text='ROSClaw GoalForge Hat Trick':x=34:y=18:fontsize=36:fontcolor=white",
        f"drawtext={font_option}text='SIM EVIDENCE REPLAY · VISUALIZATION ONLY':x=34:y=h-46:fontsize=22:fontcolor=0x8DD8FF",
    ]
    offset = 0.0
    for source, duration in zip(sources, durations, strict=True):
        end = offset + duration
        result = source.result
        wobble_reduction = 100.0 * float(
            source.recovery_comparison.get("tail_wobble_reduction", 0.0)
        )
        metrics = (
            f"{source.title}  ·  {float(result['ball_speed_mps']):.2f} m/s  ·  "
            f"error {float(result['target_error_m']):.3f} m  ·  "
            f"RECOVERY WOBBLE -{wobble_reduction:.0f} pct"
        )
        filters.append(
            f"drawtext={font_option}text='{metrics}':x=34:y=68:fontsize=22:"
            f"fontcolor=0x65F59A:enable='between(t,{offset:.6f},{end:.6f})'"
        )
        if source.comparison is not None:
            filters.extend(
                (
                    f"drawtext={font_option}text='FEEDBACK OFF · FALL / UNSAFE':x=120:y=145:fontsize=22:fontcolor=0xFF6262:enable='between(t,{offset:.6f},{end:.6f})'",
                    f"drawtext={font_option}text='FEEDBACK + CEREBELLUM · GOAL + SETTLED':x=700:y=145:fontsize=20:fontcolor=0x65F59A:enable='between(t,{offset:.6f},{end:.6f})'",
                )
            )
        offset = end
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
    "HatTrickVideoClip",
    "HatTrickVideoResult",
    "render_goalforge_hat_trick_video",
]
