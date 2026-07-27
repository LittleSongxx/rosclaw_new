"""Web dashboard for ROSClaw CMU ARE task artifacts."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


_STATE_FILE = "dashboard_state.json"
_TRASH_DIR = ".trash"
_SAFE_FILE_NAMES = {
    "summary.json",
    "task_events.jsonl",
    "odom_trace.jsonl",
    "waypoints.jsonl",
    "path_trace.jsonl",
    "ros_topics.txt",
}
_SORT_FIELDS = {"mtime", "duration", "error", "status", "type", "instruction", "updated", "episode"}
_CAMPUS_PREVIEW = (
    Path(__file__).resolve().parents[3]
    / "third_party/ros1/are/src/vehicle_simulator/mesh/campus/preview/overview.png"
)
_DEFAULT_BACKGROUND_SETTINGS = {
    "enabled": False,
    "opacity": 0.3,
    "scale": 1.0,
    "offset_x": 0.0,
    "offset_y": 0.0,
    "rotation_deg": 0.0,
}


@dataclass(frozen=True)
class CmuDashboardConfig:
    output_dir: str | Path = "practice_data/app_runs"
    max_points: int = 2000
    connect_ros: bool = True


class CmuDashboardStore:
    """Managed view over CMU app artifacts."""

    def __init__(self, output_dir: str | Path = "practice_data/app_runs", *, max_points: int = 2000) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.max_points = int(max_points)
        self.state_path = self.output_dir / _STATE_FILE
        self.trash_dir = self.output_dir / _TRASH_DIR

    def health(self, *, connect_ros: bool = True) -> dict[str, Any]:
        ros = get_ros_status() if connect_ros else {"available": False, "reason": "disabled"}
        state = self._load_state()
        return {
            "status": "ok",
            "output_dir": str(self.output_dir),
            "output_dir_exists": self.output_dir.exists(),
            "task_count": len(self.list_tasks()),
            "trash_count": len(self._trash_tasks(state)),
            "metadata_path": str(self.state_path),
            "campus_preview_available": _CAMPUS_PREVIEW.exists(),
            "ros": ros,
            "timestamp": time.time(),
        }

    def list_tasks(self) -> list[dict[str, Any]]:
        return self.query_tasks()["tasks"]

    def query_tasks(
        self,
        *,
        q: str = "",
        status: str = "",
        task_type: str = "",
        tag: str = "",
        archived: bool = False,
        deleted: bool = False,
        sort: str = "mtime",
        order: str = "desc",
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        state = self._load_state()
        tasks = self._trash_tasks(state) if deleted else self._active_tasks(state)
        if not deleted:
            tasks = [task for task in tasks if bool(task.get("archived")) == bool(archived)]
        tasks = _filter_tasks(tasks, q=q, status=status, task_type=task_type, tag=tag)
        tasks = _sort_tasks(tasks, sort=sort, order=order, pinned_first=not deleted)
        total = len(tasks)
        offset = max(0, int(offset or 0))
        if limit is not None:
            limit = max(0, int(limit))
            tasks = tasks[offset : offset + limit]
        else:
            tasks = tasks[offset:]
        return {
            "tasks": tasks,
            "count": len(tasks),
            "total": total,
            "offset": offset,
            "limit": limit,
            "filters": {
                "q": q,
                "status": status,
                "type": task_type,
                "tag": tag,
                "archived": archived,
                "deleted": deleted,
                "sort": sort if sort in _SORT_FIELDS else "mtime",
                "order": "asc" if str(order).lower() == "asc" else "desc",
            },
        }

    def selected_tasks(self, episode_ids: list[str]) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for episode_id in _safe_ids(episode_ids):
            try:
                tasks.append(self.get_task(episode_id))
            except FileNotFoundError:
                continue
        return tasks

    def _active_tasks(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        if not self.output_dir.exists():
            return tasks
        for summary_path in self.output_dir.glob("app_cmu_*/summary.json"):
            summary = _read_json(summary_path)
            if not isinstance(summary, dict):
                continue
            episode_id = str(summary.get("episode_id") or summary_path.parent.name)
            tasks.append(_task_overview(summary_path.parent, summary, metadata=_task_metadata(state, episode_id)))
        return tasks

    def _trash_tasks(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        if not self.trash_dir.exists():
            return tasks
        trash_state = state.get("trash") if isinstance(state.get("trash"), dict) else {}
        for summary_path in self.trash_dir.glob("*/summary.json"):
            summary = _read_json(summary_path)
            if not isinstance(summary, dict):
                continue
            trash_id = summary_path.parent.name
            trash_meta = trash_state.get(trash_id) if isinstance(trash_state.get(trash_id), dict) else {}
            original_episode_id = str(trash_meta.get("original_episode_id") or summary.get("episode_id") or trash_id)
            task = _task_overview(
                summary_path.parent,
                summary,
                metadata=_task_metadata(state, original_episode_id),
            )
            task.update(
                {
                    "episode_id": original_episode_id,
                    "trash_id": trash_id,
                    "deleted": True,
                    "deleted_at": trash_meta.get("deleted_at"),
                    "archived": False,
                    "pinned": False,
                }
            )
            tasks.append(task)
        return tasks

    def get_task(self, episode_id: str) -> dict[str, Any]:
        state = self._load_state()
        task_dir = self._task_dir(episode_id)
        return self._task_detail(task_dir, episode_id=episode_id, state=state)

    def get_trash_task(self, trash_id: str) -> dict[str, Any]:
        state = self._load_state()
        trash_dir = self._trash_task_dir(trash_id)
        trash = state.get("trash") if isinstance(state.get("trash"), dict) else {}
        trash_meta = trash.get(trash_id) if isinstance(trash.get(trash_id), dict) else {}
        episode_id = str(trash_meta.get("original_episode_id") or trash_id.split("__", 1)[0])
        task = self._task_detail(trash_dir, episode_id=episode_id, state=state)
        task.update({"episode_id": episode_id, "trash_id": trash_id, "deleted": True, "deleted_at": trash_meta.get("deleted_at")})
        return task

    def _task_detail(self, task_dir: Path, *, episode_id: str, state: dict[str, Any]) -> dict[str, Any]:
        summary_path = task_dir / "summary.json"
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            raise FileNotFoundError(episode_id)
        task = _task_overview(task_dir, summary, metadata=_task_metadata(state, episode_id))
        return {
            **task,
            "summary": summary,
            "events": _read_jsonl(task_dir / "task_events.jsonl", limit=500),
            "ros_topics": _read_text_lines(task_dir / "ros_topics.txt"),
            "files": sorted(path.name for path in task_dir.iterdir() if path.is_file() and path.name in _SAFE_FILE_NAMES),
        }

    def trajectory(self, episode_id: str) -> dict[str, Any]:
        task_dir = self._task_dir(episode_id)
        return self._trajectory_from_dir(task_dir, episode_id=episode_id)

    def trash_trajectory(self, trash_id: str) -> dict[str, Any]:
        task_dir = self._trash_task_dir(trash_id)
        state = self._load_state()
        trash = state.get("trash") if isinstance(state.get("trash"), dict) else {}
        trash_meta = trash.get(trash_id) if isinstance(trash.get(trash_id), dict) else {}
        episode_id = str(trash_meta.get("original_episode_id") or trash_id.split("__", 1)[0])
        return self._trajectory_from_dir(task_dir, episode_id=episode_id)

    def _trajectory_from_dir(self, task_dir: Path, *, episode_id: str) -> dict[str, Any]:
        if not task_dir.exists():
            raise FileNotFoundError(episode_id)
        odom = _sample_points(_read_jsonl(task_dir / "odom_trace.jsonl"), self.max_points)
        waypoints = _sample_points(_read_jsonl(task_dir / "waypoints.jsonl"), self.max_points)
        path_trace = _sample_points(_read_jsonl(task_dir / "path_trace.jsonl"), min(self.max_points, 500))
        events = _read_jsonl(task_dir / "task_events.jsonl", limit=1000)
        summary = _read_json(task_dir / "summary.json") or {}
        return {
            "episode_id": episode_id,
            "summary": summary,
            "odom": odom,
            "waypoints": waypoints,
            "path_trace": path_trace,
            "events": events,
            "bounds": _bounds([*odom, *waypoints, *path_trace]),
            "counts": {
                "odom": len(odom),
                "waypoints": len(waypoints),
                "path_trace": len(path_trace),
                "events": len(events),
            },
        }

    def stats(self) -> dict[str, Any]:
        tasks = self.list_tasks()
        total = len(tasks)
        successes = [task for task in tasks if task.get("status") == "success"]
        failures = [task for task in tasks if task.get("status") not in {"success", None}]
        durations = [float(task["duration_sec"]) for task in tasks if isinstance(task.get("duration_sec"), (int, float))]
        errors = [
            float(task["distance_to_goal"])
            for task in tasks
            if isinstance(task.get("distance_to_goal"), (int, float))
        ]
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        tags: dict[str, int] = {}
        for task in tasks:
            key = str(task.get("intent_type") or task.get("task_kind") or "unknown")
            by_type[key] = by_type.get(key, 0) + 1
            status = str(task.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            for tag in task.get("tags") or []:
                tags[str(tag)] = tags.get(str(tag), 0) + 1
        return {
            "total": total,
            "success": len(successes),
            "failed": len(failures),
            "success_rate": round(len(successes) / total, 3) if total else 0.0,
            "avg_duration_sec": round(sum(durations) / len(durations), 3) if durations else None,
            "avg_goal_error_m": round(sum(errors) / len(errors), 3) if errors else None,
            "by_type": by_type,
            "by_status": by_status,
            "tags": tags,
            "recent_failures": failures[:5],
        }

    def diagnostics(self) -> dict[str, Any]:
        tasks = self.list_tasks()
        issues: list[dict[str, Any]] = []
        for task in tasks:
            task_issues: list[str] = []
            if task.get("status") in {"failed", "timeout", "cancelled"}:
                task_issues.append(f"状态为 {task.get('status')}")
            if not task.get("has_odom"):
                task_issues.append("缺少 odom_trace.jsonl")
            if not task.get("has_events"):
                task_issues.append("缺少 task_events.jsonl")
            if isinstance(task.get("distance_to_goal"), (int, float)) and float(task["distance_to_goal"]) > 2.0:
                task_issues.append(f"最终误差偏大：{float(task['distance_to_goal']):.2f}m")
            if task_issues:
                issues.append({"episode_id": task["episode_id"], "instruction": task.get("instruction", ""), "issues": task_issues})
        return {"issue_count": len(issues), "issues": issues[:50]}

    def compare_tasks(self, episode_ids: list[str]) -> dict[str, Any]:
        tasks = self.selected_tasks(episode_ids)
        rows: list[dict[str, Any]] = []
        for task in tasks:
            trajectory = self.trajectory(task["episode_id"])
            rows.append(
                {
                    "episode_id": task["episode_id"],
                    "instruction": task.get("instruction", ""),
                    "status": task.get("status"),
                    "type_label": task.get("type_label"),
                    "duration_sec": task.get("duration_sec"),
                    "distance_to_goal": task.get("distance_to_goal"),
                    "odom_points": trajectory["counts"]["odom"],
                    "waypoints": trajectory["counts"]["waypoints"],
                    "path_points": trajectory["counts"]["path_trace"],
                    "trajectory_length_m": _trajectory_length(trajectory["odom"]),
                }
            )
        return {"count": len(rows), "tasks": rows}

    def update_metadata(self, episode_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        task_dir = self._task_dir(episode_id)
        if not (task_dir / "summary.json").exists():
            raise FileNotFoundError(episode_id)
        state = self._load_state()
        tasks = state.setdefault("tasks", {})
        if not isinstance(tasks, dict):
            tasks = {}
            state["tasks"] = tasks
        metadata = _task_metadata(state, episode_id)
        if "pinned" in patch:
            metadata["pinned"] = bool(patch.get("pinned"))
        if "archived" in patch:
            metadata["archived"] = bool(patch.get("archived"))
        if "note" in patch:
            metadata["note"] = str(patch.get("note") or "")[:2000]
        if "tags" in patch:
            metadata["tags"] = _sanitize_tags(patch.get("tags"))
        metadata["updated_at"] = time.time()
        tasks[episode_id] = metadata
        self._save_state(state)
        return self.get_task(episode_id)

    def batch_update_metadata(self, episode_ids: list[str], patch: dict[str, Any]) -> dict[str, Any]:
        updated: list[str] = []
        missing: list[str] = []
        for episode_id in _safe_ids(episode_ids):
            try:
                current = self.get_task(episode_id)
                next_patch = dict(patch)
                if "add_tags" in next_patch:
                    next_patch["tags"] = [*current.get("tags", []), *_sanitize_tags(next_patch.pop("add_tags"))]
                self.update_metadata(episode_id, next_patch)
                updated.append(episode_id)
            except FileNotFoundError:
                missing.append(episode_id)
        return {"updated": updated, "missing": missing, "count": len(updated)}

    def soft_delete(self, episode_id: str) -> dict[str, Any]:
        task_dir = self._task_dir(episode_id)
        if not (task_dir / "summary.json").exists():
            raise FileNotFoundError(episode_id)
        state = self._load_state()
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        base_trash_id = f"{episode_id}__{int(time.time())}"
        trash_id = base_trash_id
        suffix = 1
        while (self.trash_dir / trash_id).exists():
            suffix += 1
            trash_id = f"{base_trash_id}_{suffix}"
        shutil.move(str(task_dir), str(self.trash_dir / trash_id))
        trash = state.setdefault("trash", {})
        if not isinstance(trash, dict):
            trash = {}
            state["trash"] = trash
        trash[trash_id] = {"original_episode_id": episode_id, "deleted_at": time.time()}
        self._save_state(state)
        return {"status": "deleted", "episode_id": episode_id, "trash_id": trash_id}

    def batch_soft_delete(self, episode_ids: list[str]) -> dict[str, Any]:
        deleted: list[dict[str, Any]] = []
        missing: list[str] = []
        for episode_id in _safe_ids(episode_ids):
            try:
                deleted.append(self.soft_delete(episode_id))
            except FileNotFoundError:
                missing.append(episode_id)
        return {"deleted": deleted, "missing": missing, "count": len(deleted)}

    def restore_trash(self, trash_id: str) -> dict[str, Any]:
        trash_dir = self._trash_task_dir(trash_id)
        if not (trash_dir / "summary.json").exists():
            raise FileNotFoundError(trash_id)
        state = self._load_state()
        trash = state.get("trash") if isinstance(state.get("trash"), dict) else {}
        trash_meta = trash.get(trash_id) if isinstance(trash.get(trash_id), dict) else {}
        episode_id = str(trash_meta.get("original_episode_id") or trash_id.split("__", 1)[0])
        target = self.output_dir / episode_id
        if target.exists():
            target = self.output_dir / f"{episode_id}_restored_{int(time.time())}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(trash_dir), str(target))
        if isinstance(trash, dict):
            trash.pop(trash_id, None)
            state["trash"] = trash
            self._save_state(state)
        return {"status": "restored", "episode_id": target.name, "trash_id": trash_id}

    def batch_restore_trash(self, trash_ids: list[str]) -> dict[str, Any]:
        restored: list[dict[str, Any]] = []
        missing: list[str] = []
        for trash_id in _safe_ids(trash_ids):
            try:
                restored.append(self.restore_trash(trash_id))
            except FileNotFoundError:
                missing.append(trash_id)
        return {"restored": restored, "missing": missing, "count": len(restored)}

    def delete_trash(self, trash_id: str) -> dict[str, Any]:
        trash_dir = self._trash_task_dir(trash_id)
        if not trash_dir.exists():
            raise FileNotFoundError(trash_id)
        shutil.rmtree(trash_dir)
        state = self._load_state()
        trash = state.get("trash") if isinstance(state.get("trash"), dict) else {}
        if isinstance(trash, dict):
            trash.pop(trash_id, None)
            state["trash"] = trash
            self._save_state(state)
        return {"status": "purged", "trash_id": trash_id}

    def batch_delete_trash(self, trash_ids: list[str]) -> dict[str, Any]:
        purged: list[dict[str, Any]] = []
        missing: list[str] = []
        for trash_id in _safe_ids(trash_ids):
            try:
                purged.append(self.delete_trash(trash_id))
            except FileNotFoundError:
                missing.append(trash_id)
        return {"purged": purged, "missing": missing, "count": len(purged)}

    def read_file(self, episode_id: str, filename: str) -> str:
        if filename not in _SAFE_FILE_NAMES:
            raise FileNotFoundError(filename)
        path = self._task_dir(episode_id) / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return path.read_text(encoding="utf-8", errors="replace")

    def read_trash_file(self, trash_id: str, filename: str) -> str:
        if filename not in _SAFE_FILE_NAMES:
            raise FileNotFoundError(filename)
        path = self._trash_task_dir(trash_id) / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return path.read_text(encoding="utf-8", errors="replace")

    def export_task(self, episode_id: str, *, format: str = "json") -> Tuple[str, str, str]:
        detail = self.get_task(episode_id)
        trajectory = self.trajectory(episode_id)
        if format == "json":
            body = json.dumps({"task": detail, "trajectory": trajectory}, ensure_ascii=False, indent=2)
            return body, "application/json", f"{episode_id}.json"
        if format == "csv":
            return _tasks_to_csv([detail]), "text/csv; charset=utf-8", f"{episode_id}.csv"
        raise ValueError(format)

    def export_tasks_csv(self, **query: Any) -> str:
        return _tasks_to_csv(self.query_tasks(**query)["tasks"])

    def export_selected_csv(self, episode_ids: list[str]) -> str:
        return _tasks_to_csv(self.selected_tasks(episode_ids))

    def dashboard_settings(self) -> dict[str, Any]:
        state = self._load_state()
        settings = state.get("settings") if isinstance(state.get("settings"), dict) else {}
        background = settings.get("plot_background") if isinstance(settings.get("plot_background"), dict) else {}
        return {
            "plot_background": {
                **_DEFAULT_BACKGROUND_SETTINGS,
                **{key: background[key] for key in _DEFAULT_BACKGROUND_SETTINGS if key in background},
                "preview_available": _CAMPUS_PREVIEW.exists(),
            }
        }

    def update_dashboard_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        state = self._load_state()
        settings = state.setdefault("settings", {})
        if not isinstance(settings, dict):
            settings = {}
            state["settings"] = settings
        background_patch = patch.get("plot_background") if isinstance(patch.get("plot_background"), dict) else {}
        current = dict(self.dashboard_settings()["plot_background"])
        current.update(_sanitize_background_settings(background_patch))
        settings["plot_background"] = current
        self._save_state(state)
        return self.dashboard_settings()

    def _load_state(self) -> dict[str, Any]:
        state = _read_json(self.state_path)
        if not isinstance(state, dict):
            return {"version": 1, "tasks": {}, "trash": {}, "settings": {}}
        if not isinstance(state.get("tasks"), dict):
            state["tasks"] = {}
        if not isinstance(state.get("trash"), dict):
            state["trash"] = {}
        if not isinstance(state.get("settings"), dict):
            state["settings"] = {}
        state.setdefault("version", 1)
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.state_path)

    def _task_dir(self, episode_id: str) -> Path:
        if not _is_safe_id(episode_id):
            raise FileNotFoundError(episode_id)
        return self.output_dir / episode_id

    def _trash_task_dir(self, trash_id: str) -> Path:
        if not _is_safe_id(trash_id):
            raise FileNotFoundError(trash_id)
        return self.trash_dir / trash_id


def create_cmu_dashboard_app(config: Optional[CmuDashboardConfig] = None) -> Any:
    """Create the FastAPI app lazily so importing data helpers does not require FastAPI."""

    try:
        from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, PlainTextResponse, Response
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("FastAPI is required for cmu-dashboard") from exc

    cfg = config or CmuDashboardConfig()
    store = CmuDashboardStore(cfg.output_dir, max_points=cfg.max_points)
    app = FastAPI(title="ROSClaw CMU Dashboard", version="1.1.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _HTML

    @app.get("/api/health")
    async def health() -> dict:
        return store.health(connect_ros=cfg.connect_ros)

    @app.get("/api/tasks")
    async def tasks(
        q: str = "",
        status: str = "",
        type: str = Query("", alias="type"),
        tag: str = "",
        archived: bool = False,
        deleted: bool = False,
        sort: str = "mtime",
        order: str = "desc",
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> dict:
        return store.query_tasks(
            q=q,
            status=status,
            task_type=type,
            tag=tag,
            archived=archived,
            deleted=deleted,
            sort=sort,
            order=order,
            limit=limit,
            offset=offset,
        )

    @app.post("/api/tasks/batch/metadata")
    async def batch_update_metadata(payload: dict = Body(...)) -> dict:
        ids = payload.get("episode_ids") if isinstance(payload, dict) else []
        patch = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return store.batch_update_metadata(ids if isinstance(ids, list) else [], patch)

    @app.post("/api/tasks/batch/delete")
    async def batch_delete_tasks(payload: dict = Body(...)) -> dict:
        ids = payload.get("episode_ids") if isinstance(payload, dict) else []
        return store.batch_soft_delete(ids if isinstance(ids, list) else [])

    @app.post("/api/trash/batch/restore")
    async def batch_restore_trash(payload: dict = Body(...)) -> dict:
        ids = payload.get("trash_ids") if isinstance(payload, dict) else []
        return store.batch_restore_trash(ids if isinstance(ids, list) else [])

    @app.post("/api/trash/batch/purge")
    async def batch_purge_trash(payload: dict = Body(...)) -> dict:
        ids = payload.get("trash_ids") if isinstance(payload, dict) else []
        return store.batch_delete_trash(ids if isinstance(ids, list) else [])

    @app.post("/api/export/tasks/selected")
    async def export_selected_tasks(payload: dict = Body(...)) -> Response:
        ids = payload.get("episode_ids") if isinstance(payload, dict) else []
        body = store.export_selected_csv(ids if isinstance(ids, list) else [])
        return Response(
            content=body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="cmu_selected_tasks.csv"'},
        )

    @app.post("/api/tasks/compare")
    async def compare_tasks(payload: dict = Body(...)) -> dict:
        ids = payload.get("episode_ids") if isinstance(payload, dict) else []
        return store.compare_tasks(ids if isinstance(ids, list) else [])

    @app.get("/api/diagnostics")
    async def diagnostics() -> dict:
        return store.diagnostics()

    @app.get("/api/settings")
    async def settings() -> dict:
        return store.dashboard_settings()

    @app.patch("/api/settings")
    async def update_settings(payload: dict = Body(...)) -> dict:
        return store.update_dashboard_settings(payload if isinstance(payload, dict) else {})

    @app.get("/api/worlds/campus/preview")
    async def campus_preview() -> Response:
        if not _CAMPUS_PREVIEW.exists():
            raise HTTPException(status_code=404, detail="campus preview not found")
        return Response(content=_CAMPUS_PREVIEW.read_bytes(), media_type="image/png")

    @app.get("/api/tasks/{episode_id}")
    async def task_detail(episode_id: str) -> dict:
        try:
            return store.get_task(episode_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/api/tasks/{episode_id}/trajectory")
    async def trajectory(episode_id: str) -> dict:
        try:
            return store.trajectory(episode_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/api/trash/{trash_id}/trajectory")
    async def trash_trajectory(trash_id: str) -> dict:
        try:
            return store.trash_trajectory(trash_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="trash item not found") from exc

    @app.get("/api/trash/{trash_id}/detail")
    async def trash_detail(trash_id: str) -> dict:
        try:
            return store.get_trash_task(trash_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="trash item not found") from exc

    @app.patch("/api/tasks/{episode_id}/metadata")
    async def update_metadata(episode_id: str, payload: dict = Body(...)) -> dict:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="metadata patch must be an object")
        try:
            return store.update_metadata(episode_id, payload)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.post("/api/tasks/{episode_id}/delete")
    async def soft_delete(episode_id: str) -> dict:
        try:
            return store.soft_delete(episode_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.post("/api/trash/{trash_id}/restore")
    async def restore_trash(trash_id: str) -> dict:
        try:
            return store.restore_trash(trash_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="trash item not found") from exc

    @app.delete("/api/trash/{trash_id}")
    async def delete_trash(trash_id: str) -> dict:
        try:
            return store.delete_trash(trash_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="trash item not found") from exc

    @app.get("/api/tasks/{episode_id}/files/{filename}", response_class=PlainTextResponse)
    async def read_task_file(episode_id: str, filename: str) -> str:
        try:
            return store.read_file(episode_id, filename)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="file not found") from exc

    @app.get("/api/trash/{trash_id}/files/{filename}", response_class=PlainTextResponse)
    async def read_trash_file(trash_id: str, filename: str) -> str:
        try:
            return store.read_trash_file(trash_id, filename)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="file not found") from exc

    @app.get("/api/tasks/{episode_id}/export")
    async def export_task(episode_id: str, format: str = "json") -> Response:
        try:
            body, media_type, filename = store.export_task(episode_id, format=format)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="unsupported format") from exc
        return Response(
            content=body,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/export/tasks")
    async def export_tasks(
        q: str = "",
        status: str = "",
        type: str = Query("", alias="type"),
        tag: str = "",
        archived: bool = False,
        deleted: bool = False,
        sort: str = "mtime",
        order: str = "desc",
    ) -> Response:
        body = store.export_tasks_csv(
            q=q,
            status=status,
            task_type=type,
            tag=tag,
            archived=archived,
            deleted=deleted,
            sort=sort,
            order=order,
        )
        return Response(
            content=body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="cmu_tasks.csv"'},
        )

    @app.get("/api/stats")
    async def stats() -> dict:
        return store.stats()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json({"type": "health", "data": store.health(connect_ros=cfg.connect_ros)})
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            return

    return app


def run_cmu_dashboard(
    *,
    host: str = "0.0.0.0",
    port: int = 8770,
    output_dir: str | Path = "practice_data/app_runs",
    max_points: int = 2000,
    connect_ros: bool = True,
) -> None:
    try:
        import uvicorn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("uvicorn is required for cmu-dashboard") from exc

    app = create_cmu_dashboard_app(
        CmuDashboardConfig(output_dir=output_dir, max_points=max_points, connect_ros=connect_ros)
    )
    uvicorn.run(app, host=host, port=int(port), log_level="warning")


def get_ros_status() -> dict[str, Any]:
    """Best-effort ROS1 status snapshot. Safe outside ROS1."""

    try:
        from rosclaw.apps.cmu_are_bridge import CmuAreBridge

        bridge = CmuAreBridge(node_name="rosclaw_cmu_dashboard")
        topics = bridge.list_topics()
        required = ["/state_estimation", "/cmd_vel", "/way_point", "/path"]
        return {
            "available": True,
            "topics": topics,
            "required": {topic: topic in topics for topic in required},
            "pose": bridge.current_pose,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}


def _task_overview(task_dir: Path, summary: dict[str, Any], *, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    metadata = metadata or _empty_metadata()
    intent = summary.get("intent") if isinstance(summary.get("intent"), dict) else {}
    task = summary.get("task") if isinstance(summary.get("task"), dict) else {}
    task_type = intent.get("type") or task.get("kind") or "unknown"
    return {
        "episode_id": task_dir.name,
        "instruction": summary.get("instruction", ""),
        "status": summary.get("status", "unknown"),
        "duration_sec": summary.get("duration_sec"),
        "distance_to_goal": summary.get("distance_to_goal"),
        "intent_type": intent.get("type"),
        "task_kind": task.get("kind"),
        "command": intent.get("command") or task.get("command"),
        "place": intent.get("place"),
        "target": {"x": intent.get("x"), "y": intent.get("y")} if intent.get("x") is not None else None,
        "artifact_dir": str(task_dir),
        "mtime": task_dir.stat().st_mtime,
        "has_events": (task_dir / "task_events.jsonl").exists(),
        "has_odom": (task_dir / "odom_trace.jsonl").exists(),
        "type_label": task_type,
        "pinned": bool(metadata.get("pinned")),
        "archived": bool(metadata.get("archived")),
        "deleted": False,
        "tags": list(metadata.get("tags") or []),
        "note": str(metadata.get("note") or ""),
        "updated_at": metadata.get("updated_at"),
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _read_jsonl(path: Path, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    items.append(value)
                    if limit is not None and len(items) >= limit:
                        break
    except OSError:
        return []
    return items


def _read_text_lines(path: Path) -> list[str]:
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        return []


def _sample_points(items: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if max_points <= 0 or len(items) <= max_points:
        return items
    if max_points == 1:
        return [items[-1]]
    last_index = len(items) - 1
    sampled: list[dict[str, Any]] = []
    for index in range(max_points):
        source_index = round(index * last_index / (max_points - 1))
        sampled.append(items[source_index])
    return sampled


def _bounds(points: Iterable[dict[str, Any]]) -> Optional[dict[str, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        try:
            x = float(point["x"])
            y = float(point["y"])
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return None
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}


def _empty_metadata() -> dict[str, Any]:
    return {"pinned": False, "archived": False, "tags": [], "note": "", "updated_at": None}


def _task_metadata(state: dict[str, Any], episode_id: str) -> dict[str, Any]:
    tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
    raw = tasks.get(episode_id) if isinstance(tasks.get(episode_id), dict) else {}
    metadata = _empty_metadata()
    metadata.update(
        {
            "pinned": bool(raw.get("pinned")),
            "archived": bool(raw.get("archived")),
            "tags": _sanitize_tags(raw.get("tags")),
            "note": str(raw.get("note") or ""),
            "updated_at": raw.get("updated_at"),
        }
    )
    return metadata


def _sanitize_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tags = value.replace("，", ",").split(",")
    elif isinstance(value, list):
        raw_tags = value
    else:
        raw_tags = []
    tags: list[str] = []
    for item in raw_tags:
        tag = str(item).strip()[:32]
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) >= 12:
            break
    return tags


def _filter_tasks(
    tasks: list[dict[str, Any]],
    *,
    q: str = "",
    status: str = "",
    task_type: str = "",
    tag: str = "",
) -> list[dict[str, Any]]:
    query = q.strip().lower()
    status = status.strip()
    task_type = task_type.strip()
    tag = tag.strip()
    filtered: list[dict[str, Any]] = []
    for task in tasks:
        haystack = " ".join(
            str(part or "")
            for part in [
                task.get("episode_id"),
                task.get("instruction"),
                task.get("status"),
                task.get("intent_type"),
                task.get("task_kind"),
                task.get("command"),
                task.get("place"),
                task.get("note"),
                " ".join(task.get("tags") or []),
            ]
        ).lower()
        if query and query not in haystack:
            continue
        if status and str(task.get("status")) != status:
            continue
        if task_type and str(task.get("type_label") or "") != task_type:
            continue
        if tag and tag not in (task.get("tags") or []):
            continue
        filtered.append(task)
    return filtered


def _sort_tasks(tasks: list[dict[str, Any]], *, sort: str = "mtime", order: str = "desc", pinned_first: bool = True) -> list[dict[str, Any]]:
    sort = sort if sort in _SORT_FIELDS else "mtime"
    reverse = str(order).lower() != "asc"

    def value(task: dict[str, Any]) -> Any:
        if sort == "duration":
            return _number_or_default(task.get("duration_sec"), -1.0)
        if sort == "error":
            return _number_or_default(task.get("distance_to_goal"), -1.0)
        if sort == "status":
            return str(task.get("status") or "")
        if sort == "type":
            return str(task.get("type_label") or "")
        if sort == "instruction":
            return str(task.get("instruction") or "")
        if sort == "updated":
            return _number_or_default(task.get("updated_at"), 0.0)
        if sort == "episode":
            return str(task.get("episode_id") or "")
        return _number_or_default(task.get("mtime"), 0.0)

    sorted_tasks = sorted(tasks, key=value, reverse=reverse)
    if pinned_first:
        sorted_tasks = sorted(sorted_tasks, key=lambda task: not bool(task.get("pinned")))
    return sorted_tasks


def _number_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return default
    return number if math.isfinite(number) else default


def _tasks_to_csv(tasks: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fieldnames = [
        "episode_id",
        "instruction",
        "status",
        "type_label",
        "duration_sec",
        "distance_to_goal",
        "pinned",
        "archived",
        "tags",
        "note",
        "artifact_dir",
        "mtime",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for task in tasks:
        row = dict(task)
        row["tags"] = ",".join(str(tag) for tag in row.get("tags") or [])
        writer.writerow(row)
    return output.getvalue()


def _safe_ids(values: list[str]) -> list[str]:
    ids: list[str] = []
    for value in values:
        text = str(value)
        if _is_safe_id(text) and text not in ids:
            ids.append(text)
    return ids


def _sanitize_background_settings(value: dict[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if "enabled" in value:
        settings["enabled"] = bool(value.get("enabled"))
    for key, low, high in [
        ("opacity", 0.0, 1.0),
        ("scale", 0.2, 5.0),
        ("offset_x", -2000.0, 2000.0),
        ("offset_y", -2000.0, 2000.0),
        ("rotation_deg", -180.0, 180.0),
    ]:
        if key not in value:
            continue
        try:
            number = float(value[key])
        except Exception:  # noqa: BLE001
            continue
        if math.isfinite(number):
            settings[key] = min(high, max(low, number))
    return settings


def _trajectory_length(points: list[dict[str, Any]]) -> float:
    total = 0.0
    previous: Optional[Tuple[float, float]] = None
    for point in points:
        try:
            current = (float(point["x"]), float(point["y"]))
        except Exception:  # noqa: BLE001
            continue
        if previous is not None:
            total += math.hypot(current[0] - previous[0], current[1] - previous[1])
        previous = current
    return round(total, 3)


def _is_safe_id(value: str) -> bool:
    return bool(value) and "/" not in value and "\\" not in value and value not in {".", ".."} and "\x00" not in value


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROSClaw CMU Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg:#eaf0f6; --surface:#ffffff; --soft:#f8fafc; --line:#d9e2ec;
      --text:#17202c; --muted:#64748b; --muted-2:#94a3b8;
      --accent:#2563eb; --ok:#15803d; --bad:#b42318; --warn:#b45309;
      --plot:#07111f; --shadow:0 1px 2px rgba(15,23,42,.06),0 8px 24px rgba(15,23,42,.06);
    }
    * { box-sizing:border-box; }
    html, body { height:100%; overflow:hidden; }
    body { margin:0; font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--text); background:var(--bg); }
    button,input,select,textarea { font:inherit; }
    button { cursor:pointer; }
    .app { height:100vh; display:grid; grid-template-rows:56px minmax(0,1fr); }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:0 16px; border-bottom:1px solid var(--line); background:var(--surface); }
    .brand { display:flex; align-items:center; gap:12px; min-width:260px; }
    .mark { width:34px; height:34px; border-radius:8px; display:grid; place-items:center; background:#102033; color:#fff; font-weight:800; }
    h1 { margin:0; font-size:17px; line-height:1.1; }
    .sub { color:var(--muted); font-size:12px; }
    .topStatus { display:flex; align-items:center; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
    .layout { min-height:0; display:grid; grid-template-columns:360px minmax(0,1fr); overflow:hidden; }
    aside { min-height:0; display:grid; grid-template-rows:auto minmax(0,1fr) auto; border-right:1px solid var(--line); background:var(--surface); overflow:hidden; }
    .filters { padding:10px; border-bottom:1px solid var(--line); display:grid; gap:8px; background:#fbfcfe; }
    .filterRow { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .input,.select,textarea { width:100%; border:1px solid var(--line); background:#fff; color:var(--text); border-radius:6px; padding:8px 9px; outline:none; min-width:0; }
    .input:focus,.select:focus,textarea:focus { border-color:#93b4f4; box-shadow:0 0 0 3px rgba(37,99,235,.12); }
    textarea { min-height:72px; resize:vertical; }
    .seg { display:grid; grid-template-columns:repeat(3,1fr); border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .seg button { border:0; border-right:1px solid var(--line); background:#fff; padding:7px 6px; color:var(--muted); }
    .seg button:last-child { border-right:0; }
    .seg button.active { background:#e8f0ff; color:#164ebd; font-weight:650; }
    .sideActions { display:flex; align-items:center; gap:7px; flex-wrap:wrap; }
    .taskList { min-height:0; overflow:auto; background:#fff; }
    .task { width:100%; text-align:left; border:0; border-bottom:1px solid var(--line); background:#fff; padding:10px; display:grid; grid-template-columns:24px minmax(0,1fr); gap:7px; align-items:start; }
    .task:hover { background:#f7fbff; }
    .task.active { background:#eaf2ff; box-shadow:inset 3px 0 0 var(--accent); }
    .taskCheck { margin-top:2px; }
    .taskTop { display:flex; align-items:center; justify-content:space-between; gap:8px; min-width:0; }
    .taskTitle { min-width:0; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .taskMeta { display:flex; align-items:center; gap:7px; color:var(--muted); font-size:12px; flex-wrap:wrap; margin-top:4px; }
    .taskTags { display:flex; gap:5px; flex-wrap:wrap; margin-top:5px; }
    .batchBar { border-top:1px solid var(--line); padding:9px 10px; display:grid; gap:8px; background:#fbfcfe; }
    .batchTop { display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:12px; gap:8px; }
    .batchActions { display:flex; gap:6px; flex-wrap:wrap; }
    main { min-width:0; min-height:0; display:grid; grid-template-rows:auto minmax(0,1fr); gap:10px; padding:12px; overflow:hidden; }
    .stats { display:grid; grid-template-columns:repeat(6,minmax(110px,1fr)); gap:10px; min-height:60px; }
    .stat { border:1px solid var(--line); border-radius:8px; padding:9px 10px; background:var(--surface); box-shadow:var(--shadow); min-width:0; }
    .stat span { display:block; color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .stat b { display:block; margin-top:1px; font-size:20px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .workspace { min-height:0; display:grid; grid-template-columns:minmax(0,1.45fr) minmax(390px,.8fr); gap:10px; align-items:stretch; }
    .panel { min-width:0; min-height:0; border:1px solid var(--line); border-radius:8px; background:var(--surface); box-shadow:var(--shadow); overflow:hidden; }
    .plotPanel { display:grid; grid-template-rows:auto minmax(0,1fr); }
    .detailPanel { display:grid; grid-template-rows:auto auto minmax(0,1fr); }
    .panelHead { min-height:46px; display:flex; align-items:center; justify-content:space-between; gap:10px; padding:9px 11px; border-bottom:1px solid var(--line); background:#fff; }
    .panelHead h2 { margin:0; font-size:15px; }
    .panelBody { min-height:0; padding:10px; }
    .plotBody { min-height:0; display:grid; grid-template-rows:auto minmax(0,1fr) auto; gap:8px; }
    .plotToolbar { display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap; align-items:center; }
    .plotCanvasWrap { min-height:0; position:relative; }
    canvas { width:100%; height:100%; min-height:360px; border:1px solid #1f2c40; border-radius:8px; background:var(--plot); display:block; }
    .layers,.actions { display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
    .bgControls { display:grid; grid-template-columns:repeat(5,minmax(90px,1fr)); gap:7px; align-items:end; padding:8px; border:1px solid var(--line); border-radius:6px; background:#f8fafc; }
    .bgControls label { color:var(--muted); font-size:12px; display:grid; gap:4px; }
    .check { display:inline-flex; align-items:center; gap:5px; color:var(--muted); font-size:12px; }
    .badge,.tag { display:inline-flex; align-items:center; gap:4px; min-height:22px; padding:2px 7px; border-radius:999px; font-size:12px; border:1px solid var(--line); background:#fff; color:var(--muted); white-space:nowrap; }
    .success { color:var(--ok); border-color:#a7e0b4; background:#eefbf1; }
    .failed,.timeout { color:var(--bad); border-color:#f0b5ad; background:#fff1ef; }
    .cancelled { color:var(--warn); border-color:#f5cc88; background:#fff8e8; }
    .pinned { color:#164ebd; border-color:#b7ccff; background:#eef4ff; }
    .quality { color:#7c3aed; border-color:#ddd6fe; background:#f5f3ff; }
    .btn { border:1px solid var(--line); background:#fff; color:var(--text); border-radius:6px; min-height:32px; padding:6px 9px; }
    .btn:hover { background:#f5f7fb; }
    .btn.primary { border-color:#1d4ed8; background:#2563eb; color:#fff; }
    .btn.danger { border-color:#f0b5ad; color:#b42318; }
    .btn.small { min-height:27px; padding:4px 8px; font-size:12px; }
    .tabs { display:flex; gap:4px; padding:8px 8px 0; border-bottom:1px solid var(--line); background:#fbfcfe; overflow:auto; }
    .tab { border:0; border-radius:6px 6px 0 0; background:transparent; color:var(--muted); padding:8px 10px; white-space:nowrap; }
    .tab.active { background:#fff; color:var(--text); box-shadow:0 -1px 0 var(--line),1px 0 0 var(--line),-1px 0 0 var(--line); }
    #detail { overflow:auto; }
    .kv { display:grid; grid-template-columns:112px minmax(0,1fr); gap:6px 10px; align-items:start; }
    .kv dt { color:var(--muted); }
    .kv dd { margin:0; min-width:0; word-break:break-word; }
    .timeline,.plainList { list-style:none; margin:0; padding:0; }
    .timeline li { border-left:3px solid var(--line); padding:0 0 12px 11px; color:var(--muted); }
    .timeline strong { color:var(--text); }
    .plainList li { padding:8px 0; border-bottom:1px solid var(--line); }
    .compareTable { width:100%; border-collapse:collapse; font-size:12px; }
    .compareTable th,.compareTable td { text-align:left; padding:7px; border-bottom:1px solid var(--line); vertical-align:top; }
    pre { max-height:420px; overflow:auto; margin:0; padding:10px; border-radius:8px; background:#111827; color:#e5e7eb; font-size:12px; line-height:1.45; }
    code { color:#334155; background:#f1f5f9; border-radius:4px; padding:1px 4px; }
    .fileList { display:grid; gap:6px; }
    .fileItem { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:8px; border:1px solid var(--line); border-radius:6px; }
    .toast { position:fixed; right:16px; bottom:16px; max-width:460px; padding:10px 12px; border:1px solid var(--line); border-radius:8px; background:#fff; box-shadow:var(--shadow); color:var(--text); display:none; z-index:10; }
    .muted { color:var(--muted); }
    .empty { padding:24px; color:var(--muted); text-align:center; }
    @media (max-width:1100px) {
      html,body { overflow:auto; }
      .app { height:auto; min-height:100vh; }
      .layout { grid-template-columns:1fr; overflow:visible; }
      aside { max-height:460px; border-right:0; border-bottom:1px solid var(--line); }
      main { overflow:visible; }
      .workspace,.stats,.bgControls { grid-template-columns:1fr; }
      .detailPanel,.plotPanel { min-height:520px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand"><div class="mark">RC</div><div><h1>ROSClaw CMU 任务中控台</h1><div class="sub">任务复盘、轨迹诊断、实验数据管理</div></div></div>
      <div class="topStatus" id="topStatus"><span class="badge">loading</span></div>
    </header>
    <div class="layout">
      <aside>
        <div class="filters">
          <input id="search" class="input" placeholder="搜索指令、episode、标签、备注">
          <div class="filterRow">
            <select id="statusFilter" class="select"><option value="">全部状态</option></select>
            <select id="typeFilter" class="select"><option value="">全部类型</option></select>
          </div>
          <div class="filterRow">
            <select id="sortBy" class="select">
              <option value="mtime">按时间</option><option value="duration">按耗时</option><option value="error">按误差</option>
              <option value="status">按状态</option><option value="type">按类型</option><option value="instruction">按指令</option>
            </select>
            <select id="sortOrder" class="select"><option value="desc">降序</option><option value="asc">升序</option></select>
          </div>
          <div class="seg">
            <button id="viewActive" class="active" onclick="setView('active')">任务</button>
            <button id="viewArchived" onclick="setView('archived')">归档</button>
            <button id="viewTrash" onclick="setView('trash')">回收站</button>
          </div>
          <div class="sideActions">
            <button class="btn small" onclick="refresh()">刷新</button>
            <button class="btn small" onclick="selectVisible()">全选当前</button>
            <button class="btn small" onclick="clearSelectionOnly()">取消选择</button>
          </div>
        </div>
        <div id="taskList" class="taskList"></div>
        <div class="batchBar">
          <div class="batchTop"><span id="listCount">0 项</span><span id="selectedCount">已选 0</span></div>
          <div class="batchActions" id="batchActions"></div>
        </div>
      </aside>
      <main>
        <div class="stats" id="stats"></div>
        <div class="workspace">
          <div class="panel plotPanel">
            <div class="panelHead">
              <h2>轨迹回放</h2>
              <div class="layers">
                <label class="check"><input type="checkbox" id="layerOdom" checked onchange="redraw()">odom</label>
                <label class="check"><input type="checkbox" id="layerWaypoints" checked onchange="redraw()">waypoints</label>
                <label class="check"><input type="checkbox" id="layerPath" checked onchange="redraw()">path</label>
                <label class="check"><input type="checkbox" id="layerEndpoints" checked onchange="redraw()">起终点</label>
                <label class="check"><input type="checkbox" id="layerCampus" onchange="toggleCampusBackground()">campus</label>
              </div>
            </div>
            <div class="panelBody plotBody">
              <div class="bgControls" id="bgControls">
                <label>透明度<input class="input" id="bgOpacity" type="number" min="0" max="1" step="0.05"></label>
                <label>缩放<input class="input" id="bgScale" type="number" min="0.2" max="5" step="0.05"></label>
                <label>X 偏移<input class="input" id="bgOffsetX" type="number" step="5"></label>
                <label>Y 偏移<input class="input" id="bgOffsetY" type="number" step="5"></label>
                <label>旋转<input class="input" id="bgRotation" type="number" min="-180" max="180" step="1"></label>
              </div>
              <div class="plotCanvasWrap"><canvas id="plot" width="1200" height="720"></canvas></div>
              <p id="plotHint" class="muted"></p>
            </div>
          </div>
          <div class="panel detailPanel">
            <div class="panelHead">
              <h2>任务详情</h2>
              <div class="actions" id="taskActions"></div>
            </div>
            <div class="tabs">
              <button class="tab active" onclick="setTab('overview')" id="tabOverview">概览</button>
              <button class="tab" onclick="setTab('timeline')" id="tabTimeline">时间线</button>
              <button class="tab" onclick="setTab('files')" id="tabFiles">文件</button>
              <button class="tab" onclick="setTab('raw')" id="tabRaw">JSON</button>
              <button class="tab" onclick="setTab('compare')" id="tabCompare">对比</button>
              <button class="tab" onclick="setTab('diagnostics')" id="tabDiagnostics">诊断</button>
            </div>
            <div class="panelBody" id="detail"><div class="empty">选择左侧任务查看详情</div></div>
          </div>
        </div>
      </main>
    </div>
  </div>
  <div id="toast" class="toast"></div>
  <script>
    const state = { view:'active', tasks:[], selected:null, selectedIds:new Set(), detail:null, traj:null, stats:null, health:null, settings:null, diagnostics:null, tab:'overview', timer:null, campusImage:null };
    const fmt = n => typeof n === 'number' && Number.isFinite(n) ? Number(n).toFixed(2).replace(/\.00$/,'') : '—';
    const cls = s => ['success','failed','timeout','cancelled'].includes(s) ? s : '';
    const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const qs = id => document.getElementById(id);
    async function getJSON(url, opts){ const r = await fetch(url, opts); if(!r.ok) throw new Error(await r.text()); return r.json(); }
    function rowId(t){ return state.view === 'trash' ? (t.trash_id || t.episode_id) : t.episode_id; }
    function params(){
      const p = new URLSearchParams();
      const q = qs('search').value.trim(), st = qs('statusFilter').value, ty = qs('typeFilter').value;
      if(q) p.set('q', q); if(st) p.set('status', st); if(ty) p.set('type', ty);
      p.set('sort', qs('sortBy').value); p.set('order', qs('sortOrder').value);
      if(state.view === 'archived') p.set('archived','true');
      if(state.view === 'trash') p.set('deleted','true');
      p.set('limit','300');
      return p;
    }
    async function refresh(){
      try {
        const [health, stats, tasks, settings, diagnostics] = await Promise.all([getJSON('/api/health'), getJSON('/api/stats'), getJSON('/api/tasks?'+params()), getJSON('/api/settings'), getJSON('/api/diagnostics')]);
        state.health = health; state.stats = stats; state.tasks = tasks.tasks; state.settings = settings; state.diagnostics = diagnostics;
        pruneSelection(); renderTop(); renderStats(stats); renderFilterOptions(stats); applyBackgroundSettings(settings); renderTasks(tasks); renderBatchActions();
        if(!state.selected && state.tasks[0]) await selectTask(rowId(state.tasks[0]));
        if(state.selected && !state.tasks.some(t => rowId(t) === state.selected)) clearSelection();
        if(state.tab === 'diagnostics' || state.tab === 'compare') renderDetail();
      } catch(err) { toast('刷新失败：' + err.message); }
    }
    function renderTop(){
      const h = state.health || {};
      qs('topStatus').innerHTML = `
        <span class="badge ${h.ros?.available ? 'success' : 'cancelled'}">${h.ros?.available ? 'ROS connected' : 'artifact only'}</span>
        <span class="badge">任务 ${h.task_count ?? 0}</span>
        <span class="badge">回收站 ${h.trash_count ?? 0}</span>
        <span class="badge ${h.campus_preview_available ? 'success' : 'cancelled'}">campus ${h.campus_preview_available ? 'ready' : 'missing'}</span>
        <span class="badge">${new Date((h.timestamp||Date.now()/1000)*1000).toLocaleTimeString()}</span>`;
    }
    function renderStats(s){
      const ros = state.health?.ros || {};
      const pose = ros.pose ? `${fmt(ros.pose.x)}, ${fmt(ros.pose.y)}` : '—';
      const items = [['任务数', s.total], ['成功率', Math.round((s.success_rate||0)*100)+'%'], ['成功', s.success], ['失败/中止', s.failed], ['平均误差', fmt(s.avg_goal_error_m)+'m'], ['当前位姿', pose]];
      qs('stats').innerHTML = items.map(([k,v])=>`<div class="stat"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
    }
    function renderFilterOptions(s){
      const curStatus = qs('statusFilter').value, curType = qs('typeFilter').value;
      const statuses = Object.keys(s.by_status||{}).sort();
      const types = Object.keys(s.by_type||{}).sort();
      qs('statusFilter').innerHTML = '<option value="">全部状态</option>' + statuses.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
      qs('typeFilter').innerHTML = '<option value="">全部类型</option>' + types.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
      qs('statusFilter').value = curStatus; qs('typeFilter').value = curType;
    }
    function taskQuality(t){
      const items = [];
      if(!t.has_odom) items.push('无 odom');
      if(!t.has_events) items.push('无事件');
      if(typeof t.distance_to_goal === 'number' && t.distance_to_goal > 2) items.push('误差高');
      return items;
    }
    function renderTasks(payload){
      qs('listCount').textContent = `${payload.total ?? state.tasks.length} 项`;
      qs('selectedCount').textContent = `已选 ${state.selectedIds.size}`;
      qs('taskList').innerHTML = state.tasks.length ? state.tasks.map(t => {
        const id = rowId(t);
        const tagHtml = (t.tags||[]).slice(0,4).map(tag=>`<span class="tag">${esc(tag)}</span>`).join('');
        const qualityHtml = taskQuality(t).map(x=>`<span class="tag quality">${esc(x)}</span>`).join('');
        return `<div class="task ${id===state.selected?'active':''}">
          <input class="taskCheck" type="checkbox" ${state.selectedIds.has(id)?'checked':''} onchange="toggleSelection('${esc(id)}', this.checked)">
          <div onclick="selectTask('${esc(id)}')">
            <div class="taskTop"><div class="taskTitle">${t.pinned?'★ ':''}${esc(t.instruction||t.episode_id)}</div><span class="badge ${cls(t.status)}">${esc(t.status)}</span></div>
            <div class="taskMeta"><span>${esc(t.type_label||t.intent_type||t.task_kind||'task')}</span><span>${fmt(t.duration_sec)}s</span><span>误差 ${fmt(t.distance_to_goal)}m</span></div>
            ${tagHtml || qualityHtml ? `<div class="taskTags">${tagHtml}${qualityHtml}</div>` : ''}
          </div>
        </div>`;
      }).join('') : '<div class="empty">没有匹配的任务</div>';
    }
    function renderBatchActions(){
      const count = state.selectedIds.size;
      const disabled = count ? '' : 'disabled';
      if(state.view === 'trash'){
        qs('batchActions').innerHTML = `<button class="btn small primary" ${disabled} onclick="batchRestore()">批量恢复</button><button class="btn small danger" ${disabled} onclick="batchPurge()">永久删除</button>`;
        return;
      }
      qs('batchActions').innerHTML = `<button class="btn small" ${disabled} onclick="batchArchive(true)">归档</button><button class="btn small" ${disabled} onclick="batchArchive(false)">取消归档</button><button class="btn small" ${disabled} onclick="batchAddTags()">加标签</button><button class="btn small" ${disabled} onclick="exportSelected()">导出所选</button><button class="btn small danger" ${disabled} onclick="batchDelete()">软删除</button>`;
    }
    async function selectTask(id){
      state.selected = id; renderTasks({total:state.tasks.length});
      try {
        const base = state.view === 'trash' ? '/api/trash/' : '/api/tasks/';
        const detailUrl = state.view === 'trash' ? base+encodeURIComponent(id)+'/detail' : base+encodeURIComponent(id);
        const [detail, traj] = await Promise.all([getJSON(detailUrl), getJSON(base+encodeURIComponent(id)+'/trajectory')]);
        state.detail = detail; state.traj = traj; renderActions(); renderDetail(); drawTrajectory(traj);
      } catch(err) { toast('读取任务失败：' + err.message); }
    }
    function clearSelection(){ state.selected=null; state.detail=null; state.traj=null; qs('detail').innerHTML='<div class="empty">选择左侧任务查看详情</div>'; qs('taskActions').innerHTML=''; redraw(); }
    function toggleSelection(id, checked){ checked ? state.selectedIds.add(id) : state.selectedIds.delete(id); renderBatchActions(); qs('selectedCount').textContent = `已选 ${state.selectedIds.size}`; if(state.tab==='compare') renderDetail(); }
    function selectVisible(){ state.tasks.forEach(t => state.selectedIds.add(rowId(t))); renderTasks({total:state.tasks.length}); renderBatchActions(); if(state.tab==='compare') renderDetail(); }
    function clearSelectionOnly(){ state.selectedIds.clear(); renderTasks({total:state.tasks.length}); renderBatchActions(); if(state.tab==='compare') renderDetail(); }
    function pruneSelection(){ const visible = new Set(state.tasks.map(rowId)); for(const id of Array.from(state.selectedIds)){ if(!visible.has(id)) state.selectedIds.delete(id); } }
    function selectedPayload(){ return state.view === 'trash' ? {trash_ids:Array.from(state.selectedIds)} : {episode_ids:Array.from(state.selectedIds)}; }
    async function batchArchive(archived){ await getJSON('/api/tasks/batch/metadata', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({episode_ids:Array.from(state.selectedIds), metadata:{archived}})}); toast('批量归档状态已更新'); await refresh(); }
    async function batchAddTags(){ const tags = prompt('输入要添加的标签，用逗号分隔'); if(tags === null) return; await getJSON('/api/tasks/batch/metadata', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({episode_ids:Array.from(state.selectedIds), metadata:{add_tags:tags}})}); toast('标签已添加'); await refresh(); }
    async function batchDelete(){ if(!state.selectedIds.size || !confirm('将所选任务移入回收站？')) return; await getJSON('/api/tasks/batch/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(selectedPayload())}); state.selectedIds.clear(); toast('已批量移入回收站'); await refresh(); }
    async function batchRestore(){ await getJSON('/api/trash/batch/restore', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(selectedPayload())}); state.selectedIds.clear(); toast('已批量恢复'); await refresh(); }
    async function batchPurge(){ if(!state.selectedIds.size || !confirm('永久删除所选回收站任务？此操作不可撤销。')) return; await getJSON('/api/trash/batch/purge', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(selectedPayload())}); state.selectedIds.clear(); toast('已批量永久删除'); await refresh(); }
    async function exportSelected(){
      const r = await fetch('/api/export/tasks/selected', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({episode_ids:Array.from(state.selectedIds)})});
      const blob = await r.blob(); downloadBlob(blob, 'cmu_selected_tasks.csv');
    }
    function renderActions(){
      const d = state.detail; if(!d){ qs('taskActions').innerHTML=''; return; }
      if(state.view === 'trash'){ qs('taskActions').innerHTML = `<button class="btn small primary" onclick="restoreTask('${esc(d.trash_id||'')}')">恢复</button><button class="btn small danger" onclick="purgeTask('${esc(d.trash_id||'')}')">永久删除</button>`; return; }
      qs('taskActions').innerHTML = `<button class="btn small" onclick="togglePin()">${d.pinned?'取消置顶':'置顶'}</button><button class="btn small" onclick="toggleArchive()">${d.archived?'取消归档':'归档'}</button><button class="btn small" onclick="downloadTask('json')">JSON</button><button class="btn small" onclick="downloadTask('csv')">CSV</button><button class="btn small danger" onclick="deleteTask()">删除</button>`;
    }
    function setTab(tab){ state.tab=tab; ['Overview','Timeline','Files','Raw','Compare','Diagnostics'].forEach(x=>qs('tab'+x).classList.toggle('active', tab===x.toLowerCase())); renderDetail(); }
    async function renderDetail(){
      const d = state.detail, t = state.traj;
      if(state.tab === 'compare'){ renderCompare(); return; }
      if(state.tab === 'diagnostics'){ renderDiagnostics(); return; }
      if(!d){ qs('detail').innerHTML='<div class="empty">选择左侧任务查看详情</div>'; return; }
      if(state.tab === 'timeline'){
        const ev = (d.events||[]).slice(-100).map(e=>`<li><strong>${esc(e.phase||'event')}</strong> <span>${esc(e.status||'')}</span><br>${esc(e.message||'')}</li>`).join('');
        qs('detail').innerHTML = `<ul class="timeline">${ev || '<li>无 task_events，显示旧 artifact。</li>'}</ul>`; return;
      }
      if(state.tab === 'files'){
        qs('detail').innerHTML = `<div class="fileList">${(d.files||[]).map(f=>`<div class="fileItem"><code>${esc(f)}</code><button class="btn small" onclick="openFile('${esc(f)}')">查看</button></div>`).join('') || '<div class="empty">无可查看文件</div>'}</div><pre id="fileViewer" style="margin-top:10px">选择文件查看内容</pre>`; return;
      }
      if(state.tab === 'raw'){ qs('detail').innerHTML = `<pre>${esc(JSON.stringify(d.summary,null,2))}</pre>`; return; }
      const quality = taskQuality(d).map(x=>`<span class="tag quality">${esc(x)}</span>`).join('');
      qs('detail').innerHTML = `<dl class="kv">
          <dt>状态</dt><dd><span class="badge ${cls(d.status)}">${esc(d.status)}</span> ${d.pinned?'<span class="badge pinned">置顶</span>':''} ${d.archived?'<span class="badge cancelled">已归档</span>':''} ${quality}</dd>
          <dt>指令</dt><dd><b>${esc(d.instruction||'')}</b></dd><dt>Episode</dt><dd><code>${esc(d.episode_id)}</code></dd>
          <dt>类型</dt><dd>${esc(d.type_label||'unknown')}</dd><dt>耗时</dt><dd>${fmt(d.duration_sec)} s</dd><dt>最终误差</dt><dd>${fmt(d.distance_to_goal)} m</dd>
          <dt>轨迹点</dt><dd>odom ${t?.counts?.odom ?? 0}，waypoint ${t?.counts?.waypoints ?? 0}，path ${t?.counts?.path_trace ?? 0}</dd>
          <dt>标签</dt><dd>${(d.tags||[]).map(tag=>`<span class="tag">${esc(tag)}</span>`).join('') || '<span class="muted">无</span>'}</dd>
          <dt>Artifact</dt><dd><code>${esc(d.artifact_dir)}</code></dd>
        </dl>
        <h2 style="font-size:14px;margin:16px 0 8px">管理信息</h2>
        <input id="tagInput" class="input" placeholder="标签，用逗号分隔" value="${esc((d.tags||[]).join(','))}">
        <textarea id="noteInput" style="margin-top:8px" placeholder="备注">${esc(d.note||'')}</textarea>
        <div class="actions" style="margin-top:8px"><button class="btn primary" onclick="saveMetadata()">保存标签和备注</button></div>`;
      qs('plotHint').textContent = `odom ${t?.counts?.odom ?? 0} 点，waypoint ${t?.counts?.waypoints ?? 0} 点，事件 ${t?.counts?.events ?? 0} 条；campus 预览背景未精确配准`;
    }
    async function renderCompare(){
      if(!state.selectedIds.size){ qs('detail').innerHTML='<div class="empty">勾选多个任务后查看对比</div>'; return; }
      const data = await getJSON('/api/tasks/compare', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({episode_ids:Array.from(state.selectedIds)})});
      qs('detail').innerHTML = `<table class="compareTable"><thead><tr><th>任务</th><th>状态</th><th>类型</th><th>耗时</th><th>误差</th><th>轨迹长度</th><th>点数</th></tr></thead><tbody>${data.tasks.map(t=>`<tr><td>${esc(t.instruction||t.episode_id)}<br><code>${esc(t.episode_id)}</code></td><td>${esc(t.status)}</td><td>${esc(t.type_label)}</td><td>${fmt(t.duration_sec)}s</td><td>${fmt(t.distance_to_goal)}m</td><td>${fmt(t.trajectory_length_m)}m</td><td>${t.odom_points}/${t.waypoints}/${t.path_points}</td></tr>`).join('')}</tbody></table>`;
    }
    function renderDiagnostics(){
      const issues = state.diagnostics?.issues || [];
      qs('detail').innerHTML = issues.length ? `<ul class="plainList">${issues.map(item=>`<li><b>${esc(item.instruction||item.episode_id)}</b><br><code>${esc(item.episode_id)}</code><br>${(item.issues||[]).map(x=>`<span class="tag quality">${esc(x)}</span>`).join('')}</li>`).join('')}</ul>` : '<div class="empty">当前筛选外的活跃任务没有明显数据质量问题</div>';
    }
    async function patchMetadata(patch){ const d = await getJSON('/api/tasks/'+encodeURIComponent(state.selected)+'/metadata', {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify(patch)}); state.detail = d; renderActions(); renderDetail(); await refresh(); toast('已保存'); }
    function saveMetadata(){ patchMetadata({tags:qs('tagInput').value, note:qs('noteInput').value}); }
    function togglePin(){ patchMetadata({pinned:!state.detail.pinned}); }
    function toggleArchive(){ patchMetadata({archived:!state.detail.archived}); }
    async function deleteTask(){ if(!state.selected || !confirm('将任务移入回收站？')) return; await getJSON('/api/tasks/'+encodeURIComponent(state.selected)+'/delete', {method:'POST'}); toast('已移入回收站'); state.selected=null; await refresh(); }
    async function restoreTask(id){ if(!id) return; await getJSON('/api/trash/'+encodeURIComponent(id)+'/restore', {method:'POST'}); toast('已恢复任务'); state.selected=null; await refresh(); }
    async function purgeTask(id){ if(!id || !confirm('永久删除回收站任务？此操作不可撤销。')) return; await getJSON('/api/trash/'+encodeURIComponent(id), {method:'DELETE'}); toast('已永久删除'); state.selected=null; await refresh(); }
    async function openFile(name){ const base = state.view === 'trash' ? '/api/trash/' : '/api/tasks/'; const r = await fetch(base+encodeURIComponent(state.selected)+'/files/'+encodeURIComponent(name)); qs('fileViewer').textContent = r.ok ? await r.text() : '读取失败'; }
    function downloadTask(format){ window.location = '/api/tasks/'+encodeURIComponent(state.selected)+'/export?format='+format; }
    function exportList(){ window.location = '/api/export/tasks?'+params(); }
    function setView(view){ state.view=view; state.selected=null; state.selectedIds.clear(); ['Active','Archived','Trash'].forEach(x=>qs('view'+x).classList.toggle('active', view===x.toLowerCase())); refresh(); }
    function applyBackgroundSettings(settings){
      const bg = settings?.plot_background || {};
      qs('layerCampus').checked = !!bg.enabled; qs('bgOpacity').value = bg.opacity ?? 0.3; qs('bgScale').value = bg.scale ?? 1; qs('bgOffsetX').value = bg.offset_x ?? 0; qs('bgOffsetY').value = bg.offset_y ?? 0; qs('bgRotation').value = bg.rotation_deg ?? 0;
      if(bg.enabled && !state.campusImage) loadCampusImage();
    }
    async function saveBackgroundSettings(){
      const payload = {plot_background:{enabled:qs('layerCampus').checked, opacity:Number(qs('bgOpacity').value), scale:Number(qs('bgScale').value), offset_x:Number(qs('bgOffsetX').value), offset_y:Number(qs('bgOffsetY').value), rotation_deg:Number(qs('bgRotation').value)}};
      state.settings = await getJSON('/api/settings', {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}); redraw();
    }
    function toggleCampusBackground(){ if(qs('layerCampus').checked && !state.campusImage) loadCampusImage(); saveBackgroundSettings(); }
    function loadCampusImage(){ const img = new Image(); img.onload = () => { state.campusImage = img; redraw(); }; img.src = '/api/worlds/campus/preview'; }
    function redraw(){ if(state.traj) drawTrajectory(state.traj); }
    function drawTrajectory(t){
      const c = qs('plot'), ctx = c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle='#07111f'; ctx.fillRect(0,0,c.width,c.height);
      if(!t || !t.bounds){ drawCampus(ctx, c, null); ctx.fillStyle='#cbd5e1'; ctx.fillText('无轨迹数据', 30, 42); return; }
      const b=t.bounds, pad=42, sx=(c.width-pad*2)/Math.max(1e-6,b.max_x-b.min_x), sy=(c.height-pad*2)/Math.max(1e-6,b.max_y-b.min_y), s=Math.min(sx,sy);
      const p = q => [pad+(Number(q.x)-b.min_x)*s, c.height-pad-(Number(q.y)-b.min_y)*s];
      drawCampus(ctx, c, b);
      ctx.strokeStyle='rgba(148,163,184,.18)'; ctx.lineWidth=1; for(let i=0;i<9;i++){ ctx.beginPath(); ctx.moveTo(pad,pad+i*(c.height-pad*2)/8); ctx.lineTo(c.width-pad,pad+i*(c.height-pad*2)/8); ctx.stroke(); ctx.beginPath(); ctx.moveTo(pad+i*(c.width-pad*2)/8,pad); ctx.lineTo(pad+i*(c.width-pad*2)/8,c.height-pad); ctx.stroke(); }
      if(qs('layerPath').checked) drawLine(ctx, t.path_trace||[], p, '#64748b', 2);
      if(qs('layerOdom').checked) drawLine(ctx, t.odom||[], p, '#38bdf8', 3);
      if(qs('layerWaypoints').checked) drawPoints(ctx, t.waypoints||[], p, '#facc15', 5);
      if(qs('layerEndpoints').checked && (t.odom||[]).length){ dot(ctx, p(t.odom[0]), '#22c55e', 7); dot(ctx, p(t.odom[t.odom.length-1]), '#fb7185', 7); }
    }
    function drawCampus(ctx, c){
      if(!qs('layerCampus').checked || !state.campusImage) return;
      const bg = state.settings?.plot_background || {};
      const opacity = Number(qs('bgOpacity').value || bg.opacity || .3), scale = Number(qs('bgScale').value || bg.scale || 1), ox = Number(qs('bgOffsetX').value || bg.offset_x || 0), oy = Number(qs('bgOffsetY').value || bg.offset_y || 0), rot = Number(qs('bgRotation').value || bg.rotation_deg || 0) * Math.PI / 180;
      const iw = state.campusImage.width, ih = state.campusImage.height, base = Math.min(c.width/iw, c.height/ih) * scale;
      ctx.save(); ctx.globalAlpha = opacity; ctx.translate(c.width/2 + ox, c.height/2 + oy); ctx.rotate(rot); ctx.drawImage(state.campusImage, -iw*base/2, -ih*base/2, iw*base, ih*base); ctx.restore();
    }
    function drawLine(ctx, pts, p, color, width){ if(pts.length<2) return; ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); pts.forEach((pt,i)=>{ const q=p(pt); i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]); }); ctx.stroke(); }
    function drawPoints(ctx, pts, p, color, r){ pts.forEach(pt=>dot(ctx,p(pt),color,r)); }
    function dot(ctx, q, color, r){ ctx.fillStyle=color; ctx.beginPath(); ctx.arc(q[0],q[1],r,0,Math.PI*2); ctx.fill(); }
    function downloadBlob(blob, name){ const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href), 1000); }
    function toast(msg){ const el=qs('toast'); el.textContent=msg; el.style.display='block'; clearTimeout(state.timer); state.timer=setTimeout(()=>el.style.display='none',3200); }
    ['search','statusFilter','typeFilter','sortBy','sortOrder'].forEach(id=>qs(id).addEventListener('input',()=>refresh()));
    ['bgOpacity','bgScale','bgOffsetX','bgOffsetY','bgRotation'].forEach(id=>qs(id).addEventListener('change',()=>saveBackgroundSettings()));
    refresh(); setInterval(refresh, 7000);
  </script>
</body>
</html>"""
