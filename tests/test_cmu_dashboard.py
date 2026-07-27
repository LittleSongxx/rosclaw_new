import json
from pathlib import Path

import pytest

from rosclaw.apps.cmu_dashboard import CmuDashboardStore, _sample_points, create_cmu_dashboard_app


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_task(output: Path, episode_id: str, **summary):
    task_dir = output / episode_id
    task_dir.mkdir(parents=True)
    payload = {
        "episode_id": episode_id,
        "instruction": episode_id,
        "status": "success",
        "duration_sec": 1.0,
        "distance_to_goal": 0.1,
        "intent": {"type": "relative"},
    }
    payload.update(summary)
    (task_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    return task_dir


def test_cmu_dashboard_scans_new_and_legacy_artifacts(tmp_path: Path):
    output = tmp_path / "app_runs"
    new_task = output / "app_cmu_chat_task"
    old_task = output / "app_cmu_go_1"
    new_task.mkdir(parents=True)
    old_task.mkdir(parents=True)
    (new_task / "summary.json").write_text(
        json.dumps(
            {
                "episode_id": "app_cmu_chat_task",
                "instruction": "以半径2米转一圈",
                "status": "success",
                "duration_sec": 2.0,
                "distance_to_goal": 0.2,
                "intent": {"type": "absolute", "x": 1.0, "y": 2.0},
                "task": {"kind": "sequence"},
            }
        ),
        encoding="utf-8",
    )
    (new_task / "task_events.jsonl").write_text('{"phase":"start","message":"开始"}\n', encoding="utf-8")
    (old_task / "summary.json").write_text(
        json.dumps(
            {
                "episode_id": "app_cmu_go_1",
                "instruction": "向上走3米",
                "status": "timeout",
                "duration_sec": 5.0,
                "intent": {"type": "relative"},
            }
        ),
        encoding="utf-8",
    )

    store = CmuDashboardStore(output)
    tasks = store.list_tasks()
    stats = store.stats()

    assert {task["episode_id"] for task in tasks} == {"app_cmu_chat_task", "app_cmu_go_1"}
    assert stats["total"] == 2
    assert stats["success"] == 1
    assert stats["failed"] == 1
    assert stats["by_type"]["absolute"] == 1
    assert stats["by_type"]["relative"] == 1


def test_cmu_dashboard_handles_missing_empty_and_bad_jsonl(tmp_path: Path):
    task_dir = tmp_path / "app_runs" / "app_cmu_chat_bad"
    task_dir.mkdir(parents=True)
    (task_dir / "summary.json").write_text(
        json.dumps({"episode_id": "app_cmu_chat_bad", "status": "success", "intent": {"type": "place"}}),
        encoding="utf-8",
    )
    (task_dir / "odom_trace.jsonl").write_text('{"x":0,"y":0}\nnot-json\n{"x":1,"y":1}\n', encoding="utf-8")
    (task_dir / "waypoints.jsonl").write_text("", encoding="utf-8")

    store = CmuDashboardStore(tmp_path / "app_runs", max_points=10)
    detail = store.get_task("app_cmu_chat_bad")
    trajectory = store.trajectory("app_cmu_chat_bad")

    assert detail["events"] == []
    assert trajectory["counts"]["odom"] == 2
    assert trajectory["counts"]["waypoints"] == 0
    assert trajectory["bounds"] == {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0}


def test_cmu_dashboard_sampling_preserves_first_and_last():
    items = [{"x": i, "y": i * 2} for i in range(100)]

    sampled = _sample_points(items, 10)

    assert len(sampled) == 10
    assert sampled[0] == items[0]
    assert sampled[-1] == items[-1]


def test_cmu_dashboard_missing_task_raises(tmp_path: Path):
    store = CmuDashboardStore(tmp_path / "app_runs")

    with pytest.raises(FileNotFoundError):
        store.get_task("missing")


def test_cmu_dashboard_filters_sorts_and_paginates(tmp_path: Path):
    output = tmp_path / "app_runs"
    _write_task(output, "app_cmu_a", instruction="向上走3米", status="success", duration_sec=3, intent={"type": "relative"})
    _write_task(output, "app_cmu_b", instruction="开始探索", status="timeout", duration_sec=9, intent={"type": "explore_control"})
    store = CmuDashboardStore(output)

    assert store.query_tasks(q="探索")["tasks"][0]["episode_id"] == "app_cmu_b"
    assert store.query_tasks(status="timeout")["total"] == 1
    assert store.query_tasks(task_type="relative")["total"] == 1
    assert [task["episode_id"] for task in store.query_tasks(sort="duration", order="asc")["tasks"]] == [
        "app_cmu_a",
        "app_cmu_b",
    ]
    page = store.query_tasks(sort="duration", order="asc", limit=1, offset=1)
    assert page["count"] == 1
    assert page["total"] == 2
    assert page["tasks"][0]["episode_id"] == "app_cmu_b"


def test_cmu_dashboard_metadata_persists_and_archives(tmp_path: Path):
    output = tmp_path / "app_runs"
    _write_task(output, "app_cmu_tagged")
    store = CmuDashboardStore(output)

    detail = store.update_metadata(
        "app_cmu_tagged",
        {"pinned": True, "archived": True, "tags": "review, tunnel,review", "note": "needs rerun"},
    )

    assert detail["pinned"] is True
    assert detail["archived"] is True
    assert detail["tags"] == ["review", "tunnel"]
    assert detail["note"] == "needs rerun"
    assert store.query_tasks()["total"] == 0
    archived = CmuDashboardStore(output).query_tasks(archived=True)
    assert archived["total"] == 1
    assert archived["tasks"][0]["tags"] == ["review", "tunnel"]
    assert archived["tasks"][0]["note"] == "needs rerun"


def test_cmu_dashboard_soft_delete_restore_and_purge(tmp_path: Path):
    output = tmp_path / "app_runs"
    _write_task(output, "app_cmu_delete_me")
    store = CmuDashboardStore(output)

    deleted = store.soft_delete("app_cmu_delete_me")
    trash_id = deleted["trash_id"]

    assert not (output / "app_cmu_delete_me").exists()
    assert store.query_tasks()["total"] == 0
    trash = store.query_tasks(deleted=True)
    assert trash["total"] == 1
    assert trash["tasks"][0]["trash_id"] == trash_id
    assert store.get_trash_task(trash_id)["episode_id"] == "app_cmu_delete_me"

    restored = store.restore_trash(trash_id)
    assert restored["episode_id"] == "app_cmu_delete_me"
    assert (output / "app_cmu_delete_me").exists()

    deleted_again = store.soft_delete("app_cmu_delete_me")
    store.delete_trash(deleted_again["trash_id"])
    assert store.query_tasks(deleted=True)["total"] == 0
    with pytest.raises(FileNotFoundError):
        store.delete_trash("../app_cmu_delete_me")


def test_cmu_dashboard_exports_and_safe_file_access(tmp_path: Path):
    output = tmp_path / "app_runs"
    task_dir = _write_task(output, "app_cmu_export", instruction="向右走5米")
    _write_jsonl(task_dir / "odom_trace.jsonl", [{"x": 0, "y": 0}, {"x": 1, "y": 0}])
    (task_dir / "unsafe.txt").write_text("hidden", encoding="utf-8")
    store = CmuDashboardStore(output)

    json_body, json_type, _ = store.export_task("app_cmu_export", format="json")
    csv_body, csv_type, _ = store.export_task("app_cmu_export", format="csv")

    assert "向右走5米" in json_body
    assert json_type == "application/json"
    assert "episode_id" in csv_body
    assert csv_type.startswith("text/csv")
    assert "summary.json" in store.get_task("app_cmu_export")["files"]
    assert "unsafe.txt" not in store.get_task("app_cmu_export")["files"]
    with pytest.raises(FileNotFoundError):
        store.read_file("app_cmu_export", "unsafe.txt")


def test_cmu_dashboard_batch_management_and_selected_export(tmp_path: Path):
    output = tmp_path / "app_runs"
    _write_task(output, "app_cmu_batch_a", instruction="a", intent={"type": "relative"})
    _write_task(output, "app_cmu_batch_b", instruction="b", intent={"type": "absolute"})
    store = CmuDashboardStore(output)

    updated = store.batch_update_metadata(
        ["app_cmu_batch_a", "app_cmu_batch_b", "../bad"],
        {"archived": True, "add_tags": "batch,review"},
    )

    assert updated["count"] == 2
    assert store.query_tasks()["total"] == 0
    archived = store.query_tasks(archived=True)
    assert archived["total"] == 2
    assert all(task["tags"] == ["batch", "review"] for task in archived["tasks"])

    store.batch_update_metadata(["app_cmu_batch_a", "app_cmu_batch_b"], {"archived": False})
    csv_body = store.export_selected_csv(["app_cmu_batch_a"])
    assert "app_cmu_batch_a" in csv_body
    assert "app_cmu_batch_b" not in csv_body

    deleted = store.batch_soft_delete(["app_cmu_batch_a", "missing"])
    assert deleted["count"] == 1
    assert deleted["missing"] == ["missing"]
    trash_id = deleted["deleted"][0]["trash_id"]
    assert store.batch_restore_trash([trash_id])["count"] == 1
    deleted_again = store.batch_soft_delete(["app_cmu_batch_a"])
    assert store.batch_delete_trash([deleted_again["deleted"][0]["trash_id"], "../bad"])["count"] == 1


def test_cmu_dashboard_settings_diagnostics_and_compare(tmp_path: Path):
    output = tmp_path / "app_runs"
    good_dir = _write_task(output, "app_cmu_good", instruction="good", status="success", distance_to_goal=0.2)
    bad_dir = _write_task(output, "app_cmu_bad", instruction="bad", status="timeout", distance_to_goal=3.5)
    _write_jsonl(good_dir / "odom_trace.jsonl", [{"x": 0, "y": 0}, {"x": 3, "y": 4}])
    store = CmuDashboardStore(output)

    settings = store.update_dashboard_settings(
        {"plot_background": {"enabled": True, "opacity": 2, "scale": 0.1, "offset_x": 20, "rotation_deg": 300}}
    )
    background = settings["plot_background"]

    assert background["enabled"] is True
    assert background["opacity"] == 1.0
    assert background["scale"] == 0.2
    assert background["offset_x"] == 20.0
    assert background["rotation_deg"] == 180.0
    assert store.diagnostics()["issue_count"] >= 1
    compare = store.compare_tasks(["app_cmu_good", "missing"])
    assert compare["count"] == 1
    assert compare["tasks"][0]["trajectory_length_m"] == 5.0


def test_cmu_dashboard_api_routes_when_fastapi_available(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from rosclaw.apps.cmu_dashboard import CmuDashboardConfig

    task_dir = tmp_path / "app_runs" / "app_cmu_go_1"
    task_dir.mkdir(parents=True)
    (task_dir / "summary.json").write_text(
        json.dumps({"episode_id": "app_cmu_go_1", "status": "success", "intent": {"type": "relative"}}),
        encoding="utf-8",
    )
    _write_jsonl(task_dir / "odom_trace.jsonl", [{"x": 0, "y": 0}, {"x": 1, "y": 1}])

    app = create_cmu_dashboard_app(
        CmuDashboardConfig(output_dir=tmp_path / "app_runs", connect_ros=False)
    )
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/tasks").json()["count"] == 1
    assert client.get("/api/tasks/app_cmu_go_1").status_code == 200
    assert client.get("/api/tasks/app_cmu_go_1/trajectory").json()["counts"]["odom"] == 2
    assert client.get("/api/tasks/missing").status_code == 404


def test_cmu_dashboard_management_api_routes_when_fastapi_available(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from rosclaw.apps.cmu_dashboard import CmuDashboardConfig

    output = tmp_path / "app_runs"
    _write_task(output, "app_cmu_api", instruction="开始探索", status="success", intent={"type": "explore_control"})
    app = create_cmu_dashboard_app(CmuDashboardConfig(output_dir=output, connect_ros=False))
    client = TestClient(app)

    patched = client.patch(
        "/api/tasks/app_cmu_api/metadata",
        json={"tags": ["demo"], "note": "good run", "pinned": True},
    )
    assert patched.status_code == 200
    assert patched.json()["tags"] == ["demo"]
    assert client.get("/api/tasks?q=good").json()["total"] == 1
    assert "app_cmu_api" in client.get("/api/export/tasks").text
    assert "app_cmu_api" in client.get("/api/tasks/app_cmu_api/export?format=csv").text

    deleted = client.post("/api/tasks/app_cmu_api/delete")
    assert deleted.status_code == 200
    trash_id = deleted.json()["trash_id"]
    assert client.get(f"/api/trash/{trash_id}/detail").status_code == 200
    assert client.get(f"/api/trash/{trash_id}/trajectory").status_code == 200
    assert client.post(f"/api/trash/{trash_id}/restore").status_code == 200
    assert client.get("/api/tasks").json()["total"] == 1


def test_cmu_dashboard_batch_and_background_api_routes_when_fastapi_available(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from rosclaw.apps.cmu_dashboard import CmuDashboardConfig

    output = tmp_path / "app_runs"
    _write_task(output, "app_cmu_api_batch_a", instruction="a")
    _write_task(output, "app_cmu_api_batch_b", instruction="b")
    app = create_cmu_dashboard_app(CmuDashboardConfig(output_dir=output, connect_ros=False))
    client = TestClient(app)

    assert client.post(
        "/api/tasks/batch/metadata",
        json={"episode_ids": ["app_cmu_api_batch_a", "app_cmu_api_batch_b"], "metadata": {"add_tags": "api"}},
    ).json()["count"] == 2
    assert "app_cmu_api_batch_a" in client.post(
        "/api/export/tasks/selected",
        json={"episode_ids": ["app_cmu_api_batch_a"]},
    ).text
    assert client.post(
        "/api/tasks/compare",
        json={"episode_ids": ["app_cmu_api_batch_a", "app_cmu_api_batch_b"]},
    ).json()["count"] == 2
    assert client.get("/api/diagnostics").status_code == 200
    patched_settings = client.patch(
        "/api/settings",
        json={"plot_background": {"enabled": True, "opacity": 0.4}},
    )
    assert patched_settings.status_code == 200
    assert patched_settings.json()["plot_background"]["enabled"] is True
    assert client.get("/api/worlds/campus/preview").status_code in {200, 404}

    deleted = client.post(
        "/api/tasks/batch/delete",
        json={"episode_ids": ["app_cmu_api_batch_a"]},
    ).json()
    trash_id = deleted["deleted"][0]["trash_id"]
    assert client.post("/api/trash/batch/restore", json={"trash_ids": [trash_id]}).json()["count"] == 1
