from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rosclaw.integrations.cmu_are.executor import CMU_STOP_CAPABILITY
from rosclaw.sim.assets import verify_assets
from rosclaw.sim.cli import _make_action, dispatch_sim_argv


def test_cmu_are_help_and_contract_json(capsys) -> None:
    assert dispatch_sim_argv(["sim", "cmu-are", "--help"]) == 0
    output = capsys.readouterr().out
    assert "navigate" in output

    assert dispatch_sim_argv(["sim", "cmu-are", "check", "--contract-only", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["launchable"] is False


def test_asset_verifier_recognizes_directory_mounts(tmp_path: Path) -> None:
    manifest = tmp_path / "assets.yaml"
    (tmp_path / "mesh").mkdir()
    manifest.write_text(
        """
schema_version: rosclaw.cmu_are.assets.v1
assets:
  - asset_id: mesh
    relative_path: mesh/
    source: fixture
    upstream_version_or_commit: test
    size_bytes: null
    sha256: null
    license: test
    mount_path: /opt/mesh/
    required_for: [gazebo_world]
""",
        encoding="utf-8",
    )
    report = verify_assets(project_root=tmp_path, manifest_path=manifest)
    assert report["ok"] is True
    assert report["assets"][0]["status"] == "ok"


def test_asset_verifier_supports_external_third_party_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "assets"
    project.mkdir()
    external.mkdir()
    payload = b"approved-cmu-asset"
    (external / "model.bin").write_bytes(payload)
    manifest = project / "assets.yaml"
    manifest.write_text(
        f"""
schema_version: rosclaw.cmu_are.assets.v1
assets:
  - asset_id: model
    relative_path: third_party/model.bin
    source: fixture
    upstream_version_or_commit: test
    size_bytes: {len(payload)}
    sha256: {hashlib.sha256(payload).hexdigest()}
    license: test
    mount_path: /opt/model.bin
    required_for: [ariadne2_exploration]
""",
        encoding="utf-8",
    )
    report = verify_assets(
        project_root=project,
        manifest_path=manifest,
        asset_root=external,
    )
    assert report["ok"] is True
    assert report["asset_root"] == str(external.resolve())


def test_cmu_action_has_a_gateway_reusable_trace_id() -> None:
    action = _make_action(
        capability=CMU_STOP_CAPABILITY,
        arguments={"schema_version": "cmu_are.stop.v1", "timeout_sec": 1.0},
        timeout_sec=1.0,
    )
    assert action.parent_trace_id is not None
    assert action.parent_trace_id.startswith("trace_cmu_are_")


def test_stop_timeout_fails_closed_as_json(capsys) -> None:
    assert dispatch_sim_argv(["sim", "cmu-are", "stop", "--timeout", "nan", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "CMU_ARE_COMMAND_FAILED"


def test_compose_mounts_assets_at_ros_package_runtime_paths() -> None:
    root = Path(__file__).parents[2]
    compose = (root / "docker/ros1/docker-compose.ros1-are.yml").read_text(encoding="utf-8")
    for target in (
        "/opt/rosclaw/ros1_ws/src/ariadne2/scripts/model/checkpoint.pth:ro",
        "/opt/rosclaw/ros1_ws/src/local_planner/paths/correspondences.txt:ro",
        "/opt/rosclaw/ros1_ws/src/local_planner/paths/paths.ply:ro",
        "/opt/rosclaw/ros1_ws/src/local_planner/paths/startPaths.ply:ro",
        "/opt/rosclaw/ros1_ws/src/local_planner/paths/pathList.ply:ro",
        "/opt/rosclaw/ros1_ws/src/vehicle_simulator/mesh:ro",
    ):
        assert target in compose
