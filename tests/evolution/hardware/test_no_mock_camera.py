"""No-mock-camera gate tests (PR-EVO-HW-1, 真机自进化v2 §2.2).

Formal acceptance NEVER falls back to a mock camera: physical perception
unavailable → BLOCKED, and no "all modules passed" claim may be produced.
"""

from __future__ import annotations

from rosclaw.evolution.hardware.contracts import (
    BLOCKED_PHYSICAL_PERCEPTION_UNAVAILABLE,
    load_config,
)
from rosclaw.evolution.hardware.preflight import run_preflight

CONFIG_PATH = "configs/acceptance/evo_rps_v1.yaml"

NO_CAMERA = {"available": False, "reason": "no_device_enumerated"}
CAMERA_OK = {"available": True, "devices": [{"name": "D435I", "serial": "x"}]}
SERIAL_OK = {"available": True, "ports": ["/dev/ttyUSB0", "/dev/ttyUSB1"]}
STORE_OK = {"available": True}


def _config():
    return load_config(CONFIG_PATH)


def test_camera_absent_blocks_formal_acceptance() -> None:
    report = run_preflight(
        _config(),
        camera_probe=lambda: NO_CAMERA,
        serial_probe=lambda: SERIAL_OK,
        store_probe=lambda: STORE_OK,
    )
    assert report.ok is False
    assert BLOCKED_PHYSICAL_PERCEPTION_UNAVAILABLE in report.blocked
    assert report.dev_mode is False
    assert not report.probes["camera"].get("mock_used")


def test_dev_bypass_is_disclosed_and_never_formal() -> None:
    report = run_preflight(
        _config(),
        camera_probe=lambda: NO_CAMERA,
        serial_probe=lambda: SERIAL_OK,
        store_probe=lambda: STORE_OK,
        dev_allow_mock=True,
    )
    assert report.ok is True  # harness development only
    assert report.dev_mode is True
    assert report.probes["camera"]["mock_used"] is True


def test_camera_present_passes_with_serial_and_store() -> None:
    report = run_preflight(
        _config(),
        camera_probe=lambda: CAMERA_OK,
        serial_probe=lambda: SERIAL_OK,
        store_probe=lambda: STORE_OK,
    )
    assert report.ok is True
    assert report.blocked == []
    assert report.dev_mode is False


def test_mock_allowed_in_config_is_a_contract_violation(tmp_path) -> None:
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as _fh:
        raw = yaml.safe_load(_fh)
    raw["experiment"]["allow_mock_camera"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    config = load_config(str(path))
    report = run_preflight(
        config,
        camera_probe=lambda: CAMERA_OK,
        serial_probe=lambda: SERIAL_OK,
        store_probe=lambda: STORE_OK,
    )
    assert report.ok is False
    assert any("allow_mock_camera" in b for b in report.blocked)


def test_serial_absent_blocks() -> None:
    report = run_preflight(
        _config(),
        camera_probe=lambda: CAMERA_OK,
        serial_probe=lambda: {"available": False, "ports": []},
        store_probe=lambda: STORE_OK,
    )
    assert "BLOCKED: RH56_SERIAL_UNAVAILABLE" in report.blocked


def test_camera_env_subprocess_probe_used_when_in_process_missing(monkeypatch) -> None:
    """The acceptance CLI may run in an interpreter without pyrealsense2;
    the probe must fall back to the camera-capable task env before ever
    declaring perception unavailable (live finding 2026-07-25)."""
    import json
    import subprocess as sp

    import rosclaw.evolution.hardware.preflight as pf

    monkeypatch.setitem(__import__("sys").modules, "pyrealsense2", None)
    completed = sp.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps([{"name": "D435I", "serial": "x", "firmware": "f"}]) + "\n",
        stderr="",
    )
    monkeypatch.setattr(sp, "run", lambda *a, **k: completed)
    probe = pf._subprocess_camera_probe()
    assert probe["available"] is True
    assert probe["via"] == "camera_env"
    assert probe["devices"][0]["serial"] == "x"


def test_camera_env_probe_honest_when_env_missing(monkeypatch) -> None:
    import rosclaw.evolution.hardware.preflight as pf

    monkeypatch.setattr(pf, "CAMERA_ENV_PY", "/nonexistent/python")
    probe = pf._subprocess_camera_probe()
    assert probe["available"] is False
    assert "pyrealsense2_not_installed" in probe["reason"]
