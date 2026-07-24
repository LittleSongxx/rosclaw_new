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
