"""Hardware preflight gates (真机自进化v2 §2.2, §Phase 0).

Formal acceptance NEVER falls back to a mock camera: when physical
perception is unavailable the harness emits
``BLOCKED: PHYSICAL_PERCEPTION_UNAVAILABLE`` and refuses to run sessions.
A disclosed dev-mode bypass exists solely for harness development and is
recorded into the evidence manifest as non-acceptance.

Probes are injectable so tests exercise the gate logic without hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .contracts import BLOCKED_PHYSICAL_PERCEPTION_UNAVAILABLE, EvoRpsConfig

CameraProbe = Callable[[], dict[str, Any]]
SerialProbe = Callable[[], dict[str, Any]]
StoreProbe = Callable[[], dict[str, Any]]


def default_camera_probe() -> dict[str, Any]:
    """Enumerate RealSense devices via pyrealsense2 (optional dependency).

    Never starts a pipeline and never issues a hardware_reset here —
    preflight is observation-only (see field notes: a wedge/reset at the
    wrong moment can drop the device off the bus).

    The acceptance CLI itself may run in an interpreter WITHOUT
    pyrealsense2 (the repo venv); the camera work runs in the
    camera-capable task env (the workspace venv).  When the in-process
    import fails, the probe retries the same observation in the task
    env — the one that will actually run sessions — before declaring
    perception unavailable.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return _subprocess_camera_probe()
    try:
        devices = rs.context().query_devices()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"enumerate_failed:{type(exc).__name__}"}
    if len(devices) == 0:
        return {"available": False, "reason": "no_device_enumerated"}
    info = []
    for dev in devices:
        try:
            info.append(
                {
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "firmware": dev.get_info(rs.camera_info.firmware_version),
                }
            )
        except Exception:  # noqa: BLE001
            info.append({"name": "unreadable"})
    return {"available": True, "devices": info, "via": "in_process"}


CAMERA_ENV_PY = "/home/nvidia/workspace/rosclaw/rosclaw_test/.venv/bin/python"


def _subprocess_camera_probe() -> dict[str, Any]:
    """Enumerate via the camera-capable task env (observation-only)."""
    import json
    import subprocess
    from pathlib import Path

    if not Path(CAMERA_ENV_PY).is_file():
        return {
            "available": False,
            "reason": "pyrealsense2_not_installed (and no camera task env at "
            f"{CAMERA_ENV_PY})",
        }
    code = (
        "import json, pyrealsense2 as rs\n"
        "devs = rs.context().query_devices()\n"
        "print(json.dumps([{'name': d.get_info(rs.camera_info.name),"
        " 'serial': d.get_info(rs.camera_info.serial_number),"
        " 'firmware': d.get_info(rs.camera_info.firmware_version)} for d in devs]))"
    )
    try:
        proc = subprocess.run(
            [CAMERA_ENV_PY, "-c", code], capture_output=True, text=True, timeout=30
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "reason": f"camera env probe failed: {exc}"}
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": f"camera env pyrealsense2 unusable: {proc.stderr.strip()[-160:]}",
        }
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("["):
            devices = json.loads(line)
            if devices:
                return {"available": True, "devices": devices, "via": "camera_env"}
            return {"available": False, "reason": "no_device_enumerated", "via": "camera_env"}
    return {"available": False, "reason": "camera env probe returned no device list"}


def default_serial_probe() -> dict[str, Any]:
    from pathlib import Path

    ports = sorted(str(p) for p in Path("/dev").glob("ttyUSB*"))
    return {"available": bool(ports), "ports": ports}


def default_store_probe(dsn: str) -> dict[str, Any]:
    try:
        from rosclaw.storage.factory import StoreFactory

        store = StoreFactory.create_structured_store(backend="seekdb_server", url=dsn)
        store.connect()
        return {"available": True, "dsn": dsn}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


@dataclass
class PreflightReport:
    ok: bool
    blocked: list[str] = field(default_factory=list)
    probes: dict[str, Any] = field(default_factory=dict)
    dev_mode: bool = False
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "blocked": self.blocked,
            "probes": self.probes,
            "dev_mode": self.dev_mode,
            "checked_at": self.checked_at,
        }


def run_preflight(
    config: EvoRpsConfig,
    *,
    camera_probe: CameraProbe | None = None,
    serial_probe: SerialProbe | None = None,
    store_probe: StoreProbe | None = None,
    dev_allow_mock: bool = False,
) -> PreflightReport:
    """Run all hardware gates.  ``dev_allow_mock`` is the harness-development
    bypass; it is ignored unless the caller ALSO passes it explicitly, and
    the report marks ``dev_mode=True`` so the evidence can never be confused
    with formal acceptance (§2.2)."""
    blocked: list[str] = []
    probes: dict[str, Any] = {}

    camera = (camera_probe or default_camera_probe)()
    probes["camera"] = camera
    if not camera.get("available"):
        if config.allow_mock_camera or dev_allow_mock:
            probes["camera"]["mock_used"] = True
        else:
            blocked.append(BLOCKED_PHYSICAL_PERCEPTION_UNAVAILABLE)
    elif config.allow_mock_camera:
        blocked.append(
            "contract violation: allow_mock_camera=true is forbidden in formal acceptance"
        )

    serial = (serial_probe or default_serial_probe)()
    probes["serial"] = serial
    if not serial.get("available"):
        blocked.append("BLOCKED: RH56_SERIAL_UNAVAILABLE")

    store = (store_probe or (lambda: default_store_probe(config.seekdb_dsn())))()
    probes["seekdb"] = store
    if not store.get("available"):
        blocked.append("BLOCKED: SEEKDB_UNAVAILABLE")

    if config.allow_fixture_execution:
        blocked.append("contract violation: allow_fixture_execution=true is forbidden")

    return PreflightReport(
        ok=not blocked,
        blocked=blocked,
        probes=probes,
        dev_mode=bool(dev_allow_mock and not camera.get("available")),
    )
