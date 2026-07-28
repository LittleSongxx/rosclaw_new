"""Thermal window gate for matched-regime canary starts (§Phase 6).

The protocol demands 相同初始温度区间: every canary slot must start in
the same temperature band.  Before each slot the gate reads the hand
temperatures; above ``start_max_temp_c`` it waits (recorded) until the
hands cool into the window, or — past ``max_wait_s`` — blocks the matrix
honestly rather than running a mismatched experiment.

The probe runs in the camera/hand-capable task env (the repo venv has no
pyserial), mirroring the camera probe pattern.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TASK_ENV_PY = "/home/nvidia/workspace/rosclaw/rosclaw_test/.venv/bin/python"
RPS_ROOT = "/home/nvidia/workspace/rosclaw/rosclaw_test/examples/rh56_rps"
RH56_SRC = "/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src"

_PROBE_CODE = (
    "import json\n"
    "from rosclaw_rps.hand.rh56_controller import build_hand_controller\n"
    "out = {}\n"
    "for cfg, side in (({'controller': 'rh56', 'port': '/dev/ttyUSB0', 'device_id': 2}, 'right'),"
    " ({'controller': 'rh56', 'port': '/dev/ttyUSB1', 'device_id': 1}, 'left')):\n"
    "    hand = build_hand_controller(cfg)\n"
    "    try:\n"
    "        tel = hand.read_telemetry()\n"
    "        vals = [v for v in (tel.temperature_c or {}).values() if isinstance(v, (int, float)) and v > 0]\n"
    "        out[side] = max(vals) if vals else None\n"
    "    except Exception:\n"
    "        out[side] = None\n"
    "    finally:\n"
    "        try: hand.close()\n"
    "        except Exception: pass\n"
    "print(json.dumps(out))\n"
)

TempProbe = Callable[[], dict[str, float | None]]


def default_temp_probe() -> dict[str, float | None]:
    import os

    env = {
        **os.environ,
        "PYTHONPATH": f"{RPS_ROOT}/scripts:{RPS_ROOT}/src:{RH56_SRC}:/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src",
    }
    proc = subprocess.run(
        [TASK_ENV_PY, "-c", _PROBE_CODE],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
        cwd=RPS_ROOT,
    )
    if proc.returncode != 0:
        return {"right": None, "left": None, "error": proc.stderr.strip()[-160:]}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"right": None, "left": None, "error": "probe returned no temps"}


@dataclass
class ThermalWindowResult:
    ok: bool
    waited_s: float = 0.0
    temps: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def wait_for_thermal_window(
    *,
    probe: TempProbe = default_temp_probe,
    start_max_temp_c: float = 46.0,
    max_wait_s: float = 900.0,
    poll_s: float = 60.0,
) -> ThermalWindowResult:
    """Wait until both hands read at or below start_max_temp_c (missing
    readings never count as cool — unknown is not cold)."""
    started = time.monotonic()
    while True:
        temps = probe()
        values = [v for k, v in temps.items() if k in ("right", "left") and v is not None]
        if values and max(values) <= start_max_temp_c:
            return ThermalWindowResult(
                ok=True,
                waited_s=time.monotonic() - started,
                temps=temps,
                reason=f"in window (max {max(values):.0f}°C ≤ {start_max_temp_c}°C)",
            )
        waited = time.monotonic() - started
        if waited >= max_wait_s:
            return ThermalWindowResult(
                ok=False,
                waited_s=waited,
                temps=temps,
                reason=(
                    f"thermal window not reached within {max_wait_s:.0f}s "
                    f"(last temps {temps})"
                ),
            )
        time.sleep(poll_s)
