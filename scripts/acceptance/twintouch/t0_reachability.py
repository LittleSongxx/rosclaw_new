#!/usr/bin/env python3
"""TwinTouch T0 reachability probe (v4 §11 T0 pregate, reachability half).

HISTORICAL — DO NOT RE-RUN IN THIS FORM.  This script predates the
Contact Supervisor (PR-TT-4).  Its second run answered the reachability
question (mutual full curl closes the ~0.045 m open-pose gap to physical
CONTACT — reachability proven), but it answered it UNSAFELY: a mutual
full-range curl sweep with no live force monitoring, no micro-steps and
a lateral-extreme metric that was blind to the resulting finger
interleaving (incident 2026-07-31; fingers squeezed, no damage at
force_set=200 — luck, not design).  All future dual-hand approach must
go through the Bimanual ActionGateway + Contact Supervisor with
bilateral force consensus and 3D nearest-point contact detection.

What this script got RIGHT and keeps:

1. Both hands start OPEN; force baselines recorded in the open pose
   (T0's force-sensor baseline requirement).
2. Visual measurement only after settle; UNKNOWN states recorded, never
   interpolated.
3. Adaptive x-column valley split of one combined-ROI near cluster —
   fixed ROI boundaries clip reaching fingertips; the valley split
   separates the two hands without a hardcoded midline.

Verdict classes:

  REACHABLE      lateral gap <= 0.005 m at some step (overlap within noise)
  MARGINAL       0.005 < gap <= 0.03 m (contact uncertain, needs T1 probe)
  NOT_REACHABLE  gap > 0.03 m at every step — mount adjustment required,
                 quantified for the operator.

Measured results (2026-07-31, run 20260731T083536Z): gap 0.045 m open →
~0.011 m at full curl with rising valley fill — then visual contact in
the saved frames.  Verdict: REACHABLE (proven by contact, not by the
metric).
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

REPO_SRC = "/home/nvidia/workspace/rosclaw/rosclaw_test/rosclaw/src"
sys.path.insert(0, REPO_SRC)
sys.path.insert(0, "/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src")
sys.path.insert(0, "/home/nvidia/workspace/rosclaw/rosclaw_test/examples/rh56_rps/src")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyrealsense2 as rs  # noqa: E402

from rosclaw.evolution.hardware.camera import D435iCapture  # noqa: E402
from rosclaw.evolution.hardware.thermal import default_temp_probe  # noqa: E402
from rosclaw.perception.handcam import (  # noqa: E402
    CameraPoseContract,
    estimate_hand_state,
    intrinsics_from_pyrealsense,
)

RIGHT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04LBR0-if00-port0"
LEFT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04LB62-if00-port0"
# ROIs re-derived 2026-07-31 T0 run 1: the v3 ROIs ({0..230}/{410..640})
# are STALE — the mounts/camera were moved since v3; both hands now sit
# center-frame (left ~x270-430, right ~x450-580).  First run measured a
# meaningless 0.25-0.79 m "gap" from empty background.  (CameraPoseContract
# invalidation exists precisely for this.)
RIGHT_ROI = {"x0": 445, "y0": 200, "x1": 610, "y1": 480}
LEFT_ROI = {"x0": 250, "y0": 200, "x1": 445, "y1": 480}
COMBINED_ROI = {"x0": 250, "y0": 200, "x1": 610, "y1": 480}
CAMERA_SERIAL = "231122070092"
START_MAX_TEMP_C = 46.0
OUT_ROOT = Path("/home/nvidia/.rosclaw/acceptance/twintouch/t0")

JOINTS = ("little", "ring", "middle", "index", "thumb", "thumb_rot")
SLAVE_CANDIDATES = (1, 2)

SWEEP_RAW = [1000, 850, 700, 550, 400, 250]
SWEEP_SPEED = 150
SWEEP_FORCE = 200
SETTLE_S = 2.5
FRAMES_PER_STEP = 3


def open_probed_controller(port: str, label: str):
    """Open a hand controller and probe its slave id by response (v3 §4.4:
    identity by response, never by port-name guessing).  Per-hand lock
    paths so one process can own both ports without self-deadlock."""
    from rosclaw_rh56.transport.base import TransportConfig
    from rosclaw_rh56.transport.serial_rs485 import SerialRS485Transport
    from rosclaw_rps.hand.rh56_controller import RH56Controller

    transport = SerialRS485Transport(
        TransportConfig(
            kind="serial_rs485",
            port=port,
            baudrate=115200,
            timeout_s=1.0,
            lock_path=f"/tmp/rosclaw_rh56_serial_{label}.lock",
        )
    )
    ctl = RH56Controller(port=port, transport=transport)
    ctl.connect()
    proto = ctl._proto
    transport = ctl._transport
    for candidate in SLAVE_CANDIDATES:
        for _attempt in range(3):
            proto.device_id = candidate
            transport.flush_input()
            transport.write(proto.read_angle_actual())
            time.sleep(0.5)
            if transport.read(64, timeout_s=1.0):
                return ctl, candidate
            time.sleep(0.2)
    ctl.close()
    raise RuntimeError(f"{label} hand: no modbus response at any candidate slave id")


def _cluster_points_m(
    depth_mm: np.ndarray,
    roi: dict,
    intr,
    *,
    near_window_mm: float = 80.0,
    min_pixels: int = 300,
    bin_mm: int = 25,
) -> np.ndarray | None:
    """Nearest-depth-cluster points in meters (same histogram approach as
    PE-3; duplicated here because the probe needs the raw points for
    lateral extremes, which the library summary does not expose)."""
    x0, y0 = roi.get("x0", 0), roi.get("y0", 0)
    x1 = roi.get("x1", depth_mm.shape[1])
    y1 = roi.get("y1", depth_mm.shape[0])
    region = depth_mm[y0:y1, x0:x1]
    valid = region[region > 0]
    if len(valid) < min_pixels:
        return None
    max_depth = int(valid.max()) + bin_mm
    counts, edges = np.histogram(valid, bins=np.arange(0, max_depth + bin_mm, bin_mm))
    thresh_low = None
    for index, count in enumerate(counts):
        if count >= min_pixels:
            thresh_low = float(edges[index])
            thresh_high = float(edges[index + 1] + near_window_mm)
            break
    if thresh_low is None:
        return None
    mask = (region > 0) & (region >= thresh_low) & (region <= thresh_high)
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    depths_m = region[ys, xs].astype(np.float64) / 1000.0
    us = xs.astype(np.float64) + x0
    vs = ys.astype(np.float64) + y0
    return np.stack(
        [
            (us - intr.ppx) * depths_m / intr.fx,
            (vs - intr.ppy) * depths_m / intr.fy,
            depths_m,
        ],
        axis=1,
    )


def _robust_extreme(points: np.ndarray, side: str) -> tuple[float, float, float] | None:
    """Lateral extreme toward the OTHER hand, robust to pixel noise:
    mean of points within 1 cm of the 99th (left hand, max-x) or 1st
    (right hand, min-x) x-percentile."""
    if points is None or len(points) < 50:
        return None
    xs = points[:, 0]
    q = np.percentile(xs, 99.0 if side == "left" else 1.0)
    near = points[np.abs(xs - q) <= 0.01]
    if len(near) < 10:
        return None
    return tuple(float(v) for v in near.mean(axis=0))


def _adaptive_split_gap(points_m: np.ndarray, us: np.ndarray) -> dict:
    """Inter-hand gap from ONE combined-ROI near cluster via the x-column
    valley between the two hands.

    Fixed ROI boundaries clip reaching fingertips (conservative but
    blind to actual touch); instead: histogram the cluster's pixel
    columns, find the left and right modes, and split at the valley
    between them.  If the valley is shallow (<20% of the lower mode)
    the two clusters are CONNECTED in image space — the hands are
    touching or within one depth bin: gap ~ 0, flagged merged=True.
    """
    if points_m is None or len(points_m) < 600:
        return {"state": "unknown", "reason": "combined cluster too small"}
    u_min, u_max = int(us.min()), int(us.max())
    if u_max - u_min < 40:
        return {"state": "unknown", "reason": "cluster too narrow for two hands"}
    cols = np.zeros(u_max + 1, dtype=np.int64)
    for u in us.astype(int):
        cols[u] += 1
    # smooth over 3 px to suppress single-column noise
    smooth = np.convolve(cols, np.ones(3) / 3, mode="same")
    occupied = np.nonzero(smooth > 2)[0]
    if len(occupied) < 40:
        return {"state": "unknown", "reason": "too few occupied columns"}
    left_edge, right_edge = int(occupied[0]), int(occupied[-1])
    mid = (left_edge + right_edge) // 2
    left_mode_x = left_edge + int(np.argmax(smooth[left_edge : mid + 1]))
    right_mode_x = mid + int(np.argmax(smooth[mid : right_edge + 1]))
    if right_mode_x - left_mode_x < 20:
        return {"state": "unknown", "reason": "hand modes not separable"}
    valley_x = left_mode_x + int(np.argmin(smooth[left_mode_x : right_mode_x + 1]))
    valley = float(smooth[valley_x])
    lower_mode = float(min(smooth[left_mode_x], smooth[right_mode_x]))
    merged = valley >= 0.2 * lower_mode
    left_points = points_m[us <= valley_x]
    right_points = points_m[us > valley_x]
    left_ext = _robust_extreme(left_points, "left")
    right_ext = _robust_extreme(right_points, "right")
    out = {
        "state": "ok",
        "split_col_u": int(valley_x),
        "valley_fill_ratio": round(valley / lower_mode, 3) if lower_mode else None,
        "merged": merged,
        "left_points": int(len(left_points)),
        "right_points": int(len(right_points)),
        "left_extreme": left_ext,
        "right_extreme": right_ext,
    }
    if left_ext is not None and right_ext is not None:
        out["gap_lateral_m"] = round(right_ext[0] - left_ext[0], 4)
        out["gap_3d_extremes_m"] = round(
            float(np.linalg.norm(np.array(right_ext) - np.array(left_ext))), 4
        )
        if merged and out["gap_lateral_m"] <= 0.02:
            # connected clusters + extremes within noise => visual contact
            out["gap_lateral_m"] = min(out["gap_lateral_m"], 0.0)
    else:
        out["gap_lateral_m"] = None
        out["gap_3d_extremes_m"] = None
    return out


def measure_once(cap, contracts, intr, label: str) -> dict:
    """One measurement: PE-3 states per hand + adaptive-split inter-hand gap."""
    frame = cap.read()
    states = {s: estimate_hand_state(frame.depth, c) for s, c in contracts.items()}
    points = _cluster_points_m(frame.depth, COMBINED_ROI, intr)
    if points is None:
        gap = {"state": "unknown", "reason": "no combined cluster"}
    else:
        # recompute pixel columns for the same mask (kept in sync with
        # _cluster_points_m's deprojection)
        x0, y0 = COMBINED_ROI["x0"], COMBINED_ROI["y0"]
        region = frame.depth[y0 : COMBINED_ROI["y1"], x0 : COMBINED_ROI["x1"]]
        valid = region[region > 0]
        counts, edges = np.histogram(valid, bins=np.arange(0, int(valid.max()) + 50, 25))
        thresh_low = thresh_high = None
        for index, count in enumerate(counts):
            if count >= 300:
                thresh_low = float(edges[index])
                thresh_high = float(edges[index + 1] + 80.0)
                break
        mask = (region > 0) & (region >= thresh_low) & (region <= thresh_high)
        ys, xs = np.nonzero(mask)
        us = xs.astype(np.float64) + x0
        gap = _adaptive_split_gap(points, us)
    return {
        "label": label,
        "ts": time.time(),
        "states": {s: states[s].to_dict() for s in states},
        "inter_hand_gap": gap,
    }


def main() -> int:
    probe = default_temp_probe()
    temps = [v for v in probe.values() if isinstance(v, (int, float))]
    if temps and max(temps) > START_MAX_TEMP_C:
        print(json.dumps({"ok": False, "blocked": f"start {max(temps)}°C > {START_MAX_TEMP_C}°C"}))
        return 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    outcomes: dict = {"run_id": run_id, "start_temps": probe, "steps": [], "verdict": None}
    controllers: dict = {}
    cap = D435iCapture()
    try:
        for label, port in (("right", RIGHT_PORT), ("left", LEFT_PORT)):
            controllers[label], slave = open_probed_controller(port, label)
            outcomes.setdefault("slave_ids", {})[label] = slave

        cap.start(serial=CAMERA_SERIAL)
        intr = intrinsics_from_pyrealsense(cap._pipeline.get_active_profile(), rs.stream.depth)
        contracts = {
            "left": CameraPoseContract(
                camera_pose_id="twintouch_t0_left", camera_id="d435i",
                intrinsics=intr, roi=LEFT_ROI,
            ),
            "right": CameraPoseContract(
                camera_pose_id="twintouch_t0_right", camera_id="d435i",
                intrinsics=intr, roi=RIGHT_ROI,
            ),
        }

        # T0 force baseline: open hand, no contact — 5 reads per hand.
        baselines: dict = {}
        for _label, ctl in controllers.items():
            reads = []
            for _ in range(5):
                tel = ctl.read_telemetry()
                reads.append(tel.force_act or {})
                time.sleep(0.15)
            baselines[label] = {
                j: round(float(np.mean([r.get(j) for r in reads if r.get(j) is not None])), 1)
                if any(r.get(j) is not None for r in reads)
                else None
                for j in JOINTS
            }
        outcomes["force_baseline_open"] = baselines

        # Curl sweep.
        best_gap = None
        best_step = None
        for raw in SWEEP_RAW:
            target = [raw] * len(JOINTS)
            for _label, ctl in controllers.items():
                ctl.move_to_gesture(f"sweep{raw}", target, SWEEP_SPEED, SWEEP_FORCE)
            time.sleep(SETTLE_S)
            step_measures = []
            for frame_index in range(FRAMES_PER_STEP):
                measure = measure_once(cap, contracts, intr, f"sweep{raw}_f{frame_index}")
                step_measures.append(measure)
                time.sleep(0.1)
            frame = cap.read()
            cv2.imwrite(str(run_dir / f"sweep{raw}.png"), frame.color)
            gaps = [
                m["inter_hand_gap"]["gap_lateral_m"]
                for m in step_measures
                if m["inter_hand_gap"].get("gap_lateral_m") is not None
            ]
            merged_any = any(m["inter_hand_gap"].get("merged") for m in step_measures)
            med_gap = float(np.median(gaps)) if gaps else None
            outcomes["steps"].append(
                {
                    "raw": raw,
                    "median_gap_lateral_m": None if med_gap is None else round(med_gap, 4),
                    "merged_any": merged_any,
                    "measurements": step_measures,
                }
            )
            if med_gap is not None and (best_gap is None or med_gap < best_gap):
                best_gap = med_gap
                best_step = raw

        outcomes["best"] = {"raw": best_step, "gap_lateral_m": best_gap}
        if best_gap is None:
            outcomes["verdict"] = "UNKNOWN_NO_VISUAL"
        elif best_gap <= 0.005:
            outcomes["verdict"] = "REACHABLE"
        elif best_gap <= 0.03:
            outcomes["verdict"] = "MARGINAL"
        else:
            outcomes["verdict"] = "NOT_REACHABLE"
            outcomes["operator_action"] = (
                f"fingertips cannot meet: minimum lateral gap {best_gap:.3f} m at curl "
                f"raw={best_step}; move mounts at least {best_gap + 0.01:.3f} m closer "
                f"together (plus ~1 cm contact margin) or angle palms toward each other"
            )
    finally:
        for ctl in controllers.values():
            try:
                ctl.safe_open()
                time.sleep(0.3)
                # coast: low holding force to limit self-heating
                ctl.move_to_gesture("open", [1000] * len(JOINTS), 300, 50)
                ctl.close()
            except Exception:  # noqa: BLE001
                pass
        with contextlib.suppress(Exception):
            cap.stop()

    report_path = run_dir / "t0_reachability_report.json"
    report_path.write_text(json.dumps(outcomes, indent=2, ensure_ascii=False, default=str))
    outcomes["report_path"] = str(report_path)
    summary = {
        "run_id": run_id,
        "verdict": outcomes["verdict"],
        "best": outcomes["best"],
        "steps": [
            {"raw": s["raw"], "median_gap_lateral_m": s["median_gap_lateral_m"]}
            for s in outcomes["steps"]
        ],
        "force_baseline_open": outcomes.get("force_baseline_open"),
        "report_path": str(report_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0 if outcomes["verdict"] in ("REACHABLE", "MARGINAL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
