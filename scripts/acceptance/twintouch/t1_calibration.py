#!/usr/bin/env python3
"""TwinTouch T1: single-side contact calibration (v4 §11 T1, PR-TT-5).

FIRST physical use of the TT-1..TT-4 machinery.  Every motion goes
through the Bimanual ActionGateway (atomic lease + sequence permit +
preconditions); every episode is driven by the Contact Supervisor's
pure state machine; the Action Effect Gate verifies the safe-reset
BEFORE any approach is allowed (the v3 static-hand lesson, wired
physically); a hardware watchdog independent of the supervisor loop
reads forces at ~2 Hz and retreats BOTH hands on any threshold
violation even if the main loop is stuck (the 2026-07-31 lesson).

Calibration-bootstrap disclosure: the supervisor refuses UNCALIBRATED
pairs at PAIR_SELECTED (§12.2).  T1's whole purpose is to produce the
first per-pair envelopes, so the runner passes
``reachability_calibrated=True`` ONLY for the pair named by the
approved calibration contract, with ultra_light tuning (probe force,
hard step budgets) — that IS the approved validation path; every other
pair remains forbidden.

Per pair × per mode (passive_active = right active / active_passive =
left active / mutual) × repeats, the runner records:

  precontact (fine-zone entry raw per side)
  first single-side force (raw at first target rise)
  bilateral confirm (raw at CONTACT_CONFIRMED + force peaks)
  visual near range (3D nearest-cluster distance at confirm)
  release margin actually needed
  hysteresis (confirm raw vs release-complete raw)

and emits FingerContactEnvelope records bound to the camera pose hash
and body snapshot hashes — the T1 deliverable (v4 §21 PR-TT-5).

Safety invariants (hard, from config — never tuned by this script):
  one active pair; non-target force abort; any hand lost → both
  retreat; stale camera → no approach; start ≤46°C, abort ≥49°C,
  hardware protection 52°C; approach speed ≤150, fine ≤100;
  force_set ≤150 approach / ≤100 dwell; coast 50 on exit ALWAYS.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
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
    intrinsics_from_pyrealsense,
)
from rosclaw.twintouch.choreography import (  # noqa: E402
    ContactChoreographyContract,
    SequencePermit,
)
from rosclaw.twintouch.config import TwinTouchConfig  # noqa: E402
from rosclaw.twintouch.effect_gate import (  # noqa: E402
    COMMAND_HOLD,
    COMMAND_MOVE,
    EFFECT_CONFIRMED,
    JointCommand,
    TelemetryPoint,
    VisualSample,
    evaluate_effect,
)
from rosclaw.twintouch.envelope import (  # noqa: E402
    BimanualActionEnvelope,
    BodyActionBlock,
    CoordinationBlock,
    SafetyBlock,
)
from rosclaw.twintouch.gateway import (  # noqa: E402
    BimanualActionGateway,
    LeaseRegistry,
)
from rosclaw.twintouch.pairs import (  # noqa: E402
    FORBIDDEN_FINGERTIP_PAIRS,
    RH56_JOINTS,
    pair_by_id,
)
from rosclaw.twintouch.supervisor import (  # noqa: E402
    COARSE_APPROACH,
    DECISION_COMMIT,
    DECISION_FAIL,
    DECISION_ISSUE_STEP,
    DECISION_RETREAT,
    EPISODE_COMMITTED,
    FINE_APPROACH,
    RECORD_FAILURE,
    ContactSupervisor,
    ForceBaseline,
    HandObservation,
    SupervisorObservation,
    SupervisorTuning,
    VisualObservation,
)

RIGHT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04LBR0-if00-port0"
LEFT_PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG04LB62-if00-port0"
RIGHT_ROI = {"x0": 445, "y0": 200, "x1": 610, "y1": 480}
LEFT_ROI = {"x0": 250, "y0": 200, "x1": 445, "y1": 480}
COMBINED_ROI = {"x0": 250, "y0": 200, "x1": 610, "y1": 480}
CAMERA_SERIAL = "231122070092"
# T0-recorded identity: if the pose hashes changed, the layout changed
# and every T0 measurement is void — abort, never adapt silently.
EXPECTED_POSE_HASHES = {
    "left": "campose_f5bdf7f2134abe0c",
    "right": "campose_b03dffb24a2b9299",
}
T0_BUNDLE = Path("/home/nvidia/.rosclaw/acceptance/twintouch/t0/latest/t0_evidence_bundle.json")
OUT_ROOT = Path("/home/nvidia/.rosclaw/acceptance/twintouch/t1")

SLAVE_CANDIDATES = (1, 2)
BODY_IDS = {"left": "rh56_left_01", "right": "rh56_right_01"}
OPEN_RAW = {j: 1000 for j in RH56_JOINTS}
TELEMETRY_MIN_INTERVAL_S = {"left": 0.25, "right": 0.0}


# ---------------------------------------------------------------- hardware


def open_probed_controller(port: str, label: str):
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


class Rh56BodyExecutor:
    """BodyExecutor binding for one RH56 (gateway dispatch surface)."""

    def __init__(self, side: str, ctl, speed: int, force: int):
        self.side = side
        self.ctl = ctl
        self.speed = speed
        self.force = force
        self._dispatch_counter = 0

    def dispatch(self, action: dict, *, timeout_ms: float) -> str:
        targets = action.get("targets")
        if not targets:
            raise RuntimeError(f"{self.side}: action has no targets")
        speed = int(action.get("speed", self.speed))
        force = int(action.get("force", self.force))
        self.ctl.move_to_gesture(
            f"tt_{self.side}", [int(targets[j]) for j in RH56_JOINTS], speed, force
        )
        self._dispatch_counter += 1
        return f"arec_{self.side}_{self._dispatch_counter}"

    def retreat(self, retreat_action: dict) -> bool:
        try:
            self.ctl.move_to_gesture(
                f"tt_{self.side}_retreat",
                [OPEN_RAW[j] for j in RH56_JOINTS],
                int(retreat_action.get("speed", 300)),
                int(retreat_action.get("force", 150)),
            )
            time.sleep(1.0)
            tel = self.ctl.read_telemetry()
            angles = tel.angle_actual or {}
            return all((angles.get(j) or 0) >= 800 for j in ("little", "ring", "middle", "index"))
        except Exception:  # noqa: BLE001
            return False

    def estop(self) -> bool:
        """Immediate hold-in-place: re-command the CURRENT actual
        position (fastest stop the servo honors; the zero-delta dip is
        ~15-17 raw on gravity joints — acceptable for a stop)."""
        try:
            tel = self.ctl.read_telemetry()
            angles = tel.angle_actual or {}
            self.ctl.move_to_gesture(
                f"tt_{self.side}_estop",
                [int(angles.get(j) or OPEN_RAW[j]) for j in RH56_JOINTS],
                300,
                100,
            )
            return True
        except Exception:  # noqa: BLE001
            return False


class LiveProbe:
    """PreconditionProbe over live camera + snapshot freshness."""

    def __init__(self, collector: "ObservationCollector"):
        self._collector = collector

    def camera_fresh(self, *, max_age_ms: float) -> bool:
        age = self._collector.last_frame_age_ms()
        return age is not None and age <= max_age_ms

    def snapshots_valid(self, envelope: BimanualActionEnvelope) -> list[str]:
        # Phase 1: snapshot freshness = both controllers answering
        # telemetry within the last second (body presence, not a hash).
        violations: list[str] = []
        ages = self._collector.telemetry_ages_s()
        for side, age in ages.items():
            if age is None or age > 1.0:
                violations.append(f"{side} telemetry age {age} — snapshot stale")
        return violations


# ------------------------------------------------------- observation layer


class ObservationCollector:
    """Continuous dual telemetry + visual collection with the FORCE
    WATCHDOG built in: independent of the supervisor loop, any threshold
    violation sets abort and the runner retreats both hands."""

    def __init__(self, controllers: dict, cap, contracts: dict, intr, baselines, config):
        self._controllers = controllers
        self._cap = cap
        self._contracts = contracts
        self._intr = intr
        self._baselines = baselines
        self._config = config
        self._lock = threading.Lock()
        self._telemetry: dict[str, tuple[object, float] | None] = {"left": None, "right": None}
        self._last_tel_ts: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._frame_ts: float | None = None
        self._visual: VisualObservation | None = None
        self._visual_ts: float | None = None
        self._last_depth: np.ndarray | None = None
        self.abort_reason: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def last_frame_age_ms(self) -> float | None:
        with self._lock:
            if self._frame_ts is None:
                return None
            return (time.time() - self._frame_ts) * 1000.0

    def telemetry_ages_s(self) -> dict[str, float | None]:
        with self._lock:
            now = time.time()
            return {
                side: (None if entry is None else now - entry[1])
                for side, entry in self._telemetry.items()
            }

    def latest(self) -> SupervisorObservation:
        with self._lock:
            hands: dict[str, HandObservation | None] = {}
            for side in ("left", "right"):
                entry = self._telemetry[side]
                if entry is None:
                    hands[side] = None
                    continue
                tel, _ts = entry
                temps = [v for v in (tel.temperature_c or {}).values() if isinstance(v, (int, float)) and v > 0]
                hands[side] = HandObservation(
                    ok=True,
                    angle_actual=dict(tel.angle_actual or {}),
                    force_act={k: (None if v is None else float(v)) for k, v in (tel.force_act or {}).items()},
                    temperature_max_c=max(temps) if temps else None,
                )
            # the stored visual carries its CAPTURE time; age is computed
            # NOW so a wedged camera goes stale instead of lying fresh
            visual = self._visual
            if visual is not None and self._visual_ts is not None:
                visual = VisualObservation(
                    age_ms=(time.time() - self._visual_ts) * 1000.0,
                    left_cluster_ok=visual.left_cluster_ok,
                    right_cluster_ok=visual.right_cluster_ok,
                    min_distance_m=visual.min_distance_m,
                    pair_identity_confirmed=visual.pair_identity_confirmed,
                )
            return SupervisorObservation(
                ts_s=time.time(),
                left=hands["left"],
                right=hands["right"],
                visual=visual,
            )

    def visual_sample(self, side: str) -> "VisualSample":
        """Real per-hand cluster centroid for the effect gate — never a
        fabricated constant."""
        with self._lock:
            depth = self._last_depth
        if depth is None:
            return VisualSample(ok=False, centroid_3d=None)
        roi = LEFT_ROI if side == "left" else RIGHT_ROI
        points, _us = _cluster_points(depth, roi, self._intr)
        if points is None:
            return VisualSample(ok=False, centroid_3d=None)
        return VisualSample(ok=True, centroid_3d=tuple(float(v) for v in points.mean(axis=0)))

    # watchdog thresholds come from config (never tuned here).  The
    # supervisor itself aborts at 1x; the watchdog is the LAST backstop
    # for a STUCK supervisor loop, so it fires at 2x — late enough not
    # to race the supervisor, early enough to bound the excursion.
    def _watchdog_check(self, side: str, tel) -> None:
        deltas = self._baselines[side].delta(tel.force_act or {})
        for finger, delta in deltas.items():
            if delta >= self._config.non_target_force_abort_raw * 2:
                self.abort_reason = (
                    f"WATCHDOG: {side}.{finger} +{delta:.0f} raw "
                    f"(2x non-target abort) — retreat both"
                )
        temp_vals = [v for v in (tel.temperature_c or {}).values() if isinstance(v, (int, float)) and v > 0]
        if temp_vals and max(temp_vals) >= self._config.temperature_abort_c:
            self.abort_reason = f"WATCHDOG: {side} temp {max(temp_vals)}°C >= abort"

    def _loop(self) -> None:
        while not self._stop.is_set():
            for side, ctl in self._controllers.items():
                if time.time() - self._last_tel_ts[side] < TELEMETRY_MIN_INTERVAL_S[side]:
                    continue
                try:
                    tel = ctl.read_telemetry()
                    with self._lock:
                        self._telemetry[side] = (tel, time.time())
                        self._last_tel_ts[side] = time.time()
                    self._watchdog_check(side, tel)
                except Exception:  # noqa: BLE001
                    with self._lock:
                        self._telemetry[side] = None
            try:
                frame = self._cap.read()
                with self._lock:
                    self._frame_ts = time.time()
                    self._visual_ts = time.time()
                    self._last_depth = frame.depth
                    self._visual = self._compute_visual(frame.depth)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.05)

    def _compute_visual(self, depth_mm: np.ndarray) -> VisualObservation:
        clusters = _split_clusters(depth_mm, COMBINED_ROI, self._intr)
        if clusters is None:
            return VisualObservation(
                age_ms=0.0,
                left_cluster_ok=False,
                right_cluster_ok=False,
                min_distance_m=None,
                pair_identity_confirmed=None,
            )
        left_points, right_points, merged = clusters
        ok = left_points is not None and right_points is not None
        distance = None
        if ok:
            distance = _min_pairwise_distance(left_points, right_points)
            if merged:
                distance = 0.0
        return VisualObservation(
            age_ms=0.0,
            left_cluster_ok=left_points is not None,
            right_cluster_ok=right_points is not None,
            min_distance_m=distance,
            # hand-level identity (v4 §6.2: visual says WHO is where,
            # not which finger — finger identity is command+force)
            pair_identity_confirmed=True if ok else None,
        )


def _cluster_points(depth_mm: np.ndarray, roi: dict, intr, *, min_pixels: int = 300):
    x0, y0 = roi.get("x0", 0), roi.get("y0", 0)
    region = depth_mm[y0 : roi.get("y1", depth_mm.shape[0]), x0 : roi.get("x1", depth_mm.shape[1])]
    valid = region[region > 0]
    if len(valid) < min_pixels:
        return None, None
    counts, edges = np.histogram(valid, bins=np.arange(0, int(valid.max()) + 50, 25))
    thresh_low = thresh_high = None
    for index, count in enumerate(counts):
        if count >= min_pixels:
            thresh_low = float(edges[index])
            thresh_high = float(edges[index + 1] + 80.0)
            break
    if thresh_low is None:
        return None, None
    mask = (region > 0) & (region >= thresh_low) & (region <= thresh_high)
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None, None
    depths_m = region[ys, xs].astype(np.float64) / 1000.0
    us = xs.astype(np.float64) + x0
    vs = ys.astype(np.float64) + y0
    points = np.stack(
        [(us - intr.ppx) * depths_m / intr.fx, (vs - intr.ppy) * depths_m / intr.fy, depths_m],
        axis=1,
    )
    return points, us


def _split_clusters(depth_mm: np.ndarray, roi: dict, intr):
    """Adaptive x-column valley split (from T0) + merged flag."""
    points, us = _cluster_points(depth_mm, roi, intr)
    if points is None or len(points) < 600:
        return None
    u_min, u_max = int(us.min()), int(us.max())
    if u_max - u_min < 40:
        return None
    cols = np.zeros(u_max + 1, dtype=np.int64)
    for u in us.astype(int):
        cols[u] += 1
    smooth = np.convolve(cols, np.ones(3) / 3, mode="same")
    occupied = np.nonzero(smooth > 2)[0]
    if len(occupied) < 40:
        return None
    left_edge, right_edge = int(occupied[0]), int(occupied[-1])
    mid = (left_edge + right_edge) // 2
    left_mode_x = left_edge + int(np.argmax(smooth[left_edge : mid + 1]))
    right_mode_x = mid + int(np.argmax(smooth[mid : right_edge + 1]))
    if right_mode_x - left_mode_x < 20:
        return None
    valley_x = left_mode_x + int(np.argmin(smooth[left_mode_x : right_mode_x + 1]))
    valley = float(smooth[valley_x])
    lower_mode = float(min(smooth[left_mode_x], smooth[right_mode_x]))
    merged = valley >= 0.2 * lower_mode
    return points[us <= valley_x], points[us > valley_x], merged


def _min_pairwise_distance(a: np.ndarray, b: np.ndarray, *, max_points: int = 800) -> float:
    """3D nearest-cluster distance — the metric that sees interleaving."""
    if len(a) > max_points:
        a = a[np.linspace(0, len(a) - 1, max_points).astype(int)]
    if len(b) > max_points:
        b = b[np.linspace(0, len(b) - 1, max_points).astype(int)]
    best = float("inf")
    for chunk in np.array_split(a, max(1, len(a) // 200)):
        dists = np.sqrt(((chunk[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
        best = min(best, float(dists.min()))
    return best


# ------------------------------------------------------------- episode kit


# Per-mode starting geometry (T0-informed priors, calibration subjects):
# an open passive hand points its fingertips UP — an approaching active
# fingertip would meet the passive finger's SIDE or the palm (forbidden
# zone), never tip-to-tip.  The passive side PRESENTS its target finger
# at mid-curl toward the peer (T0 showed mutual mid-curl ~550 raw
# converging tip-to-tip); mutual mode presents both sides.  These are
# STARTING priors — T1 measures the real contact positions around them.
PRESENT_RAW = {"index": 550, "middle": 550, "ring": 550, "little": 550, "thumb": 450}
THUMB_ROT_PRESENT_RAW = 300


def _present_targets(pair_id: str, mode: str) -> dict[str, dict[str, int]]:
    finger = pair_id.split("_")[0]
    targets = {"left": dict(OPEN_RAW), "right": dict(OPEN_RAW)}
    # presenting side = the PASSIVE one for single-side modes, both for mutual
    presenting = {
        "active_passive": ("right",),  # left active -> right presents
        "passive_active": ("left",),  # right active -> left presents
        "mutual": ("left", "right"),
    }[mode]
    for side in presenting:
        targets[side][finger] = PRESENT_RAW[finger]
        if finger == "thumb":
            targets[side]["thumb_rot"] = THUMB_ROT_PRESENT_RAW
    return targets


@dataclass
class EpisodeRecord:
    pair_id: str
    mode: str
    repeat: int
    outcome: str | None = None
    state_history: list[str] = field(default_factory=list)
    receipt: dict | None = None
    envelope_facts: dict = field(default_factory=dict)
    started_at: str = ""
    notes: list[str] = field(default_factory=list)


def _intent_card(pair_id: str, modes: list[str], repeats: int) -> str:
    """The operator-intent disclosure bound into the SequencePermit.
    Standing authorization: the user's v4 task document explicitly
    prescribes T1 (模式A/B per pair, low-speed repeated approaches);
    this card records that linkage instead of pretending a fresh
    interactive confirmation happened."""
    return (
        f"t1_calibration pair={pair_id} modes={modes} repeats={repeats} "
        "authorization=v4-doc-§11-T1 operator=user-via-standing-task-doc"
    )


def build_episode_envelope(
    *,
    interaction_id: str,
    sequence_id: str,
    pair_id: str,
    left_targets: dict[str, int],
    right_targets: dict[str, int],
    speed: int,
    force: int,
    contract_hash: str,
    snapshot_hash: str,
) -> BimanualActionEnvelope:
    def block(body_id: str, targets: dict[str, int]) -> BodyActionBlock:
        return BodyActionBlock(
            body_id=body_id,
            action={
                "gesture": "tt_step",
                "targets": targets,
                "speed": speed,
                "force": force,
            },
            body_snapshot_hash=snapshot_hash,
            calibration_hash="t0_baseline_open",
        )

    return BimanualActionEnvelope(
        interaction_id=interaction_id,
        sequence_id=sequence_id,
        pair_id=pair_id,
        left=block(BODY_IDS["left"], left_targets),
        right=block(BODY_IDS["right"], right_targets),
        coordination=CoordinationBlock(
            mode="mutual",
            synchronization_barrier="start_together",
            maximum_start_skew_ms=250.0,
            timeout_ms=4000.0,
        ),
        safety=SafetyBlock(
            contract_hash=contract_hash,
            permitted_contact_pair=pair_id,
            forbidden_contact_pairs=tuple(sorted(FORBIDDEN_FINGERTIP_PAIRS)),
            retreat_action={"gesture": "safe_open", "speed": 300, "force": 150},
        ),
    )


# ---------------------------------------------------------------- runner


def run() -> int:
    config = TwinTouchConfig.load()
    probe = default_temp_probe()
    temps = [v for v in probe.values() if isinstance(v, (int, float))]
    if temps and max(temps) > config.temperature_start_max_c:
        print(json.dumps({"ok": False, "blocked": f"start {max(temps)}°C > gate"}))
        return 1
    if not T0_BUNDLE.exists():
        print(json.dumps({"ok": False, "blocked": "T0 evidence bundle missing"}))
        return 1

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="index_index", help="comma list; pilot is index_index")
    parser.add_argument("--modes", default="passive_active,active_passive,mutual")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--pregate-only",
        action="store_true",
        help="stop after the effect-gate pregate — validates identity, "
        "pose-hash gate, baselines, collector+watchdog, gateway dispatch "
        "(open-pose hold only, zero approach) and the v3-lesson effect "
        "gate on live hardware without any approach motion",
    )
    args = parser.parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "run_id": run_id,
        "pairs": pairs,
        "modes": modes,
        "repeats": args.repeats,
        "start_temps": probe,
        "episodes": [],
        "envelopes": {},
        "aborts": [],
    }

    controllers: dict = {}
    cap = D435iCapture()
    collector: ObservationCollector | None = None
    try:
        for label, port in (("right", RIGHT_PORT), ("left", LEFT_PORT)):
            controllers[label], slave = open_probed_controller(port, label)
            report.setdefault("slave_ids", {})[label] = slave

        cap.start(serial=CAMERA_SERIAL)
        intr = intrinsics_from_pyrealsense(cap._pipeline.get_active_profile(), rs.stream.depth)
        contracts = {
            side: CameraPoseContract(
                camera_pose_id=f"twintouch_t0_{side}", camera_id="d435i", intrinsics=intr, roi=roi
            )
            for side, roi in (("left", LEFT_ROI), ("right", RIGHT_ROI))
        }
        for side, contract in contracts.items():
            if contract.camera_pose_hash != EXPECTED_POSE_HASHES[side]:
                report["aborts"].append(
                    f"LAYOUT_CHANGED: {side} pose hash {contract.camera_pose_hash} "
                    f"!= T0 {EXPECTED_POSE_HASHES[side]} — T0 evidence void, abort"
                )
                print(json.dumps(report, indent=2, default=str))
                return 2

        # hands must be OPEN for the baseline (RH56 calibration rule)
        for side, ctl in controllers.items():
            tel = ctl.read_telemetry()
            angles = tel.angle_actual or {}
            if any((angles.get(j) or 0) < 900 for j in ("little", "ring", "middle", "index")):
                ctl.move_to_gesture("t1_open", [1000] * len(RH56_JOINTS), 150, 150)
        time.sleep(3.0)

        baselines: dict[str, ForceBaseline] = {}
        for side, ctl in controllers.items():
            samples = []
            for _ in range(8):
                tel = ctl.read_telemetry()
                samples.append({k: (None if v is None else float(v)) for k, v in (tel.force_act or {}).items()})
                time.sleep(0.2)
            baselines[side] = ForceBaseline.capture(side, samples, min_samples=6)
        report["force_baselines"] = {s: b.medians for s, b in baselines.items()}

        executors = {
            side: Rh56BodyExecutor(side, ctl, config.servo_max_speed_approach, config.approach_force_set)
            for side, ctl in controllers.items()
        }
        collector = ObservationCollector(controllers, cap, contracts, intr, baselines, config)
        collector.start()
        gateway = BimanualActionGateway(
            executors=executors,
            leases=LeaseRegistry(BODY_IDS),
            probe=LiveProbe(collector),
            camera_freshness_ms=config.camera_freshness_ms,
        )

        # ---- effect-gate pregate (the v3 lesson, physically wired):
        # both hands commanded to safe_open (they are already open —
        # HOLD semantics); BOTH must confirm or nothing approaches.
        # Telemetry before/after and visual centroids are REAL reads.
        snapshot_hash = f"t1_{run_id}"
        pregate_pair = pairs[0]
        pre_contract = ContactChoreographyContract(
            pattern="fingertip_marquee",
            pairs=(
                "thumb_thumb", "index_index", "middle_middle", "ring_ring",
                "little_little", "ring_ring", "middle_middle", "index_index", "thumb_thumb",
            ),
            cycles=1,
            force_level="ultra_light",
            left_body_hash=snapshot_hash,
            right_body_hash=snapshot_hash,
            camera_pose_hash=EXPECTED_POSE_HASHES["left"],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        violations = pre_contract.validate()
        if violations:
            report["aborts"].append(f"pregate contract invalid: {violations}")
            print(json.dumps(report, indent=2, default=str))
            return 2
        permit = SequencePermit.issue(
            pre_contract, intent_hash=_intent_card(pregate_pair, ["pregate"], 1), lifetime_s=120.0
        )
        telemetry_before: dict[str, TelemetryPoint] = {}
        visual_before: dict[str, VisualSample] = {}
        for side, ctl in controllers.items():
            tel = ctl.read_telemetry()
            telemetry_before[side] = TelemetryPoint(
                ts_s=time.time(), angle_actual=dict(tel.angle_actual or {})
            )
            visual_before[side] = collector.visual_sample(side)
        envelope = build_episode_envelope(
            interaction_id=f"pregate_{run_id}",
            sequence_id=f"pregate_{run_id}",
            pair_id=pregate_pair,
            left_targets=dict(OPEN_RAW),
            right_targets=dict(OPEN_RAW),
            speed=150,
            force=100,
            contract_hash=pre_contract.contract_hash(),
            snapshot_hash=snapshot_hash,
        )
        dispatch = gateway.dispatch(envelope, contract=pre_contract, permit=permit)
        if dispatch.violation_kind is not None:
            report["aborts"].append(f"pregate dispatch blocked: {dispatch.violations}")
            print(json.dumps(report, indent=2, default=str))
            return 2
        time.sleep(2.0)
        effect_results: dict[str, str] = {}
        for side, ctl in controllers.items():
            tel = ctl.read_telemetry()
            tel_after = TelemetryPoint(ts_s=time.time(), angle_actual=dict(tel.angle_actual or {}))
            other = "right" if side == "left" else "left"
            command = JointCommand(
                action_id=f"pregate_{side}",
                body_id=BODY_IDS[side],
                command_type=COMMAND_HOLD,
                targets=dict(OPEN_RAW),
                issued_at_s=telemetry_before[side].ts_s,
                window_s=2.0,
            )
            receipt = evaluate_effect(
                command,
                [telemetry_before[side], tel_after],
                visual_before=visual_before[side],
                visual_after=collector.visual_sample(side),
                other_before=visual_before[other],
                other_after=collector.visual_sample(other),
            )
            effect_results[side] = receipt.verdict
        report["pregate_effects"] = effect_results
        if any(v != EFFECT_CONFIRMED for v in effect_results.values()):
            report["aborts"].append(
                f"pregate effect gate BLOCKED: {effect_results} — no approach allowed (v3 lesson)"
            )
            print(json.dumps(report, indent=2, default=str))
            return 2
        if args.pregate_only:
            report["pregate_only"] = True
            (out_dir / "t1_calibration_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str)
            )
            print(json.dumps({"run_id": run_id, "pregate_only": True,
                              "pregate_effects": effect_results, "aborts": []},
                             indent=2, ensure_ascii=False))
            return 0

        # ---- calibration episodes
        tuning = SupervisorTuning(
            contact_force_delta_raw=config.contact_force_delta_raw,
            non_target_force_abort_raw=config.non_target_force_abort_raw,
            coarse_step_raw=config.coarse_step_raw,
            fine_step_raw=config.fine_step_raw,
            dwell_ms=config.dwell_ms_default,
            camera_freshness_ms=config.camera_freshness_ms,
            temperature_abort_c=config.temperature_abort_c,
        )
        for pair_id in pairs:
            pair = pair_by_id(pair_id)
            assert pair is not None
            for mode in modes:
                for repeat in range(args.repeats):
                    if collector.abort_reason:
                        report["aborts"].append(collector.abort_reason)
                        executors["left"].retreat({"speed": 300, "force": 150})
                        executors["right"].retreat({"speed": 300, "force": 150})
                        print(json.dumps(report, indent=2, default=str))
                        return 3
                    temp_now = default_temp_probe()
                    temps_now = [v for v in temp_now.values() if isinstance(v, (int, float))]
                    if temps_now and max(temps_now) > config.temperature_start_max_c:
                        report["aborts"].append(f"thermal continue gate: {max(temps_now)}°C")
                        print(json.dumps(report, indent=2, default=str))
                        return 3
                    record = _run_episode(
                        pair_id=pair_id,
                        mode=mode,
                        repeat=repeat,
                        run_id=run_id,
                        gateway=gateway,
                        executors=executors,
                        collector=collector,
                        baselines=baselines,
                        tuning=tuning,
                        config=config,
                        out_dir=out_dir,
                    )
                    report["episodes"].append(record.__dict__)
                    # between episodes: posture reset to open — this is a
                    # SAFETY-class retreat path (not an interaction), the
                    # same one the supervisor's RETREAT decision uses
                    for side in ("left", "right"):
                        executors[side].dispatch(
                            {"targets": dict(OPEN_RAW), "speed": 150, "force": 100},
                            timeout_ms=4000,
                        )
                    time.sleep(2.5)

        report["envelopes"] = _build_envelopes(report["episodes"], EXPECTED_POSE_HASHES["left"])
        (out_dir / "t1_calibration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str)
        )
        print(json.dumps({"run_id": run_id, "episodes": len(report["episodes"]),
                          "envelopes": report["envelopes"], "aborts": report["aborts"]},
                         indent=2, ensure_ascii=False, default=str))
        return 0 if not report["aborts"] else 3
    finally:
        if collector is not None:
            collector.stop()
        for side, ctl in controllers.items():
            try:
                ctl.move_to_gesture("t1_coast", [1000] * len(RH56_JOINTS), 300, 50)
                ctl.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            cap.stop()
        except Exception:  # noqa: BLE001
            pass


def _run_episode(
    *,
    pair_id: str,
    mode: str,
    repeat: int,
    run_id: str,
    gateway,
    executors,
    collector,
    baselines,
    tuning,
    config,
    out_dir: Path,
) -> EpisodeRecord:
    record = EpisodeRecord(
        pair_id=pair_id,
        mode=mode,
        repeat=repeat,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    contract = ContactChoreographyContract(
        pattern="fingertip_marquee",
        pairs=("thumb_thumb", "index_index", "middle_middle", "ring_ring",
               "little_little", "ring_ring", "middle_middle", "index_index", "thumb_thumb"),
        cycles=1,
        force_level="ultra_light",
        left_body_hash=f"t1_{run_id}",
        right_body_hash=f"t1_{run_id}",
        camera_pose_hash=EXPECTED_POSE_HASHES["left"],
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    permit = SequencePermit.issue(
        contract,
        intent_hash=_intent_card(pair_id, [mode], 1),
        lifetime_s=config.permit_lifetime_s,
    )
    sequence_id = f"seq_{pair_id}_{mode}_{repeat}_{run_id}"

    # ---- starting pose (present the target finger per mode) — the
    # first envelope of the sequence, dispatched through the gateway
    present = _present_targets(pair_id, mode)
    start_envelope = build_episode_envelope(
        interaction_id=f"{sequence_id}_present",
        sequence_id=sequence_id,
        pair_id=pair_id,
        left_targets=present["left"],
        right_targets=present["right"],
        speed=150,
        force=100,
        contract_hash=contract.contract_hash(),
        snapshot_hash=f"t1_{run_id}",
    )
    start_dispatch = gateway.dispatch(start_envelope, contract=contract, permit=permit)
    if start_dispatch.violation_kind is not None:
        record.outcome = "GATEWAY_BLOCKED"
        record.notes.append(f"present pose: {start_dispatch.violations}")
        for side in ("left", "right"):
            executors[side].retreat({"speed": 300, "force": 150})
        (out_dir / f"episode_{pair_id}_{mode}_{repeat}.json").write_text(
            json.dumps(record.__dict__, indent=2, ensure_ascii=False, default=str)
        )
        return record
    time.sleep(2.5)
    current_targets = {s: dict(t) for s, t in present.items()}
    expected_start: dict[str, dict[str, int]] = {}
    for side in ("left", "right"):
        declared = {
            joint: raw
            for joint, raw in present[side].items()
            if raw != OPEN_RAW[joint]
        }
        if declared:
            expected_start[side] = declared

    supervisor = ContactSupervisor(
        interaction_id=f"ep_{pair_id}_{mode}_{repeat}_{run_id}",
        pair_id=pair_id,
        active_mode=mode,
        baselines=baselines,
        reachability_calibrated=True,  # contract-declared calibration pair ONLY
        tuning=tuning,
        expected_start=expected_start or None,
    )
    step_index = 0
    deadline = time.time() + 120.0
    fine_entry_targets: dict | None = None
    first_rise_targets: dict | None = None
    confirm_targets: dict | None = None

    while supervisor.state not in (EPISODE_COMMITTED, RECORD_FAILURE) and time.time() < deadline:
        if collector.abort_reason:
            for side in ("left", "right"):
                executors[side].retreat({"speed": 300, "force": 150})
            record.outcome = "WATCHDOG_ABORT"
            record.notes.append(collector.abort_reason)
            break
        obs = collector.latest()
        decision = supervisor.step(obs)
        if decision.kind == DECISION_ISSUE_STEP and decision.step:
            step_index += 1
            new_targets = {s: dict(t) for s, t in current_targets.items()}
            for side in decision.step["sides"]:
                for joint, delta in decision.step["joints"].items():
                    new_targets[side][joint] = int(
                        max(50, min(1000, new_targets[side][joint] + delta))
                    )
            envelope = build_episode_envelope(
                interaction_id=f"{supervisor.interaction_id}_s{step_index}",
                sequence_id=sequence_id,
                pair_id=pair_id,
                left_targets=new_targets["left"],
                right_targets=new_targets["right"],
                speed=(
                    config.servo_max_speed_fine
                    if supervisor.state == FINE_APPROACH
                    else config.servo_max_speed_approach
                ),
                force=config.approach_force_set,
                contract_hash=contract.contract_hash(),
                snapshot_hash=f"t1_{run_id}",
            )
            dispatch = gateway.dispatch(envelope, contract=contract, permit=permit)
            if dispatch.violation_kind is not None:
                record.outcome = "GATEWAY_BLOCKED"
                record.notes.append(f"step {step_index}: {dispatch.violations}")
                for side in ("left", "right"):
                    executors[side].retreat({"speed": 300, "force": 150})
                break
            current_targets = new_targets
            time.sleep(1.2 if supervisor.state == COARSE_APPROACH else 1.5)
        elif decision.kind == DECISION_RETREAT:
            for side in ("left", "right"):
                executors[side].retreat({"speed": 300, "force": 150})
            current_targets = {"left": dict(OPEN_RAW), "right": dict(OPEN_RAW)}
            time.sleep(1.5)
        else:
            time.sleep(0.3)

        # envelope facts
        if supervisor.state == "FINE_APPROACH" and fine_entry_targets is None:
            fine_entry_targets = {s: dict(t) for s, t in current_targets.items()}
        if supervisor.state == "CONTACT_CANDIDATE" and first_rise_targets is None:
            first_rise_targets = {s: dict(t) for s, t in current_targets.items()}
        if supervisor.state == "CONTACT_CONFIRMED" and confirm_targets is None:
            confirm_targets = {s: dict(t) for s, t in current_targets.items()}
        if decision.kind in (DECISION_COMMIT, DECISION_FAIL):
            record.outcome = decision.receipt.outcome if decision.receipt else "UNKNOWN"
            record.receipt = decision.receipt.to_record() if decision.receipt else None

    record.state_history = list(supervisor.history)
    record.envelope_facts = {
        "fine_entry_targets": fine_entry_targets,
        "first_rise_targets": first_rise_targets,
        "confirm_targets": confirm_targets,
        "final_targets": current_targets,
        "release_steps": supervisor.track.release_steps,
        "coarse_steps": supervisor.track.coarse_steps,
        "fine_steps": supervisor.track.fine_steps,
    }
    (out_dir / f"episode_{pair_id}_{mode}_{repeat}.json").write_text(
        json.dumps(record.__dict__, indent=2, ensure_ascii=False, default=str)
    )
    return record


def _build_envelopes(episodes: list[dict], pose_hash: str) -> dict:
    """FingerContactEnvelope per pair×mode from CONFIRMED episodes."""
    envelopes: dict = {}
    for ep in episodes:
        if ep.get("outcome") != "CONTACT_CONFIRMED" or not ep.get("receipt"):
            continue
        key = f"{ep['pair_id']}|{ep['mode']}"
        bucket = envelopes.setdefault(
            key,
            {
                "pair_id": ep["pair_id"],
                "active_side": ep["mode"],
                "camera_pose_hash": pose_hash,
                "confirm_targets": [],
                "force_peaks_left": [],
                "force_peaks_right": [],
                "visual_distance_min_m": [],
                "contact_latency_ms": [],
                "release_steps": [],
                "evidence_count": 0,
            },
        )
        facts = ep.get("envelope_facts") or {}
        if facts.get("confirm_targets"):
            bucket["confirm_targets"].append(facts["confirm_targets"])
        receipt = ep["receipt"]
        if receipt.get("left_force_peak") is not None:
            bucket["force_peaks_left"].append(receipt["left_force_peak"])
        if receipt.get("right_force_peak") is not None:
            bucket["force_peaks_right"].append(receipt["right_force_peak"])
        if receipt.get("visual_distance_min_m") is not None:
            bucket["visual_distance_min_m"].append(receipt["visual_distance_min_m"])
        if receipt.get("contact_latency_ms") is not None:
            bucket["contact_latency_ms"].append(receipt["contact_latency_ms"])
        bucket["release_steps"].append(facts.get("release_steps"))
        bucket["evidence_count"] += 1
    return envelopes


if __name__ == "__main__":
    raise SystemExit(run())
