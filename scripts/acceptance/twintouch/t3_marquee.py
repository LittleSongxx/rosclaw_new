#!/usr/bin/env python3
"""TwinTouch T3: fingertip marquee baseline (v4 §11 T3, PR-TT-6).

The canonical 9-pair sequence (thumb → little → back), forward and
reverse, each pair a full supervisor-driven contact episode, chained at
the PRE-APPROACH level: pair K's clearance flows into pair K+1's
present WITHOUT returning to open (the marquee rhythm).

BASELINE discipline (v4 §11 T3): fixed envelopes, fixed approach step,
fixed timing — NO memory, NO self residual, NO auto-compensation.  An
uncalibrated pair is refused by the supervisor (§12.2): with only
index_index calibrated, the default sequence is the index cell x5 —
the full sequence enables pair by pair as their T1 envelopes land.

Pair transitions: after pair K's CLEARANCE_VERIFIED the runner
dispatches pair K+1's present pose THROUGH THE GATEWAY under the same
Sequence Permit (one permit per sequence, many envelopes — TT-2 §18),
re-baselines at the new pre-approach pose, and starts the next
episode.  The previous target finger returns toward open (out of the
corridor).  Any anomaly completes the recovery spine and aborts the
sequence honestly — never continue the marquee past a failed pair.

Metrics (v4 §11 T3): pair contact SR, sequence completion SR, wrong
finger rate, unintended contact rate, force overshoot, contact
latency, release failures, cycle time — plus a live state file per
cycle for the Dashboard's first marquee page.

Safety: identical to the T1 runner (watchdog, gates, thermal,
one-pair-at-a-time).  Every motion through the gateway.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO_SRC = str(Path(__file__).resolve().parents[2] / "src")
sys.path.insert(0, REPO_SRC)
sys.path.insert(
    0,
    os.environ.get(
        "ROSCLAW_RH56_RUNTIME_SRC",
        "/home/nvidia/workspace/rosclaw_rh56_real/rosclaw-rh56-runtime/src",
    ),
)
sys.path.insert(
    0,
    os.environ.get(
        "ROSCLAW_RH56_RPS_SRC",
        "/home/nvidia/workspace/rosclaw/rosclaw_test/examples/rh56_rps/src",
    ),
)

# reuse the proven T1 machinery as a library (registered for dataclasses)
_spec = importlib.util.spec_from_file_location(
    "t1c", str(Path(__file__).resolve().parent / "t1_calibration.py")
)
t1 = importlib.util.module_from_spec(_spec)
sys.modules["t1c"] = t1
_spec.loader.exec_module(t1)

import pyrealsense2 as rs  # noqa: E402

from rosclaw.evolution.hardware.camera import D435iCapture  # noqa: E402
from rosclaw.evolution.hardware.thermal import default_temp_probe  # noqa: E402
from rosclaw.perception.handcam import CameraPoseContract, intrinsics_from_pyrealsense  # noqa: E402
from rosclaw.twintouch.choreography import (  # noqa: E402
    CANONICAL_MARQUEE_PAIRS,
    ContactChoreographyContract,
    SequencePermit,
)
from rosclaw.twintouch.config import TwinTouchConfig  # noqa: E402
from rosclaw.twintouch.gateway import BimanualActionGateway, LeaseRegistry  # noqa: E402
from rosclaw.twintouch.pairs import RH56_JOINTS, is_valid_pair_id  # noqa: E402
from rosclaw.twintouch.supervisor import (  # noqa: E402
    EPISODE_COMMITTED,
    RECORD_FAILURE,
    ContactSupervisor,
    SupervisorTuning,
)

OUT_ROOT = Path("/home/nvidia/.rosclaw/acceptance/twintouch/t3")
EXPECTED_POSE_HASHES = dict(t1.EXPECTED_POSE_HASHES)


def _present_for(pair_id: str, mode: str) -> dict[str, dict[str, int]]:
    return t1._present_targets(pair_id, mode)


def run_marquee(
    *,
    pairs: list[str],
    mode: str,
    cycles: int,
    pregate_only: bool = False,
) -> int:
    config = TwinTouchConfig.load()
    probe = default_temp_probe()
    temps = [v for v in probe.values() if isinstance(v, (int, float))]
    if temps and max(temps) > config.temperature_start_max_c:
        print(json.dumps({"ok": False, "blocked": f"start {max(temps)}°C > gate"}))
        return 1
    for pair_id in pairs:
        if not is_valid_pair_id(pair_id):
            print(json.dumps({"ok": False, "blocked": f"{pair_id} not a valid pair"}))
            return 1

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "marquee_state.jsonl"
    report: dict = {
        "run_id": run_id,
        "pairs": pairs,
        "mode": mode,
        "cycles": cycles,
        "start_temps": probe,
        "cells": [],
        "metrics": {},
        "aborts": [],
    }

    controllers: dict = {}
    cap = D435iCapture()
    collector = None
    try:
        for label, port in (("right", t1.RIGHT_PORT), ("left", t1.LEFT_PORT)):
            controllers[label], slave = t1.open_probed_controller(port, label)
            report.setdefault("slave_ids", {})[label] = slave

        cap.start(serial=t1.CAMERA_SERIAL)
        intr = intrinsics_from_pyrealsense(cap._pipeline.get_active_profile(), rs.stream.depth)
        contracts = {
            side: CameraPoseContract(
                camera_pose_id=f"twintouch_t0_{side}", camera_id="d435i",
                intrinsics=intr, roi=roi,
            )
            for side, roi in (("left", t1.LEFT_ROI), ("right", t1.RIGHT_ROI))
        }
        for side, contract in contracts.items():
            if contract.camera_pose_hash != EXPECTED_POSE_HASHES[side]:
                report["aborts"].append(f"LAYOUT_CHANGED on {side} — T0 evidence void")
                print(json.dumps(report, indent=2, default=str))
                return 2

        # hands open + session baselines (open pose, for the watchdog pre-present)
        for _side, ctl in controllers.items():
            tel = ctl.read_telemetry()
            angles = tel.angle_actual or {}
            if any((angles.get(j) or 0) < 900 for j in ("little", "ring", "middle", "index")):
                ctl.move_to_gesture("t3_open", [1000] * len(RH56_JOINTS), 150, 150)
        time.sleep(2.5)
        from rosclaw.twintouch.supervisor import ForceBaseline

        session_baselines: dict[str, ForceBaseline] = {}
        for _side, ctl in controllers.items():
            samples = []
            for _ in range(8):
                tel = ctl.read_telemetry()
                samples.append({k: (None if v is None else float(v)) for k, v in (tel.force_act or {}).items()})
                time.sleep(0.2)
            session_baselines[side] = ForceBaseline.capture(side, samples, min_samples=6)

        executors = {
            side: t1.Rh56BodyExecutor(side, ctl, config.servo_max_speed_approach, config.approach_force_set)
            for side, ctl in controllers.items()
        }
        collector = t1.ObservationCollector(controllers, cap, contracts, intr, session_baselines, config)
        collector.start()
        if not collector.wait_warm(timeout_s=12.0):
            report["aborts"].append("collector never warmed")
            print(json.dumps(report, indent=2, default=str))
            return 2
        gateway = BimanualActionGateway(
            executors=executors,
            leases=LeaseRegistry(t1.BODY_IDS),
            probe=t1.LiveProbe(collector),
            camera_freshness_ms=config.camera_freshness_ms,
        )

        # one choreography contract for the WHOLE marquee (v4 §18): the
        # canonical sequence is the APPROVAL SCOPE; execution is gated
        # pair-by-pair by T1 calibration — uncalibrated pairs are
        # recorded as skipped cells, never silently dropped and never
        # attempted.  Sequence Permits are re-issued per pair/cycle
        # cell (fresh intent hash per cell) so every permit stays
        # within config.permit_lifetime_s — a hard, never-raisable
        # limit the runner must never override.
        snapshot_hashes = {"left": f"t3_{run_id}_left", "right": f"t3_{run_id}_right"}
        contract = ContactChoreographyContract(
            pattern="fingertip_marquee",
            pairs=tuple(CANONICAL_MARQUEE_PAIRS),
            cycles=cycles,
            force_level="ultra_light",
            left_body_hash=snapshot_hashes["left"],
            right_body_hash=snapshot_hashes["right"],
            camera_pose_hash=EXPECTED_POSE_HASHES["left"],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        violations = contract.validate()
        if violations:
            report["aborts"].append(f"marquee contract invalid: {violations}")
            print(json.dumps(report, indent=2, default=str))
            return 2

        tuning = SupervisorTuning(
            contact_force_delta_raw=35.0,
            non_target_force_abort_raw=config.non_target_force_abort_raw,
            visual_conflict_distance_m=0.10,
            coarse_step_raw=config.coarse_step_raw,
            fine_step_raw=config.fine_step_raw,
            max_fine_steps=16,
            coarse_to_fine_distance_m=0.10,
            max_release_steps=8,
            dwell_ms=config.dwell_ms_default,
            camera_freshness_ms=config.camera_freshness_ms,
            temperature_abort_c=config.temperature_abort_c,
        )

        def _emit_state(cell: dict) -> None:
            with state_path.open("a") as fh:
                fh.write(json.dumps(cell, ensure_ascii=False, default=str) + "\n")

        if pregate_only:
            report["pregate_only"] = True
            print(json.dumps(report, indent=2, default=str))
            return 0

        current_targets = {"left": dict(t1.OPEN_RAW), "right": dict(t1.OPEN_RAW)}
        sequence_ok = True
        cell_index = 0
        t_sequence_start = time.time()
        # pairs the supervisor may attempt TODAY (T1-calibrated only);
        # every other canonical pair in the execution subset is an
        # explicit skipped cell, never silently dropped
        calibrated_pairs = {"index_index"}
        for cycle in range(cycles):
            if not sequence_ok:
                break
            for pair_id in pairs:
                if collector.abort_reason:
                    report["aborts"].append(collector.abort_reason)
                    sequence_ok = False
                    break
                temp_now = default_temp_probe()
                temps_now = [v for v in temp_now.values() if isinstance(v, (int, float))]
                if temps_now and max(temps_now) > config.temperature_start_max_c:
                    report["aborts"].append(f"thermal gate: {max(temps_now)}°C")
                    sequence_ok = False
                    break
                cell: dict = {
                    "cell": cell_index, "cycle": cycle, "pair_id": pair_id, "mode": mode,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                cell_index += 1
                if pair_id not in calibrated_pairs:
                    cell["outcome"] = "SKIPPED_UNCALIBRATED"
                    cell["notes"] = ["pair has no T1 envelope — skipped, never attempted"]
                    _emit_state(cell)
                    report["cells"].append(cell)
                    continue
                # fresh Sequence Permit per pair/cycle cell: a fresh
                # intent hash per cell and a lifetime that never
                # exceeds config.permit_lifetime_s (hard limit)
                permit = SequencePermit.issue(
                    contract,
                    intent_hash=(
                        f"t3_marquee pair={pair_id} cycle={cycle} cell={cell['cell']} "
                        f"mode={mode} authorization=v4-doc-§11-T3 "
                        "operator=user-via-standing-task-doc"
                    ),
                    lifetime_s=config.permit_lifetime_s,
                )
                sequence_ok = _run_cell(
                    cell=cell,
                    run_id=run_id,
                    gateway=gateway,
                    executors=executors,
                    collector=collector,
                    controllers=controllers,
                    contract=contract,
                    permit=permit,
                    tuning=tuning,
                    config=config,
                    snapshot_hashes=snapshot_hashes,
                    current_targets=current_targets,
                    emit=_emit_state,
                )
                report["cells"].append({k: v for k, v in cell.items() if k != "history"})
                if not sequence_ok:
                    break

        report["metrics"] = _metrics(report["cells"], time.time() - t_sequence_start)
        (out_dir / "t3_marquee_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str)
        )
        print(json.dumps({"run_id": run_id, "cells": len(report["cells"]),
                          "metrics": report["metrics"], "aborts": report["aborts"]},
                         indent=2, ensure_ascii=False, default=str))
        return 0 if not report["aborts"] else 3
    finally:
        if collector is not None:
            collector.stop()
        for _side, ctl in controllers.items():
            try:
                ctl.move_to_gesture("t3_coast", [1000] * len(RH56_JOINTS), 300, 50)
                ctl.close()
            except Exception:  # noqa: BLE001
                pass
        with contextlib.suppress(Exception):
            cap.stop()


def _run_cell(
    *,
    cell: dict,
    run_id: str,
    gateway,
    executors,
    collector,
    controllers,
    contract,
    permit,
    tuning,
    config,
    snapshot_hashes,
    current_targets,
    emit,
) -> bool:
    """One marquee cell: present pair → supervisor episode → clearance.
    Returns True iff the sequence may continue."""
    pair_id = cell["pair_id"]
    mode = cell["mode"]
    present = t1._present_targets(pair_id, mode)
    # pair transition: previous non-target fingers are already open;
    # dispatch BOTH hands' present through the gateway under the
    # sequence permit (staggered: passive first for single-side modes)
    if mode == "mutual":
        stages = ["left", "right"]
    else:
        passive = "right" if mode == "active_passive" else "left"
        active = "left" if mode == "active_passive" else "right"
        stages = [passive, active]
    staged = {"left": dict(current_targets["left"]), "right": dict(current_targets["right"])}
    finger = pair_id.split("_")[0]
    envelope_index = 0
    for side in stages:
        side_present = present[side]
        overlays = [side_present]
        if (
            finger != "thumb"
            and t1.HAND_TUCK_POLICY[side]
            and side_present.get("thumb") == t1.THUMB_TUCK_RAW
        ):
            overlays = [{"thumb": t1.THUMB_TUCK_RAW}, side_present]
        for overlay in overlays:
            if collector.abort_reason:
                cell["outcome"] = "WATCHDOG_ABORT"
                cell["notes"] = [collector.abort_reason]
                for s in ("left", "right"):
                    executors[s].retreat({"speed": 300, "force": 150})
                emit(dict(cell))
                return False
            staged[side].update(overlay)
            # previous target fingers of this side return toward open
            for other in ("index", "middle", "ring", "little"):
                if other != finger and staged[side][other] != t1.RETRACT_RAW:
                    staged[side][other] = t1.RETRACT_RAW
            envelope = t1.build_episode_envelope(
                interaction_id=f"cell{cell['cell']}_present{envelope_index}",
                sequence_id=f"seq_{run_id}",
                pair_id=pair_id,
                left_targets=staged["left"],
                right_targets=staged["right"],
                speed=150,
                force=100,
                contract_hash=contract.contract_hash(),
                snapshot_hashes=snapshot_hashes,
            )
            envelope_index += 1
            dispatch = gateway.dispatch(envelope, contract=contract, permit=permit)
            if dispatch.violation_kind is not None:
                cell["outcome"] = "GATEWAY_BLOCKED"
                cell["notes"] = [f"present: {dispatch.violations}"]
                for s in ("left", "right"):
                    executors[s].retreat({"speed": 300, "force": 150})
                emit(dict(cell))
                return False
            time.sleep(1.6 if envelope_index < 2 else 2.2)
    # thread the staged pose back into the SHARED targets dict — the
    # next cell's transition starts from this pose, not from OPEN
    current_targets.update(staged)

    # per-cell re-baseline at the pre-approach pose
    episode_baselines: dict = {}
    from rosclaw.twintouch.supervisor import ForceBaseline

    for side, ctl in controllers.items():
        samples = []
        for _ in range(7):
            tel = ctl.read_telemetry()
            samples.append({k: (None if v is None else float(v)) for k, v in (tel.force_act or {}).items()})
            time.sleep(0.15)
        episode_baselines[side] = ForceBaseline.capture(side, samples, min_samples=5)
    collector.set_baselines(episode_baselines)
    collector.set_watch_target(finger)

    expected_start: dict[str, dict[str, int]] = {}
    for side in ("left", "right"):
        declared = {j: r for j, r in present[side].items() if r != t1.OPEN_RAW[j]}
        if declared:
            expected_start[side] = declared
    supervisor = ContactSupervisor(
        interaction_id=f"cell{cell['cell']}_{pair_id}_{run_id}",
        pair_id=pair_id,
        active_mode=mode,
        baselines=episode_baselines,
        reachability_calibrated=True,  # pair envelopes T1-calibrated only
        tuning=tuning,
        expected_start=expected_start or None,
    )
    step_index = 0
    deadline = time.time() + 240.0
    t_cell_start = time.time()
    cell["history"] = []
    while supervisor.state not in (EPISODE_COMMITTED, RECORD_FAILURE) and time.time() < deadline:
        if collector.abort_reason:
            # thermal/watchdog abort unloads the hands fully — never
            # hold a flexed pose (mirror of the episode-boundary path)
            for side in ("left", "right"):
                executors[side].retreat({"speed": 300, "force": 150})
            current_targets.update({s: dict(t1.OPEN_RAW) for s in ("left", "right")})
            cell["outcome"] = "WATCHDOG_ABORT"
            cell["notes"] = [collector.abort_reason]
            break
        obs = collector.latest()
        decision = supervisor.step(obs)
        if decision.kind == "ISSUE_STEP" and decision.step:
            step_index += 1
            new_targets = {s: dict(t) for s, t in current_targets.items()}
            for side in decision.step["sides"]:
                for joint, delta in decision.step["joints"].items():
                    new_val = new_targets[side][joint] + delta
                    if delta > 0:
                        new_val = min(new_val, present[side].get(joint, 1000) + 100)
                    new_targets[side][joint] = int(max(50, min(1000, new_val)))
            envelope = t1.build_episode_envelope(
                interaction_id=f"cell{cell['cell']}_s{step_index}",
                sequence_id=f"seq_{run_id}",
                pair_id=pair_id,
                left_targets=new_targets["left"],
                right_targets=new_targets["right"],
                speed=config.servo_max_speed_fine,
                force=config.approach_force_set,
                contract_hash=contract.contract_hash(),
                snapshot_hashes=snapshot_hashes,
            )
            dispatch = gateway.dispatch(envelope, contract=contract, permit=permit)
            if dispatch.violation_kind is not None:
                cell["outcome"] = "GATEWAY_BLOCKED"
                cell["notes"] = [f"step {step_index}: {dispatch.violations}"]
                for side in ("left", "right"):
                    executors[side].dispatch(
                        {"targets": present[side], "speed": 300, "force": 100}, timeout_ms=4000
                    )
                break
            current_targets.update(new_targets)
            time.sleep(1.5)
        elif decision.kind == "RETREAT":
            for side in ("left", "right"):
                executors[side].dispatch(
                    {"targets": present[side], "speed": 300, "force": 100}, timeout_ms=4000
                )
            current_targets.update({s: dict(t) for s, t in present.items()})
            time.sleep(1.5)
        else:
            time.sleep(0.3)
        if decision.kind in ("COMMIT", "FAIL"):
            cell["outcome"] = decision.receipt.outcome if decision.receipt else "UNKNOWN"
            cell["receipt"] = decision.receipt.to_record() if decision.receipt else None
    if "outcome" not in cell or cell["outcome"] is None:
        cell["outcome"] = supervisor.track.anomaly or "DEADLINE_EXCEEDED"
        if supervisor.track.anomaly:
            cell["notes"] = [supervisor.track.anomaly_detail]
    cell["history"] = list(supervisor.history)
    cell["cycle_time_s"] = round(time.time() - t_cell_start, 1)
    cell["force_peaks"] = {
        "left": supervisor.track.left_force_peak,
        "right": supervisor.track.right_force_peak,
    }
    emit({k: v for k, v in cell.items() if k != "receipt"})
    return cell["outcome"] == "CONTACT_CONFIRMED"


def _metrics(cells: list[dict], sequence_time_s: float) -> dict:
    total = len(cells)
    confirmed = sum(1 for c in cells if c.get("outcome") == "CONTACT_CONFIRMED")
    wrong = sum(1 for c in cells if c.get("outcome") == "WRONG_FINGER_CONTACT")
    unintended = sum(1 for c in cells if c.get("outcome") == "UNINTENDED_CONTACT")
    release_failed = sum(1 for c in cells if c.get("outcome") == "RELEASE_FAILED")
    latencies = [
        (c.get("receipt") or {}).get("contact_latency_ms")
        for c in cells
        if (c.get("receipt") or {}).get("contact_latency_ms") is not None
    ]
    peaks_l = [(c.get("force_peaks") or {}).get("left") for c in cells]
    peaks_r = [(c.get("force_peaks") or {}).get("right") for c in cells]
    overshoot = max(
        [abs(p) for p in peaks_l + peaks_r if isinstance(p, (int, float))] or [0]
    )
    return {
        "cells": total,
        "pair_contact_sr": round(confirmed / total, 3) if total else 0.0,
        "sequence_completed": confirmed == total and total > 0,
        "wrong_finger_rate": round(wrong / total, 3) if total else 0.0,
        "unintended_contact_rate": round(unintended / total, 3) if total else 0.0,
        "release_failure_count": release_failed,
        "force_overshoot_max_raw": overshoot,
        "contact_latency_ms_median": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "cycle_times_s": [c.get("cycle_time_s") for c in cells],
        "sequence_time_s": round(sequence_time_s, 1),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        default="index_index",
        help="comma list; full canonical sequence only when all pairs are T1-calibrated",
    )
    parser.add_argument("--mode", default="mutual")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--pregate-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_marquee(
            pairs=[p.strip() for p in args.pairs.split(",") if p.strip()],
            mode=args.mode,
            cycles=args.cycles,
            pregate_only=args.pregate_only,
        )
    )
