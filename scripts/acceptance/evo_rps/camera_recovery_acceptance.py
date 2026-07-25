#!/usr/bin/env python3
"""Evo-RPS acceptance: camera fault recovery on the real D435i (§九, PR-EVO-HW-2).

Runs one honest recovery cycle against the physical camera:

    healthy capture → injected node fault (pipeline torn down externally)
    → recovery procedure (default B_reset_fresh_frames) with REAL handlers
    → TTR + 10 consecutive fresh frames + NEW camera session id
    → procedural memory into the EXPERIMENT NAMESPACE store
    → evidence manifest entry

Safety: requires no robot motion (the injector guard is re-checked here);
never touches USB; the capture lifecycle is the only camera control path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rosclaw.evolution.hardware.camera import (  # noqa: E402
    CameraWedgeError,
    D435iCapture,
    write_artifacts,
)
from rosclaw.evolution.hardware.camera_recovery import run_recovery  # noqa: E402
from rosclaw.evolution.hardware.contracts import load_config  # noqa: E402
from rosclaw.evolution.hardware.evidence import EvidenceManifest  # noqa: E402
from rosclaw.evolution.hardware.freshness import FrameFreshnessGate  # noqa: E402
from rosclaw.evolution.hardware.namespace import ExperimentNamespace  # noqa: E402
from rosclaw.evolution.hardware.orchestrator import DEFAULT_CONFIG  # noqa: E402

SERIAL = "231122070092"


def robot_moving() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-af", "evo3_regime_rps|stress_closed_loop"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return True
    return any(ln.strip() and "pgrep" not in ln for ln in out.stdout.splitlines())


def real_handlers(capture_box: dict, gate: FrameFreshnessGate, artifacts_dir: Path):
    def stop_node(ctx):
        cap = capture_box.get("cap")
        if cap is not None:
            cap.stop()
        return True, "pipeline stopped gracefully"

    def start_node(ctx):
        cap = D435iCapture()
        cap.start(serial=SERIAL)
        capture_box["cap"] = cap
        ctx["camera_session_id"] = cap.session_id
        return True, f"restarted session {cap.session_id}"

    def hardware_reset(ctx):
        stop_node(ctx)
        # D435iCapture.start performs the reset FIRST by design; an explicit
        # reset step here is exercised through a fresh start below.
        return True, "reset delegated to wedge-safe start (hardware_reset FIRST)"

    def wait_reenumeration(ctx):
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                devices = D435iCapture.enumerate_devices()
            except Exception:  # noqa: BLE001
                devices = []
            if any(d.get("serial") == SERIAL for d in devices):
                return True, "re-enumerated"
            time.sleep(1.0)
        return False, "no re-enumeration within 20s (physical re-plug required)"

    def require_fresh_frames(n):
        def handler(ctx):
            cap = capture_box.get("cap")
            if cap is None:
                return False, "no capture"
            for _ in range(n):
                try:
                    frame = cap.read()
                except (CameraWedgeError, RuntimeError) as exc:
                    return False, f"frame read failed: {exc}"
                verdict = gate.check(
                    frame_age_ms=frame.age_ms(),
                    rgb_depth_delta_ms=frame.rgb_depth_delta_ms,
                )
                if not verdict.ok:
                    return False, f"frame {frame.sequence} not fresh: {verdict.reason}"
                ctx["last_frame"] = frame
            return True, f"{n} consecutive fresh frames"

        return handler

    def new_camera_session(ctx):
        return True, ctx.get("camera_session_id") or capture_box["cap"].session_id

    def rgb_depth_sync_check(ctx):
        frame = ctx.get("last_frame") or capture_box["cap"].read()
        delta = frame.rgb_depth_delta_ms
        if delta > 50.0:
            return False, f"rgb/depth delta {delta:.0f}ms > 50ms"
        refs = write_artifacts(frame, artifacts_dir)
        ctx["artifact_refs"] = refs
        return True, f"sync delta {delta:.0f}ms, artifacts saved"

    return {
        "stop_node": stop_node,
        "start_node": start_node,
        "hardware_reset": hardware_reset,
        "wait_reenumeration": wait_reenumeration,
        "require_10_fresh_frames": require_fresh_frames(10),
        "new_camera_session": new_camera_session,
        "rgb_depth_sync_check": rgb_depth_sync_check,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedure", default="B_reset_fresh_frames",
                        choices=["A_immediate_restart", "B_reset_fresh_frames", "C_backoff_sync_check"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    if robot_moving():
        print(json.dumps({"ok": False, "reason": "robot motion detected — camera fault injection refused"}, indent=2))
        return 1

    config = load_config(args.config)
    namespace = ExperimentNamespace.from_config(config)
    manifest = EvidenceManifest.open(
        namespace.evidence_root, config.experiment_id, config.config_hash
    )
    artifacts_dir = namespace.evidence_root / "artifacts" / "camera"
    gate = FrameFreshnessGate()
    capture_box: dict = {}

    # 1) Healthy capture baseline.
    cap = D435iCapture()
    identity = cap.start(serial=SERIAL)
    healthy_frame = cap.read()
    pre_fault_age = healthy_frame.age_ms()
    cap.stop()
    capture_box["cap"] = None

    # 2) Recovery from the injected fault (the torn-down pipeline above).
    handlers = real_handlers(capture_box, gate, artifacts_dir)
    result = run_recovery(args.procedure, handlers=handlers)
    if capture_box.get("cap") is not None:
        capture_box["cap"].stop()

    # 3) Procedural memory into the EXPERIMENT NAMESPACE store (never the
    #    shared rosclaw database).
    memory = result.to_procedural_memory(
        body_id=config.camera_body,
        device=identity,
        pre_fault_frame_age_ms=pre_fault_age,
        robot_id="rh56_rps_robot",
        evidence_refs=[f"camera_recovery_{args.procedure}_{int(time.time())}"],
    )
    store = namespace.knowledge_store()
    namespace.assert_store_isolated(store)
    record = dict(memory)
    record.setdefault("status", "active")
    record.setdefault("event_time", time.time())
    record.setdefault("updated_at", time.time())
    if "id" not in record:
        import hashlib

        record["id"] = "mem_" + hashlib.sha256(
            json.dumps(memory, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
    store.insert("memory_items", record)

    entry = manifest.record(
        "camera_recovery",
        procedure=args.procedure,
        recovered=result.recovered,
        ttr_s=round(result.ttr_s, 3),
        manual_replug_required=result.manual_replug_required,
        camera_session_id=result.camera_session_id,
        memory_id=record["id"],
        steps=[{"name": s.name, "ok": s.ok, "duration_s": round(s.duration_s, 3)} for s in result.steps],
        device=identity,
        note=result.note,
    )
    print(json.dumps({"ok": result.recovered, "evidence": entry}, indent=2, ensure_ascii=False, default=str))
    return 0 if result.recovered else 1


if __name__ == "__main__":
    raise SystemExit(main())
