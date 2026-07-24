#!/usr/bin/env python3
"""Evo-RPS acceptance: per-session metric extraction from the evidence manifest."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rosclaw.evolution.hardware.orchestrator import DEFAULT_CONFIG, orchestrator_for  # noqa: E402

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    orchestrator = orchestrator_for(config_path)
    report = orchestrator.report()
    manifest = orchestrator._open_manifest()
    sessions = manifest.by_kind("baseline_session")
    print(json.dumps({"report": report, "sessions": sessions}, indent=2, ensure_ascii=False, default=str))
