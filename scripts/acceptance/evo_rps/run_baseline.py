#!/usr/bin/env python3
"""Evo-RPS acceptance: baseline sessions (real hardware, strict verify)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rosclaw.evolution.hardware.cli import cmd_acceptance_evo_rps_baseline  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(cmd_acceptance_evo_rps_baseline(args))
