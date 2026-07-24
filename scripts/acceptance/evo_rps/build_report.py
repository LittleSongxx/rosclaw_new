#!/usr/bin/env python3
"""Evo-RPS acceptance: evidence manifest → report."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rosclaw.evolution.hardware.cli import cmd_acceptance_evo_rps_report  # noqa: E402


class _Args:
    config = None


if __name__ == "__main__":
    raise SystemExit(cmd_acceptance_evo_rps_report(_Args()))
