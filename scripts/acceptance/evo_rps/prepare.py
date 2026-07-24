#!/usr/bin/env python3
"""Evo-RPS acceptance: prepare (namespace + preflight + evidence init)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from rosclaw.evolution.hardware.cli import cmd_acceptance_evo_rps_prepare  # noqa: E402


class _Args:
    config = None
    dev_allow_mock = "--dev-allow-mock" in sys.argv


if __name__ == "__main__":
    raise SystemExit(cmd_acceptance_evo_rps_prepare(_Args()))
