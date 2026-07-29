from __future__ import annotations

import json
from pathlib import Path

from rosclaw.continual import service_validation
from rosclaw.continual.cli import dispatch_continual_argv


def test_continual_services_validate_dispatches_and_reports_result(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    captured = {}

    def fake_validation(**kwargs):
        captured.update(kwargs)
        return {
            "passed": True,
            "checks": {"strict_replay": True},
            "report_hash": "sha256:" + "1" * 64,
        }

    monkeypatch.setattr(service_validation, "run_g1_service_validation", fake_validation)
    output = tmp_path / "report.json"
    exit_code = dispatch_continual_argv(
        [
            "continual",
            "services",
            "validate",
            "--asset-root",
            str(tmp_path / "assets"),
            "--candidate",
            str(tmp_path / "candidate.bin"),
            "--matched-report",
            str(tmp_path / "matched.json"),
            "--state-root",
            str(tmp_path / "state"),
            "--output",
            str(output),
            "--learner-device",
            "cuda:0",
            "--learner-updates",
            "3",
        ]
    )
    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["learner_device"] == "cuda:0"
    assert captured["learner_updates"] == 3
    assert captured["output_path"] == output
    assert printed["passed"] is True


def test_continual_dispatch_leaves_unknown_namespace_for_legacy_cli() -> None:
    assert dispatch_continual_argv(["continual", "unknown"]) is None
