from __future__ import annotations

import subprocess
from pathlib import Path

from rosclaw.integrations.lerobot import reference_policy_setup


def test_pip_install_uses_uv_for_minimal_runtime(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(args)
        if args[1:4] == ["-m", "pip", "install"]:
            return subprocess.CompletedProcess(
                args,
                1,
                stdout="",
                stderr="runtime: No module named pip",
            )
        return subprocess.CompletedProcess(args, 0, stdout="installed", stderr="")

    monkeypatch.setattr(reference_policy_setup.subprocess, "run", fake_run)
    monkeypatch.setattr(reference_policy_setup.shutil, "which", lambda name: "/opt/uv")

    result = reference_policy_setup._pip_install(Path("/runtime/python"), "/plugin")

    assert result.returncode == 0
    assert calls == [
        ["/runtime/python", "-m", "pip", "install", "/plugin"],
        ["/opt/uv", "pip", "install", "--python", "/runtime/python", "/plugin"],
    ]
