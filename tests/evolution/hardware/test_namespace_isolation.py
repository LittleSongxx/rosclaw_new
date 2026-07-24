"""Namespace isolation tests (PR-EVO-HW-1, 真机自进化v2 §2.7)."""

from __future__ import annotations

import pytest

from rosclaw.evolution.hardware.contracts import load_config
from rosclaw.evolution.hardware.namespace import ExperimentNamespace, NamespaceError

CONFIG_PATH = "configs/acceptance/evo_rps_v1.yaml"


def _config(tmp_path, database="rosclaw_evo_rps_test"):
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as _fh:
        raw = yaml.safe_load(_fh)
    raw["namespace"]["database"] = database
    for key in ("practice_root", "trace_root", "evidence_root"):
        raw["namespace"][key] = str(tmp_path / key)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    return load_config(str(path))


def test_namespace_roots_and_dsn_are_experiment_scoped(tmp_path) -> None:
    config = _config(tmp_path)
    ns = ExperimentNamespace.from_config(config)
    assert ns.database == "rosclaw_evo_rps_test"
    assert ns.database in ns.dsn
    assert "rosclaw_evo_rps_test" in ns.dsn
    assert ns.practice_root == tmp_path / "practice_root"
    assert ns.trace_root == tmp_path / "trace_root"
    assert ns.evidence_root == tmp_path / "evidence_root"


def test_shared_database_is_a_violation(tmp_path) -> None:
    import yaml

    with open(CONFIG_PATH, encoding="utf-8") as _fh:
        raw = yaml.safe_load(_fh)
    raw["namespace"]["database"] = "rosclaw"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))
    with pytest.raises(Exception, match="isolated database"):
        load_config(str(path))


def test_assert_store_isolated_rejects_foreign_store(tmp_path) -> None:
    config = _config(tmp_path)
    ns = ExperimentNamespace.from_config(config)

    class _Foreign:
        _dsn = "seekdb://root@127.0.0.1:2881/rosclaw"

    with pytest.raises(NamespaceError, match="outside the experiment namespace"):
        ns.assert_store_isolated(_Foreign())

    class _Mine:
        _dsn = "seekdb://root@127.0.0.1:2881/rosclaw_evo_rps_test"

    ns.assert_store_isolated(_Mine())  # no raise


def test_provision_marker_records_config_hash(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    ns = ExperimentNamespace.from_config(config)
    monkeypatch.setattr(ns, "_ensure_database", lambda: True)
    marker = ns.provision()
    assert marker["database_created"] is True
    assert (tmp_path / "evidence_root" / "namespace.json").is_file()
    assert marker["config_hash"] == config.config_hash
