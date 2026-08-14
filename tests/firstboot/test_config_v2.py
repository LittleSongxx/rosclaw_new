"""PR-DF-02: Config v2 — canonical storage.*/knowledge/evolution sections
with legacy-field normalization (ADR-0010 §11)."""

import logging

from rosclaw.firstboot.config import FirstbootConfig


def test_v2_defaults_present():
    cfg = FirstbootConfig()
    assert cfg.schema_version == "2.0"
    structured = cfg.storage["structured"]
    assert structured["backend"] == "sqlite"
    assert structured["path"].endswith("knowledge.sqlite")  # pre-v2 path kept
    retrieval = cfg.storage["retrieval"]
    assert retrieval["backend"] == "seekdb_native"
    assert retrieval["mode"] == "embedded"
    assert retrieval["port"] == 2881
    assert cfg.storage["outbox"]["batch_size"] == 100
    assert cfg.storage["artifacts"]["backend"] == "filesystem"
    assert cfg.knowledge["enabled"] is True
    assert cfg.knowledge["mode"] == "inprocess"
    assert "trigger_failure_threshold" in cfg.evolution


def test_legacy_seekdb_fields_map_to_structured(caplog):
    cfg = FirstbootConfig(
        runtime={
            "seekdb_backend": "mysql",
            "seekdb_url": "mysql://root@127.0.0.1:2881/rosclaw",
            "seekdb_path": "/tmp/x.sqlite",
        }
    )
    structured = cfg.storage["structured"]
    assert structured["backend"] == "mysql"
    assert structured["dsn"] == "mysql://root@127.0.0.1:2881/rosclaw"
    assert structured["path"] == "/tmp/x.sqlite"
    assert any("DEPRECATED CONFIG" in r.message for r in caplog.records)
    # legacy mirrors still populated for pre-DF-03 readers
    assert cfg.runtime["seekdb_backend"] == "mysql"


def test_explicit_v2_wins_over_legacy(caplog):
    cfg = FirstbootConfig(
        runtime={"seekdb_backend": "mysql"},
        storage={"structured": {"backend": "sqlite"}},
    )
    assert cfg.storage["structured"]["backend"] == "sqlite"
    caplog.clear()
    # no deprecation note: the v2 key was already set, legacy not consumed
    cfg2 = FirstbootConfig(storage={"structured": {"backend": "sqlite"}})
    assert cfg2.storage["structured"]["backend"] == "sqlite"
    assert not [r for r in caplog.records if "DEPRECATED" in r.message]


def test_memory_backend_vocabulary_maps(caplog):
    assert FirstbootConfig(memory={"backend": "seekdb"}).storage["structured"]["backend"] == "mysql"
    assert FirstbootConfig(memory={"backend": "local"}).storage["structured"]["backend"] == "sqlite"


def test_vector_enabled_maps_to_retrieval():
    cfg = FirstbootConfig(storage={"vector_enabled": True})
    assert cfg.storage["retrieval"]["enabled"] is True
    # and the flat legacy key is preserved
    assert cfg.storage["vector_enabled"] is True


def test_know_auto_mirrored_to_canonical_and_back():
    cfg = FirstbootConfig(know={"asset_dir": "/tmp/assets"}, auto={"enabled": True})
    assert cfg.knowledge["asset_dir"] == "/tmp/assets"
    assert cfg.evolution["enabled"] is True
    # mirror keeps legacy sections in sync
    assert cfg.know["asset_dir"] == "/tmp/assets"
    assert cfg.auto["enabled"] is True
    # canonical defaults land in the legacy mirror too
    assert cfg.know["mode"] == "inprocess"
    assert cfg.auto["require_human_approval"] is True


def test_to_dict_carries_both_vocabularies():
    d = FirstbootConfig().to_dict()
    for key in ("knowledge", "evolution", "know", "auto"):
        assert key in d and isinstance(d[key], dict) and d[key]
    assert d["storage"]["structured"]["backend"] == "sqlite"


def test_no_legacy_input_no_deprecation_noise(caplog):
    with caplog.at_level(logging.WARNING, logger="rosclaw.firstboot.config"):
        FirstbootConfig()
    assert not [r for r in caplog.records if "DEPRECATED CONFIG" in r.message]
