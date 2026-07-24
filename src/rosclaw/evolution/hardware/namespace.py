"""Experiment namespace isolation (真机自进化v2 §2.7).

A formal experiment must not read historical campaign memory: it gets its
own SeekDB database, practice root, trace root, and evidence root.  The
:class:`ExperimentNamespace` provisions those roots and is the ONLY way the
harness resolves storage — passing the shared ``rosclaw`` database or a
foreign root is a contract violation, not a silent fallback.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import EvoRpsConfig

SHARED_DATABASE = "rosclaw"


class NamespaceError(RuntimeError):
    """Namespace provisioning or isolation violation."""


@dataclass
class ExperimentNamespace:
    config: EvoRpsConfig
    database: str
    dsn: str
    practice_root: Path
    trace_root: Path
    evidence_root: Path

    @classmethod
    def from_config(cls, config: EvoRpsConfig) -> ExperimentNamespace:
        database = str(config.namespace.get("database") or "")
        if config.require_clean_namespace and database == SHARED_DATABASE:
            raise NamespaceError(
                f"experiment {config.experiment_id} must not use the shared "
                f"{SHARED_DATABASE!r} database (§2.7)"
            )
        return cls(
            config=config,
            database=database,
            dsn=config.seekdb_dsn(),
            practice_root=Path(os.path.expanduser(str(config.namespace["practice_root"]))),
            trace_root=Path(os.path.expanduser(str(config.namespace["trace_root"]))),
            evidence_root=Path(os.path.expanduser(str(config.namespace["evidence_root"]))),
        )

    # ------------------------------------------------------------------

    def provision(self) -> dict[str, Any]:
        """Create the roots and the isolated SeekDB database (idempotent)."""
        for root in (self.practice_root, self.trace_root, self.evidence_root):
            root.mkdir(parents=True, exist_ok=True)
        created = self._ensure_database()
        marker = {
            "experiment_id": self.config.experiment_id,
            "database": self.database,
            "dsn": self.dsn,
            "config_hash": self.config.config_hash,
            "practice_root": str(self.practice_root),
            "trace_root": str(self.trace_root),
        }
        (self.evidence_root / "namespace.json").write_text(
            json.dumps(marker, indent=2, ensure_ascii=False)
        )
        return {"database_created": created, **marker}

    def _ensure_database(self) -> bool:
        """CREATE DATABASE IF NOT EXISTS via the MySQL protocol (SeekDB)."""
        try:
            import pymysql
        except ImportError as exc:  # pragma: no cover - dependency present in prod env
            raise NamespaceError("pymysql is required for namespace provisioning") from exc
        ns = self.config.namespace
        conn = pymysql.connect(
            host=str(ns["seekdb_host"]),
            port=int(ns["seekdb_port"]),
            user=str(ns["seekdb_user"]),
            password=str(ns.get("seekdb_password") or ""),
            autocommit=True,
            connect_timeout=10,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.database}`"
                )
                created = cur.rowcount > 0
        finally:
            conn.close()
        return bool(created)

    # ------------------------------------------------------------------

    def knowledge_store(self) -> Any:
        """The experiment's own knowledge store — never the shared one."""
        from rosclaw.storage.factory import StorageFactory

        store = StorageFactory.create_knowledge_store(backend="seekdb_server", url=self.dsn)
        store.connect()
        return store

    def assert_store_isolated(self, store: Any) -> None:
        """Fail loudly if a store points anywhere but this namespace."""
        dsn = getattr(store, "_dsn", None) or getattr(store, "dsn", None) or ""
        if self.database not in str(dsn):
            raise NamespaceError(
                f"knowledge store DSN {dsn!r} is outside the experiment "
                f"namespace {self.database!r} (§2.7)"
            )
