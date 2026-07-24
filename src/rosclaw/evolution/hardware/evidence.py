"""Evidence manifest (真机自进化v2 §Phase 0.6, §2.8).

Every run of the harness appends hash-bound evidence to
``<evidence_root>/evidence_manifest.json`` — config hash, namespace ids,
sessions (practice ids + verify/reconcile outcomes), memories, decisions,
and driver code hashes.  The manifest is the link between a claim in the
report and the raw artifacts; reports must never cite results whose
evidence entries are missing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "rosclaw.acceptance.evidence.v1"


def file_sha256(path: Path, *, head_bytes: int = 1 << 20) -> str:
    """Content hash of a file (first MiB is enough for scripts/configs)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(head_bytes))
    return digest.hexdigest()


@dataclass
class EvidenceManifest:
    root: Path
    experiment_id: str
    config_hash: str
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return self.root / "evidence_manifest.json"

    @classmethod
    def open(cls, root: Path, experiment_id: str, config_hash: str) -> EvidenceManifest:
        root.mkdir(parents=True, exist_ok=True)
        if (root / "evidence_manifest.json").is_file():
            blob = json.loads((root / "evidence_manifest.json").read_text())
            manifest = cls(
                root=root,
                experiment_id=blob["experiment_id"],
                config_hash=blob["config_hash"],
                entries=list(blob.get("entries") or []),
            )
            if manifest.experiment_id != experiment_id:
                raise ValueError(
                    f"evidence root {root} belongs to {manifest.experiment_id}, "
                    f"not {experiment_id}"
                )
            if manifest.config_hash != config_hash:
                raise ValueError(
                    f"config hash changed ({manifest.config_hash} → {config_hash}); "
                    "start a new experiment id instead of mutating the contract"
                )
            return manifest
        manifest = cls(root=root, experiment_id=experiment_id, config_hash=config_hash)
        manifest._write()
        return manifest

    def record(self, kind: str, **fields: Any) -> dict[str, Any]:
        entry = {"kind": kind, "recorded_at": time.time(), **fields}
        self.entries.append(entry)
        self._write()
        return entry

    def by_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.entries if e.get("kind") == kind]

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for entry in self.entries:
            kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
        return {
            "schema": SCHEMA,
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "entries": len(self.entries),
            "by_kind": kinds,
            "path": str(self.path),
        }

    def _write(self) -> None:
        payload = {
            "schema": SCHEMA,
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "entries": self.entries,
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
