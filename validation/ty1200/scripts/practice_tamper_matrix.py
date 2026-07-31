#!/usr/bin/env python3
"""Practice tamper-detection matrix (任务书 §14.3).

For each tamper type, copy the pristine practice dataset, mutate exactly one
thing, and require `practice verify --strict` to reject it. A tamper that
passes verification is a FAIL of this harness (and a real defect).

Tamper types: jsonl_line | yaml_asset | image | trace_ref | catalog_sqlite |
artifact_hash | episode_metadata
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROSCLAW = ".venv/bin/rosclaw"
FIXTURE = "tests/fixtures/practice/rh56_minimal_loop.json"
PRACTICE_ID = "practice_rh56_minimal_loop"


def run(cmd: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)


def verify(data_root: Path, env: dict) -> bool:
    """Return True if strict verify ACCEPTS the dataset."""
    proc = run(
        [ROSCLAW, "practice", "verify", PRACTICE_ID,
         "--data-root", str(data_root), "--strict", "--json"],
        env,
    )
    return proc.returncode == 0


def build_pristine(base: Path, env: dict) -> Path:
    data = base / "pristine" / "data"
    data.mkdir(parents=True)
    proc = run([ROSCLAW, "practice", "record", "--fixture", FIXTURE,
                "--out", str(data), "--json"], env)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise SystemExit("pristine record failed")
    # distill produces the artifact assets (failures/candidates/...) whose
    # sha256 is registered in the practice catalog.
    proc = run([ROSCLAW, "practice", "distill", PRACTICE_ID,
                "--data-root", str(data), "--json"], env)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr)
        raise SystemExit("pristine distill failed")
    return data


def _first(data: Path, pattern: str) -> Path:
    matches = sorted(data.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} under {data}")
    return matches[0]


def tamper_jsonl_line(data: Path) -> None:
    f = _first(data, "events.jsonl")
    lines = f.read_text().splitlines()
    obj = json.loads(lines[len(lines) // 2])
    obj["tampered"] = True
    for key in ("outcome", "status", "result"):
        if key in obj:
            obj[key] = "success" if str(obj[key]) != "success" else "failure"
    lines[len(lines) // 2] = json.dumps(obj)
    f.write_text("\n".join(lines) + "\n")


def tamper_yaml_asset(data: Path) -> None:
    f = _first(data, "failures_asset_*.yaml")
    text = f.read_text()
    f.write_text(text + "\n# tampered: injected fake recovery note\n")


def tamper_image(data: Path) -> None:
    candidates = list(data.rglob("*.png")) + list(data.rglob("*.jpg")) + list(data.rglob("*.jpeg"))
    if not candidates:
        # Fixture has no media: simulate a media-bearing practice by adding a
        # frame artifact and registering it (correct sha) in the episode
        # artifact manifest, then corrupt the image bytes.
        import hashlib

        manifest_path = _first(data, "artifact_manifest.yaml")
        frame = manifest_path.parent / "frames" / "frame_0001.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"\x89PNG\r\n\x1a\nLEGITIMATE-FRAME")
        sha = hashlib.sha256(frame.read_bytes()).hexdigest()
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["artifacts"]["frame_0001"] = {
            "artifact_id": "frame_0001",
            "artifact_type": "frame",
            "path": str(frame),
            "sha256": sha,
            "size_bytes": frame.stat().st_size,
            "schema_name": "image.png",
            "created_at": "2026-07-31T00:00:00Z",
            "session_id": manifest.get("session_id"),
            "episode_id": manifest.get("episode_id"),
            "metadata": {},
        }
        manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True))
        candidates = [frame]
    candidates[0].write_bytes(b"\x89PNG\r\n\x1a\nTAMPERED")


def tamper_trace_ref(data: Path) -> None:
    f = _first(data, "events.jsonl")
    text = f.read_text()
    if "trace_" in text:
        text = text.replace("trace_", "trace_TAMPERED_", 1)
    else:
        text += "\n"
    f.write_text(text)


def tamper_catalog_sqlite(data: Path) -> None:
    db = _first(data, "practice_catalog.sqlite")
    con = sqlite3.connect(db)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        target = next((t for t in tables if "episode" in t or "session" in t), tables[0])
        con.execute(f"UPDATE {target} SET rowid = rowid WHERE 0")  # no-op probe
        # real tamper: flip a text column if one exists
        cols = con.execute(f"PRAGMA table_info({target})").fetchall()
        text_cols = [c[1] for c in cols if c[2].upper().startswith(("TEXT", "CHAR"))]
        if text_cols:
            con.execute(f"UPDATE {target} SET {text_cols[0]} = {text_cols[0]} || '_TAMPERED' LIMIT 1" if False else f"UPDATE {target} SET {text_cols[0]} = 'TAMPERED' WHERE rowid = (SELECT MIN(rowid) FROM {target})")
        con.commit()
    finally:
        con.close()


def tamper_artifact_hash(data: Path) -> None:
    """Forge the catalog-recorded sha256 of one artifact (attacker rewrites DB)."""
    db = _first(data, "practice_catalog.sqlite")
    con = sqlite3.connect(db)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        table = next(t for t in tables if t == "practice_artifacts")
        row = con.execute(
            f"SELECT rowid FROM {table} WHERE sha256 IS NOT NULL AND sha256 != '' LIMIT 1"
        ).fetchone()
        if row is None:
            raise FileNotFoundError("no artifact with sha256 in catalog")
        con.execute(
            f"UPDATE {table} SET sha256 = 'sha256:' || lower(hex(randomblob(32))) WHERE rowid = ?",
            (row[0],),
        )
        con.commit()
    finally:
        con.close()


def tamper_episode_metadata(data: Path) -> None:
    f = _first(data, "episode.json")
    obj = json.loads(f.read_text())
    obj["tampered_metadata"] = "injected"
    if "outcome" in obj:
        obj["outcome"] = "success" if obj["outcome"] != "success" else "failure"
    f.write_text(json.dumps(obj, indent=2))


TAMPERS = {
    "jsonl_line": tamper_jsonl_line,
    "yaml_asset": tamper_yaml_asset,
    "image": tamper_image,
    "trace_ref": tamper_trace_ref,
    "catalog_sqlite": tamper_catalog_sqlite,
    "artifact_hash": tamper_artifact_hash,
    "episode_metadata": tamper_episode_metadata,
}


def _rebase_paths(data: Path, old_root: Path, new_root: Path) -> None:
    """Rewrite absolute paths stored in the catalog + manifests after copytree.

    Artifact records persist absolute paths; a copied dataset must point at
    its own files or verify would check the pristine originals.
    """
    old, new = str(old_root), str(new_root)
    db = _first(data, "practice_catalog.sqlite")
    con = sqlite3.connect(db)
    try:
        for table, col in [
            ("practices", "manifest_path"),
            ("practices", "events_jsonl_path"),
            ("practices", "replay_path"),
            ("practices", "failure_report_path"),
            ("practice_artifacts", "path"),
            ("artifacts", "path"),
        ]:
            try:
                con.execute(
                    f"UPDATE {table} SET {col} = REPLACE({col}, ?, ?) WHERE {col} LIKE ?",
                    (old, new, f"{old}%"),
                )
            except sqlite3.OperationalError:
                pass  # table/column absent in this catalog version
        con.commit()
    finally:
        con.close()
    for manifest in data.rglob("artifact_manifest.yaml"):
        manifest.write_text(manifest.read_text().replace(old, new))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="ty1200-practice-tamper-"))
    import os
    env = dict(os.environ)
    env["ROSCLAW_HOME"] = str(work / "home")
    (work / "home").mkdir(parents=True)

    pristine = build_pristine(work, env)
    if not verify(pristine, env):
        print("FATAL: pristine dataset failed strict verify", file=sys.stderr)
        return 2

    results = {}
    for name, fn in TAMPERS.items():
        data = work / name / "data"
        shutil.copytree(pristine, data)
        _rebase_paths(data, pristine, data)
        try:
            fn(data)
        except FileNotFoundError as exc:
            results[name] = {"status": "BLOCKED", "detail": str(exc)}
            continue
        accepted = verify(data, env)
        results[name] = {
            "status": "PASS" if not accepted else "FAIL",
            "tampered": True,
            "verify_accepted": accepted,
            "detail": "strict verify rejected the tamper" if not accepted
            else "DEFECT: tampered dataset passed strict verify",
        }
        print(f"[{results[name]['status']:7}] {name}: {results[name]['detail']}")

    # control: pristine copy untouched must still pass
    data = work / "control" / "data"
    shutil.copytree(pristine, data)
    _rebase_paths(data, pristine, data)
    ok = verify(data, env)
    results["control_untouched"] = {"status": "PASS" if ok else "FAIL",
                                    "detail": "untouched copy accepted" if ok else "false positive!"}

    detected = sum(1 for k, v in results.items() if k != "control_untouched" and v["status"] == "PASS")
    total = len(TAMPERS)
    summary = {
        "tamper_detection_rate": f"{detected}/{total}",
        "all_detected": detected == total,
        "control_ok": results["control_untouched"]["status"] == "PASS",
        "cases": results,
        "overall": "PASS" if detected == total and results["control_untouched"]["status"] == "PASS" else "FAIL",
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
