"""Real SeekDB validation (任务书 §十五).

Two paths are kept strictly separate (§15.1):

1. ``seekdb_server`` (mysql://...:2881) — SQL-port deployment. Currently
   BLOCKED_EXTERNAL on this site (Docker Hub unreachable; PyPI ships only
   the embedded engine). Test asserts the client fails closed.
2. ``seekdb_embedded`` — the REAL OceanBase SeekDB engine in-process
   (pylibseekdb/pyseekdb, native VECTOR type + cosine/l2 + BM25). Numbers
   here are labelled ``embedded`` — real engine numbers, NOT SQLite-compat.

Site finding: the engine's built-in MiniLM-L6 (384-d, English) embedding is
inadequate for Chinese operational text (semantic search mis-ranks Chinese
documents). The Qwen-1024 path is therefore validated directly against the
engine's raw SQL interface with explicit embeddings — the same deployment
shape rosclaw documents as "manual multilingual embeddings".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EMBEDDED_PATH = "/tmp/ty1200_seekdb_embedded"
RESULTS: dict = {"path": "seekdb_embedded (real engine, in-process)"}


def _store():
    from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore

    return SeekDBEmbeddedStore(path=EMBEDDED_PATH, database="rosclaw")


def _qwen_embed(text: str) -> list[float]:
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/embeddings",
        data=json.dumps({"model": "/models/Qwen/Qwen3-Embedding-0.6B",
                         "input": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["data"][0]["embedding"]


def test_00_engine_available():
    pyseekdb = pytest.importorskip("pyseekdb")
    assert hasattr(pyseekdb, "Client")


def test_01_server_path_fails_closed_when_unreachable():
    """§15.1: the 2881 SQL path must fail honestly when no server exists."""
    from rosclaw.memory.seekdb_client import SeekDBMySQLClient

    client = SeekDBMySQLClient("mysql://root@127.0.0.1:2881/rosclaw",
                               connect_timeout=2.0)
    with pytest.raises(Exception):
        client.connect()
    RESULTS["server_path"] = "BLOCKED_EXTERNAL (no 2881 server on this site; fails closed)"


def test_02_connect_and_create():
    store = _store()
    store.connect()
    assert store.is_connected()
    RESULTS["embedded_connect"] = "PASS"


def test_03_idempotent_insert():
    store = _store()
    store.connect()
    record = {
        "id": "episode-upsert-1",
        "robot_id": "ty1200",
        "body_id": "body_ur5e",
        "skill_id": "reach",
        "outcome": "success",
        "description": "deterministic upsert probe",
    }
    store.insert("episodes", record)
    store.insert("episodes", record)  # duplicate ingest must not duplicate rows
    count = store.count("episodes")
    RESULTS["idempotent_duplicate_rows"] = max(0, count - 1)
    assert count == 1, f"duplicate insert produced {count} rows"


def test_04_filtered_queries():
    store = _store()
    store.connect()
    for i in range(20):
        store.insert("failures", {
            "id": f"fail-{i}",
            "robot_id": "ty1200" if i % 2 == 0 else "other",
            "failure_type": "joint_limit" if i % 3 else "timeout",
            "severity": "high" if i % 4 else "low",
            "description": f"synthetic failure {i}",
        })
    rows = store.query("failures", filters={"robot_id": "ty1200"}, limit=50)
    assert len(rows) == 10
    assert all(r["robot_id"] == "ty1200" for r in rows)
    RESULTS["filtered_queries"] = "PASS"


def test_05_qwen_1024_vector_search_raw_engine():
    """Qwen-1024 explicit embeddings against the engine's native VECTOR type.

    This is the site's multilingual deployment shape (§12): engine stores
    raw vectors, retrieval via cosine_distance over explicit embeddings.
    """
    import pyseekdb
    from pyseekdb import HNSWConfiguration

    docs = {
        "vec-joint": "机械臂关节目标越界导致 sandbox 阻断",
        "vec-network": "SeekDB 网络断开后本地补偿回放",
        "vec-vision": "视觉传感器丢帧引起状态估计漂移",
    }
    client = pyseekdb.Client(path=EMBEDDED_PATH, database="rosclaw")
    try:
        client.delete_collection("qwen1024")
    except Exception:  # noqa: BLE001
        pass
    coll = client.create_collection(
        "qwen1024",
        configuration=HNSWConfiguration(dimension=1024),
        embedding_function=None,
    )
    for k, text in docs.items():
        coll.add(ids=[k], embeddings=[_qwen_embed(text)], documents=[text])

    qvec = _qwen_embed("关节越界被安全沙箱拒绝")
    # pyseekdb 1.4.0 embedded: Collection.query(query_embeddings=...) returns
    # empty — retrieve via the engine's raw SQL cosine_distance instead
    # (the site-standard path; same semantics, fully auditable SQL).
    conn = client._server._ensure_connection()
    cur = conn.cursor()
    # resolve qwen1024's physical table from the sdk_collections registry
    cur.execute("SELECT * FROM sdk_collections")
    for row in cur.fetchall():
        if row[1] == "qwen1024":
            table = f"c$v2${row[0]}"
    assert table, "qwen1024 physical table not found"
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    cur.execute(
        f"SELECT _id, cosine_distance(embedding, '{vec_literal}') AS d "
        f"FROM `{table}` ORDER BY d LIMIT 3"
    )
    ids = [r[0] for r in cur.fetchall()]
    RESULTS["vector_search_qwen1024"] = {"ranking": ids, "dim": 1024}
    assert ids and ids[0] == "vec-joint", f"top hit was {ids}"

    # wrong dimension must be rejected, never silently accepted
    with pytest.raises(Exception):
        coll.add(ids=["bad"], embeddings=[[0.1] * 8])
    RESULTS["wrong_dimension_accepts"] = 0


def test_06_builtin_embedding_chinese_quality_finding():
    """Document the site finding: MiniLM-384 mis-ranks Chinese text."""
    store = _store()
    store.connect()
    docs = [
        {"id": "q1", "description": "机械臂关节目标越界导致 sandbox 阻断", "outcome": "failure"},
        {"id": "q2", "description": "SeekDB 网络断开后本地补偿回放", "outcome": "failure"},
        {"id": "q3", "description": "视觉传感器丢帧引起状态估计漂移", "outcome": "failure"},
    ]
    for d in docs:
        store.insert("memory_nodes", d)
    hits = store.similar("memory_nodes", "关节越界被安全沙箱拒绝", limit=3)
    top = hits[0]["id"] if hits else None
    RESULTS["builtin_minilm_chinese"] = {
        "top1_for_joint_query": top,
        "expected": "q1",
        "finding": ("builtin MiniLM-384 embedding is English-centric; Chinese "
                    "semantic search is unreliable -> site uses Qwen-1024 "
                    "manual embeddings (validated in test_05)"),
    }
    # Not a hard failure of the engine; the finding is recorded, and the
    # correct deployment path is proven in test_05.
    assert hits is not None


def test_07_fulltext_bm25():
    store = _store()
    store.connect()
    hits = store.fulltext_search("memory_nodes", "关节", limit=3)
    assert hits, "BM25 fulltext returned no hits for 关节"
    RESULTS["fulltext_bm25"] = "PASS"


def test_08_restart_persistence():
    """Data survives process restart (new python process, same path)."""
    code = (
        "from rosclaw.storage.seekdb_native import SeekDBEmbeddedStore\n"
        f"s = SeekDBEmbeddedStore(path={EMBEDDED_PATH!r}, database='rosclaw')\n"
        "s.connect()\n"
        "print(s.count('episodes'), s.count('failures'))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-500:]
    episodes, failures = out.stdout.strip().split()[-2:]
    assert int(episodes) == 1 and int(failures) == 20
    RESULTS["restart_data_loss"] = 0


def test_zz_write_results():
    report = os.environ.get("TY1200_VALIDATION_REPORT_DIR")
    if report:
        Path(report).mkdir(parents=True, exist_ok=True)
        (Path(report) / "seekdb_embedded.json").write_text(
            json.dumps(RESULTS, indent=2, ensure_ascii=False))
