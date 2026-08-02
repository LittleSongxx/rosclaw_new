#!/usr/bin/env python3
"""Minimal repro: pyseekdb 1.4.0 incompatibility with the embedded SeekDB engine.

Observed on TY1200 (x86_64, glibc 2.35, Python 3.12.13, seekdb-lib 0.0.1.dev5,
pylibseekdb 1.3.0.post3):

  pyseekdb 1.3.0 + embedded engine -> all operations work.
  pyseekdb 1.4.0 + embedded engine -> TWO failure classes:

  (a) ``get``/filtered queries raise::

        pylibseekdb.SeekdbError: You have an error in your SQL syntax;
        check the manual that corresponds to your OceanBase version for the
        right syntax to use near 'DESC, `__pk_increment` LIMIT 5' at line 1
        failed: code=1064

  (b) ``hybrid_search`` with only the BM25 (query) leg returns empty lists
      for every query, including substrings that exist in the documents.

Run:

    python -m venv /tmp/repro-venv && /tmp/repro-venv/bin/pip install \
        "pyseekdb==1.4.0" seekdb seekdb-lib pylibseekdb
    /tmp/repro-venv/bin/python validation/ty1200/scripts/repro_pyseekdb_140.py

Expected with 1.3.0: prints "PASS". Expected with 1.4.0: prints the two
failure classes above and exits 1.
"""

from __future__ import annotations

import sys
import tempfile

import pyseekdb
from pyseekdb import HNSWConfiguration


def main() -> int:
    path = tempfile.mkdtemp(prefix="repro-140-")
    client = pyseekdb.AdminClient(path=path)
    client.create_database("repro")
    client = pyseekdb.Client(path=path, database="repro")
    coll = client.create_collection(
        "docs", configuration=HNSWConfiguration(dimension=4), embedding_function=None
    )
    coll.add(
        ids=["a", "b"],
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        documents=["joint limit exceeded in sandbox", "network retry succeeded"],
    )

    failures = []

    # (a) filtered get()
    try:
        rows = coll.get(where={"robot": "ty1200"}, limit=5, include=["metadatas"])
        print("get() ok:", (rows or {}).get("ids"))
    except Exception as exc:  # noqa: BLE001
        failures.append(f"get() raised: {type(exc).__name__}: {str(exc)[:160]}")

    # (b) BM25-only hybrid leg
    try:
        res = coll.hybrid_search(
            query={"where_document": {"$contains": "joint"}, "n_results": 3},
            n_results=3,
            include=["metadatas"],
        )
        ids = (res or {}).get("ids")
        print("hybrid BM25 leg:", ids)
        if not ids or ids == [[]]:
            failures.append("hybrid_search BM25 leg returned empty for an existing substring")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"hybrid_search raised: {type(exc).__name__}: {str(exc)[:160]}")

    if failures:
        print("\nREPRO FAILURES (pyseekdb", getattr(pyseekdb, "__version__", "?"), "):")
        for f in failures:
            print(" -", f)
        return 1
    print("PASS (pyseekdb", getattr(pyseekdb, "__version__", "?"), "works with embedded engine)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
