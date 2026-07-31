#!/usr/bin/env python3
"""Build the local ROSClaw Wiki (任务书 §17.1/§17.2).

Pipeline: file scan → secret scan → chunk → Qwen embedding → store.
Store backend: sqlite file (validation workspace). When the real SeekDB
server is available the same chunks flow through `practice ingest-seekdb`/
Memory paths; this builder is the deterministic offline index.

Every chunk carries the §17.1 document model fields. No secrets, tokens,
or site addresses are accepted into the index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.request
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(password|passwd|approval[_-]?token)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]{10,}"),
]
SITE_ADDR = re.compile(r"\b10\.10\.217\.108\b")  # site-local LLM address must not leak


def secret_scan(text: str, path: str) -> list[str]:
    hits = []
    for pat in SECRET_PATTERNS + [SITE_ADDR]:
        for m in pat.finditer(text):
            hits.append(f"{path}: matched {pat.pattern[:40]} at {m.start()}")
    return hits


def chunk_markdown(text: str, doc_id: str, source_uri: str, source_hash: str,
                   trust: str, max_chars: int = 1200) -> list[dict]:
    """Split markdown into header-aware chunks with line spans."""
    lines = text.splitlines()
    sections: list[tuple[str, int, int]] = []  # (title, start, end)
    current_title = doc_id
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            if i > start:
                sections.append((current_title, start, i))
            current_title = line.lstrip("#").strip() or doc_id
            start = i
    sections.append((current_title, start, len(lines)))

    chunks = []
    for title, s, e in sections:
        body = "\n".join(lines[s:e]).strip()
        if len(body) < 80:
            continue  # skip title-only / near-empty chunks (retrieval noise)
        # split oversized sections on paragraph boundaries
        parts, buf = [], ""
        for para in body.split("\n\n"):
            if len(buf) + len(para) > max_chars and buf:
                parts.append(buf)
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            parts.append(buf)
        for idx, part in enumerate(parts):
            chunks.append({
                "chunk_id": f"{doc_id}::{s}:{idx}",
                "document_id": doc_id,
                "title": title[:200],
                "source_uri": source_uri,
                "source_hash": source_hash,
                "git_commit": "tarball-2026-07-31",
                "section": title[:120],
                "line_start": s,
                "line_end": e,
                "content": part,
                "embedding_model": "qwen3-embedding-0.6b",
                "embedding_dimension": 1024,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "visibility": "internal",
                "trust_level": trust,
                "supersedes": None,
            })
    return chunks


def embed(texts: list[str], endpoint: str, model: str, timeout: float = 60.0) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/embeddings", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return [item["embedding"] for item in sorted(body["data"], key=lambda d: d["index"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="validation/ty1200/fixtures/wiki_documents")
    ap.add_argument("--db", required=True)
    ap.add_argument("--endpoint", default=os.environ.get("TY1200_EMBEDDING_ENDPOINT",
                                                        "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default="/models/Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--report")
    args = ap.parse_args()

    docs_dir = Path(args.docs)
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"no markdown files in {docs_dir}")

    all_chunks: list[dict] = []
    secret_hits: list[str] = []
    redactions: list[str] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        # Site-local addresses are redacted in place (document still indexed);
        # true secrets reject the whole document (fail closed).
        redacted_text, n = SITE_ADDR.subn("[SITE-LOCAL-ADDR]", text)
        if n:
            redactions.append(f"{f.name}: {n} site address(es) redacted")
            text = redacted_text
        hits = secret_scan(text, f.name)
        secret_hits.extend(hits)
        if hits:
            continue  # fail closed: document with secrets never enters the index
        trust = "official" if f.name.startswith(("rosclaw_", "ty1200_platform")) else "local"
        doc_hash = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
        all_chunks.extend(chunk_markdown(text, f.stem, f"validation/ty1200/fixtures/wiki_documents/{f.name}",
                                         doc_hash, trust))

    # embed in batches
    embeddings: list[list[float]] = []
    batch = 16
    t0 = time.perf_counter()
    for i in range(0, len(all_chunks), batch):
        group = all_chunks[i:i + batch]
        embeddings.extend(embed([c["content"][:2000] for c in group], args.endpoint, args.model))
    embed_s = time.perf_counter() - t0

    con = sqlite3.connect(args.db)
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute(
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT, title TEXT,"
        " source_uri TEXT, source_hash TEXT, git_commit TEXT, section TEXT,"
        " line_start INTEGER, line_end INTEGER, content TEXT, embedding BLOB,"
        " embedding_model TEXT, embedding_dimension INTEGER, created_at TEXT,"
        " visibility TEXT, trust_level TEXT, supersedes TEXT)"
    )
    import array
    for chunk, vec in zip(all_chunks, embeddings):
        con.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (chunk["chunk_id"], chunk["document_id"], chunk["title"], chunk["source_uri"],
             chunk["source_hash"], chunk["git_commit"], chunk["section"],
             chunk["line_start"], chunk["line_end"], chunk["content"],
             array.array("f", vec).tobytes(), chunk["embedding_model"],
             chunk["embedding_dimension"], chunk["created_at"], chunk["visibility"],
             chunk["trust_level"], chunk["supersedes"]),
        )
    con.commit()
    con.close()

    report = {
        "documents_indexed": len(files) - len({h.split(':')[0] for h in secret_hits}),
        "documents_rejected_secret_scan": sorted({h.split(":")[0] for h in secret_hits}),
        "site_address_redactions": redactions,
        "secret_scan_hits": secret_hits,
        "chunks": len(all_chunks),
        "embed_seconds": round(embed_s, 2),
        "embedding_model": args.model,
        "db": args.db,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not secret_hits else 0  # rejected docs are reported, not fatal


if __name__ == "__main__":
    raise SystemExit(main())
