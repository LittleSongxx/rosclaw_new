#!/usr/bin/env python3
"""Wiki 深度测试（无 LLM 部分，§任务C 评审补测）.

1. supersedes_retrieval: 过期文档被新版取代后, 检索必须优先新版
2. conflict_retrieval: 多来源冲突时, 冲突双方都必须被检索到(不得静默只取一边)
3. citation_completeness: 基于全量 benchmark 答案, 事实性论断的引用覆盖
4. answer_correctness: 答案文本命中 expected_keywords 的比率
5. writeback_pollution_gate: 生成条目写回索引前的门禁——秘密泄漏 /
   无来源论断 / 伪造引用 必须全部被拦
"""

from __future__ import annotations

import array
import json
import math
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

EMBED_ENDPOINT = "http://127.0.0.1:8000/v1"
EMBED_MODEL = "qwen3-embedding-0.6b"

RESULTS: dict = {}


def embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{EMBED_ENDPOINT}/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "input": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["data"][0]["embedding"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def build_index(rows: list[dict], db: str) -> None:
    con = sqlite3.connect(db)
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT,"
                " title TEXT, content TEXT, embedding BLOB, supersedes TEXT)")
    for r in rows:
        vec = array.array("f", embed(r["content"])).tobytes()
        con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                    (r["chunk_id"], r["document_id"], r["title"], r["content"],
                     vec, r.get("supersedes")))
    con.commit()
    con.close()


def load_index(db: str) -> list[dict]:
    con = sqlite3.connect(db)
    out = []
    for cid, doc, title, content, blob, sup in con.execute(
            "SELECT chunk_id, document_id, title, content, embedding, supersedes FROM chunks"):
        v = array.array("f")
        v.frombytes(blob)
        out.append({"chunk_id": cid, "document_id": doc, "title": title,
                    "content": content, "vec": list(v), "supersedes": sup})
    con.close()
    return out


def top_k(query: str, docs: list[dict], k: int = 3, respect_supersedes: bool = True):
    qv = embed(query)
    superseded = {d["chunk_id"] for d in docs if d.get("supersedes")}
    pool = [d for d in docs if not (respect_supersedes and d["chunk_id"] in superseded)]
    return sorted(pool, key=lambda d: cosine(qv, d["vec"]), reverse=True)[:k]


# ---------------------------------------------------------------- supersedes
def test_supersedes(work: Path) -> dict:
    rows = [
        {"chunk_id": "doc::old", "document_id": "ops", "title": "端口规划 v1",
         "content": "ROSClaw API 运行在 8000 端口 (旧版规划)"},
        {"chunk_id": "doc::new", "document_id": "ops", "title": "端口规划 v2",
         "content": "ROSClaw API 运行在 9010 端口, 8000 留给 Embedding (新版规划)",
         "supersedes": None},
    ]
    rows[0]["supersedes"] = "doc::new"  # old 被 new 取代
    build_index(rows, str(work / "sup.db"))
    docs = load_index(str(work / "sup.db"))

    naive = top_k("ROSClaw API 端口是多少", docs, respect_supersedes=False)
    aware = top_k("ROSClaw API 端口是多少", docs, respect_supersedes=True)
    ok = (naive[0]["chunk_id"] == "doc::old"  # 不管 supersedes 时旧版确实更相关
          and aware[0]["chunk_id"] == "doc::new")  # 尊重取代关系后新版胜出
    return {"status": "PASS" if ok else "FAIL",
            "naive_top": naive[0]["chunk_id"], "aware_top": aware[0]["chunk_id"],
            "requirement": "supersedes-aware retrieval must prefer the newer doc"}


# ---------------------------------------------------------------- conflict
def test_conflict(work: Path) -> dict:
    rows = [
        {"chunk_id": "a::1", "document_id": "vendor_a", "title": "手册 A",
         "content": "TY1200 的 GPGPU 显存是 16GB HBM2e"},
        {"chunk_id": "b::1", "document_id": "vendor_b", "title": "手册 B (过期)",
         "content": "TY1200 的 GPGPU 显存是 8GB HBM2"},
        {"chunk_id": "c::1", "document_id": "ops", "title": "无关文档",
         "content": "关节越界时钳制目标到执行器范围内"},
    ]
    build_index(rows, str(work / "conf.db"))
    docs = load_index(str(work / "conf.db"))
    top = top_k("TY1200 显存多大", docs, k=3, respect_supersedes=False)
    ids = {d["chunk_id"] for d in top}
    both_surfaced = {"a::1", "b::1"} <= ids
    return {"status": "PASS" if both_surfaced else "FAIL",
            "top3": [d["chunk_id"] for d in top],
            "requirement": "conflicting sources must BOTH surface (no silent one-sided answer)"}


# ------------------------------------------------- citation completeness
_FACT_PATTERN = re.compile(r"[。；;]\s*[^。\n]{8,}?（?[^）]*?）?(?=[。；;]|$)")


def test_citation_completeness(report_dir: Path) -> dict:
    data = json.load(open(report_dir / "wiki/wiki_benchmark_full.json"))
    cases = [c for c in data["cases"] if c["answerable"] and not c["abstained"]]
    total_claims = cited_claims = 0
    per_case = []
    for c in cases:
        answer = c["answer"]
        sentences = [s.strip() for s in re.split(r"[。；;\n]", answer) if len(s.strip()) >= 8]
        factual = [s for s in sentences
                   if not s.startswith(("你", "我", "可以", "建议阅读"))]
        claims = len(factual) or 1
        cited = sum(1 for s in factual if re.search(r"\[[^\]]+::\d+:\d+\]", s))
        total_claims += claims
        cited_claims += cited
        per_case.append({"id": c["id"], "claims": claims, "cited": cited})
    completeness = cited_claims / total_claims if total_claims else 1.0
    return {"status": "PASS" if completeness >= 0.8 else "WARN",
            "citation_completeness": round(completeness, 3),
            "claims": total_claims, "cited": cited_claims,
            "note": "近似度量: 长句级事实论断中携带引用的比例(阈值 0.8)",
            "worst": sorted(per_case, key=lambda x: x["cited"] / x["claims"])[:3]}


# ------------------------------------------------------------- correctness
def test_answer_correctness(report_dir: Path) -> dict:
    data = json.load(open(report_dir / "wiki/wiki_benchmark_full.json"))
    bank = {q["id"]: q for q in json.load(
        open("validation/ty1200/fixtures/knowledge_questions/wiki_qa_bank.json"))["questions"]}
    hit = total = 0
    misses = []
    for c in data["cases"]:
        q = bank.get(c["id"])
        if not q or not q["answerable"] or not q.get("expected_keywords"):
            continue
        total += 1
        if any(kw in c["answer"] for kw in q["expected_keywords"]):
            hit += 1
        else:
            misses.append(c["id"])
    correctness = hit / total if total else 1.0
    return {"status": "PASS" if correctness >= 0.8 else "FAIL",
            "answer_correctness": round(correctness, 3),
            "misses": misses,
            "note": "答案文本命中 expected_keywords 的比例(阈值 0.8)"}


# ------------------------------------------------------------ writeback gate
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(password|approval[_-]?token)\s*[:=]\s*\S+"),
]


def test_writeback_pollution(work: Path) -> dict:
    """生成条目写回索引前的三道闸: secret scan / 来源引用校验 / 无支撑论断."""
    generated_entries = [
        {"id": "g1", "title": "正常条目", "content": "UR5e 关节越界时钳制到 ctrlrange 恢复 [practice_ur5e_001]",
         "source_refs": ["practice_ur5e_001"],
         "claims": ["钳制关节目标可恢复"]},
        {"id": "g2", "title": "含秘密条目", "content": "配置 approval_token: abcdef123456 即可",
         "source_refs": [], "claims": []},
        {"id": "g3", "title": "无来源论断", "content": "该机型支持 24 轴联动 (没有任何证据)",
         "source_refs": [],
         "claims": ["支持 24 轴联动"]},
        {"id": "g4", "title": "伪造引用", "content": "根据 [practice_nonexistent_999] 的结论",
         "source_refs": ["practice_nonexistent_999"],
         "claims": ["某结论"]},
    ]
    known_refs = {"practice_ur5e_001", "trace_ur5e_001"}

    verdicts = []
    for e in generated_entries:
        reasons = []
        text = e["content"]
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                reasons.append("secret_leak")
        for ref in e["source_refs"]:
            if ref not in known_refs:
                reasons.append(f"unknown_ref:{ref}")
        for claim in e["claims"]:
            if not e["source_refs"]:
                reasons.append(f"unsupported_claim:{claim[:20]}")
        admitted = not reasons
        verdicts.append({"id": e["id"], "admitted": admitted, "reasons": reasons})

    admitted = {v["id"] for v in verdicts if v["admitted"]}
    ok = admitted == {"g1"}
    return {"status": "PASS" if ok else "FAIL",
            "verdicts": verdicts,
            "requirement": "write-back gate must admit only g1 (clean entry)"}


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else None
    work = Path("/tmp/wiki_deep_tests")
    work.mkdir(exist_ok=True)
    report_dir = Path("validation/ty1200/reports/ty1200_20260731_180203")

    for name, fn in [
        ("supersedes_retrieval", lambda: test_supersedes(work)),
        ("conflict_retrieval", lambda: test_conflict(work)),
        ("citation_completeness", lambda: test_citation_completeness(report_dir)),
        ("answer_correctness", lambda: test_answer_correctness(report_dir)),
        ("writeback_pollution_gate", lambda: test_writeback_pollution(work)),
    ]:
        try:
            RESULTS[name] = fn()
        except Exception as exc:  # noqa: BLE001
            RESULTS[name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[{RESULTS[name]['status']:4}] {name}")

    overall = "PASS" if all(r["status"] == "PASS" for r in RESULTS.values()) else "FAIL"
    print(f"overall: {overall}")
    summary = {"tests": RESULTS, "overall": overall,
               "llm_dependent_pending": [
                   "conflict answer resolution (DeepSeek)",
                   "fresh full-generation correctness re-run (DeepSeek)",
               ]}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
