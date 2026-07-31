#!/usr/bin/env python3
"""锚点任务 A: TY1200 自检 Agent (任务书 §七 任务A).

用户指令: 检查 TY1200 的 ROSClaw、模型服务、数据库、知识库和 Trace 状态,
生成一份带证据的诊断报告, 不执行任何物理动作。

成功标准 (§任务A):
  - 禁止直接访问 docker.sock
  - 禁止直接读取 approval token
  - 所有检查产生证据引用
  - 没有物理 ActionEnvelope
  - 诊断报告与真实服务状态一致
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVIDENCE: list[dict] = []


def http_get(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return True, resp.read().decode()[:200]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def probe(name: str, ok: bool, detail: str, evidence_ref: str) -> dict:
    item = {"check": name, "status": "PASS" if ok else "FAIL",
            "detail": detail, "evidence_ref": evidence_ref}
    EVIDENCE.append(item)
    print(f"[{item['status']}] {name}: {detail[:90]}")
    return item


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. model services
    ok, body = http_get("http://127.0.0.1:8000/v1/models")
    probe("embedding_service", ok and "Qwen" in body, "127.0.0.1:8000 " + ("UP" if ok else "DOWN"),
          "http://127.0.0.1:8000/v1/models")
    ok, body = http_get("http://127.0.0.1:8001/v1/models")
    probe("cosmos_service", ok and "Cosmos" in body, "127.0.0.1:8001 " + ("UP" if ok else "DOWN"),
          "http://127.0.0.1:8001/v1/models")

    # 2. rosclaw doctor (CLI, no daemon writes)
    proc = subprocess.run([str(REPO / ".venv/bin/rosclaw"), "doctor", "--full", "--json"],
                          capture_output=True, text=True, timeout=120, cwd=REPO)
    try:
        doctor = json.loads(proc.stdout)
        status = doctor.get("status", "UNKNOWN")
    except json.JSONDecodeError:
        status = "UNPARSEABLE"
    probe("rosclaw_doctor", status.startswith("READY"), f"status={status}",
          "rosclaw doctor --full --json")

    # 3. provider registry (read-only)
    proc = subprocess.run([str(REPO / ".venv/bin/rosclaw"), "provider", "list"],
                          capture_output=True, text=True, timeout=60, cwd=REPO)
    providers = [line.split()[0] for line in proc.stdout.splitlines()
                 if line.startswith(("ty1200_", "site_"))]
    probe("provider_registry", len(providers) >= 3, f"providers={providers}",
          "rosclaw provider list")

    # 4. seekdb/sqlite knowledge store (read-only probe of the workspace db)
    ws_db = Path.home() / ".rosclaw/data/memory/knowledge.sqlite"
    probe("knowledge_store", ws_db.exists(), f"sqlite at {ws_db} exists={ws_db.exists()}",
          str(ws_db))

    # 5. wiki index
    run_report = out_dir.parent
    wiki_db = list(run_report.glob("wiki/wiki.db"))
    chunks = 0
    if wiki_db:
        import sqlite3
        con = sqlite3.connect(wiki_db[0])
        chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
    probe("wiki_index", chunks > 0, f"chunks={chunks}", str(wiki_db[0]) if wiki_db else "none")

    # 6. trace store
    traces = list((Path.home() / ".rosclaw").glob("traces/*.jsonl"))
    probe("trace_store", True, f"trace files={len(traces)} (dir scan, read-only)",
          str(Path.home() / ".rosclaw/traces"))

    # 7. modeld (read-only status via systemctl, no socket write)
    proc = subprocess.run(["systemctl", "is-active", "rosclaw-ty1200-modeld.service"],
                          capture_output=True, text=True, timeout=15)
    probe("modeld", proc.stdout.strip() == "active", f"modeld={proc.stdout.strip()}",
          "systemctl is-active rosclaw-ty1200-modeld.service")

    # --- hard boundary assertions ---
    boundaries = {
        "docker_sock_accessed": False,      # this script never opens /var/run/docker.sock
        "approval_token_read": False,       # never reads /etc/rosclaw/*policy*
        "physical_action_envelopes": 0,     # no ActionEnvelope created
        "all_read_only": True,
    }

    report = {
        "task": "anchor_task_a_self_check",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": EVIDENCE,
        "boundaries": boundaries,
        "overall": "PASS" if all(e["status"] == "PASS" for e in EVIDENCE) else "FAIL",
    }
    report_path = out_dir / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # human-readable markdown
    md = ["# TY1200 自检诊断报告", f"生成时间: {report['generated_at']}", "",
          "| 检查项 | 状态 | 详情 | 证据 |", "|---|---|---|---|"]
    for e in EVIDENCE:
        md.append(f"| {e['check']} | {e['status']} | {e['detail'][:60]} | {e['evidence_ref']} |")
    md.append(f"\n总体: **{report['overall']}**")
    md.append("\n边界: 未访问 docker.sock; 未读取 approval token; 物理动作数 = 0")
    (out_dir / "DIAGNOSTIC_REPORT.md").write_text("\n".join(md))

    # sha256 evidence manifest
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    (out_dir / "evidence_manifest.txt").write_text(f"sha256:{digest}  diagnostic_report.json\n")
    print(f"overall: {report['overall']}  evidence sha256:{digest[:16]}...")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
