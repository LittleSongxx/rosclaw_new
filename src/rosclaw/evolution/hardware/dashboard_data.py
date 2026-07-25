"""Dashboard data assembly for the Evo-RPS evolution page (§十三).

Six sections, all sourced from the experiment's evidence manifest and the
namespace registry — the page shows the truth, including rolled-back
candidates and aborted canaries; a "green wall" it is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SECTION_ORDER = (
    "physical_system",
    "flywheel",
    "candidate_comparison",
    "decision_evidence",
    "physical_execution",
    "promotion_state",
)

FLYWHEEL_STAGES = [
    ("prepare", "Prepare"),
    ("baseline_session", "Baseline"),
    ("distill_gate", "Distill"),
    ("propose", "Propose"),
    ("candidate_evaluated", "Validate"),
    ("canary_session", "Canary"),
    ("promotion_decision", "Promotion"),
    ("recurrence_session", "Recurrence"),
]


def _loads(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def assemble_evolution_page(
    *,
    experiment_id: str,
    evidence_root: Path,
    candidates: list[dict[str, Any]],
    promoted_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_path = evidence_root / "evidence_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"entries": []}
    )
    entries = manifest.get("entries", [])
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_kind.setdefault(entry["kind"], []).append(entry)

    baseline = by_kind.get("baseline_session", [])
    canary = by_kind.get("canary_session", [])
    aborts = by_kind.get("canary_aborted", [])
    decisions = by_kind.get("promotion_decision", [])
    recurrence = by_kind.get("recurrence_session", [])
    blocked = by_kind.get("recurrence_blocked", [])

    physical_system = {
        "experiment_id": experiment_id,
        "config_hash": manifest.get("config_hash"),
        "baseline_sessions": len(baseline),
        "peak_temperatures": [s.get("peak_temperature") for s in baseline + canary],
        "camera": "realsense (formal; mock forbidden)",
        "safety_aborts": len(aborts),
    }

    flywheel = [
        {
            "stage": label,
            "count": len(by_kind.get(kind, [])),
            "done": bool(by_kind.get(kind)),
        }
        for kind, label in FLYWHEEL_STAGES
    ]

    comparison: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for session in canary:
        arm = str(session.get("arm") or "")
        key = arm[0] if arm[:1] in comparison else None
        if key:
            comparison[key].append(
                {
                    "block": session.get("block"),
                    "invalid_rate": session.get("invalid_rate"),
                    "verified_rate": session.get("verified_rate"),
                    "peak_temperature": session.get("peak_temperature"),
                }
            )
    comparison["baseline"] = [
        {
            "invalid_rate": s.get("invalid_rate"),
            "verified_rate": s.get("verified_rate"),
            "peak_temperature": s.get("peak_temperature"),
        }
        for s in baseline
    ]

    decision_evidence = [
        {
            "candidate_id": c.get("candidate_id"),
            "state": c.get("state"),
            "changes": _loads(c.get("changes")),
            "failed_gate": c.get("failed_gate"),
            "gate_verdicts": _loads(c.get("gate_verdicts")) or [],
        }
        for c in candidates
    ]

    physical_execution = {
        "canary_sessions": [
            {
                "arm": s.get("arm"),
                "practice_id": s.get("practice_id"),
                "verify_rc": (s.get("verify") or {}).get("rc"),
                "candidate_lifecycle": s.get("candidate_lifecycle"),
            }
            for s in canary
        ],
        "aborts": aborts,
    }

    promotion_state = {
        "decisions": [
            {
                "candidate_id": d.get("candidate_id"),
                "decision": d.get("decision"),
                "scope": d.get("scope"),
                "checks": d.get("checks"),
            }
            for d in decisions
        ],
        "promoted_rules": [
            {
                "rule_id": r.get("rule_id"),
                "candidate_id": r.get("candidate_id"),
                "scope": r.get("scope"),
                "status": r.get("status"),
            }
            for r in promoted_rules
        ],
        "recurrence": [
            {
                "rule_id": r.get("rule_id"),
                "invalid_rate": r.get("invalid_rate"),
                "proof": r.get("proof"),
            }
            for r in recurrence
        ],
        "recurrence_blocked": blocked,
    }

    return {
        "experiment_id": experiment_id,
        "sections": {
            "physical_system": physical_system,
            "flywheel": flywheel,
            "candidate_comparison": comparison,
            "decision_evidence": decision_evidence,
            "physical_execution": physical_execution,
            "promotion_state": promotion_state,
        },
    }


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>Evo-RPS {experiment_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0f1419; color: #e6e6e6; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.05rem; color: #8ab4f8; margin-top: 1.6rem; }}
table {{ border-collapse: collapse; margin: 0.4rem 0; }}
td, th {{ border: 1px solid #2a3644; padding: 0.25rem 0.6rem; font-size: 0.85rem; }}
.stage-done {{ color: #7ee787; }} .stage-pending {{ color: #6e7681; }}
.bad {{ color: #ff7b72; }} .good {{ color: #7ee787; }}
pre {{ background: #161d26; padding: 0.6rem; overflow-x: auto; font-size: 0.8rem; }}
</style></head>
<body>
<h1>ROSClaw Evo-RPS 自进化证据 — {experiment_id}</h1>
<div id="app">loading…</div>
<script>
fetch("/api/evolution/evo-rps/{experiment_id}").then(r => r.json()).then(data => {{
  const s = data.sections;
  let html = "";
  html += "<h2>1. 当前物理系统</h2><pre>" + JSON.stringify(s.physical_system, null, 2) + "</pre>";
  html += "<h2>2. 飞轮</h2><table><tr><th>阶段</th><th>状态</th></tr>" +
    s.flywheel.map(f => `<tr><td>${{f.stage}}</td><td class="${{f.done ? 'stage-done' : 'stage-pending'}}">${{f.done ? '✓ ' + f.count : '—'}}</td></tr>`).join("") + "</table>";
  html += "<h2>3. Candidate 对比</h2><pre>" + JSON.stringify(s.candidate_comparison, null, 2) + "</pre>";
  html += "<h2>4. Decision Evidence</h2><table><tr><th>candidate</th><th>state</th><th>changes</th><th>failed gate</th></tr>" +
    s.decision_evidence.map(c => `<tr><td>${{c.candidate_id}}</td><td class="${{c.state === 'ROLLED_BACK' || c.state === 'REJECTED' ? 'bad' : 'good'}}">${{c.state}}</td><td>${{JSON.stringify(c.changes)}}</td><td>${{c.failed_gate || ''}}</td></tr>`).join("") + "</table>";
  html += "<h2>5. 物理执行</h2><pre>" + JSON.stringify(s.physical_execution, null, 2) + "</pre>";
  html += "<h2>6. 晋升状态</h2><pre>" + JSON.stringify(s.promotion_state, null, 2) + "</pre>";
  document.getElementById("app").innerHTML = html;
}});
</script>
</body></html>"""


def render_evolution_page_html(experiment_id: str) -> str:
    return _PAGE_TEMPLATE.replace("{experiment_id}", experiment_id)
