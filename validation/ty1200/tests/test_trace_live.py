"""Trace live validation (任务书 §十三).

Hard gates:
    trace_parent_integrity == 100%
    secret_leaks == 0
    raw_cot_leaks_standard_mode == 0
    binary_payload_embedded == 0
    blocked_status_accuracy == 100%

Covers: causal tree integrity, one-root-per-mission, time monotonicity,
redaction of secrets/<think>/binary, BLOCKED status surfacing, JSONL export
+ re-read, and TraceStore query over a large span set.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from rosclaw.observability.exporters.jsonl import JsonlTraceExporter
from rosclaw.observability.redaction import TraceRedactor
from rosclaw.observability.schema import CaptureMode
from rosclaw.observability.store import TraceStore
from rosclaw.observability.tracer import Tracer

SECRET_LEAKS = 0
RAW_COT_LEAKS = 0
BINARY_EMBEDDED = 0


def _mission_trace(exporter: JsonlTraceExporter) -> str:
    """Build the §13.2 mission span tree and return its trace_id."""
    tracer = Tracer(exporters=[exporter])
    spans = [
        ("knowledge.preflight", "CONTEXT"),
        ("memory.retrieve", "MEMORY"),
        ("provider.invoke", "LLM"),
        ("decision", "PLANNER"),
        ("validate", "SANDBOX"),
        ("simulation", "ROBOT_ACTION"),
        ("observation", "ROBOT_STATE"),
        ("evaluation", "CRITIC"),
        ("store", "MEMORY"),
        ("knowledge usage", "CONTEXT"),
    ]
    # Spans attach to their parent only when entered (contextvars), so build
    # the tree with real context managers — the same way runtime code does.
    with tracer.start_span("mission", "MISSION", mission_id="m-trace-live") as root:
        for name, kind in spans:
            with tracer.start_span(name, kind) as child:
                child.set_attribute("api_key", "sk-should-never-appear")
                child.set_input({"prompt": "<think>raw private reasoning</think> do the thing"})
                child.set_output({"blob": b"\x89binary-frame", "note": "ok"})
        with tracer.start_span("validate-blocked", "SANDBOX") as blocked:
            blocked.set_status("BLOCKED", "collision predicted")
        trace_id = root.trace_id
    tracer.close()
    return trace_id


def test_causal_tree_integrity(tmp_path: Path):
    exporter = JsonlTraceExporter(output_path=tmp_path / "traces.jsonl")
    trace_id = _mission_trace(exporter)

    records = [
        json.loads(line)
        for line in (tmp_path / "traces.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 12  # root + 10 children + blocked child
    by_id = {r["span_id"]: r for r in records}
    roots = [r for r in records if not r.get("parent_span_id")]
    assert len(roots) == 1, "exactly one root span per mission"

    integrity_failures = 0
    for r in records:
        parent = r.get("parent_span_id")
        if parent and parent not in by_id:
            integrity_failures += 1
        if r.get("ended_at") is not None and r["ended_at"] < r["started_at"]:
            integrity_failures += 1
        if r["trace_id"] != trace_id:
            integrity_failures += 1
    assert integrity_failures == 0


def test_redaction_hard_gates(tmp_path: Path):
    global SECRET_LEAKS, RAW_COT_LEAKS, BINARY_EMBEDDED
    exporter = JsonlTraceExporter(output_path=tmp_path / "traces.jsonl")
    _mission_trace(exporter)
    raw = (tmp_path / "traces.jsonl").read_text()

    SECRET_LEAKS += raw.count("sk-should-never-appear")
    RAW_COT_LEAKS += raw.count("raw private reasoning") + raw.count("<think>")
    BINARY_EMBEDDED += raw.count("binary-frame")

    assert SECRET_LEAKS == 0
    assert RAW_COT_LEAKS == 0
    assert BINARY_EMBEDDED == 0
    # Positive controls: markers must be present instead.
    assert "[REDACTED]" in raw
    assert "THINK_BLOCK_OMITTED" in raw
    assert "inline-binary-omitted" in raw


def test_blocked_status_accuracy(tmp_path: Path):
    exporter = JsonlTraceExporter(output_path=tmp_path / "traces.jsonl")
    _mission_trace(exporter)
    records = [
        json.loads(line)
        for line in (tmp_path / "traces.jsonl").read_text().splitlines()
        if line.strip()
    ]
    blocked = [r for r in records if r["name"] == "validate-blocked"]
    assert len(blocked) == 1
    assert blocked[0]["status"] == "BLOCKED"
    # no error/blocked span may be written as OK
    bad = [r for r in records if r["status"] == "OK" and "blocked" in r["name"]]
    assert bad == []


def test_export_reread_via_store(tmp_path: Path):
    exporter = JsonlTraceExporter(output_path=tmp_path / "traces.jsonl")
    trace_id = _mission_trace(exporter)
    store = TraceStore(path=tmp_path / "traces.jsonl")
    trace = store.get_trace(trace_id)
    assert trace is not None
    spans = trace.get("spans") or trace.get("records") or []
    assert len(spans) == 12


def test_store_query_performance_100k(tmp_path: Path):
    """100k-span store: list/get queries must stay interactive."""
    path = tmp_path / "bulk.jsonl"
    n = 100_000
    t0 = time.perf_counter()
    with path.open("w") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {
                        "trace_id": f"trace_{i % 1000:024d}",
                        "span_id": f"span_{i}",
                        "parent_span_id": None,
                        "name": f"op-{i % 50}",
                        "span_kind": "MISSION",
                        "source": "bench",
                        "operation": "bench",
                        "started_at": 1000.0 + i,
                        "ended_at": 1000.0 + i + 0.001,
                        "duration_ms": 1.0,
                        "status": "OK",
                        "attributes": {},
                    }
                )
                + "\n"
            )
    write_s = time.perf_counter() - t0

    store = TraceStore(path=path)
    t1 = time.perf_counter()
    listing = store.list_traces(limit=50)
    list_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    store.get_trace("trace_000000000000000000000042")
    get_s = time.perf_counter() - t2

    metrics = {
        "spans": n,
        "write_s": round(write_s, 2),
        "list_traces_s": round(list_s, 2),
        "get_trace_s": round(get_s, 2),
        "listed": len(listing),
    }
    report = os.environ.get("TY1200_VALIDATION_REPORT_DIR")
    if report:
        Path(report).mkdir(parents=True, exist_ok=True)
        (Path(report) / "trace_store_perf.json").write_text(json.dumps(metrics, indent=2))
    # interactive thresholds (local NVMe): list < 30s, single get < 30s
    assert list_s < 30 and get_s < 30
    assert listing
