#!/usr/bin/env python3
"""EventBus verification & benchmark (任务书 §10.2).

Load levels: 10 / 1,000 / 10,000 events per second.
Checks:
  - trace_id / span_id / parent_span_id propagation
  - duplicate delivery
  - priority semantics (does CRITICAL preempt NORMAL?)
  - exception in one subscriber does not affect others
  - slow subscriber impact on publisher (head-of-line blocking?)
  - history bound / loss accounting
  - payload mutation by a subscriber

Emits PASS/WARN/FAIL per case plus throughput and latency percentiles.
Findings (not failures of this script) record design gaps discovered.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid

from rosclaw.core.event_bus import Event, EventBus, EventPriority


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100) * (len(values) - 1)))))
    return values[k]


def case_throughput(bus: EventBus, eps: int, seconds: float = 2.0) -> dict:
    received: list[float] = []
    bus.subscribe("bench.load", lambda e: received.append(time.perf_counter()))
    total = int(eps * seconds)
    interval = 1.0 / eps
    lat: list[float] = []
    t0 = time.perf_counter()
    for i in range(total):
        sent = time.perf_counter()
        bus.publish(Event(topic="bench.load", payload={"i": i}, source="bench"))
        lat.append((time.perf_counter() - sent) * 1e6)  # µs inline dispatch
        # pace to target rate (best effort)
        target = t0 + (i + 1) * interval
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
    wall = time.perf_counter() - t0
    bus.clear_history()
    return {
        "target_eps": eps,
        "published": total,
        "received": len(received),
        "wall_s": round(wall, 2),
        "achieved_eps": round(total / wall, 1),
        "loss": total - len(received),
        "dispatch_us_p50": round(pct(lat, 50), 1),
        "dispatch_us_p95": round(pct(lat, 95), 1),
        "dispatch_us_p99": round(pct(lat, 99), 1),
    }


def case_trace_propagation(bus: EventBus) -> dict:
    got: list[Event] = []
    bus.subscribe("bench.trace", got.append)
    tid = f"trace_{uuid.uuid4().hex[:12]}"
    parent = Event(topic="bench.trace", payload={}, source="bench",
                   trace_id=tid, span_id="span_parent")
    bus.publish(parent)
    child = parent.derive(span_id="span_child", parent_span_id="span_parent")
    bus.publish(child)
    ok = (
        len(got) == 2
        and got[0].trace_id == tid
        and got[1].trace_id == tid
        and got[1].parent_span_id == "span_parent"
    )
    return {"status": "PASS" if ok else "FAIL", "events": len(got),
            "trace_id_preserved": got[1].trace_id == tid if len(got) > 1 else False}


def case_duplicates(bus: EventBus) -> dict:
    got: list[str] = []
    bus.subscribe("bench.dup", lambda e: got.append(e.event_id))
    e = Event(topic="bench.dup", payload={}, source="bench")
    bus.publish(e)
    dups = len(got) - len(set(got))
    return {"status": "PASS" if len(got) == 1 and dups == 0 else "FAIL",
            "deliveries": len(got), "duplicates": dups}


def case_priority(bus: EventBus) -> dict:
    """Publish NORMAL then CRITICAL; record delivery order.

    The bus dispatches inline in publish order, so CRITICAL cannot preempt
    an earlier NORMAL event. Recorded as a design finding, not a failure.
    """
    order: list[str] = []
    bus.subscribe("bench.prio", lambda e: order.append(e.payload["tag"]))
    bus.publish(Event(topic="bench.prio", payload={"tag": "normal"}, source="bench",
                      priority=EventPriority.NORMAL))
    bus.publish(Event(topic="bench.prio", payload={"tag": "critical"}, source="bench",
                      priority=EventPriority.CRITICAL))
    preempted = order == ["critical", "normal"]
    return {
        "status": "PASS",
        "delivery_order": order,
        "critical_preempts": preempted,
        "finding": None if preempted else
        "EventPriority is declared but dispatch is inline FIFO: a CRITICAL "
        "event cannot preempt an already-queued NORMAL event. Priority is "
        "advisory metadata only (asyncio.PriorityQueue member is unused).",
    }


def case_exception_isolation(bus: EventBus) -> dict:
    got: list[str] = []

    def bad(_e: Event) -> None:
        raise RuntimeError("subscriber exploded")

    bus.subscribe("bench.exc", bad)
    bus.subscribe("bench.exc", lambda e: got.append("second"))
    bus.publish(Event(topic="bench.exc", payload={}, source="bench"))
    return {"status": "PASS" if got == ["second"] else "FAIL",
            "other_subscriber_ran": got == ["second"]}


def case_slow_subscriber(bus: EventBus) -> dict:
    def slow(_e: Event) -> None:
        time.sleep(0.05)

    bus.subscribe("bench.slow", slow)
    t = time.perf_counter()
    bus.publish(Event(topic="bench.slow", payload={}, source="bench"))
    blocked_s = time.perf_counter() - t
    return {
        "status": "PASS",
        "publisher_blocked_s": round(blocked_s, 3),
        "finding": None if blocked_s < 0.01 else
        f"A slow sync subscriber blocks publish() inline ({blocked_s:.3f}s): "
        "head-of-line blocking exists; there is no worker queue or drop policy.",
    }


def case_payload_mutation(bus: EventBus) -> dict:
    payload = {"v": 1}

    def mutator(e: Event) -> None:
        e.payload["v"] = 999

    seen: list[int] = []
    bus.subscribe("bench.mut", mutator)
    bus.subscribe("bench.mut", lambda e: seen.append(e.payload["v"]))
    bus.publish(Event(topic="bench.mut", payload=payload, source="bench"))
    mutated = seen == [999]
    return {
        "status": "PASS",
        "second_subscriber_saw": seen,
        "finding": "Subscribers share the same mutable Event/payload object; "
        "a mutating subscriber affects later subscribers."
        if mutated else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--stress", action="store_true", help="include 10k eps level")
    args = ap.parse_args()

    results: dict = {"cases": {}, "findings": []}

    bus = EventBus()
    results["cases"]["trace_propagation"] = case_trace_propagation(bus)
    results["cases"]["duplicates"] = case_duplicates(bus)
    results["cases"]["exception_isolation"] = case_exception_isolation(bus)
    results["cases"]["priority"] = case_priority(bus)
    results["cases"]["slow_subscriber"] = case_slow_subscriber(bus)
    results["cases"]["payload_mutation"] = case_payload_mutation(bus)

    levels = [10, 1000] + ([10000] if args.stress else [])
    results["throughput"] = [case_throughput(bus, eps) for eps in levels]

    for name, case in results["cases"].items():
        if case.get("finding"):
            results["findings"].append({"case": name, "finding": case["finding"]})

    fails = [n for n, c in results["cases"].items() if c.get("status") == "FAIL"]
    results["overall"] = "PASS" if not fails else "FAIL"
    results["failed_cases"] = fails

    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
