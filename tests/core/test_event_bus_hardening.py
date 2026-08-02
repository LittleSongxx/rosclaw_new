"""EventBus hardening tests (评审 P1 任务).

Covers:
- per-subscriber payload immutability (cross-subscriber isolation)
- json/yaml serializers keep working on frozen payloads
- sync subscriber timeout isolation (publisher not blocked past bound)
- circuit breaker (repeat-timeout subscriber gets unsubscribed)
- priority history lane (CRITICAL evidence survives bulk traffic)
- metrics exposure
- EventBus() 默认行为与全局硬化总线行为分层
"""

from __future__ import annotations

import json
import time

import pytest
import yaml

from rosclaw.core.event_bus import Event, EventBus, EventPriority, get_global_event_bus
from rosclaw.core.immutable import FrozenDict, FrozenList, freeze, thaw


class TestImmutablePayload:
    def test_subscriber_cannot_mutate_payload(self):
        bus = EventBus()
        seen = []

        def mutator(e: Event) -> None:
            with pytest.raises(TypeError):
                e.payload["v"] = 999
            seen.append("mutator-blocked")

        bus.subscribe("t", mutator)
        bus.subscribe("t", lambda e: seen.append(e.payload["v"]))
        bus.publish(Event(topic="t", payload={"v": 1}, source="test"))
        assert seen == ["mutator-blocked", 1]
        # 第二个订阅者看到的是原始值, 历史也是
        assert bus.get_history("t")[0].payload["v"] == 1

    def test_nested_mutation_blocked(self):
        bus = EventBus()

        def mutator(e: Event) -> None:
            with pytest.raises(TypeError):
                e.payload["nested"]["x"].append(4)

        bus.subscribe("t", mutator)
        bus.publish(Event(topic="t", payload={"nested": {"x": [1, 2]}}, source="test"))

    def test_frozen_payload_serializes_normally(self):
        bus = EventBus()
        bus.subscribe("t", lambda e: None)
        bus.publish(Event(topic="t", payload={"a": [1, {"b": 2}]}, source="test"))
        payload = bus.get_history("t")[0].payload
        assert json.loads(json.dumps(payload)) == {"a": [1, {"b": 2}]}
        assert yaml.safe_load(yaml.safe_dump(payload)) == {"a": [1, {"b": 2}]}

    def test_thaw_restores_mutable_copy(self):
        frozen = freeze({"a": [1, 2]})
        plain = thaw(frozen)
        plain["a"].append(3)
        assert plain == {"a": [1, 2, 3]}
        assert frozen["a"] == [1, 2]

    def test_opt_out_keeps_legacy_mutable_delivery(self):
        bus = EventBus(freeze_payloads=False)

        def mutator(e: Event) -> None:
            e.payload["v"] = 999

        seen = []
        bus.subscribe("t", mutator)
        bus.subscribe("t", lambda e: seen.append(e.payload["v"]))
        bus.publish(Event(topic="t", payload={"v": 1}, source="test"))
        # legacy 模式: 订阅者与 history 共享同一可变对象 (即 finding #3 的机制;
        # 仅供需要旧语义的调用方显式选择)
        assert seen == [999]
        assert bus.get_history("t")[0].payload["v"] == 999


class TestTimeoutIsolation:
    def test_slow_subscriber_bounded(self):
        bus = EventBus(subscriber_timeout=0.2)

        def slow(_e: Event) -> None:
            time.sleep(1.0)

        bus.subscribe("t", slow)
        bus.subscribe("t", lambda e: None)
        started = time.perf_counter()
        bus.publish(Event(topic="t", payload={}, source="test"))
        elapsed = time.perf_counter() - started
        assert elapsed < 0.6, f"publisher blocked {elapsed:.2f}s"
        stats = bus.get_stats()
        assert stats["metrics"]["subscriber_timeouts"] == 1

    def test_circuit_breaker_unsubscribes_poison_subscriber(self):
        bus = EventBus(subscriber_timeout=0.05, max_timeouts_per_subscriber=3)

        def slow(_e: Event) -> None:
            time.sleep(1.0)

        bus.subscribe("t", slow)
        for _ in range(3):
            bus.publish(Event(topic="t", payload={}, source="test"))
        assert bus.subscriber_count("t") == 0  # 已被摘除
        assert bus.get_stats()["metrics"]["circuit_breaks"] == 1

    def test_healthy_subscriber_stays(self):
        bus = EventBus(subscriber_timeout=0.5)
        got = []
        bus.subscribe("t", lambda e: got.append(e.payload["i"]))
        for i in range(3):
            bus.publish(Event(topic="t", payload={"i": i}, source="test"))
        assert got == [0, 1, 2]
        assert bus.subscriber_count("t") == 1


class TestPriorityLane:
    def test_critical_events_survive_bulk_traffic(self):
        bus = EventBus()
        bus._max_history = 100
        bus.subscribe("#", lambda e: None)
        bus.publish(Event(topic="estop", payload={}, source="s",
                          priority=EventPriority.CRITICAL))
        for i in range(500):
            bus.publish(Event(topic=f"bulk.{i % 5}", payload={"i": i}, source="s",
                              priority=EventPriority.LOW))
        # 主流水线被挤出, 但预留道保留 CRITICAL 证据
        assert not any(e.topic == "estop" for e in bus.get_history())
        lane = bus._priority_history
        assert any(e.topic == "estop" for e in lane)
        assert bus.get_stats()["metrics"]["dropped_events"] > 0


class TestMetrics:
    def test_stats_expose_hardening_metrics(self):
        bus = EventBus(subscriber_timeout=0.5)
        bus.subscribe("t", lambda e: None)
        bus.publish(Event(topic="t", payload={}, source="test"))
        stats = bus.get_stats()
        assert stats["metrics"]["published"] == 1
        assert stats["metrics"]["delivered"] == 1
        assert stats["hardening"]["freeze_payloads"] is True
        assert stats["hardening"]["subscriber_timeout"] == 0.5

    def test_global_bus_is_hardened(self):
        bus = get_global_event_bus()
        stats = bus.get_stats()
        assert stats["hardening"]["subscriber_timeout"] is not None


class TestFreezeThawUnit:
    def test_freeze_idempotent(self):
        once = freeze({"a": 1})
        assert freeze(once) is once

    def test_frozen_types_are_native_subclasses(self):
        f = freeze({"k": [1]})
        assert isinstance(f, dict) and isinstance(f["k"], list)
        assert isinstance(f, FrozenDict) and isinstance(f["k"], FrozenList)

    def test_tuple_becomes_tuple_of_frozen(self):
        f = freeze(({"a": 1},))
        assert isinstance(f, tuple)
        with pytest.raises(TypeError):
            f[0]["a"] = 2
