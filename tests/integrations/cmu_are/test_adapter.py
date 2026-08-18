from __future__ import annotations

from collections import deque

import pytest

from rosclaw.connectors.ros.transport import RosbridgeTransport, RosTransportResult
from rosclaw.integrations.cmu_are.adapter import (
    CmuAreRosbridgeAdapter,
    CmuAreTransportError,
)


class _ScriptedTransport:
    def __init__(self, responses: list[RosTransportResult]) -> None:
        self.responses = deque(responses)
        self.connected = False
        self.requests: list[dict[str, object]] = []

    def connect(self) -> RosTransportResult:
        self.connected = True
        return RosTransportResult(ok=True)

    def call_service(self, service: str, args: dict[str, object], **_kwargs):
        self.requests.append({"op": "call_service", "service": service, "args": args})
        return self.responses.popleft() if self.responses else RosTransportResult(ok=False)

    def subscribe_once(self, topic: str, **_kwargs):
        self.requests.append({"op": "subscribe", "topic": topic})
        return (
            self.responses.popleft()
            if self.responses
            else RosTransportResult(ok=False, error=f"No message received on {topic} within 0.1s")
        )

    def advertise(self, topic: str, msg_type: str, **_kwargs):
        self.requests.append({"op": "advertise", "topic": topic, "type": msg_type})
        return RosTransportResult(ok=True)

    def publish(self, topic: str, msg: dict[str, object]):
        self.requests.append({"op": "publish", "topic": topic, "msg": msg})
        return RosTransportResult(ok=True)


def _timeout(topic: str = "/state_estimation") -> RosTransportResult:
    return RosTransportResult(ok=False, error=f"No message received on {topic} within 0.1s")


def test_observation_timeout_does_not_change_connection_generation() -> None:
    transport = _ScriptedTransport([_timeout(), _timeout(), _timeout()])
    adapter = CmuAreRosbridgeAdapter(transport)

    first = adapter.connect()
    assert adapter.read_exploration_state(timeout_sec=0.2) is None
    assert adapter.connection is not None
    assert adapter.connection.generation == first.generation == 1


def test_broken_observation_invalidates_connection_and_next_connect_increments() -> None:
    transport = _ScriptedTransport([RosTransportResult(ok=False, error="connection reset by peer")])
    adapter = CmuAreRosbridgeAdapter(transport)
    first = adapter.connect()

    with pytest.raises(CmuAreTransportError):
        adapter.read_odom(timeout_sec=0.1)
    assert adapter.connection is None

    second = adapter.connect()
    assert second.generation == first.generation + 1
    assert second.connection_id != first.connection_id


def test_rosbridge_receive_timeout_keeps_the_live_socket_generation() -> None:
    class _IdleSocket:
        connected = True

        def settimeout(self, _timeout: float) -> None:
            return None

        def recv(self):
            raise TimeoutError("idle subscription")

    transport = RosbridgeTransport(max_retries=0)
    socket = _IdleSocket()
    transport._ws = socket

    result = transport.receive(timeout_sec=0.01)
    assert result.ok is False
    assert transport._ws is socket
