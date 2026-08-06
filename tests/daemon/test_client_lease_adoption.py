from __future__ import annotations

import time

from rosclaw.daemon.client import DaemonClient


def test_track_action_lease_reads_owner_checked_status_and_schedules_renewal(
    monkeypatch,
) -> None:
    client = DaemonClient.__new__(DaemonClient)
    client._action_leases = {}
    status = {
        "action_id": "action-via-operatord",
        "session_id": "action-via-operatord",
        "state": "RUNNING",
        "action_lease": {"renew_interval_ms": 250},
    }
    monkeypatch.setattr(client, "get_action_status", lambda action_id: dict(status))

    observed = client.track_action_lease("action-via-operatord")

    assert observed == status
    assert client._action_leases["action-via-operatord"][0] == "action-via-operatord"
    assert client._action_leases["action-via-operatord"][2] == 0.25


def test_status_poll_renews_adopted_lease(monkeypatch) -> None:
    client = DaemonClient.__new__(DaemonClient)
    client._action_leases = {
        "action-via-operatord": ("action-via-operatord", time.monotonic() - 1.0, 0.25)
    }
    calls = []

    def call(method, params):
        calls.append((method, params))
        if method == "action.lease.renew":
            return {"action_id": params["action_id"], "action_lease": {"active": True}}
        return {"action_id": params["action_id"], "state": "RUNNING"}

    monkeypatch.setattr(client, "call", call)

    status = client.get_action_status("action-via-operatord")

    assert status["state"] == "RUNNING"
    assert [method for method, _params in calls] == [
        "action.lease.renew",
        "action.status",
    ]
    assert client._action_leases["action-via-operatord"][1] > time.monotonic()
