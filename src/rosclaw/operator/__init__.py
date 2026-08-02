"""Operator Broker: approvals and MissionGrants (ADR-0006).

Maturity: **experimental**. The broker owns authorization decisions; the
agent only ever sees public grant scope. Private signatures never leave
``rosclaw.operator``.
"""

from rosclaw.operator.broker import GrantDeniedError, OperatorBroker

__all__ = ["GrantDeniedError", "OperatorBroker"]

MATURITY = "experimental"
