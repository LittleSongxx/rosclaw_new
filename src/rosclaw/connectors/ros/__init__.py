"""ROS Connector package.

ROS integration through rosbridge. No ROS Python client libraries are
imported at the top level so that ROSClaw remains installable without ROS.
"""

from __future__ import annotations

from rosclaw.connectors.ros import compiler, discovery, embodiment, provider, transport
from rosclaw.connectors.ros.embodiment import (
    EmbodimentCard,
    EmbodimentCardError,
    load_embodiment_card,
)

__all__ = [
    "EmbodimentCard",
    "EmbodimentCardError",
    "compiler",
    "discovery",
    "embodiment",
    "load_embodiment_card",
    "provider",
    "transport",
]
