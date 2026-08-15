"""
ROSClaw Memory - Experience Grounding Engine

Interface to SeekDB (Shared Knowledge Plane).
Stores and retrieves robot experiences, skills, and world knowledge.
"""

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import (
    ROSCLAW_STRUCTURED_SCHEMAS,
    SEEKDB_SCHEMAS,
    InMemoryKnowledgeStore,
    InMemoryStructuredStore,
    SeekDBClient,
    SeekDBMemoryClient,
    SeekDBMySQLClient,
    SeekDBSQLiteClient,
    SeekDBSQLStore,
    SQLiteKnowledgeStore,
    SQLiteStructuredStore,
    StructuredStore,
)
from rosclaw.memory.types import ArtifactRef, FailureMemory, PraxisEvent

# Backward-compatible aliases for documentation
SQLiteSeekDB = SQLiteStructuredStore
MemorySeekDB = InMemoryStructuredStore

__all__ = [
    "SeekDBClient",
    "InMemoryKnowledgeStore",
    "SQLiteKnowledgeStore",
    "SeekDBMySQLClient",
    "SEEKDB_SCHEMAS",
    "ROSCLAW_STRUCTURED_SCHEMAS",
    "MemoryInterface",
    "StructuredStore",
    "InMemoryStructuredStore",
    "SeekDBMemoryClient",
    "MemorySeekDB",
    "SeekDBSQLStore",
    "SQLiteStructuredStore",
    "SeekDBSQLiteClient",
    "SQLiteSeekDB",
    "PraxisEvent",
    "FailureMemory",
    "ArtifactRef",
]
