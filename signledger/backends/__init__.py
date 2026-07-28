"""Storage backends (architecture contract, section 4).

``StorageBackend`` is the protocol; ``BackendConformanceSuite`` is the executable proof
that an implementation of it is interchangeable with the shipped ones.

Only the dependency-free backends are imported eagerly. ``postgresql`` and ``mongodb``
need optional drivers, so they stay importable by path
(``signledger.backends.postgresql``) without making ``import signledger.backends`` fail
on a machine that has neither. Each backend has exactly one module name - no aliases.
"""

from signledger.backends.base import (
    CAP_APPEND_ONLY_ENFORCED,
    CAP_ATOMIC_APPEND,
    CAP_DURABLE_FSYNC,
    CAP_UNIQUE_CHAIN_LINK,
    CAP_WORM,
    CAPABILITY_TOKENS,
    BackendConformanceSuite,
    ConformanceTestMixin,
    StorageBackend,
)
from signledger.backends.memory import MemoryBackend
from signledger.backends.sqlite import SQLiteBackend

__all__ = [
    "CAPABILITY_TOKENS",
    "CAP_APPEND_ONLY_ENFORCED",
    "CAP_ATOMIC_APPEND",
    "CAP_DURABLE_FSYNC",
    "CAP_UNIQUE_CHAIN_LINK",
    "CAP_WORM",
    "BackendConformanceSuite",
    "ConformanceTestMixin",
    "MemoryBackend",
    "SQLiteBackend",
    "StorageBackend",
]
