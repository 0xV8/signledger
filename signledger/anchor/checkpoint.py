"""``Checkpoint`` - a signed statement about a chain head (contract section 8).

A checkpoint is what makes **tail truncation** detectable. A bare hash chain cannot see
it: lopping the last N entries off leaves a chain that is internally perfectly
consistent. Binding ``head_hash`` and ``entry_count`` into a record published somewhere
the adversary does not control turns that silent deletion into a ``HEAD_MISMATCH`` plus
a ``COUNT_MISMATCH``.

This module holds the record only. Signing lives in ``signledger/crypto/signatures.py``
and publishing in the anchor sinks.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

__all__ = ["Checkpoint"]


@dataclass(frozen=True)
class Checkpoint:
    """An immutable snapshot of a ledger's head at a point in time.

    ``signature``, ``key_id`` and ``format_version`` carry defaults so an unsigned
    checkpoint can be built in one call; the full positional form given in the contract
    is unaffected.
    """

    ledger_id: str
    sequence: int
    head_hash: str
    entry_count: int
    merkle_root: str
    timestamp: datetime
    signature: Optional[str] = None
    key_id: Optional[str] = None
    format_version: int = 2
