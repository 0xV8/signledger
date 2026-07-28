"""In-process storage backend.

Useful for tests, for short-lived processes and as the reference implementation of the
section 4 semantics: whatever this module does, every durable backend must do too.

It is deliberately **not** a WORM store. Nothing here survives the process, so
``capabilities()`` never claims ``append_only_enforced`` or ``durable_fsync`` - an
in-memory ledger reporting WORM would be a lie that ``assurance_level()`` would repeat.
"""

import threading
from datetime import datetime
from typing import Callable, Dict, FrozenSet, Iterator, List, Optional

from signledger.anchor.checkpoint import Checkpoint
from signledger.backends.base import (
    CAP_ATOMIC_APPEND,
    CAP_UNIQUE_CHAIN_LINK,
    check_link,
    checkpoint_from_row,
    checkpoint_to_row,
    entry_to_row,
    retry_on_conflict,
    validate_window,
)
from signledger.core.entry import Entry
from signledger.core.exceptions import StorageError

__all__ = ["MemoryBackend"]


class MemoryBackend:
    """Keeps one ledger's chain in process memory."""

    def __init__(self) -> None:
        # Re-entrant: ``append_atomic`` calls the build callback while holding the lock,
        # and a callback that reads the backend should be slow, not deadlocked.
        self._lock = threading.RLock()
        self._entries: List[Entry] = []
        self._by_entry_id: Dict[str, Entry] = {}
        self._previous_hashes: Dict[str, int] = {}
        self._checkpoints: List[Checkpoint] = []
        self._closed = False

    # -- lifecycle ---------------------------------------------------------------------

    def close(self) -> None:
        """Release the store. Idempotent."""
        with self._lock:
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise StorageError("backend is closed")

    # -- append ------------------------------------------------------------------------

    def append_atomic(self, build: Callable[[Optional[Entry]], Entry]) -> Entry:
        """Read the tip and insert the entry it produces, under one lock.

        The lock spans the tip read *and* the insert. Reading the head, building, then
        writing under separate locks is a read-modify-write, and that is how v1 forked a
        chain between two workers.
        """
        return retry_on_conflict(self._append_once(build), _is_conflict)

    def _append_once(self, build: Callable[[Optional[Entry]], Entry]) -> Callable[[], Entry]:
        def attempt() -> Entry:
            with self._lock:
                self._check_open()
                tip = self._entries[-1] if self._entries else None
                entry = build(tip)
                check_link(tip, entry)
                self._reject_duplicates(entry)
                stored = _detach(entry)
                self._entries.append(stored)
                self._by_entry_id[stored.entry_id] = stored
                self._previous_hashes[stored.previous_hash] = stored.sequence
                return entry

        return attempt

    def _reject_duplicates(self, entry: Entry) -> None:
        """Stand in for the ``UNIQUE`` constraints a SQL schema declares."""
        if entry.entry_id in self._by_entry_id:
            raise _ConstraintViolationError(f"entry_id of sequence {entry.sequence} already exists")
        if entry.previous_hash in self._previous_hashes:
            raise _ConstraintViolationError(
                f"previous_hash of sequence {entry.sequence} already links sequence "
                f"{self._previous_hashes[entry.previous_hash]}"
            )

    # -- reads -------------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> Optional[Entry]:
        with self._lock:
            self._check_open()
            found = self._by_entry_id.get(entry_id)
            return None if found is None else _detach(found)

    def get_by_sequence(self, sequence: int) -> Optional[Entry]:
        with self._lock:
            self._check_open()
            if 0 <= sequence < len(self._entries):
                return _detach(self._entries[sequence])
            return None

    def get_head(self) -> Optional[Entry]:
        with self._lock:
            self._check_open()
            return _detach(self._entries[-1]) if self._entries else None

    def get_oldest(self) -> Optional[Entry]:
        with self._lock:
            self._check_open()
            return _detach(self._entries[0]) if self._entries else None

    def count(self) -> int:
        with self._lock:
            self._check_open()
            return len(self._entries)

    def iter_entries(
        self,
        *,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Iterator[Entry]:
        validate_window(
            start_seq=start_seq,
            end_seq=end_seq,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
        with self._lock:
            self._check_open()
            selected = [
                entry
                for entry in self._entries
                if _matches(entry, start_seq, end_seq, start_time, end_time)
            ]
        # Offset applies *after* filtering, or a filtered page never advances.
        window = selected[offset:] if limit is None else selected[offset : offset + limit]
        return iter([_detach(entry) for entry in window])

    def size_bytes(self) -> int:
        """Bytes of the canonical storage form of every row held."""
        with self._lock:
            self._check_open()
            return sum(_row_bytes(entry) for entry in self._entries)

    # -- capabilities ------------------------------------------------------------------

    def capabilities(self) -> FrozenSet[str]:
        """Atomic and fork-proof, but never durable and never WORM.

        ``append_only_enforced`` is deliberately absent: nothing here outlives the
        process, and a store that claims WORM to ``assurance_level()`` while living in a
        heap is exactly the kind of unearned assurance this rebuild exists to remove.
        """
        return frozenset({CAP_ATOMIC_APPEND, CAP_UNIQUE_CHAIN_LINK})

    # -- checkpoints -------------------------------------------------------------------

    def save_checkpoint(self, cp: Checkpoint) -> None:
        # Round-tripped through the shared row form so the normalisation (UTC timestamps
        # above all) is identical to what a durable backend does.
        normalised = checkpoint_from_row(checkpoint_to_row(cp))
        with self._lock:
            self._check_open()
            self._checkpoints.append(normalised)

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        with self._lock:
            self._check_open()
            if not self._checkpoints:
                return None
            return max(self._checkpoints, key=lambda cp: cp.sequence)

    def iter_checkpoints(self) -> Iterator[Checkpoint]:
        with self._lock:
            self._check_open()
            return iter(sorted(self._checkpoints, key=lambda cp: cp.sequence))


def _matches(
    entry: Entry,
    start_seq: Optional[int],
    end_seq: Optional[int],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
) -> bool:
    if start_seq is not None and entry.sequence < start_seq:
        return False
    if end_seq is not None and entry.sequence > end_seq:
        return False
    if start_time is not None and entry.timestamp < start_time:
        return False
    return not (end_time is not None and entry.timestamp > end_time)


def _row_bytes(entry: Entry) -> int:
    row = entry_to_row(entry)
    return sum(len(str(value).encode("utf-8")) for value in row.values() if value is not None)


def _detach(entry: Entry) -> Entry:
    """Return an independent copy, so a caller mutating ``.data`` cannot reach the store.

    ``Entry`` is only shallow-frozen: ``entry.data["x"] = 1`` still works and would
    silently change what a stored entry hashes to. A durable backend gets this isolation
    for free by serializing; memory has to do it explicitly, or it is not interchangeable.
    """
    return Entry.from_dict(entry.model_dump())


class _ConstraintViolationError(Exception):
    """A uniqueness violation, mirroring what a SQL backend's driver raises."""


def _is_conflict(err: BaseException) -> bool:
    """Only a uniqueness clash is a lost race; everything else must surface at once."""
    return isinstance(err, _ConstraintViolationError)
