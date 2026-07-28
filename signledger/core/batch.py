"""Bulk appends - ``BatchProcessor``.

A hash chain is **serial at the tip**: entry *n* cannot exist until entry *n-1* has a
digest. So the only bulk path worth having is one that does less work per entry, not one
that spreads the same work over more threads.

v1 had it backwards. Its ``BatchProcessor`` measured 0.89x a plain ``append`` loop, and
its ``parallel_processing=True`` mode measured 0.67x because the threads only ever
contended for a single lock. ``TransactionalBatch.commit()`` raised ``AttributeError`` on
the first line of its body and ``_rollback()`` logged a warning and returned, so nothing
was atomic anywhere. Neither comes back: there is no thread pool here, no flag to enable
one, and no second class holding a rollback that does not roll back.

Where the speedup comes from
----------------------------
Each item is validated, deep-copied, hashed and signed **once**, before the first write,
and directly at its projected chain position. The write phase then only re-links and
inserts. A ``Ledger.append`` loop instead rebuilds every entry inside the write
transaction and pushes each one through the read cache, which canonicalizes both payloads
again just to measure them - so the loop pays SLCS/1 encoding roughly seven times per
entry where a batch pays it five (three on a backend that stores JSON rather than
detaching in memory).

What "atomic" means here, precisely
-----------------------------------
Every failure the *caller* can cause - a payload SLCS/1 refuses, a naive timestamp, an
``entry_id`` repeated inside the batch, a signer that throws - happens in the prepare
phase with nothing written. When ``atomic`` is on (the default), one such failure
abandons the whole batch: not a single entry reaches the store.

It is **not** a distributed transaction, and this module does not pretend otherwise.
:class:`~signledger.backends.base.StorageBackend` exposes ``append_atomic`` - one entry,
one transaction - and no bulk primitive, so a storage failure part-way through the write
phase cannot be undone by a protocol that has no DELETE. That case raises
:class:`PartialBatchError`, which names exactly how many entries landed. Reporting it is
the whole difference from v1, which logged a warning and returned as though nothing had
happened.
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import ValidationError as PydanticValidationError

from signledger.core.entry import CURRENT_FORMAT_VERSION, Entry
from signledger.core.exceptions import (
    ConfigurationError,
    SignLedgerError,
    StorageError,
    ValidationError,
)
from signledger.crypto.hashing import genesis_hash

__all__ = [
    "DEFAULT_MAX_BATCH_SIZE",
    "BatchItem",
    "BatchProcessor",
    "BatchResult",
    "DeadLetter",
    "PartialBatchError",
]

#: Upper bound on one batch. The whole chain is built in memory before anything is
#: written, so an unbounded batch is an unbounded allocation: a caller streaming a
#: million rows through one call would exhaust the process instead of being told to page.
DEFAULT_MAX_BATCH_SIZE = 10_000

#: Stand-in ``previous_hash`` while entry zero's genesis anchor is derived. Never
#: stored, and never hashed: it is replaced before the digest is computed. It only has to
#: be non-empty, because ``Entry`` refuses an empty one.
_UNLINKED = "0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PartialBatchError(StorageError):
    """The store failed part-way through a batch, after some entries were committed.

    This is the one failure a batch cannot take back. An append-only store has no DELETE
    to undo the rows that already landed, so the honest answer is to name the extent of
    the damage: :attr:`committed` holds exactly the entries that reached the chain.
    v1's equivalent logged a warning and returned normally, leaving a half-written ledger
    nobody was told about.
    """

    def __init__(self, message: str, committed: Sequence[Entry]) -> None:
        super().__init__(message)
        self.committed: Tuple[Entry, ...] = tuple(committed)


@dataclass(frozen=True)
class BatchItem:
    """One entry-to-be.

    A bare ``dict`` passed to :meth:`BatchProcessor.append_batch` is always the entry's
    ``data``; this is the only way to attach metadata, pin a timestamp or supply an
    ``entry_id``. Keeping the two forms unambiguous matters more than the convenience of
    guessing: ``{"data": ...}`` is a payload with a key called ``data``, nothing else.
    """

    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    entry_id: Optional[str] = None


@dataclass(frozen=True)
class DeadLetter:
    """One item that never became an entry, and why.

    ``index`` is the item's position in the batch as submitted, so a caller can line the
    failures back up against its own input.
    """

    index: int
    error: str
    message: str


@dataclass(frozen=True)
class BatchResult:
    """The outcome of one batch: what was committed, and what was refused."""

    entries: Tuple[Entry, ...] = ()
    dead_letters: Tuple[DeadLetter, ...] = ()
    submitted: int = 0
    duration_seconds: float = 0.0

    @property
    def committed(self) -> int:
        """How many entries reached the store."""
        return len(self.entries)

    @property
    def failed(self) -> int:
        """How many items were refused."""
        return len(self.dead_letters)

    @property
    def ok(self) -> bool:
        """True only when every submitted item became an entry."""
        return not self.dead_letters and len(self.entries) == self.submitted


class BatchProcessor:
    """Appends many entries to one :class:`~signledger.core.ledger.Ledger`.

    There is no ``parallel_processing`` option and there will not be one: the tip is a
    serial dependency, so extra threads can only contend for it.

    With ``atomic=True`` (the default) one unusable item abandons the whole submission and
    nothing is written. With ``atomic=False`` the usable items are committed and the rest
    come back in :attr:`BatchResult.dead_letters`.
    """

    def __init__(
        self,
        ledger: Any,
        *,
        atomic: bool = True,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        if max_batch_size < 1:
            raise ConfigurationError(f"max_batch_size must be at least 1, got {max_batch_size}")
        self._ledger = ledger
        self.atomic = atomic
        self.max_batch_size = max_batch_size
        # Coupling to the ledger's internals, in one place and on purpose. A batched
        # entry has to be indistinguishable from one ``Ledger.append`` produced - same
        # digest algorithm, same signing identity inside the preimage - so it uses the
        # ledger's own chain and signer rather than a second configuration that could
        # drift out of step with it.
        self.backend = ledger.backend
        self._chain = ledger._chain
        self._signer = ledger._signer if ledger._enable_signatures else None

    @property
    def ledger_id(self) -> str:
        """Read through to the ledger, never a snapshot.

        This was a copy taken in ``__init__``. Once ``Ledger`` learned to adopt the store's
        identity at cold start (:meth:`~signledger.core.ledger.Ledger._adopt_or_refuse`), a
        processor built before that adoption held the abandoned id - so one ledger answered
        two different identities depending on whether the caller went through ``append`` or
        through a batch, and only the batch failed.
        """
        return str(self._ledger.ledger_id)

    def append_batch(
        self, items: Iterable[Union[BatchItem, Dict[str, Any]]]
    ) -> BatchResult:
        """Validate the whole submission, then write the part of it that survived.

        A bare ``dict`` is the entry's ``data``; use :class:`BatchItem` to attach
        metadata, pin a timestamp or supply an ``entry_id``.

        Raises :class:`~signledger.core.exceptions.ValidationError` for an empty or
        oversized submission and :class:`PartialBatchError` if the store fails after some
        entries have already been committed.
        """
        started = time.perf_counter()
        submitted = list(items)
        if not submitted:
            # A bulk call that was handed nothing did nothing, and a result whose ``ok``
            # is True would be a success tick over an empty range - the same defect
            # contract section 5 rule 1 forbids in the verifier.
            raise ValidationError("append_batch requires at least one item")
        if len(submitted) > self.max_batch_size:
            raise ValidationError(
                f"batch holds {len(submitted)} items; the maximum is {self.max_batch_size}"
            )
        prepared, dead_letters = self._prepare(submitted)
        if dead_letters and self.atomic:
            return BatchResult(
                dead_letters=tuple(dead_letters),
                submitted=len(submitted),
                duration_seconds=time.perf_counter() - started,
            )
        committed = self._commit(prepared)
        return BatchResult(
            entries=tuple(committed),
            dead_letters=tuple(dead_letters),
            submitted=len(submitted),
            duration_seconds=time.perf_counter() - started,
        )

    # -- phase 1: prepare ---------------------------------------------------------------

    def _prepare(
        self, submitted: List[Union[BatchItem, Dict[str, Any]]]
    ) -> Tuple[List["_Prepared"], List[DeadLetter]]:
        """Validate, hash and sign the whole chain. Writes nothing.

        This is where the atomicity guarantee is bought: everything that can go wrong on
        the caller's side - a payload SLCS/1 refuses, a naive timestamp, a repeated
        ``entry_id``, a signer that throws - goes wrong here, before a single row exists.

        Each entry is built directly at its projected chain position, so a payload is
        canonicalized and deep-copied exactly once. That projection is a *guess* about
        where the tip will still be; :meth:`_commit` checks it against the tip the backend
        reads inside the transaction and re-seals only if another writer moved it.
        """
        moment = _utcnow()
        tip = self.backend.get_head()
        previous = None if tip is None else tip.hash
        sequence = 0 if tip is None else tip.sequence + 1

        prepared: List[_Prepared] = []
        dead_letters: List[DeadLetter] = []
        seen_ids: Dict[str, int] = {}
        for index, raw in enumerate(submitted):
            item = raw if isinstance(raw, BatchItem) else BatchItem(data=raw)
            try:
                entry = self._seal(item, moment, sequence, previous)
                _reject_repeat(entry.entry_id, index, seen_ids)
            except (SignLedgerError, PydanticValidationError) as err:
                dead_letters.append(
                    DeadLetter(index=index, error=type(err).__name__, message=str(err))
                )
                continue
            prepared.append(_Prepared(index=index, entry=entry, predecessor=previous))
            previous = entry.hash
            sequence += 1
        return prepared, dead_letters

    def _seal(
        self, item: BatchItem, moment: datetime, sequence: int, previous: Optional[str]
    ) -> Entry:
        """Build, validate, hash and sign one entry at a known chain position.

        ``previous is None`` means the store is empty, so entry 0's ``previous_hash`` comes
        from its **own** timestamp - the same rule ``Ledger.append`` uses. A reopened
        ledger recomputes that anchor from the oldest entry it can see, and any other
        choice makes a pristine chain report GENESIS_MISMATCH on restart.
        """
        signer = self._signer
        draft = Entry(
            ledger_id=self.ledger_id,
            sequence=sequence,
            entry_id=str(uuid.uuid4()) if item.entry_id is None else item.entry_id,
            timestamp=moment if item.timestamp is None else item.timestamp,
            data=item.data,
            metadata={} if item.metadata is None else item.metadata,
            # Linked below once ``Entry`` has validated the timestamp the genesis anchor
            # is derived from. Deriving it first would report a naive timestamp as a
            # CanonicalizationError from deep inside the encoder on an empty store and as
            # a ValidationError on a populated one - the same mistake, two different
            # answers, depending on something the caller cannot see.
            previous_hash=_UNLINKED if previous is None else previous,
            # Contract section 6: the signing identity is inside the preimage, so it is
            # fixed here, before the digest that covers it exists.
            key_id=None if signer is None else signer.key_id,
            signature_algorithm=None if signer is None else signer.algorithm,
        )
        if previous is None:
            draft = draft.model_copy(
                update={"previous_hash": self._genesis(draft.timestamp)}
            )
        return self._finish(draft)

    def _reseal(self, entry: Entry, sequence: int, previous: Optional[str]) -> Entry:
        """Re-link an already validated entry onto a tip that moved, and re-sign it.

        ``ledger_id`` is re-stamped along with the position. The tip moving is exactly the
        moment an adoption can have happened, and an entry prepared under the identity this
        process minted would otherwise be written into a chain that identity does not own -
        stamped forever with a name the store never agreed to.
        """
        draft = entry.model_copy(
            update={
                "ledger_id": self.ledger_id,
                "sequence": sequence,
                "previous_hash": (
                    self._genesis(entry.timestamp) if previous is None else previous
                ),
                "hash": "",
                "signature": None,
            }
        )
        return self._finish(draft)

    def _finish(self, draft: Entry) -> Entry:
        """Attach the digest, then the signature over it."""
        sealed = draft.model_copy(update={"hash": self._chain.calculate_hash(draft)})
        if self._signer is None:
            return sealed
        # Deferred for the same reason as in ``Ledger._seal``: importing
        # ``signledger.crypto.signatures`` at module scope pulls ``cryptography`` - an
        # optional extra - into every `import signledger`.
        from signledger.crypto.signatures import signing_payload

        signature = self._signer.sign(signing_payload(sealed.ledger_id, sealed.hash))
        return sealed.model_copy(update={"signature": signature})

    def _genesis(self, timestamp: datetime) -> str:
        return genesis_hash(
            self.ledger_id, timestamp, CURRENT_FORMAT_VERSION, algorithm=self._chain.algorithm
        )

    # -- phase 2: commit ----------------------------------------------------------------

    def _commit(self, prepared: List["_Prepared"]) -> List[Entry]:
        """Write the prepared chain, one backend transaction per entry.

        The build callback always re-links against the tip the backend read *inside* the
        transaction. Reading the head here and trusting it would be a read-modify-write,
        which is precisely how two v1 workers forked a chain permanently.
        """
        committed: List[Entry] = []
        for step in prepared:
            try:
                entry = self.backend.append_atomic(self._linker(step))
            except Exception as err:
                if not committed:
                    # Nothing landed, so nothing is partial. Wrapping it would send an
                    # operator looking for a half-written chain that does not exist.
                    raise
                raise PartialBatchError(
                    f"batch failed after committing {len(committed)} of {len(prepared)} "
                    f"entries: {type(err).__name__}",
                    committed,
                ) from err
            committed.append(entry)
            # The ledger now has a stake in this chain, so any later tip belonging to
            # someone else is a fork rather than a cold start. Without this, a processor
            # would keep adopting foreign identities forever and silently splice two
            # independent logs - strictly worse than the refusal it replaced.
            self._ledger._has_appended = True
            # Same bounded, off-thread queue ``Ledger.append`` uses. A subscriber must not
            # go blind because the caller switched to the bulk path, and it must not be
            # able to slow a batch down either.
            self._ledger._publish_event(entry)
        return committed

    def _linker(self, step: "_Prepared") -> Any:
        """The build callback for one prepared entry, run inside the transaction."""

        def build(tip: Optional[Entry]) -> Entry:
            if tip is None:
                actual, sequence = None, 0
            else:
                if tip.ledger_id != self.ledger_id:
                    # Delegated, not duplicated. This used to be a verbatim copy of the
                    # ledger's refusal, so when ``Ledger`` learned to adopt an identity it
                    # minted itself and had not yet written under, the batch path kept
                    # refusing - and the README's own bulk example stayed broken for every
                    # worker that lost the race for entry zero.
                    #
                    # Continuing another ledger's chain under this one's identity splices
                    # two logs together and every hash in the result verifies perfectly, so
                    # the decision has to live in exactly one place.
                    self._ledger._adopt_or_refuse(tip)
                actual, sequence = tip.hash, tip.sequence + 1
            if actual == step.predecessor and step.entry.ledger_id == self.ledger_id:
                # The identity check is not redundant with the position check. An adoption
                # a line above changes who this ledger is WITHOUT moving the tip, so the
                # prepared entry can still sit at the right position while carrying the
                # name this process invented and then gave up.
                return step.entry
            # Another writer moved the tip between preparation and this transaction, so
            # the projection is stale. Re-link against what the transaction actually read
            # - never against what we assumed before it started.
            return self._reseal(step.entry, sequence, actual)

        return build


@dataclass(frozen=True)
class _Prepared:
    """A validated, hashed and signed entry waiting to be confirmed into the chain.

    ``predecessor`` is the tip digest the projection assumed - ``None`` for a store the
    preparation phase found empty. The write phase compares it against the tip the
    transaction really read.
    """

    index: int
    entry: Entry
    predecessor: Optional[str]


def _reject_repeat(entry_id: str, index: int, seen: Dict[str, int]) -> None:
    """Refuse an ``entry_id`` this batch has already used.

    ``UNIQUE(entry_id)`` would catch it at insert time, but only after the earlier
    entries had been committed - and the backend's retry would rebuild the same duplicate
    five times before giving up. Catching it here keeps the failure inside the prepare
    phase, where a batch can still be abandoned whole.
    """
    if entry_id in seen:
        raise ValidationError(
            f"entry_id of item {index} is already used by item {seen[entry_id]} in this batch"
        )
    seen[entry_id] = index
