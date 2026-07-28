"""Storage backend contract (architecture contract, section 4).

Two things live here:

* :class:`StorageBackend` - the ``typing.Protocol`` every backend implements.
* :class:`BackendConformanceSuite` - a pytest mixin of behavioural tests. Subclass it,
  supply a ``backend`` fixture, and any backend that does not behave like the others
  fails at test time. Interchangeability that is not executed is interchangeability that
  is not true.

Plus the small shared pieces every backend needs to agree on: identifier validation,
payload serialization (``metadata`` is **never** NULL), microsecond timestamp encoding
and the bounded append retry.
"""

import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from signledger.anchor.checkpoint import Checkpoint
from signledger.core.entry import Entry
from signledger.core.exceptions import ConfigurationError, StorageError, ValidationError
from signledger.crypto.hashing import HashChain, genesis_hash

__all__ = [
    "APPEND_RETRY_BASE_DELAY",
    "CAP_APPEND_ONLY_ENFORCED",
    "CAP_ATOMIC_APPEND",
    "CAP_DURABLE_FSYNC",
    "CAP_UNIQUE_CHAIN_LINK",
    "CAP_WORM",
    "CAPABILITY_TOKENS",
    "IDENTIFIER_PATTERN",
    "MAX_APPEND_ATTEMPTS",
    "BackendConformanceSuite",
    "ConformanceTestMixin",
    "StorageBackend",
    "build_next_entry",
    "checkpoint_from_row",
    "checkpoint_to_row",
    "deserialize_payload",
    "entry_from_row",
    "entry_to_row",
    "micros_to_timestamp",
    "retry_on_conflict",
    "serialize_payload",
    "timestamp_to_micros",
    "validate_identifier",
]

T = TypeVar("T")

# --------------------------------------------------------------------------------------
# capability tokens (contract section 4)
# --------------------------------------------------------------------------------------

CAP_ATOMIC_APPEND = "atomic_append"
CAP_UNIQUE_CHAIN_LINK = "unique_chain_link"
CAP_APPEND_ONLY_ENFORCED = "append_only_enforced"
CAP_WORM = "worm"
CAP_DURABLE_FSYNC = "durable_fsync"

#: Every token a backend may report. ``capabilities()`` returning anything else is a bug -
#: ``assurance_level()`` reads these, so an invented token is a silent security claim.
CAPABILITY_TOKENS: FrozenSet[str] = frozenset(
    {
        CAP_ATOMIC_APPEND,
        CAP_UNIQUE_CHAIN_LINK,
        CAP_APPEND_ONLY_ENFORCED,
        CAP_WORM,
        CAP_DURABLE_FSYNC,
    }
)

# --------------------------------------------------------------------------------------
# append retry
# --------------------------------------------------------------------------------------

#: Bounded, because an unbounded retry under contention is a livelock, not a repair.
MAX_APPEND_ATTEMPTS = 5

#: First backoff step, doubled each attempt.
APPEND_RETRY_BASE_DELAY = 0.005

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

#: Identifiers are interpolated into DDL, so they are matched, never escaped.
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_MAX_IDENTIFIER_LENGTH = 63


# --------------------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class StorageBackend(Protocol):
    """The storage surface a :class:`~signledger.core.ledger.Ledger` writes through."""

    def append_atomic(self, build: Callable[[Optional[Entry]], Entry]) -> Entry: ...

    def get_entry(self, entry_id: str) -> Optional[Entry]: ...

    def get_by_sequence(self, sequence: int) -> Optional[Entry]: ...

    def iter_entries(
        self,
        *,
        start_seq: Optional[int] = None,
        end_seq: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Iterator[Entry]: ...

    def get_head(self) -> Optional[Entry]: ...

    def get_oldest(self) -> Optional[Entry]: ...

    def count(self) -> int: ...

    def size_bytes(self) -> int: ...

    def capabilities(self) -> FrozenSet[str]: ...

    def save_checkpoint(self, cp: Checkpoint) -> None: ...

    def get_latest_checkpoint(self) -> Optional[Checkpoint]: ...

    def iter_checkpoints(self) -> Iterator[Checkpoint]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


def validate_identifier(name: str, *, kind: str = "identifier") -> str:
    """Return ``name`` if it is a safe SQL identifier, else raise.

    Table names reach DDL by interpolation - there is no parameter binding for an
    identifier - so the only safe policy is an allowlist. Anything carrying a quote, a
    semicolon, a space or a comment marker is refused outright rather than escaped.
    """
    if not isinstance(name, str) or not IDENTIFIER_PATTERN.match(name):
        raise ConfigurationError(
            f"{kind} must match {IDENTIFIER_PATTERN.pattern}; "
            f"got a {type(name).__name__} of length {len(name) if isinstance(name, str) else 0}"
        )
    if len(name) > _MAX_IDENTIFIER_LENGTH:
        raise ConfigurationError(
            f"{kind} is {len(name)} characters; the maximum is {_MAX_IDENTIFIER_LENGTH}"
        )
    return name


def serialize_payload(payload: Optional[Dict[str, Any]]) -> str:
    """Encode ``data``/``metadata`` for storage - ``None`` becomes ``{}``, never NULL.

    This single line is what broke every v1 SQLite and Postgres ledger: ``{}`` is falsy,
    so ``metadata or None`` stored NULL, and reading it back handed ``None`` to
    ``Entry(metadata=...)``, which refuses it. A ledger that cannot read its own rows is
    not a ledger.
    """
    import json

    return json.dumps({} if payload is None else payload, ensure_ascii=False, sort_keys=True)


def deserialize_payload(raw: Optional[str]) -> Dict[str, Any]:
    """Decode a stored payload. A NULL column from a legacy row still reads as ``{}``."""
    import json

    if raw is None or raw == "":
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise StorageError(f"stored payload decoded to {type(loaded).__name__}, expected an object")
    return loaded


def timestamp_to_micros(value: datetime) -> int:
    """Encode a tz-aware datetime as integer microseconds since the Unix epoch.

    Integer microseconds are the only representation that sorts correctly, compares
    exactly and survives a round trip without a timezone renegotiation. A naive datetime
    is refused: guessing its zone is how v1 re-hashed every row differently under a
    non-UTC session.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValidationError("timestamp must be timezone-aware")
    delta = value.astimezone(timezone.utc) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def micros_to_timestamp(micros: int) -> datetime:
    """Decode integer microseconds back into a tz-aware UTC datetime."""
    return _EPOCH + timedelta(microseconds=micros)


def entry_to_row(entry: Entry) -> Dict[str, Any]:
    """Flatten an entry into the column mapping every SQL backend shares."""
    return {
        "sequence": entry.sequence,
        "entry_id": entry.entry_id,
        "ledger_id": entry.ledger_id,
        "format_version": entry.format_version,
        "timestamp": entry.timestamp.isoformat(),
        "timestamp_us": timestamp_to_micros(entry.timestamp),
        "data": serialize_payload(entry.data),
        "metadata": serialize_payload(entry.metadata),
        "previous_hash": entry.previous_hash,
        "hash": entry.hash,
        "key_id": entry.key_id,
        "signature_algorithm": entry.signature_algorithm,
        "signature": entry.signature,
    }


def entry_from_row(row: Dict[str, Any]) -> Entry:
    """Rebuild an entry from a stored row, re-validating it on the way in."""
    return Entry(
        format_version=row["format_version"],
        ledger_id=row["ledger_id"],
        sequence=row["sequence"],
        entry_id=row["entry_id"],
        timestamp=micros_to_timestamp(row["timestamp_us"]),
        data=deserialize_payload(row["data"]),
        metadata=deserialize_payload(row["metadata"]),
        previous_hash=row["previous_hash"],
        key_id=row["key_id"],
        signature_algorithm=row["signature_algorithm"],
        hash=row["hash"],
        signature=row["signature"],
    )


def checkpoint_to_row(cp: Checkpoint) -> Dict[str, Any]:
    """Flatten a checkpoint into its column mapping."""
    return {
        "ledger_id": cp.ledger_id,
        "sequence": cp.sequence,
        "head_hash": cp.head_hash,
        "entry_count": cp.entry_count,
        "merkle_root": cp.merkle_root,
        "timestamp": cp.timestamp.isoformat(),
        "timestamp_us": timestamp_to_micros(cp.timestamp),
        "signature": cp.signature,
        "key_id": cp.key_id,
        "format_version": cp.format_version,
    }


def checkpoint_from_row(row: Dict[str, Any]) -> Checkpoint:
    """Rebuild a checkpoint from a stored row."""
    return Checkpoint(
        ledger_id=row["ledger_id"],
        sequence=row["sequence"],
        head_hash=row["head_hash"],
        entry_count=row["entry_count"],
        merkle_root=row["merkle_root"],
        timestamp=micros_to_timestamp(row["timestamp_us"]),
        signature=row["signature"],
        key_id=row["key_id"],
        format_version=row["format_version"],
    )


def validate_window(
    *,
    start_seq: Optional[int],
    end_seq: Optional[int],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    limit: Optional[int],
    offset: int,
) -> None:
    """Reject nonsensical pagination before it reaches SQL.

    ``limit`` and ``offset`` are checked with ``is not None``, never for truthiness:
    ``limit=0`` means *zero rows*, and v1's ``if limit:`` turned that into *every row*.
    """
    if limit is not None and limit < 0:
        raise ValidationError(f"limit must be zero or greater, got {limit}")
    if offset < 0:
        raise ValidationError(f"offset must be zero or greater, got {offset}")
    for name, value in (("start_seq", start_seq), ("end_seq", end_seq)):
        if value is not None and value < 0:
            raise ValidationError(f"{name} must be zero or greater, got {value}")
    for name, moment in (("start_time", start_time), ("end_time", end_time)):
        if moment is not None and (
            moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None
        ):
            raise ValidationError(f"{name} must be timezone-aware")


def retry_on_conflict(
    operation: Callable[[], T],
    is_conflict: Callable[[BaseException], bool],
    *,
    attempts: int = MAX_APPEND_ATTEMPTS,
    base_delay: float = APPEND_RETRY_BASE_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``operation``, retrying while ``is_conflict`` recognises the failure.

    Bounded and exponential. A uniqueness violation under contention means another writer
    won the race, so the fix is to re-read the tip and rebuild - which is exactly what
    running ``operation`` again does. Anything ``is_conflict`` does not recognise
    propagates untouched on the first attempt: a broken payload must not be retried five
    times before surfacing.

    ``sleep`` is injectable so a test can observe the backoff schedule without waiting.
    """
    if attempts < 1:
        raise ConfigurationError(f"attempts must be at least 1, got {attempts}")
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as err:
            if not is_conflict(err):
                raise
            attempt += 1
            if attempt >= attempts:
                raise StorageError(
                    f"append conflicted on {attempt} consecutive attempts; "
                    f"the chain tip is contended"
                ) from err
            sleep(base_delay * (2 ** (attempt - 1)))


def build_next_entry(
    tip: Optional[Entry],
    *,
    ledger_id: str,
    created_at: datetime,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
    entry_id: Optional[str] = None,
    chain: Optional[HashChain] = None,
    **fields: Any,
) -> Entry:
    """Build the sealed entry that must follow ``tip``.

    ``tip`` is what :meth:`StorageBackend.append_atomic` read inside the transaction:
    ``None`` for an empty store, in which case the result is the genesis-linked entry
    zero. Exposed because every backend's conformance run needs to produce a valid chain.
    """
    hasher = HashChain() if chain is None else chain
    if tip is None:
        sequence = 0
        previous_hash = genesis_hash(ledger_id, created_at, algorithm=hasher.algorithm)
    else:
        sequence = tip.sequence + 1
        previous_hash = tip.hash
    draft = Entry(
        ledger_id=ledger_id,
        sequence=sequence,
        entry_id=str(uuid.uuid4()) if entry_id is None else entry_id,
        timestamp=created_at + timedelta(seconds=sequence) if timestamp is None else timestamp,
        data=data,
        metadata={} if metadata is None else metadata,
        previous_hash=previous_hash,
        **fields,
    )
    return draft.model_copy(update={"hash": hasher.calculate_hash(draft)})


def check_link(tip: Optional[Entry], entry: Entry) -> None:
    """Refuse an entry that does not continue the chain. Appends **fail closed**.

    A genesis link is accepted only when the tip really is absent, read inside the same
    transaction. v1 caught the head-read failure, logged a warning and wrote an unlinked
    entry, which forks a ledger permanently with no repair path.
    """
    if tip is None:
        if entry.sequence != 0:
            raise StorageError(
                f"cannot append sequence {entry.sequence} to an empty store; expected 0"
            )
        return
    if entry.sequence != tip.sequence + 1:
        raise StorageError(
            f"sequence must increment by exactly 1: tip is {tip.sequence}, "
            f"entry is {entry.sequence}"
        )
    if entry.previous_hash != tip.hash:
        raise StorageError(
            f"previous_hash of sequence {entry.sequence} does not match the hash of "
            f"sequence {tip.sequence}"
        )


# --------------------------------------------------------------------------------------
# conformance suite
# --------------------------------------------------------------------------------------


class _BuildFailedError(Exception):
    """Raised by a conformance build callback to prove errors are not retried."""


def _always_fails(tip: Optional[Entry]) -> Entry:
    raise _BuildFailedError("the caller's callback blew up")


def _raises(exception: type) -> Any:
    """``pytest.raises``, imported lazily.

    The conformance suite ships inside the installed package so that a third-party
    backend can run it, but pytest must never become a runtime dependency of a library
    whose job is writing audit logs.
    """
    import pytest

    return pytest.raises(exception)


class BackendConformanceSuite:
    """Behavioural tests every :class:`StorageBackend` must pass.

    Subclass it in a test module and provide a ``backend`` fixture::

        class TestMyBackend(BackendConformanceSuite):
            @pytest.fixture
            def backend(self):
                store = MyBackend(...)
                yield store
                store.close()

    The suite builds its own valid chains, so nothing beyond that fixture is required.
    """

    ledger_id = "conformance-0f1c9a2e-4b7d-4c1e-9b4a-5f2d8c3e7a10"
    created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    base_timestamp = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    # -- helpers ----------------------------------------------------------------------

    def append(self, backend: Any, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Entry:
        """Append one correctly linked entry and return it."""
        payload = {"event": "conformance"} if data is None else data
        appended: Entry = backend.append_atomic(
            lambda tip: build_next_entry(
                tip,
                ledger_id=self.ledger_id,
                created_at=self.created_at,
                data=payload,
                **kwargs,
            )
        )
        return appended

    def append_many(self, backend: Any, count: int, **kwargs: Any) -> List[Entry]:
        """Append ``count`` entries whose timestamps advance one second at a time."""
        return [
            self.append(
                backend,
                {"event": "conformance", "index": index},
                timestamp=self.base_timestamp + timedelta(seconds=index),
                **kwargs,
            )
            for index in range(count)
        ]

    # -- lifecycle ---------------------------------------------------------------------

    def test_a_fresh_backend_is_empty(self, backend: Any) -> None:
        assert backend.count() == 0
        assert backend.get_head() is None
        assert backend.get_oldest() is None
        assert list(backend.iter_entries()) == []

    def test_close_is_idempotent(self, backend: Any) -> None:
        backend.close()
        backend.close()

    # -- append ------------------------------------------------------------------------

    def test_append_atomic_returns_the_entry_it_stored(self, backend: Any) -> None:
        entry = self.append(backend, {"event": "created"})

        assert backend.count() == 1
        assert backend.get_head() == entry

    def test_append_atomic_offers_none_as_the_tip_of_an_empty_store(self, backend: Any) -> None:
        seen: List[Optional[Entry]] = []

        def build(tip: Optional[Entry]) -> Entry:
            seen.append(tip)
            return build_next_entry(
                tip, ledger_id=self.ledger_id, created_at=self.created_at, data={"n": 1}
            )

        backend.append_atomic(build)

        assert seen == [None]

    def test_append_atomic_offers_the_head_as_the_tip(self, backend: Any) -> None:
        first = self.append(backend)
        seen: List[Optional[Entry]] = []

        def build(tip: Optional[Entry]) -> Entry:
            seen.append(tip)
            return build_next_entry(
                tip, ledger_id=self.ledger_id, created_at=self.created_at, data={"n": 2}
            )

        backend.append_atomic(build)

        assert seen == [first]

    # -- reads -------------------------------------------------------------------------

    def test_get_entry_finds_an_entry_by_its_id(self, backend: Any) -> None:
        entries = self.append_many(backend, 3)

        assert backend.get_entry(entries[1].entry_id) == entries[1]

    def test_get_entry_returns_none_for_an_unknown_id(self, backend: Any) -> None:
        self.append_many(backend, 3)

        assert backend.get_entry("no-such-entry") is None

    def test_get_by_sequence_finds_an_entry_by_its_position(self, backend: Any) -> None:
        entries = self.append_many(backend, 3)

        assert backend.get_by_sequence(2) == entries[2]

    def test_get_by_sequence_returns_none_past_the_head(self, backend: Any) -> None:
        self.append_many(backend, 3)

        assert backend.get_by_sequence(3) is None

    def test_get_oldest_returns_sequence_zero(self, backend: Any) -> None:
        entries = self.append_many(backend, 4)

        assert backend.get_oldest() == entries[0]

    def test_get_head_returns_the_highest_sequence(self, backend: Any) -> None:
        entries = self.append_many(backend, 4)

        assert backend.get_head() == entries[-1]

    def test_iter_entries_yields_every_entry_in_sequence_order(self, backend: Any) -> None:
        entries = self.append_many(backend, 6)

        assert [e.sequence for e in backend.iter_entries()] == [0, 1, 2, 3, 4, 5]
        assert list(backend.iter_entries()) == entries

    def test_iter_entries_returns_an_iterator(self, backend: Any) -> None:
        self.append_many(backend, 2)
        result = backend.iter_entries()

        assert iter(result) is result

    # -- ordering ----------------------------------------------------------------------

    def test_iteration_orders_by_sequence_even_when_timestamps_are_identical(
        self, backend: Any
    ) -> None:
        """The ordering authority is ``sequence``. v1 ordered by wall clock and
        returned entries sharing a timestamp in reverse."""
        same = datetime(2024, 3, 3, 3, 3, 3, tzinfo=timezone.utc)
        entries = [self.append(backend, {"index": i}, timestamp=same) for i in range(8)]

        assert list(backend.iter_entries()) == entries

    def test_iteration_order_is_stable_across_repeated_calls(self, backend: Any) -> None:
        same = datetime(2024, 3, 3, 3, 3, 3, tzinfo=timezone.utc)
        for index in range(6):
            self.append(backend, {"index": index}, timestamp=same)

        first_pass = [e.entry_id for e in backend.iter_entries()]
        second_pass = [e.entry_id for e in backend.iter_entries()]

        assert first_pass == second_pass

    def test_iteration_ignores_a_timestamp_that_steps_backwards(self, backend: Any) -> None:
        """An NTP step-back must not reorder a pristine ledger."""
        self.append(backend, {"index": 0}, timestamp=self.base_timestamp)
        self.append(backend, {"index": 1}, timestamp=self.base_timestamp - timedelta(hours=3))
        self.append(backend, {"index": 2}, timestamp=self.base_timestamp + timedelta(hours=1))

        assert [e.data["index"] for e in backend.iter_entries()] == [0, 1, 2]

    # -- metadata is never NULL --------------------------------------------------------

    def test_an_entry_appended_without_metadata_reads_back_with_an_empty_dict(
        self, backend: Any
    ) -> None:
        """The single line that broke every v1 SQLite and Postgres ledger.

        ``{}`` is falsy, so ``metadata or None`` stored NULL, and ``Entry(metadata=None)``
        then refused to validate on read.
        """
        appended = self.append(backend, {"event": "no-metadata"})
        loaded = backend.get_by_sequence(0)

        assert loaded.metadata == {}
        assert loaded == appended

    def test_an_entry_with_empty_metadata_still_re_hashes_to_the_stored_digest(
        self, backend: Any
    ) -> None:
        appended = self.append(backend, {"event": "no-metadata"}, metadata={})
        loaded = backend.get_by_sequence(0)

        assert HashChain().verify_hash(loaded)
        assert loaded.hash == appended.hash

    def test_metadata_survives_the_round_trip_unchanged(self, backend: Any) -> None:
        metadata = {"actor": "svc", "tags": ["a", "b"], "retries": 0, "ok": False, "x": None}
        self.append(backend, {"event": "meta"}, metadata=metadata)

        assert backend.get_by_sequence(0).metadata == metadata

    # -- filtering ---------------------------------------------------------------------

    def test_sequence_range_is_inclusive_at_both_ends(self, backend: Any) -> None:
        self.append_many(backend, 8)

        found = backend.iter_entries(start_seq=2, end_seq=5)

        assert [e.sequence for e in found] == [2, 3, 4, 5]

    def test_start_seq_alone_runs_to_the_head(self, backend: Any) -> None:
        self.append_many(backend, 5)

        assert [e.sequence for e in backend.iter_entries(start_seq=3)] == [3, 4]

    def test_end_seq_alone_runs_from_the_oldest(self, backend: Any) -> None:
        self.append_many(backend, 5)

        assert [e.sequence for e in backend.iter_entries(end_seq=1)] == [0, 1]

    def test_time_range_is_inclusive_at_both_ends(self, backend: Any) -> None:
        self.append_many(backend, 8)

        found = backend.iter_entries(
            start_time=self.base_timestamp + timedelta(seconds=2),
            end_time=self.base_timestamp + timedelta(seconds=4),
        )

        assert [e.sequence for e in found] == [2, 3, 4]

    def test_start_time_alone_runs_to_the_head(self, backend: Any) -> None:
        self.append_many(backend, 5)

        found = backend.iter_entries(start_time=self.base_timestamp + timedelta(seconds=3))

        assert [e.sequence for e in found] == [3, 4]

    def test_end_time_alone_runs_from_the_oldest(self, backend: Any) -> None:
        self.append_many(backend, 5)

        found = backend.iter_entries(end_time=self.base_timestamp + timedelta(seconds=1))

        assert [e.sequence for e in found] == [0, 1]

    def test_a_time_window_matching_nothing_yields_nothing(self, backend: Any) -> None:
        self.append_many(backend, 5)

        found = backend.iter_entries(start_time=self.base_timestamp + timedelta(days=400))

        assert list(found) == []

    # -- limit and offset --------------------------------------------------------------

    def test_limit_zero_yields_no_entries(self, backend: Any) -> None:
        """``limit`` is tested with ``is not None``. v1's ``if limit:`` turned a request
        for zero rows into a request for every row."""
        self.append_many(backend, 5)

        assert list(backend.iter_entries(limit=0)) == []

    def test_limit_caps_the_number_of_entries(self, backend: Any) -> None:
        self.append_many(backend, 5)

        assert [e.sequence for e in backend.iter_entries(limit=2)] == [0, 1]

    def test_a_limit_larger_than_the_ledger_yields_everything(self, backend: Any) -> None:
        self.append_many(backend, 3)

        assert [e.sequence for e in backend.iter_entries(limit=99)] == [0, 1, 2]

    def test_offset_without_limit_skips_and_yields_the_remainder(self, backend: Any) -> None:
        """SQLite has no bare ``OFFSET``; v1 emitted one and raised OperationalError."""
        self.append_many(backend, 6)

        assert [e.sequence for e in backend.iter_entries(offset=4)] == [4, 5]

    def test_offset_zero_is_not_confused_with_no_offset(self, backend: Any) -> None:
        self.append_many(backend, 3)

        assert [e.sequence for e in backend.iter_entries(offset=0)] == [0, 1, 2]

    def test_offset_past_the_end_yields_nothing(self, backend: Any) -> None:
        self.append_many(backend, 3)

        assert list(backend.iter_entries(offset=10)) == []

    def test_limit_and_offset_page_through_every_entry(self, backend: Any) -> None:
        self.append_many(backend, 7)

        pages = [
            [e.sequence for e in backend.iter_entries(limit=3, offset=start)]
            for start in (0, 3, 6, 9)
        ]

        assert pages == [[0, 1, 2], [3, 4, 5], [6], []]

    def test_offset_applies_after_the_time_filter_so_pages_advance(self, backend: Any) -> None:
        """v1 consumed the offset counter on rows the filter had already discarded, so
        the same page came back forever."""
        self.append_many(backend, 10)
        window = {
            "start_time": self.base_timestamp + timedelta(seconds=4),
            "end_time": self.base_timestamp + timedelta(seconds=9),
        }

        pages = [
            [e.sequence for e in backend.iter_entries(limit=2, offset=start, **window)]
            for start in (0, 2, 4)
        ]

        assert pages == [[4, 5], [6, 7], [8, 9]]

    def test_offset_applies_after_the_sequence_filter(self, backend: Any) -> None:
        self.append_many(backend, 10)

        found = backend.iter_entries(start_seq=6, offset=1, limit=2)

        assert [e.sequence for e in found] == [7, 8]

    # -- pagination validation ---------------------------------------------------------

    def test_a_negative_limit_is_rejected(self, backend: Any) -> None:
        with _raises(ValidationError):
            backend.iter_entries(limit=-1)

    def test_a_negative_offset_is_rejected(self, backend: Any) -> None:
        with _raises(ValidationError):
            backend.iter_entries(offset=-1)

    def test_a_negative_start_seq_is_rejected(self, backend: Any) -> None:
        with _raises(ValidationError):
            backend.iter_entries(start_seq=-1)

    def test_a_negative_end_seq_is_rejected(self, backend: Any) -> None:
        with _raises(ValidationError):
            backend.iter_entries(end_seq=-3)

    def test_a_naive_start_time_is_rejected(self, backend: Any) -> None:
        with _raises(ValidationError):
            backend.iter_entries(start_time=datetime(2024, 1, 1))

    def test_a_naive_end_time_is_rejected(self, backend: Any) -> None:
        with _raises(ValidationError):
            backend.iter_entries(end_time=datetime(2024, 1, 1))

    # -- fail closed -------------------------------------------------------------------

    def test_append_refuses_an_entry_that_does_not_follow_the_tip(self, backend: Any) -> None:
        """A genesis link is only legal when the store really is empty."""
        self.append(backend)

        with _raises(StorageError):
            backend.append_atomic(
                lambda tip: build_next_entry(
                    None, ledger_id=self.ledger_id, created_at=self.created_at, data={"n": 1}
                )
            )

        assert backend.count() == 1

    def test_append_refuses_an_entry_whose_previous_hash_is_wrong(self, backend: Any) -> None:
        self.append(backend)

        def build(tip: Optional[Entry]) -> Entry:
            good = build_next_entry(
                tip, ledger_id=self.ledger_id, created_at=self.created_at, data={"n": 2}
            )
            draft = good.model_copy(update={"previous_hash": "0" * 64})
            return draft.model_copy(update={"hash": HashChain().calculate_hash(draft)})

        with _raises(StorageError):
            backend.append_atomic(build)

        assert backend.count() == 1

    def test_append_refuses_a_non_zero_sequence_on_an_empty_store(self, backend: Any) -> None:
        first = build_next_entry(
            None, ledger_id=self.ledger_id, created_at=self.created_at, data={"n": 1}
        )
        stranded = build_next_entry(
            first, ledger_id=self.ledger_id, created_at=self.created_at, data={"n": 2}
        )

        with _raises(StorageError):
            backend.append_atomic(lambda tip: stranded)

        assert backend.count() == 0

    def test_append_propagates_a_build_error_without_retrying(self, backend: Any) -> None:
        calls: List[Optional[Entry]] = []

        def build(tip: Optional[Entry]) -> Entry:
            calls.append(tip)
            raise _BuildFailedError("the caller's callback blew up")

        with _raises(_BuildFailedError):
            backend.append_atomic(build)

        assert len(calls) == 1
        assert backend.count() == 0

    def test_a_failed_append_leaves_the_head_untouched(self, backend: Any) -> None:
        entries = self.append_many(backend, 3)

        with _raises(_BuildFailedError):
            backend.append_atomic(_always_fails)

        assert backend.get_head() == entries[-1]
        assert backend.count() == 3

    # -- retry -------------------------------------------------------------------------

    def test_append_retries_after_a_uniqueness_conflict_and_then_succeeds(
        self, backend: Any
    ) -> None:
        """A duplicate ``entry_id`` is what a lost race looks like at the schema level.
        Re-reading the tip and rebuilding is the repair, so the retry must rebuild."""
        first = self.append(backend)
        calls: List[Optional[Entry]] = []

        def build(tip: Optional[Entry]) -> Entry:
            calls.append(tip)
            return build_next_entry(
                tip,
                ledger_id=self.ledger_id,
                created_at=self.created_at,
                data={"n": 2},
                entry_id=first.entry_id if len(calls) == 1 else None,
            )

        appended = backend.append_atomic(build)

        assert len(calls) == 2
        assert backend.count() == 2
        assert backend.get_head() == appended

    def test_append_gives_up_after_a_bounded_number_of_conflicts(self, backend: Any) -> None:
        first = self.append(backend)
        calls: List[Optional[Entry]] = []

        def build(tip: Optional[Entry]) -> Entry:
            calls.append(tip)
            return build_next_entry(
                tip,
                ledger_id=self.ledger_id,
                created_at=self.created_at,
                data={"n": 2},
                entry_id=first.entry_id,
            )

        with _raises(StorageError):
            backend.append_atomic(build)

        assert len(calls) == MAX_APPEND_ATTEMPTS
        assert backend.count() == 1

    def test_a_duplicate_previous_hash_is_refused(self, backend: Any) -> None:
        """``UNIQUE(previous_hash)`` is what stops two entries claiming the same parent -
        a fork - even if every other column differs."""
        entries = self.append_many(backend, 2)

        def build(tip: Optional[Entry]) -> Entry:
            draft = Entry(
                ledger_id=self.ledger_id,
                sequence=2,
                entry_id="fork-entry",
                timestamp=self.base_timestamp,
                data={"n": "fork"},
                previous_hash=entries[0].hash,
            )
            return draft.model_copy(update={"hash": HashChain().calculate_hash(draft)})

        with _raises(StorageError):
            backend.append_atomic(build)

        assert backend.count() == 2

    # -- isolation ---------------------------------------------------------------------

    def test_the_store_is_detached_from_the_entry_the_caller_holds(self, backend: Any) -> None:
        """``Entry`` is only shallow-frozen, so ``entry.data[...] = x`` still works. It
        must not reach into the store and change what a row hashes to."""
        appended = self.append(backend, {"event": "x", "nested": {"n": 1}})

        appended.data["nested"]["n"] = 999

        assert backend.get_by_sequence(0).data == {"event": "x", "nested": {"n": 1}}

    def test_the_store_is_detached_from_an_entry_it_returned(self, backend: Any) -> None:
        self.append(backend, {"event": "x", "nested": {"n": 1}})

        loaded = backend.get_by_sequence(0)
        loaded.data["nested"]["n"] = 999

        assert backend.get_by_sequence(0).data["nested"]["n"] == 1

    # -- capabilities ------------------------------------------------------------------

    def test_capabilities_reports_only_known_tokens(self, backend: Any) -> None:
        """An invented token is a silent security claim: ``assurance_level()`` reads
        these to decide whether to say WORM."""
        tokens = backend.capabilities()

        assert isinstance(tokens, frozenset)
        assert tokens <= CAPABILITY_TOKENS

    def test_every_backend_offers_atomic_append_and_a_unique_chain_link(
        self, backend: Any
    ) -> None:
        tokens = backend.capabilities()

        assert CAP_ATOMIC_APPEND in tokens
        assert CAP_UNIQUE_CHAIN_LINK in tokens

    def test_size_bytes_is_non_negative_when_empty(self, backend: Any) -> None:
        assert backend.size_bytes() >= 0

    def test_size_bytes_does_not_shrink_as_entries_are_added(self, backend: Any) -> None:
        empty = backend.size_bytes()
        self.append_many(backend, 40)

        assert backend.size_bytes() >= empty

    # -- checkpoints -------------------------------------------------------------------

    def checkpoint(self, sequence: int = 0, **kwargs: Any) -> Checkpoint:
        fields: Dict[str, Any] = {
            "ledger_id": self.ledger_id,
            "sequence": sequence,
            "head_hash": f"{sequence:064x}",
            "entry_count": sequence + 1,
            "merkle_root": f"{sequence:064X}",
            "timestamp": self.base_timestamp + timedelta(minutes=sequence),
        }
        fields.update(kwargs)
        return Checkpoint(**fields)

    def test_get_latest_checkpoint_is_none_before_any_is_saved(self, backend: Any) -> None:
        assert backend.get_latest_checkpoint() is None
        assert list(backend.iter_checkpoints()) == []

    def test_a_saved_checkpoint_reads_back_field_for_field(self, backend: Any) -> None:
        saved = self.checkpoint(3, signature="c2ln", key_id="kid-1", format_version=2)
        backend.save_checkpoint(saved)

        assert backend.get_latest_checkpoint() == saved

    def test_an_unsigned_checkpoint_reads_back_with_null_signature_fields(
        self, backend: Any
    ) -> None:
        saved = self.checkpoint(1)
        backend.save_checkpoint(saved)
        loaded = backend.get_latest_checkpoint()

        assert loaded.signature is None
        assert loaded.key_id is None

    def test_a_checkpoint_timestamp_normalises_to_utc(self, backend: Any) -> None:
        """Two backends that disagree about the stored zone are not interchangeable."""
        zone = timezone(timedelta(hours=5, minutes=30))
        moment = datetime(2024, 5, 5, 6, 7, 8, 123456, tzinfo=zone)
        backend.save_checkpoint(self.checkpoint(0, timestamp=moment))
        loaded = backend.get_latest_checkpoint()

        assert loaded.timestamp == moment
        assert loaded.timestamp.utcoffset() == timedelta(0)

    def test_get_latest_checkpoint_returns_the_highest_sequence(self, backend: Any) -> None:
        for sequence in (2, 9, 5):
            backend.save_checkpoint(self.checkpoint(sequence))

        assert backend.get_latest_checkpoint().sequence == 9

    def test_iter_checkpoints_yields_them_in_sequence_order(self, backend: Any) -> None:
        for sequence in (7, 1, 4):
            backend.save_checkpoint(self.checkpoint(sequence))

        assert [cp.sequence for cp in backend.iter_checkpoints()] == [1, 4, 7]

    def test_checkpoints_are_independent_of_entries(self, backend: Any) -> None:
        backend.save_checkpoint(self.checkpoint(0))

        assert backend.count() == 0

    # -- closed backend ----------------------------------------------------------------

    def test_reading_after_close_raises_storage_error(self, backend: Any) -> None:
        self.append(backend)
        backend.close()

        with _raises(StorageError):
            backend.count()

    def test_appending_after_close_raises_storage_error(self, backend: Any) -> None:
        backend.close()

        with _raises(StorageError):
            self.append(backend)

    def test_iterating_after_close_raises_storage_error(self, backend: Any) -> None:
        self.append(backend)
        backend.close()

        with _raises(StorageError):
            list(backend.iter_entries())


#: The contract names this ``ConformanceTestMixin``; the file list names it
#: ``BackendConformanceSuite``. They are the same class.
ConformanceTestMixin = BackendConformanceSuite
