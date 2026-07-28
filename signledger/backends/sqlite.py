"""SQLite storage backend.

Three things here are not optional and each of them was a real v1 defect:

* ``BEGIN IMMEDIATE`` wraps the tip read *and* the insert. A read-modify-write under a
  process-local lock forks the chain the moment a second process exists.
* ``ORDER BY sequence``. Never ``ORDER BY timestamp`` - equal timestamps came back
  reversed and an NTP step-back reordered a pristine ledger.
* ``metadata`` is stored as ``json.dumps(... or {})``. ``{}`` is falsy, so v1 wrote NULL
  and could no longer read its own rows.

Connections are per-thread and tracked in a lock-protected registry, so ``close()``
really closes all of them rather than leaking every worker thread's handle.
"""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Tuple,
)

from signledger.anchor.checkpoint import Checkpoint
from signledger.backends.base import (
    CAP_APPEND_ONLY_ENFORCED,
    CAP_ATOMIC_APPEND,
    CAP_DURABLE_FSYNC,
    CAP_UNIQUE_CHAIN_LINK,
    check_link,
    checkpoint_from_row,
    checkpoint_to_row,
    entry_from_row,
    entry_to_row,
    retry_on_conflict,
    timestamp_to_micros,
    validate_identifier,
    validate_window,
)
from signledger.core.entry import Entry
from signledger.core.exceptions import StorageError

__all__ = ["SQLiteBackend"]

DEFAULT_ENTRIES_TABLE = "ledger_entries"
DEFAULT_CHECKPOINTS_TABLE = "ledger_checkpoints"

_ENTRY_COLUMNS: Tuple[str, ...] = (
    "sequence",
    "entry_id",
    "ledger_id",
    "format_version",
    "timestamp",
    "timestamp_us",
    "data",
    "metadata",
    "previous_hash",
    "hash",
    "key_id",
    "signature_algorithm",
    "signature",
)

_CHECKPOINT_COLUMNS: Tuple[str, ...] = (
    "ledger_id",
    "sequence",
    "head_hash",
    "entry_count",
    "merkle_root",
    "timestamp",
    "timestamp_us",
    "signature",
    "key_id",
    "format_version",
)

#: sqlite3 error text that means "another writer got there first", not "your data is bad".
_CONFLICT_MARKERS = ("locked", "busy")


class SQLiteBackend:
    """A durable, append-only ledger store in a single SQLite database."""

    def __init__(
        self,
        path: str = ":memory:",
        *,
        entries_table: str = DEFAULT_ENTRIES_TABLE,
        checkpoints_table: str = DEFAULT_CHECKPOINTS_TABLE,
        timeout: float = 5.0,
        check_same_thread: bool = True,
    ) -> None:
        self.path = path
        self.entries_table = validate_identifier(entries_table, kind="entries table name")
        self.checkpoints_table = validate_identifier(
            checkpoints_table, kind="checkpoints table name"
        )
        self.timeout = timeout
        self._closed = False
        self._registry_lock = threading.RLock()
        self._connections: Dict[int, sqlite3.Connection] = {}

        # ``check_same_thread=True`` (the default) means "do not share a connection
        # between threads", so every thread is handed its own. ``False`` means the caller
        # accepts sharing, which here is one connection serialised by a lock.
        #
        # An in-memory database has no choice: it cannot be reopened by name, so a second
        # connection would silently be a second, empty database.
        self._in_memory = _is_memory_path(path)
        self._shared = self._in_memory or not check_same_thread
        self._check_same_thread = check_same_thread
        # Typed as a context manager so both arms are one type: a real lock when a
        # connection is shared, and nothing at all when the transaction is the only
        # authority that matters.
        self._access_lock: ContextManager[Any]
        if self._shared:
            self._access_lock = threading.RLock()
        else:
            self._access_lock = _NullLock()

        first = self._open_connection()
        with self._registry_lock:
            self._connections[_SHARED_KEY if self._shared else threading.get_ident()] = first
        self._create_schema(first)

    # -- connections -------------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        # Translated here rather than at the call sites so that *every* route to a new
        # connection inherits it - the constructor, and the lazy per-thread open in
        # `_connection`. Opening is a backend operation like any other: contract section 4
        # requires it to fail as StorageError, and section 10 forbids the driver message
        # (which quotes the file path) from reaching the caller.
        with _translating("open"):
            return self._connect()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout,
            # The registry is the guarantee, not this flag: a connection is either owned
            # by exactly one thread or serialised by ``_access_lock``. The driver's own
            # check has to be off regardless, because ``close()`` must be able to reclaim
            # the connection of a worker thread that has already exited - otherwise every
            # thread that ever touched the ledger leaks a file handle.
            check_same_thread=False,
            isolation_level=None,  # we issue BEGIN IMMEDIATE / COMMIT ourselves
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        if not self._in_memory:
            connection.execute("PRAGMA journal_mode = WAL")
            # WAL defaults to synchronous=NORMAL, which does not fsync on commit. The
            # backend advertises durable_fsync, so it must actually be durable.
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _connection(self) -> sqlite3.Connection:
        """Return this thread's connection, opening and registering one if needed."""
        if self._closed:
            raise StorageError("backend is closed")
        key = _SHARED_KEY if self._shared else threading.get_ident()
        with self._registry_lock:
            connection = self._connections.get(key)
            if connection is None:
                connection = self._open_connection()
                self._connections[key] = connection
            return connection

    def close(self) -> None:
        """Close every connection this backend opened, on every thread. Idempotent."""
        with self._registry_lock:
            self._closed = True
            connections = list(self._connections.values())
            self._connections.clear()
        for connection in connections:
            connection.close()

    # -- schema ------------------------------------------------------------------------

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        with _translating("create_schema"):
            self._create_schema_once(connection)

    def _create_schema_once(self, connection: sqlite3.Connection) -> None:
        entries, checkpoints = self.entries_table, self.checkpoints_table
        connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {entries} (
                sequence            INTEGER NOT NULL,
                entry_id            TEXT    NOT NULL,
                ledger_id           TEXT    NOT NULL,
                format_version      INTEGER NOT NULL,
                timestamp           TEXT    NOT NULL,
                timestamp_us        INTEGER NOT NULL,
                data                TEXT    NOT NULL,
                metadata            TEXT    NOT NULL,
                previous_hash       TEXT    NOT NULL,
                hash                TEXT    NOT NULL,
                key_id              TEXT,
                signature_algorithm TEXT,
                signature           TEXT,
                UNIQUE (sequence),
                UNIQUE (previous_hash),
                UNIQUE (entry_id),
                UNIQUE (hash)
            );

            CREATE INDEX IF NOT EXISTS {entries}_timestamp_us
                ON {entries} (timestamp_us);

            CREATE TABLE IF NOT EXISTS {checkpoints} (
                ledger_id      TEXT    NOT NULL,
                sequence       INTEGER NOT NULL,
                head_hash      TEXT    NOT NULL,
                entry_count    INTEGER NOT NULL,
                merkle_root    TEXT    NOT NULL,
                timestamp      TEXT    NOT NULL,
                timestamp_us   INTEGER NOT NULL,
                signature      TEXT,
                key_id         TEXT,
                format_version INTEGER NOT NULL
            );

            -- Append-only enforcement. A row that can be edited in place is not evidence
            -- of anything, so the database itself refuses UPDATE and DELETE.
            CREATE TRIGGER IF NOT EXISTS {entries}_no_update
            BEFORE UPDATE ON {entries}
            BEGIN
                SELECT RAISE(ABORT, 'signledger: entries are append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS {entries}_no_delete
            BEFORE DELETE ON {entries}
            BEGIN
                SELECT RAISE(ABORT, 'signledger: entries are append-only');
            END;
            """
        )

    # -- append ------------------------------------------------------------------------

    def append_atomic(self, build: Callable[[Optional[Entry]], Entry]) -> Entry:
        """Read the tip and insert its successor inside one ``BEGIN IMMEDIATE``."""
        with _translating("append"):
            return retry_on_conflict(self._append_once(build), _is_conflict)

    def _append_once(self, build: Callable[[Optional[Entry]], Entry]) -> Callable[[], Entry]:
        def attempt() -> Entry:
            connection = self._connection()
            with self._access_lock:
                # IMMEDIATE, not DEFERRED: the write lock is taken before the tip is read,
                # so no other writer can slip an entry in between the read and the insert.
                connection.execute("BEGIN IMMEDIATE")
                try:
                    tip = self._select_head(connection)
                    entry = build(tip)
                    check_link(tip, entry)
                    self._insert(connection, entry)
                except BaseException:
                    connection.rollback()
                    raise
                connection.commit()
                return entry

        return attempt

    def _insert(self, connection: sqlite3.Connection, entry: Entry) -> None:
        row = entry_to_row(entry)
        columns = ", ".join(_ENTRY_COLUMNS)
        placeholders = ", ".join("?" for _ in _ENTRY_COLUMNS)
        connection.execute(
            f"INSERT INTO {self.entries_table} ({columns}) VALUES ({placeholders})",  # noqa: S608 - identifier is allowlisted by validate_identifier()
            [row[column] for column in _ENTRY_COLUMNS],
        )

    # -- reads -------------------------------------------------------------------------

    def _select_head(self, connection: sqlite3.Connection) -> Optional[Entry]:
        return self._select_one(connection, "ORDER BY sequence DESC LIMIT 1", ())

    def _select_one(
        self, connection: sqlite3.Connection, tail: str, params: Tuple[Any, ...]
    ) -> Optional[Entry]:
        columns = ", ".join(_ENTRY_COLUMNS)
        cursor = connection.execute(
            f"SELECT {columns} FROM {self.entries_table} {tail}", list(params)  # noqa: S608 - identifier is allowlisted by validate_identifier()
        )
        row = cursor.fetchone()
        return None if row is None else entry_from_row(dict(row))

    def get_entry(self, entry_id: str) -> Optional[Entry]:
        with _translating("get_entry"):
            return self._select_one(
                self._connection(), "WHERE entry_id = ? LIMIT 1", (entry_id,)
            )

    def get_by_sequence(self, sequence: int) -> Optional[Entry]:
        with _translating("get_by_sequence"):
            return self._select_one(
                self._connection(), "WHERE sequence = ? LIMIT 1", (sequence,)
            )

    def get_head(self) -> Optional[Entry]:
        with _translating("get_head"):
            return self._select_head(self._connection())

    def get_oldest(self) -> Optional[Entry]:
        with _translating("get_oldest"):
            return self._select_one(self._connection(), "ORDER BY sequence ASC LIMIT 1", ())

    def count(self) -> int:
        with _translating("count"):
            cursor = self._connection().execute(f"SELECT COUNT(*) FROM {self.entries_table}")  # noqa: S608 - identifier is allowlisted by validate_identifier()
            return int(cursor.fetchone()[0])

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
        sql, params = build_entry_query(
            self.entries_table,
            start_seq=start_seq,
            end_seq=end_seq,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
        with _translating("iter_entries"):
            cursor = self._connection().execute(sql, params)
            return iter([entry_from_row(dict(row)) for row in cursor.fetchall()])

    def size_bytes(self) -> int:
        """On-disk size of the database, pages included."""
        connection = self._connection()
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        return page_count * page_size

    # -- capabilities ------------------------------------------------------------------

    def capabilities(self) -> FrozenSet[str]:
        """Report what this deployment actually enforces, never what it aspires to."""
        tokens = {CAP_ATOMIC_APPEND, CAP_UNIQUE_CHAIN_LINK, CAP_APPEND_ONLY_ENFORCED}
        if not self._in_memory and _synchronous_is_durable(self._connection()):
            tokens.add(CAP_DURABLE_FSYNC)
        return frozenset(tokens)

    # -- checkpoints -------------------------------------------------------------------

    def save_checkpoint(self, cp: Checkpoint) -> None:
        row = checkpoint_to_row(cp)
        columns = ", ".join(_CHECKPOINT_COLUMNS)
        placeholders = ", ".join("?" for _ in _CHECKPOINT_COLUMNS)
        with _translating("save_checkpoint"):
            connection = self._connection()
            with self._access_lock:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        f"INSERT INTO {self.checkpoints_table} ({columns}) "  # noqa: S608 - identifier is allowlisted by validate_identifier()
                        f"VALUES ({placeholders})",
                        [row[column] for column in _CHECKPOINT_COLUMNS],
                    )
                except BaseException:
                    connection.rollback()
                    raise
                connection.commit()

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        columns = ", ".join(_CHECKPOINT_COLUMNS)
        with _translating("get_latest_checkpoint"):
            cursor = self._connection().execute(
                f"SELECT {columns} FROM {self.checkpoints_table} "  # noqa: S608 - identifier is allowlisted by validate_identifier()
                f"ORDER BY sequence DESC, rowid DESC LIMIT 1"
            )
            row = cursor.fetchone()
        return None if row is None else checkpoint_from_row(dict(row))

    def iter_checkpoints(self) -> Iterator[Checkpoint]:
        columns = ", ".join(_CHECKPOINT_COLUMNS)
        with _translating("iter_checkpoints"):
            cursor = self._connection().execute(
                f"SELECT {columns} FROM {self.checkpoints_table} ORDER BY sequence ASC, rowid ASC"  # noqa: S608 - identifier is allowlisted by validate_identifier()
            )
            return iter([checkpoint_from_row(dict(row)) for row in cursor.fetchall()])


# --------------------------------------------------------------------------------------
# query construction
# --------------------------------------------------------------------------------------


def build_entry_query(
    table: str,
    *,
    start_seq: Optional[int] = None,
    end_seq: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Tuple[str, List[Any]]:
    """Build the ``SELECT`` for :meth:`SQLiteBackend.iter_entries`.

    Two details are load-bearing:

    * ``ORDER BY sequence`` - the ordering authority is the sequence number, never the
      wall clock.
    * SQLite has no bare ``OFFSET``: ``... OFFSET 5`` without a ``LIMIT`` is a syntax
      error. When an offset is given without a limit the query must say
      ``LIMIT -1 OFFSET ?``, which means "no limit, skip N".
    """
    clauses: List[str] = []
    params: List[Any] = []
    if start_seq is not None:
        clauses.append("sequence >= ?")
        params.append(start_seq)
    if end_seq is not None:
        clauses.append("sequence <= ?")
        params.append(end_seq)
    if start_time is not None:
        clauses.append("timestamp_us >= ?")
        params.append(timestamp_to_micros(start_time))
    if end_time is not None:
        clauses.append("timestamp_us <= ?")
        params.append(timestamp_to_micros(end_time))

    sql = f"SELECT {', '.join(_ENTRY_COLUMNS)} FROM {table}"  # noqa: S608 - identifier is allowlisted by validate_identifier()
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY sequence ASC"

    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    elif offset:
        sql += " LIMIT -1"
    if offset:
        sql += " OFFSET ?"
        params.append(offset)
    return sql, params


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


@contextmanager
def _translating(operation: str) -> Iterator[None]:
    """Re-raise driver errors as :class:`StorageError`.

    A caller must never have to catch ``sqlite3.OperationalError`` to use a ledger, and
    an append that hits one has to fail closed rather than be swallowed into a log line.
    Only the exception *type* is reported: a driver message can quote the row it choked
    on, and contract section 10 forbids payloads in error text.
    """
    try:
        yield
    except sqlite3.Error as err:
        raise StorageError(f"{operation} failed: {type(err).__name__}") from err


def _is_memory_path(path: str) -> bool:
    return path in ("", ":memory:") or path.startswith("file::memory:")


def _is_conflict(err: BaseException) -> bool:
    """Recognise "someone else won the race" as distinct from "your data is wrong"."""
    if isinstance(err, sqlite3.IntegrityError):
        return True
    if isinstance(err, sqlite3.OperationalError):
        text = str(err).lower()
        return any(marker in text for marker in _CONFLICT_MARKERS)
    return False


def _synchronous_is_durable(connection: sqlite3.Connection) -> bool:
    """True when SQLite fsyncs on commit (``synchronous`` is FULL or EXTRA)."""
    return int(connection.execute("PRAGMA synchronous").fetchone()[0]) >= 2


class _NullLock:
    """No-op lock for file databases, where the transaction is the only authority.

    A process-local lock must never be what makes an append safe - that was exactly v1's
    mistake, and it forks the chain the moment a second process starts.
    """

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


_SHARED_KEY = -1
