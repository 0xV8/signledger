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

Four more were found by running the library rather than reading it, and each is held
down by a test in ``tests/test_sqlite_concurrency.py``:

* ``PRAGMA journal_mode = WAL`` is **not** covered by ``PRAGMA busy_timeout`` - SQLite
  does not consult the busy handler for a journal-mode change - so four workers opening
  one fresh file meant three of them raised. It is retried, and the mode the pragma
  *answers with* is what decides success.
* every ``PRAGMA`` cursor is closed. It holds an open statement, and on an interpreter
  without refcounting that statement is still open at the first ``COMMIT``.
* a constructor that gives up closes the connection it opened, rather than leaving an fd
  for the cycle collector and a ``ResourceWarning`` for some unrelated test.
* ``capabilities()`` probes for the append-only triggers instead of asserting them, and
  the schema is never re-created around an entries table that already exists.

``create=False`` opens a store that already exists and provisions nothing: no
``CREATE TABLE``, no triggers, and - the one that surprises people - no
``PRAGMA journal_mode = WAL``, which is itself a write to the database header. See
:meth:`SQLiteBackend._configure`.
"""

import os
import pathlib
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

#: The journal mode the backend runs in, and the answer ``PRAGMA journal_mode`` must give.
_WAL_MODE = "wal"

#: Suffixes of the two ``BEFORE`` triggers that make the entries table append-only.
_APPEND_ONLY_TRIGGER_SUFFIXES = ("_no_update", "_no_delete")



def _file_uri(path: "pathlib.PurePath") -> str:
    """The ``file:`` URI naming ``path``, in the spelling SQLite documents.

    Hand-assembling this as ``"file:" + quote(path)`` is right on POSIX and wrong on
    Windows: ``quote`` escapes the drive colon and every backslash, SQLite percent-decodes
    the result back to ``C:\\Users\\me\\audit.db``, and since that does not begin with
    ``/`` it is treated as a **relative** filename resolved against the working directory.
    A read-only verify would then open some other file, or nothing.

    ``PurePath.as_uri`` is the platform's own answer - ``file:///C:/Users/me/audit.db`` on
    Windows, ``file:///var/lib/audit.db`` on POSIX - and it escapes ``?`` and ``#``, which
    SQLite would otherwise read as the start of its own query string.
    """
    return path.as_uri()


class SQLiteBackend:
    """A durable, append-only ledger store in a single SQLite database.

    ``create`` defaults to True, which is the behaviour every writer wants: the schema is
    provisioned on open. Pass ``create=False`` to read a store that already exists,
    provisioning nothing and issuing no write at all - what ``signledger verify`` does,
    because an integrity check that provisions the thing it is checking cannot be run
    against a copy this process may not write, and a mistyped path was answered with
    "your ledger is empty" rather than with "that is not a ledger".
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        entries_table: str = DEFAULT_ENTRIES_TABLE,
        checkpoints_table: str = DEFAULT_CHECKPOINTS_TABLE,
        timeout: float = 5.0,
        check_same_thread: bool = True,
        create: bool = True,
    ) -> None:
        self.path = path
        self.create = bool(create)
        self.entries_table = validate_identifier(entries_table, kind="entries table name")
        self.checkpoints_table = validate_identifier(
            checkpoints_table, kind="checkpoints table name"
        )
        self._trigger_names = frozenset(
            f"{self.entries_table}{suffix}" for suffix in _APPEND_ONLY_TRIGGER_SUFFIXES
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

        if not self.create:
            # Before the first connect, because ``sqlite3.connect`` on a path that is not
            # there *creates* it: a mistyped path used to leave a 32 KiB database behind
            # and then report that the ledger it had just invented was empty.
            self._require_existing_file()

        first = self._open_connection()
        try:
            with self._registry_lock:
                self._connections[_SHARED_KEY if self._shared else threading.get_ident()] = first
            if self.create:
                self._create_schema(first)
            else:
                self._require_existing_tables(first)
        except BaseException:
            # The caller never receives this instance, so nothing else can ever close
            # ``first``. Left alone it holds an fd until the cycle collector runs and
            # then raises ``ResourceWarning: unclosed database`` from inside whatever
            # unrelated code was executing at the time - which, with
            # ``filterwarnings = ["error"]``, is a red test somewhere else entirely.
            self.close()
            raise

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
        connection = self._raw_connect()
        try:
            self._configure(connection)
        except BaseException:
            # Nothing holds this handle yet, so a pragma that raises would strand it.
            connection.close()
            raise
        return connection

    def _raw_connect(self) -> sqlite3.Connection:
        """Open the file. Read-only opens go through a URI so they can touch nothing.

        ``sqlite3.connect(path)`` always opens read-write, whatever statements follow. For
        a WAL database that means SQLite wants the ``-shm`` wal-index, so a ``create=False``
        open of an evidence snapshot both littered a writable directory with ``-wal`` and
        ``-shm``, and *failed outright* on read-only media - the exact case this mode
        exists for. An auditor is handed the ``.db``; the sidecars are transient and do not
        come with it.

        ``mode=ro`` is tried first because it stays correct against a store another process
        is still appending to, which is what ``signledger verify`` does in production. Only
        if SQLite cannot proceed - which on a WAL database means it could not create the
        wal-index, i.e. nobody can be writing through this directory either - does it fall
        back to ``immutable=1``, which needs no sidecar at all. Getting that order wrong
        would read a live ledger as though it were frozen.
        """
        if self.create:
            return self._open(self.path, uri=False)

        uri = _file_uri(pathlib.Path(os.path.abspath(self.path)))
        connection = self._open(uri + "?mode=ro", uri=True)
        try:
            # Force SQLite to reach the wal-index now rather than on the first read, so
            # the fallback below is chosen here or not at all.
            connection.execute("SELECT count(*) FROM sqlite_master").close()
        except sqlite3.OperationalError:
            # Close the one being abandoned before opening its replacement. Left open it
            # is reachable by nothing and is finalised whenever the collector next runs,
            # emitting ``ResourceWarning: unclosed database`` from inside an unrelated
            # test - which under ``filterwarnings = ["error"]`` fails that test instead of
            # this code. CPython 3.11 reclaimed it quietly; 3.13 and 3.14 do not.
            connection.close()
            return self._open(uri + "?mode=ro&immutable=1", uri=True)
        return connection

    def _open(self, target: str, *, uri: bool) -> sqlite3.Connection:
        return sqlite3.connect(
            target,
            timeout=self.timeout,
            # The registry is the guarantee, not this flag: a connection is either owned
            # by exactly one thread or serialised by ``_access_lock``. The driver's own
            # check has to be off regardless, because ``close()`` must be able to reclaim
            # the connection of a worker thread that has already exited - otherwise every
            # thread that ever touched the ledger leaks a file handle.
            check_same_thread=False,
            isolation_level=None,  # we issue BEGIN IMMEDIATE / COMMIT ourselves
            uri=uri,
        )

    def _configure(self, connection: sqlite3.Connection) -> None:
        """Set this connection up. Under ``create=False`` it may not write to the file.

        ``PRAGMA journal_mode = WAL`` is not a session setting - it rewrites the database
        header - so a read-only open must not issue it, and does not. The mode the file
        is already in is the mode we read it in; a rollback-journal ledger stays one.

        ``PRAGMA query_only`` then makes SQLite itself refuse every write on this
        connection, which is the difference between a backend that intends not to write
        and one that cannot. ``synchronous`` and ``foreign_keys`` are per-connection
        settings that touch no file, so they are applied either way - keeping
        :meth:`capabilities` answering the same thing about a store however it was opened.

        What this does *not* claim, because it would not be true: SQLite's own housekeeping
        is still SQLite's. Reading a WAL database whose ``-shm`` wal-index is absent
        materialises one, and closing the last connection to a WAL database checkpoints
        whatever an earlier *writer* left in the ``-wal``. Neither is a statement this
        backend issues - ``query_only`` refuses every one of those - and neither alters a
        stored row; but a directory that cannot be written at all still cannot host a
        WAL-mode read, which is a SQLite requirement rather than a signledger one.
        """
        connection.row_factory = sqlite3.Row
        _pragma(connection, "PRAGMA foreign_keys = ON")
        _pragma(connection, f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        if not self._in_memory:
            if self.create:
                self._enable_wal(connection)
            # WAL defaults to synchronous=NORMAL, which does not fsync on commit. The
            # backend advertises durable_fsync, so it must actually be durable.
            _pragma(connection, "PRAGMA synchronous = FULL")
        if not self.create:
            _pragma(connection, "PRAGMA query_only = ON")

    def _enable_wal(self, connection: sqlite3.Connection) -> None:
        """Put the file into WAL mode, tolerating another process doing it right now.

        SQLite does **not** run the busy handler for a journal-mode change, so the
        ``busy_timeout`` set on the line above does not cover this statement: four
        gunicorn workers starting together on one fresh file meant three of them got
        SQLITE_BUSY straight out of ``__init__`` and the master never came up. The retry
        is the same bounded, exponential one the append path uses - contention is not a
        data error, it is someone else being first.

        The pragma's *answer* is what decides success, not the absence of an exception.
        Once any process has switched the file, every later ``PRAGMA journal_mode = WAL``
        replies ``wal`` without needing the write lock at all: that is a win, not a race
        to re-run. A store that replies with some other mode never switched, and running
        a journal the backend does not advertise is exactly the silent divergence this
        library exists to refuse.
        """

        def attempt() -> str:
            return _pragma_value(connection, "PRAGMA journal_mode = WAL")

        try:
            mode = retry_on_conflict(attempt, _is_conflict)
        except StorageError as err:
            raise StorageError(
                "open failed: the database file is locked by another writer"
            ) from err
        if mode != _WAL_MODE:
            raise StorageError(
                f"open failed: journal mode is {mode!r}, not {_WAL_MODE!r}"
            )

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

    def _require_existing_file(self) -> None:
        """Refuse to conjure the store a read-only open was pointed at.

        A ``StorageError`` and never an integrity verdict: "that store does not exist" and
        "this ledger does not verify" are different answers to different questions, and
        the caller above turns them into different exit codes.
        """
        if self._in_memory:
            raise StorageError(
                "open failed: an in-memory sqlite database is created empty every time it "
                "is opened, so there is nothing in it to read; name a file instead"
            )
        if not os.path.exists(self.path):
            raise StorageError(
                f"open failed: the sqlite database {self.path} does not exist, and this "
                f"backend was opened with create=False, which never creates a store"
            )

    def _require_existing_tables(self, connection: sqlite3.Connection) -> None:
        """Refuse a database that holds no ledger, naming the table that is absent.

        A file that exists but holds no entries table is a different fault from an empty
        ledger, and much the likelier one: a mistyped ``--entries-table``, or a path that
        names somebody else's database.
        """
        with _translating("open"):
            missing = [
                name
                for name in (self.entries_table, self.checkpoints_table)
                if not _object_exists(connection, "table", name)
            ]
        if missing:
            raise StorageError(
                f"open failed: this store does not exist - the sqlite database "
                f"{self.path} holds no table called {', '.join(missing)}, and this "
                f"backend was opened with create=False, which never creates one"
            )

    def _refuse_write(self, operation: str) -> None:
        """A store opened for reading is not written to, and says so in one message.

        Without this the refusal arrives from SQLite as ``attempt to write a readonly
        database``, which reads like a permissions problem on the operator's file rather
        than the deliberate choice it is.
        """
        if not self.create:
            raise StorageError(
                f"{operation} failed: this backend was opened with create=False, which "
                f"reads an existing store and never writes to it"
            )

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        with _translating("create_schema"):
            self._create_schema_once(connection)

    def _create_schema_once(self, connection: sqlite3.Connection) -> None:
        # One transaction, not ``executescript``. ``executescript`` commits between
        # statements, so a second process opening the same fresh file could see the
        # entries table after it was created but before its triggers landed - which is
        # indistinguishable from a table whose triggers an insider removed. Wrapped in
        # ``BEGIN IMMEDIATE`` the schema appears whole or not at all, and concurrent
        # creators serialise on the write lock (which *is* covered by busy_timeout).
        connection.execute("BEGIN IMMEDIATE").close()
        try:
            fresh = not _object_exists(connection, "table", self.entries_table)
            for statement in self._schema_statements(fresh=fresh):
                connection.execute(statement).close()
        except BaseException:
            connection.rollback()
            raise
        connection.commit()

    def _schema_statements(self, *, fresh: bool) -> List[str]:
        """The DDL for this store. Triggers only when the entries table is being made.

        An entries table that already exists is not re-armed. ``CREATE TRIGGER IF NOT
        EXISTS`` used to put the append-only triggers back on every open, which silently
        repaired the single most informative piece of evidence a ledger can carry - that
        somebody with DDL rights removed them. It is reported instead: the triggers stay
        gone, :meth:`capabilities` stops claiming ``append_only_enforced``, and
        ``assurance_level()`` therefore stops saying ``worm``.

        Opening still *succeeds*, deliberately. Refusing would make a tampered ledger
        unreadable by the one tool able to describe the damage, and ``signledger verify``
        on exactly such a store is the flow this library exists for.
        """
        entries, checkpoints = self.entries_table, self.checkpoints_table
        statements = [
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
            )
            """,
            f"CREATE INDEX IF NOT EXISTS {entries}_timestamp_us ON {entries} (timestamp_us)",
            f"""
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
            )
            """,
        ]
        if fresh:
            # Append-only enforcement. A row that can be edited in place is not evidence
            # of anything, so the database itself refuses UPDATE and DELETE. No
            # ``IF NOT EXISTS``: the table was just created, so the only way one of these
            # names is taken is that something else in the file already owns it, and a
            # store that cannot arm its own triggers must say so rather than shrug.
            statements += [
                f"""
                CREATE TRIGGER {entries}_no_update
                BEFORE UPDATE ON {entries}
                BEGIN
                    SELECT RAISE(ABORT, 'signledger: entries are append-only');
                END
                """,
                f"""
                CREATE TRIGGER {entries}_no_delete
                BEFORE DELETE ON {entries}
                BEGIN
                    SELECT RAISE(ABORT, 'signledger: entries are append-only');
                END
                """,
            ]
        return statements

    # -- append ------------------------------------------------------------------------

    def append_atomic(self, build: Callable[[Optional[Entry]], Entry]) -> Entry:
        """Read the tip and insert its successor inside one ``BEGIN IMMEDIATE``."""
        self._refuse_write("append")
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
        page_count = int(_pragma_value(connection, "PRAGMA page_count"))
        page_size = int(_pragma_value(connection, "PRAGMA page_size"))
        return page_count * page_size

    # -- capabilities ------------------------------------------------------------------

    def capabilities(self) -> FrozenSet[str]:
        """Report what this deployment actually enforces, never what it aspires to.

        Every token here is **probed**. ``append_only_enforced`` used to be asserted
        unconditionally, one line above a properly probed ``durable_fsync``, so a store
        whose triggers had been dropped still reported the top assurance level - a
        library whose whole job is saying what its guarantees are worth, saying the wrong
        thing about the one guarantee an insider can remove in a single statement.
        """
        tokens = {CAP_ATOMIC_APPEND, CAP_UNIQUE_CHAIN_LINK}
        with _translating("capabilities"):
            connection = self._connection()
            if self._append_only_is_enforced(connection):
                tokens.add(CAP_APPEND_ONLY_ENFORCED)
            if not self._in_memory and _synchronous_is_durable(connection):
                tokens.add(CAP_DURABLE_FSYNC)
        return frozenset(tokens)

    def _append_only_is_enforced(self, connection: sqlite3.Connection) -> bool:
        """True only when **both** append-only triggers are attached right now.

        Half the guarantee is none of it: a table that refuses UPDATE but permits DELETE
        is a table an insider can truncate, and truncation is the attack a bare hash
        chain cannot see.
        """
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = ?",
            (self.entries_table,),
        )
        try:
            found = {str(row[0]) for row in cursor.fetchall()}
        finally:
            cursor.close()
        return self._trigger_names <= found

    # -- checkpoints -------------------------------------------------------------------

    def save_checkpoint(self, cp: Checkpoint) -> None:
        self._refuse_write("save_checkpoint")
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


def _pragma(connection: sqlite3.Connection, statement: str) -> None:
    """Run a ``PRAGMA`` and close the cursor it hands back.

    ``connection.execute`` returns a cursor whose statement is still open, and
    ``PRAGMA journal_mode`` returns a row, so the statement really is mid-flight.
    CPython's refcounting reclaims the cursor on the next line and hides that; PyPy does
    not, and the first ``COMMIT`` fails with "cannot commit transaction - SQL statements
    in progress" - every on-disk append, from the very first one.
    """
    connection.execute(statement).close()


def _pragma_value(connection: sqlite3.Connection, statement: str) -> str:
    """Run a ``PRAGMA``, return its single answer, and close the cursor either way."""
    cursor = connection.execute(statement)
    try:
        return str(cursor.fetchone()[0]).lower()
    finally:
        cursor.close()


def _object_exists(connection: sqlite3.Connection, kind: str, name: str) -> bool:
    """True when the schema already holds an object of ``kind`` called ``name``."""
    cursor = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ? LIMIT 1", (kind, name)
    )
    try:
        return cursor.fetchone() is not None
    finally:
        cursor.close()


def _synchronous_is_durable(connection: sqlite3.Connection) -> bool:
    """True when SQLite fsyncs on commit (``synchronous`` is FULL or EXTRA)."""
    return int(_pragma_value(connection, "PRAGMA synchronous")) >= 2


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
