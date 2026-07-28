"""PostgreSQL storage backend (architecture contract, section 4).

Every decision below is a v1 defect that had to stop being possible.

* **Identifiers go through** :class:`psycopg.sql.Identifier`, never an f-string. v1 had
  roughly 25 interpolation sites; ``validate_identifier`` allowlists the name and
  ``Identifier`` quotes it, so both halves of the door are shut.
* **Timestamps are re-read from ``timestamp_us``**, a ``BIGINT`` of integer epoch
  microseconds, never from the ``TIMESTAMPTZ``. v1 rebuilt entries from the ``TIMESTAMPTZ``,
  which Postgres renders in the *session* zone, so a server running ``America/New_York``
  re-hashed every row differently and reported a pristine log as tampered. The
  ``TIMESTAMPTZ`` is kept for operators to query by hand; it is never part of a digest.
* **``metadata`` is ``json.dumps(entry.metadata or {})`` and the column is ``NOT NULL``.**
  ``{}`` is falsy, so v1's ``metadata or None`` wrote NULL and the ledger could no longer
  read its own rows.
* **The tip read and the insert are one transaction**, held by
  ``SELECT ... ORDER BY sequence DESC LIMIT 1 FOR UPDATE`` plus a transaction-scoped
  advisory lock. v1 did read-modify-write under a process-local lock, which forks the
  chain the moment a second gunicorn worker exists.
* **A connection is rolled back before it goes back to the pool, on every path.** v1
  returned connections still carrying a failed transaction, so the next borrower inherited
  ``current transaction is aborted`` and the ledger wedged.
* **``ORDER BY sequence``**, never ``ORDER BY timestamp``.
"""

import hashlib
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, FrozenSet, Iterator, List, Optional, Sequence, Tuple, cast

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

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
from signledger.core.exceptions import ConfigurationError, StorageError

__all__ = [
    "DEFAULT_CHECKPOINTS_TABLE",
    "DEFAULT_ENTRIES_TABLE",
    "PostgresBackend",
    "append_only_revoke_script",
]

DEFAULT_ENTRIES_TABLE = "ledger_entries"
DEFAULT_CHECKPOINTS_TABLE = "ledger_checkpoints"

#: Postgres truncates anything longer, silently. Derived names are shortened explicitly
#: so two tables whose first 63 characters agree cannot share one trigger.
MAX_IDENTIFIER_LENGTH = 63

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

#: ``synchronous_commit`` settings under which a committed transaction is on disk locally.
#: ``local`` and ``off`` are not: ``off`` acknowledges before fsync, and ``local`` is only
#: durable if ``fsync`` is also on, which is checked separately.
_DURABLE_COMMIT_MODES = frozenset({"on", "remote_apply", "remote_write", "local"})


# --------------------------------------------------------------------------------------
# connection pool
# --------------------------------------------------------------------------------------


class _ConnectionPool:
    """A bounded, thread-safe pool of PostgreSQL connections.

    Threaded, not "simple": every operation is guarded by a condition variable, so two
    worker threads cannot be handed the same connection. v1 used a non-threadsafe pool
    from a threaded web server, which interleaves two transactions on one socket.
    """

    def __init__(self, dsn: str, *, minconn: int = 1, maxconn: int = 8, **kwargs: Any) -> None:
        # Validated before the first connect, so a misconfigured pool cannot leak a socket
        # on its way to raising.
        if minconn < 0:
            raise ConfigurationError(f"minconn must be zero or greater, got {minconn}")
        if maxconn < 1:
            raise ConfigurationError(f"maxconn must be at least 1, got {maxconn}")
        if minconn > maxconn:
            raise ConfigurationError(f"minconn ({minconn}) cannot exceed maxconn ({maxconn})")
        self._dsn = dsn
        self._kwargs = kwargs
        self._maxconn = maxconn
        self._condition = threading.Condition(threading.Lock())
        self._idle: List[psycopg.Connection] = []
        self._opened = 0
        self._closed = False
        for _ in range(minconn):
            self._idle.append(self._connect())
            self._opened += 1

    def _connect(self) -> psycopg.Connection:
        # autocommit stays off: a ledger append is a transaction, not three statements
        # that happen to run next to each other.
        return psycopg.connect(self._dsn, autocommit=False, **self._kwargs)

    def getconn(self) -> psycopg.Connection:
        with self._condition:
            while True:
                if self._closed:
                    raise StorageError("backend is closed")
                if self._idle:
                    return self._idle.pop()
                if self._opened < self._maxconn:
                    self._opened += 1
                    break
                self._condition.wait()
        try:
            return self._connect()
        except BaseException:
            with self._condition:
                self._opened -= 1
                self._condition.notify()
            raise

    def putconn(self, connection: psycopg.Connection, *, discard: bool = False) -> None:
        """Return a connection. ``discard`` closes it instead - a broken socket must not
        be handed to the next borrower."""
        with self._condition:
            if discard or self._closed:
                self._opened -= 1
                self._condition.notify()
            else:
                self._idle.append(connection)
                self._condition.notify()
                return
        connection.close()

    def closeall(self) -> None:
        with self._condition:
            self._closed = True
            idle, self._idle = self._idle, []
            self._opened -= len(idle)
            self._condition.notify_all()
        for connection in idle:
            connection.close()

    @property
    def idle_count(self) -> int:
        with self._condition:
            return len(self._idle)

    @property
    def opened_count(self) -> int:
        with self._condition:
            return self._opened


# --------------------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------------------


class PostgresBackend:
    """A durable, append-only ledger store in PostgreSQL."""

    def __init__(
        self,
        dsn: str,
        *,
        entries_table: str = DEFAULT_ENTRIES_TABLE,
        checkpoints_table: str = DEFAULT_CHECKPOINTS_TABLE,
        minconn: int = 1,
        maxconn: int = 8,
        connect_timeout: int = 10,
    ) -> None:
        self.entries_table = validate_identifier(entries_table, kind="entries table name")
        self.checkpoints_table = validate_identifier(
            checkpoints_table, kind="checkpoints table name"
        )
        self.timestamp_index = _derived_identifier(self.entries_table, "_ts_us_idx")
        self.append_only_function = _derived_identifier(self.entries_table, "_append_only")
        self.append_only_trigger = _derived_identifier(self.entries_table, "_no_write")
        self._closed = False
        self._session_timezone: Optional[str] = None
        #: Derived from the table name, so two *processes* writing the same ledger contend
        #: on one lock. A process-local lock is exactly what forked the v1 chain.
        self._lock_key = _advisory_key(self.entries_table)
        with _translating("connect"):
            self._pool = _ConnectionPool(
                dsn, minconn=minconn, maxconn=maxconn, connect_timeout=connect_timeout
            )
        try:
            self._create_schema()
        except BaseException:
            self._pool.closeall()
            raise

    # -- connections -------------------------------------------------------------------

    @contextmanager
    def _borrow(self) -> Iterator[psycopg.Connection]:
        """Lend a connection and **always** roll it back before taking it back.

        A connection returned mid-transaction poisons its next borrower: every statement
        answers ``current transaction is aborted`` until someone rolls back. v1 leaked
        exactly that, so one bad append disabled the ledger until the process restarted.
        """
        if self._closed:
            raise StorageError("backend is closed")
        connection = self._pool.getconn()
        try:
            self._apply_session_settings(connection)
            yield connection
        except BaseException:
            self._pool.putconn(connection, discard=_unwind(connection))
            raise
        else:
            self._pool.putconn(connection, discard=_unwind(connection))

    def _apply_session_settings(self, connection: psycopg.Connection) -> None:
        """Push the requested session zone onto a freshly borrowed connection.

        Committed immediately: a plain ``SET`` belongs to the open transaction, and
        ``_unwind`` rolls that transaction back, so an uncommitted setting would silently
        evaporate before the caller ever saw it.
        """
        if self._session_timezone is None:
            return
        with connection.cursor() as cur:
            cur.execute("SELECT set_config('TimeZone', %s, false)", (self._session_timezone,))
        connection.commit()

    def set_session_timezone(self, zone: str) -> None:
        """Set the PostgreSQL session ``TimeZone`` on every connection this backend uses.

        It exists so that the zone can be proven *harmless*. An entry is rebuilt from
        ``timestamp_us``, never from the ``TIMESTAMPTZ``, so no session zone can move a
        digest - which is exactly what v1 got wrong, reporting a healthy log as tampered
        on any server not running UTC.
        """
        if not isinstance(zone, str) or not zone:
            raise ConfigurationError(
                f"session timezone must be a non-empty string, got a {type(zone).__name__}"
            )
        previous = self._session_timezone
        self._session_timezone = zone
        try:
            with _translating("set_session_timezone"), self._borrow():
                pass
        except BaseException as err:
            self._session_timezone = previous
            cause = err.__cause__
            if isinstance(cause, psycopg.errors.InvalidParameterValue):
                raise ConfigurationError(f"unknown session timezone: {zone!r}") from err
            raise

    def close(self) -> None:
        """Close every pooled connection. Idempotent."""
        self._closed = True
        self._pool.closeall()

    # -- schema ------------------------------------------------------------------------

    def _create_schema(self) -> None:
        entries = sql.Identifier(self.entries_table)
        checkpoints = sql.Identifier(self.checkpoints_table)
        with self._borrow() as connection, connection.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {entries} (
                        sequence            BIGINT      NOT NULL,
                        entry_id            TEXT        NOT NULL,
                        ledger_id           TEXT        NOT NULL,
                        format_version      INTEGER     NOT NULL,
                        timestamp           TIMESTAMPTZ NOT NULL,
                        timestamp_us        BIGINT      NOT NULL,
                        data                TEXT        NOT NULL,
                        metadata            TEXT        NOT NULL,
                        previous_hash       TEXT        NOT NULL,
                        hash                TEXT        NOT NULL,
                        key_id              TEXT,
                        signature_algorithm TEXT,
                        signature           TEXT,
                        UNIQUE (sequence),
                        UNIQUE (previous_hash),
                        UNIQUE (entry_id),
                        UNIQUE (hash)
                    )
                    """
                ).format(entries=entries)
            )
            # sequence, entry_id, hash and previous_hash are already indexed by their
            # UNIQUE constraints. timestamp_us is the only other queried column, and it
            # is what every time filter compares on.
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {index} ON {entries} (timestamp_us)").format(
                    index=sql.Identifier(self.timestamp_index), entries=entries
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {checkpoints} (
                        id             BIGSERIAL PRIMARY KEY,
                        ledger_id      TEXT        NOT NULL,
                        sequence       BIGINT      NOT NULL,
                        head_hash      TEXT        NOT NULL,
                        entry_count    BIGINT      NOT NULL,
                        merkle_root    TEXT        NOT NULL,
                        timestamp      TIMESTAMPTZ NOT NULL,
                        timestamp_us   BIGINT      NOT NULL,
                        signature      TEXT,
                        key_id         TEXT,
                        format_version INTEGER     NOT NULL
                    )
                    """
                ).format(checkpoints=checkpoints)
            )
            self._create_append_only_guard(cur, entries)
            connection.commit()

    def _create_append_only_guard(self, cur: psycopg.Cursor, entries: sql.Identifier) -> None:
        """Make the database itself refuse ``UPDATE`` and ``DELETE`` on entries.

        The contract names "rules" here. A ``RULE ... DO INSTEAD NOTHING`` *silently
        discards* the statement, so a delete an auditor believes was refused reports
        success. A ``BEFORE`` trigger that raises is the same prohibition that also tells
        the truth, so that is what ships.
        """
        cur.execute(
            sql.SQL(
                """
                CREATE OR REPLACE FUNCTION {function}() RETURNS trigger AS $guard$
                BEGIN
                    RAISE EXCEPTION 'signledger: entries are append-only';
                END;
                $guard$ LANGUAGE plpgsql
                """
            ).format(function=sql.Identifier(self.append_only_function))
        )
        cur.execute(
            sql.SQL("DROP TRIGGER IF EXISTS {trigger} ON {entries}").format(
                trigger=sql.Identifier(self.append_only_trigger), entries=entries
            )
        )
        cur.execute(
            sql.SQL(
                "CREATE TRIGGER {trigger} BEFORE UPDATE OR DELETE ON {entries} "
                "FOR EACH ROW EXECUTE FUNCTION {function}()"
            ).format(
                trigger=sql.Identifier(self.append_only_trigger),
                entries=entries,
                function=sql.Identifier(self.append_only_function),
            )
        )

    # -- append ------------------------------------------------------------------------

    def append_atomic(self, build: Callable[[Optional[Entry]], Entry]) -> Entry:
        """Read the tip and insert its successor inside one transaction."""
        with _translating("append"):
            return retry_on_conflict(self._append_once(build), _is_conflict)

    def _append_once(self, build: Callable[[Optional[Entry]], Entry]) -> Callable[[], Entry]:
        def attempt() -> Entry:
            with self._borrow() as connection:
                with connection.cursor(row_factory=dict_row) as cur:
                    # FOR UPDATE locks the tip row - but an empty table has no row to
                    # lock, which is precisely when two writers both believe they are
                    # writing genesis. The transaction-scoped advisory lock covers that
                    # gap and is released by COMMIT or ROLLBACK, never left behind.
                    cur.execute("SELECT pg_advisory_xact_lock(%s)", (self._lock_key,))
                    tip = self._select_one(
                        cur, sql.SQL("ORDER BY sequence DESC LIMIT 1 FOR UPDATE"), ()
                    )
                entry = build(tip)
                check_link(tip, entry)
                with connection.cursor() as cur:
                    self._insert(cur, entry)
                connection.commit()
                return entry

        return attempt

    def _insert(self, cur: psycopg.Cursor, entry: Entry) -> None:
        row = entry_to_row(entry)
        # The TIMESTAMPTZ is written from the real datetime for operators; the digest is
        # rebuilt from timestamp_us, so no session zone can touch it.
        row["timestamp"] = entry.timestamp
        cur.execute(
            sql.SQL("INSERT INTO {entries} ({columns}) VALUES ({placeholders})").format(
                entries=sql.Identifier(self.entries_table),
                columns=sql.SQL(", ").join(sql.Identifier(c) for c in _ENTRY_COLUMNS),
                placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in _ENTRY_COLUMNS),
            ),
            [row[column] for column in _ENTRY_COLUMNS],
        )

    # -- reads -------------------------------------------------------------------------

    def _select_one(
        self, cur: "psycopg.Cursor[Any]", tail: sql.Composable, params: Tuple[Any, ...]
    ) -> Optional[Entry]:
        cur.execute(
            sql.SQL("SELECT {columns} FROM {entries} {tail}").format(
                columns=sql.SQL(", ").join(sql.Identifier(c) for c in _ENTRY_COLUMNS),
                entries=sql.Identifier(self.entries_table),
                tail=tail,
            ),
            list(params),
        )
        row = cur.fetchone()
        return None if row is None else entry_from_row(dict(row))

    def _fetch_one(self, tail: sql.Composable, params: Tuple[Any, ...]) -> Optional[Entry]:
        with self._borrow() as connection, connection.cursor(row_factory=dict_row) as cur:
            return self._select_one(cur, tail, params)

    def get_entry(self, entry_id: str) -> Optional[Entry]:
        with _translating("get_entry"):
            return self._fetch_one(sql.SQL("WHERE entry_id = %s LIMIT 1"), (entry_id,))

    def get_by_sequence(self, sequence: int) -> Optional[Entry]:
        with _translating("get_by_sequence"):
            return self._fetch_one(sql.SQL("WHERE sequence = %s LIMIT 1"), (sequence,))

    def get_head(self) -> Optional[Entry]:
        with _translating("get_head"):
            return self._fetch_one(sql.SQL("ORDER BY sequence DESC LIMIT 1"), ())

    def get_oldest(self) -> Optional[Entry]:
        with _translating("get_oldest"):
            return self._fetch_one(sql.SQL("ORDER BY sequence ASC LIMIT 1"), ())

    def count(self) -> int:
        with _translating("count"), self._borrow() as connection, connection.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {entries}").format(
                    entries=sql.Identifier(self.entries_table)
                )
            )
            return int(_one_row(cur)[0])

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
        query, params = build_entry_query(
            self.entries_table,
            start_seq=start_seq,
            end_seq=end_seq,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )
        with _translating("iter_entries"), self._borrow() as connection:
            with connection.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return iter([entry_from_row(dict(row)) for row in rows])

    def size_bytes(self) -> int:
        """Bytes the entries and checkpoints relations occupy, indexes included."""
        with _translating("size_bytes"), self._borrow() as connection, connection.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(pg_total_relation_size(to_regclass(%s)), 0) "
                "+ COALESCE(pg_total_relation_size(to_regclass(%s)), 0)",
                (self.entries_table, self.checkpoints_table),
            )
            return int(_one_row(cur)[0])

    # -- capabilities ------------------------------------------------------------------

    def capabilities(self) -> FrozenSet[str]:
        """Report what this deployment **actually** enforces, read from the catalog.

        ``append_only_enforced`` is not a constant: an operator can drop the trigger, and
        a backend that kept claiming it would let ``assurance_level()`` say WORM about a
        table anyone can edit. ``worm`` is never reported at all - a trigger can be
        dropped by anyone who can create one.
        """
        tokens = {CAP_ATOMIC_APPEND, CAP_UNIQUE_CHAIN_LINK}
        with _translating("capabilities"), self._borrow() as connection:
            with connection.cursor() as cur:
                enforced = self._append_only_trigger_exists(cur)
                fsync, commit_mode = self._fsync_settings(cur)
        if enforced:
            tokens.add(CAP_APPEND_ONLY_ENFORCED)
        if fsync == "on" and commit_mode in _DURABLE_COMMIT_MODES:
            tokens.add(CAP_DURABLE_FSYNC)
        return frozenset(tokens)

    def _append_only_trigger_exists(self, cur: psycopg.Cursor) -> bool:
        """True when the append-only trigger is still attached to the entries table."""
        cur.execute(
            "SELECT COUNT(*) FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
            "WHERE c.relname = %s AND t.tgname = %s AND NOT t.tgisinternal",
            (self.entries_table, self.append_only_trigger),
        )
        return int(_one_row(cur)[0]) > 0

    def _fsync_settings(self, cur: psycopg.Cursor) -> Tuple[str, str]:
        """``(fsync, synchronous_commit)`` as the *server* currently has them."""
        cur.execute("SELECT current_setting('fsync'), current_setting('synchronous_commit')")
        row = _one_row(cur)
        return (str(row[0]), str(row[1]))

    # -- checkpoints -------------------------------------------------------------------

    def save_checkpoint(self, cp: Checkpoint) -> None:
        row = checkpoint_to_row(cp)
        row["timestamp"] = cp.timestamp
        with _translating("save_checkpoint"), self._borrow() as connection:
            with connection.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {checkpoints} ({columns}) VALUES ({placeholders})"
                    ).format(
                        checkpoints=sql.Identifier(self.checkpoints_table),
                        columns=sql.SQL(", ").join(
                            sql.Identifier(c) for c in _CHECKPOINT_COLUMNS
                        ),
                        placeholders=sql.SQL(", ").join(
                            sql.Placeholder() for _ in _CHECKPOINT_COLUMNS
                        ),
                    ),
                    [row[column] for column in _CHECKPOINT_COLUMNS],
                )
            connection.commit()

    def _checkpoint_query(self, tail: sql.Composable) -> sql.Composed:
        return sql.SQL("SELECT {columns} FROM {checkpoints} {tail}").format(
            columns=sql.SQL(", ").join(sql.Identifier(c) for c in _CHECKPOINT_COLUMNS),
            checkpoints=sql.Identifier(self.checkpoints_table),
            tail=tail,
        )

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        with _translating("get_latest_checkpoint"), self._borrow() as connection:
            with connection.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    self._checkpoint_query(sql.SQL("ORDER BY sequence DESC, id DESC LIMIT 1"))
                )
                row = cur.fetchone()
        return None if row is None else checkpoint_from_row(dict(row))

    def iter_checkpoints(self) -> Iterator[Checkpoint]:
        with _translating("iter_checkpoints"), self._borrow() as connection:
            with connection.cursor(row_factory=dict_row) as cur:
                cur.execute(self._checkpoint_query(sql.SQL("ORDER BY sequence ASC, id ASC")))
                rows = cur.fetchall()
        return iter([checkpoint_from_row(dict(row)) for row in rows])


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
) -> Tuple[sql.Composed, List[Any]]:
    """Build the ``SELECT`` for :meth:`PostgresBackend.iter_entries`.

    Three details are load-bearing:

    * ``ORDER BY sequence`` - the ordering authority is the sequence number, never the
      wall clock.
    * Time filters compare ``timestamp_us``, the integer the digest is rebuilt from, so a
      filter and a re-hash can never disagree about what an entry's instant is.
    * ``LIMIT`` is emitted when ``limit is not None``. ``if limit:`` turned a request for
      zero rows into a request for every row.
    """
    clauses: List[sql.Composable] = []
    params: List[Any] = []
    if start_seq is not None:
        clauses.append(sql.SQL("sequence >= %s"))
        params.append(start_seq)
    if end_seq is not None:
        clauses.append(sql.SQL("sequence <= %s"))
        params.append(end_seq)
    if start_time is not None:
        clauses.append(sql.SQL("timestamp_us >= %s"))
        params.append(timestamp_to_micros(start_time))
    if end_time is not None:
        clauses.append(sql.SQL("timestamp_us <= %s"))
        params.append(timestamp_to_micros(end_time))

    parts: List[sql.Composable] = [
        sql.SQL("SELECT {columns} FROM {entries}").format(
            columns=sql.SQL(", ").join(sql.Identifier(c) for c in _ENTRY_COLUMNS),
            entries=sql.Identifier(table),
        )
    ]
    if clauses:
        parts.append(sql.SQL("WHERE "))
        parts.append(sql.SQL(" AND ").join(clauses))
    parts.append(sql.SQL(" ORDER BY sequence ASC"))
    if limit is not None:
        parts.append(sql.SQL(" LIMIT %s"))
        params.append(limit)
    # Postgres, unlike SQLite, accepts a bare OFFSET, and OFFSET 0 is a no-op - so the
    # clause is emitted unconditionally rather than on a truthiness test.
    parts.append(sql.SQL(" OFFSET %s"))
    params.append(offset)
    return sql.Composed(parts), params


# --------------------------------------------------------------------------------------
# append-only privileges
# --------------------------------------------------------------------------------------


def append_only_revoke_script(
    role: str,
    *,
    entries_table: str = DEFAULT_ENTRIES_TABLE,
    checkpoints_table: str = DEFAULT_CHECKPOINTS_TABLE,
) -> str:
    """SQL that leaves ``role`` able to append and read, and nothing else.

    The trigger stops the application's own bugs. It does **not** stop anyone who can
    ``ALTER TABLE ... DISABLE TRIGGER``, so the deployment story is privileges: run this
    as the table owner, then run the application as ``role``. It is returned as text
    rather than executed, because a process granting privileges to itself proves nothing -
    the owner has to do it.
    """
    validate_identifier(role, kind="role name")
    validate_identifier(entries_table, kind="entries table name")
    validate_identifier(checkpoints_table, kind="checkpoints table name")
    statements = (
        "REVOKE ALL ON TABLE {entries} FROM {role};",
        "REVOKE ALL ON TABLE {checkpoints} FROM {role};",
        "GRANT SELECT, INSERT ON TABLE {entries} TO {role};",
        "GRANT SELECT, INSERT ON TABLE {checkpoints} TO {role};",
        "REVOKE UPDATE, DELETE, TRUNCATE ON TABLE {entries} FROM {role};",
        "REVOKE UPDATE, DELETE, TRUNCATE ON TABLE {checkpoints} FROM {role};",
    )
    rendered = [
        sql.SQL(statement)  # noqa: S608 - every placeholder is an allowlisted Identifier
        .format(
            entries=sql.Identifier(entries_table),
            checkpoints=sql.Identifier(checkpoints_table),
            role=sql.Identifier(role),
        )
        .as_string(None)
        for statement in statements
    ]
    return "\n".join(rendered) + "\n"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------



def _one_row(cur: "psycopg.Cursor[Any]") -> Sequence[Any]:
    """The single row an aggregate query is guaranteed to produce.

    ``fetchone()`` is typed ``Optional`` because most queries can return nothing, but
    ``SELECT count(*)`` and ``SELECT current_setting(...)`` always yield exactly one row.
    A ``cast`` states that, where an ``if row is None: raise`` would add a branch no test
    could reach - and an unreachable branch is a coverage hole pretending to be a guard.
    """
    return cast(Sequence[Any], cur.fetchone())


@contextmanager
def _translating(operation: str) -> Iterator[None]:
    """Re-raise driver errors as :class:`StorageError`.

    Only the exception *type* is reported. A psycopg message quotes the offending row and
    a connection error quotes the DSN, which can carry a password - contract section 10
    forbids either from reaching an exception message.
    """
    try:
        yield
    except psycopg.Error as err:
        raise StorageError(f"{operation} failed: {type(err).__name__}") from err


def _unwind(connection: psycopg.Connection) -> bool:
    """Roll a connection back before it returns to the pool. True means "discard it".

    Called on the success path too: a plain ``SELECT`` still opens a transaction when
    autocommit is off, and an idle-in-transaction connection holds its snapshot and blocks
    vacuum forever.
    """
    try:
        connection.rollback()
    except psycopg.Error:
        return True
    return False


def _is_conflict(err: BaseException) -> bool:
    """Recognise "another writer won the race" - never "your data is wrong".

    The append-only trigger also raises, as ``RaiseException``. Retrying that would hammer
    the server five times over a mistake that will never succeed, so it is excluded.
    """
    return isinstance(
        err,
        (
            psycopg.errors.UniqueViolation,
            psycopg.errors.SerializationFailure,
            psycopg.errors.DeadlockDetected,
            psycopg.errors.LockNotAvailable,
        ),
    )


def _derived_identifier(base: str, suffix: str) -> str:
    """Build ``<base><suffix>`` short enough that Postgres will not truncate it.

    Postgres silently cuts an identifier at 63 bytes, so two tables agreeing on their
    first 63 characters would otherwise end up sharing one trigger - and dropping the
    trigger on one would quietly un-protect the other.
    """
    return validate_identifier(
        base[: MAX_IDENTIFIER_LENGTH - len(suffix)] + suffix, kind="derived identifier"
    )


def _advisory_key(table: str) -> int:
    """A stable signed 64-bit advisory-lock key for a table name.

    Derived from the name and nothing else, so two processes - the case a process-local
    lock cannot cover - serialise against each other.
    """
    digest = hashlib.blake2b(table.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)
