"""SQLite defects that only appear when something else is holding the file.

Four of them, every one observed in a real deployment rather than imagined:

* opening a fresh database from several processes at once raised ``OperationalError``
  in the losers, because SQLite does not run the busy handler for a journal-mode change
  and ``PRAGMA busy_timeout`` therefore does not cover ``PRAGMA journal_mode = WAL``;
* the cursor a ``PRAGMA`` hands back was never closed, so on an interpreter without
  refcounting the statement was still open at the first ``COMMIT`` and every on-disk
  append failed;
* a ``PRAGMA`` that raised during construction stranded the connection, leaking an fd
  and eventually a ``ResourceWarning`` attributed to some unrelated test;
* ``capabilities()`` asserted ``append_only_enforced`` instead of probing for it, so a
  store whose triggers had been dropped still reported the top assurance level.

Everything here is **deterministic**: locks are held explicitly by a second connection
and handed over at a named point, never by starting workers and hoping they collide. The
one genuinely multi-process test is marked ``slow``. A flaky test is worse than no test.
"""

import multiprocessing
import sqlite3
import time
from typing import Any, Callable, List, Optional, Set

import pytest

from signledger.backends.base import (
    CAP_APPEND_ONLY_ENFORCED,
    MAX_APPEND_ATTEMPTS,
    BackendConformanceSuite,
)
from signledger.backends.sqlite import SQLiteBackend
from signledger.core.exceptions import StorageError

TRIGGER_SUFFIXES = ("_no_update", "_no_delete")


# --------------------------------------------------------------------------------------
# instrumentation
#
# A real sqlite3 connection on a real file, watched. Nothing is simulated except where a
# test says so explicitly, and each hook exists because the behaviour it exposes cannot
# be produced any other way on a local filesystem.
# --------------------------------------------------------------------------------------


class FrozenJournalMode:
    """The answer SQLite gives when it could not switch the file's journal mode.

    ``PRAGMA journal_mode = WAL`` reports the mode that is *actually* in force, and on a
    store that cannot support WAL - a network share is the documented case - it answers
    with the old mode and does not raise. That cannot be produced on a local disk, so it
    is the one thing here that is injected rather than provoked.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.closed = False

    def fetchone(self) -> Any:
        return (self.mode,)

    def close(self) -> None:
        self.closed = True


class Probe:
    """What the backend asked its connections to do, and a place to intervene."""

    def __init__(self) -> None:
        self.connections: List[sqlite3.Connection] = []
        self.pragma_cursors: List[Any] = []
        self.journal_mode_attempts = 0
        #: Called with the attempt number *before* each ``PRAGMA journal_mode``.
        self.before_journal_mode: Optional[Callable[[int], None]] = None
        #: When set, ``PRAGMA journal_mode`` answers with this instead of running.
        self.journal_mode_answer: Optional[str] = None
        #: Keep every pragma cursor alive, the way a non-refcounting interpreter does.
        self.hoard_pragma_cursors = False

    def leaked_connections(self) -> List[sqlite3.Connection]:
        return [c for c in self.connections if not is_closed(c)]


def is_closed(connection: sqlite3.Connection) -> bool:
    """A closed sqlite3 connection refuses every operation with ``ProgrammingError``."""
    try:
        connection.execute("SELECT 1").close()
    except sqlite3.ProgrammingError:
        return True
    except sqlite3.Error:
        # It answered the way an open connection does - unhappily, but open. A file that
        # is not a database is the case that gets here.
        return False
    return False


@pytest.fixture
def probe(monkeypatch: Any) -> Probe:
    """Hand every ``sqlite3.connect`` in this test a watched connection subclass."""
    watcher = Probe()
    real_connect = sqlite3.connect

    class Watched(sqlite3.Connection):
        def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
            statement = sql.strip().upper()
            if statement.startswith("PRAGMA JOURNAL_MODE"):
                watcher.journal_mode_attempts += 1
                if watcher.before_journal_mode is not None:
                    watcher.before_journal_mode(watcher.journal_mode_attempts)
                if watcher.journal_mode_answer is not None:
                    return FrozenJournalMode(watcher.journal_mode_answer)
            cursor = super().execute(sql, *args, **kwargs)
            if watcher.hoard_pragma_cursors and statement.startswith("PRAGMA"):
                watcher.pragma_cursors.append(cursor)
            return cursor

    def watched_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        kwargs["factory"] = Watched
        connection = real_connect(*args, **kwargs)
        watcher.connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", watched_connect)
    return watcher


def hold_the_write_lock(path: str) -> sqlite3.Connection:
    """Return a connection sitting inside ``BEGIN IMMEDIATE`` on ``path``.

    This is exactly the state a second worker process is in for the microsecond it takes
    to set up its own database, and it is what makes ``PRAGMA journal_mode = WAL`` raise
    SQLITE_BUSY without ever consulting the busy handler.
    """
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("CREATE TABLE anvil (x)").close()
    holder.execute("BEGIN IMMEDIATE").close()
    holder.execute("INSERT INTO anvil VALUES (1)").close()
    return holder


def trigger_names(path: str, table: str = "ledger_entries") -> Set[str]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = ?",
            (table,),
        ).fetchall()
    finally:
        connection.close()
    return {str(row[0]) for row in rows}


def drop_triggers(path: str, *suffixes: str, table: str = "ledger_entries") -> None:
    """Remove the append-only triggers, as the insider of contract section 1 would."""
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        for suffix in suffixes or TRIGGER_SUFFIXES:
            connection.execute(f"DROP TRIGGER IF EXISTS {table}{suffix}").close()
    finally:
        connection.close()


# --------------------------------------------------------------------------------------
# A1 - the WAL pragma race
# --------------------------------------------------------------------------------------


class TestOpeningAContendedDatabase:
    """Four gunicorn workers opening one fresh file is not an exotic deployment."""

    def test_the_wal_pragma_is_retried_when_another_writer_holds_the_file(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        """The lock is handed over at a named point, so the schedule is not a guess.

        Attempt one runs against a genuinely locked database and must fail; the file is
        released immediately before attempt two, which must therefore succeed.
        """
        path = str(tmp_path / "contended.db")
        holder = hold_the_write_lock(path)

        def release(attempt: int) -> None:
            if attempt == 2:
                holder.rollback()

        probe.before_journal_mode = release
        try:
            backend = SQLiteBackend(path)
            try:
                assert probe.journal_mode_attempts == 2
                assert BackendConformanceSuite().append(backend).sequence == 0
            finally:
                backend.close()
        finally:
            holder.close()

    def test_a_file_that_stays_locked_is_a_storage_error_after_bounded_attempts(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        """Bounded, because an unbounded retry under contention is a livelock."""
        path = str(tmp_path / "wedged.db")
        holder = hold_the_write_lock(path)
        try:
            with pytest.raises(StorageError) as caught:
                SQLiteBackend(path)

            assert probe.journal_mode_attempts == MAX_APPEND_ATTEMPTS
            # Contract section 10: no path, no payload, no connection string.
            assert path not in str(caught.value)
        finally:
            holder.rollback()
            holder.close()

    def test_a_store_that_never_switches_to_wal_is_reported_not_assumed(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        """The pragma's answer is the authority, not the absence of an exception."""
        probe.journal_mode_answer = "delete"

        with pytest.raises(StorageError) as caught:
            SQLiteBackend(str(tmp_path / "network-share.db"))

        assert "delete" in str(caught.value)

    def test_nothing_is_left_open_when_the_journal_mode_never_becomes_wal(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        probe.journal_mode_answer = "delete"

        with pytest.raises(StorageError):
            SQLiteBackend(str(tmp_path / "network-share-2.db"))

        assert probe.leaked_connections() == []


# --------------------------------------------------------------------------------------
# A2 - pragma cursors that outlive their statement
# --------------------------------------------------------------------------------------


class TestPragmaCursorsAreClosed:
    """``PRAGMA journal_mode`` returns a row, so its cursor holds an open statement.

    CPython's refcounting collects it on the next line and hides the bug. PyPy does not,
    and the first ``COMMIT`` fails with "cannot commit transaction - SQL statements in
    progress" - the backend is simply non-functional on disk there. Holding a reference
    to every pragma cursor reproduces that here, deterministically, on any interpreter.
    """

    def test_an_append_commits_when_the_construction_pragmas_are_not_collected(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        probe.hoard_pragma_cursors = True
        backend = SQLiteBackend(str(tmp_path / "hoarded.db"))
        try:
            assert BackendConformanceSuite().append(backend).sequence == 0
        finally:
            backend.close()

    def test_no_pragma_anywhere_leaves_its_cursor_open(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        """Every pragma the backend issues, not only the ones on the open path.

        ``size_bytes`` and the ``synchronous`` probe behind ``capabilities`` run one
        each, and a cursor left open by either is the same latent COMMIT failure.
        """
        probe.hoard_pragma_cursors = True
        backend = SQLiteBackend(str(tmp_path / "hoarded-closed.db"))
        try:
            backend.capabilities()
            backend.size_bytes()

            assert len(probe.pragma_cursors) >= 5
            for cursor in probe.pragma_cursors:
                with pytest.raises(sqlite3.ProgrammingError):
                    cursor.fetchone()
        finally:
            backend.close()


# --------------------------------------------------------------------------------------
# A3 - a failed construction must not strand a file descriptor
# --------------------------------------------------------------------------------------


class TestAFailedConstructionClosesItsConnection:
    """``filterwarnings = ["error"]`` turns the eventual ``ResourceWarning: unclosed
    database`` into a hard failure attached to whichever unrelated test the collector
    happens to interrupt. The fd has to go back when the constructor gives up."""

    def test_a_pragma_that_raises_closes_the_connection_before_re_raising(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        path = tmp_path / "not-a-database.db"
        path.write_bytes(b"this file is not a database\n" * 64)

        with pytest.raises(StorageError):
            SQLiteBackend(str(path))

        assert probe.leaked_connections() == []

    def test_a_failing_schema_creation_closes_the_connection_before_re_raising(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        """A decoy trigger already owns the name the append-only trigger needs.

        The trigger cannot be created, so the store can never enforce append-only - and
        ``CREATE TRIGGER IF NOT EXISTS`` would have shrugged that off and carried on.
        """
        path = str(tmp_path / "decoy.db")
        decoy = sqlite3.connect(path, isolation_level=None)
        decoy.execute("CREATE TABLE decoy (x)").close()
        decoy.execute(
            "CREATE TRIGGER ledger_entries_no_update BEFORE UPDATE ON decoy "
            "BEGIN SELECT 1; END"
        ).close()
        decoy.close()

        with pytest.raises(StorageError):
            SQLiteBackend(path)

        assert probe.leaked_connections() == []

    def test_a_half_built_schema_is_rolled_back_rather_than_left_behind(
        self, tmp_path: Any
    ) -> None:
        """The entries table and its triggers land together or not at all - a table
        without its triggers is indistinguishable from a tampered one."""
        path = str(tmp_path / "decoy-rollback.db")
        decoy = sqlite3.connect(path, isolation_level=None)
        decoy.execute("CREATE TABLE decoy (x)").close()
        decoy.execute(
            "CREATE TRIGGER ledger_entries_no_update BEFORE UPDATE ON decoy "
            "BEGIN SELECT 1; END"
        ).close()
        decoy.close()

        with pytest.raises(StorageError):
            SQLiteBackend(path)

        connection = sqlite3.connect(path)
        try:
            names = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        finally:
            connection.close()
        assert "ledger_entries" not in names


# --------------------------------------------------------------------------------------
# A4 - capabilities() must probe, never assert
# --------------------------------------------------------------------------------------


class TestAppendOnlyEnforcementIsProbed:
    """Contract section 1: the guarantee is deployment-dependent and reported at
    runtime. A store whose triggers an insider dropped enforces nothing, and saying
    ``worm`` about it is the exact contradiction this library exists to prevent."""

    def test_a_store_with_both_triggers_reports_append_only_enforcement(
        self, tmp_path: Any
    ) -> None:
        backend = SQLiteBackend(str(tmp_path / "intact.db"))
        try:
            assert CAP_APPEND_ONLY_ENFORCED in backend.capabilities()
        finally:
            backend.close()

    def test_dropping_both_triggers_withdraws_the_claim(self, tmp_path: Any) -> None:
        backend = SQLiteBackend(str(tmp_path / "unlocked.db"))
        try:
            assert CAP_APPEND_ONLY_ENFORCED in backend.capabilities()

            drop_triggers(backend.path)

            assert CAP_APPEND_ONLY_ENFORCED not in backend.capabilities()
        finally:
            backend.close()

    @pytest.mark.parametrize("suffix", TRIGGER_SUFFIXES)
    def test_one_surviving_trigger_is_not_append_only_enforcement(
        self, tmp_path: Any, suffix: str
    ) -> None:
        """Half the guarantee is none of it: a store that refuses UPDATE but permits
        DELETE is a store an insider can truncate."""
        backend = SQLiteBackend(str(tmp_path / f"half{suffix}.db"))
        try:
            drop_triggers(backend.path, suffix)

            assert CAP_APPEND_ONLY_ENFORCED not in backend.capabilities()
        finally:
            backend.close()

    def test_reopening_does_not_silently_restore_dropped_triggers(
        self, tmp_path: Any
    ) -> None:
        """``CREATE TRIGGER IF NOT EXISTS`` on an existing table repaired the evidence
        and handed the ``worm`` claim straight back."""
        path = str(tmp_path / "reopened.db")
        first = SQLiteBackend(path)
        BackendConformanceSuite().append(first)
        first.close()
        drop_triggers(path)

        second = SQLiteBackend(path)
        try:
            assert trigger_names(path) == set()
            assert CAP_APPEND_ONLY_ENFORCED not in second.capabilities()
        finally:
            second.close()

    def test_a_tampered_store_still_opens_so_that_it_can_be_audited(
        self, tmp_path: Any
    ) -> None:
        """Refusing to open would make the one tool that can report the damage useless.
        The damage is reported through ``capabilities()``, which is where contract
        section 1 says a deployment-dependent guarantee is reported."""
        path = str(tmp_path / "auditable.db")
        first = SQLiteBackend(path)
        BackendConformanceSuite().append_many(first, 3)
        first.close()
        drop_triggers(path)

        second = SQLiteBackend(path)
        try:
            assert second.count() == 3
            assert [e.sequence for e in second.iter_entries()] == [0, 1, 2]
        finally:
            second.close()

    def test_reopening_an_intact_store_leaves_its_triggers_alone(
        self, tmp_path: Any
    ) -> None:
        path = str(tmp_path / "intact-reopen.db")
        first = SQLiteBackend(path)
        BackendConformanceSuite().append(first)
        first.close()

        second = SQLiteBackend(path)
        try:
            assert trigger_names(path) == {
                f"ledger_entries{suffix}" for suffix in TRIGGER_SUFFIXES
            }
            assert CAP_APPEND_ONLY_ENFORCED in second.capabilities()
            assert second.count() == 1
        finally:
            second.close()

    def test_an_in_memory_store_reports_enforcement_from_its_own_triggers(self) -> None:
        backend = SQLiteBackend(":memory:")
        try:
            assert CAP_APPEND_ONLY_ENFORCED in backend.capabilities()
        finally:
            backend.close()


# --------------------------------------------------------------------------------------
# the deployment that reported A1: several processes, one fresh file
# --------------------------------------------------------------------------------------


def open_and_close(path: str, start_at: float) -> str:
    """Open the backend at ``start_at``, in a separate OS process."""
    from signledger.backends.sqlite import SQLiteBackend as Backend

    while time.time() < start_at:
        pass
    Backend(path).close()
    return "ok"


@pytest.mark.slow
def test_four_processes_opening_one_fresh_file_all_come_up(tmp_path: Any) -> None:
    """The gunicorn ``-w 4`` case. Before the retry, 11 of 20 opens raised."""
    context = multiprocessing.get_context("spawn")
    for trial in range(5):
        path = str(tmp_path / f"trial{trial}.db")
        start_at = time.time() + 1.0
        with context.Pool(4) as pool:
            results = pool.starmap(open_and_close, [(path, start_at)] * 4)

        assert results == ["ok"] * 4

        reader = SQLiteBackend(path)
        try:
            assert reader.count() == 0
            assert CAP_APPEND_ONLY_ENFORCED in reader.capabilities()
        finally:
            reader.close()
