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

import hashlib
import multiprocessing
import shutil
import sqlite3
import time
from contextlib import closing, suppress
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

import pytest

from signledger.backends.base import (
    CAP_APPEND_ONLY_ENFORCED,
    MAX_APPEND_ATTEMPTS,
    BackendConformanceSuite,
    build_next_entry,
)
from signledger.backends.sqlite import SQLiteBackend
from signledger.core.entry import Entry
from signledger.core.exceptions import StorageError

TRIGGER_SUFFIXES = ("_no_update", "_no_delete")

#: The shapes an evidence snapshot arrives in, and the files a read-only open of each may
#: leave behind beyond the ones it was handed.
#:
#: * ``db only`` - the ``.db`` of a WAL store on its own, which is what an auditor is
#:   emailed. The ``-wal`` and ``-shm`` are transient and do not travel with it.
#: * ``db and its wal`` - what ``cp -a`` of a running service's directory captures. Not
#:   exotic, and the shape that made the first CI run fail: SQLite can reach the wal-index
#:   from here, so the ``mode=ro`` open succeeds and materialises a ``-shm``.
#: * ``rollback journal db`` - a store whose file is not in WAL mode at all, the network
#:   share ``SQLiteBackend._configure`` documents. No wal-index is possible, so a read-only
#:   open of one must add *nothing*, on every build.
#:
#: The permitted additions are SQLite's own wal-index, never a signledger write:
#: ``query_only`` refuses every write statement on a read-only connection.
SNAPSHOT_SHAPES: Dict[str, Tuple[str, ...]] = {
    "db only": ("-wal", "-shm"),
    "db and its wal": ("-wal", "-shm"),
    "rollback journal db": (),
}


class AuditorSnapshot(NamedTuple):
    """The evidence an auditor was handed, and the ledger it is supposed to contain."""

    directory: Any
    path: Any
    written: List[Entry]


def _pragma_answer(connection: sqlite3.Connection, statement: str) -> str:
    """Run a ``PRAGMA`` on a raw connection, returning its answer and closing the cursor."""
    cursor = connection.execute(statement)
    try:
        return str(cursor.fetchone()[0]).lower()
    finally:
        cursor.close()


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
        #: When True, the first probe read a read-only open performs raises
        #: ``OperationalError``, which is how SQLite reports that it could not reach the
        #: wal-index. That is the branch that falls back to ``immutable=1``.
        self.fail_first_readonly_probe = False

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
def probe(monkeypatch: Any) -> Iterator[Probe]:
    """Hand every ``sqlite3.connect`` in this test a watched connection subclass.

    The teardown is not optional. This fixture deliberately keeps a reference to every
    connection it hands out - that is how ``leaked_connections()`` can answer at all - and
    several tests here open connections the library is *supposed* to abandon. Without a
    teardown those stay reachable until the ``Probe`` itself becomes garbage, and are then
    finalised at whatever arbitrary later moment the collector runs, emitting
    ``ResourceWarning: unclosed database`` from inside an unrelated test.

    Under ``filterwarnings = ["error"]`` that is a hard failure attributed to the wrong
    test. It was invisible on CPython 3.11 and failed the suite on 3.13 and 3.14, on every
    OS, blaming a Hypothesis tamper-detection test that only ever touches MemoryBackend.
    """
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
            if watcher.fail_first_readonly_probe and statement.startswith(
                "SELECT COUNT(*) FROM SQLITE_MASTER"
            ):
                watcher.fail_first_readonly_probe = False
                raise sqlite3.OperationalError("unable to open database file")
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
    yield watcher

    # Close what this fixture kept alive, and nothing else. Assertions have already run,
    # so a test that checks the library closed its own connections is unaffected.
    watcher.pragma_cursors.clear()
    for connection in watcher.connections:
        with suppress(sqlite3.Error):
            connection.close()
    watcher.connections.clear()


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


class TestAReadOnlyOpenWorksOnReadOnlyMedia:
    """``create=False`` exists so an auditor can read an evidence snapshot.

    That snapshot is very often on media the auditor cannot write: a file copied off with
    its permissions intact, a mounted image, a directory the audit team owns and the
    reader does not. ``sqlite3.connect(path)`` opens read-write even when every statement
    afterwards is read-only, so SQLite tries to create the ``-wal``/``-shm`` sidecars and
    the open fails outright on a 0444 file in a 0555 directory - failing precisely the
    case the mode was built for. Opening through a ``file:...?mode=ro`` URI asks SQLite
    for a read-only connection and lets it use the immutable path instead.

    **What is asserted here, and what deliberately is not.**

    The first CI run failed on ``assert ['evidence.db', 'evidence.db-shm',
    'evidence.db-wal'] == ['evidence.db']`` on the ubuntu and macos runners while passing
    on the machine it was written on, and it was the assertion that was wrong, not the
    backend. "No sidecar appears" is one SQLite build's answer, not a promise this library
    is in a position to make: materialising a wal-index is SQLite's own housekeeping, this
    backend issues no write at all on a read-only connection (``query_only`` refuses every
    one), and whether the housekeeping succeeds or is refused varies by build. sqlite 3.51
    here declines to build a wal-index for a snapshot that arrived without its ``-wal`` and
    falls back to ``immutable=1``; the runners' sqlite succeeds and writes both sidecars.
    Neither is a defect and neither is a guarantee.

    What *is* guaranteed, holds on every build, and is what an auditor actually needs:

    * every file the auditor was handed is byte-identical after the open;
    * every entry, hash and link reads back exactly as written;
    * nothing is removed, and the only files that may appear are SQLite's transient
      wal-index beside a WAL database - beside a rollback-journal database, nothing at all;
    * media that genuinely cannot be written to - a 0555 directory - still opens and reads.

    Each shape in :data:`SNAPSHOT_SHAPES` drives a different route through
    ``SQLiteBackend._raw_connect``, and all three run on every platform, so the routes are
    covered here rather than left to whichever one a given SQLite build happens to take.
    """

    def snapshot(self, directory: Any) -> Any:
        return sorted(p.name for p in directory.iterdir())

    def digests(self, directory: Any) -> Any:
        """Every file in ``directory`` by name, with the sha256 of its bytes."""
        return {
            entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
            for entry in directory.iterdir()
        }

    def handed_to_an_auditor(self, tmp_path: Any, shape: str = "db only") -> Any:
        """Copy the evidence out of a live store, in one of the shapes it arrives in.

        See :data:`SNAPSHOT_SHAPES`. The WAL is checkpointed before the copy, so the
        ``.db`` is complete on its own and every route must report the same ledger -
        without that, ``immutable=1`` (which ignores the ``-wal`` entirely) would honestly
        report a shorter ledger than ``mode=ro`` and no portable assertion would be
        possible.
        """
        rollback_journal = shape == "rollback journal db"
        source, written, bystander = self.build_store(
            tmp_path / "live", rollback_journal=rollback_journal
        )
        with closing(bystander):
            snapshot_dir = tmp_path / "snapshot"
            snapshot_dir.mkdir()
            target = snapshot_dir / "evidence.db"
            shutil.copy2(source, target)
            if shape == "db and its wal":
                shutil.copy2(f"{source}-wal", f"{target}-wal")
        return AuditorSnapshot(snapshot_dir, target, written)

    def build_store(self, tmp_path: Any, *, rollback_journal: bool = False) -> Any:
        """Three linked entries, and a second connection holding the ``-wal`` in place.

        SQLite checkpoints and deletes the ``-wal`` when the *last* connection to a WAL
        database closes, so without a bystander open there is no sidecar left to copy - and
        a bystander is exactly what production has: the service that is still running while
        the auditor takes the copy. The caller closes it.
        """
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "evidence.db"

        backend = SQLiteBackend(str(path))
        written = []
        try:
            for index in range(3):
                written.append(
                    backend.append_atomic(
                        lambda tip, i=index: build_next_entry(
                            tip,
                            ledger_id="evidence-ledger",
                            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                            data={"index": i},
                            timestamp=datetime(2024, 6, 1, 12, 0, i, tzinfo=timezone.utc),
                        )
                    )
                )
            bystander = sqlite3.connect(str(path))
        finally:
            backend.close()
        try:
            if rollback_journal:
                # The network-share deployment `SQLiteBackend._configure` names: a ledger
                # whose file is not in WAL mode at all. Converting checkpoints the -wal and
                # removes it, so this shape arrives as a lone .db by construction.
                mode = _pragma_answer(bystander, "PRAGMA journal_mode = DELETE")
                assert mode == "delete"
            else:
                # FULL, so every frame the -wal holds is also in the .db. The snapshot is
                # then complete whichever route the read-only open takes.
                _pragma_answer(bystander, "PRAGMA wal_checkpoint(FULL)")
        except BaseException:
            bystander.close()
            raise
        return path, written, bystander

    @pytest.mark.parametrize("shape", SNAPSHOT_SHAPES)
    def test_a_read_only_open_of_read_only_media_reads_every_entry(
        self, tmp_path: Any, shape: str
    ) -> None:
        """The case ``create=False`` exists for, and the one that holds everywhere.

        A 0555 directory is media nothing can be written to at all - no wal-index, no
        journal, nothing - so no SQLite build has a way to litter here and none has a way
        to disagree. This is also the assertion that fails outright if the read-only URI is
        ever dropped: a plain ``sqlite3.connect`` of the two WAL shapes here is
        ``StorageError: open failed: OperationalError``, measured.
        """
        snapshot = self.handed_to_an_auditor(tmp_path, shape)
        for entry in snapshot.directory.iterdir():
            entry.chmod(0o444)
        snapshot.directory.chmod(0o555)
        try:
            backend = SQLiteBackend(str(snapshot.path), create=False)
            try:
                assert backend.count() == 3
                assert list(backend.iter_entries()) == snapshot.written
            finally:
                backend.close()
        finally:
            snapshot.directory.chmod(0o755)
            for entry in snapshot.directory.iterdir():
                entry.chmod(0o644)

    @pytest.mark.parametrize("shape", SNAPSHOT_SHAPES)
    def test_a_read_only_open_leaves_every_file_it_was_handed_byte_identical(
        self, tmp_path: Any, shape: str
    ) -> None:
        """The guarantee an auditor actually needs, and a portable one.

        Not "no sidecar appeared" - that is an SQLite build's answer, and both answers are
        legitimate - but "the evidence I gave you came back unchanged".

        The snapshot is left **writable** on purpose. At 0444 the operating system is what
        prevents a write, so the assertion would pass even against a backend that opened
        read-write and would be measuring the chmod rather than the library. Writable, the
        only thing standing between the evidence and a write is the backend's own choice
        of ``mode=ro`` and ``query_only``. Measured teeth: on the ``rollback journal db``
        shape an open that provisions - ``create=True``, i.e. ``PRAGMA journal_mode = WAL``
        rewriting the header - changes ``evidence.db`` and fails this.
        """
        snapshot = self.handed_to_an_auditor(tmp_path, shape)
        before = self.digests(snapshot.directory)

        backend = SQLiteBackend(str(snapshot.path), create=False)
        try:
            assert backend.count() == 3
        finally:
            backend.close()

        after = self.digests(snapshot.directory)
        assert {name: after.get(name) for name in before} == before

    @pytest.mark.parametrize("shape", SNAPSHOT_SHAPES)
    def test_a_read_only_open_adds_nothing_but_sqlites_own_wal_index(
        self, tmp_path: Any, shape: str
    ) -> None:
        """A writable directory is the subtler case: the open succeeds, and may litter.

        Whether it litters is SQLite's decision, so what is pinned is the *bound*: at most
        the transient wal-index of the database being read, and beside a rollback-journal
        database not even that, on any build. A ``-journal``, a second database, a renamed
        or deleted evidence file all fail this.
        """
        snapshot = self.handed_to_an_auditor(tmp_path, shape)
        before = set(self.snapshot(snapshot.directory))

        backend = SQLiteBackend(str(snapshot.path), create=False)
        try:
            assert backend.count() == 3
        finally:
            backend.close()

        after = set(self.snapshot(snapshot.directory))
        permitted = {
            f"{snapshot.path.name}{suffix}" for suffix in SNAPSHOT_SHAPES[shape]
        }
        assert after - before <= permitted
        assert before - after == set()

    @pytest.mark.parametrize("shape", SNAPSHOT_SHAPES)
    def test_a_read_only_open_reads_back_every_entry_and_every_digest(
        self, tmp_path: Any, shape: str
    ) -> None:
        """No stored row, hash or link changes - the same guarantee, read from inside.

        Byte-identity says the file did not move; this says the ledger inside it did not
        either, which is what a verdict is computed from.
        """
        snapshot = self.handed_to_an_auditor(tmp_path, shape)

        backend = SQLiteBackend(str(snapshot.path), create=False)
        try:
            assert list(backend.iter_entries()) == snapshot.written
            assert backend.get_head() == snapshot.written[-1]
            assert backend.get_oldest() == snapshot.written[0]
        finally:
            backend.close()


class TestTheReadOnlyFileUriIsBuiltForThePlatform:
    """A ``create=False`` open must name an ABSOLUTE file on every platform.

    The read-only route builds a ``file:`` URI so SQLite opens without creating anything.
    Hand-assembling it as ``"file:" + quote(path)`` is correct on POSIX only: on Windows
    ``quote`` escapes the drive colon and the backslashes, SQLite percent-decodes the
    result to ``C:\\Users\\me\\audit.db``, and because that does not begin with ``/`` it is
    classified as a **relative** filename and resolved against the process working
    directory instead of the path the caller asked for.

    Nobody caught it because this route had never executed on Windows: every CLI test
    there died earlier, in URI parsing, before a backend was ever constructed.

    ``PurePath.as_uri()`` is the platform's own answer and is what SQLite documents -
    ``file:///C:/Users/me/audit.db``. These assert the rule on both flavours, so the
    Windows half is checked from any machine.
    """

    def test_a_windows_path_becomes_a_triple_slash_file_uri(self) -> None:
        from signledger.backends.sqlite import _file_uri

        assert _file_uri(PureWindowsPath(r"C:\Users\me\audit.db")) == (
            "file:///C:/Users/me/audit.db"
        )

    def test_a_posix_path_is_unchanged_in_shape(self) -> None:
        from signledger.backends.sqlite import _file_uri

        assert _file_uri(PurePosixPath("/var/lib/audit.db")) == "file:///var/lib/audit.db"

    def test_a_character_sqlite_would_read_as_its_own_query_is_escaped(self) -> None:
        """``?`` and ``#`` end the filename in a SQLite URI, so they must not survive raw."""
        from signledger.backends.sqlite import _file_uri

        built = _file_uri(PurePosixPath("/var/lib/od?d#b.db"))

        assert "?" not in built and "#" not in built
        assert built.startswith("file:///var/lib/")

    def test_a_read_only_open_reads_the_file_the_caller_named(self, tmp_path: Any) -> None:
        """The functional half, on whatever platform is running this."""
        from signledger.backends.base import build_next_entry
        from signledger.backends.sqlite import SQLiteBackend

        path = tmp_path / "named.db"
        writer = SQLiteBackend(str(path))
        try:
            writer.append_atomic(
                lambda tip: build_next_entry(
                    tip,
                    ledger_id="uri-ledger",
                    created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    data={"event": "one"},
                    timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
                )
            )
        finally:
            writer.close()

        reader = SQLiteBackend(str(path), create=False)
        try:
            assert reader.count() == 1
        finally:
            reader.close()


class TestTheReadOnlyFallbackDoesNotStrandItsFirstConnection:
    """Falling back to ``immutable=1`` must close the connection it is abandoning.

    ``_raw_connect`` opens with ``mode=ro``, then forces SQLite to reach the wal-index. If
    that raises, it opens a second connection with ``immutable=1`` and returns that - and
    the first one was left open, reachable by nothing, until the collector finalised it and
    emitted ``ResourceWarning: unclosed database`` inside whatever test happened to be
    running at the time.

    On CPython 3.11 the collector reclaimed it quietly. On 3.13 and 3.14 it surfaced as an
    unraisable exception and, under ``filterwarnings = ["error"]``, failed a Hypothesis
    tamper-detection test in another file that never opens a SQLite database at all.
    """

    def store(self, tmp_path: Any) -> Any:
        path = tmp_path / "fallback.db"
        backend = SQLiteBackend(str(path))
        try:
            BackendConformanceSuite().append(backend)
        finally:
            backend.close()
        return path

    def test_the_abandoned_connection_is_closed(self, tmp_path: Any, probe: Probe) -> None:
        path = self.store(tmp_path)
        probe.fail_first_readonly_probe = True

        backend = SQLiteBackend(str(path), create=False)
        try:
            assert backend.count() == 1
        finally:
            backend.close()

        assert probe.leaked_connections() == []

    def test_the_fallback_still_returns_a_usable_connection(
        self, tmp_path: Any, probe: Probe
    ) -> None:
        """Closing the first one must not cost us the second."""
        path = self.store(tmp_path)
        probe.fail_first_readonly_probe = True

        backend = SQLiteBackend(str(path), create=False)
        try:
            assert [entry.sequence for entry in backend.iter_entries()] == [0]
        finally:
            backend.close()
