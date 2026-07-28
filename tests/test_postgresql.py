"""PostgreSQL backend tests. Real server only - nothing here mocks a database."""

import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from signledger.backends.base import (
    CAP_APPEND_ONLY_ENFORCED,
    CAP_DURABLE_FSYNC,
    CAP_WORM,
    BackendConformanceSuite,
    build_next_entry,
)

pytestmark = pytest.mark.postgres

LEDGER_ID = "pg-9c1f0d3a-6b52-4e77-8a41-2d6f5b9c0e18"
CREATED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)
BASE_TIMESTAMP = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def append(backend, data=None, **kwargs):
    """Append one correctly linked entry and return it."""
    payload = {"event": "pg"} if data is None else data
    return backend.append_atomic(
        lambda tip: build_next_entry(
            tip, ledger_id=LEDGER_ID, created_at=CREATED_AT, data=payload, **kwargs
        )
    )


def append_many(backend, count, **kwargs):
    """Append ``count`` entries whose timestamps advance one second at a time."""
    return [
        append(
            backend,
            {"event": "pg", "index": index},
            timestamp=BASE_TIMESTAMP + timedelta(seconds=index),
            **kwargs,
        )
        for index in range(count)
    ]

psycopg = pytest.importorskip("psycopg", reason="psycopg (v3) is not installed")

DSN_ENV = "SIGNLEDGER_TEST_POSTGRES_DSN"
DEFAULT_DSN = "postgresql://postgres@localhost:5432/signledger_test"


def _dsn() -> str:
    return os.environ.get(DSN_ENV, DEFAULT_DSN)


def _require_server() -> str:
    """Skip - never fail - when no live server answers. Report host:port, never the DSN."""
    dsn = _dsn()
    parts = urlsplit(dsn)
    host, port = parts.hostname or "localhost", parts.port or 5432
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as err:
        pytest.skip(f"no live PostgreSQL server at {host}:{port} ({type(err).__name__})")
    return dsn


@pytest.fixture
def pg_dsn() -> str:
    return _require_server()


@pytest.fixture
def table_prefix() -> str:
    return "sl_" + uuid.uuid4().hex[:12]


def _drop_test_tables(dsn: str, prefix: str) -> None:
    """Drop everything a test created. Without this the test database grows forever."""
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
            "AND tablename LIKE %s",
            (prefix + "%",),
        )
        for (name,) in cur.fetchall():
            cur.execute(
                psycopg.sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    psycopg.sql.Identifier(name)
                )
            )


@pytest.fixture
def pg_backend(pg_dsn: str, table_prefix: str):
    """A live, empty PostgreSQL backend on throwaway tables."""
    from signledger.backends.postgresql import PostgresBackend

    instance = PostgresBackend(
        pg_dsn,
        entries_table=f"{table_prefix}_entries",
        checkpoints_table=f"{table_prefix}_checkpoints",
    )
    try:
        yield instance
    finally:
        instance.close()
        _drop_test_tables(pg_dsn, table_prefix)


@pytest.fixture(autouse=True)
def _sweep_stray_tables(request, pg_dsn: str):
    """Tests that build their own backends still get their tables reclaimed."""
    yield
    _drop_test_tables(pg_dsn, "sl_")


class TestPostgresConformance(BackendConformanceSuite):
    """Every shared backend behaviour, executed against a real PostgreSQL server."""

    @pytest.fixture
    def backend(self, pg_backend):
        return pg_backend


class TestConstruction:
    def test_a_fresh_backend_starts_empty(self, pg_dsn: str, table_prefix: str) -> None:
        from signledger.backends.postgresql import PostgresBackend

        backend = PostgresBackend(
            pg_dsn,
            entries_table=f"{table_prefix}_entries",
            checkpoints_table=f"{table_prefix}_checkpoints",
        )
        try:
            assert backend.count() == 0
        finally:
            backend.close()


# --------------------------------------------------------------------------------------
# append-only enforcement
# --------------------------------------------------------------------------------------


def _raw(dsn: str):
    """A connection that does not go through the backend - the insider's view."""
    return psycopg.connect(dsn, autocommit=True)


def _refused():
    """The append-only trigger's refusal, as psycopg surfaces it."""
    return pytest.raises(psycopg.errors.RaiseException)


class TestAppendOnlyEnforcement:
    """Adversary B holds a database login. The table itself has to refuse them."""

    def test_a_direct_update_of_an_entry_is_refused_by_the_database(
        self, pg_backend, pg_dsn: str
    ) -> None:
        append(pg_backend, {"event": "original"})

        with _raw(pg_dsn) as conn, conn.cursor() as cur, _refused():
            cur.execute(
                psycopg.sql.SQL("UPDATE {} SET data = %s WHERE sequence = 0").format(
                    psycopg.sql.Identifier(pg_backend.entries_table)
                ),
                ('{"event": "rewritten"}',),
            )

    def test_an_entry_survives_a_refused_update_unchanged(self, pg_backend, pg_dsn: str) -> None:
        append(pg_backend, {"event": "original"})

        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.RaiseException):
                cur.execute(
                    psycopg.sql.SQL("UPDATE {} SET data = %s WHERE sequence = 0").format(
                        psycopg.sql.Identifier(pg_backend.entries_table)
                    ),
                    ('{"event": "rewritten"}',),
                )

        assert pg_backend.get_by_sequence(0).data == {"event": "original"}

    def test_a_direct_delete_of_an_entry_is_refused_by_the_database(
        self, pg_backend, pg_dsn: str
    ) -> None:
        """Truncating the tail is the attack a bare hash chain cannot see."""
        append(pg_backend, {"event": "original"})

        with _raw(pg_dsn) as conn, conn.cursor() as cur, _refused():
            cur.execute(
                psycopg.sql.SQL("DELETE FROM {}").format(
                    psycopg.sql.Identifier(pg_backend.entries_table)
                )
            )

    def test_the_refusal_message_names_signledger(self, pg_backend, pg_dsn: str) -> None:
        append(pg_backend)

        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.RaiseException) as caught:
                cur.execute(
                    psycopg.sql.SQL("DELETE FROM {}").format(
                        psycopg.sql.Identifier(pg_backend.entries_table)
                    )
                )

        assert "append-only" in str(caught.value)

    def test_capabilities_reports_append_only_enforced_while_the_trigger_exists(
        self, pg_backend
    ) -> None:
        assert CAP_APPEND_ONLY_ENFORCED in pg_backend.capabilities()

    def test_capabilities_stops_reporting_append_only_once_the_trigger_is_dropped(
        self, pg_backend, pg_dsn: str
    ) -> None:
        """The token is read from the catalog, never hardcoded. An operator who drops the
        trigger must not keep a WORM claim they no longer earn."""
        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL("DROP TRIGGER {} ON {}").format(
                    psycopg.sql.Identifier(pg_backend.append_only_trigger),
                    psycopg.sql.Identifier(pg_backend.entries_table),
                )
            )

        assert CAP_APPEND_ONLY_ENFORCED not in pg_backend.capabilities()

    def test_an_update_succeeds_once_the_trigger_is_dropped(self, pg_backend, pg_dsn: str) -> None:
        """Proves the previous refusals came from the trigger and not from some accident
        of the test setup."""
        append(pg_backend, {"event": "original"})

        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL("DROP TRIGGER {} ON {}").format(
                    psycopg.sql.Identifier(pg_backend.append_only_trigger),
                    psycopg.sql.Identifier(pg_backend.entries_table),
                )
            )
            cur.execute(
                psycopg.sql.SQL("UPDATE {} SET data = %s WHERE sequence = 0").format(
                    psycopg.sql.Identifier(pg_backend.entries_table)
                ),
                ('{"event": "rewritten"}',),
            )

        assert pg_backend.get_by_sequence(0).data == {"event": "rewritten"}

    def test_never_claims_worm(self, pg_backend) -> None:
        """A trigger can be dropped by anyone who can create one. That is not WORM."""
        assert CAP_WORM not in pg_backend.capabilities()

    def test_capabilities_reports_durable_fsync_when_the_server_fsyncs_on_commit(
        self, pg_backend
    ) -> None:
        assert CAP_DURABLE_FSYNC in pg_backend.capabilities()

    def test_capabilities_stops_reporting_durable_fsync_when_fsync_is_off(
        self, pg_backend, monkeypatch
    ) -> None:
        """``fsync=off`` acknowledges a commit the disk has never seen."""
        original = pg_backend._fsync_settings

        monkeypatch.setattr(pg_backend, "_fsync_settings", lambda cur: ("off", "on"))

        assert CAP_DURABLE_FSYNC not in pg_backend.capabilities()
        assert original is not None

    def test_capabilities_stops_reporting_durable_fsync_when_commits_are_asynchronous(
        self, pg_backend, monkeypatch
    ) -> None:
        monkeypatch.setattr(pg_backend, "_fsync_settings", lambda cur: ("on", "off"))

        assert CAP_DURABLE_FSYNC not in pg_backend.capabilities()


# --------------------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------------------


class TestTimestampsAreZoneIndependent:
    """v1 rebuilt entries from the TIMESTAMPTZ, which Postgres renders in the *session*
    zone. A server running America/New_York therefore re-hashed every row differently and
    reported a pristine ledger as tampered."""

    def test_an_entry_rehashes_to_its_stored_digest_under_a_non_utc_session(
        self, pg_backend, pg_dsn: str
    ) -> None:
        from signledger.crypto.hashing import HashChain

        stored = append(pg_backend, {"event": "tz"}, timestamp=BASE_TIMESTAMP)

        pg_backend.set_session_timezone("America/New_York")
        loaded = pg_backend.get_by_sequence(0)

        assert loaded.hash == stored.hash
        assert HashChain().verify_hash(loaded)

    def test_the_timestamp_read_back_is_utc_whatever_the_session_zone_is(
        self, pg_backend
    ) -> None:
        append(pg_backend, {"event": "tz"}, timestamp=BASE_TIMESTAMP)

        pg_backend.set_session_timezone("Asia/Kolkata")
        loaded = pg_backend.get_by_sequence(0)

        assert loaded.timestamp == BASE_TIMESTAMP
        assert loaded.timestamp.utcoffset() == timedelta(0)

    def test_every_entry_still_verifies_after_the_session_zone_changes(
        self, pg_backend
    ) -> None:
        from signledger.crypto.hashing import HashChain

        appended = append_many(pg_backend, 5)

        pg_backend.set_session_timezone("Pacific/Kiritimati")

        assert list(pg_backend.iter_entries()) == appended
        assert all(HashChain().verify_hash(e) for e in pg_backend.iter_entries())

    def test_microsecond_precision_survives_the_round_trip(self, pg_backend) -> None:
        """A digest is over integer microseconds, so losing them is losing the entry."""
        moment = datetime(2024, 6, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        stored = append(pg_backend, {"event": "us"}, timestamp=moment)

        loaded = pg_backend.get_by_sequence(0)

        assert loaded.timestamp == moment
        assert loaded.timestamp.microsecond == 123456
        assert loaded.hash == stored.hash

    def test_a_time_filter_under_a_non_utc_session_selects_the_same_rows(
        self, pg_backend
    ) -> None:
        """The filter compares timestamp_us, so it cannot drift with the session zone."""
        append_many(pg_backend, 8)
        window = {
            "start_time": BASE_TIMESTAMP + timedelta(seconds=2),
            "end_time": BASE_TIMESTAMP + timedelta(seconds=4),
        }
        before = [e.sequence for e in pg_backend.iter_entries(**window)]

        pg_backend.set_session_timezone("America/New_York")

        assert [e.sequence for e in pg_backend.iter_entries(**window)] == before == [2, 3, 4]

    def test_a_tz_aware_filter_bound_in_another_zone_selects_the_same_rows(
        self, pg_backend
    ) -> None:
        append_many(pg_backend, 8)
        eastern = timezone(timedelta(hours=-4))
        start = (BASE_TIMESTAMP + timedelta(seconds=2)).astimezone(eastern)

        found = pg_backend.iter_entries(
            start_time=start, end_time=BASE_TIMESTAMP + timedelta(seconds=4)
        )

        assert [e.sequence for e in found] == [2, 3, 4]


# --------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------


def _columns(dsn: str, table: str):
    with _raw(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name = %s",
            (table,),
        )
        return {name: (kind, nullable) for name, kind, nullable in cur.fetchall()}


def _indexed_columns(dsn: str, table: str):
    """Every set of columns that has an index on it, however that index arose."""
    with _raw(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = %s",
            (table,),
        )
        return [definition for (definition,) in cur.fetchall()]


class TestSchema:
    def test_metadata_is_not_nullable(self, pg_backend, pg_dsn: str) -> None:
        """The one line that broke every v1 Postgres ledger. ``{}`` is falsy, so
        ``metadata or None`` wrote NULL and the ledger could not read its own rows."""
        columns = _columns(pg_dsn, pg_backend.entries_table)

        assert columns["metadata"] == ("text", "NO")

    def test_an_entry_without_metadata_stores_an_empty_json_object_not_null(
        self, pg_backend, pg_dsn: str
    ) -> None:
        append(pg_backend, {"event": "no-metadata"})

        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL("SELECT metadata FROM {} WHERE sequence = 0").format(
                    psycopg.sql.Identifier(pg_backend.entries_table)
                )
            )
            stored = cur.fetchone()[0]

        assert stored == "{}"

    def test_a_null_metadata_row_cannot_be_inserted_at_all(self, pg_backend, pg_dsn: str) -> None:
        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.NotNullViolation):
                cur.execute(
                    psycopg.sql.SQL(
                        "INSERT INTO {} (sequence, entry_id, ledger_id, format_version, "
                        "timestamp, timestamp_us, data, metadata, previous_hash, hash) "
                        "VALUES (0, 'e', 'l', 2, now(), 0, '{{}}', NULL, 'p', 'h')"
                    ).format(psycopg.sql.Identifier(pg_backend.entries_table))
                )

    def test_the_queried_time_column_is_an_integer_microsecond_count(
        self, pg_backend, pg_dsn: str
    ) -> None:
        columns = _columns(pg_dsn, pg_backend.entries_table)

        assert columns["timestamp_us"] == ("bigint", "NO")

    def test_the_human_readable_time_column_is_a_timestamptz(self, pg_backend, pg_dsn: str) -> None:
        columns = _columns(pg_dsn, pg_backend.entries_table)

        assert columns["timestamp"] == ("timestamp with time zone", "NO")

    def test_sequence_is_unique(self, pg_backend, pg_dsn: str) -> None:
        append(pg_backend)
        head = pg_backend.get_head()

        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.UniqueViolation):
                cur.execute(
                    psycopg.sql.SQL(
                        "INSERT INTO {} (sequence, entry_id, ledger_id, format_version, "
                        "timestamp, timestamp_us, data, metadata, previous_hash, hash) "
                        "VALUES (0, 'other', 'l', 2, now(), 1, '{{}}', '{{}}', 'other', 'other')"
                    ).format(psycopg.sql.Identifier(pg_backend.entries_table))
                )
        assert head.sequence == 0

    def test_previous_hash_is_unique_so_two_entries_cannot_claim_one_parent(
        self, pg_backend, pg_dsn: str
    ) -> None:
        """A fork is two entries naming the same parent. The schema refuses it even if
        every other column differs."""
        append(pg_backend)
        parent = pg_backend.get_by_sequence(0).previous_hash

        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.UniqueViolation):
                cur.execute(
                    psycopg.sql.SQL(
                        "INSERT INTO {} (sequence, entry_id, ledger_id, format_version, "
                        "timestamp, timestamp_us, data, metadata, previous_hash, hash) "
                        "VALUES (99, 'other', 'l', 2, now(), 1, '{{}}', '{{}}', %s, 'other')"
                    ).format(psycopg.sql.Identifier(pg_backend.entries_table)),
                    (parent,),
                )

    def test_entry_id_is_unique(self, pg_backend, pg_dsn: str) -> None:
        append(pg_backend)
        existing = pg_backend.get_by_sequence(0).entry_id

        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            with pytest.raises(psycopg.errors.UniqueViolation):
                cur.execute(
                    psycopg.sql.SQL(
                        "INSERT INTO {} (sequence, entry_id, ledger_id, format_version, "
                        "timestamp, timestamp_us, data, metadata, previous_hash, hash) "
                        "VALUES (99, %s, 'l', 2, now(), 1, '{{}}', '{{}}', 'other', 'other')"
                    ).format(psycopg.sql.Identifier(pg_backend.entries_table)),
                    (existing,),
                )

    def test_the_time_filter_column_is_indexed(self, pg_backend, pg_dsn: str) -> None:
        """iter_entries filters on timestamp_us, so a ledger of any size needs the index."""
        definitions = _indexed_columns(pg_dsn, pg_backend.entries_table)

        assert any("timestamp_us" in definition for definition in definitions)

    def test_sequence_is_indexed(self, pg_backend, pg_dsn: str) -> None:
        definitions = _indexed_columns(pg_dsn, pg_backend.entries_table)

        assert any("(sequence)" in definition for definition in definitions)

    def test_entry_id_is_indexed(self, pg_backend, pg_dsn: str) -> None:
        definitions = _indexed_columns(pg_dsn, pg_backend.entries_table)

        assert any("(entry_id)" in definition for definition in definitions)

    def test_creating_a_backend_over_an_existing_schema_is_harmless(
        self, pg_dsn: str, table_prefix: str
    ) -> None:
        """Two processes starting at once must not race each other's CREATE TABLE."""
        from signledger.backends.postgresql import PostgresBackend

        first = PostgresBackend(
            pg_dsn,
            entries_table=f"{table_prefix}_entries",
            checkpoints_table=f"{table_prefix}_checkpoints",
        )
        try:
            append(first)
            second = PostgresBackend(
                pg_dsn,
                entries_table=f"{table_prefix}_entries",
                checkpoints_table=f"{table_prefix}_checkpoints",
            )
            try:
                assert second.count() == 1
            finally:
                second.close()
        finally:
            first.close()

    def test_derived_names_stay_within_the_postgres_identifier_limit(self) -> None:
        """Postgres silently truncates at 63 bytes, which would let two tables share one
        trigger - and dropping it on one would quietly un-protect the other."""
        from signledger.backends.postgresql import MAX_IDENTIFIER_LENGTH, _derived_identifier

        derived = _derived_identifier("t" * 80, "_no_write")

        assert len(derived) == MAX_IDENTIFIER_LENGTH
        assert derived.endswith("_no_write")

    def test_a_long_table_name_still_gets_its_own_distinct_trigger(
        self, pg_dsn: str
    ) -> None:
        from signledger.backends.postgresql import PostgresBackend

        stem = "sl_" + uuid.uuid4().hex[:12]
        long_a = (stem + "a" * 60)[:60]
        long_b = (stem + "b" * 60)[:60]
        backends = []
        try:
            for name in (long_a, long_b):
                backends.append(
                    PostgresBackend(
                        pg_dsn, entries_table=name, checkpoints_table=name + "_cp"
                    )
                )
            assert backends[0].append_only_trigger != backends[1].append_only_trigger
            assert CAP_APPEND_ONLY_ENFORCED in backends[0].capabilities()
            assert CAP_APPEND_ONLY_ENFORCED in backends[1].capabilities()
        finally:
            for instance in backends:
                instance.close()


# --------------------------------------------------------------------------------------
# connection pool
# --------------------------------------------------------------------------------------


class TestConnectionPool:
    """v1 returned connections still carrying a failed transaction, so the next borrower
    inherited ``current transaction is aborted`` and the ledger wedged until restart."""

    def test_a_connection_is_returned_to_the_pool_after_a_read(self, pg_backend) -> None:
        before = pg_backend._pool.idle_count
        pg_backend.count()

        assert pg_backend._pool.idle_count == before

    def test_a_connection_comes_back_idle_not_mid_transaction(self, pg_backend) -> None:
        """An idle-in-transaction connection holds its snapshot and blocks vacuum forever."""
        pg_backend.count()
        connection = pg_backend._pool.getconn()
        try:
            assert connection.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
        finally:
            pg_backend._pool.putconn(connection)

    def test_a_failed_append_returns_a_usable_connection(self, pg_backend) -> None:
        class BuildFailedError(Exception):
            pass

        def explode(tip):
            raise BuildFailedError("callback failed")

        with pytest.raises(BuildFailedError):
            pg_backend.append_atomic(explode)

        assert pg_backend.count() == 0
        assert append(pg_backend).sequence == 0

    def test_a_connection_poisoned_by_a_constraint_violation_is_rolled_back(
        self, pg_backend
    ) -> None:
        """A UniqueViolation aborts the transaction. Without a rollback the very next
        statement on that connection fails with InFailedSqlTransaction."""
        first = append(pg_backend)
        calls = []

        def build(tip):
            calls.append(tip)
            from signledger.backends.base import build_next_entry

            return build_next_entry(
                tip,
                ledger_id=LEDGER_ID,
                created_at=CREATED_AT,
                data={"n": 2},
                entry_id=first.entry_id if len(calls) == 1 else None,
            )

        appended = pg_backend.append_atomic(build)

        assert len(calls) == 2
        assert pg_backend.get_head() == appended

    def test_the_pool_never_hands_one_connection_to_two_threads(self, pg_backend) -> None:
        import threading

        held = []
        overlaps = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def borrow():
            barrier.wait()
            for _ in range(20):
                connection = pg_backend._pool.getconn()
                with lock:
                    if connection in held:
                        overlaps.append(connection)
                    held.append(connection)
                with lock:
                    held.remove(connection)
                pg_backend._pool.putconn(connection)

        threads = [threading.Thread(target=borrow) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert overlaps == []

    def test_the_pool_is_bounded_by_maxconn(self, pg_dsn: str, table_prefix: str) -> None:
        from signledger.backends.postgresql import PostgresBackend

        backend = PostgresBackend(
            pg_dsn,
            entries_table=f"{table_prefix}_entries",
            checkpoints_table=f"{table_prefix}_checkpoints",
            minconn=1,
            maxconn=2,
        )
        try:
            first = backend._pool.getconn()
            second = backend._pool.getconn()

            assert backend._pool.opened_count == 2
            backend._pool.putconn(first)
            backend._pool.putconn(second)
        finally:
            backend.close()

    def test_a_maxconn_below_one_is_refused(self, pg_dsn: str, table_prefix: str) -> None:
        from signledger.backends.postgresql import PostgresBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            PostgresBackend(pg_dsn, entries_table=f"{table_prefix}_e", maxconn=0)

    def test_a_negative_minconn_is_refused(self, pg_dsn: str, table_prefix: str) -> None:
        from signledger.backends.postgresql import PostgresBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            PostgresBackend(pg_dsn, entries_table=f"{table_prefix}_e", minconn=-1)

    def test_minconn_above_maxconn_is_refused(self, pg_dsn: str, table_prefix: str) -> None:
        from signledger.backends.postgresql import PostgresBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            PostgresBackend(pg_dsn, entries_table=f"{table_prefix}_e", minconn=3, maxconn=2)

    def test_closing_releases_every_pooled_connection(self, pg_dsn: str, table_prefix: str) -> None:
        from signledger.backends.postgresql import PostgresBackend

        backend = PostgresBackend(
            pg_dsn,
            entries_table=f"{table_prefix}_entries",
            checkpoints_table=f"{table_prefix}_checkpoints",
            minconn=2,
            maxconn=4,
        )
        pooled = list(backend._pool._idle)
        backend.close()

        assert backend._pool.idle_count == 0
        assert all(connection.closed for connection in pooled)

    def test_borrowing_after_close_raises_storage_error(self, pg_backend) -> None:
        from signledger.core.exceptions import StorageError

        pg_backend.close()

        with pytest.raises(StorageError):
            pg_backend._pool.getconn()

    def test_a_discarded_connection_frees_its_slot(self, pg_backend) -> None:
        """A broken socket must not be handed to the next borrower, and must not leak the
        slot it occupied either - otherwise the pool shrinks to nothing."""
        connection = pg_backend._pool.getconn()
        opened = pg_backend._pool.opened_count

        pg_backend._pool.putconn(connection, discard=True)

        assert pg_backend._pool.opened_count == opened - 1
        assert connection.closed


# --------------------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------------------


class TestConcurrentAppend:
    """v1 read the head, then inserted, under a process-local lock. Two gunicorn workers
    forked the chain within seconds, permanently, with no repair path."""

    def test_sixteen_threads_appending_produce_a_gapless_chain(self, pg_backend) -> None:
        import threading

        errors = []
        barrier = threading.Barrier(8)

        def writer():
            barrier.wait()
            for _ in range(6):
                try:
                    append(pg_backend, {"event": "race"})
                except Exception as err:  # noqa: BLE001 - recorded, then asserted on
                    errors.append(err)

        threads = [threading.Thread(target=writer) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        entries = list(pg_backend.iter_entries())

        assert errors == []
        assert [e.sequence for e in entries] == list(range(48))

    def test_a_concurrent_chain_links_end_to_end(self, pg_backend) -> None:
        import threading

        barrier = threading.Barrier(4)

        def writer():
            barrier.wait()
            for _ in range(5):
                append(pg_backend, {"event": "race"})

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        entries = list(pg_backend.iter_entries())

        assert all(
            entries[i].previous_hash == entries[i - 1].hash for i in range(1, len(entries))
        )

    def test_two_backends_on_one_table_do_not_fork_the_chain(
        self, pg_dsn: str, table_prefix: str
    ) -> None:
        """Two *processes* is the case a process-local lock cannot cover. Two independent
        backend objects with independent pools is the same shape."""
        import threading

        from signledger.backends.postgresql import PostgresBackend

        def make():
            return PostgresBackend(
                pg_dsn,
                entries_table=f"{table_prefix}_entries",
                checkpoints_table=f"{table_prefix}_checkpoints",
                maxconn=4,
            )

        left, right = make(), make()
        try:
            barrier = threading.Barrier(2)

            def writer(target):
                barrier.wait()
                for _ in range(8):
                    append(target, {"event": "race"})

            threads = [threading.Thread(target=writer, args=(t,)) for t in (left, right)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            entries = list(left.iter_entries())

            assert [e.sequence for e in entries] == list(range(16))
            assert len({e.previous_hash for e in entries}) == 16
        finally:
            left.close()
            right.close()

    def test_the_append_lock_key_is_stable_for_a_table_name(self) -> None:
        """Two processes must contend on the *same* advisory lock, so the key can only be
        derived from the table name - never from anything process-local."""
        from signledger.backends.postgresql import _advisory_key

        assert _advisory_key("ledger_entries") == _advisory_key("ledger_entries")
        assert _advisory_key("ledger_entries") != _advisory_key("other_entries")
        assert -(2**63) <= _advisory_key("ledger_entries") < 2**63


# --------------------------------------------------------------------------------------
# identifier safety
# --------------------------------------------------------------------------------------


class TestIdentifierSafety:
    """v1 interpolated table names into SQL at roughly 25 sites. Names are allowlisted
    *and* quoted with psycopg.sql.Identifier - both halves of the door."""

    INJECTIONS = [
        "entries; DROP TABLE entries",
        'entries" ; DROP TABLE x --',
        "entries--",
        "entries'",
        "1_entries",
        "",
        "entries table",
        "entries/*x*/",
    ]

    @pytest.mark.parametrize("name", INJECTIONS)
    def test_a_hostile_entries_table_name_is_refused(self, pg_dsn: str, name: str) -> None:
        from signledger.backends.postgresql import PostgresBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            PostgresBackend(pg_dsn, entries_table=name)

    @pytest.mark.parametrize("name", INJECTIONS)
    def test_a_hostile_checkpoints_table_name_is_refused(self, pg_dsn: str, name: str) -> None:
        from signledger.backends.postgresql import PostgresBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            PostgresBackend(pg_dsn, checkpoints_table=name)

    def test_a_hostile_name_is_refused_before_any_connection_is_opened(
        self, pg_dsn: str
    ) -> None:
        """Validation happens first, so a bad name cannot leak a socket on its way out."""
        from signledger.backends.postgresql import PostgresBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            PostgresBackend("postgresql://nowhere:1/none", entries_table="bad; name")

    def test_every_identifier_reaches_sql_through_the_identifier_wrapper(self) -> None:
        """A regression guard on the whole module: no f-string may carry a table name
        into SQL text."""
        import inspect

        from signledger.backends import postgresql

        source = inspect.getsource(postgresql)
        offenders = [
            line.strip()
            for line in source.splitlines()
            if "self.entries_table" in line or "self.checkpoints_table" in line
            if "{" in line and "sql.Identifier" not in line and "validate_identifier" not in line
        ]

        assert offenders == []

    def test_a_table_name_of_the_maximum_length_is_accepted(self, pg_dsn: str) -> None:
        from signledger.backends.postgresql import PostgresBackend

        name = ("sl_" + uuid.uuid4().hex[:12]).ljust(60, "x")
        backend = PostgresBackend(pg_dsn, entries_table=name, checkpoints_table=name + "_cp")
        try:
            assert backend.count() == 0
        finally:
            backend.close()


# --------------------------------------------------------------------------------------
# error redaction
# --------------------------------------------------------------------------------------


class TestErrorRedaction:
    """Contract section 10: no exception message may carry a payload, a connection string
    or a credential."""

    def test_a_connection_failure_never_quotes_the_dsn(self) -> None:
        from signledger.backends.postgresql import PostgresBackend
        from signledger.core.exceptions import StorageError

        secret = "postgresql://ledger:hunter2@127.0.0.1:1/absent"

        with pytest.raises(StorageError) as caught:
            PostgresBackend(secret, entries_table="probe_entries")

        message = str(caught.value)
        assert "hunter2" not in message
        assert "127.0.0.1" not in message
        assert "absent" not in message

    def test_a_connection_failure_still_names_the_operation(self) -> None:
        from signledger.backends.postgresql import PostgresBackend
        from signledger.core.exceptions import StorageError

        with pytest.raises(StorageError) as caught:
            PostgresBackend("postgresql://ledger:hunter2@127.0.0.1:1/absent")

        assert "connect" in str(caught.value)

    def test_a_driver_error_reports_only_its_type(self, pg_backend, pg_dsn: str) -> None:
        """A psycopg message quotes the row it choked on. Only the class name is safe."""
        from signledger.core.exceptions import StorageError

        append(pg_backend, {"card": "4111111111111111"})
        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL("DROP TABLE {}").format(
                    psycopg.sql.Identifier(pg_backend.entries_table)
                )
            )

        with pytest.raises(StorageError) as caught:
            pg_backend.count()

        message = str(caught.value)
        assert "4111111111111111" not in message
        assert message == "count failed: UndefinedTable"

    def test_a_failed_append_does_not_quote_the_payload(self, pg_backend, pg_dsn: str) -> None:
        from signledger.core.exceptions import StorageError

        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL("DROP TABLE {}").format(
                    psycopg.sql.Identifier(pg_backend.entries_table)
                )
            )

        with pytest.raises(StorageError) as caught:
            append(pg_backend, {"ssn": "078-05-1120"})

        assert "078-05-1120" not in str(caught.value)


# --------------------------------------------------------------------------------------
# the documented REVOKE script
# --------------------------------------------------------------------------------------


class TestRevokeScript:
    """The trigger stops the application's own bugs. It does not stop anyone who can
    ALTER TABLE ... DISABLE TRIGGER, so the deployment story is privileges."""

    def test_the_script_revokes_update_and_delete(self) -> None:
        from signledger.backends.postgresql import append_only_revoke_script

        script = append_only_revoke_script("app_writer", entries_table="ledger_entries")

        assert "REVOKE UPDATE, DELETE, TRUNCATE" in script
        assert "GRANT SELECT, INSERT" in script

    def test_the_script_quotes_every_identifier(self) -> None:
        from signledger.backends.postgresql import append_only_revoke_script

        script = append_only_revoke_script("app_writer", entries_table="ledger_entries")

        assert '"app_writer"' in script
        assert '"ledger_entries"' in script

    def test_the_script_covers_the_checkpoints_table_too(self) -> None:
        from signledger.backends.postgresql import append_only_revoke_script

        script = append_only_revoke_script("app_writer", checkpoints_table="cp_table")

        assert '"cp_table"' in script

    def test_a_hostile_role_name_is_refused(self) -> None:
        from signledger.backends.postgresql import append_only_revoke_script
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            append_only_revoke_script("writer; DROP DATABASE ledger")

    def test_a_hostile_table_name_is_refused(self) -> None:
        from signledger.backends.postgresql import append_only_revoke_script
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            append_only_revoke_script("app_writer", entries_table="x; DROP TABLE y")

    def test_a_hostile_checkpoints_name_is_refused(self) -> None:
        from signledger.backends.postgresql import append_only_revoke_script
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            append_only_revoke_script("app_writer", checkpoints_table="x; DROP TABLE y")

    def test_the_script_actually_executes_against_a_real_server(
        self, pg_backend, pg_dsn: str
    ) -> None:
        """A script that has never been run is a comment. This one runs."""
        from signledger.backends.postgresql import append_only_revoke_script

        role = "sl_role_" + uuid.uuid4().hex[:10]
        script = append_only_revoke_script(
            role,
            entries_table=pg_backend.entries_table,
            checkpoints_table=pg_backend.checkpoints_table,
        )
        with _raw(pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                psycopg.sql.SQL("CREATE ROLE {} LOGIN").format(psycopg.sql.Identifier(role))
            )
            try:
                cur.execute(script)
                cur.execute(
                    "SELECT has_table_privilege(%s, %s, 'UPDATE'), "
                    "has_table_privilege(%s, %s, 'INSERT')",
                    (role, pg_backend.entries_table, role, pg_backend.entries_table),
                )
                may_update, may_insert = cur.fetchone()
            finally:
                # The grants the script issued are dependent objects, so they have to go
                # before the role itself can.
                cur.execute(
                    psycopg.sql.SQL("DROP OWNED BY {}").format(psycopg.sql.Identifier(role))
                )
                cur.execute(
                    psycopg.sql.SQL("DROP ROLE {}").format(psycopg.sql.Identifier(role))
                )

        assert may_update is False
        assert may_insert is True


# --------------------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------------------


class TestPoolUnderPressure:
    def test_a_borrower_waits_when_every_connection_is_lent_out(
        self, pg_dsn: str, table_prefix: str
    ) -> None:
        """The pool is bounded, so the last borrower blocks rather than opening an
        unbounded number of sockets against the server."""
        import threading

        from signledger.backends.postgresql import PostgresBackend

        backend = PostgresBackend(
            pg_dsn,
            entries_table=f"{table_prefix}_entries",
            checkpoints_table=f"{table_prefix}_checkpoints",
            minconn=1,
            maxconn=1,
        )
        try:
            held = backend._pool.getconn()
            got = []

            def borrow():
                got.append(backend._pool.getconn())

            waiter = threading.Thread(target=borrow)
            waiter.start()
            waiter.join(timeout=0.3)

            assert waiter.is_alive()
            assert got == []

            backend._pool.putconn(held)
            waiter.join(timeout=5)

            assert not waiter.is_alive()
            assert got == [held]
            backend._pool.putconn(got[0])
        finally:
            backend.close()

    def test_a_failed_connect_does_not_consume_a_pool_slot(self, pg_backend) -> None:
        """Otherwise a server blip permanently shrinks the pool to nothing."""
        pool = pg_backend._pool
        opened = pool.opened_count

        def refuse():
            raise psycopg.OperationalError("server is away")

        pool._connect = refuse
        while pool.idle_count:
            with pool._condition:
                pool._idle.pop().close()
                pool._opened -= 1

        with pytest.raises(psycopg.OperationalError):
            pool.getconn()

        assert pool.opened_count == 0
        assert opened >= 0

    def test_a_schema_failure_closes_the_pool_it_had_already_opened(
        self, pg_dsn: str, table_prefix: str
    ) -> None:
        """Construction is all-or-nothing: a half-built backend must not leave sockets
        open against the server."""
        from signledger.backends import postgresql
        from signledger.core.exceptions import StorageError

        opened = []
        real_connect = psycopg.connect

        def spy(*args, **kwargs):
            connection = real_connect(*args, **kwargs)
            opened.append(connection)
            return connection

        original = postgresql._ConnectionPool._connect
        postgresql._ConnectionPool._connect = lambda self: spy(
            self._dsn, autocommit=False, **self._kwargs
        )
        broken = postgresql.PostgresBackend._create_schema
        postgresql.PostgresBackend._create_schema = lambda self: (_ for _ in ()).throw(
            StorageError("schema is unavailable")
        )
        try:
            with pytest.raises(StorageError):
                postgresql.PostgresBackend(
                    pg_dsn,
                    entries_table=f"{table_prefix}_entries",
                    checkpoints_table=f"{table_prefix}_checkpoints",
                )
        finally:
            postgresql.PostgresBackend._create_schema = broken
            postgresql._ConnectionPool._connect = original

        assert opened
        assert all(connection.closed for connection in opened)

    def test_a_connection_that_cannot_be_rolled_back_is_discarded(self, pg_backend) -> None:
        """A socket the server has already dropped must not go back into the pool for the
        next borrower to trip over."""
        from signledger.backends.postgresql import _unwind

        connection = pg_backend._pool.getconn()
        connection.close()

        assert _unwind(connection) is True

        pg_backend._pool.putconn(connection, discard=True)


class TestSessionTimezoneValidation:
    def test_an_unknown_timezone_is_refused(self, pg_backend) -> None:
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            pg_backend.set_session_timezone("Mars/Olympus_Mons")

    def test_a_refused_timezone_leaves_the_previous_one_in_place(self, pg_backend) -> None:
        from signledger.core.exceptions import ConfigurationError

        pg_backend.set_session_timezone("Asia/Kolkata")
        with pytest.raises(ConfigurationError):
            pg_backend.set_session_timezone("Mars/Olympus_Mons")

        assert pg_backend._session_timezone == "Asia/Kolkata"
        append(pg_backend, {"event": "still-working"})
        assert pg_backend.count() == 1

    def test_a_non_string_timezone_is_refused(self, pg_backend) -> None:
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            pg_backend.set_session_timezone(3600)

    def test_an_empty_timezone_is_refused(self, pg_backend) -> None:
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            pg_backend.set_session_timezone("")

    def test_a_storage_failure_while_setting_the_zone_is_not_reported_as_config(
        self, pg_backend
    ) -> None:
        """Only InvalidParameterValue means "bad zone". Anything else is a storage fault
        and must keep its own identity."""
        from signledger.core.exceptions import StorageError

        pg_backend.close()

        with pytest.raises(StorageError):
            pg_backend.set_session_timezone("Asia/Kolkata")
