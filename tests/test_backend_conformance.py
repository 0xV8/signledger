"""Contract tests for every storage backend (architecture contract, section 4).

``TestBackendConformance`` runs the whole of :class:`BackendConformanceSuite` against the
parametrized ``backend`` fixture, so the identical assertions execute against memory,
SQLite, PostgreSQL and MongoDB. Backends that are not provably interchangeable are not
backends, they are four different libraries wearing one name.

The classes after it cover what cannot be shared: the SQL text SQLite emits, the triggers
that make its rows append-only, its connection registry, and the small helpers every
backend is built from.
"""

import hashlib
import os
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from signledger.backends.base import (
    APPEND_RETRY_BASE_DELAY,
    CAP_APPEND_ONLY_ENFORCED,
    CAP_DURABLE_FSYNC,
    MAX_APPEND_ATTEMPTS,
    BackendConformanceSuite,
    ConformanceTestMixin,
    StorageBackend,
    build_next_entry,
    check_link,
    checkpoint_from_row,
    checkpoint_to_row,
    deserialize_payload,
    entry_from_row,
    entry_to_row,
    micros_to_timestamp,
    retry_on_conflict,
    serialize_payload,
    timestamp_to_micros,
    validate_identifier,
)
from signledger.backends.memory import MemoryBackend
from signledger.backends.sqlite import SQLiteBackend, build_entry_query
from signledger.core.entry import Entry
from signledger.core.exceptions import ConfigurationError, StorageError, ValidationError

UTC = timezone.utc


class TestBackendConformance(BackendConformanceSuite):
    """Every backend, every shared behaviour. The ``backend`` fixture is parametrized."""


# --------------------------------------------------------------------------------------
# identifier safety
# --------------------------------------------------------------------------------------


class TestIdentifierSafety:
    """Table names reach DDL by interpolation, so they are allowlisted, never escaped."""

    INJECTIONS = [
        "entries; DROP TABLE ledger_entries",
        "entries--",
        'entries" OR "1"="1',
        "entries' OR '1'='1",
        "ledger entries",
        "ledger.entries",
        "1entries",
        "",
        "entries)",
        "entries\nDROP TABLE x",
        "entries/*comment*/",
        "ent`ries",
    ]

    @pytest.mark.parametrize("name", INJECTIONS)
    def test_an_entries_table_name_that_is_not_a_bare_identifier_is_refused(self, tmp_path, name):
        with pytest.raises(ConfigurationError):
            SQLiteBackend(str(tmp_path / "x.db"), entries_table=name)

    @pytest.mark.parametrize("name", INJECTIONS)
    def test_a_checkpoints_table_name_that_is_not_a_bare_identifier_is_refused(
        self, tmp_path, name
    ):
        with pytest.raises(ConfigurationError):
            SQLiteBackend(str(tmp_path / "x.db"), checkpoints_table=name)

    def test_a_rejected_table_name_never_reaches_the_database(self, tmp_path):
        path = tmp_path / "untouched.db"

        with pytest.raises(ConfigurationError):
            SQLiteBackend(str(path), entries_table="e; DROP TABLE x")

        assert not path.exists()

    def test_a_valid_custom_table_name_is_accepted(self, tmp_path):
        backend = SQLiteBackend(
            str(tmp_path / "custom.db"), entries_table="Audit_Log_2", checkpoints_table="_cps"
        )
        try:
            assert backend.entries_table == "Audit_Log_2"
            assert backend.count() == 0
        finally:
            backend.close()

    def test_validate_identifier_returns_the_name_it_accepted(self):
        assert validate_identifier("ledger_entries") == "ledger_entries"

    def test_validate_identifier_rejects_a_non_string(self):
        with pytest.raises(ConfigurationError):
            validate_identifier(7)  # type: ignore[arg-type]

    def test_validate_identifier_rejects_an_over_long_name(self):
        with pytest.raises(ConfigurationError):
            validate_identifier("a" * 64)

    def test_validate_identifier_accepts_the_longest_allowed_name(self):
        assert validate_identifier("a" * 63) == "a" * 63

    def test_the_rejection_message_never_repeats_the_offending_name(self):
        secret = "tbl_with_a_secret_looking_name"
        with pytest.raises(ConfigurationError) as caught:
            validate_identifier(secret + "; DROP")

        assert secret not in str(caught.value)


# --------------------------------------------------------------------------------------
# append-only enforcement
# --------------------------------------------------------------------------------------


class TestSQLiteAppendOnlyEnforcement:
    """Adversary B edits a row directly. The database itself must refuse."""

    @pytest.fixture
    def populated(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "worm.db"))
        suite = BackendConformanceSuite()
        suite.append_many(backend, 3)
        try:
            yield backend
        finally:
            backend.close()

    def test_a_direct_update_of_an_entry_row_is_aborted(self, populated):
        raw = sqlite3.connect(populated.path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(
                    f"UPDATE {populated.entries_table} SET data = '{{}}' WHERE sequence = 1"
                )
        finally:
            raw.close()

    def test_a_direct_delete_of_an_entry_row_is_aborted(self, populated):
        raw = sqlite3.connect(populated.path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(f"DELETE FROM {populated.entries_table} WHERE sequence = 1")
        finally:
            raw.close()

    def test_a_refused_update_leaves_the_entry_byte_identical(self, populated):
        before = populated.get_by_sequence(1)
        raw = sqlite3.connect(populated.path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(f"UPDATE {populated.entries_table} SET hash = 'x'")
        finally:
            raw.close()

        assert populated.get_by_sequence(1) == before

    def test_a_refused_delete_leaves_the_count_unchanged(self, populated):
        raw = sqlite3.connect(populated.path)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(f"DELETE FROM {populated.entries_table}")
        finally:
            raw.close()

        assert populated.count() == 3

    def insert_variant(self, backend, *, source_sequence=1, **overrides):
        """Re-insert an existing row with some columns changed, bypassing ``check_link``.

        The uniqueness constraints are defence in depth *behind* the link check, so the
        only way to prove the schema really declares them is to go around it.
        """
        columns = [
            "sequence", "entry_id", "ledger_id", "format_version", "timestamp",
            "timestamp_us", "data", "metadata", "previous_hash", "hash", "key_id",
            "signature_algorithm", "signature",
        ]
        selected = [repr(overrides[c]) if c in overrides else c for c in columns]
        raw = sqlite3.connect(backend.path)
        try:
            raw.execute(
                f"INSERT INTO {backend.entries_table} ({', '.join(columns)}) "
                f"SELECT {', '.join(selected)} FROM {backend.entries_table} "
                f"WHERE sequence = {source_sequence}"
            )
            raw.commit()
        finally:
            raw.close()

    def test_the_schema_declares_unique_sequence(self, populated):
        with pytest.raises(sqlite3.IntegrityError):
            self.insert_variant(
                populated, entry_id="dup-1", previous_hash="a" * 64, hash="b" * 64
            )

    def test_the_schema_declares_unique_previous_hash(self, populated):
        """Two entries claiming the same parent is a fork, and the schema must refuse
        it even if the application-level link check is somehow bypassed."""
        with pytest.raises(sqlite3.IntegrityError):
            self.insert_variant(populated, sequence=99, entry_id="dup-2", hash="c" * 64)

    def test_the_schema_declares_unique_entry_id(self, populated):
        with pytest.raises(sqlite3.IntegrityError):
            self.insert_variant(
                populated, sequence=98, previous_hash="d" * 64, hash="e" * 64
            )

    def test_the_schema_declares_unique_hash(self, populated):
        with pytest.raises(sqlite3.IntegrityError):
            self.insert_variant(
                populated, sequence=97, entry_id="dup-4", previous_hash="f" * 64
            )

    def test_sqlite_reports_append_only_enforced(self, populated):
        assert CAP_APPEND_ONLY_ENFORCED in populated.capabilities()

    def test_memory_does_not_claim_append_only_enforcement(self):
        backend = MemoryBackend()
        try:
            assert CAP_APPEND_ONLY_ENFORCED not in backend.capabilities()
        finally:
            backend.close()


# --------------------------------------------------------------------------------------
# emitted SQL
# --------------------------------------------------------------------------------------


class TestSQLiteQueryText:
    """SQLite has no bare ``OFFSET``. v1 emitted one and every offset query raised."""

    def test_an_offset_without_a_limit_emits_limit_minus_one(self):
        sql, params = build_entry_query("t", offset=5)

        assert "LIMIT -1 OFFSET ?" in sql
        assert params == [5]

    def test_a_bare_query_emits_neither_limit_nor_offset(self):
        sql, params = build_entry_query("t")

        assert "LIMIT" not in sql
        assert "OFFSET" not in sql
        assert params == []

    def test_a_limit_with_an_offset_emits_both_as_parameters(self):
        sql, params = build_entry_query("t", limit=10, offset=20)

        assert "LIMIT ? OFFSET ?" in sql
        assert params == [10, 20]

    def test_a_limit_without_an_offset_emits_no_offset(self):
        sql, params = build_entry_query("t", limit=10)

        assert "LIMIT ?" in sql
        assert "OFFSET" not in sql
        assert params == [10]

    def test_a_zero_limit_is_still_emitted(self):
        sql, params = build_entry_query("t", limit=0)

        assert "LIMIT ?" in sql
        assert params == [0]

    def test_the_query_orders_by_sequence_and_never_by_timestamp(self):
        sql, _ = build_entry_query(
            "t",
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert "ORDER BY sequence ASC" in sql
        assert "ORDER BY timestamp" not in sql

    def test_time_filters_are_bound_as_integer_microseconds(self):
        moment = datetime(2024, 1, 1, tzinfo=UTC)
        _, params = build_entry_query("t", start_time=moment)

        assert params == [timestamp_to_micros(moment)]

    def test_every_filter_becomes_a_bound_parameter(self):
        sql, params = build_entry_query(
            "t",
            start_seq=1,
            end_seq=9,
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            end_time=datetime(2024, 2, 1, tzinfo=UTC),
        )

        assert sql.count("?") == len(params) == 4


# --------------------------------------------------------------------------------------
# connections and threads
# --------------------------------------------------------------------------------------


class TestSQLiteConnections:
    def test_entries_appended_from_many_threads_form_one_gapless_chain(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "threads.db"), timeout=30.0)
        suite = BackendConformanceSuite()
        errors = []

        def worker():
            try:
                for _ in range(15):
                    suite.append(backend, {"thread": threading.get_ident()})
            except BaseException as err:  # pragma: no cover - only on a real failure
                errors.append(err)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert errors == []
            entries = list(backend.iter_entries())
            assert [e.sequence for e in entries] == list(range(120))
            assert all(
                entries[i].previous_hash == entries[i - 1].hash for i in range(1, len(entries))
            )
        finally:
            backend.close()

    def test_each_thread_gets_its_own_connection(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "registry.db"))
        try:
            backend.count()

            def worker():
                backend.count()

            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert len(backend._connections) == 4
        finally:
            backend.close()

    def test_close_closes_every_registered_connection(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "closeall.db"))
        backend.count()

        def worker():
            backend.count()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        tracked = list(backend._connections.values())
        assert len(tracked) == 2

        backend.close()

        for connection in tracked:
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

    def test_close_empties_the_connection_registry(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "empty.db"))
        backend.count()

        backend.close()

        assert backend._connections == {}

    def test_check_same_thread_false_shares_a_single_connection(self, tmp_path):
        """``False`` is the caller accepting a shared connection, so there is one."""
        backend = SQLiteBackend(str(tmp_path / "cst2.db"), check_same_thread=False)
        try:
            backend.count()

            def worker():
                backend.count()

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

            assert len(backend._connections) == 1
        finally:
            backend.close()

    def test_a_shared_connection_still_appends_correctly_from_two_threads(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "cst3.db"), check_same_thread=False)
        suite = BackendConformanceSuite()
        try:

            def worker():
                for _ in range(10):
                    suite.append(backend, {"t": threading.get_ident()})

            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert [e.sequence for e in backend.iter_entries()] == list(range(30))
        finally:
            backend.close()

    def test_a_connection_belonging_to_a_finished_thread_is_still_closed(self, tmp_path):
        """A worker thread that has exited cannot close its own handle, so ``close()``
        has to be able to - otherwise every thread that ever wrote leaks one."""
        backend = SQLiteBackend(str(tmp_path / "orphan.db"))
        captured = []

        def worker():
            backend.count()
            captured.append(backend._connections[threading.get_ident()])

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        backend.close()

        with pytest.raises(sqlite3.ProgrammingError):
            captured[0].execute("SELECT 1")

    def test_an_in_memory_database_shares_one_connection_across_threads(self):
        backend = SQLiteBackend(":memory:")
        suite = BackendConformanceSuite()
        try:
            suite.append(backend)
            seen = []

            def worker():
                seen.append(backend.count())

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

            assert seen == [1]
            assert len(backend._connections) == 1
        finally:
            backend.close()

    def test_an_in_memory_database_does_not_claim_durable_fsync(self):
        backend = SQLiteBackend(":memory:")
        try:
            assert CAP_DURABLE_FSYNC not in backend.capabilities()
        finally:
            backend.close()

    def test_a_file_database_claims_durable_fsync(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "durable.db"))
        try:
            assert CAP_DURABLE_FSYNC in backend.capabilities()
        finally:
            backend.close()

    def test_a_file_database_actually_synchronises_on_commit(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "sync.db"))
        try:
            level = backend._connection().execute("PRAGMA synchronous").fetchone()[0]
            assert int(level) >= 2
        finally:
            backend.close()

    def test_entries_survive_closing_and_reopening_the_database(self, tmp_path):
        path = str(tmp_path / "persist.db")
        suite = BackendConformanceSuite()
        writer = SQLiteBackend(path)
        written = suite.append_many(writer, 4)
        writer.close()

        reader = SQLiteBackend(path)
        try:
            assert list(reader.iter_entries()) == written
        finally:
            reader.close()

    def test_reopening_does_not_duplicate_the_schema(self, tmp_path):
        path = str(tmp_path / "reopen.db")
        first = SQLiteBackend(path)
        BackendConformanceSuite().append(first)
        first.close()

        second = SQLiteBackend(path)
        try:
            assert second.count() == 1
        finally:
            second.close()

    def test_size_bytes_of_a_file_database_grows_with_the_ledger(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "size.db"))
        try:
            empty = backend.size_bytes()
            BackendConformanceSuite().append_many(backend, 200)

            assert backend.size_bytes() > empty
        finally:
            backend.close()

    def test_a_closed_backend_refuses_to_open_a_new_connection(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "closed.db"))
        backend.close()

        with pytest.raises(StorageError):
            backend.get_head()

    def test_saving_a_checkpoint_after_close_raises_storage_error(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "closedcp.db"))
        backend.close()

        with pytest.raises(StorageError):
            backend.save_checkpoint(BackendConformanceSuite().checkpoint(0))

    def test_a_failing_checkpoint_insert_rolls_back_and_raises_storage_error(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "cpfail.db"))
        try:
            raw = sqlite3.connect(backend.path)
            raw.execute(f"DROP TABLE {backend.checkpoints_table}")
            raw.commit()
            raw.close()

            with pytest.raises(StorageError):
                backend.save_checkpoint(BackendConformanceSuite().checkpoint(0))
        finally:
            backend.close()

    def test_the_connection_is_usable_after_a_failed_checkpoint_insert(self, tmp_path):
        """A rollback that leaves the transaction open would wedge the connection."""
        backend = SQLiteBackend(str(tmp_path / "cprecover.db"))
        try:
            raw = sqlite3.connect(backend.path)
            raw.execute(f"DROP TABLE {backend.checkpoints_table}")
            raw.commit()
            raw.close()

            with pytest.raises(StorageError):
                backend.save_checkpoint(BackendConformanceSuite().checkpoint(0))

            assert BackendConformanceSuite().append(backend).sequence == 0
        finally:
            backend.close()

    @pytest.mark.parametrize(
        "call",
        [
            lambda b: b.count(),
            lambda b: b.get_head(),
            lambda b: b.get_oldest(),
            lambda b: b.get_entry("x"),
            lambda b: b.get_by_sequence(0),
            lambda b: list(b.iter_entries()),
        ],
        ids=["count", "head", "oldest", "by_id", "by_sequence", "iter"],
    )
    def test_a_read_against_a_broken_database_raises_storage_error(self, tmp_path, call):
        """Every public method speaks the library's error vocabulary. A caller must not
        have to catch ``sqlite3.OperationalError`` to use a ledger."""
        backend = SQLiteBackend(str(tmp_path / "brokenread.db"))
        try:
            raw = sqlite3.connect(backend.path)
            raw.execute(f"DROP TABLE {backend.entries_table}")
            raw.commit()
            raw.close()

            with pytest.raises(StorageError):
                call(backend)
        finally:
            backend.close()

    @pytest.mark.parametrize(
        "call",
        [lambda b: b.get_latest_checkpoint(), lambda b: list(b.iter_checkpoints())],
        ids=["latest", "iter"],
    )
    def test_a_checkpoint_read_against_a_broken_database_raises_storage_error(
        self, tmp_path, call
    ):
        backend = SQLiteBackend(str(tmp_path / "brokencp.db"))
        try:
            raw = sqlite3.connect(backend.path)
            raw.execute(f"DROP TABLE {backend.checkpoints_table}")
            raw.commit()
            raw.close()

            with pytest.raises(StorageError):
                call(backend)
        finally:
            backend.close()

    def test_size_bytes_of_a_broken_database_still_answers(self, tmp_path):
        """``page_count`` does not depend on the table, so this must not start failing."""
        backend = SQLiteBackend(str(tmp_path / "sizeok.db"))
        try:
            assert backend.size_bytes() > 0
        finally:
            backend.close()

    def test_a_driver_error_surfaces_as_a_storage_error(self, tmp_path):
        """``append`` fails closed: a backend failure is a ``StorageError``, never a
        warning followed by an unlinked row."""
        backend = SQLiteBackend(str(tmp_path / "broken.db"))
        try:
            raw = sqlite3.connect(backend.path)
            raw.execute(f"DROP TRIGGER {backend.entries_table}_no_update")
            raw.execute(f"ALTER TABLE {backend.entries_table} RENAME TO gone")
            raw.commit()
            raw.close()

            with pytest.raises(StorageError):
                BackendConformanceSuite().append(backend)
        finally:
            backend.close()


class TestMemoryBackendSpecifics:
    def test_size_bytes_grows_with_each_entry(self):
        backend = MemoryBackend()
        try:
            empty = backend.size_bytes()
            BackendConformanceSuite().append_many(backend, 5)

            assert backend.size_bytes() > empty
        finally:
            backend.close()

    def test_a_second_close_is_harmless_after_entries_exist(self):
        backend = MemoryBackend()
        BackendConformanceSuite().append(backend)
        backend.close()
        backend.close()

        with pytest.raises(StorageError):
            backend.count()

    def test_checkpoint_reads_after_close_raise_storage_error(self):
        backend = MemoryBackend()
        backend.save_checkpoint(BackendConformanceSuite().checkpoint(0))
        backend.close()

        with pytest.raises(StorageError):
            backend.get_latest_checkpoint()

    def test_iterating_checkpoints_after_close_raises_storage_error(self):
        backend = MemoryBackend()
        backend.close()

        with pytest.raises(StorageError):
            list(backend.iter_checkpoints())

    def test_saving_a_checkpoint_after_close_raises_storage_error(self):
        backend = MemoryBackend()
        backend.close()

        with pytest.raises(StorageError):
            backend.save_checkpoint(BackendConformanceSuite().checkpoint(0))

    def test_reads_by_id_after_close_raise_storage_error(self):
        backend = MemoryBackend()
        backend.close()

        with pytest.raises(StorageError):
            backend.get_entry("x")

    def test_reads_by_sequence_after_close_raise_storage_error(self):
        backend = MemoryBackend()
        backend.close()

        with pytest.raises(StorageError):
            backend.get_by_sequence(0)

    def test_reading_the_oldest_after_close_raises_storage_error(self):
        backend = MemoryBackend()
        backend.close()

        with pytest.raises(StorageError):
            backend.get_oldest()

    def test_reading_the_head_after_close_raises_storage_error(self):
        backend = MemoryBackend()
        backend.close()

        with pytest.raises(StorageError):
            backend.get_head()

    def test_size_bytes_after_close_raises_storage_error(self):
        backend = MemoryBackend()
        backend.close()

        with pytest.raises(StorageError):
            backend.size_bytes()

    def test_a_duplicate_previous_hash_is_refused_by_the_uniqueness_check(self):
        """The stand-in for ``UNIQUE(previous_hash)``. It sits behind ``check_link``, so
        it is reached directly here for the same reason the SQL schema test bypasses the
        link check: a fork must be impossible at the storage layer too."""
        from signledger.backends.memory import _ConstraintViolationError

        backend = MemoryBackend()
        try:
            suite = BackendConformanceSuite()
            entries = suite.append_many(backend, 2)
            clash = build_next_entry(
                entries[0],
                ledger_id=suite.ledger_id,
                created_at=suite.created_at,
                data={"n": "fork"},
            )

            with pytest.raises(_ConstraintViolationError):
                backend._reject_duplicates(clash)
        finally:
            backend.close()

    def test_a_negative_sequence_read_returns_none_rather_than_the_tail(self):
        """Python indexing would happily read ``entries[-1]`` and hand back the head."""
        backend = MemoryBackend()
        try:
            BackendConformanceSuite().append_many(backend, 3)

            assert backend.get_by_sequence(-1) is None
        finally:
            backend.close()


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


class TestPayloadSerialization:
    def test_none_metadata_serializes_to_an_empty_object_not_null(self):
        """The one line that broke every v1 SQLite and Postgres ledger."""
        assert serialize_payload(None) == "{}"

    def test_an_empty_dict_serializes_to_an_empty_object(self):
        assert serialize_payload({}) == "{}"

    def test_a_payload_serializes_with_sorted_keys(self):
        assert serialize_payload({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'

    def test_non_ascii_survives_serialization_unescaped(self):
        assert serialize_payload({"k": "café"}) == '{"k": "café"}'

    def test_null_reads_back_as_an_empty_dict(self):
        assert deserialize_payload(None) == {}

    def test_empty_text_reads_back_as_an_empty_dict(self):
        assert deserialize_payload("") == {}

    def test_an_object_reads_back_as_a_dict(self):
        assert deserialize_payload('{"a": 1}') == {"a": 1}

    def test_a_stored_array_is_refused(self):
        with pytest.raises(StorageError):
            deserialize_payload("[1, 2]")

    def test_a_stored_scalar_is_refused(self):
        with pytest.raises(StorageError):
            deserialize_payload("42")

    def test_the_refusal_message_never_quotes_the_payload(self):
        with pytest.raises(StorageError) as caught:
            deserialize_payload('"hunter2"')

        assert "hunter2" not in str(caught.value)


class TestTimestampEncoding:
    def test_a_naive_datetime_is_refused(self):
        with pytest.raises(ValidationError):
            timestamp_to_micros(datetime(2024, 1, 1))

    def test_the_epoch_encodes_to_zero(self):
        assert timestamp_to_micros(datetime(1970, 1, 1, tzinfo=UTC)) == 0

    def test_microsecond_precision_survives_the_round_trip(self):
        moment = datetime(2024, 6, 1, 12, 0, 0, 999999, tzinfo=UTC)

        assert micros_to_timestamp(timestamp_to_micros(moment)) == moment

    def test_a_single_microsecond_is_not_lost(self):
        moment = datetime(2024, 6, 1, 12, 0, 0, 1, tzinfo=UTC)

        assert micros_to_timestamp(timestamp_to_micros(moment)) == moment

    def test_an_offset_timestamp_encodes_as_the_same_instant_as_its_utc_form(self):
        east = datetime(2024, 6, 1, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        utc = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

        assert timestamp_to_micros(east) == timestamp_to_micros(utc)

    def test_a_pre_epoch_timestamp_encodes_negative(self):
        assert timestamp_to_micros(datetime(1969, 12, 31, tzinfo=UTC)) < 0

    def test_decoding_returns_a_utc_aware_datetime(self):
        assert micros_to_timestamp(0).tzinfo is UTC


class TestRowMapping:
    def build(self, **kwargs):
        fields = {
            "ledger_id": "led-1",
            "sequence": 0,
            "entry_id": "e-1",
            "timestamp": datetime(2024, 6, 1, 12, 0, 0, 123456, tzinfo=UTC),
            "data": {"event": "x"},
            "previous_hash": "0" * 64,
        }
        fields.update(kwargs)
        draft = Entry(**fields)
        from signledger.crypto.hashing import HashChain

        return draft.model_copy(update={"hash": HashChain().calculate_hash(draft)})

    def test_metadata_is_stored_as_an_object_even_when_absent(self):
        assert entry_to_row(self.build())["metadata"] == "{}"

    def test_no_column_of_a_minimal_entry_holds_a_null_payload(self):
        row = entry_to_row(self.build())

        assert row["data"] is not None
        assert row["metadata"] is not None

    def test_a_row_rebuilds_into_an_equal_entry(self):
        entry = self.build(metadata={"a": 1})

        assert entry_from_row(entry_to_row(entry)) == entry

    def test_a_row_preserves_the_optional_signature_columns(self):
        entry = self.build(key_id="kid", signature_algorithm="ed25519")
        rebuilt = entry_from_row(entry_to_row(entry))

        assert (rebuilt.key_id, rebuilt.signature_algorithm) == ("kid", "ed25519")

    def test_a_legacy_null_metadata_column_still_loads(self):
        """Reading a v1 row must not be a second outage."""
        row = entry_to_row(self.build())
        row["metadata"] = None

        assert entry_from_row(row).metadata == {}

    def test_a_checkpoint_row_rebuilds_into_an_equal_checkpoint(self):
        cp = BackendConformanceSuite().checkpoint(4, signature="sig", key_id="kid")

        assert checkpoint_from_row(checkpoint_to_row(cp)) == cp


# --------------------------------------------------------------------------------------
# decoding a stored row fails closed (contract sections 4 and 10)
# --------------------------------------------------------------------------------------


class TestRowMappingFailsClosed:
    """A row missing a column is a ``StorageError``, never a bare ``KeyError``.

    Observed for real: a PostgreSQL connection pool inherited across ``fork()`` handed a
    gunicorn worker a cross-talked result row, ``entry_from_row`` indexed a column that
    was not there, and the ``KeyError`` escaped ``append()`` untouched. Contract section 4
    makes every storage failure a ``StorageError``; section 10 forbids the message from
    carrying payload.
    """

    secret = "4111-1111-1111-1111"

    def row(self, without=None):
        from signledger.crypto.hashing import HashChain

        draft = Entry(
            ledger_id="led-1",
            sequence=0,
            entry_id="e-1",
            timestamp=datetime(2024, 6, 1, 12, 0, 0, 123456, tzinfo=UTC),
            data={"event": "charge", "card": self.secret},
            previous_hash="0" * 64,
        )
        sealed = draft.model_copy(update={"hash": HashChain().calculate_hash(draft)})
        row = entry_to_row(sealed)
        if without is not None:
            del row[without]
        return row

    def test_a_row_missing_a_column_raises_storage_error_not_key_error(self):
        with pytest.raises(StorageError):
            entry_from_row(self.row(without="format_version"))

    def test_the_error_names_the_missing_column(self):
        with pytest.raises(StorageError) as caught:
            entry_from_row(self.row(without="format_version"))

        assert "format_version" in str(caught.value)

    def test_a_missing_column_read_late_is_named_just_as_precisely(self):
        """``signature`` is the last column read, so the whole row decodes first."""
        with pytest.raises(StorageError) as caught:
            entry_from_row(self.row(without="signature"))

        assert "signature" in str(caught.value)

    def test_the_error_derives_from_the_package_base_error(self):
        from signledger.core.exceptions import SignLedgerError

        with pytest.raises(SignLedgerError):
            entry_from_row(self.row(without="timestamp_us"))

    def test_the_error_never_quotes_the_payload(self):
        """Contract section 10. The row still holds ``data`` when the decode fails."""
        with pytest.raises(StorageError) as caught:
            entry_from_row(self.row(without="format_version"))

        assert self.secret not in str(caught.value)

    def test_a_complete_row_still_rebuilds(self):
        assert entry_from_row(self.row()).entry_id == "e-1"


class TestPayloadDeserializationFailsClosed:
    """A stored payload that is not JSON text is corruption, not a ``TypeError``.

    MongoDB will happily hold a subdocument in ``data``, and ``json.loads`` then raised a
    bare ``TypeError`` while the :class:`~signledger.core.ledger.Ledger` was still being
    built - so ``check_integrity()`` never got the chance to report the corruption at all.
    """

    def test_a_document_payload_raises_storage_error_not_type_error(self):
        with pytest.raises(StorageError):
            deserialize_payload({"event": "charge"})

    def test_a_numeric_payload_raises_storage_error(self):
        with pytest.raises(StorageError):
            deserialize_payload(42)

    def test_a_list_payload_raises_storage_error(self):
        with pytest.raises(StorageError):
            deserialize_payload(["a"])

    def test_the_error_names_the_type_it_found(self):
        with pytest.raises(StorageError) as caught:
            deserialize_payload({"event": "charge"})

        assert "dict" in str(caught.value)

    def test_the_error_derives_from_the_package_base_error(self):
        from signledger.core.exceptions import SignLedgerError

        with pytest.raises(SignLedgerError):
            deserialize_payload({"event": "charge"})

    def test_the_error_never_quotes_the_payload_value(self):
        """Contract section 10 - naming the type is fine, printing the value is not.

        This library already shipped a payload leak once. A ``dict`` interpolated into an
        f-string prints its keys *and* its values straight into the log line.
        """
        with pytest.raises(StorageError) as caught:
            deserialize_payload({"card": "4111-1111-1111-1111"})

        message = str(caught.value)
        assert "4111-1111-1111-1111" not in message
        assert "card" not in message

    def test_bytes_are_still_decoded(self):
        """``json.loads`` accepts them, so refusing them would be a new outage."""
        assert deserialize_payload(b'{"a": 1}') == {"a": 1}

    def test_a_bytearray_is_still_decoded(self):
        assert deserialize_payload(bytearray(b'{"a": 1}')) == {"a": 1}


class TestChainLinkChecking:
    def entry(self, tip=None, **kwargs):
        return build_next_entry(
            tip, ledger_id="led-1", created_at=datetime(2024, 1, 1, tzinfo=UTC),
            data={"n": 1}, **kwargs
        )

    def test_a_genesis_entry_is_accepted_on_an_empty_store(self):
        check_link(None, self.entry())

    def test_a_successor_is_accepted(self):
        first = self.entry()

        check_link(first, self.entry(first))

    def test_a_non_zero_sequence_is_refused_on_an_empty_store(self):
        first = self.entry()

        with pytest.raises(StorageError):
            check_link(None, self.entry(first))

    def test_a_skipped_sequence_is_refused(self):
        first = self.entry()
        second = self.entry(first)
        third = self.entry(second)

        with pytest.raises(StorageError):
            check_link(first, third)

    def test_a_skipped_sequence_is_refused_even_when_the_hash_link_is_valid(self):
        """Isolates the sequence rule from the hash rule.

        Every other case trips the ``previous_hash`` comparison first, so without this
        one the sequence check could be deleted and the suite would stay green - which a
        mutation run duly proved. ``sequence`` is the ordering authority and gaps are a
        ``SEQUENCE_GAP`` failure, so the storage layer must refuse to create one.
        """
        first = self.entry()
        forged = self.entry(first).model_copy(update={"sequence": 7})

        assert forged.previous_hash == first.hash

        with pytest.raises(StorageError):
            check_link(first, forged)

    def test_a_repeated_sequence_is_refused_even_when_the_hash_link_is_valid(self):
        first = self.entry()
        forged = self.entry(first).model_copy(update={"sequence": first.sequence})

        with pytest.raises(StorageError):
            check_link(first, forged)

    def test_a_mismatched_previous_hash_is_refused(self):
        first = self.entry()
        second = self.entry(first)
        forged = second.model_copy(update={"previous_hash": "9" * 64})

        with pytest.raises(StorageError):
            check_link(first, forged)


class TestRetryOnConflict:
    def test_a_successful_operation_runs_once(self):
        calls = []

        result = retry_on_conflict(lambda: (calls.append(1), "ok")[1], lambda err: True)

        assert result == "ok"
        assert len(calls) == 1

    def test_a_conflict_is_retried_until_it_succeeds(self):
        calls = []

        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise KeyError("conflict")
            return "ok"

        result = retry_on_conflict(
            operation, lambda err: isinstance(err, KeyError), sleep=lambda _: None
        )

        assert result == "ok"
        assert len(calls) == 3

    def test_an_unrecognised_error_is_raised_on_the_first_attempt(self):
        calls = []

        def operation():
            calls.append(1)
            raise ValueError("not a conflict")

        with pytest.raises(ValueError):
            retry_on_conflict(operation, lambda err: False, sleep=lambda _: None)

        assert len(calls) == 1

    def test_persistent_conflict_becomes_a_storage_error(self):
        def operation():
            raise KeyError("conflict")

        with pytest.raises(StorageError):
            retry_on_conflict(operation, lambda err: True, sleep=lambda _: None)

    def test_the_number_of_attempts_is_bounded(self):
        calls = []

        def operation():
            calls.append(1)
            raise KeyError("conflict")

        with pytest.raises(StorageError):
            retry_on_conflict(operation, lambda err: True, sleep=lambda _: None)

        assert len(calls) == MAX_APPEND_ATTEMPTS

    def test_the_backoff_doubles_between_attempts(self):
        delays = []

        def operation():
            raise KeyError("conflict")

        with pytest.raises(StorageError):
            retry_on_conflict(operation, lambda err: True, sleep=delays.append)

        assert delays == [APPEND_RETRY_BASE_DELAY * (2**n) for n in range(MAX_APPEND_ATTEMPTS - 1)]

    def test_the_original_conflict_is_kept_as_the_cause(self):
        def operation():
            raise KeyError("conflict")

        with pytest.raises(StorageError) as caught:
            retry_on_conflict(operation, lambda err: True, sleep=lambda _: None)

        assert isinstance(caught.value.__cause__, KeyError)

    def test_zero_attempts_is_a_configuration_error(self):
        with pytest.raises(ConfigurationError):
            retry_on_conflict(lambda: "ok", lambda err: True, attempts=0)

    def test_a_single_attempt_is_allowed(self):
        assert retry_on_conflict(lambda: "ok", lambda err: True, attempts=1) == "ok"


class TestProtocolConformance:
    def test_the_memory_backend_satisfies_the_storage_backend_protocol(self):
        backend = MemoryBackend()
        try:
            assert isinstance(backend, StorageBackend)
        finally:
            backend.close()

    def test_the_sqlite_backend_satisfies_the_storage_backend_protocol(self, tmp_path):
        backend = SQLiteBackend(str(tmp_path / "proto.db"))
        try:
            assert isinstance(backend, StorageBackend)
        finally:
            backend.close()

    def test_an_object_missing_a_method_does_not_satisfy_the_protocol(self):
        class Incomplete:
            def close(self):
                return None

        assert not isinstance(Incomplete(), StorageBackend)

    def test_the_contract_name_for_the_mixin_is_the_same_class(self):
        assert ConformanceTestMixin is BackendConformanceSuite


class TestPackageExports:
    def test_the_backends_package_exports_both_shipped_backends(self):
        import signledger.backends as package

        assert package.MemoryBackend is MemoryBackend
        assert package.SQLiteBackend is SQLiteBackend

    def test_the_backends_package_exports_the_protocol(self):
        import signledger.backends as package

        assert package.StorageBackend is StorageBackend

    def test_the_backends_package_exports_the_conformance_suite(self):
        import signledger.backends as package

        assert package.BackendConformanceSuite is BackendConformanceSuite

    def test_everything_named_in_all_is_importable(self):
        import signledger.backends as package

        for name in package.__all__:
            assert hasattr(package, name), name


# --------------------------------------------------------------------------------------
# Backend construction must fail closed too (contract sections 4 and 10)
# --------------------------------------------------------------------------------------


class TestSQLiteConstructionFailsClosed:
    """Opening a store is as much a backend operation as reading from one.

    Wave 5 found that ``_open_connection`` sits outside ``_translating``, so a failure to
    open escapes as a raw ``sqlite3`` error. Two contract rules break at once: section 4
    requires backend failures to surface as ``StorageError``, and section 10 promises one
    ``except SignLedgerError`` contains the library. A caller who wraps
    ``SQLiteBackend(path)`` in ``except SignLedgerError`` - the documented way to use it -
    currently gets an unhandled ``sqlite3.OperationalError`` instead.

    This is also the surface the reported gunicorn cold-start race arrives through: two
    workers opening a brand-new ledger at the same instant, one of them refused
    ``PRAGMA journal_mode = WAL``. That race is timing-dependent and not reproduced here;
    these cases are its deterministic siblings.
    """

    def test_a_path_whose_directory_does_not_exist_raises_storage_error(self, tmp_path):
        from signledger.backends.sqlite import SQLiteBackend

        with pytest.raises(StorageError):
            SQLiteBackend(str(tmp_path / "no-such-directory" / "ledger.db"))

    def test_a_path_that_is_a_directory_raises_storage_error(self, tmp_path):
        from signledger.backends.sqlite import SQLiteBackend

        with pytest.raises(StorageError):
            SQLiteBackend(str(tmp_path))

    def test_a_file_that_is_not_a_database_raises_storage_error(self, tmp_path):
        from signledger.backends.sqlite import SQLiteBackend

        corrupt = tmp_path / "corrupt.db"
        corrupt.write_bytes(b"definitely not a sqlite database " * 64)
        with pytest.raises(StorageError):
            SQLiteBackend(str(corrupt))

    def test_the_open_failure_does_not_leak_the_path(self, tmp_path):
        """Section 10: a path can carry a customer name or a deployment secret."""
        from signledger.backends.sqlite import SQLiteBackend

        secret = "tenant-acme-do-not-log"
        with pytest.raises(StorageError) as caught:
            SQLiteBackend(str(tmp_path / "missing" / f"{secret}.db"))
        assert secret not in f"{caught.value!s} {caught.value!r}"

    def test_the_open_failure_still_names_the_driver_error_type(self, tmp_path):
        """Redaction must not leave an operator with nothing to diagnose."""
        from signledger.backends.sqlite import SQLiteBackend

        with pytest.raises(StorageError, match="OperationalError"):
            SQLiteBackend(str(tmp_path / "missing" / "ledger.db"))

    def test_the_underlying_driver_error_is_still_chained(self, tmp_path):
        """``raise ... from err`` keeps the real cause for a debugger, out of the message."""
        import sqlite3

        from signledger.backends.sqlite import SQLiteBackend

        with pytest.raises(StorageError) as caught:
            SQLiteBackend(str(tmp_path / "missing" / "ledger.db"))
        assert isinstance(caught.value.__cause__, sqlite3.Error)

    def test_a_healthy_path_still_opens(self, tmp_path):
        """Guards the tests above: a constructor that always raised would pass them all."""
        from signledger.backends.sqlite import SQLiteBackend

        backend = SQLiteBackend(str(tmp_path / "fine.db"))
        try:
            assert backend.count() == 0
        finally:
            backend.close()


class TestSQLiteSurvivesWalPragmaContention:
    """A second opener must not die because the first one is mid-transaction.

    ``PRAGMA journal_mode = WAL`` is refused with SQLITE_BUSY while another connection
    holds a write lock. That is the reported gunicorn symptom: two workers open the same
    ledger file at the same instant and one dies on startup.

    Held-lock contention is the deterministic form of that race, so it is what these
    tests use - no sleeping on a hope, and no test that passes half the time.

    Measured here, and worth recording because the answer depends on *which* lock the
    other connection holds. Against ``BEGIN EXCLUSIVE`` the pragma cannot even take its
    read lock, so SQLite consults the busy handler and the opener waits - which is what
    these two tests exercise, and why either ``sqlite3.connect(timeout=...)`` or
    ``PRAGMA busy_timeout`` alone keeps them green. Against a mere RESERVED lock
    (``BEGIN IMMEDIATE`` plus a write, i.e. what a live worker actually holds) the
    journal-mode change is refused with SQLITE_BUSY *immediately*, busy handler ignored.
    That is the case the multi-process failure came from, and it is covered by
    ``tests/test_sqlite_concurrency.py``, which pins the retry that fixes it.
    """

    @staticmethod
    def _hold_exclusive(path, hold_for, released):
        """Hold a write lock for ``hold_for`` seconds, then release it."""
        import sqlite3
        import threading

        def run():
            connection = sqlite3.connect(path, isolation_level=None)
            connection.execute("BEGIN EXCLUSIVE")
            time.sleep(hold_for)
            connection.execute("COMMIT")
            connection.close()
            released.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def test_opening_waits_out_a_writer_that_holds_the_lock(self, tmp_path):
        """The whole point: the second worker survives the first worker's transaction."""
        import sqlite3
        import threading

        from signledger.backends.sqlite import SQLiteBackend

        path = str(tmp_path / "contended.db")
        sqlite3.connect(path).close()  # the file exists, not yet in WAL mode

        released = threading.Event()
        holder = self._hold_exclusive(path, 0.75, released)
        time.sleep(0.15)  # make sure the lock is actually held before we try

        backend = SQLiteBackend(path, timeout=10.0)
        try:
            assert backend.count() == 0
        finally:
            backend.close()
            holder.join(timeout=10)
        assert released.is_set()

    #: How much longer each ``busy_timeout`` wait takes to come back than it asked for,
    #: as a multiple of the timeout itself, injected per attempt. ``0`` is this machine.
    #: ``3`` is worse than the runner that broke the old wall-clock bound: windows-latest
    #: returned 3.02s for five 0.3s waits, a multiple of about one.
    RUNNER_SLOWNESS = (0, 3)

    @pytest.mark.parametrize("slowness", RUNNER_SLOWNESS)
    def test_a_lock_held_throughout_still_fails_as_storage_error(
        self, tmp_path, monkeypatch, slowness
    ):
        """Guards the retry: waiting for ever is not an improvement on failing.

        A retry loop with no bound turns one stuck worker into a hung fleet. The lock is
        held for the whole of the attempt and released only after it, so this observes the
        bound rather than racing a sleeping thread against it.

        **Why the bound is no longer a wall clock.** This used to end in
        ``assert timeout <= elapsed < MAX_APPEND_ATTEMPTS * timeout + 1.0`` and
        windows-latest failed it with ``assert 3.020011799999992 < ((5 * 0.3) + 1.0)``.
        Nothing was wrong with the backend: the runner took about twice as long to return
        from each 0.3s wait as it was asked for. A ceiling in seconds on a shared runner
        measures the runner, and no ceiling tight enough to mean anything is loose enough
        to stay true - so the bound is asserted where this library actually imposes one,
        on the *number of attempts*.

        That is stronger than what it replaces, not weaker. ``elapsed < 2.5s`` was also
        satisfied by an unbounded loop that happened to get lucky and by a bound of any
        size that ran fast; ``attempts == MAX_APPEND_ATTEMPTS`` is satisfied only by the
        bound itself, and ``slowness`` is what proves it has teeth on a busy host: a
        runner that makes every wait four times as long moves the wall clock and changes
        nothing else here. An unbounded loop is not left to hang either - attempt six
        raises a ``RuntimeError``, which ``_is_conflict`` does not recognise and
        ``_translating`` does not catch, so it leaves the constructor immediately and this
        fails in seconds.

        The other half of the old assertion, "it did not fail instantly", was really
        asking whether ``busy_timeout`` had been applied at all - but whether SQLite
        consults the busy handler for a journal-mode change is SQLite's answer and varies
        with which lock is held (see the class docstring), so reading it off the clock is
        the same mistake in the other direction. What *this library* owes is the pragma,
        with the timeout converted to milliseconds, and that is asserted directly: a
        dropped ``* 1000`` fails the last line here without timing anything.
        """
        import sqlite3

        import signledger.backends.sqlite as sqlite_backend
        from signledger.backends.sqlite import SQLiteBackend

        timeout = 0.1
        settings = []
        attempts = []
        real_pragma = sqlite_backend._pragma
        real_pragma_value = sqlite_backend._pragma_value

        def record_setting(connection, statement):
            settings.append(statement)
            return real_pragma(connection, statement)

        def record_attempt(connection, statement):
            if not statement.strip().upper().startswith("PRAGMA JOURNAL_MODE"):
                return real_pragma_value(connection, statement)
            attempts.append(statement)
            if len(attempts) > MAX_APPEND_ATTEMPTS:
                # Deliberately not an OperationalError: `_is_conflict` says no to this, so
                # `retry_on_conflict` re-raises it untouched and an unbounded retry fails
                # here rather than running until the 300s suite timeout kills the session.
                raise RuntimeError(f"the wal retry is unbounded: attempt {len(attempts)}")
            time.sleep(slowness * timeout)  # the slow runner, reproduced on any OS
            return real_pragma_value(connection, statement)

        monkeypatch.setattr(sqlite_backend, "_pragma", record_setting)
        monkeypatch.setattr(sqlite_backend, "_pragma_value", record_attempt)

        path = str(tmp_path / "stuck.db")
        holder = sqlite3.connect(path, isolation_level=None)
        holder.execute("BEGIN EXCLUSIVE")
        try:
            with pytest.raises(StorageError):
                SQLiteBackend(path, timeout=timeout)
        finally:
            holder.execute("COMMIT")
            holder.close()

        # It gave up after exactly the attempts it is allowed, however long they took.
        assert len(attempts) == MAX_APPEND_ATTEMPTS
        # And each of those attempts was given the timeout it was configured with, in the
        # milliseconds SQLite expects - which is the part "it did not fail instantly" was
        # really about, stated without a stopwatch.
        assert [
            int(statement.split("=")[1])
            for statement in settings
            if statement.strip().upper().startswith("PRAGMA BUSY_TIMEOUT")
        ] == [int(timeout * 1000)]


class TestPayloadDecodeFailsClosed:
    """Malformed payload TEXT is a ``StorageError``, never a raw ``JSONDecodeError``.

    ``deserialize_payload`` gained a type guard, but the ``json.loads`` call one line below
    it stayed bare - so a truncated or corrupted ``data`` column raised a ``ValueError``
    subclass that is not a ``SignLedgerError`` and escaped every ``except`` in the library.
    A corrupted audit log is precisely when ``check_integrity()`` needs to run, and this
    raised during construction, before it could.
    """

    def test_malformed_json_text_is_a_storage_error(self) -> None:
        with pytest.raises(StorageError):
            deserialize_payload("not json at all")

    def test_truncated_json_is_a_storage_error(self) -> None:
        """The realistic shape of the failure: a column cut off mid-write."""
        with pytest.raises(StorageError):
            deserialize_payload('{"actor": "alice", "acti')

    def test_empty_bytes_read_as_an_absent_payload_like_an_empty_string(self) -> None:
        """``raw == ""`` is False for ``b""``, so the empty guard missed the bytes case.

        A driver that hands back ``bytes`` for a TEXT column then took the JSON path with
        nothing to decode.
        """
        assert deserialize_payload(b"") == {}

    def test_the_refusal_never_contains_the_payload(self) -> None:
        """Contract section 10. A corrupted row still holds real audit data."""
        secret = "4111-1111-1111-1111"

        with pytest.raises(StorageError) as caught:
            deserialize_payload('{"card": "' + secret + '"')

        assert secret not in str(caught.value)

    def test_a_decoded_payload_that_is_not_an_object_is_refused(self) -> None:
        """``json.loads("3")`` is valid JSON and a valid int. ``data`` is a mapping."""
        with pytest.raises(StorageError):
            deserialize_payload("3")


class TestCheckpointRowMappingFailsClosed:
    """``checkpoint_from_row`` had the same defect ``entry_from_row`` was fixed for.

    Same file, same read path, same bare ``KeyError`` - it was simply not in the brief.
    """

    def row(self, without=None):
        row = {
            "ledger_id": "led-1",
            "sequence": 4,
            "head_hash": "a" * 64,
            "entry_count": 5,
            "merkle_root": "b" * 64,
            "timestamp_us": 1717243200000000,
            "signature": None,
            "key_id": None,
            "format_version": 2,
        }
        if without is not None:
            del row[without]
        return row

    def test_a_complete_row_still_rebuilds(self) -> None:
        checkpoint = checkpoint_from_row(self.row())

        assert checkpoint.ledger_id == "led-1"
        assert checkpoint.entry_count == 5

    @pytest.mark.parametrize(
        "column",
        [
            "ledger_id",
            "sequence",
            "head_hash",
            "entry_count",
            "merkle_root",
            "timestamp_us",
            "signature",
            "key_id",
            "format_version",
        ],
    )
    def test_a_missing_column_is_a_storage_error_naming_it(self, column: str) -> None:
        with pytest.raises(StorageError) as caught:
            checkpoint_from_row(self.row(without=column))

        assert column in str(caught.value)


class TestTheSuiteRefusesToRunWithAssertionsDisabled:
    """The conformance suite is 72 bare ``assert`` statements, and ``python -O`` deletes them.

    A third-party backend author running this suite under ``-O`` gets a green result that
    proves nothing at all - the same defect as a verifier that returns success having
    checked nothing, which contract section 5 rule 1 forbids in ``check_integrity``. The
    suite has to hold itself to the rule it exists to enforce.
    """

    def test_the_suite_refuses_to_run_under_dash_o(self) -> None:
        """Run a real interpreter with -O; the suite must fail loudly, not pass quietly."""
        program = (
            "from signledger.backends.base import BackendConformanceSuite\n"
            "from signledger.backends.memory import MemoryBackend\n"
            "class MySuite(BackendConformanceSuite):\n"
            "    pass\n"
            "MySuite().test_a_fresh_backend_is_empty(MemoryBackend())\n"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", program],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode != 0, (
            "the suite ran to completion with assertions disabled - every check was a no-op"
        )
        assert "assertions are disabled" in (result.stdout + result.stderr)

    def test_the_suite_runs_normally_without_dash_o(self) -> None:
        """The guard must not break the ordinary path."""
        program = (
            "from signledger.backends.base import BackendConformanceSuite\n"
            "from signledger.backends.memory import MemoryBackend\n"
            "class MySuite(BackendConformanceSuite):\n"
            "    pass\n"
            "MySuite().test_a_fresh_backend_is_empty(MemoryBackend())\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
        )

        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


# --------------------------------------------------------------------------------------
# opening a store without provisioning it
# --------------------------------------------------------------------------------------


def _file_digest(path):
    """Every byte of the database file, so "unchanged" really means unchanged."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _journal_mode(path):
    """The journal mode recorded in the file's own header."""
    raw = sqlite3.connect(str(path))
    try:
        return str(raw.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        raw.close()


def _schema_objects(path):
    """Every table, index and trigger the file holds, by name."""
    raw = sqlite3.connect(str(path))
    try:
        return sorted(
            row[0]
            for row in raw.execute("SELECT name FROM sqlite_master WHERE name IS NOT NULL")
        )
    finally:
        raw.close()


class TestSQLiteOpensWithoutProvisioning:
    """``create=False`` reads a store that exists and refuses to bring one into being.

    Two separate defects live here. ``signledger verify`` on a mistyped path conjured a
    32 KiB database, armed its append-only triggers and reported an empty ledger - which
    an operator reads as "my audit log is gone", not as "I typed the wrong name". And
    ``PRAGMA journal_mode = WAL`` is itself a write to the database header, so an
    integrity check could not be run against a copy the process may not write at all.
    """

    @pytest.fixture
    def store(self, tmp_path):
        """A real three-entry ledger, on disk, closed."""
        path = str(tmp_path / "ledger.db")
        backend = SQLiteBackend(path)
        BackendConformanceSuite().append_many(backend, 3)
        backend.close()
        return path

    def test_a_missing_file_is_refused_rather_than_created(self, tmp_path):
        missing = tmp_path / "typo.db"

        with pytest.raises(StorageError) as caught:
            SQLiteBackend(str(missing), create=False)

        assert not missing.exists()
        assert "does not exist" in str(caught.value)

    def test_the_refusal_names_the_path_it_was_pointed_at(self, tmp_path):
        missing = tmp_path / "typo.db"

        with pytest.raises(StorageError) as caught:
            SQLiteBackend(str(missing), create=False)

        assert str(missing) in str(caught.value)

    def test_an_in_memory_database_is_refused(self):
        """It is created empty every time it is opened, so there is nothing to read."""
        with pytest.raises(StorageError) as caught:
            SQLiteBackend(":memory:", create=False)

        assert "in-memory" in str(caught.value)

    def test_a_database_that_holds_no_entries_table_is_refused(self, tmp_path):
        path = tmp_path / "unrelated.db"
        sqlite3.connect(str(path)).close()

        with pytest.raises(StorageError) as caught:
            SQLiteBackend(str(path), create=False)

        assert "ledger_entries" in str(caught.value)

    def test_a_store_whose_checkpoints_table_is_absent_is_refused(self, store):
        raw = sqlite3.connect(store)
        raw.execute("DROP TABLE ledger_checkpoints")
        raw.commit()
        raw.close()

        with pytest.raises(StorageError) as caught:
            SQLiteBackend(store, create=False)

        assert "ledger_checkpoints" in str(caught.value)

    def test_a_mistyped_table_name_is_refused_rather_than_created(self, store):
        with pytest.raises(StorageError) as caught:
            SQLiteBackend(store, entries_table="ledger_entires", create=False)

        assert "ledger_entires" in str(caught.value)
        assert "ledger_entires" not in _schema_objects(store)

    def test_the_file_is_byte_identical_after_a_read_only_open(self, store):
        before = _file_digest(store)

        backend = SQLiteBackend(store, create=False)
        try:
            assert [entry.sequence for entry in backend.iter_entries()] == [0, 1, 2]
        finally:
            backend.close()

        assert _file_digest(store) == before

    def test_the_journal_mode_recorded_in_the_file_is_left_alone(self, tmp_path):
        """``PRAGMA journal_mode = WAL`` writes to the database header. Never here."""
        path = str(tmp_path / "rollback.db")
        backend = SQLiteBackend(path)
        BackendConformanceSuite().append_many(backend, 1)
        backend.close()
        raw = sqlite3.connect(path)
        raw.execute("PRAGMA journal_mode = DELETE").fetchall()
        raw.close()

        reader = SQLiteBackend(path, create=False)
        try:
            assert reader.count() == 1
        finally:
            reader.close()

        assert _journal_mode(path) == "delete"

    def test_no_schema_object_is_created_by_a_read_only_open(self, store):
        before = _schema_objects(store)

        SQLiteBackend(store, create=False).close()

        assert _schema_objects(store) == before

    def test_a_removed_append_only_trigger_is_not_re_armed(self, store):
        """A missing trigger is evidence. Reading the store must not destroy it."""
        raw = sqlite3.connect(store)
        raw.execute("DROP TRIGGER ledger_entries_no_update")
        raw.execute("DROP TRIGGER ledger_entries_no_delete")
        raw.commit()
        raw.close()

        backend = SQLiteBackend(store, create=False)
        try:
            assert CAP_APPEND_ONLY_ENFORCED not in backend.capabilities()
        finally:
            backend.close()

        assert "ledger_entries_no_update" not in _schema_objects(store)

    def test_every_read_still_answers(self, store):
        backend = SQLiteBackend(store, create=False)
        try:
            assert backend.count() == 3
            assert backend.get_head().sequence == 2
            assert backend.get_oldest().sequence == 0
            assert backend.get_by_sequence(1).sequence == 1
            assert backend.get_entry(backend.get_head().entry_id) is not None
            assert backend.size_bytes() > 0
            assert backend.get_latest_checkpoint() is None
            assert list(backend.iter_checkpoints()) == []
            assert CAP_APPEND_ONLY_ENFORCED in backend.capabilities()
        finally:
            backend.close()

    def test_a_file_this_process_may_not_write_can_still_be_read(self, store):
        """The whole point: an integrity check run against a copy nobody may write."""
        os.chmod(store, 0o444)
        try:
            backend = SQLiteBackend(store, create=False)
            try:
                assert backend.count() == 3
            finally:
                backend.close()
        finally:
            os.chmod(store, 0o644)

    def test_an_append_is_refused_rather_than_attempted(self, store):
        backend = SQLiteBackend(store, create=False)
        try:
            with pytest.raises(StorageError) as caught:
                BackendConformanceSuite().append(backend)
        finally:
            backend.close()

        assert "create=False" in str(caught.value)

    def test_a_refused_append_leaves_the_file_byte_identical(self, store):
        before = _file_digest(store)
        backend = SQLiteBackend(store, create=False)
        try:
            with pytest.raises(StorageError):
                BackendConformanceSuite().append(backend)
        finally:
            backend.close()

        assert _file_digest(store) == before

    def test_saving_a_checkpoint_is_refused(self, store):
        suite = BackendConformanceSuite()
        backend = SQLiteBackend(store, create=False)
        try:
            with pytest.raises(StorageError) as caught:
                backend.save_checkpoint(suite.checkpoint(2))
        finally:
            backend.close()

        assert "create=False" in str(caught.value)

    def test_the_default_still_provisions_a_new_store(self, tmp_path):
        """Guards every refusal above: ordinary use is unchanged."""
        path = tmp_path / "new.db"

        backend = SQLiteBackend(str(path))
        try:
            assert backend.count() == 0
            assert CAP_APPEND_ONLY_ENFORCED in backend.capabilities()
        finally:
            backend.close()

        assert path.exists()


class TestTheMemoryBackendRefusesAReadOnlyOpen:
    """``create=False`` means "read a store that already exists", which memory never is.

    A fresh ``MemoryBackend`` is empty by construction, so opening one for reading and
    then reporting "no entries were verified" is an integrity verdict about a store that
    never existed. ``SQLiteBackend(":memory:", create=False)`` already refuses for exactly
    this reason; the two in-memory stores answered differently, and the CLI turned that
    into exit 1 (the log is damaged) where it should be exit 2 (you named nothing).
    """

    def test_a_read_only_open_is_refused(self) -> None:
        from signledger.backends.memory import MemoryBackend

        with pytest.raises(StorageError, match="every time it is opened"):
            MemoryBackend(create=False)

    def test_the_refusal_says_what_to_do_instead(self) -> None:
        from signledger.backends.memory import MemoryBackend

        with pytest.raises(StorageError) as caught:
            MemoryBackend(create=False)

        assert "sqlite:" in str(caught.value) or "file" in str(caught.value)

    def test_the_ordinary_open_is_unchanged(self) -> None:
        from signledger.backends.memory import MemoryBackend

        backend = MemoryBackend()
        try:
            assert backend.count() == 0
        finally:
            backend.close()
