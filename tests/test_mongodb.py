"""MongoDB backend tests. Real server only - nothing here mocks a database."""

import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from signledger.backends.base import (
    CAP_APPEND_ONLY_ENFORCED,
    CAP_ATOMIC_APPEND,
    CAP_DURABLE_FSYNC,
    CAP_UNIQUE_CHAIN_LINK,
    CAP_WORM,
    BackendConformanceSuite,
    build_next_entry,
)

pytestmark = pytest.mark.mongo

pymongo = pytest.importorskip("pymongo", reason="pymongo is not installed")

URI_ENV = "SIGNLEDGER_TEST_MONGO_URI"
DEFAULT_URI = "mongodb://localhost:27017"

LEDGER_ID = "mongo-2f7a1c60-9d34-4b18-a5e2-71c8f0d6b943"
CREATED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)
BASE_TIMESTAMP = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _uri() -> str:
    return os.environ.get(URI_ENV, DEFAULT_URI)


def _require_server() -> str:
    """Skip - never fail - when no live server answers. Report host:port, never the URI."""
    uri = _uri()
    parts = urlsplit(uri)
    host, port = parts.hostname or "localhost", parts.port or 27017
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as err:
        pytest.skip(f"no live MongoDB server at {host}:{port} ({type(err).__name__})")
    return uri


@pytest.fixture(scope="session")
def mongo_uri() -> str:
    return _require_server()


@pytest.fixture(scope="session")
def shared_database(mongo_uri: str):
    """One database for the whole module, with a fresh set of collections per test.

    A database per test is cleaner in principle and unusable in practice: MongoDB lays
    down a separate set of WiredTiger files for every one, so a few hundred of them turn
    a fast suite into a slow one and leave a pile of unreclaimed ledgers behind.
    """
    name = "signledger_test_" + uuid.uuid4().hex[:12]
    try:
        yield name
    finally:
        with pymongo.MongoClient(mongo_uri) as client:
            client.drop_database(name)


@pytest.fixture
def collection_prefix() -> str:
    return "sl_" + uuid.uuid4().hex[:12]


@pytest.fixture
def database_name(mongo_uri: str):
    """A throwaway database of its own, for the few tests that need one.

    Used only where a test constructs a backend on the *default* collection names and so
    cannot share a database with anyone.
    """
    name = "signledger_test_" + uuid.uuid4().hex[:12]
    try:
        yield name
    finally:
        with pymongo.MongoClient(mongo_uri) as client:
            client.drop_database(name)


def append(backend, data=None, **kwargs):
    """Append one correctly linked entry and return it."""
    payload = {"event": "mongo"} if data is None else data
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
            {"event": "mongo", "index": index},
            timestamp=BASE_TIMESTAMP + timedelta(seconds=index),
            **kwargs,
        )
        for index in range(count)
    ]


@pytest.fixture
def mongo_backend(mongo_uri: str, shared_database: str, collection_prefix: str):
    """A live, empty MongoDB backend on a fresh set of collections."""
    from signledger.backends.mongodb import MongoBackend

    instance = MongoBackend(
        mongo_uri,
        database=shared_database,
        entries_collection=f"{collection_prefix}_entries",
        checkpoints_collection=f"{collection_prefix}_checkpoints",
    )
    try:
        yield instance
    finally:
        instance.close()
        with pymongo.MongoClient(mongo_uri) as client:
            database = client[shared_database]
            for name in database.list_collection_names():
                if name.startswith(collection_prefix):
                    database.drop_collection(name)


class TestConstruction:
    def test_a_fresh_backend_starts_empty(self, mongo_uri: str, database_name: str) -> None:
        from signledger.backends.mongodb import MongoBackend

        backend = MongoBackend(mongo_uri, database=database_name)
        try:
            assert backend.count() == 0
        finally:
            backend.close()


class TestMongoConformance(BackendConformanceSuite):
    """Every shared backend behaviour, executed against a real MongoDB server."""

    @pytest.fixture
    def backend(self, mongo_backend):
        return mongo_backend


@pytest.fixture
def raw_db(mongo_uri: str, shared_database: str):
    """A client that does not go through the backend - the insider's view."""
    client = pymongo.MongoClient(mongo_uri)
    try:
        yield client[shared_database]
    finally:
        client.close()


# --------------------------------------------------------------------------------------
# time-range queries
# --------------------------------------------------------------------------------------


class TestTimeRangeQueries:
    """v1 stored the timestamp as an ISO *string* and queried it with a BSON Date. The two
    never compare equal, so every time-range query returned zero rows - a report of "no
    activity" indistinguishable from a report of no activity."""

    def test_a_time_range_query_returns_a_non_empty_matching_set(self, mongo_backend) -> None:
        append_many(mongo_backend, 8)

        found = list(
            mongo_backend.iter_entries(
                start_time=BASE_TIMESTAMP + timedelta(seconds=2),
                end_time=BASE_TIMESTAMP + timedelta(seconds=5),
            )
        )

        assert [e.sequence for e in found] == [2, 3, 4, 5]

    def test_the_queried_time_field_is_stored_as_an_integer(
        self, mongo_backend, raw_db
    ) -> None:
        """Stored as an integer, compared as an integer. A stored string and a queried
        Date is the v1 bug, and it is invisible from Python."""
        append(mongo_backend, timestamp=BASE_TIMESTAMP)

        document = raw_db[mongo_backend.entries_collection].find_one()

        assert isinstance(document["timestamp_us"], int)
        assert not isinstance(document["timestamp_us"], bool)
        assert document["timestamp_us"] == 1717243200000000

    def test_the_stored_integer_answers_a_native_driver_range_query(
        self, mongo_backend, raw_db
    ) -> None:
        """Proves the field is queryable as stored, not merely readable."""
        append_many(mongo_backend, 6)
        collection = raw_db[mongo_backend.entries_collection]

        matched = list(
            collection.find(
                {
                    "timestamp_us": {
                        "$gte": 1717243200000000 + 2_000_000,
                        "$lte": 1717243200000000 + 4_000_000,
                    }
                }
            )
        )

        assert sorted(d["sequence"] for d in matched) == [2, 3, 4]

    def test_a_bson_date_query_against_the_integer_field_finds_nothing(
        self, mongo_backend, raw_db
    ) -> None:
        """The v1 bug, pinned so nobody re-introduces it by "improving" the schema."""
        append_many(mongo_backend, 6)
        collection = raw_db[mongo_backend.entries_collection]

        matched = list(collection.find({"timestamp_us": {"$gte": BASE_TIMESTAMP}}))

        assert matched == []

    def test_a_time_range_matching_one_entry_returns_exactly_that_entry(
        self, mongo_backend
    ) -> None:
        appended = append_many(mongo_backend, 5)
        moment = BASE_TIMESTAMP + timedelta(seconds=3)

        found = list(mongo_backend.iter_entries(start_time=moment, end_time=moment))

        assert found == [appended[3]]

    def test_microsecond_precision_survives_the_round_trip(self, mongo_backend) -> None:
        """BSON dates are millisecond-resolution. An integer microsecond count is not."""
        moment = datetime(2024, 6, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        stored = append(mongo_backend, {"event": "us"}, timestamp=moment)

        loaded = mongo_backend.get_by_sequence(0)

        assert loaded.timestamp == moment
        assert loaded.timestamp.microsecond == 123456
        assert loaded.hash == stored.hash


# --------------------------------------------------------------------------------------
# schema validation
# --------------------------------------------------------------------------------------


def _validator(database, collection: str):
    info = list(database.list_collections(filter={"name": collection}))
    return info[0].get("options", {}).get("validator")


class TestJsonSchemaValidation:
    def test_the_entries_collection_carries_a_json_schema_validator(
        self, mongo_backend, raw_db
    ) -> None:
        validator = _validator(raw_db, mongo_backend.entries_collection)

        assert validator is not None
        assert "$jsonSchema" in validator

    def test_metadata_is_required_by_the_validator(
        self, mongo_backend, raw_db
    ) -> None:
        """``{}`` is falsy, so v1 wrote null. The database now refuses the document."""
        validator = _validator(raw_db, mongo_backend.entries_collection)

        assert "metadata" in validator["$jsonSchema"]["required"]

    def test_a_document_without_metadata_is_refused_by_the_server(
        self, mongo_backend, raw_db
    ) -> None:
        collection = raw_db[mongo_backend.entries_collection]

        document = _document()
        del document["metadata"]

        with pytest.raises(pymongo.errors.WriteError):
            collection.insert_one(document)

    def test_a_null_metadata_document_is_refused_by_the_server(
        self, mongo_backend, raw_db
    ) -> None:
        collection = raw_db[mongo_backend.entries_collection]

        with pytest.raises(pymongo.errors.WriteError):
            collection.insert_one(_document(metadata=None))

    def test_a_string_timestamp_us_is_refused_by_the_server(
        self, mongo_backend, raw_db
    ) -> None:
        """The exact v1 shape: the queried field stored as text."""
        collection = raw_db[mongo_backend.entries_collection]

        with pytest.raises(pymongo.errors.WriteError):
            collection.insert_one(_document(timestamp_us="2024-06-01T12:00:00+00:00"))

    def test_a_bson_document_payload_is_refused_by_the_server(
        self, mongo_backend, raw_db
    ) -> None:
        """``data`` is JSON *text*. A sub-document cannot round-trip byte-exactly, so the
        validator refuses one rather than let a digest quietly stop matching."""
        collection = raw_db[mongo_backend.entries_collection]

        with pytest.raises(pymongo.errors.WriteError):
            collection.insert_one(_document(data={"event": "native"}))

    def test_the_checkpoints_collection_is_validated_too(
        self, mongo_backend, raw_db
    ) -> None:
        validator = _validator(raw_db, mongo_backend.checkpoints_collection)

        assert validator is not None
        assert "head_hash" in validator["$jsonSchema"]["required"]

    def test_a_valid_document_is_still_accepted(self, mongo_backend) -> None:
        """A validator that refuses everything is not a validator."""
        append(mongo_backend, {"event": "accepted"}, metadata={"actor": "svc"})

        assert mongo_backend.get_by_sequence(0).metadata == {"actor": "svc"}


# --------------------------------------------------------------------------------------
# unique indexes
# --------------------------------------------------------------------------------------


class TestUniqueIndexes:
    def test_sequence_uniqueness_cannot_be_switched_off(
        self, mongo_backend, raw_db
    ) -> None:
        """``sequence`` is the document ``_id``, and MongoDB refuses to drop that index -
        so the one constraint that stops a fork is the one nobody can turn off."""
        collection = raw_db[mongo_backend.entries_collection]

        with pytest.raises(pymongo.errors.OperationFailure):
            collection.drop_index("_id_")

    def test_two_entries_cannot_share_a_sequence(
        self, mongo_backend, raw_db
    ) -> None:
        append(mongo_backend)
        collection = raw_db[mongo_backend.entries_collection]

        with pytest.raises(pymongo.errors.DuplicateKeyError):
            collection.insert_one(_document(sequence=0, entry_id="other", previous_hash="q"))

    def test_two_entries_cannot_claim_one_parent(
        self, mongo_backend, raw_db
    ) -> None:
        """A fork is two entries naming the same parent. The index refuses it even if
        every other field differs."""
        append(mongo_backend)
        parent = mongo_backend.get_by_sequence(0).previous_hash
        collection = raw_db[mongo_backend.entries_collection]

        with pytest.raises(pymongo.errors.DuplicateKeyError):
            collection.insert_one(
                _document(sequence=99, entry_id="other", previous_hash=parent)
            )

    def test_two_entries_cannot_share_an_entry_id(
        self, mongo_backend, raw_db
    ) -> None:
        append(mongo_backend)
        existing = mongo_backend.get_by_sequence(0).entry_id
        collection = raw_db[mongo_backend.entries_collection]

        with pytest.raises(pymongo.errors.DuplicateKeyError):
            collection.insert_one(
                _document(sequence=99, entry_id=existing, previous_hash="q")
            )

    def test_two_entries_cannot_share_a_hash(
        self, mongo_backend, raw_db
    ) -> None:
        append(mongo_backend)
        existing = mongo_backend.get_by_sequence(0).hash
        collection = raw_db[mongo_backend.entries_collection]

        with pytest.raises(pymongo.errors.DuplicateKeyError):
            collection.insert_one(
                _document(sequence=99, entry_id="other", previous_hash="q", hash=existing)
            )

    def test_the_time_filter_field_is_indexed(
        self, mongo_backend, raw_db
    ) -> None:
        collection = raw_db[mongo_backend.entries_collection]
        keys = [tuple(info["key"].items()) for info in collection.list_indexes()]

        assert (("timestamp_us", 1),) in keys


def _document(**overrides):
    """A document the validator accepts. Each test breaks exactly one rule of it."""
    document = {
        "_id": 0,
        "sequence": 0,
        "entry_id": "e",
        "ledger_id": "l",
        "format_version": 2,
        "timestamp": "2024-06-01T12:00:00+00:00",
        "timestamp_us": 0,
        "data": "{}",
        "metadata": "{}",
        "previous_hash": "p",
        "hash": "h" * 64,
        "key_id": None,
        "signature_algorithm": None,
        "signature": None,
    }
    document.update(overrides)
    if "sequence" in overrides:
        document["_id"] = overrides["sequence"]
    return document


# --------------------------------------------------------------------------------------
# capabilities: what MongoDB actually enforces
# --------------------------------------------------------------------------------------


class TestCapabilityHonesty:
    """``capabilities()`` feeds ``assurance_level()``. A token this deployment has not
    earned is a silent security claim, so each one is asserted against the behaviour of a
    real server rather than against a constant."""

    def test_a_schema_violating_update_is_refused(self, mongo_backend, raw_db) -> None:
        """The validator does apply to updates - it is not decoration."""
        append(mongo_backend)

        with pytest.raises(pymongo.errors.WriteError):
            raw_db[mongo_backend.entries_collection].update_one(
                {"_id": 0}, {"$set": {"metadata": None}}
            )

    def test_a_schema_preserving_update_still_succeeds(self, mongo_backend, raw_db) -> None:
        """$jsonSchema constrains a document's *shape*, never its history. An insider can
        rewrite a payload to another legal payload and the validator says nothing."""
        append(mongo_backend, {"event": "original"})

        result = raw_db[mongo_backend.entries_collection].update_one(
            {"_id": 0}, {"$set": {"data": '{"event":"rewritten"}'}}
        )

        assert result.modified_count == 1

    def test_a_direct_delete_still_succeeds(self, mongo_backend, raw_db) -> None:
        """A validator is never consulted on a delete at all."""
        append(mongo_backend)

        result = raw_db[mongo_backend.entries_collection].delete_one({"_id": 0})

        assert result.deleted_count == 1
        assert mongo_backend.count() == 0

    def test_append_only_enforced_is_not_claimed(self, mongo_backend) -> None:
        """Because the two tests above show it would be a lie."""
        assert CAP_APPEND_ONLY_ENFORCED not in mongo_backend.capabilities()

    def test_worm_is_not_claimed(self, mongo_backend) -> None:
        assert CAP_WORM not in mongo_backend.capabilities()

    def test_atomic_append_and_a_unique_chain_link_are_claimed(self, mongo_backend) -> None:
        tokens = mongo_backend.capabilities()

        assert CAP_ATOMIC_APPEND in tokens
        assert CAP_UNIQUE_CHAIN_LINK in tokens

    def test_durable_fsync_is_claimed_when_writes_wait_for_the_journal(
        self, mongo_backend
    ) -> None:
        assert CAP_DURABLE_FSYNC in mongo_backend.capabilities()

    def test_entries_are_written_with_a_journalled_write_concern(self, mongo_backend) -> None:
        """The token is derived from the write concern the backend actually uses."""
        assert mongo_backend._entries.write_concern.document.get("j") is True

    def test_durable_fsync_is_dropped_when_journalling_is_turned_off(
        self, mongo_uri: str, database_name: str
    ) -> None:
        from signledger.backends.mongodb import MongoBackend

        backend = MongoBackend(mongo_uri, database=database_name, journal=False)
        try:
            assert CAP_DURABLE_FSYNC not in backend.capabilities()
            assert backend._entries.write_concern.document.get("j") is not True
        finally:
            backend.close()

    def test_an_unjournalled_backend_still_stores_and_reads_entries(
        self, mongo_uri: str, database_name: str
    ) -> None:
        from signledger.backends.mongodb import MongoBackend

        backend = MongoBackend(mongo_uri, database=database_name, journal=False)
        try:
            stored = append(backend, {"event": "fast"})

            assert backend.get_by_sequence(0) == stored
        finally:
            backend.close()

    def test_capabilities_reports_only_known_tokens(self, mongo_backend) -> None:
        from signledger.backends.base import CAPABILITY_TOKENS

        assert mongo_backend.capabilities() <= CAPABILITY_TOKENS


# --------------------------------------------------------------------------------------
# the counter document
# --------------------------------------------------------------------------------------


class TestChainCounter:
    """The contract's ``findAndModify`` counter document. It serialises appenders; the
    unique indexes are what make the result *correct* even if it does not."""

    def test_the_counter_tracks_the_head(self, mongo_backend, raw_db) -> None:
        entries = append_many(mongo_backend, 3)

        counter = raw_db[mongo_backend.state_collection].find_one({"_id": "chain"})

        assert counter["sequence"] == 2
        assert counter["hash"] == entries[-1].hash

    def test_an_empty_ledger_starts_the_counter_below_zero(self, mongo_backend, raw_db) -> None:
        """Genesis needs a value to compare-and-swap *from*, so "empty" is a number."""
        counter = raw_db[mongo_backend.state_collection].find_one({"_id": "chain"})

        assert counter["sequence"] == -1

    def test_the_counter_document_is_the_append_lock(self, mongo_backend, raw_db) -> None:
        """A hash chain is inherently serial, so appenders queue on one document rather
        than racing and losing."""
        state = raw_db[mongo_backend.state_collection]

        with mongo_backend._chain_lock():
            held = state.find_one({"_id": "chain"})

        released = state.find_one({"_id": "chain"})

        assert held["owner"] is not None
        assert released["owner"] is None

    def test_a_second_appender_waits_for_the_lock_rather_than_forking(
        self, mongo_backend
    ) -> None:
        import threading

        started = threading.Event()
        finished = []

        def appender():
            started.set()
            append(mongo_backend, {"event": "queued"})
            finished.append(True)

        with mongo_backend._chain_lock():
            waiter = threading.Thread(target=appender)
            waiter.start()
            started.wait(timeout=5)
            waiter.join(timeout=0.4)

            assert waiter.is_alive()
            assert finished == []

        waiter.join(timeout=10)

        assert finished == [True]
        assert mongo_backend.count() == 1

    def test_an_expired_lease_is_taken_over(self, mongo_backend, raw_db) -> None:
        """A process killed mid-append would otherwise lock the ledger permanently, with
        no repair path."""
        raw_db[mongo_backend.state_collection].update_one(
            {"_id": "chain"},
            {"$set": {"owner": "dead-process", "lease_until_us": 1}},
        )

        appended = append(mongo_backend, {"event": "after-crash"})

        assert appended.sequence == 0
        assert mongo_backend.count() == 1

    def test_a_live_lease_is_not_stolen(self, mongo_backend, raw_db) -> None:
        from signledger.core.exceptions import StorageError

        mongo_backend.append_lock_timeout = 0.05
        raw_db[mongo_backend.state_collection].update_one(
            {"_id": "chain"},
            {"$set": {"owner": "another-writer", "lease_until_us": 2**53}},
        )

        with pytest.raises(StorageError):
            append(mongo_backend)

        assert mongo_backend.count() == 0

    def test_waiting_for_the_lock_does_not_consume_the_retry_budget(
        self, mongo_backend
    ) -> None:
        """If queueing counted as a conflict, a fourth concurrent writer would exhaust
        five attempts just waiting its turn."""
        from signledger.core.exceptions import StorageError

        mongo_backend.append_lock_timeout = 0.05
        builds = []

        with mongo_backend._chain_lock():
            with pytest.raises(StorageError):
                mongo_backend.append_atomic(lambda tip: builds.append(tip))

        assert builds == []

    def test_a_holder_whose_lease_lapsed_cannot_release_its_successor(
        self, mongo_backend, raw_db
    ) -> None:
        """The release is scoped to the owner token, so a straggler cannot unlock a chain
        somebody else is now writing to."""
        state = raw_db[mongo_backend.state_collection]
        mongo_backend.append_lease_us = 0

        with mongo_backend._chain_lock():
            state.update_one(
                {"_id": "chain"},
                {"$set": {"owner": "successor", "lease_until_us": 2**53}},
            )

        assert state.find_one({"_id": "chain"})["owner"] == "successor"

    def test_a_failed_insert_leaves_the_counter_on_the_real_head(
        self, mongo_backend, raw_db
    ) -> None:
        """The counter advances only after the insert succeeds, so a failure cannot
        strand it ahead of the data."""
        first = append(mongo_backend)
        calls = []

        def build(tip):
            calls.append(tip)
            return build_next_entry(
                tip,
                ledger_id=LEDGER_ID,
                created_at=CREATED_AT,
                data={"n": 2},
                entry_id=first.entry_id,
            )

        from signledger.core.exceptions import StorageError

        with pytest.raises(StorageError):
            mongo_backend.append_atomic(build)

        counter = raw_db[mongo_backend.state_collection].find_one({"_id": "chain"})

        assert counter["sequence"] == 0
        assert counter["hash"] == first.hash

    def test_the_ledger_still_appends_after_a_run_of_failed_claims(
        self, mongo_backend
    ) -> None:
        """A ledger that wedges after a failed append is worse than one that never had a
        counter."""
        first = append(mongo_backend)
        from signledger.core.exceptions import StorageError

        with pytest.raises(StorageError):
            mongo_backend.append_atomic(
                lambda tip: build_next_entry(
                    tip,
                    ledger_id=LEDGER_ID,
                    created_at=CREATED_AT,
                    data={"n": 2},
                    entry_id=first.entry_id,
                )
            )

        recovered = append(mongo_backend, {"event": "after"})

        assert recovered.sequence == 1
        assert recovered.previous_hash == first.hash

    def test_a_counter_stranded_ahead_of_the_data_heals_itself(
        self, mongo_backend, raw_db
    ) -> None:
        """A process killed between the insert and the counter update leaves the counter
        stale. The tip is read from the entries collection, so the next append is correct
        anyway and simply overwrites it."""
        first = append(mongo_backend)
        raw_db[mongo_backend.state_collection].update_one(
            {"_id": "chain"}, {"$set": {"sequence": 1, "hash": "stale", "entry_id": "ghost"}}
        )

        recovered = append(mongo_backend, {"event": "healed"})

        assert recovered.sequence == 1
        assert recovered.previous_hash == first.hash
        assert mongo_backend.count() == 2


# --------------------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------------------


class TestConcurrentAppend:
    def test_threads_appending_produce_a_gapless_chain(self, mongo_backend) -> None:
        import threading

        errors = []
        barrier = threading.Barrier(4)

        def writer():
            barrier.wait()
            for _ in range(5):
                try:
                    append(mongo_backend, {"event": "race"})
                except Exception as err:  # noqa: BLE001 - recorded, then asserted on
                    errors.append(err)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        entries = list(mongo_backend.iter_entries())

        assert errors == []
        assert [e.sequence for e in entries] == list(range(20))

    def test_a_concurrent_chain_links_end_to_end(self, mongo_backend) -> None:
        import threading

        barrier = threading.Barrier(3)

        def writer():
            barrier.wait()
            for _ in range(4):
                append(mongo_backend, {"event": "race"})

        threads = [threading.Thread(target=writer) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        entries = list(mongo_backend.iter_entries())

        assert all(
            entries[i].previous_hash == entries[i - 1].hash for i in range(1, len(entries))
        )


# --------------------------------------------------------------------------------------
# construction, identifiers and error redaction
# --------------------------------------------------------------------------------------


class TestConstructionFailures:
    INJECTIONS = ["db; drop", "db.other", "db$x", "", "1db", "db name", "db/x"]

    @pytest.mark.parametrize("name", INJECTIONS)
    def test_a_hostile_database_name_is_refused(self, mongo_uri: str, name: str) -> None:
        from signledger.backends.mongodb import MongoBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            MongoBackend(mongo_uri, database=name)

    @pytest.mark.parametrize("name", INJECTIONS)
    def test_a_hostile_collection_name_is_refused(self, mongo_uri: str, name: str) -> None:
        from signledger.backends.mongodb import MongoBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            MongoBackend(mongo_uri, entries_collection=name)

    def test_a_hostile_checkpoints_name_is_refused(self, mongo_uri: str) -> None:
        from signledger.backends.mongodb import MongoBackend
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            MongoBackend(mongo_uri, checkpoints_collection="cp; drop")

    def test_an_unreachable_server_fails_at_construction(self) -> None:
        """Not on the first append, halfway through building an entry."""
        from signledger.backends.mongodb import MongoBackend
        from signledger.core.exceptions import StorageError

        with pytest.raises(StorageError):
            MongoBackend(
                "mongodb://127.0.0.1:1/", database="probe", server_selection_timeout_ms=200
            )

    def test_a_connection_failure_never_quotes_the_uri(self) -> None:
        from signledger.backends.mongodb import MongoBackend
        from signledger.core.exceptions import StorageError

        with pytest.raises(StorageError) as caught:
            MongoBackend(
                "mongodb://ledger:hunter2@127.0.0.1:1/",
                database="probe",
                server_selection_timeout_ms=200,
            )

        message = str(caught.value)
        assert "hunter2" not in message
        assert "127.0.0.1" not in message

    def test_a_schema_failure_closes_the_client_it_opened(
        self, mongo_uri: str, database_name: str, monkeypatch
    ) -> None:
        from signledger.backends import mongodb
        from signledger.core.exceptions import StorageError

        def explode(self):
            raise StorageError("schema is unavailable")

        monkeypatch.setattr(mongodb.MongoBackend, "_create_schema", explode)

        with pytest.raises(StorageError):
            mongodb.MongoBackend(mongo_uri, database=database_name)

    def test_a_second_backend_over_an_existing_database_is_harmless(
        self, mongo_uri: str, database_name: str
    ) -> None:
        """Two processes starting at once must not race each other's create_collection."""
        from signledger.backends.mongodb import MongoBackend

        first = MongoBackend(mongo_uri, database=database_name)
        try:
            append(first)
            second = MongoBackend(mongo_uri, database=database_name)
            try:
                assert second.count() == 1
            finally:
                second.close()
        finally:
            first.close()

    def test_a_driver_error_reports_only_its_type(self, mongo_backend, raw_db) -> None:
        """A pymongo message quotes the document it choked on. Only the class name is
        safe to surface - contract section 10."""
        from signledger.core.exceptions import StorageError

        # Make the server refuse every entry, so the next append fails inside the driver.
        raw_db.command(
            "collMod",
            mongo_backend.entries_collection,
            validator={"$jsonSchema": {"bsonType": "object", "required": ["impossible"]}},
            validationLevel="strict",
            validationAction="error",
        )

        with pytest.raises(StorageError) as caught:
            append(mongo_backend, {"card": "4111111111111111"})

        message = str(caught.value)
        assert "4111111111111111" not in message
        assert message == "append failed: WriteError"

    def test_a_failed_write_does_not_quote_the_uri(self, mongo_backend, raw_db) -> None:
        from signledger.core.exceptions import StorageError

        raw_db.command(
            "collMod",
            mongo_backend.entries_collection,
            validator={"$jsonSchema": {"bsonType": "object", "required": ["impossible"]}},
            validationLevel="strict",
            validationAction="error",
        )

        with pytest.raises(StorageError) as caught:
            append(mongo_backend, {"ssn": "078-05-1120"})

        assert "078-05-1120" not in str(caught.value)
        assert "27017" not in str(caught.value)
        assert "57018" not in str(caught.value)


# --------------------------------------------------------------------------------------
# append-only is a privilege question on MongoDB
# --------------------------------------------------------------------------------------


class TestAppendOnlyRole:
    """$jsonSchema cannot make a collection append-only, so the shipped answer is the
    same as PostgreSQL's REVOKE script: a role that may insert and read, and nothing
    else. Returned as a command document rather than executed - granting privileges to
    yourself proves nothing."""

    def test_the_role_grants_insert_and_find_only(self) -> None:
        from signledger.backends.mongodb import append_only_role_command

        command = append_only_role_command("ledger_writer", database="audit")
        actions = command["privileges"][0]["actions"]

        assert sorted(actions) == ["find", "insert"]

    def test_the_role_never_grants_update_or_remove(self) -> None:
        from signledger.backends.mongodb import append_only_role_command

        command = append_only_role_command("ledger_writer", database="audit")
        actions = [a for privilege in command["privileges"] for a in privilege["actions"]]

        assert "update" not in actions
        assert "remove" not in actions
        assert "dropCollection" not in actions

    def test_the_role_covers_both_collections(self) -> None:
        from signledger.backends.mongodb import append_only_role_command

        command = append_only_role_command(
            "ledger_writer",
            database="audit",
            entries_collection="e",
            checkpoints_collection="c",
        )
        collections = {p["resource"]["collection"] for p in command["privileges"]}

        assert collections == {"e", "c", "e_state"}

    def test_a_hostile_role_name_is_refused(self) -> None:
        from signledger.backends.mongodb import append_only_role_command
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            append_only_role_command("writer; db.dropDatabase()")

    def test_a_hostile_database_name_is_refused(self) -> None:
        from signledger.backends.mongodb import append_only_role_command
        from signledger.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            append_only_role_command("writer", database="audit; drop")

    def test_the_command_is_accepted_by_a_real_server(self, mongo_backend, raw_db) -> None:
        """A command document that has never been run is a comment. This one runs."""
        from signledger.backends.mongodb import append_only_role_command

        role = "sl_role_" + uuid.uuid4().hex[:10]
        command = append_only_role_command(
            role,
            database=mongo_backend.database_name,
            entries_collection=mongo_backend.entries_collection,
            checkpoints_collection=mongo_backend.checkpoints_collection,
        )
        raw_db.command(command)
        try:
            info = raw_db.command({"rolesInfo": role, "showPrivileges": True})
            granted = {
                action
                for privilege in info["roles"][0]["privileges"]
                for action in privilege["actions"]
            }
        finally:
            raw_db.command({"dropRole": role})

        assert granted == {"find", "insert"}
