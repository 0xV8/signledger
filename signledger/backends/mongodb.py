"""MongoDB storage backend (architecture contract, section 4).

Every decision below is a v1 defect that had to stop being possible.

* **The queried time field is the same type it is stored as.** ``timestamp_us`` is an
  integer microsecond count, written as an integer and compared against integers. v1
  stored the timestamp as an ISO *string* and queried it with a BSON ``Date``, so every
  time-range query silently returned zero rows - a report of "no activity" that looked
  exactly like a report of no activity.
* **``data`` and ``metadata`` are stored as JSON text, not BSON sub-documents.** BSON
  cannot hold every JSON object a payload may legally contain (a key with a dot or a
  leading ``$``), and it does not distinguish ``1`` from ``1.0`` the way the canonical
  encoder must. Text round-trips byte-exactly; a sub-document does not.
* **``metadata`` is ``json.dumps(entry.metadata or {})`` and the validator requires it.**
  ``{}`` is falsy, so v1's ``metadata or None`` wrote null and the ledger could no longer
  read its own rows.
* **The tip read and the insert are serialised by a ``findAndModify`` counter document**,
  and arbitrated by unique indexes. v1 did read-modify-write under a process-local lock,
  which forks the chain the moment a second worker exists.
* **``sort("sequence")``**, never ``sort("timestamp")``.
* **``create=False`` provisions nothing at all** - no collection, no validator, no index,
  no counter document. Creating them is what stopped ``signledger verify`` from being
  runnable by a read-only user or against a secondary, and re-applying a validator or an
  index on open silently repairs the evidence that somebody removed it. See
  :meth:`MongoBackend._require_existing_schema`.
"""

import contextlib
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, Iterator, Optional, Tuple

import pymongo
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.write_concern import WriteConcern

from signledger.anchor.checkpoint import Checkpoint
from signledger.backends.base import (
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

__all__ = [
    "DEFAULT_CHECKPOINTS_COLLECTION",
    "DEFAULT_DATABASE",
    "DEFAULT_ENTRIES_COLLECTION",
    "MongoBackend",
    "append_only_role_command",
]

DEFAULT_DATABASE = "signledger"
DEFAULT_ENTRIES_COLLECTION = "ledger_entries"
DEFAULT_CHECKPOINTS_COLLECTION = "ledger_checkpoints"

#: The single document every appender contends on. One document, so ``findAndModify`` is
#: atomic over the whole of it.
CHAIN_DOCUMENT_ID = "chain"

#: The counter's value on an empty ledger. Not ``None``: the compare-and-set filter has to
#: be able to name "still empty" as a value, or genesis has no starting point to swap from.
EMPTY_SEQUENCE = -1

MAX_COLLECTION_NAME_LENGTH = 63

#: Backoff bounds while queueing for the append lock. Short enough that an uncontended
#: hand-off is not noticeably delayed, capped so a long queue does not become a spin.
_LOCK_POLL_MIN_SECONDS = 0.002
_LOCK_POLL_MAX_SECONDS = 0.05

_ENTRY_FIELDS: Tuple[str, ...] = (
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

_CHECKPOINT_FIELDS: Tuple[str, ...] = (
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


#: BSON has no single "integer" type; a value can arrive as int32 or int64 depending on
#: its magnitude and the driver, so both are accepted wherever an integer is required.
_INTEGER = ["int", "long"]
_NULLABLE_STRING = ["string", "null"]

#: The ``$jsonSchema`` contract section 4 requires of the entries collection.
#:
#: ``metadata`` is **required** and must be a ``string``: that single rule is what makes
#: the v1 "``metadata or None`` wrote null" defect unrepresentable on disk. ``data`` and
#: ``metadata`` are text rather than sub-documents because BSON cannot round-trip every
#: legal JSON object byte-exactly, and a digest over a payload that changed shape in
#: transit is worse than no digest at all.
ENTRY_SCHEMA: Dict[str, Any] = {
    "bsonType": "object",
    "required": ["_id", *(_ENTRY_FIELDS)],
    "additionalProperties": False,
    "properties": {
        "_id": {"bsonType": _INTEGER, "minimum": 0},
        "sequence": {"bsonType": _INTEGER, "minimum": 0},
        "entry_id": {"bsonType": "string"},
        "ledger_id": {"bsonType": "string"},
        "format_version": {"bsonType": _INTEGER, "minimum": 1},
        "timestamp": {"bsonType": "string"},
        "timestamp_us": {"bsonType": _INTEGER},
        "data": {"bsonType": "string"},
        "metadata": {"bsonType": "string"},
        "previous_hash": {"bsonType": "string"},
        "hash": {"bsonType": "string"},
        "key_id": {"bsonType": _NULLABLE_STRING},
        "signature_algorithm": {"bsonType": _NULLABLE_STRING},
        "signature": {"bsonType": _NULLABLE_STRING},
    },
}

#: The same treatment for checkpoints - the anchored head hash and entry count are what
#: make tail truncation detectable, so they may not be absent or of the wrong type.
CHECKPOINT_SCHEMA: Dict[str, Any] = {
    "bsonType": "object",
    "required": list(_CHECKPOINT_FIELDS),
    "properties": {
        "ledger_id": {"bsonType": "string"},
        "sequence": {"bsonType": _INTEGER, "minimum": 0},
        "head_hash": {"bsonType": "string"},
        "entry_count": {"bsonType": _INTEGER, "minimum": 0},
        "merkle_root": {"bsonType": "string"},
        "timestamp": {"bsonType": "string"},
        "timestamp_us": {"bsonType": _INTEGER},
        "signature": {"bsonType": _NULLABLE_STRING},
        "key_id": {"bsonType": _NULLABLE_STRING},
        "format_version": {"bsonType": _INTEGER, "minimum": 1},
    },
}


class MongoBackend:
    """A ledger store in MongoDB.

    ``create`` defaults to True, which is what every writer wants: the collections, their
    validators and their indexes are provisioned on open. Pass ``create=False`` to read a
    store that already exists without creating anything - what ``signledger verify`` does,
    so that an integrity check can be run by a user holding ``find`` and nothing else, and
    so that a mistyped database name is answered with "that store does not exist" rather
    than with a verdict about an audit log that was never there.
    """

    def __init__(
        self,
        uri: str,
        *,
        database: str = DEFAULT_DATABASE,
        entries_collection: str = DEFAULT_ENTRIES_COLLECTION,
        checkpoints_collection: str = DEFAULT_CHECKPOINTS_COLLECTION,
        server_selection_timeout_ms: int = 5000,
        journal: bool = True,
        append_lease_ms: int = 30_000,
        append_lock_timeout: float = 30.0,
        create: bool = True,
    ) -> None:
        self.create = bool(create)
        self.database_name = validate_identifier(database, kind="database name")
        self.entries_collection = validate_identifier(
            entries_collection, kind="entries collection name"
        )
        self.checkpoints_collection = validate_identifier(
            checkpoints_collection, kind="checkpoints collection name"
        )
        self.state_collection = _derived_name(self.entries_collection, "_state")
        # j=True means the server acknowledges a write only once the journal is on disk.
        # That is what durable_fsync claims, so the claim is derived from this and not
        # from a constant a deployment could quietly falsify.
        self.write_concern = WriteConcern(w=1, j=True) if journal else WriteConcern(w=1)
        self.append_lease_us = int(append_lease_ms) * 1000
        self.append_lock_timeout = float(append_lock_timeout)
        self._closed = False
        self._client: Optional[pymongo.MongoClient[Dict[str, Any]]] = pymongo.MongoClient(
            uri, serverSelectionTimeoutMS=server_selection_timeout_ms
        )
        # Construction is all-or-nothing. The client is built before the first round trip,
        # so *both* the reachability check and the schema have to be able to close it -
        # otherwise an unreachable server leaks a client and its monitor threads on every
        # construction attempt.
        try:
            with _translating("connect"):
                # A lazily-connecting client would surface a bad URI on the first query,
                # halfway through an append. Fail at construction instead.
                self._client.admin.command("ping")
            if self.create:
                self._create_schema()
            else:
                self._require_existing_schema()
        except BaseException:
            self._client.close()
            self._client = None
            raise

    # -- plumbing ----------------------------------------------------------------------

    @property
    def _db(self) -> Any:
        if self._closed or self._client is None:
            raise StorageError("backend is closed")
        return self._client[self.database_name]

    @property
    def _entries(self) -> Any:
        return self._db.get_collection(self.entries_collection, write_concern=self.write_concern)

    @property
    def _checkpoints(self) -> Any:
        return self._db.get_collection(
            self.checkpoints_collection, write_concern=self.write_concern
        )

    @property
    def _state(self) -> Any:
        return self._db.get_collection(self.state_collection, write_concern=self.write_concern)

    def close(self) -> None:
        """Close the client. Idempotent."""
        self._closed = True
        client, self._client = self._client, None
        if client is not None:
            client.close()

    # -- schema ------------------------------------------------------------------------

    def _require_existing_schema(self) -> None:
        """Check both collections are there, and create neither.

        One ``listCollections`` and nothing else. MongoDB brings a collection - and its
        database - into being on first use, so there is no such thing as an accidental
        read here: the check has to be explicit, or a mistyped database name is answered
        with an empty ledger, which is the same words an operator would use for "somebody
        deleted my audit log".

        A missing collection is a :class:`StorageError`, never an integrity verdict; the
        caller above turns the two into different exit codes.
        """
        wanted = [self.entries_collection, self.checkpoints_collection]
        with _translating("open"):
            present = set(self._db.list_collection_names(filter={"name": {"$in": wanted}}))
        missing = [name for name in wanted if name not in present]
        if missing:
            # Collection and database names only. A MongoDB URI carries a password
            # (contract section 10), and this is the message an operator sees when the URI
            # is what was wrong.
            raise StorageError(
                f"open failed: this store does not exist - database "
                f"{self.database_name} holds no collection called {', '.join(missing)}, "
                f"and this backend was opened with create=False, which never creates one"
            )

    def _refuse_write(self, operation: str) -> None:
        """A store opened for reading is not written to, and says so in one message."""
        if not self.create:
            raise StorageError(
                f"{operation} failed: this backend was opened with create=False, which "
                f"reads an existing store and never writes to it"
            )

    def _create_schema(self) -> None:
        # MongoDB creates a collection on first write, which would leave ``$collStats``
        # with nothing to report and checkpoints unvalidated until the first one is saved.
        for name in (
            self.entries_collection,
            self.checkpoints_collection,
            self.state_collection,
        ):
            self._create_collection(name)
        self._apply_validator(self.entries_collection, ENTRY_SCHEMA)
        self._apply_validator(self.checkpoints_collection, CHECKPOINT_SCHEMA)
        entries = self._entries
        # ``sequence`` is the document's own ``_id``, so its uniqueness is enforced by the
        # index MongoDB will not let anyone drop. The remaining constraints are explicit.
        entries.create_index([("entry_id", ASCENDING)], unique=True, name="entry_id_unique")
        # Two entries naming one parent is a fork. Two entries with one digest is a
        # collision. Both are refused by the store, not merely by the code above it.
        entries.create_index(
            [("previous_hash", ASCENDING)], unique=True, name="previous_hash_unique"
        )
        entries.create_index([("hash", ASCENDING)], unique=True, name="hash_unique")
        entries.create_index([("timestamp_us", ASCENDING)], name="timestamp_us")
        self._state.update_one(
            {"_id": CHAIN_DOCUMENT_ID},
            {
                "$setOnInsert": {
                    "sequence": EMPTY_SEQUENCE,
                    "hash": None,
                    "entry_id": None,
                    "owner": None,
                    "lease_until_us": 0,
                }
            },
            upsert=True,
        )

    def _create_collection(self, name: str) -> None:
        """Create a collection, tolerating a second backend that got there first."""
        with contextlib.suppress(pymongo.errors.CollectionInvalid):
            self._db.create_collection(name)

    def _apply_validator(self, name: str, schema: Dict[str, Any]) -> None:
        """Attach the ``$jsonSchema`` the contract requires, idempotently.

        ``collMod`` rather than a ``create_collection`` argument, so that a collection an
        earlier version created without a validator gets one on the next start rather than
        staying permanently unguarded.
        """
        self._db.command(
            "collMod",
            name,
            validator={"$jsonSchema": schema},
            validationLevel="strict",
            validationAction="error",
        )

    # -- append ------------------------------------------------------------------------

    def append_atomic(self, build: Callable[[Optional[Entry]], Entry]) -> Entry:
        """Claim the next sequence with ``findAndModify``, then insert under it."""
        self._refuse_write("append")
        with _translating("append"):
            return retry_on_conflict(self._append_once(build), _is_conflict)

    def _append_once(self, build: Callable[[Optional[Entry]], Entry]) -> Callable[[], Entry]:
        def attempt() -> Entry:
            with self._chain_lock():
                # Read inside the lock, from the entries collection - the authoritative
                # data. The counter's own sequence is a record of what happened, never the
                # thing correctness depends on, so a counter left stale by a killed
                # process cannot wedge the ledger.
                tip = self._select_head()
                entry = build(tip)
                check_link(tip, entry)
                self._entries.insert_one(entry_to_document(entry))
                self._state.update_one(
                    {"_id": CHAIN_DOCUMENT_ID},
                    {
                        "$set": {
                            "sequence": entry.sequence,
                            "hash": entry.hash,
                            "entry_id": entry.entry_id,
                        }
                    },
                )
                return entry

        return attempt

    @contextmanager
    def _chain_lock(self) -> Iterator[Dict[str, Any]]:
        """Hold the chain tip against every other appender, on any machine.

        This is the contract's ``findAndModify`` counter document. ``findAndModify`` is
        atomic over one document, so the compare-and-set that stamps an owner onto it is
        what serialises appenders - a hash chain is inherently serial and cannot be
        written in parallel however clever the schema.

        The ownership stamp carries a **lease**. A holder that is killed mid-append would
        otherwise lock the ledger permanently, with no repair path; instead the lease
        lapses and the next appender takes over. Waiting for a turn is not a conflict, so
        it does not consume the bounded append-retry budget - if it did, a fourth
        concurrent writer would exhaust five attempts just queueing.
        """
        token = uuid.uuid4().hex
        deadline = time.monotonic() + self.append_lock_timeout
        delay = _LOCK_POLL_MIN_SECONDS
        while True:
            now_us = _now_micros()
            acquired = self._state.find_one_and_update(
                {
                    "_id": CHAIN_DOCUMENT_ID,
                    "$or": [{"owner": None}, {"lease_until_us": {"$lte": now_us}}],
                },
                {"$set": {"owner": token, "lease_until_us": now_us + self.append_lease_us}},
                return_document=ReturnDocument.AFTER,
            )
            if acquired is not None:
                break
            if time.monotonic() >= deadline:
                raise StorageError(
                    f"timed out after {self.append_lock_timeout}s waiting for the chain "
                    f"append lock; another writer is holding it"
                )
            time.sleep(delay)
            delay = min(delay * 2, _LOCK_POLL_MAX_SECONDS)
        try:
            yield acquired
        finally:
            # Scoped to this token, so a holder whose lease already lapsed cannot release
            # the lock its successor now owns.
            self._state.update_one(
                {"_id": CHAIN_DOCUMENT_ID, "owner": token},
                {"$set": {"owner": None, "lease_until_us": 0}},
            )

    # -- reads -------------------------------------------------------------------------

    def _select_head(self) -> Optional[Entry]:
        document = self._entries.find_one(sort=[("sequence", DESCENDING)])
        return None if document is None else entry_from_document(document)

    def get_entry(self, entry_id: str) -> Optional[Entry]:
        with _translating("get_entry"):
            document = self._entries.find_one({"entry_id": entry_id})
        return None if document is None else entry_from_document(document)

    def get_by_sequence(self, sequence: int) -> Optional[Entry]:
        with _translating("get_by_sequence"):
            document = self._entries.find_one({"_id": sequence})
        return None if document is None else entry_from_document(document)

    def get_head(self) -> Optional[Entry]:
        with _translating("get_head"):
            return self._select_head()

    def get_oldest(self) -> Optional[Entry]:
        with _translating("get_oldest"):
            document = self._entries.find_one(sort=[("sequence", ASCENDING)])
        return None if document is None else entry_from_document(document)

    def count(self) -> int:
        with _translating("count"):
            return int(self._entries.count_documents({}))

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
        if limit == 0:
            # pymongo reads ``limit(0)`` as *no limit*. A request for zero rows would
            # therefore return every row - the exact shape of v1's ``if limit:`` bug.
            return iter([])
        query = build_entry_filter(
            start_seq=start_seq, end_seq=end_seq, start_time=start_time, end_time=end_time
        )
        with _translating("iter_entries"):
            # sort -> skip -> limit is applied server-side in that order, so the offset
            # lands after the filter rather than on rows the filter already discarded.
            cursor = self._entries.find(query).sort("sequence", ASCENDING).skip(offset)
            if limit is not None:
                cursor = cursor.limit(limit)
            documents = list(cursor)
        return iter([entry_from_document(document) for document in documents])

    def size_bytes(self) -> int:
        """Bytes the ledger's collections occupy, indexes included."""
        with _translating("size_bytes"):
            return sum(
                self._collection_size(name)
                for name in (
                    self.entries_collection,
                    self.checkpoints_collection,
                    self.state_collection,
                )
            )

    def _collection_size(self, name: str) -> int:
        stats = list(self._db[name].aggregate([{"$collStats": {"storageStats": {}}}]))
        return 0 if not stats else int(stats[0]["storageStats"]["totalSize"])

    # -- capabilities ------------------------------------------------------------------

    def capabilities(self) -> FrozenSet[str]:
        """Report what this deployment **actually** enforces.

        ``append_only_enforced`` is deliberately absent. The contract names ``$jsonSchema``
        as MongoDB's append-only mechanism, and the validator is applied - but a validator
        constrains a document's *shape*, never its history: a schema-preserving ``UPDATE``
        is accepted and a ``DELETE`` is never validated at all. Claiming the token would
        let ``assurance_level()`` say WORM about a collection an insider can rewrite, so
        the honest answer is to withhold it. Append-only on MongoDB is a *privilege*
        question - see :func:`append_only_role_command`.
        """
        tokens = {CAP_ATOMIC_APPEND, CAP_UNIQUE_CHAIN_LINK}
        if self.write_concern.document.get("j") is True:
            tokens.add(CAP_DURABLE_FSYNC)
        return frozenset(tokens)

    # -- checkpoints -------------------------------------------------------------------

    def save_checkpoint(self, cp: Checkpoint) -> None:
        self._refuse_write("save_checkpoint")
        with _translating("save_checkpoint"):
            self._checkpoints.insert_one(checkpoint_to_document(cp))

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        with _translating("get_latest_checkpoint"):
            document = self._checkpoints.find_one(
                sort=[("sequence", DESCENDING), ("_id", DESCENDING)]
            )
        return None if document is None else checkpoint_from_document(document)

    def iter_checkpoints(self) -> Iterator[Checkpoint]:
        with _translating("iter_checkpoints"):
            documents = list(
                self._checkpoints.find().sort([("sequence", ASCENDING), ("_id", ASCENDING)])
            )
        return iter([checkpoint_from_document(document) for document in documents])


# --------------------------------------------------------------------------------------
# document mapping
# --------------------------------------------------------------------------------------


def entry_to_document(entry: Entry) -> Dict[str, Any]:
    """Flatten an entry into its BSON document.

    ``_id`` is the sequence number. MongoDB will not let anyone drop the ``_id`` index, so
    "one entry per sequence" is the one constraint on this collection that cannot be
    turned off.
    """
    row = entry_to_row(entry)
    document: Dict[str, Any] = {field: row[field] for field in _ENTRY_FIELDS}
    document["_id"] = entry.sequence
    return document


def entry_from_document(document: Dict[str, Any]) -> Entry:
    """Rebuild an entry from its stored document, re-validating it on the way in."""
    return entry_from_row(document)


def checkpoint_to_document(cp: Checkpoint) -> Dict[str, Any]:
    row = checkpoint_to_row(cp)
    return {field: row[field] for field in _CHECKPOINT_FIELDS}


def checkpoint_from_document(document: Dict[str, Any]) -> Checkpoint:
    return checkpoint_from_row(document)


def build_entry_filter(
    *,
    start_seq: Optional[int] = None,
    end_seq: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the query document for :meth:`MongoBackend.iter_entries`.

    Time bounds compare ``timestamp_us``, an *integer*, because that is the type the field
    is stored as. v1 stored an ISO string and queried a BSON ``Date``; the two never
    compare equal, so every time-range query returned nothing at all.
    """
    query: Dict[str, Any] = {}
    sequence: Dict[str, Any] = {}
    moment: Dict[str, Any] = {}
    if start_seq is not None:
        sequence["$gte"] = start_seq
    if end_seq is not None:
        sequence["$lte"] = end_seq
    if start_time is not None:
        moment["$gte"] = timestamp_to_micros(start_time)
    if end_time is not None:
        moment["$lte"] = timestamp_to_micros(end_time)
    if sequence:
        query["sequence"] = sequence
    if moment:
        query["timestamp_us"] = moment
    return query


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


@contextmanager
def _translating(operation: str) -> Iterator[None]:
    """Re-raise driver errors as :class:`StorageError`.

    Only the exception *type* is reported: a pymongo message quotes the document it choked
    on and a connection error quotes the URI, which can carry a password. Contract
    section 10 forbids either from reaching an exception message.
    """
    try:
        yield
    except pymongo.errors.PyMongoError as err:
        raise StorageError(f"{operation} failed: {type(err).__name__}") from err


def _is_conflict(err: BaseException) -> bool:
    """Recognise "another writer won the race" - never "your data is wrong"."""
    return isinstance(err, pymongo.errors.DuplicateKeyError)


def _derived_name(base: str, suffix: str) -> str:
    """Build ``<base><suffix>`` short enough to stay a legal collection name."""
    return validate_identifier(
        base[: MAX_COLLECTION_NAME_LENGTH - len(suffix)] + suffix, kind="derived name"
    )


def _now_micros() -> int:
    """Wall-clock microseconds. Leases are compared against the *server's* stored value,
    so every appender has to read the same clock domain - wall clock, not monotonic."""
    return int(time.time() * 1_000_000)


def append_only_role_command(
    role: str,
    *,
    database: str = DEFAULT_DATABASE,
    entries_collection: str = DEFAULT_ENTRIES_COLLECTION,
    checkpoints_collection: str = DEFAULT_CHECKPOINTS_COLLECTION,
) -> Dict[str, Any]:
    """A ``createRole`` command granting ``insert`` and ``find`` - and nothing else.

    MongoDB's ``$jsonSchema`` constrains a document's *shape*, never its history: a
    schema-preserving ``UPDATE`` is accepted and a ``DELETE`` is never validated at all.
    So append-only on MongoDB is a privilege question, exactly as it is on PostgreSQL.
    Run this as an administrator, then run the application as ``role``.

    Returned as a command document rather than executed, because a process granting
    privileges to itself proves nothing - the administrator has to do it.
    """
    validate_identifier(role, kind="role name")
    validate_identifier(database, kind="database name")
    validate_identifier(entries_collection, kind="entries collection name")
    validate_identifier(checkpoints_collection, kind="checkpoints collection name")
    collections = (
        entries_collection,
        checkpoints_collection,
        _derived_name(entries_collection, "_state"),
    )
    return {
        "createRole": role,
        "privileges": [
            {
                "resource": {"db": database, "collection": name},
                # No "update", no "remove", no "dropCollection". Those are the whole point.
                "actions": ["find", "insert"],
            }
            for name in collections
        ],
        "roles": [],
    }
